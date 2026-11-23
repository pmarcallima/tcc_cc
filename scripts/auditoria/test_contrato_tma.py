"""Testes locais de roteamento e auditoria; não simulam execução correta na GPU.

Executar: python3 -m unittest discover -s scripts/auditoria -p 'test_contrato_tma.py' -v
Os objetos mínimos de IR/templates substituem só dependências do PyTorch.
"""

import ast
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("contract_audit", ROOT / "scripts/auditoria/verificar_contrato_tma.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Integer(int):
    """Só o tipo e a aritmética inteira usados pelo filtro, sem importar SymPy."""


class Node:
    def __init__(self, shape=(256, 256), stride=None, offset=0, dtype="float16"):
        self.shape = [Integer(s) if isinstance(s, int) else s for s in shape]
        self.stride = list(stride) if stride is not None else [self.shape[-1], 1]
        self.offset, self.dtype = offset, dtype

    def get_size(self):
        return self.shape

    def get_stride(self):
        return self.stride

    def get_layout(self):
        return types.SimpleNamespace(offset=self.offset)

    def get_dtype(self):
        return self.dtype


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.native = types.SimpleNamespace(generate=Mock(return_value="native_choice"))
        self.extended = types.SimpleNamespace(generate=Mock(return_value="extended_choice"))
        modules = {name: types.ModuleType(name) for name in (
            "torch", "torch._inductor", "torch._inductor.kernel", "torch._inductor.kernel.mm")}
        for name, module in modules.items():
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(modules[parent], child, module)
        modules["torch"].float16 = "float16"
        modules["torch._inductor.kernel.mm"].persistent_tma_mm_template = self.native
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

        @contextlib.contextmanager
        def restricted(enabled, declare_b):
            generate = self.extended.generate
            try:
                yield self.extended
            finally:
                # O contexto real cria um template novo a cada chamada.
                self.extended.generate = generate

        # Executa as definições reais do arquivo, sem importar CUDA/Inductor.
        source = ast.parse((ROOT / "src/inductor_tma_prologue_patch.py").read_text())
        definitions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                       and node.name in {"_candidate_contract_reason", "selectable_tma_template"}]
        namespace = {"contextlib": contextlib, "torch": modules["torch"],
                     "sympy": types.SimpleNamespace(Integer=Integer),
                     "restricted_tma_template": restricted}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "actual_contract", "exec"), namespace)
        self.selectable = namespace["selectable_tma_template"]
        self.options = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}

    def route(self, nodes, options=None):
        decisions = []
        with self.selectable(True, decisions) as template:
            choice = template.generate(input_nodes=nodes, **(options or self.options))
        return choice, decisions[0]

    def test_valid_candidate_uses_extension(self):
        nodes = [Node(), Node()]
        choice, decision = self.route(nodes)
        self.assertEqual(choice, "extended_choice")
        self.assertEqual(decision["reason"], "accepted")
        self.assertEqual(decision["status"], "generated")
        self.native.generate.assert_not_called()
        self.extended.generate.assert_called_once_with(input_nodes=nodes, **self.options)

    def test_rejections_delegate_original_arguments_to_native(self):
        cases = [
            ([], "input_count"),
            ([Node(shape=(256,)), Node()], "rank"),
            ([Node(shape=(object(), 256)), Node()], "dynamic_shape"),
            ([Node(dtype="bfloat16"), Node(dtype="bfloat16")], "dtype"),
            ([Node(stride=(1, 256)), Node()], "stride_a"),
            ([Node(), Node(stride=(1, 256))], "stride_b"),
            ([Node(offset=8), Node()], "offset_a"),
            ([Node(), Node(offset=8)], "offset_b"),
            ([Node(shape=(257, 256)), Node()], "tail_m"),
            ([Node(shape=(256, 264)), Node(shape=(264, 256))], "tail_k"),
        ]
        for nodes, reason in cases:
            with self.subTest(reason=reason):
                self.native.generate.reset_mock()
                choice, decision = self.route(nodes)
                self.assertEqual(choice, "native_choice")
                self.assertEqual(decision["reason"], reason)
                self.assertEqual(decision["delegated_to"], "native")
                self.native.generate.assert_called_once_with(input_nodes=nodes, **self.options)
        self.extended.generate.assert_not_called()

    def test_alignment_rejection_and_no_choice_are_recorded(self):
        self.native.generate.return_value = None
        choice, decision = self.route([Node(), Node()], {**self.options, "BLOCK_K": 4})
        self.assertIsNone(choice)
        self.assertEqual(decision["reason"], "block_k_alignment")
        self.assertEqual(decision["status"], "not_generated")

    def test_generator_exception_is_preserved(self):
        error = NotImplementedError("native rejected choice")
        self.native.generate.side_effect = error
        decisions = []
        with self.selectable(True, decisions) as template:
            with self.assertRaises(NotImplementedError) as raised:
                template.generate(input_nodes=[Node(offset=8), Node()], **self.options)
        self.assertIs(raised.exception, error)
        self.assertEqual(decisions[0]["status"], "error")

    def test_disabled_patch_returns_original_without_decisions(self):
        decisions = []
        with self.selectable(False, decisions) as template:
            self.assertIs(template, self.native)
        self.assertEqual(decisions, [])


class CandidateLimitTests(unittest.TestCase):
    def setUp(self):
        def append(kind):
            def generate(choices, **kwargs):
                choices.append(types.SimpleNamespace(name=f"triton_{kind}_0"))
            return generate

        self.common = types.SimpleNamespace(maybe_append_choice=append("mm"))
        self.tma = types.SimpleNamespace(maybe_append_choice=append("mm_persistent_tma"))
        self.select = Mock(return_value="selected")
        mm = types.SimpleNamespace(mm_template=self.common, use_triton_tma_template=Mock(return_value=True),
                                   autotune_select_algorithm=self.select)
        source = ast.parse((ROOT / "experiments/integracao/modal_contrato_tma.py").read_text())
        collect = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "collect_case")
        observer = next(node for node in collect.body if isinstance(node, ast.FunctionDef)
                        and node.name == "observe_and_limit")
        namespace = {"contextlib": contextlib, "replace": patch, "mm": mm,
                     "extension": types.SimpleNamespace(ContractError=type("FakeContractError", (Exception,), {}))}
        exec(compile(ast.Module(body=[observer], type_ignores=[]), "actual_observer", "exec"), namespace)
        self.observe = namespace["observe_and_limit"]
        self.mm = mm

    def test_distinct_temporary_lists_still_offer_one_choice_per_template(self):
        record = {}
        offered = []
        common_append, tma_append = self.common.maybe_append_choice, self.tma.maybe_append_choice
        with self.observe(self.tma, record):
            for _ in range(8):
                for template in (self.common, self.tma):
                    temporary = []
                    template.maybe_append_choice(temporary)
                    offered.extend(temporary)
            selected = self.mm.autotune_select_algorithm("mm", offered, "inputs", "layout")
        self.assertEqual(selected, "selected")
        self.assertEqual(len(offered), 2)
        self.assertEqual([c.name for c in offered], ["triton_mm_0", "triton_mm_persistent_tma_0"])
        for kind in ("common", "tma"):
            counts = record["candidate_limits"][kind]
            self.assertEqual((counts["attempted"], counts["generated"], counts["skipped"]), (1, 1, 7))
        self.assertEqual(record["candidate_limits"]["tma"]["config"], {
            "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "num_stages": 3, "num_warps": 4})
        self.assertEqual(record["offered_choices"], [{"operation": "mm", "names": [c.name for c in offered]}])
        self.select.assert_called_once_with("mm", offered, "inputs", "layout")
        self.assertIs(self.common.maybe_append_choice, common_append)
        self.assertIs(self.tma.maybe_append_choice, tma_append)

    def test_each_control_gets_a_fresh_quota(self):
        for _ in range(2):
            record = {}
            choices = []
            with self.observe(self.tma, record):
                self.tma.maybe_append_choice(choices)
            self.assertEqual(len(choices), 1)
            self.assertEqual(record["candidate_limits"]["tma"]["attempted"], 1)

    def test_failed_first_attempt_does_not_retry_other_configs(self):
        self.tma.maybe_append_choice = Mock(side_effect=NotImplementedError("unsupported choice"))
        record = {}
        with self.observe(self.tma, record):
            with self.assertRaises(NotImplementedError):
                self.tma.maybe_append_choice([])
            self.assertIsNone(self.tma.maybe_append_choice([]))
        counts = record["candidate_limits"]["tma"]
        self.assertEqual((counts["attempted"], counts["generated"], counts["skipped"]), (1, 0, 1))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tail_m").mkdir()
        self.save(self.root / "collection.json", {"status": "completed",
            "requested_cases": ["tail_m"], "completed_cases": ["tail_m"]})
        (self.root / "experiment_snapshot.py").write_text("# synthetic fixture\n")
        (self.root / "tail_m/patch_snapshot.py").write_text("# synthetic fixture\n")
        self.data = {"name": "tail_m", "paths": {}}
        for label in ("original", "patched"):
            (self.root / f"tail_m/{label}_output_code.py").write_text(
                "    triton_cast.run(x)\n    extern_kernels.mm(q, w)\n")
            (self.root / f"tail_m/{label}_fusion.log").write_text("synthetic fixture\n")
            self.data["paths"][label] = {"status": "executed", "candidate_decisions": [],
                "candidate_limits": {
                    "common": {"attempted": 1, "generated": 1, "skipped": 0, "config": None},
                    "tma": {"attempted": 0, "generated": 0, "skipped": 0, "config": None}},
                "offered_choices": [{"operation": "mm", "names": ["triton_mm_0", "mm"]}],
                "prologue_rejections": [], "native_tma_eligibility": [False],
                "correctness": [{"seed": seed, "allclose": True, "max_abs_error": 0.0}
                                for seed in (0, 1, 2)],
                "input_metadata": [
                    {"shape": [257, 256], "stride": [256, 1], "offset": 0, "dtype": "torch.int8"},
                    {"shape": [256, 256], "stride": [256, 1], "offset": 0, "dtype": "torch.float16"},
                    {"shape": [1, 256], "stride": [256, 1], "offset": 0, "dtype": "torch.float16"}],
                "profile_outputs_correct": True, "profile_calls": 3,
                "recompile_guard": "fail_on_recompile", "triton_launches": ["triton_cast"],
                "extern_calls": ["extern_kernels.mm"], "extension_in_wrapper": False}
            events = []
            for index in range(3):
                events.append({"cat": "user_annotation", "ph": "X", "ts": 100 + 100 * index,
                    "dur": 50, "name": f"contract/tail_m/{label}/call_{index}"})
                # Relógios desalinhados e matmul externo com DOIS kernels.
                for kernel_index, kernel in enumerate(("triton_cast", "vendor_mm_1", "vendor_mm_2")):
                    correlation = index * 3 + kernel_index
                    events.extend([
                        {"cat": "cuda_runtime", "ph": "X", "name": "cudaLaunchKernel",
                         "ts": 105 + index * 100 + kernel_index, "dur": 1,
                         "args": {"correlation": correlation}},
                        {"cat": "kernel", "ph": "X", "name": kernel,
                         "ts": 5 + index * 10 + kernel_index, "dur": 1,
                         "args": {"correlation": correlation}},
                    ])
            self.save(self.root / f"tail_m/{label}_runtime_trace.json", {"traceEvents": events})
        self.save_result()

    def save(self, path, data):
        path.write_text(json.dumps(data))

    def save_result(self):
        self.save(self.root / "tail_m/results.json", self.data)

    def test_native_exclusion_is_not_filter_coverage(self):
        result = audit.audit_collection(self.root)
        self.assertTrue(result["all_executions_passed"])
        self.assertFalse(result["planned_coverage_complete"])
        self.assertFalse(result["coverage"]["tail_m"])
        self.assertTrue(result["paths"][1]["native_tma_excluded_before_filter"])
        self.assertEqual(result["paths"][1]["kernels_per_call"], [3, 3, 3])

    def test_generated_native_delegation_covers_only_target_case(self):
        patched = self.data["paths"]["patched"]
        patched["native_tma_eligibility"] = [True]
        patched["candidate_decisions"] = [{"extended": False, "reason": "tail_m",
            "delegated_to": "native", "status": "generated"}]
        patched["candidate_limits"]["tma"] = {"attempted": 1, "generated": 1, "skipped": 0,
            "config": {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "num_stages": 3, "num_warps": 4}}
        patched["offered_choices"][0]["names"].append("triton_mm_persistent_tma_1")
        self.save_result()
        result = audit.audit_collection(self.root)
        self.assertTrue(result["all_executions_passed"])
        self.assertTrue(result["protocol_conformant"])
        self.assertTrue(result["coverage"]["tail_m"])
        self.assertFalse(result["planned_coverage_complete"])

    def test_extra_offered_candidates_fail_protocol_even_when_outputs_pass(self):
        patched = self.data["paths"]["patched"]
        patched["offered_choices"][0]["names"].append("triton_mm_1")
        self.save_result()
        result = audit.audit_collection(self.root)
        self.assertTrue(result["all_executions_passed"])
        self.assertFalse(result["protocol_conformant"])
        self.assertFalse(result["paths"][1]["protocol_checks"]["actual_template_counts_limited"])

    def test_missing_counters_do_not_pass_protocol(self):
        del self.data["paths"]["patched"]["candidate_limits"]
        self.save_result()
        result = audit.audit_collection(self.root)
        self.assertTrue(result["all_executions_passed"])
        self.assertFalse(result["protocol_conformant"])

    def test_cli_rejects_protocol_violation_without_require_coverage(self):
        self.data["paths"]["patched"]["offered_choices"][0]["names"].append("triton_mm_1")
        self.save_result()
        with patch.object(sys, "argv", ["verify", str(self.root)]), patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as raised:
                audit.main()
        self.assertEqual(raised.exception.code, 1)
        saved = json.loads((self.root / "contract_verification.json").read_text())
        self.assertTrue(saved["all_executions_passed"])
        self.assertFalse(saved["protocol_conformant"])

    def test_nonfixed_tma_config_fails_protocol(self):
        counts = self.data["paths"]["patched"]["candidate_limits"]["tma"]
        counts.update(attempted=1, generated=0,
                      config={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_stages": 3, "num_warps": 4})
        self.save_result()
        result = audit.audit_collection(self.root)
        self.assertFalse(result["protocol_conformant"])
        self.assertFalse(result["paths"][1]["protocol_checks"]["tma_config_fixed"])

    def test_incorrect_output_fails_even_with_native_delegation(self):
        self.data["paths"]["patched"]["correctness"][1]["allclose"] = False
        self.save_result()
        self.assertFalse(audit.audit_collection(self.root)["all_executions_passed"])

    def test_wrong_input_shape_is_not_accepted_as_tail_case(self):
        self.data["paths"]["patched"]["input_metadata"][0]["shape"] = [256, 256]
        self.save_result()
        self.assertFalse(audit.audit_collection(self.root)["all_executions_passed"])

    def test_correct_numbers_do_not_justify_unsupported_fused_extension(self):
        patched = self.data["paths"]["patched"]
        patched.update(extern_calls=[], extension_in_wrapper=True,
                       native_tma_eligibility=[True], candidate_decisions=[{
                           "extended": True, "reason": "accepted",
                           "delegated_to": "extended", "status": "generated"}])
        self.save_result()
        (self.root / "tail_m/patched_output_code.py").write_text(
            "# TCC_RESTRICTED_TMA_PROLOGUE\n    triton_cast.run(x)\n")
        trace = self.root / "tail_m/patched_runtime_trace.json"
        data = json.loads(trace.read_text())
        data["traceEvents"] = [e for e in data["traceEvents"] if not e["name"].startswith("vendor_mm")]
        self.save(trace, data)
        result = audit.audit_collection(self.root)
        self.assertFalse(result["all_executions_passed"])
        self.assertFalse(result["paths"][1]["checks"]["no_unsupported_fused_extension"])

    def test_missing_cuda_correlation_fails(self):
        trace = self.root / "tail_m/patched_runtime_trace.json"
        data = json.loads(trace.read_text())
        next(e for e in data["traceEvents"] if e["cat"] == "kernel")["args"]["correlation"] = -1
        self.save(trace, data)
        self.assertFalse(audit.audit_collection(self.root)["all_executions_passed"])

    def test_external_matmul_before_cast_fails(self):
        trace = self.root / "tail_m/patched_runtime_trace.json"
        data = json.loads(trace.read_text())
        for event in data["traceEvents"]:
            if event["cat"] == "kernel":
                call = event["args"]["correlation"] // 3
                event["ts"] = call * 10 + {
                    "vendor_mm_1": 1, "vendor_mm_2": 2, "triton_cast": 3}[event["name"]]
        self.save(trace, data)
        result = audit.audit_collection(self.root)
        self.assertFalse(result["all_executions_passed"])
        self.assertFalse(result["paths"][1]["checks"]["kernels_match_wrapper"])

    def test_external_kernels_split_by_cast_fail(self):
        trace = self.root / "tail_m/patched_runtime_trace.json"
        data = json.loads(trace.read_text())
        for event in data["traceEvents"]:
            if event["cat"] == "kernel":
                call = event["args"]["correlation"] // 3
                event["ts"] = call * 10 + {
                    "vendor_mm_1": 1, "triton_cast": 2, "vendor_mm_2": 3}[event["name"]]
        self.save(trace, data)
        result = audit.audit_collection(self.root)
        self.assertFalse(result["all_executions_passed"])
        self.assertFalse(result["paths"][1]["checks"]["kernels_match_wrapper"])

    def test_interrupted_collection_cannot_pass(self):
        self.save(self.root / "collection.json", {"status": "interrupted",
            "requested_cases": ["tail_m"], "completed_cases": ["tail_m"]})
        self.assertFalse(audit.audit_collection(self.root)["all_executions_passed"])


class CollectionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stdout = patch("sys.stdout", new_callable=io.StringIO)
        self.stdout.start()
        self.addCleanup(self.stdout.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        filename = ROOT / "experiments/integracao/modal_contrato_tma.py"
        source = ast.parse(filename.read_text())
        main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        main.decorator_list = []
        cases = ast.literal_eval(next(node.value for node in source.body if isinstance(node, ast.Assign)
                                     and any(isinstance(t, ast.Name) and t.id == "CASES" for t in node.targets)))
        self.call = types.SimpleNamespace(object_id="synthetic-call", get=Mock(), cancel=Mock())
        self.collector = types.SimpleNamespace(spawn=Mock(return_value=self.call))
        self.call.get.return_value = {"result": {"paths": {
            "original": {"status": "executed"}, "patched": {"status": "executed"}}}, "artifacts": {}}
        namespace = {"ROOT": self.root, "Path": Path, "datetime": datetime, "timezone": timezone,
                     "json": json, "CASES": cases, "collect_case": self.collector,
                     "app": types.SimpleNamespace(app_id="synthetic-app"), "__file__": str(filename)}
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(filename), "exec"), namespace)
        self.main = namespace["main"]

    def manifest(self):
        return json.loads(next(self.root.rglob("collection.json")).read_text())

    def test_default_collects_only_positive_and_persists_result(self):
        self.main()
        self.collector.spawn.assert_called_once_with("positive")
        self.call.get.assert_called_once_with(timeout=180)
        self.call.cancel.assert_not_called()
        self.assertEqual(self.manifest()["status"], "completed")
        self.assertEqual(self.manifest()["completed_cases"], ["positive"])
        self.assertNotIn("active_call_id", self.manifest())

    def test_client_timeout_cancels_remote_container_and_preserves_ids(self):
        self.call.get.side_effect = TimeoutError("synthetic timeout")
        with self.assertRaises(TimeoutError):
            self.main(cases="all")
        self.call.cancel.assert_called_once_with(terminate_containers=True)
        self.collector.spawn.assert_called_once_with("positive")
        self.assertEqual(self.manifest()["status"], "interrupted")
        self.assertEqual(self.manifest()["active_call_id"], "synthetic-call")
        self.assertTrue(self.manifest()["active_call_cancelled"])

    def test_manifest_failure_after_spawn_cancels_before_waiting(self):
        write_text = Path.write_text
        manifest_writes = 0

        def fail_after_spawn(path, *args, **kwargs):
            nonlocal manifest_writes
            if path.name == "collection.json":
                manifest_writes += 1
                if manifest_writes == 2:
                    raise OSError("synthetic manifest failure after spawn")
            return write_text(path, *args, **kwargs)

        with patch.object(Path, "write_text", fail_after_spawn):
            with self.assertRaisesRegex(OSError, "manifest failure after spawn"):
                self.main(cases="all")
        self.collector.spawn.assert_called_once_with("positive")
        self.call.get.assert_not_called()
        self.call.cancel.assert_called_once_with(terminate_containers=True)
        self.assertEqual(self.manifest()["status"], "interrupted")
        self.assertTrue(self.manifest()["active_call_cancelled"])

    def test_spawn_failure_does_not_cancel_a_nonexistent_call(self):
        self.collector.spawn.side_effect = RuntimeError("synthetic spawn failure")
        with self.assertRaisesRegex(RuntimeError, "spawn failure"):
            self.main(cases="all")
        self.call.get.assert_not_called()
        self.call.cancel.assert_not_called()
        self.assertEqual(self.manifest()["status"], "interrupted")

    def test_case_failure_preserves_artifacts_and_stops_next_case(self):
        self.call.get.return_value["result"]["paths"]["patched"]["status"] = "failed"
        with self.assertRaises(RuntimeError):
            self.main(cases="all")
        self.collector.spawn.assert_called_once_with("positive")
        self.assertEqual(self.manifest()["status"], "interrupted")
        self.assertTrue(list(self.root.rglob("positive/results.json")))


if __name__ == "__main__":
    unittest.main()
