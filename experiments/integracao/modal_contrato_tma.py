"""Coleta diagnóstica de contrato/fallback, sem benchmark de desempenho.

Revisar o diff antes de executar. Cada caso usa uma H100 por até 120 s,
com original e extensão, mantendo ATen e Triton comum como alternativas.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent
app = modal.App("tcc-contrato-tma")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "triton==3.7.1", "numpy")
         .add_local_file(str(ROOT / "src/inductor_tma_prologue_patch.py"),
                         "/root/inductor_tma_prologue_patch.py", copy=True))

CASES = {
    "positive": {"shape_mnk": [256, 256, 256], "operation": "cast_a"},
    "tail_m": {"shape_mnk": [257, 256, 256], "operation": "cast_a"},
    "tail_k": {"shape_mnk": [256, 256, 264], "operation": "cast_a"},
    "noncontiguous": {"shape_mnk": [256, 256, 256], "operation": "cast_a"},
    "offset": {"shape_mnk": [256, 256, 256], "operation": "plain_mm"},
    "broadcast": {"shape_mnk": [256, 256, 256], "operation": "broadcast"},
    "dtype": {"shape_mnk": [256, 256, 256], "operation": "cast_bf16"},
    "prologue_b": {"shape_mnk": [256, 256, 256], "operation": "cast_b"},
}


@app.function(image=image, gpu="H100", timeout=120, retries=0)
def collect_case(name):
    import contextlib
    import io
    import logging
    import os
    import re
    import time
    import traceback
    import types
    import uuid
    from unittest.mock import patch as replace

    run_id = uuid.uuid4().hex
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/tcc_contract_{run_id}"
    import torch
    import triton
    import torch._inductor.config as cfg
    import torch._inductor.kernel.mm as mm
    from torch._inductor.scheduler import fusion_log
    import inductor_tma_prologue_patch as extension

    spec = CASES[name]
    m, n, k = spec["shape_mnk"]
    torch.backends.cuda.matmul.allow_tf32 = False
    triton.set_allocator(lambda size, alignment, stream:
                         torch.empty(size, device="cuda", dtype=torch.int8))

    def cast_a(x, w, scale):
        return x.to(torch.float16) @ w

    def plain_mm(x, w, scale):
        return x @ w

    def broadcast(x, w, scale):
        return (x.to(torch.float16) + scale) @ w

    def cast_bf16(x, w, scale):
        return x.to(torch.bfloat16) @ w

    def cast_b(x, w, scale):
        return x @ w.to(torch.float16)

    programs = {"cast_a": cast_a, "plain_mm": plain_mm, "broadcast": broadcast,
                "cast_bf16": cast_bf16, "cast_b": cast_b}

    def inputs(seed):
        torch.manual_seed(seed)
        dtype = torch.bfloat16 if name == "dtype" else torch.float16
        x = torch.randint(-8, 9, (m, k), device="cuda", dtype=torch.int8)
        w = torch.randn((k, n), device="cuda", dtype=dtype) * 0.125
        if name == "noncontiguous":
            x = torch.randint(-8, 9, (k, m), device="cuda", dtype=torch.int8).t()
        elif name == "offset":
            # View contígua com offset de 8 elementos fp16, alinhado em 16 bytes.
            x = torch.randn(m * k + 8, device="cuda", dtype=dtype)[8:].view(m, k)
        elif name == "prologue_b":
            x = torch.randn((m, k), device="cuda", dtype=dtype) * 0.125
            w = torch.randint(-8, 9, (k, n), device="cuda", dtype=torch.int8)
        scale = torch.randn((1, k), device="cuda", dtype=dtype) * 0.125
        return x, w, scale

    @contextlib.contextmanager
    def observe_and_limit(template, record):
        """Um candidato por template; preserva ATen e o Triton comum.

        O tile TMA é diagnóstico/fixo, não uma avaliação da seleção livre.
        Não modifica a elegibilidade nativa nem as regras do scheduler.
        """
        native_eligibility = mm.use_triton_tma_template
        error_init = extension.ContractError.__init__
        select_algorithm = mm.autotune_select_algorithm

        record["offered_choices"] = []

        def selection(operation, choices, *args, **kwargs):
            # Observação independente dos contadores do limitador.
            record["offered_choices"].append({
                "operation": operation,
                "names": [str(choice.name) for choice in choices],
            })
            return select_algorithm(operation, choices, *args, **kwargs)

        def eligibility(*args, **kwargs):
            eligible = native_eligibility(*args, **kwargs)
            record["native_tma_eligibility"].append(bool(eligible))
            return eligible

        def guard_error(error, message, reason="prologue_contract"):
            error_init(error, message, reason)
            record["prologue_rejections"].append({"reason": error.reason,
                                                    "message": str(error)})

        record["candidate_limits"] = {
            kind: {"attempted": 0, "generated": 0, "skipped": 0, "config": None}
            for kind in ("common", "tma")
        }

        def limited(original, kind):
            counts = record["candidate_limits"][kind]

            def append(choices, **kwargs):
                # Cada controle contém um único matmul. O compilador pode usar
                # listas temporárias distintas para suas configurações.
                if counts["attempted"]:
                    counts["skipped"] += 1
                    return None
                counts["attempted"] += 1
                if kind == "tma":
                    kwargs.update(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64,
                                  num_stages=3, num_warps=4)
                counts["config"] = {key: kwargs.get(key) for key in (
                    "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps")}
                previous = len(choices)
                try:
                    return original(choices, **kwargs)
                finally:
                    counts["generated"] += len(choices) - previous

            return append

        with contextlib.ExitStack() as stack:
            stack.enter_context(replace.object(mm, "autotune_select_algorithm", selection))
            stack.enter_context(replace.object(mm, "use_triton_tma_template", eligibility))
            stack.enter_context(replace.object(extension.ContractError, "__init__", guard_error))
            stack.enter_context(replace.object(mm.mm_template, "maybe_append_choice",
                                                limited(mm.mm_template.maybe_append_choice, "common")))
            stack.enter_context(replace.object(template, "maybe_append_choice",
                                                limited(template.maybe_append_choice, "tma")))
            yield

    started = time.monotonic()
    artifacts = {"patch_snapshot.py": Path(extension.__file__).read_text()}
    result = {"name": name, **spec,
              "environment": {"torch": str(torch.__version__), "triton": triton.__version__,
                              "gpu": torch.cuda.get_device_name(0),
                              "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid)},
              "paths": {}}
    with torch.inference_mode():
        for label, enabled in (("original", False), ("patched", True)):
            torch._dynamo.reset()
            trace_dir = Path(f"/tmp/tcc_contract_trace/{run_id}/{label}")
            record = {"status": "failed", "candidate_decisions": [],
                      "native_tma_eligibility": [], "prologue_rejections": [],
                      "correctness": []}
            result["paths"][label] = record
            capture = io.StringIO()
            handler = logging.StreamHandler(capture)
            previous_level = fusion_log.level
            fusion_log.setLevel(logging.DEBUG)
            fusion_log.addHandler(handler)
            options = {"max_autotune": True, "max_autotune_gemm": True,
                       "max_autotune_gemm_backends": "ATEN,TRITON",
                       "max_epilogue_benchmarked_choices": 8,
                       "triton.enable_persistent_tma_matmul": True,
                       "prologue_fusion": True, "epilogue_fusion": False,
                       "force_disable_caches": True, "compile_threads": 1,
                       "trace.enabled": True, "trace.output_code": True,
                       "trace.debug_dir": str(trace_dir)}
            record["config"] = options
            try:
                with cfg.patch(options), extension.selectable_tma_template(
                    enabled, record["candidate_decisions"], declare_b=True
                ) as template, observe_and_limit(template, record):
                    program = programs[spec["operation"]]
                    distinct = types.FunctionType(
                        program.__code__.replace(co_name=f"contract_{name}_{label}"),
                        program.__globals__, closure=program.__closure__)
                    runner = torch.compile(distinct, fullgraph=True, dynamic=False)
                    for seed in (0, 1, 2):
                        args = inputs(seed)
                        reference = program(*args)
                        value = runner(*args)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(value, reference, atol=0.125, rtol=0.01)
                        record["correctness"].append({"seed": seed, "allclose": True,
                            "max_abs_error": float((value.float() - reference.float()).abs().max())})
                    record["input_metadata"] = [{"dtype": str(t.dtype), "shape": list(t.shape),
                        "stride": list(t.stride()), "offset": t.storage_offset()} for t in args]
                    codes = list(trace_dir.rglob("output_code.py"))
                    if len(codes) != 1:
                        raise RuntimeError(f"Esperado um wrapper; obtidos {len(codes)}")
                    source = codes[0].read_text()
                    artifacts[f"{label}_output_code.py"] = source
                    outputs = []
                    with torch.compiler.set_stance("fail_on_recompile"):
                        runner(*args)
                        torch.cuda.synchronize()
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
                            for index in range(3):
                                with torch.profiler.record_function(f"contract/{name}/{label}/call_{index}"):
                                    outputs.append(runner(*args))
                                    torch.cuda.synchronize()
                    for output in outputs:
                        torch.testing.assert_close(output, reference, atol=0.125, rtol=0.01)
                    trace_file = trace_dir / "runtime_trace.json"
                    prof.export_chrome_trace(str(trace_file))
                    artifacts[f"{label}_runtime_trace.json"] = trace_file.read_text()
                    record.update(status="executed", profile_outputs_correct=True,
                                  recompile_guard="fail_on_recompile", profile_calls=3,
                                  triton_launches=re.findall(r"^\s+(triton_\w+)\.run\(", source, re.M),
                                  extern_calls=re.findall(r"^\s+(extern_kernels\.\w+)\(", source, re.M),
                                  extension_in_wrapper="TCC_RESTRICTED_TMA_PROLOGUE" in source)
            except Exception:
                record["error"] = traceback.format_exc()
            finally:
                fusion_log.removeHandler(handler)
                fusion_log.setLevel(previous_level)
                artifacts[f"{label}_fusion.log"] = capture.getvalue()
            print(json.dumps({"case": name, "label": label, "status": record["status"],
                              "filter_calls": len(record["candidate_decisions"]),
                              "prologue_rejections": len(record["prologue_rejections"])}))
    result["elapsed_seconds"] = time.monotonic() - started
    return {"result": result, "artifacts": artifacts}


@app.local_entrypoint()
def main(cases: str = "positive"):
    """Começa só pelo positivo; --cases all executa os oito casos em sequência."""
    names = list(CASES) if cases == "all" else cases.split(",")
    if not names or len(names) != len(set(names)) or any(name not in CASES for name in names):
        raise ValueError(f"--cases deve conter nomes distintos de {list(CASES)} ou all")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = ROOT / "artifacts/contrato_tma" / stamp
    root.mkdir(parents=True, exist_ok=False)
    (root / "experiment_snapshot.py").write_text(Path(__file__).read_text())
    manifest = {"status": "collecting", "requested_cases": names, "completed_cases": [],
                "modal_app_id": app.app_id,
                "method": {"seeds": [0, 1, 2], "profile_calls": 3,
                           "atol": 0.125, "rtol": 0.01, "case_timeout_seconds": 120,
                           "client_wait_seconds": 180,
                           "tma_tile": [64, 128, 64], "tma_warps": 4, "tma_stages": 3,
                           "candidates": "one per common/TMA template; ATen retained",
                           "scope": "diagnostic contract/fallback; not free selection or speedup"}}

    def save():
        (root / "collection.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    save()
    print(f"Evidências: {root}", flush=True)
    try:
        for index, name in enumerate(names, 1):
            print(f"[{index}/{len(names)}] Iniciando {name}; timeout remoto 120 s", flush=True)
            call = None
            try:
                call = collect_case.spawn(name)
                manifest["active_call_id"] = call.object_id
                save()
                data = call.get(timeout=180)
            except BaseException:
                # Inclui falha na persistência após spawn, timeout e interrupção.
                if call is not None:
                    try:
                        call.cancel(terminate_containers=True)
                        manifest["active_call_cancelled"] = True
                    except Exception as cancellation_error:
                        manifest["cancellation_error"] = str(cancellation_error)
                raise
            directory = root / name
            directory.mkdir()
            (directory / "results.json").write_text(json.dumps(data["result"], indent=2))
            for filename, content in data["artifacts"].items():
                (directory / filename).write_text(content)
            manifest["completed_cases"].append(name)
            manifest.pop("active_call_id", None)
            save()
            if any(row["status"] != "executed" for row in data["result"]["paths"].values()):
                raise RuntimeError(f"Falha em {name}; coleta interrompida, consulte {directory}")
            print(f"[{index}/{len(names)}] Concluído: {name}", flush=True)
    except BaseException as error:
        manifest.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        save()
        raise
    manifest["status"] = "completed"
    save()
    print(f"Coleta concluída: {root}", flush=True)
