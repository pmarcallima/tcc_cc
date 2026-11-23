"""Seleção livre instrumentada: a extensão TMA é medida como kernel fundido?

Protocolo: docs/experimentos/PROTOCOLO_SELECAO_PROLOGO_AB.md. Cada braço roda em
um container próprio. Registra, por caso, as escolhas com tempo sem fusão, os
conjuntos de entradas de prólogo, as escolhas medidas fundidas e o wrapper final.
Use --profile-runtime para exportar Chrome traces da execução após o autotuning,
antes de limpar o cache; abra os arquivos *_runtime_trace.json no Perfetto.
"""

import json
from pathlib import Path

import modal

# No container o arquivo fica em /root; ROOT só é usado localmente.
_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent
app = modal.App("tcc-selecao-prologo-ab")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "triton==3.7.1", "numpy")
         .add_local_file(str(ROOT / "src/inductor_tma_prologue_patch.py"),
                         "/root/inductor_tma_prologue_patch.py", copy=True))

SHAPES_7 = [(256, 256, 256), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
            (1024, 256, 1024), (1024, 512, 1024), (1024, 4096, 1024)]
ARMS = {
    "arm0": {"declare_b": False, "cap": None, "shapes": [(256, 256, 256), (2048, 1024, 256)],
             "negatives": False, "original": False},
    "arm1": {"declare_b": True, "cap": None, "shapes": SHAPES_7, "negatives": True, "original": True},
    "arm2": {"declare_b": True, "cap": 1000, "shapes": SHAPES_7, "negatives": True, "original": False},
}


@app.function(image=image, gpu="H100", timeout=3600)
def run_arm(arm_name, smoke=False, profile_runtime=False):
    import contextlib
    import io
    import logging
    import os
    import random
    import re
    import statistics
    import time
    import traceback
    import types

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/tcc_ab_cache_{arm_name}"
    import torch
    import triton
    import triton.testing
    import torch._inductor.config as cfg
    from torch._inductor import ir
    from torch._inductor import scheduler as sched_mod
    from torch._inductor.scheduler import fusion_log
    import inductor_tma_prologue_patch as patch

    arm = dict(ARMS[arm_name])
    if smoke:  # só confere a instrumentação, sem valor de resultado
        arm["shapes"] = [(256, 256, 256)]
    torch._logging.set_logs(fusion=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))

    # --- instrumentação -------------------------------------------------------
    state = {"case": None, "swapped": None}

    def describe(choice):
        info = {}
        with contextlib.suppress(Exception):
            info = {k: str(v) for k, v in choice.info_dict().items()}
        return {"str": str(choice), "type": type(choice).__name__,
                "allowed_prologue_inps": sorted(getattr(choice, "allowed_prologue_inps", []) or []),
                "info": info}

    orig_timings = ir.MultiTemplateBuffer.choice_timings

    def choice_timings(self, *args, **kwargs):
        timings = orig_timings(self, *args, **kwargs)
        case = state["case"]
        if case is not None and id(self) not in case["_seen_multi"]:
            case["_seen_multi"].add(id(self))
            union = sorted(getattr(self, "allowed_prologue_inps", []) or [])
            rows = []
            for choice, t in sorted(timings.items(), key=lambda kv: kv[1]):
                d = describe(choice)
                d["unfused_ms"] = t
                d["matches_union"] = d["allowed_prologue_inps"] == union
                rows.append(d)
            case["multi_nodes"].append({"union_allowed_prologue_inps": union, "choices": rows})
        return timings

    ir.MultiTemplateBuffer.choice_timings = choice_timings

    orig_swap = ir.MultiTemplateBuffer.swap_as_triton_caller

    @contextlib.contextmanager
    def swap(self, caller):
        previous = state["swapped"]
        state["swapped"] = caller
        try:
            with orig_swap(self, caller):
                yield
        finally:
            state["swapped"] = previous

    ir.MultiTemplateBuffer.swap_as_triton_caller = swap

    orig_compile = sched_mod.Scheduler.compile_kernel

    def compile_kernel(self, nodes, *args, **kwargs):
        case = state["case"]
        caller = state["swapped"]
        entry = None
        if case is not None and caller is not None:
            entry = {"choice": str(caller), "status": "compiled"}
            case["fused_compiles"].append(entry)
        try:
            return orig_compile(self, nodes, *args, **kwargs)
        except Exception as e:
            if entry is not None:
                entry["status"] = f"{type(e).__name__}: {e}"[:500]
            raise

    sched_mod.Scheduler.compile_kernel = compile_kernel

    orig_bench = sched_mod.Scheduler.benchmark_codegened_module

    def benchmark_codegened_module(self, *args, **kwargs):
        out = orig_bench(self, *args, **kwargs)
        case = state["case"]
        if case is not None and state["swapped"] is not None:
            case["fused_benchmarks"].append({"choice": str(state["swapped"]), "fused_ms": out[0]})
        return out

    sched_mod.Scheduler.benchmark_codegened_module = benchmark_codegened_module

    def capture_runtime(runner, a, b, case, reference_program):
        """Executa o wrapper já compilado; uma recompilação invalida a captura."""
        profile_record = {"calls": 5}
        case["runtime_profile"] = profile_record
        try:
            with torch.compiler.set_stance("fail_on_recompile"):
                for _ in range(3):
                    runner(a, b)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]) as prof:
                    for call_index in range(profile_record["calls"]):
                        with torch.profiler.record_function(
                            f"{arm_name}/{case['name']}/{case['label']}/call_{call_index}"
                        ):
                            value = runner(a, b)
                            torch.cuda.synchronize()
            trace_name = f"{case['name']}_{case['label']}_runtime_trace.json"
            trace_path = Path("/tmp") / trace_name
            prof.export_chrome_trace(str(trace_path))
            artifacts[trace_name] = trace_path.read_text()
            # Anotações replicadas na GPU não são kernels executados.
            chrome = json.loads(artifacts[trace_name])
            events = [e["name"] for e in chrome["traceEvents"]
                      if e.get("cat") == "kernel" and e.get("ph") == "X"]
            expected = (len(case["wrapper"]["triton_launches"])
                        + len(case["wrapper"]["extern_calls"]))
            profile_record.update(
                trace_file=trace_name, gpu_events=len(events),
                kernels_per_call=len(events) / profile_record["calls"],
                kernel_names=sorted(set(events)), expected_kernels_per_call=expected,
                count_matches_wrapper=len(events) == expected * profile_record["calls"],
                recompile_guard="fail_on_recompile",
            )
            reference = reference_program(a, b)
            torch.testing.assert_close(value, reference, atol=0.125, rtol=0.01)
            profile_record["correctness_passed"] = True
            if not profile_record["count_matches_wrapper"]:
                raise RuntimeError("Contagem CUDA difere das chamadas do wrapper; inspecione o trace")
        except Exception:
            profile_record["error"] = traceback.format_exc()[-4000:]

    # --- programas ---------------------------------------------------------------
    def pos(x, w):
        return x.to(torch.float16) @ w

    def neg_b(x, w):
        return x @ w.to(torch.float16)

    def neg_a(xt, w):
        return xt.t().to(torch.float16) @ w

    def inputs(kind, m, n, k, seed):
        torch.manual_seed(seed)
        if kind == "pos":
            return (torch.randint(-8, 9, (m, k), device="cuda", dtype=torch.int8),
                    torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.125)
        if kind == "neg_b":
            return (torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.125,
                    torch.randint(-8, 9, (k, n), device="cuda", dtype=torch.int8))
        return (torch.randint(-8, 9, (k, m), device="cuda", dtype=torch.int8),
                torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.125)

    programs = {"pos": pos, "neg_b": neg_b, "neg_a": neg_a}

    plan = [("pos", s) for s in arm["shapes"]]
    if arm["negatives"]:
        plan += [("neg_b", (1024, 1024, 1024)), ("neg_a", (1024, 1024, 1024))]
    labels = [("patched", True)] + ([("original", False)] if arm["original"] else [])

    start = time.monotonic()
    result = {"arm": arm_name, "arm_config": {k: v for k, v in arm.items() if k != "shapes"},
              "profile_runtime": profile_runtime,
              "environment": {"torch": torch.__version__, "triton": triton.__version__,
                              "gpu": torch.cuda.get_device_name(0),
                              "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid)},
              "native_config": {"benchmark_epilogue_fusion": cfg.benchmark_epilogue_fusion,
                                "max_epilogue_benchmarked_choices": cfg.max_epilogue_benchmarked_choices,
                                "prologue_fusion": cfg.prologue_fusion,
                                "gemm_backends": cfg.max_autotune_gemm_backends},
              "cases": []}
    artifacts = {"patch_snapshot.py": Path(patch.__file__).read_text()}

    with torch.inference_mode():
        for kind, (m, n, k) in plan:
            # Não invalidar o runner patched antes de executar original/latência.
            torch._dynamo.reset()
            name = f"{kind}_{m}_{n}_{k}"
            runners = {}
            for label, enabled in labels:
                case = {"name": name, "kind": kind, "shape_mnk": [m, n, k], "label": label,
                        "multi_nodes": [], "fused_compiles": [], "fused_benchmarks": [],
                        "_seen_multi": set()}
                trace = Path(f"/tmp/tcc_ab_trace/{arm_name}/{name}/{label}")
                log = io.StringIO()
                handler = logging.StreamHandler(log)
                handler.setLevel(logging.DEBUG)
                fusion_log.addHandler(handler)
                previous_level = fusion_log.level
                fusion_log.setLevel(logging.DEBUG)
                decisions = []
                options = {"trace.enabled": True, "trace.output_code": True,
                           "trace.debug_dir": str(trace), "compile_threads": 1,
                           "force_disable_caches": True, "max_autotune": True,
                           "max_autotune_gemm": True,
                           "triton.enable_persistent_tma_matmul": True}
                if arm["cap"] is not None:
                    options["max_epilogue_benchmarked_choices"] = arm["cap"]
                state["case"] = case
                try:
                    with cfg.patch(options), patch.selectable_tma_template(enabled, decisions, arm["declare_b"]):
                        fn = types.FunctionType(programs[kind].__code__.replace(co_name=f"{arm_name}_{name}_{label}"),
                                                programs[kind].__globals__,
                                                closure=programs[kind].__closure__)
                        runner = torch.compile(fn, fullgraph=True, dynamic=False)
                        checks = []
                        for seed in (0, 1, 2):
                            a, b = inputs(kind, m, n, k, seed)
                            reference = programs[kind](a, b)
                            value = runner(a, b)
                            torch.cuda.synchronize()
                            torch.testing.assert_close(value, reference, atol=0.125, rtol=0.01)
                            checks.append({"seed": seed, "max_abs_error":
                                           float((value.float() - reference.float()).abs().max())})
                    case["correctness"] = checks
                    codes = list(trace.rglob("output_code.py"))
                    source = codes[-1].read_text() if codes else ""
                    artifacts[f"{name}_{label}_output_code.py"] = source
                    case["wrapper"] = {
                        "triton_launches": re.findall(r"^\s+(triton_\w+)\.run\(", source, re.MULTILINE),
                        "extern_calls": re.findall(r"^\s+(extern_kernels\.\w+)\(", source, re.MULTILINE),
                        "extended_template_in_wrapper": "TCC_RESTRICTED_TMA_PROLOGUE" in source,
                        "tma_descriptor_in_wrapper": "make_tensor_descriptor" in source}
                    runners[label] = (runner, a, b)
                    if profile_runtime:
                        # Mesmas opções e extensão da compilação, antes do reset.
                        with cfg.patch(options), patch.selectable_tma_template(enabled, decisions, arm["declare_b"]):
                            capture_runtime(runner, a, b, case, programs[kind])
                except Exception:
                    case["error"] = traceback.format_exc()[-4000:]
                finally:
                    state["case"] = None
                    fusion_log.removeHandler(handler)
                    fusion_log.setLevel(previous_level)
                    artifacts[f"{name}_{label}_fusion.log"] = log.getvalue()
                case["candidate_decisions"] = decisions
                del case["_seen_multi"]
                result["cases"].append(case)
                print(json.dumps({k: case.get(k) for k in ("name", "label", "wrapper", "fused_benchmarks", "error")})[:3000])
            # Latência é registro auxiliar: uma falha aqui não invalida o caso.
            samples = {label: [] for label in runners}
            timing = {"name": name, "timing_us": samples}
            try:
                rng = random.Random(11)
                for _ in range(5):
                    order = list(runners)
                    rng.shuffle(order)
                    for label in order:
                        r, a, b = runners[label]
                        samples[label].append(triton.testing.do_bench_cudagraph(lambda: r(a, b), rep=30) * 1000)
                timing["median_us"] = {l: statistics.median(v) for l, v in samples.items()}
            except Exception:
                timing["error"] = traceback.format_exc()[-2000:]
                torch.cuda.synchronize()
            result["cases"].append(timing)
    result["elapsed_seconds"] = time.monotonic() - start
    # Sanitiza objetos do torch: o cliente local não tem torch para desserializar.
    return {"result": json.loads(json.dumps(result, default=str)), "artifacts": artifacts}


@app.local_entrypoint()
def main(arms: str = "arm0,arm1,arm2", smoke: bool = False, profile_runtime: bool = False):
    from datetime import datetime, timezone

    names = arms.split(",")
    experiment_snapshot = Path(__file__).read_text()
    calls = {name: run_arm.spawn(name, smoke, profile_runtime) for name in names}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = ROOT / "artifacts/selecao_prologo_ab" / (stamp + ("_smoke" if smoke else ""))
    root.mkdir(parents=True, exist_ok=False)
    for name, call in calls.items():
        d = root / name
        d.mkdir()
        try:
            data = call.get()
        except Exception as e:  # salva os demais braços mesmo se um falhar
            (d / "error.txt").write_text(f"{type(e).__name__}: {e}")
            continue
        (d / "results.json").write_text(json.dumps(data["result"], indent=2, default=str))
        for fname, text in data["artifacts"].items():
            (d / fname).write_text(text)
    (root / "experiment_snapshot.py").write_text(experiment_snapshot)
    print(root)
