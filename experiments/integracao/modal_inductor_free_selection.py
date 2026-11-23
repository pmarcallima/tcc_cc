"""Seleção nativa (padrão/max-autotune), sem fixar backend, TMA ou tiles."""

import json
from pathlib import Path

import modal

app = modal.App("tcc-inductor-free-selection")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "triton==3.7.1", "numpy")
         .add_local_file(str(Path(__file__).resolve().parents[2] / 'src/inductor_tma_prologue_patch.py'), "/root/inductor_tma_prologue_patch.py", copy=True))


@app.function(image=image, gpu="H100", timeout=300)
def experiment(allow_tma=False):
    import io
    import logging
    import os
    import random
    import re
    import statistics
    import time
    import types

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/tcc_free_cache"
    import torch
    import triton
    import triton.testing
    import torch._inductor.config as cfg
    from torch._inductor.scheduler import fusion_log
    import inductor_tma_prologue_patch as patch

    torch._logging.set_logs(fusion=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))

    def projection(x, weight):
        return x.to(torch.float16) @ weight

    start = time.monotonic()
    result = {"environment": {"torch": torch.__version__, "triton": triton.__version__,
                              "gpu": torch.cuda.get_device_name(0)},
              "native_config": {"gemm_backends": cfg.max_autotune_gemm_backends,
                                "persistent_tma_enabled": cfg.triton.enable_persistent_tma_matmul,
                                "prologue_fusion": cfg.prologue_fusion},
              "allow_tma": allow_tma,
              "method": {"seeds": [0, 1, 2], "rounds": 5, "rep_ms": 30,
                         "atol": 0.125, "rtol": 0.01}, "cases": []}
    artifacts = {"patch_snapshot.py": Path(patch.__file__).read_text()}
    with torch.inference_mode():
        cases = (("default", (256, 256, 256)),
                            ("max_autotune", (256, 256, 256)),
                            ("max_autotune", (2048, 1024, 256)))
        for mode, shape in cases:
            if allow_tma and mode == "default":
                continue
            torch._dynamo.reset()
            m, n, k = shape
            name = f"{mode}_{m}_{n}_{k}"
            case = {"name": name, "shape_mnk": list(shape), "mode": mode, "paths": {}}
            runners = {}
            originals = {}
            for label, enabled in (("original", False), ("patched", True)):
                trace = Path(f"/tmp/tcc_free_trace/{name}/{label}")
                log = io.StringIO()
                handler = logging.StreamHandler(log)
                fusion_log.addHandler(handler)
                decisions = []
                options = {"trace.enabled": True, "trace.output_code": True,
                           "trace.debug_dir": str(trace), "compile_threads": 1,
                           "force_disable_caches": True}
                if mode == "max_autotune":
                    options.update(max_autotune=True, max_autotune_gemm=True)
                if allow_tma:
                    options["triton.enable_persistent_tma_matmul"] = True
                try:
                    with cfg.patch(options), patch.selectable_tma_template(enabled, decisions):
                        fn = types.FunctionType(projection.__code__.replace(co_name=f"{name}_{label}"),
                                                projection.__globals__, closure=projection.__closure__)
                        runner = torch.compile(fn, fullgraph=True, dynamic=False)
                        checks = []
                        for seed in (0, 1, 2):
                            torch.manual_seed(seed)
                            x = torch.randint(-8, 9, (m, k), device="cuda", dtype=torch.int8)
                            w = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.125
                            reference = projection(x, w)
                            value = runner(x, w)
                            torch.cuda.synchronize()
                            torch.testing.assert_close(value, reference, atol=0.125, rtol=0.01)
                            if not enabled:
                                originals[seed] = value.clone()
                            checks.append({"seed": seed, "allclose": True,
                                           "max_abs_error": float((value.float() - reference.float()).abs().max()),
                                           "exact_vs_original": bool(torch.equal(value, originals[seed]))})
                    codes = list(trace.rglob("output_code.py"))
                    assert len(codes) == 1, codes
                    source = codes[0].read_text()
                    artifacts[f"{name}_{label}_output_code.py"] = source
                    launches = re.findall(r"^\s+(triton_\w+)\.run\(", source, re.MULTILINE)
                    extern = re.findall(r"^\s+(extern_kernels\.\w+)\(", source, re.MULTILINE)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(10):
                            runner(x, w)
                            torch.cuda.synchronize()
                    events = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    assert len(events) >= 10 * len(launches), events
                    case["paths"][label] = {"correctness": checks, "triton_launches": launches,
                                             "extern_calls": extern, "candidate_decisions": decisions,
                                             "extended_template_in_wrapper": "TCC_RESTRICTED_TMA_PROLOGUE" in source,
                                             "profiler": {"calls": 10, "gpu_events": len(events),
                                                          "names": sorted(set(events))}}
                    runners[label] = runner
                finally:
                    fusion_log.removeHandler(handler)
                    artifacts[f"{name}_{label}_fusion.log"] = log.getvalue()
            samples = {label: [] for label in runners}
            rng = random.Random(11)
            for _ in range(5):
                order = list(runners)
                rng.shuffle(order)
                for label in order:
                    samples[label].append(triton.testing.do_bench_cudagraph(lambda: runners[label](x, w), rep=30) * 1000)
            case["graph_samples_us"] = samples
            case["graph_medians_us"] = {label: statistics.median(values) for label, values in samples.items()}
            result["cases"].append(case)
            print(json.dumps(case))
    result["elapsed_seconds"] = time.monotonic() - start
    return {"result": result, "artifacts": artifacts}


@app.local_entrypoint()
def main(allow_tma: bool = False):
    data = experiment.remote(allow_tma)
    root = Path(__file__).resolve().parents[2] / ("artifacts/free_tma" if allow_tma else "artifacts/free_selection")
    root.mkdir(exist_ok=True)
    (root / "results.json").write_text(json.dumps(data["result"], indent=2))
    (root / "artifact_manifest.json").write_text(json.dumps(sorted(data["artifacts"]), indent=2))
    for name, text in data["artifacts"].items():
        (root / name).write_text(text)
