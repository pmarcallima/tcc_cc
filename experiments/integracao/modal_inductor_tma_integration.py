"""Valida uma extensão restrita do template TMA, com scheduler/codegen nativos."""

import json
from pathlib import Path

import modal


app = modal.App("tcc-inductor-tma-integration")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "triton==3.7.1", "numpy")
    .add_local_file(str(Path(__file__).resolve().parents[2] / 'src/inductor_tma_prologue_patch.py'), "/root/inductor_tma_prologue_patch.py", copy=True)
)


@app.function(image=image, gpu="H100", timeout=600)
def experiment():
    import io
    import logging
    import os
    import random
    import re
    import statistics
    import time
    import types

    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = "/tmp/tcc_integration_ir"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/tcc_integration_cache"
    import torch
    import torch._inductor.config as cfg
    import triton
    import triton.testing
    from torch._higher_order_ops import InvokeQuant
    from torch._inductor.scheduler import fusion_log
    from inductor_tma_prologue_patch import restricted_tma_template, fixed_tma_choice, SOURCE

    torch._logging.set_logs(fusion=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
    start = time.monotonic()
    result = {"environment": {"torch": torch.__version__, "triton": triton.__version__,
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
              "scope": "experimental in-memory template patch; forced TMA and fixed tiles; no custom_op",
              "method": {"tile": [64, 128, 64], "stages": 3, "warps": 4,
                         "seeds": [0, 1, 2], "benchmark_rounds": 7, "rep_ms": 30,
                         "prologue": "InvokeQuant(x*scale+(scale-1)); compiler arithmetic",
                         "atol": 0.125, "rtol": 0.01}, "cases": []}
    artifacts = {"patched_template.py.jinja": SOURCE}
    invoke = InvokeQuant()

    def prologue(x, scale):
        return x * scale + (scale - 1)

    def probe(x, scale, weight):
        q = invoke(prologue, x, scale, scheme="nf4", quant_options=invoke)
        return q @ weight

    config = {"max_autotune": True, "max_autotune_gemm": True,
              "max_autotune_gemm_backends": "TRITON", "force_disable_caches": True,
              "trace.enabled": True, "trace.output_code": True,
              "triton.enable_persistent_tma_matmul": True, "epilogue_fusion": False,
              "compile_threads": 1}
    with torch.inference_mode():
        for m, n, k in ((256, 256, 256), (512, 384, 256)):
            torch._dynamo.reset()
            runners = {}
            records = {}
            original_outputs = {}
            for label, patch, fusion in (("original", False, True), ("patched_split", True, False),
                                         ("patched_fused", True, True)):
                trace = Path(f"/tmp/tcc_integration_trace/{m}_{n}_{k}_{label}")
                trace.mkdir(parents=True, exist_ok=True)
                capture = io.StringIO()
                handler = logging.StreamHandler(capture)
                fusion_log.addHandler(handler)
                try:
                    with cfg.patch({**config, "prologue_fusion": fusion, "trace.debug_dir": str(trace)}):
                        with restricted_tma_template(patch) as template, fixed_tma_choice(template):
                            # Code objects independentes impedem reutilizar a última
                            # variante compilada através do cache global Dynamo.
                            distinct_probe = types.FunctionType(
                                probe.__code__.replace(co_name=f"probe_{label}_{m}_{n}_{k}"),
                                probe.__globals__, closure=probe.__closure__,
                            )
                            runner = torch.compile(distinct_probe, fullgraph=True, dynamic=False)
                            checks = []
                            for seed in (0, 1, 2):
                                torch.manual_seed(seed)
                                x = torch.randn((m, k), device="cuda", dtype=torch.float16)
                                scale = torch.randn_like(x) * 0.25 + 1
                                weight = torch.randn((k, n), device="cuda", dtype=torch.float16)
                                reference = prologue(x, scale) @ weight
                                value = runner(x, scale, weight)
                                torch.cuda.synchronize()
                                torch.testing.assert_close(value, reference, atol=0.125, rtol=0.01)
                                checks.append({"seed": seed, "allclose": True,
                                               "max_abs_error": float((value.float() - reference.float()).abs().max()),
                                               "relative_l2_error": float(torch.linalg.vector_norm(value.float() - reference.float())
                                                                          / torch.linalg.vector_norm(reference.float()))})
                                if label == "original":
                                    original_outputs[seed] = value.clone()
                                checks[-1]["exact_vs_original"] = bool(torch.equal(value, original_outputs[seed]))
                                checks[-1]["max_abs_vs_original"] = float((value.float() - original_outputs[seed].float()).abs().max())
                            runners[label] = runner
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                                for _ in range(10):
                                    runner(x, scale, weight)
                                    torch.cuda.synchronize()
                            events = [event for event in prof.events()
                                      if event.device_type == torch.autograd.DeviceType.CUDA
                                      and event.name.startswith("triton_")]
                            codes = list(trace.rglob("output_code.py"))
                            if len(codes) != 1:
                                raise RuntimeError(f"Esperado um output_code, recebidos {codes}")
                            source = codes[0].read_text()
                            prefix = f"m{m}_n{n}_k{k}_{label}"
                            artifacts[f"{prefix}_output_code.py"] = source
                            artifacts[f"{prefix}_fusion.log"] = capture.getvalue()
                            records[label] = {"correctness": checks, "kernel_events": len(events),
                                              "profiler_calls": 10, "kernels_per_call": len(events) / 10,
                                              "kernel_names": sorted({event.name for event in events}),
                                              "custom_op_present": "torch.ops.tcc" in source,
                                              "triton_launches": re.findall(r"^\s+(triton_\w+)\.run\(", source, re.MULTILINE)}
                            expected = 1 if label == "patched_fused" else 2
                            if len(events) != 10 * expected:
                                raise RuntimeError(f"Não confirmou fusão esperada: {records[label]}")
                finally:
                    fusion_log.removeHandler(handler)
            # Reutiliza as mesmas entradas em todos os caminhos já compilados.
            # Verifica novamente as identidades dos kernels fora dos contextos de
            # configuração, exatamente como serão chamados durante o benchmark.
            for label, runner in runners.items():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(10):
                        runner(x, scale, weight)
                        torch.cuda.synchronize()
                events = [event for event in prof.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA and event.name.startswith("triton_")]
                records[label]["benchmark_context_profiler"] = {
                    "calls": 10, "kernels_per_call": len(events) / 10,
                    "kernel_names": sorted({event.name for event in events}),
                }
                expected = 1 if label == "patched_fused" else 2
                if len(events) != expected * 10:
                    raise RuntimeError(f"Benchmark usaria o kernel errado: {label}, {records[label]}")
            values = {label: runner(x, scale, weight) for label, runner in runners.items()}
            torch.cuda.synchronize()
            pair = {label: {"exact_vs_original": bool(torch.equal(value, values["original"])),
                            "max_abs_vs_original": float((value.float() - values["original"].float()).abs().max())}
                    for label, value in values.items()}
            samples = {label: [] for label in runners}
            rng = random.Random(42)
            for _ in range(7):
                order = list(runners)
                rng.shuffle(order)
                for label in order:
                    samples[label].append(triton.testing.do_bench_cudagraph(lambda: runners[label](x, scale, weight), rep=30) * 1000)
            result["cases"].append({"shape_mnk": [m, n, k], "paths": records, "pair": pair,
                                    "samples_us": samples,
                                    "medians_us": {label: statistics.median(times) for label, times in samples.items()}})
            print(json.dumps(result["cases"][-1]))
    for path in Path("/tmp/tcc_integration_ir").rglob("*"):
        if path.suffix in (".ttir", ".ttgir", ".llir", ".ptx"):
            artifacts[f"ir/{path.parent.name}/{path.name}"] = path.read_text()
    result["elapsed_seconds"] = time.monotonic() - start
    return {"result": result, "artifacts": artifacts}


@app.local_entrypoint()
def main():
    data = experiment.remote()
    root = Path(__file__).resolve().parents[2] / "artifacts/inductor_integration"
    root.mkdir(exist_ok=True)
    (root / "results.json").write_text(json.dumps(data["result"], indent=2))
    for name, source in data["artifacts"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    print(json.dumps({"elapsed_seconds": data["result"]["elapsed_seconds"]}))
