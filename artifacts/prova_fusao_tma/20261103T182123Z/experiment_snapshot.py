"""Sondas sem InvokeQuant: decisões nativas de fusão com template TMA estendido."""

import json
from pathlib import Path

import modal


_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent
app = modal.App("tcc-inductor-plain-prologues")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "triton==3.7.1", "numpy")
    .add_local_file(str(ROOT / 'src/inductor_tma_prologue_patch.py'), "/root/inductor_tma_prologue_patch.py", copy=True)
)


@app.function(image=image, gpu="H100", timeout=240)
def experiment(suite="plain", proof_trace=False):
    import io
    import logging
    import os
    import random
    import re
    import statistics
    import time
    import traceback
    import types

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/tcc_plain_cache"
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = "/tmp/tcc_plain_ir"
    import torch
    import torch._inductor.config as cfg
    import triton
    import triton.testing
    from torch._inductor.scheduler import fusion_log
    import inductor_tma_prologue_patch as patch_module
    from inductor_tma_prologue_patch import restricted_tma_template, fixed_tma_choice

    torch._logging.set_logs(fusion=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))

    def relu_projection(x, scale, weight):
        return torch.relu(x) @ weight

    def affine_projection(x, scale, weight):
        return (x * scale + (scale - 1)) @ weight

    def cast_projection(x, scale, weight):
        return x.to(torch.float16) @ weight

    start = time.monotonic()
    result = {"environment": {"torch": torch.__version__, "triton": triton.__version__,
                              "gpu": torch.cuda.get_device_name(0),
                              "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid)},
              "proof_trace": proof_trace,
              "scope": "ordinary PyTorch expressions, no InvokeQuant/custom_op; forced fixed TMA choices",
              "method": {"tile": [64, 128, 64], "stages": 3, "warps": 4,
                         "seeds": [0, 1, 2], "rounds": 5, "graph_rep_ms": 30,
                         "atol": 0.125, "rtol": 0.01, "scheduler_heuristics": "unchanged"},
              "cases": []}
    artifacts = {"patch_snapshot.py": Path(patch_module.__file__).read_text()}
    config = {"max_autotune": True, "max_autotune_gemm": True,
              "max_autotune_gemm_backends": "TRITON", "force_disable_caches": True,
              "trace.enabled": True, "trace.output_code": True,
              "triton.enable_persistent_tma_matmul": True, "epilogue_fusion": False,
              "compile_threads": 1}
    with torch.inference_mode():
        cases = (("cast_small", cast_projection, 256, 256, 256),
                 ("cast_persistent", cast_projection, 2048, 1024, 256)) if suite == "cast" else (
                     ("relu_small", relu_projection, 256, 256, 256),
                     ("relu_persistent", relu_projection, 2048, 1024, 256),
                     ("affine_unmarked", affine_projection, 256, 256, 256))
        for name, fn, m, n, k in cases:
            torch._dynamo.reset()
            record = {"name": name, "shape_mnk": [m, n, k], "paths": {}}
            runners = {}
            original_values = {}
            for label, enabled, fusion in (("original", False, True), ("patched_split", True, False),
                                           ("patched_enabled", True, True)):
                existing_ir = set(Path("/tmp/tcc_plain_ir").rglob("*"))
                trace = Path(f"/tmp/tcc_plain_trace/{name}/{label}")
                trace.mkdir(parents=True, exist_ok=True)
                capture = io.StringIO()
                handler = logging.StreamHandler(capture)
                fusion_log.addHandler(handler)
                try:
                    with cfg.patch({**config, "prologue_fusion": fusion, "trace.debug_dir": str(trace)}):
                        with restricted_tma_template(enabled) as template, fixed_tma_choice(template):
                            distinct = types.FunctionType(fn.__code__.replace(co_name=f"{name}_{label}"), fn.__globals__, closure=fn.__closure__)
                            runner = torch.compile(distinct, fullgraph=True, dynamic=False)
                            checks = []
                            for seed in (0, 1, 2):
                                torch.manual_seed(seed)
                                x = (torch.randint(-8, 9, (m, k), device="cuda", dtype=torch.int8)
                                     if suite == "cast" else torch.randn((m, k), device="cuda", dtype=torch.float16))
                                scale = torch.ones((m, k), device="cuda", dtype=torch.float16)
                                if suite != "cast":
                                    scale = torch.randn_like(scale) * 0.25 + 1
                                weight = torch.randn((k, n), device="cuda", dtype=torch.float16)
                                if suite == "cast":
                                    weight = weight * 0.125
                                reference = fn(x, scale, weight)
                                value = runner(x, scale, weight)
                                torch.cuda.synchronize()
                                torch.testing.assert_close(value, reference, atol=0.125, rtol=0.01)
                                if label == "original":
                                    original_values[seed] = value.clone()
                                checks.append({"seed": seed, "allclose": True,
                                               "max_abs_error": float((value.float() - reference.float()).abs().max()),
                                               "exact_vs_original": bool(torch.equal(value, original_values[seed]))})
                            codes = list(trace.rglob("output_code.py"))
                            if len(codes) != 1:
                                raise RuntimeError(f"Esperado um wrapper: {codes}")
                            source = codes[0].read_text()
                            artifacts[f"{name}_{label}_output_code.py"] = source
                            launches = re.findall(r"^\s+(triton_\w+)\.run\(", source, re.MULTILINE)
                            if not 1 <= len(launches) <= 2:
                                raise RuntimeError(f"Estrutura inesperada: {launches}")
                            record["paths"][label] = {"status": "executed", "correctness": checks,
                                                      "triton_launches": launches,
                                                      "tensor_buffers": re.findall(r"\b(buf\d+) = empty_strided_cuda", source),
                                                      "fused_in_generated_code": len(launches) == 1}
                            if proof_trace:
                                expected = 1 if label == "patched_enabled" else 2
                                if len(launches) != expected:
                                    raise RuntimeError(f"Controle não confirmou a estrutura esperada: {label}/{launches}")
                                # Captura a mesma versão compilada, ainda sob as
                                # opções e o template do controle, antes do reset.
                                outputs = []
                                with torch.compiler.set_stance("fail_on_recompile"):
                                    for _ in range(3):
                                        runner(x, scale, weight)
                                    torch.cuda.synchronize()
                                    with torch.profiler.profile(activities=[
                                        torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA,
                                    ]) as prof:
                                        for call_index in range(5):
                                            with torch.profiler.record_function(f"{name}/{label}/call_{call_index}"):
                                                outputs.append(runner(x, scale, weight))
                                                torch.cuda.synchronize()
                                trace_name = f"{name}_{label}_runtime_trace.json"
                                trace_path = Path("/tmp") / trace_name
                                prof.export_chrome_trace(str(trace_path))
                                artifacts[trace_name] = trace_path.read_text()
                                chrome = json.loads(artifacts[trace_name])
                                kernels = [e for e in chrome["traceEvents"]
                                           if e.get("cat") == "kernel" and e.get("ph") == "X"]
                                if len(kernels) != 5 * expected:
                                    raise RuntimeError(f"Trace divergente: {label}/{len(kernels)} kernels")
                                for output in outputs:
                                    torch.testing.assert_close(output, reference, atol=0.125, rtol=0.01)
                                record["paths"][label]["runtime_profile"] = {
                                    "trace_file": trace_name, "calls": 5, "gpu_events": len(kernels),
                                    "kernels_per_call": len(kernels) / 5,
                                    "kernel_names": sorted({e["name"] for e in kernels}),
                                    "recompile_guard": "fail_on_recompile", "all_outputs_correct": True,
                                }
                                proof_ir = []
                                for path in sorted(set(Path("/tmp/tcc_plain_ir").rglob("*")) - existing_ir):
                                    if path.suffix in (".ttir", ".ttgir", ".ptx"):
                                        key = f"proof_ir/{name}/{label}/{path.parent.name}/{path.name}"
                                        artifacts[key] = path.read_text()
                                        proof_ir.append(key)
                                record["paths"][label]["ir_files"] = proof_ir
                            runners[label] = runner
                except Exception:
                    error = traceback.format_exc()
                    artifacts[f"{name}_{label}_error.log"] = error
                    record["paths"][label] = {"status": "failed", "error": error}
                finally:
                    artifacts[f"{name}_{label}_fusion.log"] = capture.getvalue()
                    fusion_log.removeHandler(handler)

            if proof_trace:
                # Esta rodada prova estrutura e correção; não mede speedup.
                result["cases"].append(record)
                print(json.dumps({"name": name, "proof": record["paths"]}))
                continue
            for label, runner in runners.items():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(10):
                        runner(x, scale, weight)
                        torch.cuda.synchronize()
                events = [event for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA
                          and event.name.startswith("triton_")]
                expected = len(record["paths"][label]["triton_launches"])
                if len(events) != 10 * expected:
                    raise RuntimeError(f"Profiler divergente: {name}/{label}, {len(events)} eventos")
                record["paths"][label]["profiler"] = {"calls": 10, "kernels_per_call": len(events) / 10,
                                                      "names": sorted({event.name for event in events})}
            samples = {label: [] for label in runners}
            rng = random.Random(11)
            for _ in range(5):
                order = list(runners)
                rng.shuffle(order)
                for label in order:
                    samples[label].append(triton.testing.do_bench_cudagraph(lambda: runners[label](x, scale, weight), rep=30) * 1000)
            record["graph_samples_us"] = samples
            record["graph_medians_us"] = {label: statistics.median(values) for label, values in samples.items()}
            result["cases"].append(record)
            print(json.dumps({"name": name, "status": {key: value["status"] for key, value in record["paths"].items()},
                              "kernels": {key: value.get("profiler", {}).get("kernels_per_call") for key, value in record["paths"].items()},
                              "medians": record["graph_medians_us"]}))
    for path in Path("/tmp/tcc_plain_ir").rglob("*"):
        if path.suffix in (".ttir", ".ttgir", ".ptx"):
            artifacts[f"ir/{path.parent.name}/{path.name}"] = path.read_text()
    result["elapsed_seconds"] = time.monotonic() - start
    result["suite"] = suite
    # TorchVersion é uma subclasse de str que exige torch para desserializar.
    return {"result": json.loads(json.dumps(result, default=str)), "artifacts": artifacts}


@app.local_entrypoint()
def main(suite: str = "plain", proof_trace: bool = False):
    if suite not in ("plain", "cast"):
        raise ValueError("suite deve ser plain ou cast")
    if proof_trace and suite != "cast":
        raise ValueError("--proof-trace é específico da prova controlada do cast")
    snapshot = Path(__file__).read_text()
    data = experiment.remote(suite, proof_trace)
    base = Path(__file__).resolve().parents[2]
    if proof_trace:
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = base / "artifacts/prova_fusao_tma" / stamp
        root.mkdir(parents=True, exist_ok=False)
        (root / "experiment_snapshot.py").write_text(snapshot)
    else:
        root = base / ("artifacts/inductor_cast" if suite == "cast" else "artifacts/inductor_plain")
        root.mkdir(exist_ok=True)
    (root / "results.json").write_text(json.dumps(data["result"], indent=2))
    (root / "artifact_manifest.json").write_text(json.dumps(sorted(data["artifacts"]), indent=2))
    for name, source in data["artifacts"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    print(json.dumps({"elapsed_seconds": data["result"]["elapsed_seconds"], "output": str(root)}))
