"""Valida e mede o protótipo manual TMA em uma única H100, com tempo limitado."""

import json
from pathlib import Path

import modal


app = modal.App("tcc-tma-prologue-prototype")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "triton==3.7.1", "numpy")
    .add_local_file(str(Path(__file__).resolve().parents[2] / 'src/tma_prologue_prototype.py'), "/root/tma_prologue_prototype.py", copy=True)
)


@app.function(image=image, gpu="H100", timeout=600)
def experiment():
    import random
    import statistics
    import time

    import torch
    import triton
    import triton.testing
    from tma_prologue_prototype import TMAPrologueCase

    start_time = time.monotonic()
    result = {
        "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "triton": triton.__version__,
                        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count},
        "scope": "manual persistent+TMA prototype; not a TorchInductor patch",
        "method": {"blocks": [128, 128, 64], "warps": 4, "stages": 2,
                   "enable_fp_fusion": False, "seeds": [0, 1, 2],
                   "prologue": "fp32(x)*fp32(scale)+(fp32(scale)-1), then fp16",
                   "warmup": 20, "rounds": 30, "calls_per_round": 20,
                   "clock_control": "unmodified", "cuda_graph_repeats": 3},
        "cases": [],
    }
    artifacts = {}
    shapes = [(128, 128, 128), (256, 256, 256), (512, 512, 512),
              (1024, 1024, 1024), (2048, 2048, 2048), (192, 320, 128)]
    for m, n, k in shapes:
        record = {"shape_mnk": [m, n, k], "correctness": []}
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            x = torch.randn((m, k), device="cuda", dtype=torch.float16)
            scale = torch.randn_like(x)
            weight = torch.randn((k, n), device="cuda", dtype=torch.float16)
            case = TMAPrologueCase(x, scale, weight)
            fused, split = case.fused(), case.split()
            torch.cuda.synchronize()
            q_ref = (x.float() * scale.float() + (scale.float() - 1)).half()
            # Referência de matmul em fp32, arredondada à saída fp16.
            reference = (q_ref.float() @ weight.float()).half()
            q_equal = bool(torch.equal(case.q, q_ref))
            pair_equal = bool(torch.equal(fused, split))
            # Tolerância preliminar; registra erros normalizados e máximos também.
            atol, rtol = 0.125, 0.01
            checks = {
                "seed": seed, "q_exact": q_equal, "pair_exact": pair_equal,
                "fused_allclose": bool(torch.allclose(fused, reference, atol=atol, rtol=rtol)),
                "split_allclose": bool(torch.allclose(split, reference, atol=atol, rtol=rtol)),
                "max_abs_error": float((fused.float() - reference.float()).abs().max()),
                "relative_l2_error": float(torch.linalg.vector_norm(fused.float() - reference.float())
                                           / torch.linalg.vector_norm(reference.float())),
                "atol": atol, "rtol": rtol,
            }
            record["correctness"].append(checks)
            if not all(checks[key] for key in ("q_exact", "pair_exact", "fused_allclose", "split_allclose")):
                raise RuntimeError(f"Falha de correção {record}")

        record["grid"] = list(case.grid)
        if m == 256 and n == 256:
            for label, compiled in case.compiled.items():
                artifacts[label] = {stage: compiled.asm[stage] for stage in ("ttir", "ttgir", "llir", "ptx")}
            source = artifacts["fused"]
            record["ir_evidence"] = {
                "descriptor_loads_ttir": source["ttir"].count("tt.descriptor_load"),
                "tma_copy_ttgir": "ttng.async_tma_copy_global_to_local" in source["ttgir"],
                "local_load_ttgir": "ttg.local_load" in source["ttgir"],
                "warp_group_dot_ttgir": "ttng.warp_group_dot" in source["ttgir"],
                "prologue_ttir": all(op in source["ttir"] for op in ("arith.mulf", "arith.subf", "arith.addf")),
                "stores_ttir": source["ttir"].count("tt.store"),
                "tma_ptx": "cp.async.bulk.tensor" in source["ptx"],
                "wgmma_ptx": "wgmma.mma_async" in source["ptx"],
            }
            evidence = record["ir_evidence"]
            if not all(evidence[key] for key in ("tma_copy_ttgir", "warp_group_dot_ttgir", "prologue_ttir", "tma_ptx", "wgmma_ptx")):
                raise RuntimeError(f"Pipeline esperado não confirmado: {evidence}")
            record["profiler"] = {}
            for name, runner, count in (("fused", case.fused, 1), ("split", case.split, 2)):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(10):
                        runner()
                    torch.cuda.synchronize()
                events = [ev for ev in prof.events()
                          if ev.device_type == torch.autograd.DeviceType.CUDA
                          and ("persistent_tma_prologue" in ev.name or "materialize_prologue" in ev.name)]
                record["profiler"][name] = {"events": len(events), "kernels_per_call": len(events) / 10,
                                           "kernel_names": sorted({ev.name for ev in events})}
                if len(events) != count * 10:
                    raise RuntimeError(f"Número de kernels inesperado: {record['profiler']}")

        for _ in range(20):
            case.fused()
            case.split()
        torch.cuda.synchronize()
        samples = {"fused": [], "split": []}
        rng = random.Random(0)
        for _ in range(30):
            order = ["fused", "split"]
            rng.shuffle(order)
            for name in order:
                runner = getattr(case, name)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20):
                    runner()
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end) * 1000 / 20)
        record["event_samples_us"] = samples
        record["event_medians_us"] = {name: statistics.median(values) for name, values in samples.items()}
        record["event_split_over_fused"] = record["event_medians_us"]["split"] / record["event_medians_us"]["fused"]
        # CUDA Graphs reduzem lacunas de despacho do Python. Mesmos buffers pré-alocados.
        graph_samples = {name: [] for name in ("fused", "split")}
        for _ in range(3):
            names = list(graph_samples)
            rng.shuffle(names)
            for name in names:
                graph_samples[name].append(triton.testing.do_bench_cudagraph(getattr(case, name), rep=50) * 1000)
        record["graph_samples_us"] = graph_samples
        record["graph_medians_us"] = {name: statistics.median(values) for name, values in graph_samples.items()}
        record["graph_split_over_fused"] = record["graph_medians_us"]["split"] / record["graph_medians_us"]["fused"]
        result["cases"].append(record)
        print(json.dumps({"shape": [m, n, k], "correctness": True,
                          "graph_medians_us": record["graph_medians_us"],
                          "graph_split_over_fused": record["graph_split_over_fused"]}))
    result["elapsed_seconds"] = time.monotonic() - start_time
    return {"result": result, "artifacts": artifacts}


@app.local_entrypoint()
def main():
    data = experiment.remote()
    output = Path(__file__).resolve().parents[2] / "artifacts/tma_prototype"
    output.mkdir(exist_ok=True)
    (output / "results.json").write_text(json.dumps(data["result"], indent=2))
    for label, stages in data["artifacts"].items():
        for stage, source in stages.items():
            (output / f"{label}.{stage}").write_text(source)
    print(json.dumps(data["result"], indent=2))
