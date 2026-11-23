"""Busca limitada de configuração, com validação e reavaliação dos finalistas."""

import json
from pathlib import Path

import modal


app = modal.App("tcc-tma-tuning")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "triton==3.7.1", "numpy")
    .add_local_file(str(Path(__file__).resolve().parents[2] / 'src/tma_prologue_prototype.py'), "/root/tma_prologue_prototype.py", copy=True)
)


@app.function(image=image, gpu="H100", timeout=600)
def tune():
    import random
    import statistics
    import time

    import torch
    import triton
    import triton.testing
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources
    from tma_prologue_prototype import TMAPrologueCase

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    start = time.monotonic()
    rng = random.Random(47)
    configurations = [
        dict(bm=bm, bn=bn, bk=bk, stages=stages, warps=4)
        for bm, bn, bk in ((128, 128, 64), (64, 128, 64), (128, 64, 64),
                           (64, 256, 64), (128, 256, 64), (128, 128, 128))
        for stages in (2, 3)
    ]
    baseline = dict(bm=128, bn=128, bk=64, stages=2, warps=4)
    output = {
        "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "triton": triton.__version__, "cuda": torch.version.cuda,
                        "sms": torch.cuda.get_device_properties(0).multi_processor_count,
                        "allow_tf32_reference": False},
        "method": {"configs": configurations, "baseline": baseline,
                   "discovery_repeats": 3, "discovery_rep_ms": 15,
                   "holdout_repeats": 7, "holdout_rep_ms": 40,
                   "holdout_seed": 99, "clock_control": "unmodified",
                   "reference_prologue": "fp32 arithmetic, then fp16; no FMA",
                   "benchmark": "triton.testing.do_bench_cudagraph"},
        "shapes": [],
    }
    artifacts = {}

    def make_inputs(size, seed):
        torch.manual_seed(seed)
        return (torch.randn((size, size), device="cuda", dtype=torch.float16),
                torch.randn((size, size), device="cuda", dtype=torch.float16),
                torch.randn((size, size), device="cuda", dtype=torch.float16))

    def validate(case):
        x, scale, weight = case.x, case.scale, case.weight
        q = (x.float() * scale.float() + (scale.float() - 1)).half()
        reference = (q.float() @ weight.float()).half()
        a, b = case.fused(), case.split()
        torch.cuda.synchronize()
        checks = dict(q_exact=bool(torch.equal(case.q, q)), pair_exact=bool(torch.equal(a, b)),
                      fused_allclose=bool(torch.allclose(a, reference, atol=0.125, rtol=0.01)),
                      split_allclose=bool(torch.allclose(b, reference, atol=0.125, rtol=0.01)),
                      max_pair_error=float((a.float() - b.float()).abs().max()),
                      relative_l2_error=float(torch.linalg.vector_norm(a.float() - reference.float())
                                             / torch.linalg.vector_norm(reference.float())))
        if not all(checks[key] for key in ("q_exact", "fused_allclose", "split_allclose")):
            raise RuntimeError(f"Falha numérica: {checks}, configuração {case.meta}")
        return checks

    def resources(case):
        return {name: {"registers": kernel.n_regs, "spills": kernel.n_spills,
                       "shared_bytes": kernel.metadata.shared}
                for name, kernel in case.compiled.items()}

    def statistics_us(samples):
        ordered = sorted(samples)
        return {"samples_us": samples, "median_us": statistics.median(samples),
                "min_us": ordered[0], "max_us": ordered[-1]}

    for size in (256, 2048):
        inputs = make_inputs(size, 0)
        records = []
        indices = list(range(len(configurations)))
        rng.shuffle(indices)
        for index in indices:
            config = configurations[index]
            entry = {"id": index, "config": config}
            case = TMAPrologueCase(*inputs, **config)
            try:
                entry["correctness"] = validate(case)
            except (OutOfResources, CompilationError) as error:
                entry.update(valid=False, error=f"{type(error).__name__}: {error}")
                records.append(entry)
                print(json.dumps({"size": size, "config": config, "error": entry["error"]}))
                continue
            entry.update(valid=True, resources=resources(case), grid=list(case.grid))
            samples = {"fused": [], "split": []}
            for _ in range(3):
                names = list(samples)
                rng.shuffle(names)
                for name in names:
                    samples[name].append(triton.testing.do_bench_cudagraph(getattr(case, name), rep=15) * 1000)
            entry["discovery"] = {name: statistics_us(values) for name, values in samples.items()}
            records.append(entry)
            print(json.dumps({"size": size, "id": index,
                              "fused_us": entry["discovery"]["fused"]["median_us"],
                              "split_us": entry["discovery"]["split"]["median_us"]}))

        valid = [entry for entry in records if entry["valid"]]
        if not valid:
            raise RuntimeError("Nenhuma configuração válida")
        selected = {name: min(valid, key=lambda row: row["discovery"][name]["median_us"])["id"]
                    for name in ("fused", "split")}
        baseline_id = configurations.index(baseline)
        # Reavalia vencedores + baseline em ordem intercalada, com outra semente.
        finalist_ids = sorted({baseline_id, *selected.values()})
        holdout_inputs = make_inputs(size, 99)
        finalists = {index: TMAPrologueCase(*holdout_inputs, **configurations[index]) for index in finalist_ids}
        correctness = {index: validate(case) for index, case in finalists.items()}
        holdout = {f"{index}_{name}": [] for index in finalist_ids for name in ("fused", "split")}
        for _ in range(7):
            order = list(holdout)
            rng.shuffle(order)
            for key in order:
                index, name = key.split("_")
                holdout[key].append(triton.testing.do_bench_cudagraph(getattr(finalists[int(index)], name), rep=40) * 1000)

        shape = {"size": size, "search": records, "selected_by_discovery": selected,
                 "holdout_correctness": correctness,
                 "holdout": {key: statistics_us(values) for key, values in holdout.items()},
                 "same_config_ratios": {str(index): statistics.median(holdout[f"{index}_split"])
                                        / statistics.median(holdout[f"{index}_fused"])
                                        for index in finalist_ids}}
        shape["selected_pair_ratio"] = (
            statistics.median(holdout[f"{selected['split']}_split"])
            / statistics.median(holdout[f"{selected['fused']}_fused"])
        )
        output["shapes"].append(shape)
        for index, case in finalists.items():
            for name, kernel in case.compiled.items():
                key = f"n{size}_config{index}_{name}"
                artifacts[key] = {stage: kernel.asm[stage] for stage in ("ttir", "ttgir", "ptx")}
        print(json.dumps({"size": size, "selected": selected, "holdout_ratio": shape["selected_pair_ratio"]}))
    output["elapsed_seconds"] = time.monotonic() - start
    return {"result": output, "artifacts": artifacts}


@app.local_entrypoint()
def main():
    data = tune.remote()
    directory = Path(__file__).resolve().parents[2] / "artifacts/tma_tuning"
    directory.mkdir(exist_ok=True)
    (directory / "results.json").write_text(json.dumps(data["result"], indent=2))
    for label, stages in data["artifacts"].items():
        for stage, source in stages.items():
            (directory / f"{label}.{stage}").write_text(source)
    print(json.dumps({"environment": data["result"]["environment"],
                      "elapsed_seconds": data["result"]["elapsed_seconds"]}, indent=2))
