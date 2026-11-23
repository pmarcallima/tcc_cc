"""Benchmark pré-definido comum/TMA x separado/fundido, na mesma H100.

Cada sessão usa um container novo. Resultados em diretório exclusivo por execução;
falhas de compilação são preservadas, falhas numéricas invalidam a sessão.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
app = modal.App("tcc-four-paths")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "triton==3.7.1", "numpy"))
if modal.is_local():
    image = image.add_local_file(str(ROOT / "src/four_path_benchmark.py"),
                                 "/root/four_path_benchmark.py", copy=True)

# N varia com M/K fixos: separa parcialmente repetição do prólogo e tamanho Q.
SHAPES = [(256, 256, 256), (1024, 1024, 1024), (2048, 2048, 2048),
          (4096, 4096, 4096), (1024, 256, 1024),
          (1024, 512, 1024), (1024, 4096, 1024)]
CONFIGS = [dict(bm=bm, bn=bn, bk=64, stages=st, warps=4)
           for bm, bn, st in [(64, 128, 2), (64, 128, 3),
                              (128, 128, 3), (64, 256, 3)]]


@app.function(image=image, gpu="H100", timeout=1500, single_use_containers=True)
def experiment(session: int, smoke: bool):
    import platform
    import random
    import statistics
    import subprocess
    import time

    import torch
    import triton
    import triton.testing
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources
    from four_path_benchmark import FourPathCase, PATHS

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    start = time.monotonic()
    rng = random.Random(800 + session)
    props = torch.cuda.get_device_properties(0)
    result = {
        "schema": 1, "session": session, "smoke": smoke,
        "environment": {"torch": torch.__version__, "triton": triton.__version__,
                        "cuda": torch.version.cuda, "gpu": props.name,
                        "sms": props.multi_processor_count,
                        "python": platform.python_version(),
                        "nvidia_smi": subprocess.check_output([
                            "nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem,power.limit",
                            "--format=csv"], text=True)},
        "method": {"shapes_mnk": SHAPES[:1] if smoke else SHAPES,
                   "configs": CONFIGS[:1] if smoke else CONFIGS,
                   "paths": list(PATHS), "discovery_rounds": 3,
                   "discovery_rep_ms": 15, "holdout_rounds": 9,
                   "holdout_rep_ms": 30, "correctness_seeds": [0, 1, 2],
                   "selection_seed": 0, "holdout_seed": 99,
                   "atol": 0.125, "rtol": 0.01,
                   "clock_control": "unmodified", "host_allocations": "outside timing",
                   "contract": "affine fp32 no FMA, cast fp16, dot accumulate fp32; or int8 cast fp16",
                   "scope": "manual persistent kernels; not Inductor selection; CUDA Graph replay"},
        "cases": [],
    }
    artifacts = {}

    def inputs(shape, cast, seed):
        m, n, k = shape
        torch.manual_seed(seed)
        x = (torch.randint(-8, 9, (m, k), device="cuda", dtype=torch.int8) if cast
             else torch.randn((m, k), device="cuda", dtype=torch.float16))
        s = torch.randn((m, k), device="cuda", dtype=torch.float16) * .25 + 1
        w = torch.randn((k, n), device="cuda", dtype=torch.float16) * .125
        return x, s, w

    def check(case, paths):
        q = (case.x.half() if case.cast else
             (case.x.float() * case.scale.float() + (case.scale.float() - 1)).half())
        ref = (q.float() @ case.weight.float()).half()
        checks = {}
        for path in paths:
            out = case.run(path)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, ref, atol=.125, rtol=.01)
            if path.endswith("split"):
                assert torch.equal(case.q, q), "Prólogo materializado divergente"
            checks[path] = {"allclose": True, "max_abs": float((out.float()-ref.float()).abs().max()),
                            "relative_l2": float(torch.linalg.vector_norm(out.float()-ref.float()) /
                                                 torch.linalg.vector_norm(ref.float()))}
        return checks

    def evidence(case, path, prefix):
        kernel = case.compiled[path]
        ptx = kernel.asm["ptx"]
        uses_tma = "cp.async.bulk.tensor" in ptx
        assert uses_tma == path.startswith("tma"), (path, uses_tma)
        for stage in ("ttir", "ttgir", "ptx"):
            artifacts[f"ir/{prefix}_{path}.{stage}"] = kernel.asm[stage]
        return {"tma": uses_tma, "wgmma": "wgmma.mma_async" in ptx,
                "registers": kernel.n_regs, "spills": kernel.n_spills,
                "shared_bytes": kernel.metadata.shared,
                "grid": list(case.grid), "output_tiles": case.tiles,
                "prologue_repetitions_per_element": triton.cdiv(case.n, case.meta["BN"]) if path.endswith("fused") else 1}

    def samples(runners, rounds, rep):
        measured = {p: [] for p in runners}
        for _ in range(rounds):
            order = list(runners)
            rng.shuffle(order)
            for p in order:
                measured[p].append(triton.testing.do_bench_cudagraph(runners[p], rep=rep)*1000)
        return measured

    with torch.inference_mode():
        for shape in result["method"]["shapes_mnk"]:
            for workload in ("cast", "affine"):
                cast = workload == "cast"
                name = f"{workload}_{'_'.join(map(str, shape))}"
                case_record = {"name": name, "shape_mnk": shape, "workload": workload,
                               "search": [], "holdout": {}}
                candidates = []
                order = list(enumerate(result["method"]["configs"]))
                rng.shuffle(order)
                for ci, config in order:
                    # Validate each implementation independently, retain failed candidates.
                    obj = FourPathCase(*inputs(shape, cast, 0), cast, config)
                    valid = []
                    row = {"config_id": ci, "config": config, "paths": {}}
                    for path in PATHS:
                        try:
                            correct = check(obj, [path])
                            ev = evidence(obj, path, f"{name}_c{ci}")
                        except (CompilationError, OutOfResources) as err:
                            row["paths"][path] = {"valid": False, "error": str(err)}
                            continue
                        row["paths"][path] = {"valid": True, "correctness": correct[path], **ev}
                        valid.append(path)
                        candidates.append((ci, path))
                    discovery = samples({p: (lambda p=p: obj.run(p)) for p in valid}, 3, 15)
                    for p, values in discovery.items():
                        row["paths"][p]["discovery_us"] = values
                    case_record["search"].append(row)
                by_id = {r["config_id"]: r for r in case_record["search"]}
                selected = {}
                for p in PATHS:
                    choices = [ci for ci, path in candidates if path == p]
                    if not choices:
                        raise RuntimeError(f"Nenhum candidato válido: {name}, {p}")
                    selected[p] = min(choices, key=lambda ci: statistics.median(by_id[ci]["paths"][p]["discovery_us"]))
                case_record["selected"] = selected
                # Fixed config 0 must work for all four paths; failure invalidates factorial comparison.
                assert all(by_id[0]["paths"][p]["valid"] for p in PATHS)
                finalist_ids = sorted({0, *selected.values()})
                for seed in (1, 2):
                    for ci in finalist_ids:
                        obj = FourPathCase(*inputs(shape, cast, seed), cast, CONFIGS[ci])
                        valid = [p for p in PATHS if by_id[ci]["paths"][p]["valid"]]
                        by_id[ci].setdefault("additional_correctness", {})[str(seed)] = check(obj, valid)
                finalists = {ci: FourPathCase(*inputs(shape, cast, 99), cast, CONFIGS[ci]) for ci in finalist_ids}
                runners = {}
                for ci, obj in finalists.items():
                    valid = [p for p in PATHS if by_id[ci]["paths"][p]["valid"]]
                    case_record["holdout"][str(ci)] = {"correctness": check(obj, valid)}
                    for p in valid:
                        runners[f"{ci}/{p}"] = lambda obj=obj, p=p: obj.run(p)
                # Verify actual GPU launch counts in the same runners used for timing.
                for key, runner in runners.items():
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(5):
                            runner()
                        torch.cuda.synchronize()
                    events = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    expected = 5 if key.endswith("fused") else 10
                    assert len(events) == expected, (key, events)
                    ci, p = key.split("/")
                    case_record["holdout"][ci].setdefault("profiler", {})[p] = {"calls": 5, "events": events}
                holdout = samples(runners, 9, 30)
                for key, values in holdout.items():
                    ci, p = key.split("/")
                    case_record["holdout"][ci].setdefault("samples_us", {})[p] = values
                result["cases"].append(case_record)
                print(json.dumps({"completed": name, "selected": selected,
                                  "elapsed_seconds": time.monotonic()-start}), flush=True)
    result["elapsed_seconds"] = time.monotonic()-start
    # JSON avoids pickling TorchVersion (a str subclass requiring local torch).
    return json.dumps({"result": result, "artifacts": artifacts})


@app.local_entrypoint()
def main(smoke: bool = False, sessions: int = 3):
    if not 1 <= sessions <= 3:
        raise ValueError("Use 1 a 3 sessões; máximo previsto pelo protocolo")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "artifacts/four_paths" / tag
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for p in (Path(__file__), ROOT / "src/four_path_benchmark.py"):
        content = p.read_bytes()
        (out / p.name).write_bytes(content)
        sources[p.name] = hashlib.sha256(content).hexdigest()
    (out / "protocol.json").write_text(json.dumps({"smoke": smoke, "sessions": sessions,
                                                   "sources_sha256": sources}, indent=2))
    print(f"OUTPUT_DIRECTORY={out}", flush=True)
    for session in range(1 if smoke else sessions):
        # One input per container; no concurrency on the GPU.
        data = json.loads(experiment.remote(session, smoke))
        root = out / f"session_{session}"
        root.mkdir()
        (root / "results.json").write_text(json.dumps(data["result"], indent=2))
        for name, source in data["artifacts"].items():
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(source)
    print(f"COMPLETE={out}", flush=True)
