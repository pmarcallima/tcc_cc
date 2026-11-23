"""NCU dos dois tamanhos, baseline e configurações selecionadas pela busca."""

import json
from pathlib import Path

import modal


app = modal.App("tcc-tma-prologue-ncu")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cuda-nsight-compute-13-0", "cuda-nsight-systems-13-0")
    .pip_install("torch==2.13.0", "triton==3.7.1", "numpy")
    .add_local_file(str(Path(__file__).resolve().parents[2] / 'src/tma_prologue_prototype.py'), "/root/tma_prologue_prototype.py", copy=True)
    .add_local_file(str(Path(__file__).resolve().parents[2] / 'experiments/tma/tma_profile_target.py'), "/root/tma_profile_target.py", copy=True)
)
volume = modal.Volume.from_name("tcc-tier2-out", create_if_missing=True)


@app.function(image=image, gpu="H100", volumes={"/out": volume}, timeout=600)
def profile(jobs):
    import subprocess
    import sys
    import time

    root = Path("/out/tma_prototype_ncu")
    root.mkdir(exist_ok=True)
    start = time.monotonic()
    manifest = {"clock_control": "none", "cache_control": "none", "reports": {}}
    for job in jobs:
        label, size, mode, config = job["label"], job["size"], job["mode"], job["config"]
        kernel = "materialize_prologue" if mode == "pointwise" else "persistent_tma_prologue"
        cmd = ["/usr/local/cuda/bin/ncu", "--target-processes", "all", "--kernel-name", kernel,
               "--nvtx", "--nvtx-include", "PROFILE/", "--launch-count", "1",
               "--clock-control", "none", "--cache-control", "none",
               "--export", str(root / label), "--force-overwrite"]
        for section in ("SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy", "LaunchStats", "ComputeWorkloadAnalysis"):
            cmd += ["--section", section]
        # Totais diretos: não reconstruir bytes por throughput x duração.
        cmd += ["--metrics", "dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum"]
        cmd += [sys.executable, "/root/tma_profile_target.py", "--size", str(size), "--mode", mode]
        for name, value in config.items():
            cmd += [f"--{name}", str(value)]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        log = process.stdout + process.stderr
        (root / f"{label}.log").write_text(log)
        report = root / f"{label}.ncu-rep"
        if process.returncode or not report.exists():
            volume.commit()
            raise RuntimeError(f"{label}: {log[-6000:]}")
        raw = subprocess.run(["/usr/local/cuda/bin/ncu", "--import", str(report), "--page", "raw", "--csv"],
                             capture_output=True, text=True, timeout=60, check=True)
        (root / f"{label}.csv").write_text(raw.stdout)
        manifest["reports"][label] = {**job, "returncode": process.returncode, "output_tail": log[-1500:]}
        volume.commit()
        print(json.dumps({"label": label, "profiled": True}))
    manifest["elapsed_seconds"] = time.monotonic() - start
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return manifest


@app.local_entrypoint()
def main():
    root = Path(__file__).resolve().parents[2]
    tuning = json.loads((root / "artifacts/tma_tuning/results.json").read_text())
    baseline = tuning["method"]["baseline"]
    jobs = []
    for shape in tuning["shapes"]:
        size = shape["size"]
        for mode in ("fused", "matmul", "pointwise"):
            jobs.append(dict(label=f"n{size}_baseline_{mode}", size=size, mode=mode, config=baseline))
        for mode, selected_name in (("fused", "fused"), ("matmul", "split")):
            index = shape["selected_by_discovery"][selected_name]
            config = tuning["method"]["configs"][index]
            if config != baseline:
                jobs.append(dict(label=f"n{size}_selected_{mode}", size=size, mode=mode, config=config))
    result = profile.remote(jobs)
    (root / "artifacts/tma_tuning/profile_manifest.json").write_text(json.dumps(result, indent=2))
    local = root / "artifacts/tma_tuning/ncu/tma_prototype_ncu"
    local.mkdir(parents=True, exist_ok=True)
    for label in result["reports"]:
        for suffix in ("csv", "log", "ncu-rep"):
            with (local / f"{label}.{suffix}").open("wb") as output:
                for chunk in volume.read_file(f"/tma_prototype_ncu/{label}.{suffix}"):
                    output.write(chunk)
    (local / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
