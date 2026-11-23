"""Verifica decisões distintas de suporte, heurística e execução do cast."""

import csv
import json
import re
from pathlib import Path


def main():
    base = Path(__file__).resolve().parents[2]
    rows = []
    for directory in ("artifacts/inductor_plain", "artifacts/inductor_cast"):
        root = base / directory
        data = json.loads((root / "results.json").read_text())
        for case in data["cases"]:
            for label, record in case["paths"].items():
                assert record["status"] == "executed", (case["name"], label, record)
                cast_fused = directory == "artifacts/inductor_cast" and label == "patched_enabled"
                expected = 1 if cast_fused else 2
                assert len(record["triton_launches"]) == expected
                assert record["profiler"]["kernels_per_call"] == expected
                assert len(record["tensor_buffers"]) == expected
                assert all(check["allclose"] for check in record["correctness"])
                log = (root / f"{case['name']}_{label}_fusion.log").read_text()
                if label == "original":
                    assert "template has no allowed prologue inputs" in log
                elif label == "patched_split":
                    assert "prologue fusion turned off" in log
                elif cast_fused:
                    assert "can fuse (benchmark)" in log and "fusing op0 with op1" in log
                    assert all(check["max_abs_error"] == 0 and check["exact_vs_original"]
                               for check in record["correctness"])
                elif case["name"].startswith("relu"):
                    assert "must be upcast to fp32" in log
                else:
                    assert "prologue fusion will not increase amount of bytes read in kernel" in log
                times = case["graph_samples_us"][label]
                assert len(times) == 5
                rows.append({"case": case["name"], "path": label, "kernels": expected,
                             "graph_median_us": case["graph_medians_us"][label],
                             "graph_min_us": min(times), "graph_max_us": max(times),
                             "max_abs_vs_eager": max(check["max_abs_error"] for check in record["correctness"])})

        if directory == "artifacts/inductor_cast":
            ir = list((root / "ir").rglob("triton_tem_fused__to_copy_mm_0.ttir"))
            assert len(ir) == 2
            for path in ir:
                ttir, ttgir, ptx = (path.with_suffix(suffix).read_text() for suffix in (".ttir", ".ttgir", ".ptx"))
                assert ttir.count("tt.descriptor_load") == 2 and ttir.count("tt.store") == 1
                assert "sitofp" in ttir and "tt.dot" in ttir
                assert "ttng.async_tma_copy_global_to_local" in ttgir and "ttng.warp_group_dot" in ttgir
                assert "cp.async.bulk.tensor" in ptx and "wgmma.mma_async" in ptx
            large = (root / "cast_persistent_patched_enabled_output_code.py").read_text()
            assert re.search(r"\.run\(.*?, 132, 1, 1, stream=", large)
            assert "*i8" in large and "tmp0.to(tl.float16)" in large
        with (root / "summary.csv").open("w", newline="") as output:
            selected = [row for row in rows if row["case"] in {case["name"] for case in data["cases"]}]
            writer = csv.DictWriter(output, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    print("Verificados 15 caminhos/45 inputs: suporte do template, heurísticas, profiler, correção e IR do cast fundido.")


if __name__ == "__main__":
    main()
