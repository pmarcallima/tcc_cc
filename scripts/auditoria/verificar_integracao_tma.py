"""Confere os artefatos coletados e gera a tabela da integração experimental."""

import csv
import json
import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2] / "artifacts/inductor_integration"
    data = json.loads((root / "results.json").read_text())
    rows = []
    for case in data["cases"]:
        m, n, k = case["shape_mnk"]
        for label, record in case["paths"].items():
            expected = 1 if label == "patched_fused" else 2
            assert record["kernels_per_call"] == expected
            assert record["benchmark_context_profiler"]["kernels_per_call"] == expected
            assert len(record["triton_launches"]) == expected
            assert not record["custom_op_present"]
            assert len(record["correctness"]) == 3
            assert all(check["allclose"] for check in record["correctness"])
            prefix = f"m{m}_n{n}_k{k}_{label}"
            source = (root / f"{prefix}_output_code.py").read_text()
            buffers = re.findall(r"\b(buf\d+) = empty_strided_cuda", source)
            assert len(buffers) == expected, buffers
            log = (root / f"{prefix}_fusion.log").read_text()
            message = {"original": "template has no allowed prologue inputs",
                       "patched_split": "prologue fusion turned off",
                       "patched_fused": "fusing op0 with op1"}[label]
            assert message in log
            times = case["samples_us"][label]
            assert len(times) == 7
            rows.append({"M": m, "N": n, "K": k, "path": label,
                         "kernels_per_call": expected, "tensor_buffers": len(buffers),
                         "median_graph_us": case["medians_us"][label],
                         "min_graph_us": min(times), "max_graph_us": max(times),
                         "all_seeds_bitwise_equal_original": all(check["exact_vs_original"] for check in record["correctness"]),
                         "max_abs_vs_original": max(check["max_abs_vs_original"] for check in record["correctness"])})
        split, fused = case["medians_us"]["patched_split"], case["medians_us"]["patched_fused"]
        print(f"M/N/K={m}/{n}/{k}: separado {split:.3f} us, fundido {fused:.3f} us; "
              f"redução {100 * (1-fused/split):.2f}%")
    ir_checks = []
    for path in sorted((root / "ir").rglob("triton_tem_fused_add_mm_mul_sub_0.ttir")):
        ttir = path.read_text()
        ttgir = path.with_suffix(".ttgir").read_text()
        ptx = path.with_suffix(".ptx").read_text()
        assert ttir.count("tt.descriptor_load") == 3
        assert ttir.count("tt.store") == 1
        assert all(op in ttir for op in ("arith.mulf", "arith.subf", "arith.addf", "tt.dot"))
        assert all(op in ttgir for op in ("ttng.async_tma_copy_global_to_local", "ttg.local_load", "ttng.warp_group_dot"))
        assert "cp.async.bulk.tensor" in ptx and "wgmma.mma_async" in ptx
        ir_checks.append({"path": str(path.relative_to(root)), "descriptor_loads": 3,
                          "ttir_output_stores": 1, "tma_wgmma": True})
    assert len(ir_checks) == 2
    with (root / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "verification.json").write_text(json.dumps({"cases_verified": len(data["cases"]),
                                                       "ir": ir_checks}, indent=2))
    print("Verificado: scheduler, código gerado, profiler no contexto do benchmark, correção e IR TMA/WGMMA.")


if __name__ == "__main__":
    main()
