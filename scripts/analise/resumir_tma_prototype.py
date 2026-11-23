"""Resumo reproduzível dos dados já coletados, sem executar novas medições."""

import csv
import json
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parents[2] / "artifacts/tma_prototype"
    data = json.loads((directory / "results.json").read_text())
    fields = ["M", "N", "K", "grid", "fused_graph_us", "split_graph_us",
              "split_over_fused", "fused_latency_change_pct", "pair_exact_all_seeds"]
    rows = []
    for case in data["cases"]:
        m, n, k = case["shape_mnk"]
        fused = case["graph_medians_us"]["fused"]
        split = case["graph_medians_us"]["split"]
        rows.append(dict(M=m, N=n, K=k, grid=case["grid"][0],
                         fused_graph_us=fused, split_graph_us=split,
                         split_over_fused=split / fused,
                         fused_latency_change_pct=100 * (fused / split - 1),
                         pair_exact_all_seeds=all(c["pair_exact"] for c in case["correctness"])))
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['M']}x{row['N']}x{row['K']}: "
              f"fundido {row['fused_graph_us']:.3f} us; separado {row['split_graph_us']:.3f} us; "
              f"razão {row['split_over_fused']:.3f}x; "
              f"variação fundido {row['fused_latency_change_pct']:+.2f}%")


if __name__ == "__main__":
    main()
