"""Audita evidências da seleção nativa e com TMA disponível."""

import csv
import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parents[2]
    count = 0
    for directory in ("artifacts/free_selection", "artifacts/free_tma"):
        root = base / directory
        data = json.loads((root / "results.json").read_text())
        assert data["native_config"]["persistent_tma_enabled"] is False
        rows = []
        for case in data["cases"]:
            for label, record in case["paths"].items():
                source = (root / f"{case['name']}_{label}_output_code.py").read_text()
                assert not record["extended_template_in_wrapper"]
                assert "TCC_RESTRICTED_TMA_PROLOGUE" not in source
                assert "make_tensor_descriptor" not in source
                assert len(record["correctness"]) == 3
                assert all(check["allclose"] for check in record["correctness"])
                count += 3
                expected = 2 if case["mode"] == "default" else 1
                assert record["profiler"]["gpu_events"] == expected * record["profiler"]["calls"]
                assert len(record["triton_launches"]) == 1
                if expected == 2:
                    assert record["extern_calls"] == ["extern_kernels.mm"]
                else:
                    assert not record["extern_calls"]
                    assert "tl.dot" in source and "tl.load(" in source
                    assert "*i8" in source
                    log = (root / f"{case['name']}_{label}_fusion.log").read_text()
                    assert "fusing op0 with op1" in log
                decisions = record["candidate_decisions"]
                if data.get("allow_tma", False) and label == "patched":
                    assert decisions and all(choice["extended"] for choice in decisions)
                else:
                    assert not decisions
                assert len(case["graph_samples_us"][label]) == data["method"]["rounds"]
                rows.append({"case": case["name"], "path": label,
                             "kernels_per_call": expected,
                             "extended_candidates": len(decisions),
                             "graph_median_us": case["graph_medians_us"][label]})
        with (root / "summary.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Verificados {count} inputs: seleção, wrappers, profiler, correção e candidatos TMA.")


if __name__ == "__main__":
    main()
