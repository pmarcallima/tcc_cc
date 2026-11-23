"""Consolida tuning e perfis NCU preservando as amostras e as unidades originais."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_metrics(path):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    rows = list(csv.reader(lines[start:]))
    headers, units, values = rows[:3]
    if not len(headers) == len(units) == len(values):
        raise ValueError(f"Colunas inconsistentes em {path}")
    result = {}
    for name, unit, value in zip(headers, units, values):
        try:
            number = float(value.replace(",", ""))
        except ValueError:
            number = value
        result[name] = {"unit": unit, "value": number}
    return result


def base_unit(data, multipliers):
    return data["value"] * multipliers[data["unit"]]


def main():
    directory = ROOT / "artifacts/tma_tuning"
    data = json.loads((directory / "results.json").read_text())
    rows = []
    for shape in data["shapes"]:
        configs = {entry["id"]: entry for entry in shape["search"]}
        for key, sample in shape["holdout"].items():
            index, mode = key.split("_")
            rows.append({"size": shape["size"], "config_id": index, "mode": mode,
                         **configs[int(index)]["config"],
                         "median_us": sample["median_us"], "min_us": sample["min_us"],
                         "max_us": sample["max_us"]})
    with (directory / "holdout.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for shape in data["shapes"]:
        print(f"Tamanho {shape['size']}: candidatos {len(shape['search'])}, "
              f"válidos {sum(row['valid'] for row in shape['search'])}")
        for mode, index in shape["selected_by_discovery"].items():
            print(f"  {mode} selecionado: {data['method']['configs'][index]}, "
                  f"holdout {shape['holdout'][f'{index}_{mode}']}")
        print(f"  separado/fundido selecionados: {shape['selected_pair_ratio']:.4f}x")

    profile_dir = directory / "ncu"
    metrics = {}
    brief = []
    for path in sorted(profile_dir.rglob("*.csv")):
        raw = read_metrics(path)
        tokens = ("dram__bytes", "dram_throughput", "lts__t_sectors_op", "registers", "shared_mem",
                  "occupancy", "warps_active", "pipe_tensor", "sm__throughput", "gpu__time_duration",
                  "spilling", "waves_per_multiprocessor")
        metrics[path.stem] = {name: entry for name, entry in raw.items() if any(t in name for t in tokens)}
        def get(name):
            return raw[name]["value"]
        brief.append({
            "label": path.stem,
            "duration_us": base_unit(raw["gpu__time_duration.sum"], {"ns": 0.001, "us": 1, "ms": 1000, "s": 1e6}),
            "dram_read_bytes": base_unit(raw["dram__bytes_read.sum"], {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}),
            "dram_write_bytes": base_unit(raw["dram__bytes_write.sum"], {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}),
            "l2_read_sectors": get("lts__t_sectors_op_read.sum"),
            "l2_write_sectors": get("lts__t_sectors_op_write.sum"),
            "registers_per_thread": get("launch__registers_per_thread"),
            "shared_bytes": base_unit(raw["launch__shared_mem_per_block"], {"byte/block": 1, "Kbyte/block": 1e3}),
            "sm_peak_pct": get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            "tensor_pipe_pct": get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
            "warps_active_pct": get("sm__warps_active.avg.pct_of_peak_sustained_active"),
        })
    if metrics:
        (directory / "ncu_selected_metrics.json").write_text(json.dumps(metrics, indent=2))
        (directory / "ncu_summary.json").write_text(json.dumps(brief, indent=2))
        with (directory / "ncu_summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(brief[0]))
            writer.writeheader()
            writer.writerows(brief)
        print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
