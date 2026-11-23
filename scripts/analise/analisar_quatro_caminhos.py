"""Valida coleta completa e produz CSV, figura e tabela LaTeX rastreáveis."""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics as stats
from pathlib import Path

PATHS = ("std_split", "std_fused", "tma_split", "tma_fused")


def load_sessions(root):
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["smoke"]:
        raise ValueError("Smoke não pode ser apresentado como coleta completa")
    sessions = sorted(root.glob("session_*/results.json"))
    if len(sessions) != protocol["sessions"] or len(sessions) < 3:
        raise ValueError("A síntese exige as três sessões completas previstas")
    for filename, sha in protocol["sources_sha256"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == sha
    data = []
    for file in sessions:
        d = json.loads(file.read_text())
        assert not d["smoke"] and len(d["cases"]) == 14
        assert d["environment"]["gpu"].startswith("NVIDIA H100")
        names = set()
        for c in d["cases"]:
            assert c["name"] not in names
            names.add(c["name"])
            search = {r["config_id"]: r for r in c["search"]}
            assert set(search) == set(range(4))
            for ci, holdout in c["holdout"].items():
                for path, values in holdout["samples_us"].items():
                    assert len(values) == 9 and all(math.isfinite(v) and v > 0 for v in values)
                    record = search[int(ci)]["paths"][path]
                    assert record["valid"] and record["correctness"]["allclose"]
                    assert len(record["discovery_us"]) == 3
                    assert holdout["correctness"][path]["allclose"]
                    for seed in ("1", "2"):
                        assert search[int(ci)]["additional_correctness"][seed][path]["allclose"]
                    prof = holdout["profiler"][path]
                    assert prof["calls"] == 5
                    assert len(prof["events"]) == (5 if path.endswith("fused") else 10)
                    prefix = file.parent / "ir" / f"{c['name']}_c{ci}_{path}"
                    for suffix in (".ttir", ".ttgir", ".ptx"):
                        assert prefix.with_suffix(suffix).is_file()
                    tma = "cp.async.bulk.tensor" in prefix.with_suffix(".ptx").read_text()
                    assert tma == record["tma"] == path.startswith("tma")
            for p in PATHS:
                assert p in c["holdout"]["0"]["samples_us"]
                ci = c["selected"][p]
                assert p in c["holdout"][str(ci)]["samples_us"]
                valid = [r for r in c["search"] if r["paths"][p]["valid"]]
                expected = min(valid, key=lambda r: stats.median(r["paths"][p]["discovery_us"]))
                assert ci == expected["config_id"]
        if data:
            assert d["method"] == data[0]["method"]
            assert names == {c["name"] for c in data[0]["cases"]}
        data.append(d)
    return data


def wgmma_a_modes(ptx):
    operands = re.findall(r"wgmma\.mma_async[^\s]*\s+\{[^}]+\},\s*(\{[^}]+\}|%rd\d+),", ptx)
    return sorted({"registers" if a.startswith("{") else "shared" for a in operands})


def summarize(data):
    raw = []
    for d in data:
        for c in d["cases"]:
            for mode in ("fixed", "selected"):
                times = {}
                for p in PATHS:
                    ci = 0 if mode == "fixed" else c["selected"][p]
                    times[p] = stats.median(c["holdout"][str(ci)]["samples_us"][p])
                row = dict(session=d["session"], name=c["name"], mode=mode,
                           workload=c["workload"], m=c["shape_mnk"][0],
                           n=c["shape_mnk"][1], k=c["shape_mnk"][2], **times)
                for family in ("std", "tma"):
                    row[f"{family}_reduction_pct"] = 100*(1-times[f"{family}_fused"]/times[f"{family}_split"])
                raw.append(row)
    groups = {}
    for row in raw:
        groups.setdefault((row["name"], row["mode"]), []).append(row)
    summary = []
    for rows in groups.values():
        row = {k: rows[0][k] for k in ("name", "mode", "workload", "m", "n", "k")}
        for metric in (*PATHS, "std_reduction_pct", "tma_reduction_pct"):
            vals = [r[metric] for r in rows]
            row[metric] = stats.median(vals)
            row[f"{metric}_min"] = min(vals)
            row[f"{metric}_max"] = max(vals)
        summary.append(row)
    return raw, summary


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, path, session_count=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), sharey=True)
    colors = {"std": "#2864a0", "tma": "#c35a17"}
    for ri, workload in enumerate(("cast", "affine")):
        for ci, mode in enumerate(("fixed", "selected")):
            ax = axes[ri, ci]
            subset = sorted([r for r in rows if r["workload"] == workload and r["mode"] == mode
                             and r["m"] == r["k"] == 1024], key=lambda r: r["n"])
            for fam in ("std", "tma"):
                key = f"{fam}_reduction_pct"
                y = [r[key] for r in subset]
                ax.errorbar([r["n"] for r in subset], y,
                            yerr=[[r[key]-r[key+"_min"] for r in subset],
                                  [r[key+"_max"]-r[key] for r in subset]],
                            fmt="o-", capsize=4, label=fam.upper(), color=colors[fam])
            ax.axhline(0, color="#777", linewidth=.8)
            ax.set_xscale("log", base=2)
            ax.set_xticks([r["n"] for r in subset], [str(r["n"]) for r in subset])
            ax.set_title(f"{'Cast' if workload == 'cast' else 'Fórmula'} — {'configuração fixa' if mode == 'fixed' else 'seleção por caminho'}", fontsize=10)
            ax.set_xlabel("N (M = K = 1024)")
            if ci == 0:
                ax.set_ylabel("Redução de latência pela fusão (%)")
            ax.grid(alpha=.2)
            ax.legend()
    population = "mediana e faixa entre três sessões" if session_count == 3 else f"diagnóstico de {session_count} sessão(ões)"
    fig.suptitle(f"Fusão na H100: {population}\nValores negativos indicam regressão; replay CUDA Graphs", fontsize=11)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def latex(rows):
    lines = [r"\begin{tabularx}{\linewidth}{Xrrrr}", r"\toprule",
             r"Prólogo e M/N/K & STD sep. & STD fund. & TMA sep. & TMA fund. \\", r"\midrule"]
    for r in rows:
        if r["mode"] != "selected":
            continue
        title = ("Cast" if r["workload"] == "cast" else "Fórmula") + f" {r['m']}/{r['n']}/{r['k']}"
        lines.append(title + " & " + " & ".join(f"{r[p]:.3f}".replace(".", ",") for p in PATHS) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabularx}", ""])


def latex_reductions(rows, mode):
    if mode not in ("fixed", "selected"):
        raise ValueError("Modo deve ser fixed ou selected")
    lines = [r"\begin{tabularx}{\linewidth}{Xrr}", r"\toprule",
             r"Prólogo e M/N/K & STD & TMA \\", r"\midrule"]
    for row in rows:
        if row["mode"] != mode:
            continue
        title = ("Cast" if row["workload"] == "cast" else "Fórmula") + f" {row['m']}/{row['n']}/{row['k']}"
        cells = []
        for family in ("std", "tma"):
            key = f"{family}_reduction_pct"
            median, low, high = (f"{row[field]:.2f}".replace(".", "{,}")
                                 for field in (key, key + "_min", key + "_max"))
            cells.append(rf"${median}\;[{low};\,{high}]$")
        lines.append(title + " & " + " & ".join(cells) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabularx}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--latex-dir", type=Path,
                        help="Gera somente tabelas LaTeX neste diretório, sem alterar a coleta")
    args = parser.parse_args()
    data = load_sessions(args.directory)
    raw, summary = summarize(data)
    if args.latex_dir is not None:
        args.latex_dir.mkdir(parents=True, exist_ok=True)
        tables = {"quatro_caminhos_selecionados.tex": latex(summary),
                  "quatro_caminhos_reducoes_fixas.tex": latex_reductions(summary, "fixed"),
                  "quatro_caminhos_reducoes_selecionadas.tex": latex_reductions(summary, "selected")}
        for filename, content in tables.items():
            (args.latex_dir / filename).write_text(content, encoding="utf-8")
        print(f"Auditoria OK: {len(data)} sessões; tabelas em {args.latex_dir}. Coleta preservada.")
        return
    write_csv(args.directory / "session_medians.csv", raw)
    write_csv(args.directory / "summary.csv", summary)
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2))
    checks = []
    candidates = []
    resources = []
    for session in data:
        for case in session["cases"]:
            for config in case["search"]:
                candidates.extend(config["paths"].values())
                for path, candidate in config["paths"].items():
                    if candidate["valid"]:
                        resources.append({"session": session["session"], "case": case["name"],
                                          "config_id": config["config_id"], "path": path,
                                          **config["config"],
                                          **{k: candidate[k] for k in ("registers", "spills", "shared_bytes", "output_tiles")},
                                          "ctas": candidate["grid"][0],
                                          "discovery_median_us": stats.median(candidate["discovery_us"])})
                checks.extend(p["correctness"] for p in config["paths"].values() if p["valid"])
                for seed in config.get("additional_correctness", {}).values():
                    checks.extend(seed.values())
            for config in case["holdout"].values():
                checks.extend(config["correctness"].values())
    evidence = {"sessions": len(data), "cases_per_session": len(data[0]["cases"]),
                "candidate_paths": len(candidates),
                "valid_candidate_paths": sum(c["valid"] for c in candidates),
                "failed_candidate_paths": sum(not c["valid"] for c in candidates),
                "numerical_checks": len(checks),
                "max_abs_error": max(c["max_abs"] for c in checks),
                "max_relative_l2": max(c["relative_l2"] for c in checks),
                "session_elapsed_seconds": [d["elapsed_seconds"] for d in data],
                "environments": [d["environment"] for d in data]}
    modes = {p: set() for p in PATHS}
    for session_dir in args.directory.glob("session_*"):
        for path in PATHS:
            for ptx in (session_dir / "ir").glob(f"*_{path}.ptx"):
                modes[path].update(wgmma_a_modes(ptx.read_text()))
    evidence["wgmma_a_modes_by_path"] = {p: sorted(v) for p, v in modes.items()}
    (args.directory / "evidence_summary.json").write_text(json.dumps(evidence, indent=2))
    write_csv(args.directory / "candidate_resources.csv", resources)
    plot(summary, args.directory / "fusion_vs_n")
    (args.directory / "selected_table.tex").write_text(latex(summary))
    print(f"Auditoria OK: {len(data)} sessões, {len(summary)//2} casos, dados e figuras em {args.directory}")


if __name__ == "__main__":
    main()
