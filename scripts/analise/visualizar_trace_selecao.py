"""Audita Chrome traces reais e gera um visualizador sem servidor.

Uso: python scripts/analise/visualizar_trace_selecao.py artifacts/selecao_prologo_ab/RODADA
"""

import argparse
import json
import re
from pathlib import Path


def collect(root):
    traces, checks = [], []
    for results_path in sorted(root.glob("arm*/results.json")):
        result = json.loads(results_path.read_text())
        for case in result["cases"]:
            profile = case.get("runtime_profile")
            if not profile or "trace_file" not in profile:
                if "label" in case and result.get("profile_runtime"):
                    checks.append({"arm": result["arm"], "case": case["name"],
                                   "label": case["label"], "calls": 0,
                                   "count_matches_wrapper": False, "correctness_passed": False,
                                   "error": case.get("error", profile)})
                continue
            path = results_path.parent / profile["trace_file"]
            chrome = json.loads(path.read_text())
            events = chrome["traceEvents"]
            calls = sorted((e for e in events if e.get("cat") == "user_annotation"
                            and re.search(r"/call_\d+$", e.get("name", ""))), key=lambda e: e["ts"])
            kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
            source_path = path.with_name(path.name.replace("_runtime_trace.json", "_output_code.py"))
            source = source_path.read_text()
            triton_names = re.findall(r"^\s+(triton_\w+)\.run\(", source, re.M)
            expected = len(triton_names)
            expected += len(re.findall(r"^\s+extern_kernels\.\w+\(", source, re.M))
            # Correlacionar com a API de lançamento evita atribuir kernels à
            # chamada errada quando CUPTI desalinha os relógios CPU/GPU.
            def cpu_call(event):
                return next((i for i, c in enumerate(calls)
                             if c["ts"] <= event["ts"] < c["ts"] + c["dur"]), None)

            launches = {e.get("args", {}).get("correlation"): e for e in events
                        if e.get("cat") in {"cuda_runtime", "cuda_driver"}
                        and "Launch" in e.get("name", "") and e.get("ph") == "X"}

            def event_call(event):
                if event.get("cat") == "kernel":
                    launch = launches.get(event.get("args", {}).get("correlation"))
                    return cpu_call(launch) if launch else None
                return cpu_call(event)

            per_call = [sum(event_call(k) == i for k in kernels) for i in range(len(calls))]
            assigned = sum(per_call)
            names_match = set(triton_names) <= {k["name"] for k in kernels}
            wrapper_order = re.findall(r"^\s+(triton_\w+)\.run\(|^\s+(extern_kernels\.\w+)\(", source, re.M)
            wrapper_order = [triton or "extern" for triton, _ in wrapper_order]
            order_matches = all(
                [k["name"] if k["name"].startswith("triton_") else "extern"
                 for k in sorted(kernels, key=lambda e: e["ts"]) if event_call(k) == i] == wrapper_order
                for i in range(len(calls)))
            check = {
                "arm": result["arm"], "case": case["name"], "label": case["label"],
                "trace_file": str(path.relative_to(root)), "calls": len(calls),
                "expected_kernels_per_call": expected, "observed_kernels_per_call": per_call,
                "kernel_events": len(kernels), "unassigned_kernel_events": len(kernels) - assigned,
                "count_matches_wrapper": len(calls) == profile["calls"] and expected > 0
                    and all(n == expected for n in per_call) and assigned == len(kernels) and names_match and order_matches,
                "triton_names_match_wrapper": names_match,
                "kernel_order_matches_wrapper": order_matches,
                "recompile_guard": profile.get("recompile_guard"),
                "gpu_before_cpu_launch_events": sum(
                    k["ts"] < launches[k["args"]["correlation"]]["ts"]
                    for k in kernels if k.get("args", {}).get("correlation") in launches),
                "correctness_passed": profile.get("correctness_passed", False)
                    and len(case.get("correctness", [])) == 3 and "error" not in case,
                "extension_in_wrapper": "TCC_RESTRICTED_TMA_PROLOGUE" in source,
                "tma_in_wrapper": "make_tensor_descriptor" in source,
                "kernel_names": sorted({k["name"] for k in kernels}),
                "guard_rejections": sum("ContractError" in c.get("status", "")
                                        for c in case.get("fused_compiles", [])),
            }
            checks.append(check)
            start = calls[0]["ts"] if calls else min(e["ts"] for e in kernels)
            display_events = []
            for e in events:
                category = e.get("cat")
                if e.get("ph") != "X" or category not in {
                    "user_annotation", "kernel", "cuda_runtime", "cuda_driver"
                }:
                    continue
                if category == "user_annotation" and e not in calls:
                    continue
                display_events.append({
                    "name": e["name"], "cat": category,
                    "ts": round(e["ts"] - start, 3), "dur": e["dur"],
                    "stream": e.get("args", {}).get("stream"),
                    "correlation": e.get("args", {}).get("correlation"),
                    "call": event_call(e),
                })
            traces.append({"id": f'{result["arm"]}/{case["name"]}/{case["label"]}',
                           "check": check, "events": display_events,
                           "shape": case["shape_mnk"], "environment": result["environment"]})
    if not traces:
        raise ValueError(f"Nenhum runtime trace encontrado em {root}")
    return traces, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    traces, checks = collect(args.root)
    audit = {"source": str(args.root), "traces": len(checks),
             "all_passed": all(c["count_matches_wrapper"] and c["correctness_passed"] for c in checks),
             "checks": checks}
    (args.root / "runtime_trace_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    template = Path(__file__).with_name("trace_selecao_template.html").read_text()
    data = json.dumps(traces, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = template.replace("__TRACE_DATA__", data)
    output = args.root / "trace_viewer.html"
    output.write_text(html)
    print(json.dumps({"viewer": str(output.resolve()), "traces": len(checks),
                      "calls": sum(c["calls"] for c in checks), "all_passed": audit["all_passed"]}))
    if not audit["all_passed"]:
        raise SystemExit("Falha na auditoria: consulte runtime_trace_audit.json")


if __name__ == "__main__":
    main()
