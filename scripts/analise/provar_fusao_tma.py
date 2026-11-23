"""Confere a prova controlada de fusão e gera a comparação dos traces reais.

Uso: python3 scripts/analise/provar_fusao_tma.py artifacts/prova_fusao_tma/RODADA
"""

import argparse
import json
import re
from pathlib import Path


LABELS = {
    "original": "TMA original",
    "patched_split": "Nossa extensão • fusão desligada",
    "patched_enabled": "Nossa extensão • fusão ligada",
}
EXPECTED_CASES = {
    "cast_small": [256, 256, 256],
    "cast_persistent": [2048, 1024, 256],
}


def verify_path(root, case, label, record):
    prefix = f"{case['name']}_{label}"
    source_file = root / f"{prefix}_output_code.py"
    source = source_file.read_text()
    log_file = root / f"{prefix}_fusion.log"
    log = log_file.read_text()
    profile = record["runtime_profile"]
    trace_file = root / profile["trace_file"]
    trace = json.loads(trace_file.read_text())["traceEvents"]
    calls = sorted((e for e in trace if e.get("cat") == "user_annotation"
                    and re.search(r"/call_\d+$", e.get("name", ""))), key=lambda e: e["ts"])
    kernels = [e for e in trace if e.get("cat") == "kernel" and e.get("ph") == "X"]
    launches = {e.get("args", {}).get("correlation"): e for e in trace
                if e.get("cat") in {"cuda_runtime", "cuda_driver"}
                and "Launch" in e.get("name", "") and e.get("ph") == "X"}
    by_call = [[] for _ in calls]
    unassigned = []
    for kernel in sorted(kernels, key=lambda e: e["ts"]):
        launch = launches.get(kernel.get("args", {}).get("correlation"))
        index = next((i for i, c in enumerate(calls) if launch
                      and c["ts"] <= launch["ts"] < c["ts"] + c["dur"]), None)
        if index is None:
            unassigned.append(kernel["name"])
        else:
            by_call[index].append(kernel)
    expected = 1 if label == "patched_enabled" else 2
    generated = re.findall(r"^\s+(triton_\w+)\.run\(", source, re.M)
    buffers = re.findall(r"\b(buf\d+) = empty_strided_cuda", source)
    messages = {"original": "template has no allowed prologue inputs",
                "patched_split": "prologue fusion turned off",
                "patched_enabled": "fusing op0 with op1"}
    checks = {
        "executed": record["status"] == "executed",
        "expected_generated_kernels": len(generated) == expected,
        "expected_trace_kernels": len(calls) == 5 and len(kernels) == 5 * expected
            and all(len(k) == expected for k in by_call) and not unassigned,
        "kernel_names_and_order": all([e["name"] for e in k] == generated for k in by_call),
        "scheduler_decision": messages[label] in log,
        "three_seeds_correct": len(record["correctness"]) == 3
            and all(c["allclose"] for c in record["correctness"]),
        "five_profile_outputs_correct": profile["all_outputs_correct"],
        "no_recompilation": profile["recompile_guard"] == "fail_on_recompile",
        "tensor_buffers": len(buffers) == expected,
        "extension_marker": ("TCC_RESTRICTED_TMA_PROLOGUE" in source) == (label != "original"),
        "no_external_matmul_or_custom_op": "extern_kernels.mm(" not in source and "torch.ops.tcc" not in source,
    }
    ir_paths = [root / path for path in record["ir_files"]]
    # Identidade do kernel une wrapper, trace, TTIR/TTGIR e PTX da mesma variante.
    matmul_name = generated[-1]
    matching = [p for p in ir_paths if p.suffix == ".ptx" and p.stem == matmul_name]
    checks["unique_matmul_ptx"] = len(matching) == 1
    ir_evidence = {}
    if len(matching) == 1:
        ptx_path = matching[0]
        ptx = ptx_path.read_text()
        ttir = ptx_path.with_suffix(".ttir").read_text()
        ttgir = ptx_path.with_suffix(".ttgir").read_text()
        checks["ptx_tma"] = "cp.async.bulk.tensor" in ptx
        checks["ptx_wgmma"] = "wgmma.mma_async" in ptx
        checks["ttgir_tma"] = "ttng.async_tma_copy_global_to_local" in ttgir
        if label == "patched_enabled":
            conversion = re.search(r"(%\w+) = arith.sitofp (%\w+)[^\n]*xi8[^\n]*xf16", ttir)
            checks["cast_feeds_dot"] = bool(conversion and re.search(
                rf"tt.dot {re.escape(conversion[1])},", ttir))
            checks["cast_input_loaded_by_descriptor"] = bool(conversion and re.search(
                rf"{re.escape(conversion[2])} = tt.descriptor_load[^\n]*xi8", ttir))
            checks["two_descriptor_loads_one_output_store"] = (
                ttir.count("tt.descriptor_load") == 2 and ttir.count("tt.store") == 1)
            checks["generated_int8_cast"] = "*i8" in source and ".to(tl.float16)" in source
        ir_evidence = {"ptx": str(ptx_path.relative_to(root)),
                       "ttir": str(ptx_path.with_suffix(".ttir").relative_to(root)),
                       "ttgir": str(ptx_path.with_suffix(".ttgir").relative_to(root))}
    summary = {
        "case": case["name"], "label": label, "passed": all(checks.values()),
        "checks": checks, "kernels_per_call": [len(k) for k in by_call],
        "kernel_names": generated, "tensor_buffers": buffers,
        "max_abs_error": max(c["max_abs_error"] for c in record["correctness"]),
        "all_seeds_exact_vs_original": all(c["exact_vs_original"] for c in record["correctness"]),
        "wrapper": source_file.name, "fusion_log": log_file.name,
        "trace": trace_file.name, "ir": ir_evidence,
    }
    display_calls = []
    for events in by_call:
        start = min(e["ts"] for e in events)
        display_calls.append([{
            "name": e["name"], "start": round(e["ts"] - start, 3), "duration": e["dur"],
            "stream": e["args"].get("stream"), "correlation": e["args"].get("correlation"),
            "role": "Cast + matmul TMA" if expected == 1 else "Cast" if i == 0 else "Matmul TMA",
        } for i, e in enumerate(events)])
    return summary, {"label": label, "title": LABELS[label], "calls": display_calls,
                     "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    result = json.loads((root / "results.json").read_text())
    verification, display = [], []
    for case in result["cases"]:
        variants = []
        for label in LABELS:
            try:
                summary, variant = verify_path(root, case, label, case["paths"][label])
                verification.append(summary)
                variants.append(variant)
            except Exception as error:
                verification.append({"case": case["name"], "label": label, "passed": False,
                                     "error": f"{type(error).__name__}: {error}"})
        display.append({"name": case["name"], "shape": case.get("shape_mnk"), "variants": variants})
    cases = result["cases"]
    coverage = {
        name: sum(c["name"] == name and c.get("shape_mnk") == shape for c in cases)
        for name, shape in EXPECTED_CASES.items()
    }
    complete = len(cases) == len(EXPECTED_CASES) and all(count == 1 for count in coverage.values())
    audit = {"scope": result["scope"], "environment": result["environment"],
             "all_passed": complete and all(v["passed"] for v in verification),
             "expected_shapes_mnk": EXPECTED_CASES, "matching_case_counts": coverage,
             "paths": verification}
    (root / "proof_verification.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_passed"]:
        (root / "proof_viewer.html").unlink(missing_ok=True)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        raise SystemExit("A prova não passou; consulte proof_verification.json")
    payload = json.dumps(display, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = Path(__file__).with_name("proof_tma_template.html").read_text()
    (root / "proof_viewer.html").write_text(template.replace("__PROOF_DATA__", payload))
    print(json.dumps({"all_passed": True, "controls": len(verification),
                      "calls": 30, "numerical_seed_comparisons": 18,
                      "viewer": str((root / "proof_viewer.html").resolve())}))


if __name__ == "__main__":
    main()
