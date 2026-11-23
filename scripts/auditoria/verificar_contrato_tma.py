"""Audita uma coleta de contrato/fallback usando apenas arquivos locais.

Uso: python3 scripts/auditoria/verificar_contrato_tma.py artifacts/contrato_tma/RODADA
Uma execução correta pode ter cobertura incompleta do filtro ou das guardas.
"""

import argparse
import json
import math
import re
from pathlib import Path


TARGETS = {
    "positive": ("filter", "accepted"),
    "tail_m": ("filter", "tail_m"),
    "tail_k": ("filter", "tail_k"),
    "noncontiguous": ("filter", "stride_a"),
    "offset": ("filter", "offset_a"),
    "dtype": ("filter", "dtype"),
    "broadcast": ("prologue", "prologue_load_layout"),
    "prologue_b": ("prologue", "prologue_b"),
}


def scenario_matches(name, metadata):
    """Confere o cenário recebido na GPU, antes das transformações do Inductor."""
    m, n, k = 257 if name == "tail_m" else 256, 256, 264 if name == "tail_k" else 256
    x, w, scale = metadata
    dtype = "torch.bfloat16" if name == "dtype" else "torch.float16"
    if [x["shape"], w["shape"], scale["shape"]] != [[m, k], [k, n], [1, k]]:
        return False
    if scale["dtype"] != dtype:
        return False
    x_dtype = "torch.float16" if name in {"offset", "prologue_b"} else "torch.int8"
    w_dtype = "torch.int8" if name == "prologue_b" else dtype
    if (x["dtype"], w["dtype"]) != (x_dtype, w_dtype):
        return False
    if name == "noncontiguous":
        return x["stride"] == [1, m] and x["offset"] == 0
    return x["stride"] == [k, 1] and x["offset"] == (8 if name == "offset" else 0)


def kernels_match_wrapper(kernels, wrapper_calls):
    """Preserva a ordem conjunta de Triton e trechos de kernels externos.

    Um matmul externo pode executar vários kernels consecutivos. Chamadas
    externas adjacentes não são distinguíveis só pelos nomes: nesse caso,
    a comparação falha em vez de presumir a associação.
    """
    triton_names = {name for name in wrapper_calls if name.startswith("triton_")}
    expected = [name if name in triton_names else "<external>" for name in wrapper_calls]
    observed = []
    for name in kernels:
        if name in triton_names:
            observed.append(name)
        elif not observed or observed[-1] != "<external>":
            observed.append("<external>")
    return bool(expected) and observed == expected


def candidate_protocol_checks(record, label):
    limits = record.get("candidate_limits", {})
    fields = ("attempted", "generated", "skipped")
    recorded = set(limits) == {"common", "tma"} and all(
        all(type(counts.get(key)) is int and counts[key] >= 0 for key in fields)
        for counts in limits.values())
    offered = record.get("offered_choices", [])
    single_mm = len(offered) == 1 and offered[0].get("operation") == "mm"
    names = offered[0].get("names", []) if single_mm else []
    names_recorded = single_mm and isinstance(names, list) and bool(names) and all(
        isinstance(name, str) for name in names)
    names = names if names_recorded else []
    actual = {"common": sum(bool(re.fullmatch(r"triton_mm_\d+", name)) for name in names),
              "tma": sum(bool(re.fullmatch(r"triton_mm_persistent_tma_\d+", name)) for name in names)}
    decisions = record["candidate_decisions"]
    return {
        "candidate_counters_recorded": recorded,
        "single_matmul_choices_recorded": names_recorded,
        "one_attempt_per_template": recorded and all(
            counts["generated"] <= counts["attempted"] <= 1 for counts in limits.values()),
        "actual_choices_match_counters": recorded and names_recorded and all(
            actual[kind] == limits[kind]["generated"] for kind in actual),
        "actual_template_counts_limited": names_recorded and all(count <= 1 for count in actual.values()),
        "tma_config_fixed": recorded and (limits["tma"]["attempted"] == 0 or
            limits["tma"].get("config") == {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64,
                                           "num_stages": 3, "num_warps": 4}),
        "tma_counters_match_filter": recorded and (label == "original" or (
            limits["tma"]["attempted"] == len(decisions)
            and limits["tma"]["generated"] == sum(d["status"] == "generated" for d in decisions))),
    }


def audit_path(root, name, label, record):
    source = (root / name / f"{label}_output_code.py").read_text()
    events = json.loads((root / name / f"{label}_runtime_trace.json").read_text())["traceEvents"]
    calls = sorted((e for e in events if e.get("cat") == "user_annotation"
                    and re.fullmatch(rf"contract/{name}/{label}/call_[0-2]", e.get("name", ""))),
                   key=lambda e: e["ts"])
    launches = {e.get("args", {}).get("correlation"): e for e in events
                if e.get("cat") in {"cuda_runtime", "cuda_driver"}
                and "Launch" in e.get("name", "") and e.get("ph") == "X"}
    by_call = [[] for _ in calls]
    unassigned = []
    for event in sorted((e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"),
                        key=lambda e: e["ts"]):
        launch = launches.get(event.get("args", {}).get("correlation"))
        index = next((i for i, call in enumerate(calls) if launch
                      and call["ts"] <= launch["ts"] < call["ts"] + call["dur"]), None)
        if index is None:
            unassigned.append(event["name"])
        else:
            by_call[index].append(event["name"])
    generated = re.findall(r"^\s+(triton_\w+)\.run\(", source, re.M)
    extern = re.findall(r"^\s+(extern_kernels\.\w+)\(", source, re.M)
    wrapper_calls = [name.removesuffix(".run") for name in re.findall(
        r"^\s+(triton_\w+\.run|extern_kernels\.\w+)\(", source, re.M)]
    extended = "TCC_RESTRICTED_TMA_PROLOGUE" in source
    decisions = record["candidate_decisions"]
    guards = record["prologue_rejections"]
    correct = record["correctness"]
    only_rejected = bool(decisions) and all(not d["extended"] for d in decisions)
    checks = {
        "executed": record["status"] == "executed" and "error" not in record,
        "input_scenario": scenario_matches(name, record["input_metadata"]),
        "three_seeds_correct": [c["seed"] for c in correct] == [0, 1, 2]
            and all(c["allclose"] and math.isfinite(c["max_abs_error"]) for c in correct),
        "profile_outputs_correct": record.get("profile_outputs_correct") is True,
        "no_recompilation": record.get("recompile_guard") == "fail_on_recompile",
        "three_calls_assigned": len(calls) == 3 and len({c["name"] for c in calls}) == 3
            and record.get("profile_calls") == 3 and not unassigned,
        "wrapper_matches_record": generated == record["triton_launches"]
            and extern == record["extern_calls"] and extended == record["extension_in_wrapper"],
        "kernels_match_wrapper": bool(wrapper_calls) and all(
            kernels_match_wrapper(kernels, wrapper_calls) for kernels in by_call),
        "decision_consistency": all(
            d["extended"] == (d["reason"] == "accepted")
            and d["delegated_to"] == ("extended" if d["extended"] else "native")
            and d["status"] in {"generated", "not_generated", "error"}
            for d in decisions),
        "original_unpatched": label != "original" or (not decisions and not guards and not extended),
        "extension_has_accepted_candidate": not extended or any(
            d["extended"] and d["status"] == "generated" for d in decisions),
        "filter_has_native_eligibility": not decisions or any(record["native_tma_eligibility"]),
        "rejected_candidates_not_used_as_extension": not only_rejected or not extended,
        # Nos negativos com prólogo, um kernel estendido único seria incompatível
        # com o domínio declarado. Um matmul estendido SEPARADO continua possível.
        "no_unsupported_fused_extension": name in {"positive", "offset"}
            or not extended or len(generated) + len(extern) > 1,
        "fusion_log_present": (root / name / f"{label}_fusion.log").is_file(),
    }
    target_stage, target_reason = TARGETS[name]
    target_observed = (any(d["reason"] == target_reason and d["status"] == "generated"
                           for d in decisions) if target_stage == "filter" else
                       any(g["reason"] == target_reason for g in guards))
    native_eligibility = record["native_tma_eligibility"]
    protocol_checks = candidate_protocol_checks(record, label)
    return {"case": name, "label": label, "passed": all(checks.values()), "checks": checks,
            "protocol_passed": all(protocol_checks.values()), "protocol_checks": protocol_checks,
            "target": {"stage": target_stage, "reason": target_reason},
            "target_observed": target_observed if label == "patched" else False,
            "filter_called": bool(decisions),
            "filter_reasons": sorted({d["reason"] for d in decisions}),
            "native_delegation_generated": any(d["delegated_to"] == "native"
                                                and d["status"] == "generated" for d in decisions),
            "native_tma_excluded_before_filter": bool(native_eligibility)
                and not any(native_eligibility) and not decisions,
            "prologue_reasons": sorted({g["reason"] for g in guards}),
            "extension_in_wrapper": extended,
            "tma_in_wrapper": "make_tensor_descriptor" in source,
            "kernels_per_call": [len(kernels) for kernels in by_call],
            "kernel_names": sorted({kernel for kernels in by_call for kernel in kernels}),
            "unassigned_kernels": unassigned}


def audit_collection(root):
    collection = json.loads((root / "collection.json").read_text())
    requested = collection["requested_cases"]
    checks = {"collection_completed": collection["status"] == "completed",
              "all_requested_completed": collection["completed_cases"] == requested,
              "known_distinct_cases": bool(requested) and len(requested) == len(set(requested))
                  and set(requested) <= set(TARGETS),
              "experiment_snapshot_present": (root / "experiment_snapshot.py").is_file()}
    paths, snapshots = [], []
    for name in requested:
        try:
            data = json.loads((root / name / "results.json").read_text())
            checks[f"{name}_identity"] = data["name"] == name
            checks[f"{name}_paths"] = set(data["paths"]) == {"original", "patched"}
            snapshots.append((root / name / "patch_snapshot.py").read_bytes())
            for label in ("original", "patched"):
                try:
                    paths.append(audit_path(root, name, label, data["paths"][label]))
                except Exception as error:
                    paths.append({"case": name, "label": label, "passed": False,
                                  "error": f"{type(error).__name__}: {error}"})
        except Exception as error:
            paths.append({"case": name, "passed": False,
                          "error": f"{type(error).__name__}: {error}"})
    checks["same_patch_snapshot"] = bool(snapshots) and all(s == snapshots[0] for s in snapshots)
    coverage = {name: any(p["case"] == name and p.get("label") == "patched"
                          and p["passed"] and p.get("protocol_passed") and p.get("target_observed") for p in paths)
                for name in TARGETS}
    all_passed = all(checks.values()) and len(paths) == 2 * len(requested) and all(p["passed"] for p in paths)
    protocol_conformant = all(checks.values()) and len(paths) == 2 * len(requested) and all(
        p.get("protocol_passed", False) for p in paths)
    return {"scope": "planned diagnostic cases only; no general contract or performance claim",
            "all_executions_passed": all_passed, "collection_checks": checks,
            "protocol_conformant": protocol_conformant,
            "planned_coverage_complete": all_passed and protocol_conformant and all(coverage.values()),
            "coverage": coverage, "paths": paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-coverage", action="store_true",
                        help="Falha também se algum dos oito alvos não foi exercitado")
    args = parser.parse_args()
    audit = audit_collection(args.root)
    output = args.root / "contract_verification.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps({"report": str(output.resolve()),
                      "all_executions_passed": audit["all_executions_passed"],
                      "protocol_conformant": audit["protocol_conformant"],
                      "planned_coverage_complete": audit["planned_coverage_complete"],
                      "uncovered_cases": [name for name, covered in audit["coverage"].items() if not covered]}))
    if (not audit["all_executions_passed"] or not audit["protocol_conformant"]
            or (args.require_coverage and not audit["planned_coverage_complete"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
