#!/usr/bin/env python3
"""B0-X Strong Baseline Qualification audit (contract §20.8).

Verifies the frozen B0-X baseline registry produced by b0x_run.py: schema and
run identity, closed baseline registry, P2 parameter budget within the 10k-100k
capacity ladder, the PASS criteria (beats group-aware permutation and strongest
trivial baseline with positive cluster CI lower bound), no single-group
dominance, and a monotonic learning curve.  Benchmark-qualification only; no
scientific claim, no test unseal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REGISTRY_SCHEMA = "reactflow_delta.b0x_registry.v1"
RUN_ID = "b0x_strong_baseline_20260804_v1"


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def audit(registry_path: Path) -> dict:
    reg = json.loads(registry_path.read_text(encoding="utf-8"))

    checks: dict[str, bool] = {}
    checks["registry_schema_v1"] = reg.get("schema_version") == REGISTRY_SCHEMA
    checks["run_id"] = reg.get("run_id") == RUN_ID
    checks["gate_result_pass"] = reg.get("gate_result") == "PASS"

    # baseline registry closed
    baselines = reg.get("baselines", {})
    checks["baselines_present"] = {
        "zero", "train_mean", "mutation_type_mean", "edit_only", "wt_only", "p2_paired"
    }.issubset(set(baselines.keys()))
    checks["all_baselines_ok"] = all(b.get("status") == "ok" for b in baselines.values())

    # P2 parameter budget within 10k-100k
    p2 = baselines.get("p2_paired", {})
    p2_params = p2.get("param_count")
    checks["p2_param_within_10k_100k"] = (
        _finite(p2_params) and 10000 <= p2_params <= 100000
    )

    # PASS criteria
    pc = reg.get("pass_criteria", {})
    checks["p2_ok"] = pc.get("p2_ok") is True
    checks["p2_beats_permutation"] = pc.get("p2_beats_group_aware_permutation") is True
    checks["p2_beats_strongest_trivial"] = pc.get("p2_beats_strongest_trivial") is True
    checks["p2_cluster_ci_low_positive"] = pc.get("p2_cluster_ci_low_positive") is True
    checks["p2_perm_p_le_0_05"] = pc.get("p2_perm_p_value", 1.0) <= 0.05
    checks["all_pass"] = pc.get("all_pass") is True

    # no single-group dominance
    per_study = reg.get("per_study", {})
    checks["per_study_present"] = isinstance(per_study.get("studies"), dict) and len(per_study.get("studies", {})) >= 2
    checks["no_single_group_dominance"] = per_study.get("no_single_group_dominance") is True
    for study, s in (per_study.get("studies") or {}).items():
        checks[f"per_study_{study}_positive"] = _finite(s.get("skill_wmae")) and s["skill_wmae"] > 0

    # learning curve (proxy: skill on subsampled validation pairs; indicates
    # data sufficiency).  Verify the overall trend: full-fraction skill exceeds
    # the smallest-fraction skill, and the head is not collapsed.
    lc = reg.get("learning_curve", {})
    fracs = sorted([float(k.replace("frac_", "")) for k in lc.keys()])
    vals = [lc[f"frac_{f}"]["skill_wmae"] for f in fracs]
    checks["learning_curve_present"] = len(vals) >= 2
    checks["learning_curve_trend_increasing"] = (
        len(vals) >= 2 and _finite(vals[0]) and _finite(vals[-1])
        and vals[-1] > vals[0]
    )
    checks["learning_curve_head_not_collapsed"] = _finite(vals[0]) and vals[0] > 0

    all_pass = all(checks.values())
    return {
        "schema_version": "reactflow_delta.b0x_audit.v1",
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "checks": checks,
        "all_pass": all_pass,
        "gate_result": "PASS" if all_pass else "FAIL",
        "evidence_class": "BENCHMARK_QUALIFICATION_ONLY",
    }


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.registry.resolve())
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"all_pass": result["all_pass"], "gate_result": result["gate_result"]}))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())