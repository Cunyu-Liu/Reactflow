#!/usr/bin/env python3
"""PH0-X identifiability/reliability audit.

Verifies the noise manifest, caller manifest, permutation report, and blind
test certificate produced by the PH0-X scripts.  Checks are data-qualification
only; no scientific claim, no test unseal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NOISE_SCHEMA = "reactflow_delta.ph0x_noise_manifest.v2"
CALLER_SCHEMA = "reactflow_delta.ph0x_caller.v1"
PERM_SCHEMA = "reactflow_delta.ph0x_permutation.v1"
CERT_SCHEMA = "reactflow_delta.ph0x_blind_certificate.v1"


def _load(path: Path):
    if path.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def audit(ph0x_dir: Path) -> dict:
    noise = _load(ph0x_dir / "ph0x_noise_manifest.json")
    caller = _load(ph0x_dir / "ph0x_caller.json")
    perm = _load(ph0x_dir / "ph0x_permutation.json")
    cert = _load(ph0x_dir / "ph0x_blind_certificate.json")

    checks = {
        "noise_schema_v2": noise.get("schema_version") == NOISE_SCHEMA,
        "noise_coverage_ge_80pct": noise.get("matched_noise", {}).get("coverage_ge_80pct") is True,
        "noise_coverage_100pct": noise.get("matched_noise", {}).get("coverage") == 1.0,
        "noise_no_fabricated_source": all(
            src in ("per_position", "replicate_block", "study_probe_median", "study_median")
            for src in noise.get("matched_noise", {}).get("per_source", {})
        ),
        "icc_reliable": noise.get("icc_pooled", {}).get("median", 0) >= 0.5,
        "tier_cond7_controls": noise.get("tier_b_condition_7", {}).get("ge_100_observations") is True,
        "caller_schema_v1": caller.get("schema_version") == CALLER_SCHEMA,
        "caller_frozen_train_val_only": caller.get("caller", {}).get("frozen_on") == "train+validation only",
        "training_changers_ge_100": caller.get("tier_b_conditions", {}).get("training_changers_ge_100") is True,
        "val_changers_ge_20": caller.get("tier_b_conditions", {}).get("validation_changers_ge_20") is True,
        "test_changers_ge_20": caller.get("tier_b_conditions", {}).get("test_changers_ge_20") is True,
        "perm_schema_v1": perm.get("schema_version") == PERM_SCHEMA,
        "perm_pass_real_gt_null": perm.get("pass_real_gt_group_aware_null") is True,
        "perm_p_value_le_0_05": perm.get("p_value", 1.0) <= 0.05,
        "perm_no_single_study_driven": perm.get("no_single_study_driven") is True,
        "cert_schema_v1": cert.get("schema_version") == CERT_SCHEMA,
        "cert_aggregate_only": cert.get("disclosure", "").startswith("aggregate-only"),
        "cert_test_sealed": cert.get("test_split_is_sealed") is True,
        "cert_test_changers_ge_20": cert.get("test_changers_ge_20") is True,
    }
    all_pass = all(checks.values())

    # Tier B+ aggregate qualification
    tier_b_plus = (
        noise.get("matched_noise", {}).get("coverage_ge_80pct") is True
        and caller.get("tier_b_conditions", {}).get("training_changers_ge_100") is True
        and caller.get("tier_b_conditions", {}).get("validation_changers_ge_20") is True
        and caller.get("tier_b_conditions", {}).get("test_changers_ge_20") is True
        and perm.get("pass_real_gt_group_aware_null") is True
        and perm.get("no_single_study_driven") is True
        and cert.get("test_split_is_sealed") is True
    )

    return {
        "schema_version": "reactflow_delta.ph0x_audit.v1",
        "input": {"dir": str(ph0x_dir)},
        "checks": checks,
        "all_pass": all_pass,
        "tier_b_plus_qualified": tier_b_plus,
        "tier_a_plus_data_ready": "MAINTENANCE",
        "scientific_boundary": (
            "PH0-X identifiability/reliability audit; data qualification. "
            "TIER_B_PLUS written PASS by this audit; full TIER_A_PLUS requires "
            "B0-X frozen cross-parent/cross-study learnability."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ph0x-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.ph0x_dir)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "all_pass": result["all_pass"],
        "tier_b_plus_qualified": result["tier_b_plus_qualified"],
        "failed_checks": [k for k, v in result["checks"].items() if not v],
    }, sort_keys=True))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())