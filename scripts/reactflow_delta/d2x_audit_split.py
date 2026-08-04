#!/usr/bin/env python3
"""D2-X split/exposure audit.

Verifies the split manifest, exposure audit, tier candidate, test seal,
test access ledger, blind viability certificate, and data card produced by
d2x_split.py.  Checks are data-qualification only; no scientific claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLIT_SCHEMA = "reactflow_delta.d2x_split_manifest.v1"
EXPOSURE_SCHEMA = "reactflow_delta.d2x_exposure_audit.v1"
TIER_SCHEMA = "reactflow_delta.d2x_tier_candidate.v1"
SEAL_SCHEMA = "reactflow_delta.d2x_test_seal.v1"
LEDGER_SCHEMA = "reactflow_delta.d2x_test_access_ledger.v1"
CERT_SCHEMA = "reactflow_delta.d2x_blind_viability_certificate.v1"
CARD_SCHEMA = "reactflow_delta.d2x_data_card.v1"


def _load(path: Path):
    if path.suffix == ".yaml":
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def audit(out_dir: Path) -> dict:
    split = _load(out_dir / "d2x_split_manifest.json")
    exposure = _load(out_dir / "d2x_exposure_audit.json")
    tier = _load(out_dir / "d2x_tier_candidate.json")
    seal = _load(out_dir / "d2x_test_seal.yaml")
    ledger = _load(out_dir / "d2x_test_access_ledger.json")
    cert = _load(out_dir / "d2x_blind_viability_certificate.json")
    card = _load(out_dir / "d2x_data_card.json")

    checks = {
        "split_schema_v1": split.get("schema_version") == SPLIT_SCHEMA,
        "split_outcome_blind": split.get("outcome_blind") is True,
        "all_studies_assigned": all(
            v in ("train", "validation", "test") for v in split.get("assignment", {}).values()
        ),
        "nonempty_splits": all(
            split.get("pair_counts", {}).get(k, 0) > 0 for k in ("train", "validation", "test")
        ),
        "pair_counts_reconcile": split.get("pair_counts", {}).get("train", 0)
            + split.get("pair_counts", {}).get("validation", 0)
            + split.get("pair_counts", {}).get("test", 0)
            == tier.get("observed", {}).get("n_primary_pairs"),
        "exposure_schema_v1": exposure.get("schema_version") == EXPOSURE_SCHEMA,
        "overlap_zero": exposure.get("overlap_zero") is True,
        "no_near_dup_leak": exposure.get("near_duplicate", {}).get("leakage_near_dup") is True,
        "no_publication_leak": exposure.get("publication_level", {}).get("leakage_cross_split") is False,
        "distinct_publications_ge_3": exposure.get("publication_level", {}).get("distinct_publications", 0) >= 3,
        "tier_schema_v1": tier.get("schema_version") == TIER_SCHEMA,
        "tier_candidate_present": tier.get("tier_b_plus_data_candidate") is True,
        "independent_publications_ge_3": tier.get("checklist", {}).get("independent_publications_ge_3") is True,
        "test_unconsumed": tier.get("checklist", {}).get("test_is_unconsumed") is True,
        "seal_schema_v1": seal.get("schema_version") == SEAL_SCHEMA,
        "seal_status": seal.get("seal_status") == "SEALED",
        "ledger_schema_v1": ledger.get("schema_version") == LEDGER_SCHEMA,
        "ledger_append_only": ledger.get("append_only") is True,
        "ledger_no_sample_access": not any(
            e.get("sample_level_labels_read") for e in ledger.get("entries", [])
        ),
        "cert_schema_v1": cert.get("schema_version") == CERT_SCHEMA,
        "cert_aggregate_only": cert.get("disclosure", "").startswith("aggregate-only"),
        "cert_viability": cert.get("certificate_status") == "PASS_AGGREGATE_VIABILITY",
        "card_schema_v1": card.get("schema_version") == CARD_SCHEMA,
    }
    all_pass = all(checks.values())
    return {
        "schema_version": "reactflow_delta.d2x_audit.v1",
        "input": {"out_dir": str(out_dir)},
        "checks": checks,
        "all_pass": all_pass,
        "scientific_boundary": (
            "D2-X split/exposure audit; data qualification only. Full Tier B+ "
            "requires PH0-X identifiability; full Tier A+ requires B0-X."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.out_dir)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"all_pass": result["all_pass"], "checks": result["checks"]}, sort_keys=True))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
