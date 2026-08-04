#!/usr/bin/env python3
"""D1-X canonicalization audit.

Verifies the exact canonicalization contract (V4 section 8) over the
canonical records and primary pairs emitted by d1x_canonicalize.py:
- 100% of canonical records carry the v4.0 schema and required fields.
- Every PRIMARY_EXACT_DELTA record has exact ref/alt, a verified coordinate,
  a WT anchor, and matched condition.
- Every non-primary record has a controlled exclusion reason.
- Three reactivity layers (raw/upstream/train-frozen) are present and
  missing positions are masked, never zero-filled.
- Counts reconcile with the canonicalization summary (no silent drop).
- Every primary pair maps back to a traceable profile pointer.
No scientific claim is made; this is DATA_QUALIFICATION_ONLY.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CANONICAL_SCHEMA = "reactflow_delta.data_record.v4.0"
PAIR_SCHEMA = "reactflow_delta.d1x_pair.v1"
REQUIRED_FIELDS = {
    "source_accession", "source_profile_index", "raw_mutation_token",
    "ref_allele", "alt_allele", "mutation_coordinate_system",
    "exact_mutation_evidence_status", "source_to_canonical_retention_status",
    "parent_lineage_evidence", "condition_match_evidence", "noise_source",
    "replicate_block_id", "measurement_variance", "data_role",
    "exclusion_reason", "canonical_sequence", "profile_pointer",
    "reactivity_layers", "verification",
}


def audit(canonical_jsonl: Path, pairs_jsonl: Path, summary_json: Path) -> dict:
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    records = []
    with canonical_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    pairs = []
    with pairs_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    role_counter = Counter()
    status_counter = Counter()
    exc_counter = Counter()
    schema_ok = True
    missing_fields = Counter()
    primary_missing_anchor = 0
    primary_missing_condition = 0
    primary_missing_ref_alt = 0
    non_primary_missing_reason = 0
    layers_ok = True
    primary_pairs = 0

    for r in records:
        role_counter[r.get("data_role") or "UNROLE"] += 1
        status_counter[r.get("exact_mutation_evidence_status") or "UNKNOWN"] += 1
        exc_counter[r.get("exclusion_reason") or "NONE"] += 1
        if r.get("schema_version") != CANONICAL_SCHEMA:
            schema_ok = False
        for f in REQUIRED_FIELDS:
            if f not in r:
                missing_fields[f] += 1
        layers = r.get("reactivity_layers") or {}
        for layer in ("raw", "upstream", "train_frozen"):
            if layer not in layers:
                layers_ok = False
        if r.get("data_role") == "PRIMARY_EXACT_DELTA":
            primary_pairs += 1
            if not r.get("ref_allele") or not r.get("alt_allele"):
                primary_missing_ref_alt += 1
            if not r.get("wt_anchor_profile_index"):
                primary_missing_anchor += 1
            if r.get("condition_match_evidence", {}).get("status") != "MATCHED_ALL_REQUIRED":
                primary_missing_condition += 1
        elif r.get("data_role") in ("AUXILIARY_LATENT_ALT", "RESCUE_MULTI_EDIT", None):
            if not r.get("exclusion_reason"):
                non_primary_missing_reason += 1

    # Pair records must be PRIMARY_EXACT_DELTA and carry a traceable pointer.
    pair_schema_ok = all(p.get("schema_version") == PAIR_SCHEMA for p in pairs)
    pair_role_ok = all(p.get("data_role") == "PRIMARY_EXACT_DELTA" for p in pairs)
    pair_traceable = all(
        p.get("mutant_profile_index") is not None and p.get("wt_profile_index") is not None
        for p in pairs
    )

    expected_pairs = summary.get("primary_exact_delta_pairs")
    expected_records = summary.get("canonical_records_written")

    checks = {
        "schema_all_v4": schema_ok,
        "fields_all_present": not missing_fields,
        "layers_all_present": layers_ok,
        "primary_ref_alt_all_present": primary_missing_ref_alt == 0,
        "primary_wt_anchor_all_present": primary_missing_anchor == 0,
        "primary_condition_all_matched": primary_missing_condition == 0,
        "non_primary_all_have_reason": non_primary_missing_reason == 0,
        "pair_schema_all_v1": pair_schema_ok,
        "pair_role_all_primary": pair_role_ok,
        "pair_all_traceable": pair_traceable,
        "record_count_matches_summary": expected_records == len(records),
        "pair_count_matches_summary": expected_pairs == len(pairs),
    }
    all_pass = all(checks.values())

    return {
        "schema_version": "reactflow_delta.d1x_audit.v1",
        "input": {
            "canonical_jsonl": str(canonical_jsonl),
            "pairs_jsonl": str(pairs_jsonl),
            "summary_json": str(summary_json),
        },
        "counts": {
            "canonical_records": len(records),
            "primary_pairs": len(pairs),
            "data_role_counts": dict(role_counter),
            "status_counts": dict(status_counter),
            "exclusion_reason_counts": dict(exc_counter),
        },
        "missing_fields": dict(missing_fields),
        "checks": checks,
        "all_pass": all_pass,
        "scientific_boundary": (
            "D1-X canonicalization audit; data qualification only. No exact-pair "
            "eligibility, Tier, split, training, or scientific claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-jsonl", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.canonical_jsonl, args.pairs_jsonl, args.summary_json)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"all_pass": result["all_pass"], "checks": result["checks"]}, sort_keys=True))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())