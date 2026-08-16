#!/usr/bin/env python3
"""D0-X: audit the streamed candidate inventory for per-accession disposition,
parser coverage, field retention, and silent-drop.

Streams the candidate inventory JSONL (one record per parsed file) plus the
aggregate summary, cross-checks against the frozen asset manifest and the raw
dir listing (including .headers sidecars), and emits a deterministic audit
report JSON. Candidate inventory only; no scientific claim.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_NULL_KEY = "null"


def _safe_counter(counter: Counter[str]) -> dict[str, int]:
    """Return a JSON-serializable mapping, mapping None keys to a string.

    Profile records may legitimately carry a null ``provisional_data_role`` or
    ``exact_mutation_evidence_status`` (a missing field).  ``json.dumps(..., sort_keys=True)``
    cannot sort a ``None`` key against string keys, so None is normalized to the
    literal string ``"null"`` before serialization.
    """
    return {(_NULL_KEY if k is None else str(k)): v for k, v in counter.items()}


# D0-X schema keys that must be present on every profile record (presence, not
# value non-null). These mirror the D1-X retention contract skeleton.
D0X_PROFILE_KEYS = [
    "schema_version",
    "source_accession",
    "source_profile_index",
    "source_file_sha256",
    "raw_mutation_token",
    "ref_allele",
    "alt_allele",
    "mutation_coordinate_system",
    "exact_mutation_evidence_status",
    "source_to_canonical_retention_status",
    "parent_lineage_evidence",
    "condition_match_evidence",
    "noise_source",
    "replicate_block_id",
    "measurement_variance",
    "data_role",
    "provisional_data_role",
    "data_role_status",
    "exclusion_reason",
    "resolved_annotations",
    "parsed_mutations",
    "seqpos_count",
    "missing_reactivity_count",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory-jsonl", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument("--asset-manifest", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    # ---- load manifest ----
    assets: list[dict[str, Any]] = []
    with args.asset_manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                assets.append(json.loads(line))
    manifest_names = {a["asset_name"] for a in assets}

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    failed_names = {f["asset_name"] for f in summary.get("parse_failures", [])}
    missing_names = {m["asset_name"] for m in summary.get("missing_files", [])}

    # ---- stream inventory ----
    parsed_names: set[str] = set()
    per_file: dict[str, dict[str, Any]] = {}
    role_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    silent_drop_total = 0
    seqpos_total = 0
    missing_reactivity_total = 0
    profile_total = 0
    key_present: Counter[str] = Counter()
    key_nonnull: Counter[str] = Counter()
    profile_records = 0
    source_group_counter: Counter[str] = Counter()
    with args.inventory_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            name = rec["asset_name"]
            parsed_names.add(name)
            source_group_counter[rec["source_group"]] += 1
            records = rec.get("records", [])
            per_file[name] = {
                "profile_count": len(records),
                "file_sha256": rec.get("file_sha256"),
            }
            profile_total += len(records)
            # accounting / silent-drop
            acct = rec.get("accounting", {})
            eq = acct.get("accounting_equation", {})
            silent_drop_total += int(eq.get("silent_drop_count", 0))
            for r in records:
                profile_records += 1
                role_counter[r.get("provisional_data_role")] += 1
                status_counter[r.get("exact_mutation_evidence_status")] += 1
                seqpos_total += int(r.get("seqpos_count", 0))
                missing_reactivity_total += int(r.get("missing_reactivity_count", 0))
                for k in D0X_PROFILE_KEYS:
                    if k in r:
                        key_present[k] += 1
                        if r[k] is not None:
                            key_nonnull[k] += 1

    # ---- per-accession disposition ----
    disposition: Counter[str] = Counter()
    for a in assets:
        name = a["asset_name"]
        if name in failed_names:
            disposition["PARSE_FAILED"] += 1
        elif name in missing_names:
            disposition["MISSING_FILE"] += 1
        elif name in parsed_names:
            disposition["PARSED"] += 1
        else:
            disposition["NOT_SEARCHED"] += 1

    # ---- raw dir listing / sidecar ----
    raw_files = set(os.listdir(args.raw_dir))
    sidecar = sorted(raw_files - manifest_names)
    sidecar_headers = [f for f in sidecar if f.endswith(".headers")]
    non_headers = [f for f in sidecar if not f.endswith(".headers")]
    raw_main_files = raw_files & manifest_names

    # ---- coverage ----
    frozen = len(assets)
    parsed = len(parsed_names)
    failed = len(failed_names)
    missing = len(missing_names)
    not_searched = frozen - parsed - failed - missing
    coverage = parsed / frozen if frozen else 0.0

    # ---- field retention ----
    field_retention = {
        k: {
            "present": key_present[k],
            "present_rate": round(key_present[k] / profile_records, 6) if profile_records else 0.0,
            "non_null": key_nonnull[k],
            "non_null_rate": round(key_nonnull[k] / profile_records, 6) if profile_records else 0.0,
        }
        for k in D0X_PROFILE_KEYS
    }

    report = {
        "schema_version": "reactflow_delta.d0x_inventory_audit.v1",
        "run_id": f"d0x_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S+0000')}",
        "phase_id": "D0-X",
        "inputs": {
            "inventory_jsonl": str(args.inventory_jsonl),
            "summary_json": str(args.summary_json),
            "asset_manifest": str(args.asset_manifest),
            "raw_dir": str(args.raw_dir),
        },
        "per_accession_disposition": dict(disposition),
        "parser_coverage": {
            "frozen_asset_count": frozen,
            "parsed_file_count": parsed,
            "parse_failed_file_count": failed,
            "missing_file_count": missing,
            "not_searched_count": not_searched,
            "coverage_rate": round(coverage, 6),
        },
        "profile_summary": {
            "total_profile_records": profile_records,
            "total_seqpos_count": seqpos_total,
            "total_missing_reactivity_count": missing_reactivity_total,
            "provisional_data_role_counts": _safe_counter(role_counter),
            "exact_mutation_evidence_status_counts": _safe_counter(status_counter),
            "source_group_counts": dict(source_group_counter),
        },
        "silent_drop_audit": {
            "total_silent_drop_count": silent_drop_total,
            "silent_drop_ok": silent_drop_total == 0,
        },
        "field_retention": field_retention,
        "raw_dir_files": {
            "manifest_main_files": len(raw_main_files),
            "manifest_name_count": len(manifest_names),
            "sidecar_headers_count": len(sidecar_headers),
            "sidecar_non_headers_count": len(non_headers),
            "sidecar_non_headers": non_headers,
        },
        "scientific_boundary": (
            "D0-X candidate inventory audit only; no exact pair count, "
            "eligibility, Tier, split, training, or scientific claim."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "per_accession_disposition": dict(disposition),
        "parser_coverage": report["parser_coverage"],
        "total_profile_records": profile_records,
        "silent_drop_total": silent_drop_total,
        "sidecar_headers_count": len(sidecar_headers),
    }, sort_keys=True))
    return 0 if (not_searched == 0 and silent_drop_total == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())