#!/usr/bin/env python3
"""D0-R: parse downloaded RDAT files and reconstruct candidate WT-mutant relations.

Uses the fixed RDAT parser (per-profile sequence, name mutation encoding, WT
anchor, edit-set computation) to reconstruct candidate relationships from the
downloaded RDAT files. All relations are ``candidate_only_unverified`` — lineage
confirmation requires D1.

Outputs:
  - ``artifacts/reactflow_delta/d0r/d0r_construct_audit.json``: per-profile audit
    across all downloaded RDAT files.
  - ``artifacts/reactflow_delta/d0r/m2sl5_candidate_relations.json``: M2SL5
    candidate WT-mutant relations (candidate_only).

No pair/tier/model claim is made. Construct-level audit only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reactflow.delta.rdat import (
    RdatParseError,
    classify_profile_edit,
    find_wt_anchor,
    parse_rdat,
    seqpos_to_indices,
)


SCHEMA_VERSION = "reactflow-delta-d0r-construct-audit-v1"
RELATIONS_SCHEMA_VERSION = "reactflow-delta-d0r-candidate-relations-v1"


def audit_rdat_file(rdat_path: Path) -> dict[str, Any]:
    """Parse one RDAT file and classify all profiles against the WT anchor."""

    document = parse_rdat(rdat_path)
    profiles = document["profiles"]
    seqpos_indices = seqpos_to_indices(document["seqpos"])
    wt_anchor = find_wt_anchor(profiles)

    profile_audits: list[dict[str, Any]] = []
    if wt_anchor is not None:
        for profile in profiles:
            if profile["index"] == wt_anchor["index"]:
                profile_audits.append(
                    {
                        "profile_index": profile["index"],
                        "profile_name": profile.get("profile_name"),
                        "role": "wt_anchor",
                        "wt_profile_index": wt_anchor["index"],
                        "classification": None,
                        "profile_sequence_source": profile.get("profile_sequence_source"),
                        "profile_sequence_length": len(profile["profile_sequence"]) if profile.get("profile_sequence") else 0,
                    }
                )
                continue
            classification = classify_profile_edit(profile, wt_anchor, seqpos_indices)
            profile_audits.append(
                {
                    "profile_index": profile["index"],
                    "profile_name": profile.get("profile_name"),
                    "role": "mutant_candidate",
                    "wt_profile_index": wt_anchor["index"],
                    "classification": classification,
                    "profile_sequence_source": profile.get("profile_sequence_source"),
                    "profile_sequence_length": len(profile["profile_sequence"]) if profile.get("profile_sequence") else 0,
                }
            )
    else:
        for profile in profiles:
            profile_audits.append(
                {
                    "profile_index": profile["index"],
                    "profile_name": profile.get("profile_name"),
                    "role": "no_wt_anchor",
                    "wt_profile_index": None,
                    "classification": None,
                    "profile_sequence_source": profile.get("profile_sequence_source"),
                    "profile_sequence_length": len(profile["profile_sequence"]) if profile.get("profile_sequence") else 0,
                }
            )

    edit_classes = Counter(
        pa["classification"]["edit_class"]
        for pa in profile_audits
        if pa.get("classification")
    )
    return {
        "rdat_path": str(rdat_path),
        "rdat_sha256": document["sha256"],
        "rdat_name": document["headers"]["NAME"],
        "header_sequence_masked": set(document["headers"]["SEQUENCE"]).issubset({"X", "."}),
        "seqpos_count": len(document["seqpos"]),
        "seqpos_indices": seqpos_indices,
        "profile_count": len(profiles),
        "wt_anchor_found": wt_anchor is not None,
        "wt_anchor_index": wt_anchor["index"] if wt_anchor else None,
        "wt_anchor_name": wt_anchor.get("profile_name") if wt_anchor else None,
        "profiles_with_sequence": sum(1 for p in profiles if p.get("profile_sequence")),
        "edit_class_counts": dict(sorted(edit_classes.items())),
        "profile_audits": profile_audits,
    }


def build_m2sl5_relations(audit: dict[str, Any], rdat_path: Path, modifier: str) -> list[dict[str, Any]]:
    """Extract candidate single-mutant relations from an M2SL5 audit."""

    relations: list[dict[str, Any]] = []
    wt_index = audit["wt_anchor_index"]
    if wt_index is None:
        return relations
    for pa in audit["profile_audits"]:
        cls = pa.get("classification")
        if cls is None:
            continue
        if cls["edit_class"] in ("candidate_single_from_name",):
            # Only strict candidates: name encodes 1 mutation AND functional_edit_count==1.
            # Cross-parent comparisons (name_sequence_mismatch_likely_cross_parent) are excluded.
            relations.append(
                {
                    "schema_version": RELATIONS_SCHEMA_VERSION,
                    "source": "RMDB",
                    "rmdb_id": audit["rdat_name"],
                    "rdat_sha256": audit["rdat_sha256"],
                    "rdat_path": str(rdat_path),
                    "modifier": modifier,
                    "wt_profile_index": wt_index,
                    "wt_profile_name": audit["wt_anchor_name"],
                    "mutant_profile_index": pa["profile_index"],
                    "mutant_profile_name": pa["profile_name"],
                    "name_encoded_mutations": cls["name_encoded_mutations"],
                    "edit_class": cls["edit_class"],
                    "functional_edit_count": cls["edit_set"]["functional_edit_count"],
                    "flanking_edit_count": cls["edit_set"]["flanking_edit_count"],
                    "total_edit_count": cls["edit_set"]["edit_count"],
                    "lineage_status": "candidate_only_unverified",
                    "pair_status": "candidate_not_confirmed",
                }
            )
    return relations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdat-dir", required=True, help="Directory with downloaded RDAT files")
    parser.add_argument("--manifest", required=True, help="Download manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts")
    args = parser.parse_args(argv)

    rdat_dir = Path(args.rdat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.manifest).open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    files = manifest.get("files", [])
    all_audits: list[dict[str, Any]] = []
    m2sl5_relations: list[dict[str, Any]] = []

    for entry in files:
        rmdb_id = entry["rmdb_id"]
        rdat_path = rdat_dir / entry["filename"]
        if not rdat_path.is_file():
            print(f"SKIP {rmdb_id}: file not found at {rdat_path}", file=sys.stderr)
            continue
        try:
            audit = audit_rdat_file(rdat_path)
        except (RdatParseError, FileNotFoundError) as exc:
            print(f"PARSE_ERROR {rmdb_id}: {exc}", file=sys.stderr)
            all_audits.append({"rmdb_id": rmdb_id, "error": str(exc)})
            continue
        all_audits.append({"rmdb_id": rmdb_id, **audit})

        # Build candidate relations for M2SL5 files
        if rmdb_id.startswith("M2SL5"):
            modifier = "2A3" if "2A3" in rmdb_id else ("DMS" if "DMS" in rmdb_id else "unknown")
            relations = build_m2sl5_relations(audit, rdat_path, modifier)
            m2sl5_relations.extend(relations)
            print(f"{rmdb_id}: {len(relations)} candidate relations")

    # Write construct audit
    audit_path = output_dir / "d0r_construct_audit.json"
    audit_summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "D0-R",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "input_manifest": str(Path(args.manifest).resolve()),
        "rdat_dir": str(rdat_dir.resolve()),
        "total_files": len(all_audits),
        "files_with_errors": sum(1 for a in all_audits if "error" in a),
        "files_with_wt_anchor": sum(1 for a in all_audits if a.get("wt_anchor_found")),
        "total_profiles": sum(a.get("profile_count", 0) for a in all_audits if "error" not in a),
        "total_profiles_with_sequence": sum(a.get("profiles_with_sequence", 0) for a in all_audits if "error" not in a),
        "audits": all_audits,
        "scientific_boundary": (
            "Construct-level audit only. Candidate relations are unverified "
            "(lineage_status=candidate_only_unverified). No pair, tier, or model claim."
        ),
    }
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote construct audit: {audit_path}")

    # Write M2SL5 candidate relations
    relations_path = output_dir / "m2sl5_candidate_relations.json"
    relations_summary = {
        "schema_version": RELATIONS_SCHEMA_VERSION,
        "stage": "D0-R",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source_files": [f for f in [e["rmdb_id"] for e in files] if f.startswith("M2SL5")],
        "total_candidate_relations": len(m2sl5_relations),
        "relations": m2sl5_relations,
        "scientific_boundary": (
            "Candidate WT-mutant relations only. Lineage is NOT verified — "
            "the WT anchor is identified by name convention (no mutation suffix), "
            "not by independent experimental confirmation. The functional edit "
            "is distinguished from barcode/adapter edits using SEQPOS window + "
            "name-encoded mutation. No pair, tier, or model claim."
        ),
    }
    with relations_path.open("w", encoding="utf-8") as handle:
        json.dump(relations_summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote M2SL5 candidate relations: {relations_path} ({len(m2sl5_relations)} relations)")

    # Print summary
    print(json.dumps({
        "total_files": audit_summary["total_files"],
        "files_with_errors": audit_summary["files_with_errors"],
        "files_with_wt_anchor": audit_summary["files_with_wt_anchor"],
        "total_profiles": audit_summary["total_profiles"],
        "total_profiles_with_sequence": audit_summary["total_profiles_with_sequence"],
        "m2sl5_candidate_relations": len(m2sl5_relations),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
