#!/usr/bin/env python3
"""D0-R: functional anchor candidate audit (124 nt window at offset 31).

Uses the 206 nt full anchor from SL5CV2_NOM_0002 and the 124 nt functional
anchor from COVSL5_NOM_0002 to classify M2SL5 profiles with stricter criteria
than the SEQPOS-based approach:

  1. WT anchor = profile with exact 206 nt full-anchor sequence match AND no
     mutation code in name.
  2. Candidate = exactly one name-encoded mutation AND functional Hamming == 1
     AND the functional edit matches the name-encoded mutation (pos/ref/alt,
     with DNA->RNA T->U conversion).

Expected output: 192 candidates/probe, 384 total (2A3 + DMS).
If different, the actual numbers are reported honestly — expected is NOT forced.

Also processes the 8 COVSL5/SL5CV2 files as additional candidate sources.

All candidates are labeled:
  candidate_only_pending_parent_lineage_and_functional_region_validation
  true_pair = False

Outputs (in --output-dir):
  - d0r_functional_anchor_audit.json: per-profile audit across M2SL5 files.
  - m2sl5_functional_candidate_relations.json: M2SL5 candidate relations.
  - covsl5_sl5cv2_candidate_relations.json: COVSL5/SL5CV2 candidate relations.
  - d0r_functional_anchor_summary.json: summary with counts.

The previous 744-candidate SEQPOS-based result (m2sl5_candidate_relations.json)
is NOT overwritten — it remains as historical evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reactflow.delta.d0r_functional import (
    classify_functional_candidate,
    find_wt_anchor_by_sequence,
    verify_functional_anchor,
)
from reactflow.delta.rdat import RdatParseError, parse_rdat, parse_mutations_from_name


SCHEMA_VERSION = "reactflow-delta-d0r-functional-anchor-audit-v1"
RELATIONS_SCHEMA_VERSION = "reactflow-delta-d0r-functional-candidate-relations-v1"
CANDIDATE_STATUS = "candidate_only_pending_parent_lineage_and_functional_region_validation"

# Expected anchors (from SL5CV2_NOM_0002 and COVSL5_NOM_0002 header SEQUENCE)
FULL_ANCHOR_206 = (
    "GGGAACGACUCGAGUAGAGUCGAAAAUCUACG"
    "GACACGAGUAACUCGUCUAUCUUCUGCAGGCUGCUUACGGUUUCGUCCGUGUUGCAGCCGAUCAUCAGCACAUCUAGGUUUCGUCCGGGUGUGACCGAAAGGUAAGAUGGAGAGCCUUGUCCC"
    "AAAAAUAAAACACUGGCGUUCGCGCCAGUGAAAAGAAACAACAACAACAAC"
)
FUNCTIONAL_ANCHOR_124 = (
    "GGACACGAGUAACUCGUCUAUCUUCUGCAGGCUGCUUACGGUUUCGUCCGUGUUGCAGCCGAUCAUCAGCACAUCUAGGUUUCGUCCGGGUGUGACCGAAAGGUAAGAUGGAGAGCCUUGUCCC"
)
EXPECTED_OFFSET = 31


def extract_header_sequence(rdat_path: Path) -> str | None:
    """Extract the SEQUENCE header from an RDAT file (first non-comment SEQUENCE line)."""
    for line in rdat_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("SEQUENCE\t"):
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
    return None


def audit_m2sl5_file(
    rdat_path: Path,
    full_anchor: str,
    functional_offset: int,
    functional_length: int,
) -> dict[str, Any]:
    """Parse one M2SL5 RDAT file and classify all profiles using functional anchor."""

    document = parse_rdat(rdat_path)
    profiles = document["profiles"]
    wt_anchor = find_wt_anchor_by_sequence(profiles, full_anchor)

    profile_audits: list[dict[str, Any]] = []
    if wt_anchor is not None:
        for profile in profiles:
            if profile["index"] == wt_anchor["index"]:
                profile_audits.append({
                    "profile_index": profile["index"],
                    "profile_name": profile.get("profile_name"),
                    "role": "wt_anchor",
                    "wt_profile_index": wt_anchor["index"],
                    "classification": None,
                    "lineage_status": "wt_anchor",
                    "profile_sequence_length": len(profile["profile_sequence"]) if profile.get("profile_sequence") else 0,
                })
                continue
            classification = classify_functional_candidate(
                profile, wt_anchor, functional_offset, functional_length
            )
            profile_audits.append({
                "profile_index": profile["index"],
                "profile_name": profile.get("profile_name"),
                "role": "mutant_candidate",
                "wt_profile_index": wt_anchor["index"],
                "classification": classification,
                "profile_sequence_length": len(profile["profile_sequence"]) if profile.get("profile_sequence") else 0,
            })
    else:
        for profile in profiles:
            profile_audits.append({
                "profile_index": profile["index"],
                "profile_name": profile.get("profile_name"),
                "role": "no_wt_anchor",
                "wt_profile_index": None,
                "classification": None,
                "lineage_status": "no_wt_anchor",
                "profile_sequence_length": len(profile["profile_sequence"]) if profile.get("profile_sequence") else 0,
            })

    # Count classifications
    class_counts: Counter[str] = Counter()
    exclusion_reasons: Counter[str] = Counter()
    for pa in profile_audits:
        cls = pa.get("classification")
        if cls is None:
            continue
        c = cls.get("classification", "unknown")
        class_counts[c] += 1
        if c == "excluded":
            reason = cls.get("exclusion_reason", "unknown")
            # Normalize numeric suffixes for aggregation
            if reason.startswith("functional_hamming_not_1"):
                reason = "functional_hamming_not_1"
            elif reason.startswith("name_sequence_mismatch"):
                reason = "name_sequence_mismatch"
            exclusion_reasons[reason] += 1

    return {
        "rdat_path": str(rdat_path),
        "rdat_sha256": document["sha256"],
        "rdat_name": document["headers"]["NAME"],
        "profile_count": len(profiles),
        "wt_anchor_found": wt_anchor is not None,
        "wt_anchor_index": wt_anchor["index"] if wt_anchor else None,
        "wt_anchor_name": wt_anchor.get("profile_name") if wt_anchor else None,
        "wt_anchor_sequence_matches_full_anchor": wt_anchor is not None,
        "classification_counts": dict(sorted(class_counts.items())),
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "profile_audits": profile_audits,
    }


def build_candidate_relations(
    audit: dict[str, Any],
    rdat_path: Path,
    modifier: str,
    source_label: str,
) -> list[dict[str, Any]]:
    """Extract candidate relations from a functional-anchor audit."""

    relations: list[dict[str, Any]] = []
    wt_index = audit["wt_anchor_index"]
    if wt_index is None:
        return relations
    for pa in audit["profile_audits"]:
        cls = pa.get("classification")
        if cls is None:
            continue
        if cls.get("classification") == "candidate_single_functional_anchor":
            relations.append({
                "schema_version": RELATIONS_SCHEMA_VERSION,
                "source": source_label,
                "rmdb_id": audit["rdat_name"],
                "rdat_sha256": audit["rdat_sha256"],
                "rdat_path": str(rdat_path),
                "modifier": modifier,
                "wt_profile_index": wt_index,
                "wt_profile_name": audit["wt_anchor_name"],
                "mutant_profile_index": pa["profile_index"],
                "mutant_profile_name": pa["profile_name"],
                "name_encoded_mutations": cls["name_encoded_mutations"],
                "full_hamming": cls["full_hamming"],
                "functional_hamming": cls["functional_hamming"],
                "outside_functional_region_difference_count": cls[
                    "outside_functional_region_difference_count"
                ],
                "declarative_tokens": cls["declarative_tokens"],
                "lineage_status": CANDIDATE_STATUS,
                "true_pair": False,
                "audit_method": "functional_anchor_124nt_offset31",
            })
    return relations


def audit_covsl5_sl5cv2_file(
    rdat_path: Path,
) -> dict[str, Any]:
    """Audit a COVSL5/SL5CV2 RDAT file for internal WT-mutant candidates.

    These files use ``mutation:WT`` and ``mutation:<pos><ref><alt>`` annotations
    (e.g. ``G159C``) rather than name-encoded mutations. We identify the WT by
    the ``mutation:WT`` annotation and classify profiles by their annotation
    mutation vs the computed edit set.
    """
    document = parse_rdat(rdat_path)
    profiles = document["profiles"]

    # Find WT by mutation:WT annotation
    wt_profile = None
    for p in profiles:
        ann = p.get("annotation") or {}
        mut_vals = ann.get("mutation", [])
        if any(v.strip().upper() == "WT" for v in mut_vals):
            wt_profile = p
            break

    profile_audits: list[dict[str, Any]] = []
    if wt_profile is None:
        for p in profiles:
            profile_audits.append({
                "profile_index": p["index"],
                "role": "no_wt_anchor",
                "lineage_status": "no_wt_anchor",
            })
        return {
            "rdat_path": str(rdat_path),
            "rdat_sha256": document["sha256"],
            "rdat_name": document["headers"]["NAME"],
            "profile_count": len(profiles),
            "wt_anchor_found": False,
            "profile_audits": profile_audits,
        }

    wt_seq = wt_profile.get("profile_sequence") or document["headers"].get("SEQUENCE", "")
    for p in profiles:
        if p["index"] == wt_profile["index"]:
            profile_audits.append({
                "profile_index": p["index"],
                "role": "wt_anchor",
                "lineage_status": "wt_anchor",
            })
            continue
        # Parse annotation mutation (e.g. G159C -> position 158 0-indexed, G->C)
        ann = p.get("annotation") or {}
        mut_vals = ann.get("mutation", [])
        mut_seq = p.get("profile_sequence") or ""

        # Compute edit set
        edits: list[dict[str, Any]] = []
        if wt_seq and mut_seq and len(wt_seq) == len(mut_seq):
            for i, (m, w) in enumerate(zip(mut_seq, wt_seq)):
                if m != w:
                    edits.append({"position_0indexed": i, "wt_base": w, "mutant_base": m})

        # Parse annotation mutation encoding (e.g. G159C, G1X)
        ann_mutations: list[dict[str, Any]] = []
        for mv in mut_vals:
            mv = mv.strip()
            if mv.upper() == "WT":
                continue
            # Format: <ref><pos><alt> e.g. G159C, or <ref><pos>X e.g. G1X
            if len(mv) >= 3 and mv[0] in "ACGU" and mv[-1] in "ACGUX":
                ref = mv[0]
                alt = "X" if mv[-1] == "X" else mv[-1]
                pos_str = mv[1:-1]
                if pos_str.isdigit():
                    ann_mutations.append({
                        "position_1indexed": int(pos_str),
                        "position_0indexed": int(pos_str) - 1,
                        "ref": ref,
                        "alt": alt,
                        "encoding": mv,
                    })

        is_candidate = (
            len(ann_mutations) == 1
            and len(edits) == 1
            and ann_mutations[0]["position_0indexed"] == edits[0]["position_0indexed"]
            and ann_mutations[0]["ref"] == edits[0]["wt_base"]
            and (ann_mutations[0]["alt"] == "X" or ann_mutations[0]["alt"] == edits[0]["mutant_base"])
        )

        profile_audits.append({
            "profile_index": p["index"],
            "role": "mutant_candidate",
            "annotation_mutations": ann_mutations,
            "computed_edits": edits,
            "edit_count": len(edits),
            "classification": "candidate_single_annotation" if is_candidate else "excluded",
            "lineage_status": CANDIDATE_STATUS if is_candidate else "excluded",
            "true_pair": False,
        })

    class_counts = Counter(pa.get("classification", "unknown") for pa in profile_audits)
    return {
        "rdat_path": str(rdat_path),
        "rdat_sha256": document["sha256"],
        "rdat_name": document["headers"]["NAME"],
        "profile_count": len(profiles),
        "wt_anchor_found": True,
        "wt_anchor_index": wt_profile["index"],
        "classification_counts": dict(sorted(class_counts.items())),
        "profile_audits": profile_audits,
    }


def build_covsl5_sl5cv2_relations(
    audit: dict[str, Any],
    rdat_path: Path,
) -> list[dict[str, Any]]:
    """Extract candidate relations from a COVSL5/SL5CV2 audit."""
    relations: list[dict[str, Any]] = []
    wt_index = audit.get("wt_anchor_index")
    if wt_index is None:
        return relations
    for pa in audit["profile_audits"]:
        if pa.get("classification") != "candidate_single_annotation":
            continue
        ann_muts = pa.get("annotation_mutations", [])
        relations.append({
            "schema_version": RELATIONS_SCHEMA_VERSION,
            "source": "RMDB",
            "rmdb_id": audit["rdat_name"],
            "rdat_sha256": audit["rdat_sha256"],
            "rdat_path": str(rdat_path),
            "modifier": audit["rdat_name"],
            "wt_profile_index": wt_index,
            "mutant_profile_index": pa["profile_index"],
            "annotation_mutations": ann_muts,
            "edit_count": pa.get("edit_count", 0),
            "lineage_status": CANDIDATE_STATUS,
            "true_pair": False,
            "audit_method": "annotation_mutation_match",
        })
    return relations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m2sl5-dir",
        required=True,
        help="Directory with M2SL5 RDAT files",
    )
    parser.add_argument(
        "--paper-dir",
        required=True,
        help="Directory with COVSL5/SL5CV2 RDAT files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for artifacts",
    )
    args = parser.parse_args(argv)

    m2sl5_dir = Path(args.m2sl5_dir)
    paper_dir = Path(args.paper_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Verify anchors from the COVSL5/SL5CV2 files
    sl5cv2_path = paper_dir / "SL5CV2_NOM_0002.rdat"
    covsl5_path = paper_dir / "COVSL5_NOM_0002.rdat"

    sl5cv2_header_seq = extract_header_sequence(sl5cv2_path)
    covsl5_header_seq = extract_header_sequence(covsl5_path)

    anchor_verification: dict[str, Any] = {
        "sl5cv2_full_anchor_source": str(sl5cv2_path),
        "sl5cv2_full_anchor_length": len(sl5cv2_header_seq) if sl5cv2_header_seq else 0,
        "sl5cv2_full_anchor_matches_expected": sl5cv2_header_seq == FULL_ANCHOR_206,
        "covsl5_functional_anchor_source": str(covsl5_path),
        "covsl5_functional_anchor_length": len(covsl5_header_seq) if covsl5_header_seq else 0,
        "covsl5_functional_anchor_matches_expected": covsl5_header_seq == FUNCTIONAL_ANCHOR_124,
    }

    # Use the file-derived anchors (authoritative) but verify they match expected
    full_anchor = sl5cv2_header_seq or FULL_ANCHOR_206
    functional_anchor = covsl5_header_seq or FUNCTIONAL_ANCHOR_124

    func_verify = verify_functional_anchor(full_anchor, functional_anchor, EXPECTED_OFFSET)
    anchor_verification["functional_anchor_verification"] = func_verify

    if not func_verify["valid"]:
        # Fail closed: do NOT infer region
        print("FATAL: functional anchor verification failed: " + func_verify["reason"], file=sys.stderr)
        fail_path = output_dir / "d0r_functional_anchor_FAILED.json"
        with fail_path.open("w", encoding="utf-8") as f:
            json.dump({
                "schema_version": SCHEMA_VERSION,
                "status": "validation_failed",
                "anchor_verification": anchor_verification,
                "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            }, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"Wrote failure artifact: {fail_path}")
        return 1

    functional_offset = func_verify["offset"]
    functional_length = len(functional_anchor)
    print(f"Anchor verified: full={len(full_anchor)}nt, functional={functional_length}nt, offset={functional_offset}")

    # Step 2: Audit M2SL5 files
    m2sl5_files = sorted(m2sl5_dir.glob("M2SL5_*.rdat"))
    m2sl5_audits: list[dict[str, Any]] = []
    m2sl5_relations: list[dict[str, Any]] = []

    for rdat_path in m2sl5_files:
        rmdb_id = rdat_path.stem
        try:
            audit = audit_m2sl5_file(rdat_path, full_anchor, functional_offset, functional_length)
        except (RdatParseError, FileNotFoundError) as exc:
            print(f"PARSE_ERROR {rmdb_id}: {exc}", file=sys.stderr)
            m2sl5_audits.append({"rmdb_id": rmdb_id, "error": str(exc)})
            continue
        m2sl5_audits.append({"rmdb_id": rmdb_id, **audit})

        modifier = "2A3" if "2A3" in rmdb_id else ("DMS" if "DMS" in rmdb_id else "unknown")
        relations = build_candidate_relations(audit, rdat_path, modifier, "RMDB")
        m2sl5_relations.extend(relations)
        print(f"{rmdb_id}: {len(relations)} candidates (modifier={modifier})")

    # Step 3: Audit COVSL5/SL5CV2 files
    paper_files = sorted(paper_dir.glob("*.rdat"))
    paper_audits: list[dict[str, Any]] = []
    paper_relations: list[dict[str, Any]] = []

    for rdat_path in paper_files:
        rmdb_id = rdat_path.stem
        try:
            audit = audit_covsl5_sl5cv2_file(rdat_path)
        except (RdatParseError, FileNotFoundError) as exc:
            print(f"PARSE_ERROR {rmdb_id}: {exc}", file=sys.stderr)
            paper_audits.append({"rmdb_id": rmdb_id, "error": str(exc)})
            continue
        paper_audits.append({"rmdb_id": rmdb_id, **audit})

        relations = build_covsl5_sl5cv2_relations(audit, rdat_path)
        paper_relations.extend(relations)
        print(f"{rmdb_id}: {len(relations)} candidates")

    # Step 4: Write artifacts
    now = datetime.now(timezone.utc).astimezone().isoformat()

    # M2SL5 audit
    m2sl5_audit_path = output_dir / "d0r_functional_anchor_audit.json"
    m2sl5_audit_summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "D0-R",
        "audit_method": "functional_anchor_124nt_offset31",
        "generated_at": now,
        "anchor_verification": anchor_verification,
        "full_anchor_length": len(full_anchor),
        "functional_anchor_length": functional_length,
        "functional_offset": functional_offset,
        "m2sl5_files": [a.get("rmdb_id", "") for a in m2sl5_audits],
        "total_m2sl5_candidates": len(m2sl5_relations),
        "expected_m2sl5_candidates": 384,
        "expected_per_probe": 192,
        "m2sl5_audits": m2sl5_audits,
        "scientific_boundary": (
            "Candidate WT-mutant relations only. Lineage is NOT verified. "
            "The WT anchor is identified by exact full-sequence match AND no "
            "mutation code in name. The functional edit is verified against "
            "the name-encoded mutation (pos/ref/alt with DNA->RNA conversion). "
            "All candidates are candidate_only_pending_parent_lineage_and_"
            "functional_region_validation, true_pair=false. "
            "No pair, tier, or model claim."
        ),
    }
    with m2sl5_audit_path.open("w", encoding="utf-8") as f:
        json.dump(m2sl5_audit_summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote M2SL5 audit: {m2sl5_audit_path}")

    # M2SL5 candidate relations
    m2sl5_rel_path = output_dir / "m2sl5_functional_candidate_relations.json"
    m2sl5_rel_summary = {
        "schema_version": RELATIONS_SCHEMA_VERSION,
        "stage": "D0-R",
        "audit_method": "functional_anchor_124nt_offset31",
        "generated_at": now,
        "source_files": [a.get("rmdb_id", "") for a in m2sl5_audits],
        "total_candidate_relations": len(m2sl5_relations),
        "expected_candidate_relations": 384,
        "relations": m2sl5_relations,
        "lineage_status_all": CANDIDATE_STATUS,
        "true_pair_all": False,
        "scientific_boundary": (
            "Candidate WT-mutant relations only. Lineage is NOT verified. "
            "Functional window = 124 nt at offset 31 in the 206 nt full anchor. "
            "Candidate = exactly 1 name mutation AND functional Hamming == 1 AND "
            "pos/ref/alt match. No pair, tier, or model claim."
        ),
    }
    with m2sl5_rel_path.open("w", encoding="utf-8") as f:
        json.dump(m2sl5_rel_summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote M2SL5 candidate relations: {m2sl5_rel_path} ({len(m2sl5_relations)} relations)")

    # COVSL5/SL5CV2 candidate relations
    paper_rel_path = output_dir / "covsl5_sl5cv2_candidate_relations.json"
    paper_rel_summary = {
        "schema_version": RELATIONS_SCHEMA_VERSION,
        "stage": "D0-R",
        "audit_method": "annotation_mutation_match",
        "generated_at": now,
        "source_files": [a.get("rmdb_id", "") for a in paper_audits],
        "total_candidate_relations": len(paper_relations),
        "relations": paper_relations,
        "lineage_status_all": CANDIDATE_STATUS,
        "true_pair_all": False,
        "scientific_boundary": (
            "Candidate WT-mutant relations from COVSL5/SL5CV2 files. "
            "WT identified by mutation:WT annotation. Candidate = exactly 1 "
            "annotation mutation AND 1 computed edit AND pos/ref/alt match. "
            "No pair, tier, or model claim."
        ),
    }
    with paper_rel_path.open("w", encoding="utf-8") as f:
        json.dump(paper_rel_summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote COVSL5/SL5CV2 relations: {paper_rel_path} ({len(paper_relations)} relations)")

    # Summary
    summary_path = output_dir / "d0r_functional_anchor_summary.json"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "D0-R",
        "generated_at": now,
        "anchor_verification": anchor_verification,
        "m2sl5": {
            "total_candidates": len(m2sl5_relations),
            "expected": 384,
            "expected_per_probe": 192,
            "matches_expected": len(m2sl5_relations) == 384,
            "per_file": {
                a.get("rmdb_id", ""): {
                    "profile_count": a.get("profile_count", 0),
                    "wt_anchor_found": a.get("wt_anchor_found", False),
                    "classification_counts": a.get("classification_counts", {}),
                    "exclusion_reason_counts": a.get("exclusion_reason_counts", {}),
                }
                for a in m2sl5_audits if "error" not in a
            },
        },
        "covsl5_sl5cv2": {
            "total_candidates": len(paper_relations),
            "per_file": {
                a.get("rmdb_id", ""): {
                    "profile_count": a.get("profile_count", 0),
                    "wt_anchor_found": a.get("wt_anchor_found", False),
                    "classification_counts": a.get("classification_counts", {}),
                }
                for a in paper_audits if "error" not in a
            },
        },
        "previous_seqpos_result": {
            "total_candidates": 744,
            "note": "Previous SEQPOS-based result preserved as historical evidence in m2sl5_candidate_relations.json",
            "not_overwritten": True,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote summary: {summary_path}")

    # Print summary
    print(json.dumps({
        "m2sl5_candidates": len(m2sl5_relations),
        "expected_384": len(m2sl5_relations) == 384,
        "covsl5_sl5cv2_candidates": len(paper_relations),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
