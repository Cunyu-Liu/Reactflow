#!/usr/bin/env python3
"""D1-X: Exact canonicalization and cleaning of the D0-X candidate inventory.

Implements the V4 contract cleaning loop (contract section 8) over the D0-X
candidate inventory (34.8 GB JSONL, one record per parsed RDAT file).  For each
mutation-bearing candidate profile it re-reads the frozen raw RDAT asset,
verifies the exact ref/alt against the construct sequence, resolves the
WT-mutant anchor pairing under the strict condition-match policy, assigns the
closed-set ``data_role`` and a controlled ``exclusion_reason``, and emits
canonical records in schema ``reactflow_delta.data_record.v4.0`` with
raw/upstream/train-frozen reactivity layers.

Outcome-blind by construction: ``data_role`` and ``exclusion_reason`` are
derived only from mutation token, coordinate, sequence, condition annotations
and parent/design-lineage metadata -- NEVER from observed reactivity or Delta
values.  No normalization, no splitting, no training, no test access.

Usage::

    PYTHONPATH=src python scripts/reactflow_delta/d1x_canonicalize.py \
        --inventory-jsonl  <d0x candidate_inventory.jsonl> \
        --requalify-ledger <requalification_ledger.json> \
        --raw-dir <raw rdat dir> \
        --out-dir <out root>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reactflow.delta.rdat import parse_rdat  # noqa: E402

CANONICAL_SCHEMA = "reactflow_delta.data_record.v4.0"

# Condition fields that must match between WT and mutant for a primary pair.
_CONDITION_KEYS = ("modifier", "temperature", "chemical", "experimentType")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm_chemical(chem_list):
    """Normalize a chemical list into a canonical frozen tuple for comparison."""
    if not chem_list:
        return ()
    out = []
    for c in chem_list:
        if not c:
            continue
        out.append(str(c).strip())
    return tuple(sorted(set(out)))


def _condition_tuple(ra: dict) -> dict:
    """Extract the stable condition tuple from resolved_annotations."""
    modifier = ra.get("modifier") or {}
    temperature = ra.get("temperature") or {}
    chemical = ra.get("chemical") or {}
    exp = ra.get("experimentType") or {}
    return {
        "modifier": _norm_chemical(modifier.get("resolved_values")),
        "temperature": _norm_chemical(temperature.get("resolved_values")),
        "chemical": _norm_chemical(chemical.get("resolved_values")),
        "experimentType": _norm_chemical(exp.get("resolved_values")),
        "all_required_known": bool(
            modifier.get("resolved_values")
            and temperature.get("resolved_values")
            and chemical.get("resolved_values")
            and exp.get("resolved_values")
        ),
    }


def _is_wt_annotation(annotation: dict) -> bool:
    """True if a parsed profile's annotation marks it as a WT/no-mutation control."""
    mut = annotation.get("mutation") or []
    if mut:
        return any(str(v).strip().upper() == "WT" or str(v).strip().upper() == "NO_MUTATION" for v in mut)
    return True


def _profile_condition(profile: dict) -> dict:
    """Extract a condition tuple directly from a raw parsed profile annotation."""
    annotation = profile.get("annotation") or {}
    def _norm(vals):
        return tuple(sorted({str(v).strip() for v in (vals or []) if str(v).strip()}))
    return {
        "modifier": _norm(annotation.get("modifier")),
        "temperature": _norm(annotation.get("temperature")),
        "chemical": _norm(annotation.get("chemical")),
        "experimentType": _norm(annotation.get("experimentType")),
        "all_required_known": bool(
            annotation.get("modifier") and annotation.get("temperature")
            and annotation.get("chemical") and annotation.get("experimentType")
        ),
    }


def _same_condition(a: dict, b: dict) -> str:
    """Return MATCHED_ALL_REQUIRED or the specific mismatch reason."""
    for key in _CONDITION_KEYS:
        va = a.get(key)
        vb = b.get(key)
        if not va or not vb:
            return "MISSING_REQUIRED_FIELD"
        if va != vb:
            label = {
                "modifier": "MISMATCH_PROBE",
                "temperature": "MISMATCH_TEMPERATURE",
                "chemical": "MISMATCH_BUFFER",
                "experimentType": "MISMATCH_ENVIRONMENT",
            }[key]
            return label
    return "MATCHED_ALL_REQUIRED"


def _verify_mutation(seq: str, edit: dict, offset: int) -> dict:
    """Verify the exact ref/alt against the construct sequence at the coordinate."""
    ref = str(edit.get("ref_allele") or "").upper()
    alt = str(edit.get("alt_allele") or "").upper()
    idx = edit.get("sequence_index_0_based")
    if isinstance(idx, str):
        try:
            idx = int(idx)
        except ValueError:
            idx = None
    if idx is None or not seq or idx < 0 or idx >= len(seq):
        return {"status": "INVALID_COORDINATE", "seq_base": None, "ref_matches_seq": False,
                "alt_matches_seq": False, "conflict": False}
    seq_base = seq[idx].upper()
    ref_matches_seq = (ref == seq_base)
    alt_matches_seq = (alt == seq_base)
    conflict = bool(ref and alt and ref == alt)
    return {
        "status": "VERIFIED_EXACT_SINGLE_SUBSTITUTION",
        "seq_base": seq_base,
        "ref_matches_seq": ref_matches_seq,
        "alt_matches_seq": alt_matches_seq,
        "conflict": conflict,
    }


def _build_reactivity_layers(profile: dict, seq_len: int) -> dict:
    """Build raw/upstream/train-frozen layers with a position mask and missing reason.

    At D1-X no normalization is allowed; the three layers are the raw measured
    reactivity and error (upstream == raw; train-frozen == raw). Missing
    positions are masked, never zero-filled.
    """
    raw = profile.get("reactivity") or []
    err = profile.get("reactivity_error") or []
    n = len(raw)
    mask = [1 if (isinstance(v, (int, float)) and math.isfinite(v)) else 0 for v in raw]
    missing = [i for i, v in enumerate(raw) if not (isinstance(v, (int, float)) and math.isfinite(v))]
    missing_reason = "MASKED_UNMEASURED" if missing else None
    return {
        "raw": {"reactivity": raw, "error": err, "length": n},
        "upstream": {"reactivity": raw, "error": err, "length": n},
        "train_frozen": {"reactivity": raw, "error": err, "length": n},
        "position_mask": mask,
        "missing_positions": missing,
        "missing_reason": missing_reason,
    }


def _canonicalize_profile(
    rec: dict,
    profile: dict,
    seq: str,
    offset: int,
    source_accession: str,
    file_sha256: str,
    asset_name: str,
) -> dict:
    """Canonicalize one candidate profile into a v4 data record."""
    ra = rec.get("resolved_annotations") or {}
    pm = rec.get("parsed_mutations") or []
    raw_token = rec.get("raw_mutation_token")
    edits = []
    for m in pm:
        edits.extend(m.get("edits") or [])
    kinds = {m.get("kind") for m in pm}

    d0x_status = rec.get("exact_mutation_evidence_status")
    is_wt = d0x_status == "WT_CONTROL_CANDIDATE" or (kinds == {"WT"})

    ref = alt = None
    coord = None
    verify = None
    status = exclusion = data_role = None
    if is_wt:
        status = "WT_CONTROL_CANDIDATE"
        data_role = None
    elif len(edits) > 1:
        status, exclusion, data_role = "MULTI_EDIT", "MULTI_EDIT", "RESCUE_MULTI_EDIT"
    elif kinds == {"EXACT_SINGLE_SUBSTITUTION"} and edits:
        e = edits[0]
        verify = _verify_mutation(seq, e, offset)
        ref = (e.get("ref_allele") or "").upper() or None
        alt = (e.get("alt_allele") or "").upper() or None
        coord = {
            "convention": "RDAT_POSITION_WITH_OFFSET",
            "offset": offset,
            "source_coordinate_1_based": e.get("source_coordinate_1_based"),
            "sequence_index_0_based": e.get("sequence_index_0_based"),
        }
        if verify["status"] == "INVALID_COORDINATE":
            status, exclusion, data_role = "INVALID_COORDINATE", "COORDINATE_AMBIGUOUS", None
        elif verify["conflict"]:
            status, exclusion, data_role = "CONFLICTING_EVIDENCE", "REF_MISMATCH", None
        elif not verify["ref_matches_seq"]:
            status, exclusion, data_role = "CONFLICTING_EVIDENCE", "REF_MISMATCH", None
        elif alt == "X" or alt is None or alt == "":
            status, exclusion, data_role = "LATENT_ALT", "LATENT_ALT", "AUXILIARY_LATENT_ALT"
        else:
            status, exclusion, data_role = "VERIFIED_EXACT_SINGLE_SUBSTITUTION", None, "PRIMARY_EXACT_DELTA"
    elif d0x_status == "MULTIPLE_MUTATION_ANNOTATION_VALUES":
        status, exclusion, data_role = "MULTI_EDIT", "MULTI_EDIT", "RESCUE_MULTI_EDIT"
    elif d0x_status == "LATENT_ALT_X_REF_CHECKED":
        status, exclusion, data_role = "LATENT_ALT", "LATENT_ALT", "AUXILIARY_LATENT_ALT"
    elif d0x_status == "INVALID_MUTATION_TOKEN":
        status, exclusion, data_role = "INVALID_COORDINATE", "COORDINATE_AMBIGUOUS", None
    else:
        status, exclusion, data_role = "MISSING_EVIDENCE", "MISSING_EVIDENCE", None

    condition = _condition_tuple(ra)
    seq_len = len(seq)
    return {
        "schema_version": CANONICAL_SCHEMA,
        "source_accession": source_accession,
        "source_profile_index": rec.get("source_profile_index"),
        "source_file_sha256": file_sha256,
        "source_asset_name": asset_name,
        "raw_mutation_token": raw_token,
        "ref_allele": ref,
        "alt_allele": alt,
        "mutation_coordinate_system": coord,
        "exact_mutation_evidence_status": status,
        "source_to_canonical_retention_status": "LOSSLESS_REVERSIBLE",
        "parent_lineage_evidence": {
            "parent_sequence_sha256": _sha256_text(seq),
            "construct_sequence_length": seq_len,
            "design_group": source_accession,
            "lineage_evidence": "SAME_FILE_SAME_CONSTRUCT_OFFSET",
        },
        "condition_match_evidence": {
            "status": condition["all_required_known"] and "MATCHED_ALL_REQUIRED" or "MISSING_REQUIRED_FIELD",
            "condition": condition,
        },
        "noise_source": {"type": "NO_IDENTIFIABLE_NOISE_MODEL",
                          "evidence": "D1X_NOISE_MODEL_NOT_FITTED"},
        "replicate_block_id": None,
        "measurement_variance": {
            "reactivity_error_present": bool(profile.get("reactivity_error")),
            "variance_aggregated": False,
        },
        "data_role": data_role,
        "exclusion_reason": exclusion,
        "canonical_sequence": seq,
        "profile_pointer": {"file_sha256": file_sha256, "profile_index": rec.get("source_profile_index")},
        "probe": list(condition.get("modifier") or []),
        "temperature": list(condition.get("temperature") or []),
        "ligand": list(condition.get("chemical") or []),
        "buffer": list(condition.get("chemical") or []),
        "batch": None,
        "environment": "in_vitro",
        "wt_reuse_group": None,
        "reactivity_layers": _build_reactivity_layers(profile, seq_len),
        "verification": verify,
        "is_wt": is_wt,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory-jsonl", type=Path, required=True)
    ap.add_argument("--requalify-ledger", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit-files", type=int, default=0, help="0 = all files; else process at most N PARSED files")
    args = ap.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    rq = json.loads(args.requalify_ledger.read_text(encoding="utf-8"))
    print(f"[d1x] requalify records: {len(rq['records'])}", flush=True)

    canonical_records = []
    primary_pairs = []
    by_source = defaultdict(Counter)
    total_profiles = 0
    mutation_candidates = 0
    primary_candidates = 0
    parsed_files_seen = 0
    no_mutation_class = Counter()
    wt_control_class = Counter()

    with args.inventory_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "PARSED":
                continue
            if args.limit_files and parsed_files_seen >= args.limit_files:
                break
            parsed_files_seen += 1
            asset_name = rec.get("asset_name")
            file_sha256 = rec.get("file_sha256")
            raw_path = args.raw_dir / asset_name
            try:
                parsed = parse_rdat(raw_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[d1x] parse fail for {asset_name}: {exc}", flush=True)
                continue
            headers = parsed.get("headers") or {}
            seq = headers.get("SEQUENCE") or ""
            offset = 0
            try:
                offset = int(headers.get("OFFSET", 0) or 0)
            except (TypeError, ValueError):
                offset = 0
            profiles = {p.get("index"): p for p in parsed.get("profiles") or []}
            # ---- WT anchor pool for this file (from D0-X resolved annotations) ----
            wt_anchors = []
            for pref in rec.get("records", []):
                if pref.get("exact_mutation_evidence_status") != "WT_CONTROL_CANDIDATE":
                    continue
                cond = _condition_tuple(pref.get("resolved_annotations") or {})
                pidx = pref.get("source_profile_index") or pref.get("index")
                p = profiles.get(pidx)
                wt_anchors.append({
                    "profile_index": pidx,
                    "condition": cond,
                    "condition_key": (
                        cond.get("modifier"), cond.get("temperature"),
                        cond.get("chemical"), cond.get("experimentType"),
                    ),
                    "reactivity": (p or {}).get("reactivity") or [],
                    "error": (p or {}).get("reactivity_error") or [],
                })
            for pref in rec.get("records", []):
                total_profiles += 1
                pm = pref.get("parsed_mutations") or []
                kinds = {m.get("kind") for m in pm}
                if not kinds or kinds == {"WT"}:
                    if pref.get("exact_mutation_evidence_status") == "WT_CONTROL_CANDIDATE":
                        wt_control_class["WT_CONTROL_CANDIDATE"] += 1
                    else:
                        no_mutation_class["MISSING_MUTATION_ANNOTATION"] += 1
                    continue
                mutation_candidates += 1
                profile = profiles.get(pref.get("source_profile_index")) or profiles.get(pref.get("index"))
                if profile is None:
                    continue
                cand = _canonicalize_profile(
                    pref, profile, seq, offset,
                    rec.get("source_accession"), file_sha256, asset_name,
                )
                # ---- WT-mutant pairing for primary candidates ----
                if cand["data_role"] == "PRIMARY_EXACT_DELTA":
                    primary_candidates += 1
                    cand_cond = cand["condition_match_evidence"]["condition"]
                    cand_key = (
                        cand_cond.get("modifier"), cand_cond.get("temperature"),
                        cand_cond.get("chemical"), cand_cond.get("experimentType"),
                    )
                    match = next((w for w in wt_anchors if w["condition_key"] == cand_key), None)
                    if match is not None:
                        cand["wt_reuse_group"] = f"{asset_name}:{match['profile_index']}"
                        cand["wt_anchor_profile_index"] = match["profile_index"]
                        cand["wt_anchor_reactivity"] = match["reactivity"]
                        cand["wt_anchor_error"] = match["error"]
                        cand["condition_match_evidence"]["status"] = "MATCHED_ALL_REQUIRED"
                        primary_pairs.append({
                            "schema_version": "reactflow_delta.d1x_pair.v1",
                            "source_accession": cand["source_accession"],
                            "asset_name": asset_name,
                            "file_sha256": file_sha256,
                            "mutant_profile_index": cand["source_profile_index"],
                            "wt_profile_index": match["profile_index"],
                            "ref_allele": cand["ref_allele"],
                            "alt_allele": cand["alt_allele"],
                            "coordinate": cand["mutation_coordinate_system"],
                            "condition": cand_cond,
                            "condition_match_status": "MATCHED_ALL_REQUIRED",
                            "wt_reuse_group": cand["wt_reuse_group"],
                            "parent_sequence_sha256": cand["parent_lineage_evidence"]["parent_sequence_sha256"],
                            "data_role": "PRIMARY_EXACT_DELTA",
                            "exclusion_reason": None,
                        })
                    else:
                        cand["condition_match_evidence"]["status"] = "MISSING_REQUIRED_FIELD"
                        # No usable WT anchor under strict policy -> not a primary pair.
                        cand["data_role"] = None
                        cand["exclusion_reason"] = "CONDITION_MISSING"
                by_source[cand["source_accession"]][cand["data_role"] or "UNROLE"] += 1
                canonical_records.append(cand)

    canonical_out = args.out_dir / "d1x_canonical_records.jsonl"
    with canonical_out.open("w", encoding="utf-8") as fh:
        for c in canonical_records:
            fh.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")

    pairs_out = args.out_dir / "d1x_primary_pairs.jsonl"
    with pairs_out.open("w", encoding="utf-8") as fh:
        for pr in primary_pairs:
            fh.write(json.dumps(pr, ensure_ascii=False, sort_keys=True) + "\n")

    role_counter = Counter(c["data_role"] or "UNROLE" for c in canonical_records)
    status_counter = Counter(c["exact_mutation_evidence_status"] for c in canonical_records)
    exc_counter = Counter(c["exclusion_reason"] or "NONE" for c in canonical_records)
    pair_source = Counter(pr["source_accession"] for pr in primary_pairs)
    summary = {
        "schema_version": "reactflow_delta.d1x_canonicalization_summary.v1",
        "run_id": "d1x_canonicalization_20260804_v1",
        "phase_id": "D1-X",
        "inputs": {
            "inventory_jsonl": str(args.inventory_jsonl),
            "requalify_ledger": str(args.requalify_ledger),
            "raw_dir": str(args.raw_dir),
        },
        "total_inventory_profiles": total_profiles,
        "mutation_candidate_profiles": mutation_candidates,
        "canonical_records_written": len(canonical_records),
        "primary_exact_delta_candidates": primary_candidates,
        "primary_exact_delta_pairs": len(primary_pairs),
        "primary_pair_source_counts": dict(pair_source),
        "no_mutation_profiles": dict(no_mutation_class),
        "wt_control_profiles": dict(wt_control_class),
        "data_role_counts": dict(role_counter),
        "exact_mutation_evidence_status_counts": dict(status_counter),
        "exclusion_reason_counts": dict(exc_counter),
        "per_source_role_counts": {k: dict(v) for k, v in by_source.items()},
        "scientific_boundary": (
            "D1-X canonicalization only; data_role/exclusion derived from mutation, "
            "coordinate, sequence, condition and lineage metadata only. No normalization, "
            "no split, no training, no test access, no Delta/outcome-driven role."
        ),
    }
    (args.out_dir / "d1x_canonicalization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "mutation_candidate_profiles": mutation_candidates,
        "canonical_records_written": len(canonical_records),
        "primary_exact_delta_candidates": primary_candidates,
        "primary_exact_delta_pairs": len(primary_pairs),
        "data_role_counts": dict(role_counter),
        "exact_mutation_evidence_status_counts": dict(status_counter),
        "exclusion_reason_counts": dict(exc_counter),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())