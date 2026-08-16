#!/usr/bin/env python3
"""D0-R v2 re-audit: download + parse 101 Tier A non-Ribonanza RDAT files.

READ-ONLY data feasibility re-audit (D0 scope). Does NOT enter D1
(cleaning/normalization/pair-labeling/training). No learned training.

Reuses the proven D0-R methodology (per-profile SEQUENCE:N / sequence: token
recovery, WT anchor identification, single-mutant-pair reconstruction) but
GENERALIZES beyond the M2SL5 124nt functional anchor: for each Tier A file the
functional window is defined by SEQPOS (edits within SEQPOS = functional, edits
outside = flanking/adapter/barcode), and both name-encoded (<pos><ref>-<alt>)
and annotation-encoded (e.g. G159C) mutations are matched against the single
functional edit with DNA->RNA T->U conversion.

Outputs (gitignored artifacts/):
  - d0r_reaudit_tierA_audit.json     : per-file, per-profile audit
  - d0r_reaudit_tierA_relations.json : candidate single-mutant relations
  - d0r_reaudit_tierA_summary.json   : aggregate counts per file/study/parent

Tracked (slim):
  - manifests/reactflow_delta/d0r/d0r_reaudit_tierA_manifest.json

Forward-only: failures recorded honestly, previous results preserved.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reactflow.delta.manifests import sha256_file
from reactflow.delta.rdat import (
    RdatParseError,
    parse_rdat,
    parse_mutations_from_name,
    find_wt_anchor,
    compute_edit_set,
    seqpos_to_indices,
)


SCHEMA_VERSION = "reactflow-delta-d0r-reaudit-tierA-v1"
RELATIONS_SCHEMA_VERSION = "reactflow-delta-d0r-reaudit-tierA-relations-v1"
CANDIDATE_STATUS = "candidate_only_pending_parent_lineage_and_functional_region_validation"

# Already-downloaded D0-R v1 set (skip re-downloading)
DOWNLOADED_V1 = {
    "ETERNA_R78_0000", "ETERNA_R78_0001", "ETERNA_R86_0000",
    "HC16M2R_1M7_0001", "HC16M2R_1M7_0002", "HC16M2R_1M7_0003",
    "HCVDM2_DCP_0000", "M2SL5_2A3_0000", "M2SL5_DMS_0000",
    "SPINACH_DMS_0000", "SPINACH_M2G4_0001",
    "THERM2_DMS_0001", "THERM2_GLX_0001",
}

# Ribonanza/Kaggle library exclusion (Tier B, separate pathway)
RIBO_RE = re.compile(
    r"ribonanza|kaggle|15klib|15krep|pz39|biglib|ok[0-9]|srsbcv|virwin|"
    r"m2pk|m2rfp|m2rfo|archii|rfmrep",
    re.IGNORECASE,
)

# Tier A non-Ribonanza mutation-relevant signals
TIERA_PATS = {
    "mutate_and_map": re.compile(r"mutate[-_ ]?and[-_ ]?map|mutate[-_ ]?map", re.IGNORECASE),
    "m2_seq":         re.compile(r"m2[-_ ]?seq|m2seq", re.IGNORECASE),
    "m2r_rescue":     re.compile(r"m2r|mutate[-_ ]?map[-_ ]?rescue|\brescue\b", re.IGNORECASE),
    "riboSNitch_snp": re.compile(r"ribosnitch|\bsnp\b|single[-_ ]?nucleotide|variant", re.IGNORECASE),
}

# Annotation mutation encoding: <ref><pos><alt> e.g. G159C, G1X (1-indexed)
_ANN_MUTATION = re.compile(r"^([ACGU])(\d+)([ACGUX])$")


def dna_to_rna(base: str) -> str:
    return "U" if base == "T" else base


def entry_text(rec: dict[str, Any]) -> str:
    parts = []
    for k in ("rmdb_id", "name", "description", "comments", "category"):
        v = rec.get(k, "")
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(str(v))
    return " ".join(parts)


def select_tierA_non_ribo(registry_path: Path) -> list[dict[str, Any]]:
    """Select Tier A non-Ribonanza entries not already downloaded in D0-R v1."""
    recs = [json.loads(l) for l in registry_path.open(encoding="utf-8")]
    selected = []
    for r in recs:
        t = entry_text(r)
        if RIBO_RE.search(t):
            continue
        sigs = [s for s, p in TIERA_PATS.items() if p.search(t)]
        if sigs:
            if r["rmdb_id"] in DOWNLOADED_V1:
                continue  # already downloaded in v1
            r["_tierA_signals"] = sigs
            selected.append(r)
    return selected


def _curl_download(url: str, dest: Path, timeout: int = 600) -> tuple[int, str, str, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-fsSL", "--max-time", str(timeout),
        "-w", "%{http_code}\\t%{etag}\\t%{last_modified}",
        "-D", str(dest.with_suffix(dest.suffix + ".headers")),
        "-o", str(dest),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return (0, "", "", result.stderr.strip())
    parts = result.stdout.strip().split("\t")
    while len(parts) < 3:
        parts.append("")
    return (int(parts[0]), parts[1], parts[2], "")


def download_tierA(entries: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    """Download Tier A RDAT files, return manifest records."""
    records = []
    for rec in entries:
        rmdb_id = rec["rmdb_id"]
        url = rec["rdat_url"]
        filename = f"{rmdb_id}.rdat"
        dest = output_dir / filename
        record = {
            "rmdb_id": rmdb_id,
            "filename": filename,
            "rdat_url": url,
            "owner": rec.get("owner"),
            "citation_doi": rec.get("citation", {}).get("doi"),
            "tierA_signals": rec.get("_tierA_signals"),
            "construct_count_metadata": int(rec.get("construct_count", "0")),
        }
        # Skip if already present (idempotent re-run)
        if dest.is_file() and dest.stat().st_size > 0:
            http_status, etag, last_modified, err = 200, "", "", ""
        else:
            http_status, etag, last_modified, err = _curl_download(url, dest)
        record["http_status"] = http_status
        if http_status == 200 and dest.is_file():
            record.update({
                "download_status": "downloaded",
                "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "raw_path": str(dest.resolve()),
            })
        else:
            record.update({
                "download_status": "failed",
                "error": err or f"http_status={http_status}",
                "bytes": None,
                "sha256": None,
                "raw_path": None,
            })
        records.append(record)
        print(json.dumps({
            "rmdb_id": rmdb_id, "status": record["download_status"],
            "bytes": record.get("bytes"),
        }), file=sys.stderr)
    return records


def parse_annotation_mutations(annotation: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Parse annotation-encoded mutations (e.g. mutation:G159C -> 1-indexed)."""
    out = []
    for mv in (annotation or {}).get("mutation", []):
        mv = mv.strip()
        if mv.upper() == "WT":
            continue
        m = _ANN_MUTATION.match(mv)
        if m:
            ref, pos_str, alt = m.group(1), m.group(2), m.group(3)
            out.append({
                "position_1indexed": int(pos_str),
                "position_0indexed": int(pos_str) - 1,
                "ref": ref,
                "alt": "X" if alt == "X" else alt,
                "encoding": mv,
                "source": "annotation",
            })
    return out


def classify_profile_general(
    profile: dict[str, Any],
    wt_profile: dict[str, Any],
    seqpos_indices: list[int],
) -> dict[str, Any]:
    """General single-mutant candidate classifier.

    Uses SEQPOS to partition edits into functional (within SEQPOS) vs flanking.
    Candidate criteria (ALL must hold):
      1. Exactly one name-encoded OR one annotation-encoded mutation.
      2. functional_edit_count == 1.
      3. The single functional edit matches the encoded mutation:
         position (0-indexed, with DNA->RNA for name; 1->0 indexed for annotation
         when no OFFSET, or adjusted by OFFSET), ref, alt.
    """
    name = profile.get("profile_name")
    name_muts = parse_mutations_from_name(name)
    ann_muts = parse_annotation_mutations(profile.get("annotation") or {})

    edit_info = compute_edit_set(
        profile.get("profile_sequence"),
        wt_profile.get("profile_sequence"),
        seqpos_indices,
    )

    result: dict[str, Any] = {
        "profile_index": profile["index"],
        "profile_name": name,
        "name_encoded_mutations": name_muts,
        "annotation_encoded_mutations": ann_muts,
        "edit_info": edit_info,
        "wt_profile_index": wt_profile["index"],
        "wt_profile_name": wt_profile.get("profile_name"),
        "full_edit_count": edit_info["edit_count"],
        "functional_edit_count": edit_info["functional_edit_count"],
        "flanking_edit_count": edit_info["flanking_edit_count"],
    }

    if edit_info["status"] == "skipped":
        result["classification"] = "excluded"
        result["exclusion_reason"] = edit_info["reason"]
        result["lineage_status"] = "excluded"
        result["true_pair"] = False
        return result

    # Combine encoded mutations (prefer name, fall back to annotation)
    encoded = name_muts if name_muts else ann_muts
    encoded_count = len(encoded)

    if encoded_count == 0:
        result["classification"] = "excluded"
        result["exclusion_reason"] = "no_encoded_mutation"
        result["lineage_status"] = "excluded"
        result["true_pair"] = False
        return result

    if encoded_count > 1:
        result["classification"] = "excluded"
        result["exclusion_reason"] = "multiple_encoded_mutations"
        result["lineage_status"] = "excluded"
        result["true_pair"] = False
        return result

    if edit_info["functional_edit_count"] != 1:
        result["classification"] = "excluded"
        result["exclusion_reason"] = (
            f"functional_edit_count_{edit_info['functional_edit_count']}_not_1"
        )
        result["lineage_status"] = "excluded"
        result["true_pair"] = False
        return result

    enc = encoded[0]
    func_edit = edit_info["functional_edits"][0]
    # func_edit position is 1-indexed (compute_edit_set uses position_1indexed)
    actual_pos_1idx = func_edit["position_1indexed"]
    actual_pos_0idx = actual_pos_1idx - 1
    actual_ref = func_edit["wt_base"]
    actual_alt = func_edit["mutant_base"]

    # Name mutations: position is 0-indexed; annotation: position_0indexed field
    if enc.get("source") == "annotation":
        enc_pos_0idx = enc["position_0indexed"]
    else:
        enc_pos_0idx = enc["position"]  # name mutations use "position" key, 0-indexed

    enc_ref_rna = dna_to_rna(enc["ref"])
    # name-encoded mutations use key "mut", annotation-encoded use "alt"
    enc_alt = enc.get("alt", enc.get("mut"))
    enc_alt_rna = dna_to_rna(enc_alt) if enc_alt != "X" else "X"

    pos_match = enc_pos_0idx == actual_pos_0idx
    ref_match = enc_ref_rna == actual_ref
    alt_match = (enc_alt_rna == "X") or (enc_alt_rna == actual_alt)

    if pos_match and ref_match and alt_match:
        result["classification"] = "candidate_single_functional_anchor"
        result["lineage_status"] = CANDIDATE_STATUS
        result["true_pair"] = False
        result["matched_mutation"] = {
            "encoded_position_0indexed": enc_pos_0idx,
            "encoded_ref_rna": enc_ref_rna,
            "encoded_alt_rna": enc_alt_rna,
            "actual_position_1indexed": actual_pos_1idx,
            "actual_position_0indexed": actual_pos_0idx,
            "actual_ref": actual_ref,
            "actual_alt": actual_alt,
            "encoding_source": enc.get("source", "name"),
        }
    else:
        mismatches = []
        if not pos_match:
            mismatches.append(f"position_enc_{enc_pos_0idx}_actual_{actual_pos_0idx}")
        if not ref_match:
            mismatches.append(f"ref_enc_{enc_ref_rna}_actual_{actual_ref}")
        if not alt_match:
            mismatches.append(f"alt_enc_{enc_alt_rna}_actual_{actual_alt}")
        result["classification"] = "excluded"
        result["exclusion_reason"] = "encoded_sequence_mismatch_" + "_".join(mismatches)
        result["lineage_status"] = "excluded"
        result["true_pair"] = False

    return result


def find_wt_anchor_general(
    profiles: list[dict[str, Any]], header_sequence: str | None
) -> tuple[dict[str, Any] | None, str]:
    """Find WT anchor. Returns (profile, method).

    Strategies (forward-only, no imputation):
      1. mutation:WT annotation (with OR without per-profile sequence).
      2. Exact match to header SEQUENCE with no name mutation (sequence-based).
      3. Generic find_wt_anchor fallback (sequence-based).
    """
    # Strategy 1: explicit mutation:WT annotation (works for annotation-only files)
    for p in profiles:
        ann = p.get("annotation") or {}
        mut_vals = ann.get("mutation", [])
        if any(v.strip().upper() == "WT" for v in mut_vals):
            return p, "mutation_wt_annotation"
    # Strategy 2: exact match to header SEQUENCE with no name mutation
    if header_sequence:
        matches = []
        for p in profiles:
            seq = p.get("profile_sequence")
            if seq and seq == header_sequence:
                name = p.get("profile_name")
                if not name or not parse_mutations_from_name(name):
                    matches.append(p)
        if len(matches) >= 1:
            return matches[0], "header_sequence_match"
    # Strategy 3: generic find_wt_anchor fallback
    anchor = find_wt_anchor(profiles)
    return anchor, "find_wt_anchor_fallback" if anchor else "none"


def classify_profile_annotation_only(
    profile: dict[str, Any],
    header_sequence: str | None,
    offset: int,
) -> dict[str, Any]:
    """Annotation-only classifier for files WITHOUT per-profile sequences.

    Used when mutations are encoded solely in ``mutation:`` annotations (e.g.
    standard M2-seq / mutate-and-map-seq files like RNASEP, L21RNA). Each
    construct is a single point mutant by design; the annotation is
    authoritative for the mutation position and ref base. The alt base is "X"
    (any/variable) for most M2-seq files, so alt CANNOT be verified against a
    sequence — candidates remain ``candidate_only`` (true_pair=False).

    Candidate criteria (ALL must hold):
      1. Exactly one ``<ref><pos><alt>`` annotation mutation (alt may be X).
      2. ref matches header SEQUENCE at the annotated position (1-indexed,
         construct-local; also tries OFFSET-adjusted index as a fallback).
    """
    ann_muts = parse_annotation_mutations(profile.get("annotation") or {})

    result: dict[str, Any] = {
        "profile_index": profile["index"],
        "profile_name": profile.get("profile_name"),
        "annotation_encoded_mutations": ann_muts,
        "annotation_mutation_count": len(ann_muts),
    }

    if len(ann_muts) == 0:
        result["classification"] = "excluded"
        result["exclusion_reason"] = "no_annotation_mutation"
        result["lineage_status"] = "excluded"
        result["true_pair"] = False
        return result

    if len(ann_muts) > 1:
        result["classification"] = "excluded"
        result["exclusion_reason"] = "multiple_annotation_mutations"
        result["lineage_status"] = "excluded"
        result["true_pair"] = False
        return result

    mut = ann_muts[0]
    pos_1idx = mut["position_1indexed"]
    enc_ref = mut["ref"]
    enc_alt = mut["alt"]

    # Verify ref against header SEQUENCE (construct-local 1-indexed)
    ref_match = False
    matched_index: str | None = None
    if header_sequence:
        idx_local = pos_1idx - 1
        if 0 <= idx_local < len(header_sequence):
            if header_sequence[idx_local] == enc_ref:
                ref_match = True
                matched_index = "construct_local_1indexed"
        if not ref_match and offset:
            idx_offset = pos_1idx - 1 + offset
            if 0 <= idx_offset < len(header_sequence):
                if header_sequence[idx_offset] == enc_ref:
                    ref_match = True
                    matched_index = "offset_adjusted"

    if ref_match:
        result["classification"] = "candidate_single_annotation_only"
        result["lineage_status"] = CANDIDATE_STATUS
        result["true_pair"] = False
        result["matched_mutation"] = {
            "encoded_position_1indexed": pos_1idx,
            "encoded_ref": enc_ref,
            "encoded_alt": enc_alt,
            "ref_verified_against": "header_SEQUENCE",
            "ref_match_index": matched_index,
            "alt_not_verified": enc_alt == "X",
            "encoding_source": "annotation",
            "note": "alt base is X (variable) for M2-seq; sequence-level edit not verifiable without per-profile sequence",
        }
    else:
        result["classification"] = "excluded"
        reason = "annotation_ref_mismatch_header"
        if header_sequence:
            idx_local = pos_1idx - 1
            actual = header_sequence[idx_local] if 0 <= idx_local < len(header_sequence) else "?"
            reason = f"annotation_ref_mismatch_enc_{enc_ref}_actual_{actual}_at_{pos_1idx}"
        result["exclusion_reason"] = reason
        result["lineage_status"] = "excluded"
        result["true_pair"] = False

    return result


def audit_file(rdat_path: Path) -> dict[str, Any]:
    """Parse + audit one RDAT file for single-mutant candidates."""
    try:
        document = parse_rdat(rdat_path)
    except (RdatParseError, FileNotFoundError) as exc:
        return {
            "rdat_path": str(rdat_path),
            "rdat_name": rdat_path.stem,
            "parse_status": "error",
            "parse_error": str(exc),
        }

    profiles = document["profiles"]
    header_seq = document["headers"].get("SEQUENCE")
    seqpos_indices = seqpos_to_indices(document.get("seqpos") or [])
    wt_anchor, wt_method = find_wt_anchor_general(profiles, header_seq)

    # Detect whether per-profile sequences exist (M2SL5-style) or not
    profiles_with_seq = sum(1 for p in profiles if p.get("profile_sequence"))
    annotation_only_mode = profiles_with_seq == 0

    # OFFSET for annotation ref verification
    try:
        offset_val = int(document["headers"].get("OFFSET", "0") or "0")
    except ValueError:
        offset_val = 0

    profile_audits = []
    candidate_count = 0
    exclusion_reasons: Counter[str] = Counter()

    if wt_anchor is None:
        for p in profiles:
            profile_audits.append({
                "profile_index": p["index"],
                "profile_name": p.get("profile_name"),
                "role": "no_wt_anchor",
                "lineage_status": "no_wt_anchor",
            })
        exclusion_reasons["no_wt_anchor"] = len(profiles)
    else:
        for p in profiles:
            if p["index"] == wt_anchor["index"]:
                profile_audits.append({
                    "profile_index": p["index"],
                    "profile_name": p.get("profile_name"),
                    "role": "wt_anchor",
                    "lineage_status": "wt_anchor",
                })
                continue
            if annotation_only_mode:
                cls = classify_profile_annotation_only(p, header_seq, offset_val)
                cand_label = "candidate_single_annotation_only"
            else:
                cls = classify_profile_general(p, wt_anchor, seqpos_indices)
                cand_label = "candidate_single_functional_anchor"
            if cls.get("classification") == cand_label:
                candidate_count += 1
            else:
                reason = cls.get("exclusion_reason", "unknown")
                if reason.startswith("functional_edit_count"):
                    reason = "functional_edit_count_not_1"
                elif reason.startswith("encoded_sequence_mismatch"):
                    reason = "encoded_sequence_mismatch"
                elif reason.startswith("annotation_ref_mismatch"):
                    reason = "annotation_ref_mismatch"
                exclusion_reasons[reason] += 1
            profile_audits.append({
                "profile_index": p["index"],
                "profile_name": p.get("profile_name"),
                "role": "mutant_candidate",
                "classification": cls.get("classification"),
                "exclusion_reason": cls.get("exclusion_reason"),
                "full_edit_count": cls.get("full_edit_count"),
                "functional_edit_count": cls.get("functional_edit_count"),
                "flanking_edit_count": cls.get("flanking_edit_count"),
                "annotation_mutation_count": cls.get("annotation_mutation_count"),
                "matched_mutation": cls.get("matched_mutation"),
            })

    return {
        "rdat_path": str(rdat_path),
        "rdat_sha256": document["sha256"],
        "rdat_name": document["headers"]["NAME"],
        "rdat_version": document["headers"].get("RDAT_VERSION"),
        "parse_status": "ok",
        "header_sequence_length": len(header_seq) if header_seq else 0,
        "offset": offset_val,
        "seqpos_count": len(document.get("seqpos") or []),
        "seqpos_indices_sample": seqpos_indices[:5],
        "profile_count": len(profiles),
        "profiles_with_sequence": profiles_with_seq,
        "annotation_only_mode": annotation_only_mode,
        "wt_anchor_found": wt_anchor is not None,
        "wt_anchor_method": wt_method,
        "wt_anchor_index": wt_anchor["index"] if wt_anchor else None,
        "wt_anchor_name": wt_anchor.get("profile_name") if wt_anchor else None,
        "candidate_single_count": candidate_count,
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "profile_audits": profile_audits,
    }


def build_relations(audit: dict[str, Any], rec: dict[str, Any], rdat_path: Path) -> list[dict[str, Any]]:
    """Extract candidate relations from an audit."""
    relations = []
    if audit.get("parse_status") != "ok" or not audit.get("wt_anchor_found"):
        return relations
    wt_index = audit["wt_anchor_index"]
    owner = rec.get("owner")
    annotation_only = audit.get("annotation_only_mode", False)
    audit_method = (
        "annotation_only_mutation_ref_verified_against_header"
        if annotation_only
        else "general_seqpos_functional_window_single_mutant_match"
    )
    candidate_labels = (
        {"candidate_single_annotation_only"} if annotation_only
        else {"candidate_single_functional_anchor"}
    )
    # parent = rmdb_id prefix (strip trailing _0000 batch suffix)
    rmdb_id = audit["rdat_name"]
    parent_prefix = re.sub(r"_\d+$", "", rmdb_id)
    for pa in audit["profile_audits"]:
        if pa.get("classification") not in candidate_labels:
            continue
        mm = pa.get("matched_mutation") or {}
        relations.append({
            "schema_version": RELATIONS_SCHEMA_VERSION,
            "source": "RMDB",
            "rmdb_id": rmdb_id,
            "parent_prefix": parent_prefix,
            "owner": owner,
            "citation_doi": rec.get("citation", {}).get("doi"),
            "rdat_sha256": audit["rdat_sha256"],
            "rdat_path": str(rdat_path),
            "modifier": (rec.get("annotation") or {}).get("modifier", ["unknown"])[0]
                if (rec.get("annotation") or {}).get("modifier") else "unknown",
            "wt_profile_index": wt_index,
            "mutant_profile_index": pa["profile_index"],
            "mutant_profile_name": pa["profile_name"],
            "matched_mutation": mm,
            "full_edit_count": pa.get("full_edit_count"),
            "functional_edit_count": pa.get("functional_edit_count"),
            "flanking_edit_count": pa.get("flanking_edit_count"),
            "annotation_mutation_count": pa.get("annotation_mutation_count"),
            "lineage_status": CANDIDATE_STATUS,
            "true_pair": False,
            "audit_method": audit_method,
        })
    return relations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, required=True,
                    help="d0r_accession_registry.jsonl path")
    ap.add_argument("--download-dir", type=Path, required=True,
                    help="directory for downloaded Tier A RDAT files")
    ap.add_argument("--artifacts-dir", type=Path, required=True,
                    help="gitignored artifacts output directory")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="slim tracked manifest output path")
    ap.add_argument("--skip-download", action="store_true",
                    help="skip download, audit existing files in download-dir")
    args = ap.parse_args(argv)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).astimezone().isoformat()

    # Step 1: select Tier A non-Ribonanza entries not in v1
    entries = select_tierA_non_ribo(args.registry)
    print(f"Selected {len(entries)} Tier A non-Ribonanza entries (excluding v1 downloaded)", file=sys.stderr)

    # Step 2: download
    if not args.skip_download:
        dl_records = download_tierA(entries, args.download_dir)
    else:
        dl_records = []
        for rec in entries:
            dest = args.download_dir / f"{rec['rmdb_id']}.rdat"
            r = {
                "rmdb_id": rec["rmdb_id"],
                "filename": f"{rec['rmdb_id']}.rdat",
                "rdat_url": rec["rdat_url"],
                "owner": rec.get("owner"),
                "citation_doi": rec.get("citation", {}).get("doi"),
                "tierA_signals": rec.get("_tierA_signals"),
                "construct_count_metadata": int(rec.get("construct_count", "0")),
                "http_status": 200 if dest.is_file() else 0,
            }
            if dest.is_file():
                r.update({
                    "download_status": "downloaded",
                    "bytes": dest.stat().st_size,
                    "sha256": sha256_file(dest),
                    "raw_path": str(dest.resolve()),
                })
            else:
                r.update({"download_status": "missing", "bytes": None, "sha256": None, "raw_path": None})
            dl_records.append(r)

    dl_ok = [r for r in dl_records if r["download_status"] == "downloaded"]
    dl_failed = [r for r in dl_records if r["download_status"] != "downloaded"]
    print(f"Download: {len(dl_ok)} ok, {len(dl_failed)} failed", file=sys.stderr)

    # Step 3: audit each downloaded file
    audits = []
    all_relations = []
    parse_errors = []
    for rec in entries:
        rmdb_id = rec["rmdb_id"]
        rdat_path = args.download_dir / f"{rmdb_id}.rdat"
        if not rdat_path.is_file():
            audits.append({"rmdb_id": rmdb_id, "parse_status": "missing_file"})
            continue
        audit = audit_file(rdat_path)
        audit["rmdb_id"] = rmdb_id
        audit["owner"] = rec.get("owner")
        audit["citation_doi"] = rec.get("citation", {}).get("doi")
        audit["construct_count_metadata"] = int(rec.get("construct_count", "0"))
        audit["tierA_signals"] = rec.get("_tierA_signals")
        audits.append(audit)
        if audit.get("parse_status") == "error":
            parse_errors.append({"rmdb_id": rmdb_id, "error": audit["parse_error"]})
            continue
        rels = build_relations(audit, rec, rdat_path)
        all_relations.extend(rels)
        print(f"{rmdb_id}: {audit.get('candidate_single_count', 0)} candidates "
              f"(profiles={audit.get('profile_count', 0)}, wt={audit.get('wt_anchor_found')})",
              file=sys.stderr)

    # Step 4: aggregate
    per_file = []
    for a in audits:
        per_file.append({
            "rmdb_id": a.get("rmdb_id"),
            "owner": a.get("owner"),
            "parse_status": a.get("parse_status"),
            "profile_count": a.get("profile_count"),
            "wt_anchor_found": a.get("wt_anchor_found"),
            "candidate_single_count": a.get("candidate_single_count", 0),
            "exclusion_reason_counts": a.get("exclusion_reason_counts", {}),
            "parse_error": a.get("parse_error"),
        })

    # study = (owner, doi); parent = parent_prefix
    study_counts: Counter[tuple] = Counter()
    parent_counts: Counter[str] = Counter()
    owner_counts: Counter[str] = Counter()
    for rel in all_relations:
        study_counts[(rel["owner"], rel["citation_doi"])] += 1
        parent_counts[rel["parent_prefix"]] += 1
        owner_counts[rel["owner"]] += 1

    total_candidates = len(all_relations)
    distinct_studies = len(study_counts)
    distinct_parents = len(parent_counts)
    distinct_owners = len(owner_counts)

    # Tier judgment per contract §8 gates (Tier A: >=5 studies, >=20 parents, >=5000 pairs)
    tierA_pair_gate = 5000
    tierA_study_gate = 5
    tierA_parent_gate = 20
    tierB_pair_gate = 1000
    tierB_study_gate = 3
    tierB_parent_gate = 10

    meets_tierA = (distinct_studies >= tierA_study_gate
                   and distinct_parents >= tierA_parent_gate
                   and total_candidates >= tierA_pair_gate)
    meets_tierB = (distinct_studies >= tierB_study_gate
                   and distinct_parents >= tierB_parent_gate
                   and total_candidates >= tierB_pair_gate)

    if meets_tierA:
        re_tier = "Tier A"
    elif meets_tierB:
        re_tier = "Tier B"
    else:
        re_tier = "below_Tier_B_gates"

    # Re-judge d1_allowed: forward-only, do NOT authorize D1 (v3.1 not published)
    re_d1_allowed = False
    re_triage = "reaudit_qualified_to_propose_v3_1_non_learning_d1_cleanup_only"
    if total_candidates == 0:
        re_triage = "reaudit_no_candidates_d0_incomplete"

    # Step 5: write artifacts
    audit_path = args.artifacts_dir / "d0r_reaudit_tierA_audit.json"
    relations_path = args.artifacts_dir / "d0r_reaudit_tierA_relations.json"
    summary_path = args.artifacts_dir / "d0r_reaudit_tierA_summary.json"

    with audit_path.open("w", encoding="utf-8") as f:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "stage": "D0-R v2 re-audit",
            "generated_at": now,
            "audit_method": "general_seqpos_functional_window_single_mutant_match",
            "files_audited": len(audits),
            "parse_errors": parse_errors,
            "audits": audits,
        }, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote audit: {audit_path}", file=sys.stderr)

    with relations_path.open("w", encoding="utf-8") as f:
        json.dump({
            "schema_version": RELATIONS_SCHEMA_VERSION,
            "stage": "D0-R v2 re-audit",
            "generated_at": now,
            "total_candidate_relations": total_candidates,
            "relations": all_relations,
            "lineage_status_all": CANDIDATE_STATUS,
            "true_pair_all": False,
        }, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote relations: {relations_path} ({total_candidates} relations)", file=sys.stderr)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "D0-R v2 re-audit",
        "generated_at": now,
        "audit_method": "general_seqpos_functional_window_single_mutant_match",
        "tierA_entries_selected": len(entries),
        "downloaded_ok": len(dl_ok),
        "downloaded_failed": len(dl_failed),
        "files_audited": len([a for a in audits if a.get("parse_status") == "ok"]),
        "parse_errors": len(parse_errors),
        "total_candidate_single_mutant_pairs": total_candidates,
        "distinct_studies": distinct_studies,
        "distinct_parents": distinct_parents,
        "distinct_owners": distinct_owners,
        "per_owner_candidate_counts": dict(sorted(owner_counts.items(), key=lambda x: -x[1])),
        "per_parent_candidate_counts": dict(sorted(parent_counts.items(), key=lambda x: -x[1])),
        "per_file": sorted(per_file, key=lambda x: -x.get("candidate_single_count", 0)),
        "tier_gates": {
            "Tier A": {"studies": tierA_study_gate, "parents": tierA_parent_gate, "pairs": tierA_pair_gate},
            "Tier B": {"studies": tierB_study_gate, "parents": tierB_parent_gate, "pairs": tierB_pair_gate},
        },
        "re_tier_judgment": re_tier,
        "re_d1_allowed": re_d1_allowed,
        "re_triage_decision": re_triage,
        "scientific_boundary": (
            "Read-only D0 re-audit. Candidate relations are unverified "
            "(candidate_only_pending_parent_lineage_and_functional_region_validation, "
            "true_pair=False). No pair, tier, or model claim. D1 not authorized "
            "(v3.1 contract not published). Previous D0-R v1 results preserved."
        ),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote summary: {summary_path}", file=sys.stderr)

    # Slim tracked manifest
    slim_manifest = {
        "schema_version": "reactflow-delta-d0r-reaudit-tierA-manifest-v1",
        "stage": "D0-R v2 re-audit",
        "generated_at": now,
        "tierA_entries_selected": len(entries),
        "downloaded_ok": len(dl_ok),
        "downloaded_failed": len(dl_failed),
        "files_audited": len([a for a in audits if a.get("parse_status") == "ok"]),
        "parse_errors": len(parse_errors),
        "total_candidate_single_mutant_pairs": total_candidates,
        "distinct_studies": distinct_studies,
        "distinct_parents": distinct_parents,
        "distinct_owners": distinct_owners,
        "re_tier_judgment": re_tier,
        "re_d1_allowed": re_d1_allowed,
        "re_triage_decision": re_triage,
        "artifacts_paths": {
            "audit": str(audit_path),
            "relations": str(relations_path),
            "summary": str(summary_path),
        },
        "artifacts_sha256": {
            "audit": sha256_file(audit_path),
            "relations": sha256_file(relations_path),
            "summary": sha256_file(summary_path),
        },
        "download_manifest_records": dl_records,
        "scientific_boundary": summary["scientific_boundary"],
    }
    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(slim_manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote manifest: {args.manifest}", file=sys.stderr)

    print(json.dumps({
        "total_candidates": total_candidates,
        "distinct_studies": distinct_studies,
        "distinct_parents": distinct_parents,
        "distinct_owners": distinct_owners,
        "re_tier": re_tier,
        "re_d1_allowed": re_d1_allowed,
        "parse_errors": len(parse_errors),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
