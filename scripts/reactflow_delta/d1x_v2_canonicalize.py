#!/usr/bin/env python3
"""D1-X v2: strict, versioned canonical builder with per-position eligibility codes.

This is a rebuild of the parser/canonical mask for the ReactFlowDelta M0X audit
(R2B, contract ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md
section 13.2).  It fixes the historical RDAT separator bug and rescues
salvageable parse failures, and it emits, for every primary position, an explicit
eligibility reason code from the frozen endpoint_v2 mask rule set:

    EDITED_SITE, ALIGNMENT_CHANGE, PROBE_ELIGIBILITY_CHANGE,
    MISSING_REACTIVITY, LENGTH_MISMATCH, ELIGIBLE

Contract invariants honored:
  * missing reactivity is NEVER imputed as zero -> MISSING_REACTIVITY
  * no failure is silently dropped -> asset disposition ledger
  * every primary position carries an explicit eligibility reason code
  * legacy d1x files are NOT modified; all output goes to NEW paths

Root causes fixed (confirmed by inspecting representative raw .rdat files):
  1. Mixed tab/space separators: many legacy files write ``KEY<spaces>...<TAB>...``
     so the old parser (splitting on tab because a tab exists) corrupted the key.
     Fix: whitespace-normalized tokenization.
  2. RDAT 0.4 "DATA:N" block rows (with DATA_ANNOTATION datatype tags) are used
     instead of ``REACTIVITY:N`` rows by some SHAPE-Seq files.  Fix: fall back to
     DATA rows carrying datatype:REACTIVITY / REACTIVITY_ERROR.
  3. Signed / offset SEQPOS ordinals (e.g. ``-10``, ``0``, ``g-18``, ``X0``).
     Fix: accept an optional base prefix and a signed integer.
  4. Empty annotation tokens (bare ``ANNOTATION`` lines / trailing separators)
     and orphan non-colon tokens.  Fix: skip-and-retain, never fail.
  5. Empty ``STRUCTURE`` header.  Fix: optional (default None).

Usage::

    PYTHONPATH=src python scripts/reactflow_delta/d1x_v2_canonicalize.py \
        --asset-manifest  <rmdb_release_assets_*.jsonl> \
        --raw-dir         <raw rdat dir> \
        --legacy-records  <d1x_canonical_records.jsonl> \
        --legacy-pairs    <d1x_primary_pairs.jsonl> \
        --out-registry    <data_registry/d1x_v2> \
        --out-artifacts   <artifact root>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reactflow.delta.d0x import parse_mutation_value  # noqa: E402

CANONICAL_SCHEMA_V2 = "reactflow_delta.data_record.v4.0.v2"
PAIR_SCHEMA_V2 = "reactflow_delta.d1x_pair.v2"
DISPOSITION_SCHEMA_V2 = "reactflow_delta.d1x_v2_disposition.v1"
CROSSWALK_SCHEMA_V2 = "reactflow_delta.d1x_v2_vs_v1_crosswalk.v1"
MANIFEST_SCHEMA_V2 = "reactflow_delta.d1x_v2_manifest.v1"

ELIGIBILITY_CODES = (
    "EDITED_SITE",
    "ALIGNMENT_CHANGE",
    "PROBE_ELIGIBILITY_CHANGE",
    "MISSING_REACTIVITY",
    "LENGTH_MISMATCH",
    "ELIGIBLE",
)

# indexed row keys that carry per-profile numeric/annotation blocks
_INDEXED_INT = re.compile(r"^(REACTIVITY|REACTIVITY_ERROR|DATA|ANNOTATION_DATA|DATA_ANNOTATION|SEQUENCE):([0-9]+)$")
_SEQPOS_TOKEN = re.compile(r"^([ACGUNX])?(-?[0-9]+)$", re.IGNORECASE)
_CONDITION_KEYS = ("modifier", "temperature", "chemical", "experimentType")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_map_tolerant(values: list[str]) -> dict[str, list[str]]:
    """Parse annotation tokens into {key:[values]}, retaining orphan tokens.

    Empty tokens and tokens without a ':' separator are retained in a special
    ``_orphan`` bucket instead of failing the parse (legacy files frequently
    carry bare annotation lines or space-split trailing words).  No scientific
    data is dropped: orphan tokens are recorded verbatim.
    """
    parsed: dict[str, list[str]] = {}
    for token in values:
        if not token:
            parsed.setdefault("_orphan", []).append(token)
            continue
        if ":" not in token:
            parsed.setdefault("_orphan", []).append(token)
            continue
        key, value = token.split(":", 1)
        if not key:
            parsed.setdefault("_orphan", []).append(token)
            continue
        parsed.setdefault(key, []).append(value)
    return parsed


def _numeric_values(values: list[str], key: str) -> list[float | None]:
    """Convert value tokens to floats, mapping NaN/empty/Inf to None (missing)."""
    result: list[float | None] = []
    for value in values:
        stripped = value.strip()
        if not stripped or stripped.lower() in ("nan", "n/a", "na"):
            result.append(None)
            continue
        try:
            number = float(stripped)
        except ValueError:
            # Do NOT silently drop a numeric-looking token: surface it.
            raise ValueError(f"non-numeric value in {key}: {value!r}")
        if not math.isfinite(number):
            result.append(None)
            continue
        result.append(number)
    return result


def _parse_seqpos_tokens(tokens: list[str]) -> list[int]:
    """Parse SEQPOS tokens into signed integers, tolerating a base prefix.

    Raises ValueError on the first token that cannot be interpreted, so a truly
    malformed SEQPOS is still reported (never silently dropped).
    """
    out: list[int] = []
    for ordinal, raw in enumerate(tokens, 1):
        m = _SEQPOS_TOKEN.fullmatch(raw.strip())
        if m is None:
            raise ValueError(f"malformed SEQPOS token at ordinal {ordinal}: {raw!r}")
        out.append(int(m.group(2)))
    if len(out) != len(set(out)):
        raise ValueError("duplicate SEQPOS coordinate")
    return out


def _data_reactivity_vectors(
    data_rows: dict[int, list[str]],
    data_annotations: dict[int, dict[str, list[str]]],
) -> tuple[dict[int, list[float | None]], dict[int, list[float | None]]]:
    """Build reactivity/error vectors from DATA:N rows by their datatype tags."""
    reactivity: dict[int, list[float | None]] = {}
    reactivity_error: dict[int, list[float | None]] = {}
    for index, row in data_rows.items():
        ann = data_annotations.get(index, {})
        datatypes = ann.get("datatype", [])
        for dt in datatypes:
            dt_up = dt.upper()
            if dt_up.startswith("REACTIVITY_ERROR"):
                reactivity_error[index] = _numeric_values(row, f"DATA:{index}")
                break
            if dt_up.startswith("REACTIVITY"):
                reactivity[index] = _numeric_values(row, f"DATA:{index}")
                break
    return reactivity, reactivity_error


def parse_rdat_v2(path: str | Path) -> dict[str, Any]:
    """Corrected strict RDAT parser (whitespace-normalized + DATA rows).

    Returns a dict shaped like the historical ``parse_rdat`` output plus extra
    ``seqpos_raw`` and ``profiles`` entries carrying per-profile annotation.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    headers: dict[str, str] = {}
    comments: list[str] = []
    global_annotations: list[dict[str, list[str]]] = []
    profile_annotations: dict[int, dict[str, list[str]]] = {}
    data_annotations: dict[int, dict[str, list[str]]] = {}
    data_rows: dict[int, list[str]] = {}
    seqpos_raw: list[str] | None = None
    reactivity: dict[int, list[float | None]] = {}
    reactivity_error: dict[int, list[float | None]] = {}
    profile_sequences: dict[int, str] = {}

    def _index(key: str, line_number: int) -> int:
        m = _INDEXED_INT.fullmatch(key)
        if m is None:
            raise ValueError(f"invalid indexed key at line {line_number}: {key!r}")
        index = int(m.group(2))
        if index < 1:
            raise ValueError(f"indexed key must be positive: {key!r}")
        return index

    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        fields = raw_line.split()
        if not fields:
            continue
        key = fields[0]
        values = fields[1:]
        if key == "COMMENT":
            comments.append(" ".join(values).strip())
        elif key == "ANNOTATION":
            global_annotations.append(_annotation_map_tolerant(values))
        elif key.startswith("ANNOTATION_DATA:"):
            index = _index(key, line_number)
            profile_annotations[index] = _annotation_map_tolerant(values)
        elif key.startswith("DATA_ANNOTATION:"):
            index = _index(key, line_number)
            data_annotations[index] = _annotation_map_tolerant(values)
        elif key == "SEQPOS":
            if seqpos_raw is not None:
                raise ValueError("duplicate SEQPOS")
            if not values:
                raise ValueError("SEQPOS is empty")
            seqpos_raw = list(values)
        elif key.startswith("REACTIVITY:"):
            index = _index(key, line_number)
            if index in reactivity:
                raise ValueError(f"duplicate REACTIVITY index {index}")
            reactivity[index] = _numeric_values(values, key)
        elif key.startswith("REACTIVITY_ERROR:"):
            index = _index(key, line_number)
            if index in reactivity_error:
                raise ValueError(f"duplicate REACTIVITY_ERROR index {index}")
            reactivity_error[index] = _numeric_values(values, key)
        elif key.startswith("DATA:"):
            index = _index(key, line_number)
            data_rows[index] = list(values)
        elif key.startswith("SEQUENCE:"):
            index = _index(key, line_number)
            profile_sequences[index] = "".join(values)
        elif key in {"RDAT_VERSION", "VERSION", "NAME", "SEQUENCE", "STRUCTURE", "OFFSET"}:
            if key in headers:
                raise ValueError(f"duplicate header {key}")
            if key == "STRUCTURE" and not values:
                # legacy files may leave STRUCTURE empty; tolerate it
                headers[key] = ""
                continue
            if not values:
                raise ValueError(f"header {key} requires one non-empty value")
            headers[key] = values[0] if len(values) == 1 else " ".join(values)
        elif key.startswith(("TRACE:", "READS:")):
            continue
        else:
            headers.setdefault(f"unknown:{key}", " ".join(values))

    if "VERSION" in headers and "RDAT_VERSION" not in headers:
        headers["RDAT_VERSION"] = headers["VERSION"]
    elif "VERSION" in headers and "RDAT_VERSION" in headers:
        if headers["VERSION"] != headers["RDAT_VERSION"]:
            raise ValueError("conflicting VERSION and RDAT_VERSION")

    if headers.get("RDAT_VERSION") not in {"0.4", "0.22", "0.24", "0.32", "0.34"}:
        raise ValueError(
            f"RDAT_VERSION {headers.get('RDAT_VERSION')!r} is not accepted"
        )
    for required in ("NAME", "SEQUENCE"):
        if required not in headers:
            raise ValueError(f"missing required header {required}")
    if seqpos_raw is None:
        raise ValueError("missing SEQPOS")

    # Fall back to DATA:N rows when no REACTIVITY:N rows are present.
    if not reactivity:
        reactivity, reactivity_error = _data_reactivity_vectors(
            data_rows, data_annotations
        )
    if not reactivity:
        raise ValueError("missing REACTIVITY rows")

    seqpos = _parse_seqpos_tokens(seqpos_raw)

    profiles = []
    for index in sorted(reactivity):
        values = reactivity[index]
        annotation = profile_annotations.get(index, data_annotations.get(index, {}))
        profile_sequence = profile_sequences.get(index)
        sequence_source = "sequence_indexed_line" if index in profile_sequences else None
        if profile_sequence is None:
            seq_values = annotation.get("sequence", [])
            if len(seq_values) == 1 and re.fullmatch(r"[ACGUN]+", seq_values[0], re.I):
                profile_sequence = seq_values[0]
                sequence_source = "annotation_sequence_token"
        name_values = annotation.get("name", [])
        profile_name = name_values[0] if len(name_values) == 1 else None
        profiles.append(
            {
                "index": index,
                "annotation": annotation,
                "reactivity": values,
                "reactivity_error": reactivity_error.get(index),
                "missing_reactivity_count": sum(v is None for v in values),
                "profile_sequence": profile_sequence,
                "profile_sequence_source": sequence_source,
                "profile_name": profile_name,
            }
        )
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "headers": headers,
        "comments": comments,
        "global_annotations": global_annotations,
        "seqpos": seqpos,
        "seqpos_raw": seqpos_raw,
        "profiles": profiles,
        "profile_annotations": profile_annotations,
        "orphan_annotation_indices": sorted(set(profile_annotations) - set(reactivity)),
    }


# ---------------------------------------------------------------------------
# annotation / condition resolution
# ---------------------------------------------------------------------------


def _resolve_profile_annotations(
    global_annotations: list[dict[str, list[str]]],
    profile_annotation: dict[str, list[str]],
) -> dict[str, dict[str, list[str]]]:
    """Resolve per-profile annotations with global (construct) inheritance."""
    construct: dict[str, list[str]] = {}
    for block in global_annotations:
        for k, v in block.items():
            if k == "_orphan":
                continue
            construct.setdefault(k, []).extend(v)
    keys = sorted(set(construct) | set(profile_annotation))
    resolved: dict[str, dict[str, list[str]]] = {}
    for k in keys:
        profile_values = list(profile_annotation.get(k, []))
        inherited = list(construct.get(k, []))
        use_profile = bool(profile_values)
        resolved[k] = {
            "resolved_values": profile_values if use_profile else inherited,
            "resolution": "PROFILE_OVERRIDE" if use_profile else "CONSTRUCT_INHERITED",
        }
    return resolved


def _norm(vals: list[str]) -> tuple[str, ...]:
    return tuple(sorted({str(v).strip() for v in (vals or []) if str(v).strip()}))


def _condition_tuple(ra: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    def _cv(key: str) -> tuple[str, ...]:
        return _norm((ra.get(key) or {}).get("resolved_values", []))
    return {
        "modifier": _cv("modifier"),
        "temperature": _cv("temperature"),
        "chemical": _cv("chemical"),
        "experimentType": _cv("experimentType"),
        "all_required_known": bool(
            _cv("modifier") and _cv("temperature")
            and _cv("chemical") and _cv("experimentType")
        ),
    }


def _same_condition(a: dict, b: dict) -> str:
    for key in _CONDITION_KEYS:
        va, vb = a.get(key), b.get(key)
        if not va or not vb:
            return "MISSING_REQUIRED_FIELD"
        if va != vb:
            return {
                "modifier": "MISMATCH_PROBE",
                "temperature": "MISMATCH_TEMPERATURE",
                "chemical": "MISMATCH_BUFFER",
                "experimentType": "MISMATCH_ENVIRONMENT",
            }[key]
    return "MATCHED_ALL_REQUIRED"


# ---------------------------------------------------------------------------
# mutation parsing (reuse the frozen d0x parse_mutation_value)
# ---------------------------------------------------------------------------


def _parse_mutations(profile: dict, resolved: dict, offset: int) -> list[dict[str, Any]]:
    mutation_values = (resolved.get("mutation") or {}).get("resolved_values", [])
    parsed = []
    for value in mutation_values:
        parsed.append(
            parse_mutation_value(
                value,
                header_sequence="",
                offset=offset,
                profile_sequence=profile.get("profile_sequence"),
            )
        )
    return parsed


# ---------------------------------------------------------------------------
# per-position eligibility mask
# ---------------------------------------------------------------------------


def compute_position_eligibility(
    reactivity: list[float | None],
    edited_indices: set[int],
    alignment_change_indices: set[int],
    probe_change: bool,
) -> list[str]:
    """Return one eligibility_reason_code per position (endpoint_v2 mask rules).

    Precedence (first match wins):
      1. MISSING_REACTIVITY  - NaN / non-finite / absent reactivity (never zero-filled)
      2. EDITED_SITE         - the mutation site itself
      3. ALIGNMENT_CHANGE    - position where the mutant aligns differently than WT
      4. PROBE_ELIGIBILITY_CHANGE - probe eligibility differs from the WT anchor
      5. ELIGIBLE            - usable for response evaluation
    """
    codes: list[str] = []
    for i, value in enumerate(reactivity):
        if not (isinstance(value, (int, float)) and math.isfinite(value)):
            codes.append("MISSING_REACTIVITY")
        elif i in edited_indices:
            codes.append("EDITED_SITE")
        elif i in alignment_change_indices:
            codes.append("ALIGNMENT_CHANGE")
        elif probe_change:
            codes.append("PROBE_ELIGIBILITY_CHANGE")
        else:
            codes.append("ELIGIBLE")
    return codes


def compute_alignment_change_indices(
    mutant_seq: str | None,
    wt_seq: str | None,
    seqpos_values: list[int],
) -> set[int]:
    """Indices (into seqpos order) where mutant and WT sequences differ.

    Uses the RDAT position mapping: sequence_index = position - offset - 1 is
    folded into this caller via seqpos_values aligned 1:1 with reactivity.
    Here we compare the per-position base only when both sequences are present.
    """
    if not mutant_seq or not wt_seq or len(mutant_seq) != len(wt_seq):
        return set()
    indices: set[int] = set()
    # seqpos_values[i] is the 1-based position label; the base of the construct
    # at that label is what we compare.  Fall back to ordinal alignment.
    for i, pos in enumerate(seqpos_values):
        seq_index = pos - 1
        if 0 <= seq_index < len(mutant_seq):
            if mutant_seq[seq_index].upper() != wt_seq[seq_index].upper():
                indices.add(i)
    return indices


# ---------------------------------------------------------------------------
# canonical record / pair building
# ---------------------------------------------------------------------------


def _edited_site_indices(parsed_mutations: list[dict], seqpos_values: list[int]) -> set[int]:
    """Return seqpos-order indices equal to any parsed mutation position."""
    indices: set[int] = set()
    for m in parsed_mutations:
        for edit in m.get("edits") or []:
            pos = edit.get("source_coordinate_1_based")
            if pos is None:
                continue
            for i, sp in enumerate(seqpos_values):
                if sp == pos:
                    indices.add(i)
    return indices


def _is_wt_annotation(parsed_mutations: list[dict]) -> bool:
    if not parsed_mutations:
        return True
    kinds = {m.get("kind") for m in parsed_mutations}
    if kinds == {"WT"}:
        return True
    return False


def _profile_data_role(parsed_mutations: list[dict]) -> dict[str, Any]:
    """Derive provisional data role + exclusion from parsed mutations."""
    if not parsed_mutations:
        return {"data_role": None, "exclusion_reason": None, "is_wt": True,
                "ref": None, "alt": None, "edits": []}
    edits = [e for m in parsed_mutations for e in m.get("edits") or []]
    kinds = {m.get("kind") for m in parsed_mutations}
    if kinds == {"WT"}:
        return {"data_role": None, "exclusion_reason": None, "is_wt": True,
                "ref": None, "alt": None, "edits": []}
    if len(edits) > 1:
        return {"data_role": "RESCUE_MULTI_EDIT", "exclusion_reason": "MULTI_EDIT",
                "is_wt": False, "ref": None, "alt": None, "edits": edits}
    if kinds == {"LATENT_ALT_SINGLE_SUBSTITUTION"} and edits:
        return {"data_role": "AUXILIARY_LATENT_ALT", "exclusion_reason": "LATENT_ALT",
                "is_wt": False, "ref": None, "alt": None, "edits": edits}
    if kinds == {"EXACT_SINGLE_SUBSTITUTION"} and edits:
        e = edits[0]
        ref = (e.get("ref_allele") or "").upper() or None
        alt = (e.get("alt_allele") or "").upper() or None
        if alt == "X":
            return {"data_role": "AUXILIARY_LATENT_ALT", "exclusion_reason": "LATENT_ALT",
                    "is_wt": False, "ref": ref, "alt": None, "edits": edits}
        return {"data_role": "PRIMARY_EXACT_DELTA", "exclusion_reason": None,
                "is_wt": False, "ref": ref, "alt": alt, "edits": edits}
    return {"data_role": None, "exclusion_reason": "MISSING_EVIDENCE",
            "is_wt": False, "ref": None, "alt": None, "edits": edits}


def build_canonical_record(
    *,
    asset_name: str,
    source_accession: str,
    file_sha256: str,
    profile: dict,
    profile_index: int,
    seq: str,
    offset: int,
    seqpos_values: list[int],
    global_annotations: list[dict[str, list[str]]],
) -> dict[str, Any]:
    """Build one v2 canonical record with a per-position eligibility mask."""
    resolved = _resolve_profile_annotations(global_annotations, profile.get("annotation") or {})
    parsed_mutations = _parse_mutations(profile, resolved, offset)
    edited_indices = _edited_site_indices(parsed_mutations, seqpos_values)
    role = _profile_data_role(parsed_mutations)
    is_wt = role["is_wt"]

    # Per-position eligibility (no probe-change / alignment-change at record
    # level; those are pair-level and applied when pairing).
    reactivity = profile.get("reactivity") or []
    eligibility = compute_position_eligibility(
        reactivity, edited_indices, set(), False
    )
    code_counts = Counter(eligibility)

    condition = _condition_tuple(resolved)
    seq_len = len(seq)
    raw = list(reactivity)
    err = profile.get("reactivity_error")
    missing_positions = [
        i for i, v in enumerate(raw) if not (isinstance(v, (int, float)) and math.isfinite(v))
    ]
    return {
        "schema_version": CANONICAL_SCHEMA_V2,
        "source_accession": source_accession,
        "source_asset_name": asset_name,
        "source_file_sha256": file_sha256,
        "source_profile_index": profile_index,
        "ref_allele": role["ref"],
        "alt_allele": role["alt"],
        "mutation_coordinate_system": {
            "offset": offset,
            "source_coordinate_1_based": (
                role["edits"][0].get("source_coordinate_1_based") if role["edits"] else None
            ),
            "sequence_index_0_based": (
                role["edits"][0].get("sequence_index_0_based") if role["edits"] else None
            ),
        },
        "data_role": role["data_role"],
        "exclusion_reason": role["exclusion_reason"],
        "is_wt": is_wt,
        "canonical_sequence": seq,
        "parent_lineage_evidence": {
            "parent_sequence_sha256": sha256_text(seq),
            "construct_sequence_length": seq_len,
            "design_group": source_accession,
        },
        "condition_match_evidence": {
            "status": condition["all_required_known"] and "MATCHED_ALL_REQUIRED" or "MISSING_REQUIRED_FIELD",
            "condition": condition,
        },
        "probe": list(condition.get("modifier") or []),
        "temperature": list(condition.get("temperature") or []),
        "ligand": list(condition.get("chemical") or []),
        "buffer": list(condition.get("chemical") or []),
        "reactivity_layers": {
            "raw": {"reactivity": raw, "error": err, "length": len(raw)},
            "upstream": {"reactivity": raw, "error": err, "length": len(raw)},
            "train_frozen": {"reactivity": raw, "error": err, "length": len(raw)},
            "position_mask": [1 if (isinstance(v, (int, float)) and math.isfinite(v)) else 0 for v in raw],
            "missing_positions": missing_positions,
            "missing_reason": "MASKED_UNMEASURED" if missing_positions else None,
            "eligibility_reason_codes": eligibility,
            "eligibility_code_counts": dict(code_counts),
        },
        "seqpos": seqpos_values,
        "seqpos_count": len(seqpos_values),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset-manifest", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--legacy-records", type=Path, default=None)
    ap.add_argument("--legacy-pairs", type=Path, default=None)
    ap.add_argument("--out-registry", type=Path, required=True)
    ap.add_argument("--out-artifacts", type=Path, required=True)
    args = ap.parse_args()

    out_registry = args.out_registry
    out_artifacts = args.out_artifacts
    if out_registry.exists():
        raise FileExistsError(out_registry)
    if out_artifacts.exists():
        raise FileExistsError(out_artifacts)
    out_registry.mkdir(parents=True)
    out_artifacts.mkdir(parents=True)

    assets: list[dict[str, Any]] = []
    with args.asset_manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                assets.append(json.loads(line))

    canonical_records: list[dict[str, Any]] = []
    primary_pairs: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    per_asset_status: Counter[str] = Counter()
    asset_profile_counts: Counter[str] = Counter()

    # ---- first pass: parse every asset, collect per-file profiles ----
    # We do a two-pass approach: parse each asset once, store parsed profiles in
    # memory (bounded because we only keep per-file data transiently).  Files
    # are small; the full corpus fits.
    file_profiles: dict[str, dict[str, Any]] = {}
    for asset in assets:
        name = asset["asset_name"]
        accession = asset["source_accession"]
        raw_path = args.raw_dir / name
        if not raw_path.is_file():
            dispositions.append({
                "source_accession": accession,
                "asset_name": name,
                "status": "MISSING_FILE",
                "disposition_reason": "MISSING_FILE",
            })
            per_asset_status["MISSING_FILE"] += 1
            continue
        try:
            parsed = parse_rdat_v2(raw_path)
            seq = parsed["headers"].get("SEQUENCE", "")
            offset = 0
            try:
                offset = int(parsed["headers"].get("OFFSET", "0") or "0")
            except (TypeError, ValueError):
                offset = 0
            seqpos_values = parsed["seqpos"]
            profiles = parsed["profiles"]
            # length check per profile; keep only length-consistent profiles
            kept = []
            for p in profiles:
                if len(p["reactivity"]) != len(seqpos_values):
                    dispositions.append({
                        "source_accession": accession,
                        "asset_name": name,
                        "profile_index": p["index"],
                        "status": "LENGTH_MISMATCH",
                        "disposition_reason": "LENGTH_MISMATCH",
                        "seqpos_count": len(seqpos_values),
                        "reactivity_count": len(p["reactivity"]),
                    })
                    per_asset_status["LENGTH_MISMATCH"] += 1
                    continue
                kept.append(p)
            if not kept:
                per_asset_status.setdefault("NO_USABLE_PROFILE", 0)
                continue
            file_profiles[name] = {
                "accession": accession,
                "seq": seq,
                "offset": offset,
                "seqpos_values": seqpos_values,
                "global_annotations": parsed["global_annotations"],
                "profiles": kept,
                "file_sha256": parsed["sha256"],
            }
            per_asset_status["PARSED"] += 1
        except (ValueError, FileNotFoundError) as exc:
            dispositions.append({
                "source_accession": accession,
                "asset_name": name,
                "status": "UNPARSEABLE",
                "disposition_reason": "UNPARSEABLE",
                "error": str(exc)[:300],
            })
            per_asset_status["UNPARSEABLE"] += 1

    # ---- second pass: build canonical records + WT-mutant pairs ----
    for name, info in file_profiles.items():
        accession = info["accession"]
        seq = info["seq"]
        offset = info["offset"]
        seqpos_values = info["seqpos_values"]
        global_annotations = info["global_annotations"]
        file_sha256 = info["file_sha256"]

        # WT anchor pool from this file
        wt_anchors = []
        for p in info["profiles"]:
            resolved = _resolve_profile_annotations(global_annotations, p.get("annotation") or {})
            parsed_mutations = _parse_mutations(p, resolved, offset)
            if not _is_wt_annotation(parsed_mutations):
                continue
            cond = _condition_tuple(resolved)
            wt_anchors.append({
                "profile_index": p["index"],
                "condition": cond,
                "condition_key": (cond["modifier"], cond["temperature"], cond["chemical"], cond["experimentType"]),
                "reactivity": p.get("reactivity") or [],
                "profile_sequence": p.get("profile_sequence"),
            })

        for p in info["profiles"]:
            resolved = _resolve_profile_annotations(global_annotations, p.get("annotation") or {})
            parsed_mutations = _parse_mutations(p, resolved, offset)
            rec = build_canonical_record(
                asset_name=name,
                source_accession=accession,
                file_sha256=file_sha256,
                profile=p,
                profile_index=p["index"],
                seq=seq,
                offset=offset,
                seqpos_values=seqpos_values,
                global_annotations=global_annotations,
            )
            canonical_records.append(rec)

            # Pair primary mutants with a matching WT anchor, applying
            # per-position alignment/probe eligibility.
            if rec["data_role"] == "PRIMARY_EXACT_DELTA":
                cand_cond = rec["condition_match_evidence"]["condition"]
                cand_key = (cand_cond["modifier"], cand_cond["temperature"],
                            cand_cond["chemical"], cand_cond["experimentType"])
                match = next((w for w in wt_anchors if w["condition_key"] == cand_key), None)
                if match is None:
                    rec["condition_match_evidence"]["status"] = "MISSING_REQUIRED_FIELD"
                    rec["data_role"] = None
                    rec["exclusion_reason"] = "CONDITION_MISSING"
                    # recompute eligibility without probe pairing (leave as-is)
                else:
                    mutant_seq = p.get("profile_sequence")
                    wt_seq = match["profile_sequence"]
                    alignment_change = compute_alignment_change_indices(
                        mutant_seq, wt_seq, seqpos_values
                    )
                    probe_change = bool(
                        cand_cond.get("modifier") and match["condition"]["modifier"]
                        and cand_cond["modifier"] != match["condition"]["modifier"]
                    )
                    mutant_seqpos = seqpos_values
                    edited_indices = _edited_site_indices(parsed_mutations, mutant_seqpos)
                    eligibility = compute_position_eligibility(
                        p.get("reactivity") or [],
                        edited_indices,
                        alignment_change,
                        probe_change,
                    )
                    rec["reactivity_layers"]["eligibility_reason_codes"] = eligibility
                    rec["reactivity_layers"]["eligibility_code_counts"] = dict(Counter(eligibility))
                    rec["wt_reuse_group"] = f"{name}:{match['profile_index']}"
                    rec["wt_anchor_profile_index"] = match["profile_index"]
                    rec["wt_anchor_reactivity"] = match["reactivity"]
                    rec["condition_match_evidence"]["status"] = "MATCHED_ALL_REQUIRED"
                    primary_pairs.append({
                        "schema_version": PAIR_SCHEMA_V2,
                        "source_accession": rec["source_accession"],
                        "asset_name": name,
                        "file_sha256": file_sha256,
                        "mutant_profile_index": p["index"],
                        "wt_profile_index": match["profile_index"],
                        "ref_allele": rec["ref_allele"],
                        "alt_allele": rec["alt_allele"],
                        "coordinate": rec["mutation_coordinate_system"],
                        "condition": cand_cond,
                        "condition_match_status": "MATCHED_ALL_REQUIRED",
                        "wt_reuse_group": rec["wt_reuse_group"],
                        "data_role": "PRIMARY_EXACT_DELTA",
                        "exclusion_reason": None,
                        "eligibility_reason_codes": eligibility,
                        "eligibility_code_counts": dict(Counter(eligibility)),
                    })

    # Deterministic ordering
    canonical_records.sort(key=lambda r: (r["source_accession"], r["source_profile_index"], r["source_asset_name"]))
    primary_pairs.sort(key=lambda r: (r["source_accession"], r["mutant_profile_index"], r["asset_name"]))
    dispositions.sort(key=lambda r: (r.get("source_accession", ""), r.get("asset_name", ""), r.get("profile_index", 0)))

    # ---- write outputs (registry: small; artifacts: large) ----
    rec_out = out_artifacts / "canonical_records_v2.jsonl"
    pairs_out = out_artifacts / "primary_pairs_v2.jsonl"
    disp_out = out_artifacts / "asset_disposition_v2.jsonl"
    for rec_out_, rows in ((rec_out, canonical_records), (pairs_out, primary_pairs), (disp_out, dispositions)):
        with rec_out_.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    # ---- crosswalk v1 <-> v2 ----
    legacy_records: list[dict[str, Any]] = []
    if args.legacy_records and args.legacy_records.exists():
        with args.legacy_records.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    legacy_records.append(json.loads(line))
    legacy_pairs: list[dict[str, Any]] = []
    if args.legacy_pairs and args.legacy_pairs.exists():
        with args.legacy_pairs.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    legacy_pairs.append(json.loads(line))

    crosswalk = build_crosswalk(legacy_records, legacy_pairs, canonical_records, primary_pairs)
    crosswalk_out = out_registry / "canonical_v2_vs_v1_crosswalk.json"
    crosswalk_out.write_text(
        json.dumps(crosswalk, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ---- manifest with sha256 ----
    manifest = {
        "schema_version": MANIFEST_SCHEMA_V2,
        "generated_at": "2026-08-07T00:00:00+08:00",
        "outputs": {
            p.name: {"path": str(p), "sha256": sha256_file(p)} for p in
            [rec_out, pairs_out, disp_out, crosswalk_out]
        },
    }
    manifest_out = out_registry / "canonical_v2_manifest.json"
    manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = {
        "schema_version": "reactflow_delta.d1x_v2_summary.v1",
        "frozen_asset_count": len(assets),
        "parsed_file_count": per_asset_status["PARSED"],
        "per_asset_status": dict(per_asset_status),
        "canonical_records": len(canonical_records),
        "primary_pairs": len(primary_pairs),
        "dispositions": len(dispositions),
        "disposition_reason_counts": dict(Counter(d.get("disposition_reason") for d in dispositions)),
        "eligibility_code_counts_all_records": dict(sum(
            (Counter(r["reactivity_layers"]["eligibility_code_counts"]) for r in canonical_records),
            Counter(),
        )),
    }
    (out_registry / "d1x_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_crosswalk(
    legacy_records: list[dict],
    legacy_pairs: list[dict],
    v2_records: list[dict],
    v2_pairs: list[dict],
) -> dict[str, Any]:
    """Map legacy d1x canonical records/pairs to their v2 counterparts."""
    def _key_accession(r: dict) -> str:
        return r.get("source_accession") or ""

    v2_by_key: dict[tuple, dict] = {}
    for r in v2_records:
        v2_by_key[(_key_accession(r), r.get("source_profile_index"), r.get("source_asset_name"))] = r

    record_entries = []
    v1_keys = set()
    for r in legacy_records:
        key = (r.get("source_accession"), r.get("source_profile_index"), r.get("source_asset_name"))
        v1_keys.add(key)
        v2 = v2_by_key.get(key)
        v2_codes = (v2 or {}).get("reactivity_layers", {}).get("eligibility_code_counts") or {}
        rec_entry = {
            "v1_key": [k for k in key],
            "matched_v2": v2 is not None,
            "v1_data_role": r.get("data_role"),
            "v1_exclusion_reason": r.get("exclusion_reason"),
            "v2_data_role": v2.get("data_role") if v2 else None,
            "v2_exclusion_reason": v2.get("exclusion_reason") if v2 else None,
            "v2_eligibility_code_counts": v2_codes,
            "eligibility_change": bool(
                v2 and r.get("exclusion_reason") is None and v2_codes.get("EDITED_SITE", 0)
            ),
        }
        record_entries.append(rec_entry)

    v2_only_keys = set(v2_by_key) - v1_keys
    pair_entries = []
    for pr in legacy_pairs:
        key = (pr.get("source_accession"), pr.get("mutant_profile_index"), pr.get("asset_name"))
        pair_entries.append({
            "v1_key": [k for k in key],
            "has_v2_pair": any(
                p.get("source_accession") == pr.get("source_accession")
                and p.get("mutant_profile_index") == pr.get("mutant_profile_index")
                and p.get("asset_name") == pr.get("asset_name")
                for p in v2_pairs
            ),
        })

    return {
        "schema_version": CROSSWALK_SCHEMA_V2,
        "summary": {
            "v1_canonical_records": len(legacy_records),
            "v2_canonical_records": len(v2_records),
            "v1_records_matched_to_v2": sum(1 for e in record_entries if e["matched_v2"]),
            "v2_records_without_v1_counterpart": len(v2_only_keys),
            "v1_primary_pairs": len(legacy_pairs),
            "v2_primary_pairs": len(v2_pairs),
        },
        "record_crosswalk": record_entries,
        "pair_crosswalk": pair_entries,
    }


if __name__ == "__main__":
    raise SystemExit(main())
