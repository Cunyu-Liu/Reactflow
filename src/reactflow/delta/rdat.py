"""Fail-closed RDAT parser for D0/D0-R provenance and construct audit.

D0-R extensions (backward-compatible):
  * Per-profile indexed ``SEQUENCE:N`` lines (in addition to the global header).
  * Per-profile ``sequence:`` annotation token (M2-seq style, e.g. M2SL5).
  * Mutation encoding parsed from ``name:`` annotation tokens.
  * WT anchor identification (``mutation:WT`` annotation or name without a
    mutation suffix).
  * Edit-set computation from per-profile sequence vs WT anchor, restricted to
    the SEQPOS window when available.
  * Functional-RNA vs adapter/barcode separation: edits inside the SEQPOS window
    that are NOT explained by the name-encoded mutation are reported as
    ``unexplained_edits`` (likely barcode/adapter), not auto-rejected.

D1 parser extensions (v3.1 §5, forward-only):
  * §5.1: ``VERSION`` header accepted as alias for ``RDAT_VERSION`` (TRP4P6).
  * §5.2: ``RDAT_VERSION`` 0.4 / 0.22 / 0.24 accepted in addition to 0.34
    (BSUGLY, CBAG4P, GLYCFN).
  * §5.3: Space-separated field handling for legacy files that use spaces
    instead of tabs (GLYCFN_KNK_0001/0002). When a line has no tab character,
    fields are split on whitespace; single-value headers (NAME, SEQUENCE, etc.)
    are re-joined with a single space.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .manifests import sha256_file


RDAT_CONSTRUCT_PARSE_MANIFEST_SCHEMA_VERSION = "reactflow-delta-rdat-construct-parse-manifest-v1"
D0R_PARSED_PROFILE_SCHEMA_VERSION = "reactflow-delta-d0r-parsed-profile-v1"

# D1 §5.2: accepted RDAT versions. 0.34 is the D0 canonical version; 0.4, 0.22,
# and 0.24 are forward-only additions for BSUGLY, CBAG4P, and GLYCFN files.
_ACCEPTED_RDAT_VERSIONS = frozenset({"0.34", "0.4", "0.22", "0.24"})

_RNA_BASES = set("ACGU")
_RNA_SEQUENCE = re.compile(r"^[ACGU]+$")
# Mutation tokens in name annotations: <pos><ref>-<mut>  e.g. 0G-A, 78C-T
# Accept both RNA (ACGU) and DNA (ACGT) bases since construct names often use
# DNA encoding (oligos are ordered as DNA) even though sequences are RNA.
_NAME_MUTATION = re.compile(r"(?<![A-Za-z])(\d+)([ACGUT])-([ACGUT])(?![A-Za-z])")
# WT marker in name (no mutation suffix)
_WT_NAME_HINTS = ("_wt", "-wt", "wildtype", "wild-type")


class RdatParseError(ValueError):
    """Raised when an RDAT record cannot be structurally audited."""


def parse_rdat(path: str | Path) -> dict[str, Any]:
    """Parse RDAT text without imputing missing values or experiment labels."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    headers: dict[str, str] = {}
    comments: list[str] = []
    global_annotations: list[dict[str, list[str]]] = []
    data_annotations: dict[int, dict[str, list[str]]] = {}
    seqpos: list[str] | None = None
    reactivity: dict[int, list[float | None]] = {}
    reactivity_error: dict[int, list[float | None]] = {}
    profile_sequences: dict[int, str] = {}  # D0-R: SEQUENCE:N per-profile lines

    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw_line.strip():
            continue
        # D1 §5.3: GLYCFN and other legacy files use spaces instead of tabs.
        # When no tab is present, split on whitespace. Single-value headers
        # (NAME, SEQUENCE, etc.) are re-joined downstream.
        if "\t" in raw_line:
            fields = raw_line.split("\t")
        else:
            fields = raw_line.split()
        key = fields[0].strip()
        values = [value.strip() for value in fields[1:]]
        if key == "COMMENT":
            comments.append("\t".join(values).strip())
        elif key == "ANNOTATION":
            global_annotations.append(_annotation_map(values))
        elif key.startswith("ANNOTATION_DATA:"):
            index = _parse_index(key, line_number)
            if index in data_annotations:
                raise RdatParseError(f"duplicate ANNOTATION_DATA index {index}")
            data_annotations[index] = _annotation_map(values)
        elif key == "SEQPOS":
            if seqpos is not None:
                raise RdatParseError("duplicate SEQPOS")
            if not values:
                raise RdatParseError("SEQPOS is empty")
            seqpos = values
        elif key.startswith("REACTIVITY_ERROR:"):
            index = _parse_index(key, line_number)
            if index in reactivity_error:
                raise RdatParseError(f"duplicate REACTIVITY_ERROR index {index}")
            reactivity_error[index] = _numeric_values(values, key)
        elif key.startswith("REACTIVITY:"):
            index = _parse_index(key, line_number)
            if index in reactivity:
                raise RdatParseError(f"duplicate REACTIVITY index {index}")
            reactivity[index] = _numeric_values(values, key)
        elif key in {"RDAT_VERSION", "VERSION", "NAME", "SEQUENCE", "STRUCTURE", "OFFSET"}:
            if key in headers:
                raise RdatParseError(f"duplicate header {key}")
            if not values:
                raise RdatParseError(f"header {key} requires one non-empty value")
            # D1 §5.3: space-separated files may yield multiple tokens for NAME
            # (e.g., "NAME glycine riboswitch, F. nucleatum"); re-join them.
            header_value = values[0] if len(values) == 1 else " ".join(values)
            if not header_value:
                raise RdatParseError(f"header {key} requires one non-empty value")
            headers[key] = header_value
        elif key.startswith("SEQUENCE:"):
            # D0-R: per-profile indexed sequence line SEQUENCE:N
            index = _parse_index(key, line_number)
            if index in profile_sequences:
                raise RdatParseError(f"duplicate SEQUENCE index {index}")
            if len(values) != 1 or not values[0]:
                raise RdatParseError(f"SEQUENCE:{index} requires one non-empty value")
            profile_sequences[index] = values[0]
        elif key.startswith(("TRACE:", "READS:")):
            continue
        else:
            headers.setdefault(f"unknown:{key}", "\t".join(values))

    # D1 §5.1: VERSION is an alias for RDAT_VERSION (TRP4P6 files use VERSION
    # instead of RDAT_VERSION). If both are present they must agree.
    if "VERSION" in headers and "RDAT_VERSION" not in headers:
        headers["RDAT_VERSION"] = headers["VERSION"]
    elif "VERSION" in headers and "RDAT_VERSION" in headers:
        if headers["VERSION"] != headers["RDAT_VERSION"]:
            raise RdatParseError(
                f"conflicting VERSION ({headers['VERSION']!r}) and "
                f"RDAT_VERSION ({headers['RDAT_VERSION']!r})"
            )

    # D1 §5.2: accept RDAT_VERSION 0.34 (D0 canonical), 0.4, 0.22, 0.24.
    if headers.get("RDAT_VERSION") not in _ACCEPTED_RDAT_VERSIONS:
        raise RdatParseError(
            f"RDAT_VERSION {headers.get('RDAT_VERSION')!r} is not accepted; "
            f"accepted versions: {sorted(_ACCEPTED_RDAT_VERSIONS)}"
        )
    for required in ("NAME", "SEQUENCE", "OFFSET"):
        if required not in headers:
            raise RdatParseError(f"missing required header {required}")
    if seqpos is None:
        raise RdatParseError("missing SEQPOS")
    if not reactivity:
        raise RdatParseError("missing REACTIVITY rows")

    profiles = []
    for index in sorted(reactivity):
        values = reactivity[index]
        if len(values) != len(seqpos):
            raise RdatParseError(f"REACTIVITY:{index} length does not match SEQPOS")
        errors = reactivity_error.get(index)
        if errors is not None and len(errors) != len(seqpos):
            raise RdatParseError(f"REACTIVITY_ERROR:{index} length does not match SEQPOS")
        annotation = data_annotations.get(index, {})
        # D0-R: prefer SEQUENCE:N line, fall back to sequence: annotation token
        profile_sequence = profile_sequences.get(index)
        sequence_source = "sequence_indexed_line" if index in profile_sequences else None
        if profile_sequence is None:
            seq_values = annotation.get("sequence", [])
            if len(seq_values) == 1 and _RNA_SEQUENCE.fullmatch(seq_values[0]):
                profile_sequence = seq_values[0]
                sequence_source = "annotation_sequence_token"
        name_values = annotation.get("name", [])
        profile_name = name_values[0] if len(name_values) == 1 else None
        profiles.append(
            {
                "index": index,
                "annotation": annotation,
                "reactivity": values,
                "reactivity_error": errors,
                "missing_reactivity_count": sum(value is None for value in values),
                # D0-R additions
                "profile_sequence": profile_sequence,
                "profile_sequence_source": sequence_source,
                "profile_name": profile_name,
            }
        )
    orphan_annotation_indices = sorted(set(data_annotations) - set(reactivity))
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "headers": headers,
        "comments": comments,
        "global_annotations": global_annotations,
        "seqpos": seqpos,
        "profiles": profiles,
        "orphan_annotation_indices": orphan_annotation_indices,
    }


# ---------------------------------------------------------------------------
# D0-R: mutation encoding, WT anchor, edit-set computation
# ---------------------------------------------------------------------------


def parse_mutations_from_name(name: str | None) -> list[dict[str, Any]]:
    """Parse mutation encoding from a profile ``name:`` annotation token.

    Supports single and multi-mutation encodings like:
      ``SL5_SARS_CoV_2_0G-A_5pad6_w53barcode`` -> [{position:0, ref:G, mut:A}]
      ``SL5_MERS_GCadded_78C-T_86G-C_0pad0_w53barcode`` -> [{78,C,T},{86,G,C}]

    Positions are 0-indexed relative to the functional RNA domain (the meaning
    of "position 0" depends on the construct design, e.g. after 5' padding).
    """

    if not name:
        return []
    matches = _NAME_MUTATION.findall(name)
    return [
        {"position": int(pos), "ref": ref, "mut": mut, "encoding": f"{pos}{ref}-{mut}"}
        for pos, ref, mut in matches
    ]


def is_wt_profile(profile: dict[str, Any]) -> bool:
    """Identify a WT anchor profile.

    A profile is WT if it has an explicit ``mutation:WT`` annotation, OR if its
    name lacks a mutation-encoding suffix (no ``<pos><ref>-<mut>`` tokens).
    """

    annotation = profile.get("annotation") or {}
    mutation_values = annotation.get("mutation", [])
    if mutation_values:
        return any(v.strip().upper() == "WT" for v in mutation_values)
    name = profile.get("profile_name")
    if name:
        lower = name.lower()
        if any(hint in lower for hint in _WT_NAME_HINTS):
            return True
        return not parse_mutations_from_name(name)
    # no annotation and no name: cannot establish WT
    return False


def seqpos_to_indices(seqpos: list[str]) -> list[int]:
    """Extract integer positions from SEQPOS tokens like ``X27``, ``A1``.

    Returns an empty list if any token is non-integer after stripping a leading
    single-letter nucleotide prefix. Positions are 1-indexed (RDAT convention).
    """

    indices: list[int] = []
    for token in seqpos:
        digits = re.sub(r"^[ACGUNX]", "", token)
        if digits.isdigit():
            indices.append(int(digits))
    return indices


def compute_edit_set(
    mutant_seq: str | None,
    wt_seq: str | None,
    seqpos_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Compute the edit set between a mutant and WT profile sequence.

    If ``seqpos_indices`` is provided, edits are partitioned into
    ``functional_edits`` (positions within the SEQPOS window) and
    ``flanking_edits`` (positions outside, likely adapter/barcode).

    Returns ``edit_count`` = total edits, ``functional_edit_count``, and the
    edit lists. Returns ``status`` = ``"skipped"`` if either sequence is missing
    or lengths differ.
    """

    if not mutant_seq or not wt_seq:
        return {
            "status": "skipped",
            "reason": "missing_mutant_or_wt_sequence",
            "edits": [],
            "edit_count": 0,
            "functional_edits": [],
            "functional_edit_count": 0,
            "flanking_edits": [],
            "flanking_edit_count": 0,
        }
    if len(mutant_seq) != len(wt_seq):
        return {
            "status": "skipped",
            "reason": "length_mismatch",
            "mutant_length": len(mutant_seq),
            "wt_length": len(wt_seq),
            "edits": [],
            "edit_count": 0,
            "functional_edits": [],
            "functional_edit_count": 0,
            "flanking_edits": [],
            "flanking_edit_count": 0,
        }
    edits: list[dict[str, Any]] = []
    for i, (m, w) in enumerate(zip(mutant_seq, wt_seq)):
        if m != w:
            edits.append({"position_1indexed": i + 1, "wt_base": w, "mutant_base": m})
    functional_set = set(seqpos_indices) if seqpos_indices else set()
    functional_edits = [e for e in edits if e["position_1indexed"] in functional_set] if functional_set else list(edits)
    flanking_edits = [e for e in edits if e["position_1indexed"] not in functional_set] if functional_set else []
    return {
        "status": "computed",
        "edits": edits,
        "edit_count": len(edits),
        "functional_edits": functional_edits,
        "functional_edit_count": len(functional_edits),
        "flanking_edits": flanking_edits,
        "flanking_edit_count": len(flanking_edits),
    }


def classify_profile_edit(
    profile: dict[str, Any],
    wt_profile: dict[str, Any],
    seqpos_indices: list[int],
) -> dict[str, Any]:
    """Classify a profile's edit relationship to the WT anchor.

    Combines the name-encoded mutation with the computed edit set to produce a
    candidate classification. Does NOT confirm lineage — that requires D1.
    """

    name_mutations = parse_mutations_from_name(profile.get("profile_name"))
    edit_info = compute_edit_set(
        profile.get("profile_sequence"),
        wt_profile.get("profile_sequence"),
        seqpos_indices,
    )
    # Determine if the name-encoded mutation count matches the functional edits
    name_mutation_count = len(name_mutations)
    functional_edit_count = edit_info["functional_edit_count"]

    if name_mutation_count == 0 and functional_edit_count == 0:
        edit_class = "no_edit_vs_wt"
    elif name_mutation_count == 1 and functional_edit_count == 1:
        # Name says single mutation AND exactly 1 functional edit matches.
        edit_class = "candidate_single_from_name"
    elif name_mutation_count == 1 and functional_edit_count > 1:
        # Name says single mutation but many functional edits -> cross-parent.
        edit_class = "name_sequence_mismatch_likely_cross_parent"
    elif name_mutation_count > 1 and functional_edit_count == name_mutation_count:
        edit_class = "candidate_multi_from_name"
    elif name_mutation_count > 1 and functional_edit_count > name_mutation_count:
        edit_class = "name_sequence_mismatch_likely_cross_parent"
    elif name_mutation_count == 0 and functional_edit_count == 1:
        edit_class = "candidate_single_from_sequence_only"
    elif name_mutation_count == 0 and functional_edit_count > 1:
        edit_class = "multi_edit_no_name_encoding"
    else:
        edit_class = "name_sequence_mismatch"

    return {
        "profile_index": profile["index"],
        "profile_name": profile.get("profile_name"),
        "name_encoded_mutations": name_mutations,
        "name_encoded_mutation_count": name_mutation_count,
        "edit_set": edit_info,
        "edit_class": edit_class,
        "wt_profile_index": wt_profile["index"],
        "wt_profile_name": wt_profile.get("profile_name"),
        "lineage_status": "candidate_only_unverified",
    }


def find_wt_anchor(profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find the WT anchor profile among parsed profiles.

    Preference order:
      1. Explicit ``mutation:WT`` annotation.
      2. Name without mutation suffix (and not containing WT hints).
      3. First profile with a valid per-profile sequence.

    Returns ``None`` if no candidate is found.
    """

    for profile in profiles:
        annotation = profile.get("annotation") or {}
        mutation_values = annotation.get("mutation", [])
        if any(v.strip().upper() == "WT" for v in mutation_values):
            if profile.get("profile_sequence"):
                return profile
    for profile in profiles:
        name = profile.get("profile_name")
        if name and profile.get("profile_sequence") and not parse_mutations_from_name(name):
            return profile
    for profile in profiles:
        if profile.get("profile_sequence"):
            return profile
    return None


# ---------------------------------------------------------------------------
# Original D0 helpers (unchanged for backward compatibility)
# ---------------------------------------------------------------------------


def _parse_index(key: str, line_number: int) -> int:
    try:
        index = int(key.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RdatParseError(f"invalid indexed key at line {line_number}: {key}") from exc
    if index < 1:
        raise RdatParseError(f"indexed key must be positive: {key}")
    return index


def _annotation_map(values: list[str]) -> dict[str, list[str]]:
    """Preserve every repeated RDAT annotation value in source order."""

    parsed: dict[str, list[str]] = {}
    for token in values:
        if not token:
            continue
        if ":" not in token:
            raise RdatParseError(f"annotation token lacks colon: {token!r}")
        key, value = token.split(":", 1)
        if not key or not value:
            raise RdatParseError(f"annotation token has empty key or value: {token!r}")
        parsed.setdefault(key, []).append(value)
    return parsed


def _numeric_values(values: list[str], key: str) -> list[float | None]:
    result: list[float | None] = []
    for value in values:
        if value.lower() == "nan":
            result.append(None)
            continue
        try:
            number = float(value)
        except ValueError as exc:
            raise RdatParseError(f"non-numeric value in {key}: {value!r}") from exc
        if not math.isfinite(number):
            raise RdatParseError(f"non-finite non-NaN value in {key}: {value!r}")
        result.append(number)
    return result


def build_rdat_construct_parse_manifest(fixture_manifest_path: str | Path) -> dict[str, Any]:
    """Create a construct-level audit view while retaining reactivity only in raw RDAT.

    Each fixture must remain byte-identical to the checked-in fixture manifest.
    Profile annotations are retained exactly, including repeated annotation values;
    numeric reactivities are deliberately not copied or imputed in this D0 artifact.
    """

    path = Path(fixture_manifest_path)
    with path.open(encoding="utf-8") as handle:
        fixture_manifest = json.load(handle)
    if not isinstance(fixture_manifest, dict):
        raise RdatParseError("fixture manifest must be a JSON object")
    if fixture_manifest.get("schema_version") != "reactflow-delta-rdat-fixture-manifest-v1":
        raise RdatParseError("unexpected fixture manifest schema version")
    fixtures = fixture_manifest.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise RdatParseError("fixture manifest must contain a non-empty fixtures list")

    parsed_fixtures = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise RdatParseError("fixture manifest contains a non-object fixture")
        for required in ("name", "path", "sha256", "candidate_category", "status"):
            if not isinstance(fixture.get(required), str) or not fixture[required]:
                raise RdatParseError(f"fixture lacks non-empty {required}")
        if fixture["status"] != "verified_against_release_index":
            raise RdatParseError(f"fixture is not byte-verified: {fixture['name']}")
        document = parse_rdat(fixture["path"])
        if document["sha256"] != fixture["sha256"]:
            raise RdatParseError(f"fixture checksum no longer matches manifest: {fixture['name']}")
        parsed_fixtures.append(
            {
                "name": fixture["name"],
                "path": fixture["path"],
                "sha256": fixture["sha256"],
                "candidate_category": fixture["candidate_category"],
                "rdat_version": document["headers"]["RDAT_VERSION"],
                "rdat_name": document["headers"]["NAME"],
                "sequence_length": len(document["headers"]["SEQUENCE"]),
                "seqpos_count": len(document["seqpos"]),
                "global_annotations": document["global_annotations"],
                "profiles": [
                    {
                        "index": profile["index"],
                        "annotation": profile["annotation"],
                        "missing_reactivity_count": profile["missing_reactivity_count"],
                        "reactivity_error_present": profile["reactivity_error"] is not None,
                    }
                    for profile in document["profiles"]
                ],
                "orphan_annotation_indices": document["orphan_annotation_indices"],
            }
        )

    return {
        "schema_version": RDAT_CONSTRUCT_PARSE_MANIFEST_SCHEMA_VERSION,
        "stage": "D0",
        "input_fixture_manifest": {"path": str(path.resolve()), "sha256": sha256_file(path)},
        "fixtures": parsed_fixtures,
        "fixture_count": len(parsed_fixtures),
        "scientific_boundary": (
            "This is a byte-verified RDAT structural parse only. Candidate categories remain unconfirmed, "
            "numeric reactivities remain only in immutable raw RDAT, and no construct, pair, tier, or model claim is made."
        ),
    }
