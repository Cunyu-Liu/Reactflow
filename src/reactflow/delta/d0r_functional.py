"""D0-R: functional anchor candidate audit using 124 nt window at offset 31.

This module implements the stricter candidate classification specified in the
D0-R handoff document. Instead of using the SEQPOS window (139 positions), it
uses a 124 nt functional anchor extracted from COVSL5_NOM_0002, located at
offset 31 within the 206 nt full anchor from SL5CV2_NOM_0002.

Key differences from the SEQPOS-based classify_profile_edit:
  * Functional window is [offset, offset+length) in the full sequence.
  * Candidate requires name-encoded mutation to EXACTLY MATCH the functional
    edit (position relative to window, ref, alt — with DNA->RNA T->U conversion).
  * Status label is ``candidate_only_pending_parent_lineage_and_functional_region_validation``.
  * ``true_pair`` is always ``False``.
"""

from __future__ import annotations

from typing import Any

from .rdat import parse_mutations_from_name


def dna_to_rna(base: str) -> str:
    """Convert a DNA base to RNA (T -> U). Other bases are unchanged."""
    return "U" if base == "T" else base


def compute_functional_edits(
    mutant_seq: str | None,
    wt_seq: str | None,
    functional_offset: int,
    functional_length: int,
) -> dict[str, Any]:
    """Compute edits within the functional window and full sequence.

    Parameters
    ----------
    mutant_seq, wt_seq:
        Full-length profile sequences (expected same length, e.g. 206 nt).
    functional_offset:
        0-indexed start of the functional window in the full sequence (e.g. 31).
    functional_length:
        Length of the functional window (e.g. 124).

    Returns
    -------
    dict with ``status``, ``full_hamming``, ``functional_hamming``,
    ``outside_functional_region_difference_count``, ``functional_edits``,
    ``outside_functional_edits``.
    """
    if not mutant_seq or not wt_seq:
        return {
            "status": "skipped",
            "reason": "missing_mutant_or_wt_sequence",
            "full_hamming": 0,
            "functional_hamming": 0,
            "outside_functional_region_difference_count": 0,
            "functional_edits": [],
            "outside_functional_edits": [],
        }
    if len(mutant_seq) != len(wt_seq):
        return {
            "status": "skipped",
            "reason": "length_mismatch",
            "mutant_length": len(mutant_seq),
            "wt_length": len(wt_seq),
            "full_hamming": 0,
            "functional_hamming": 0,
            "outside_functional_region_difference_count": 0,
            "functional_edits": [],
            "outside_functional_edits": [],
        }

    full_edits: list[dict[str, Any]] = []
    for i, (m, w) in enumerate(zip(mutant_seq, wt_seq)):
        if m != w:
            full_edits.append({
                "position_0indexed": i,
                "wt_base": w,
                "mutant_base": m,
            })

    func_start = functional_offset
    func_end = functional_offset + functional_length
    functional_edits: list[dict[str, Any]] = []
    outside_edits: list[dict[str, Any]] = []
    for edit in full_edits:
        pos = edit["position_0indexed"]
        if func_start <= pos < func_end:
            fe = dict(edit)
            fe["functional_position_0indexed"] = pos - func_start
            functional_edits.append(fe)
        else:
            outside_edits.append(dict(edit))

    return {
        "status": "computed",
        "full_hamming": len(full_edits),
        "functional_hamming": len(functional_edits),
        "outside_functional_region_difference_count": len(outside_edits),
        "functional_edits": functional_edits,
        "outside_functional_edits": outside_edits,
    }


def classify_functional_candidate(
    profile: dict[str, Any],
    wt_profile: dict[str, Any],
    functional_offset: int,
    functional_length: int,
) -> dict[str, Any]:
    """Classify a profile against the WT anchor using functional-window matching.

    Strict candidate criteria (ALL must hold):
      1. Exactly one name-encoded mutation.
      2. Functional Hamming distance == 1.
      3. The single functional edit matches the name-encoded mutation:
         position (relative to functional window), ref (DNA->RNA), alt (DNA->RNA).

    Excluded profiles receive ``lineage_status = "excluded"`` with a reason.
    Candidates receive ``lineage_status =
    "candidate_only_pending_parent_lineage_and_functional_region_validation"``
    and ``true_pair = False``.
    """
    name = profile.get("profile_name")
    name_mutations = parse_mutations_from_name(name)
    edit_info = compute_functional_edits(
        profile.get("profile_sequence"),
        wt_profile.get("profile_sequence"),
        functional_offset,
        functional_length,
    )

    # Extract declarative tokens from name (non-coordinate, non-mutation)
    declarative_tokens: list[str] = []
    if name:
        for token in ("5pad6", "w53barcode"):
            if token in name:
                declarative_tokens.append(token)

    result: dict[str, Any] = {
        "profile_index": profile["index"],
        "profile_name": name,
        "name_encoded_mutations": name_mutations,
        "name_encoded_mutation_count": len(name_mutations),
        "edit_info": edit_info,
        "declarative_tokens": declarative_tokens,
        "wt_profile_index": wt_profile["index"],
        "wt_profile_name": wt_profile.get("profile_name"),
        "full_hamming": edit_info["full_hamming"],
        "functional_hamming": edit_info["functional_hamming"],
        "outside_functional_region_difference_count": edit_info[
            "outside_functional_region_difference_count"
        ],
    }

    if edit_info["status"] == "skipped":
        result["classification"] = "excluded"
        result["exclusion_reason"] = edit_info["reason"]
        result["lineage_status"] = "excluded"
        result["true_pair"] = False
        return result

    mut_count = len(name_mutations)
    if mut_count == 0:
        result["classification"] = "excluded"
        result["exclusion_reason"] = "no_name_mutation_encoding"
        result["lineage_status"] = "excluded"
        result["true_pair"] = False
        return result

    if mut_count > 1:
        result["classification"] = "excluded"
        result["exclusion_reason"] = "multiple_name_mutation_encodings"
        result["lineage_status"] = "excluded"
        result["true_pair"] = False
        return result

    if edit_info["functional_hamming"] != 1:
        result["classification"] = "excluded"
        result["exclusion_reason"] = (
            "functional_hamming_not_1_got_" + str(edit_info["functional_hamming"])
        )
        result["lineage_status"] = "excluded"
        result["true_pair"] = False
        return result

    # Exactly 1 name mutation AND 1 functional edit: verify pos/ref/alt match
    name_mut = name_mutations[0]
    func_edit = edit_info["functional_edits"][0]

    name_pos = name_mut["position"]
    actual_pos = func_edit["functional_position_0indexed"]
    name_ref_rna = dna_to_rna(name_mut["ref"])
    name_alt_rna = dna_to_rna(name_mut["mut"])
    actual_ref = func_edit["wt_base"]
    actual_alt = func_edit["mutant_base"]

    pos_match = name_pos == actual_pos
    ref_match = name_ref_rna == actual_ref
    alt_match = name_alt_rna == actual_alt

    if pos_match and ref_match and alt_match:
        result["classification"] = "candidate_single_functional_anchor"
        result["lineage_status"] = (
            "candidate_only_pending_parent_lineage_and_functional_region_validation"
        )
        result["true_pair"] = False
        result["matched_mutation"] = {
            "name_position": name_pos,
            "name_ref_rna": name_ref_rna,
            "name_alt_rna": name_alt_rna,
            "actual_position_in_full": func_edit["position_0indexed"],
            "actual_functional_position": actual_pos,
            "actual_ref": actual_ref,
            "actual_alt": actual_alt,
        }
    else:
        mismatches: list[str] = []
        if not pos_match:
            mismatches.append(f"position_name_{name_pos}_actual_{actual_pos}")
        if not ref_match:
            mismatches.append(f"ref_name_{name_ref_rna}_actual_{actual_ref}")
        if not alt_match:
            mismatches.append(f"alt_name_{name_alt_rna}_actual_{actual_alt}")
        result["classification"] = "excluded"
        result["exclusion_reason"] = "name_sequence_mismatch_" + "_".join(mismatches)
        result["lineage_status"] = "excluded"
        result["true_pair"] = False

    return result


def find_wt_anchor_by_sequence(
    profiles: list[dict[str, Any]],
    full_anchor_seq: str,
) -> dict[str, Any] | None:
    """Find the WT anchor profile by exact full-sequence match.

    A profile is WT if its ``profile_sequence`` exactly equals
    ``full_anchor_seq`` AND its name has no mutation encoding.

    Returns ``None`` if no such profile is found, or if multiple are found
    (ambiguous WT).
    """
    matches: list[dict[str, Any]] = []
    for profile in profiles:
        seq = profile.get("profile_sequence")
        if not seq or seq != full_anchor_seq:
            continue
        name = profile.get("profile_name")
        if name and parse_mutations_from_name(name):
            continue  # has mutation code, not WT
        matches.append(profile)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Ambiguous: multiple exact-match no-mutation profiles
        return None
    return None


def verify_functional_anchor(
    full_anchor: str,
    functional_anchor: str,
    expected_offset: int = 31,
) -> dict[str, Any]:
    """Verify that functional_anchor appears exactly once in full_anchor.

    Returns a dict with:
      ``valid``: bool
      ``offset``: the 0-indexed offset (or None if not found)
      ``occurrences``: list of all offsets
      ``reason``: failure reason if invalid
    """
    import re

    occurrences = [
        m.start() for m in re.finditer(re.escape(functional_anchor), full_anchor)
    ]
    if len(occurrences) == 0:
        return {
            "valid": False,
            "offset": None,
            "occurrences": [],
            "reason": "functional_anchor_not_found_in_full_anchor",
        }
    if len(occurrences) > 1:
        return {
            "valid": False,
            "offset": None,
            "occurrences": occurrences,
            "reason": "functional_anchor_not_unique_in_full_anchor",
        }
    offset = occurrences[0]
    if offset != expected_offset:
        return {
            "valid": False,
            "offset": offset,
            "occurrences": occurrences,
            "reason": f"offset_{offset}_expected_{expected_offset}",
        }
    return {
        "valid": True,
        "offset": offset,
        "occurrences": occurrences,
        "reason": None,
    }
