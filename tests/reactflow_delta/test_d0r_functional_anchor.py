"""D0-R tests: functional-anchor candidate audit (124 nt window at offset 31)."""

from __future__ import annotations

from reactflow.delta.d0r_functional import (
    classify_functional_candidate,
    compute_functional_edits,
    dna_to_rna,
    find_wt_anchor_by_sequence,
    verify_functional_anchor,
)


# ---------------------------------------------------------------------------
# dna_to_rna
# ---------------------------------------------------------------------------


def test_dna_to_rna_converts_thymine_to_uracil() -> None:
    assert dna_to_rna("T") == "U"


def test_dna_to_rna_leaves_other_bases_unchanged() -> None:
    for base in ("A", "C", "G", "U", "N"):
        assert dna_to_rna(base) == base


# ---------------------------------------------------------------------------
# compute_functional_edits
# ---------------------------------------------------------------------------


def test_compute_functional_edits_full_and_functional_and_outside() -> None:
    # Full length 8, functional window [2, 6) -> positions 2,3,4,5
    wt = "AAGCUAAA"
    mutant = "AAGAUACA"  # pos3 C->A (functional), pos6 A->C (outside)
    info = compute_functional_edits(mutant, wt, functional_offset=2, functional_length=4)
    assert info["status"] == "computed"
    assert info["full_hamming"] == 2
    assert info["functional_hamming"] == 1
    assert info["outside_functional_region_difference_count"] == 1
    assert len(info["functional_edits"]) == 1
    assert info["functional_edits"][0]["position_0indexed"] == 3
    assert info["functional_edits"][0]["functional_position_0indexed"] == 1
    assert info["functional_edits"][0]["wt_base"] == "C"
    assert info["functional_edits"][0]["mutant_base"] == "A"
    assert info["outside_functional_edits"][0]["position_0indexed"] == 6


def test_compute_functional_edits_no_differences() -> None:
    wt = "AAGCUAAA"
    info = compute_functional_edits(wt, wt, functional_offset=2, functional_length=4)
    assert info["status"] == "computed"
    assert info["full_hamming"] == 0
    assert info["functional_hamming"] == 0
    assert info["outside_functional_region_difference_count"] == 0
    assert info["functional_edits"] == []
    assert info["outside_functional_edits"] == []


def test_compute_functional_edits_length_mismatch() -> None:
    info = compute_functional_edits("ACGU", "ACGUAC", functional_offset=0, functional_length=4)
    assert info["status"] == "skipped"
    assert info["reason"] == "length_mismatch"
    assert info["mutant_length"] == 4
    assert info["wt_length"] == 6
    assert info["full_hamming"] == 0
    assert info["functional_edits"] == []


def test_compute_functional_edits_missing_mutant_sequence() -> None:
    info = compute_functional_edits(None, "ACGU", functional_offset=0, functional_length=4)
    assert info["status"] == "skipped"
    assert info["reason"] == "missing_mutant_or_wt_sequence"
    assert info["functional_hamming"] == 0


def test_compute_functional_edits_missing_wt_sequence() -> None:
    info = compute_functional_edits("ACGU", "", functional_offset=0, functional_length=4)
    assert info["status"] == "skipped"
    assert info["reason"] == "missing_mutant_or_wt_sequence"


def test_compute_functional_edits_outside_only() -> None:
    # Differences only outside the functional window
    wt = "AAGCUAAA"
    mutant = "CAGCUAAC"  # pos0 A->C (outside), pos7 A->C (outside)
    info = compute_functional_edits(mutant, wt, functional_offset=2, functional_length=4)
    assert info["full_hamming"] == 2
    assert info["functional_hamming"] == 0
    assert info["outside_functional_region_difference_count"] == 2


# ---------------------------------------------------------------------------
# classify_functional_candidate
# ---------------------------------------------------------------------------


def _wt_profile(seq: str = "AAGCUAAA") -> dict:
    return {"index": 1, "profile_name": "SL5_wt", "profile_sequence": seq}


def test_classify_candidate_single_match() -> None:
    # functional window [2,6); name encodes func pos 1 G->A (DNA ref G -> RNA G)
    # wt[3]=C though, so use a name matching the actual edit.
    wt = "AAGCUAAA"
    mutant = "AAGAUAAA"  # pos3 C->A => functional pos 1, ref C, alt A
    profile = {
        "index": 2,
        "profile_name": "SL5_1C-A_5pad6_w53barcode",
        "profile_sequence": mutant,
    }
    result = classify_functional_candidate(profile, _wt_profile(wt), 2, 4)
    assert result["classification"] == "candidate_single_functional_anchor"
    assert result["true_pair"] is False
    assert result["functional_hamming"] == 1
    assert result["name_encoded_mutation_count"] == 1
    assert "candidate_only_pending_parent_lineage" in result["lineage_status"]
    assert result["matched_mutation"]["name_position"] == 1
    assert result["matched_mutation"]["actual_functional_position"] == 1
    assert result["matched_mutation"]["actual_ref"] == "C"
    assert result["matched_mutation"]["actual_alt"] == "A"


def test_classify_candidate_dna_thymine_to_rna_uracil() -> None:
    # Name encodes DNA ref/alt with T; sequence is RNA with U. dna_to_rna converts.
    wt = "AAGCUAAA"
    mutant = "AAUCUAAA"  # pos2 G->U => functional pos 0, ref G, alt U
    profile = {
        "index": 3,
        "profile_name": "SL5_0G-T_5pad6",  # DNA alt T -> RNA U
        "profile_sequence": mutant,
    }
    result = classify_functional_candidate(profile, _wt_profile(wt), 2, 4)
    assert result["classification"] == "candidate_single_functional_anchor"
    assert result["matched_mutation"]["name_alt_rna"] == "U"
    assert result["matched_mutation"]["actual_alt"] == "U"


def test_classify_excluded_no_name_mutation() -> None:
    # WT name, no mutation encoding, but sequence differs -> functional_hamming!=1 path
    # Actually 0 name mutations hits no_name_mutation_encoding first.
    wt = "AAGCUAAA"
    mutant = "AAGAUAAA"
    profile = {"index": 4, "profile_name": "SL5_wt_5pad6", "profile_sequence": mutant}
    result = classify_functional_candidate(profile, _wt_profile(wt), 2, 4)
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"] == "no_name_mutation_encoding"
    assert result["true_pair"] is False


def test_classify_excluded_multiple_name_mutations() -> None:
    wt = "AAGCUAAA"
    mutant = "AAGAUAAA"  # single edit but name claims two mutations
    profile = {
        "index": 5,
        "profile_name": "SL5_0G-A_1C-A_5pad6",
        "profile_sequence": mutant,
    }
    result = classify_functional_candidate(profile, _wt_profile(wt), 2, 4)
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"] == "multiple_name_mutation_encodings"
    assert result["name_encoded_mutation_count"] == 2


def test_classify_excluded_functional_hamming_not_1() -> None:
    # Name says single mutation, but sequence has 0 functional edits (edit is outside)
    wt = "AAGCUAAA"
    mutant = "CAGCUAAA"  # pos0 outside window, functional_hamming=0
    profile = {
        "index": 6,
        "profile_name": "SL5_0G-A_5pad6",
        "profile_sequence": mutant,
    }
    result = classify_functional_candidate(profile, _wt_profile(wt), 2, 4)
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"].startswith("functional_hamming_not_1_got_")
    assert result["functional_hamming"] == 0


def test_classify_excluded_functional_hamming_two() -> None:
    # Name says single mutation, but two functional edits
    wt = "AAGCUAAA"
    mutant = "AAGAUACA"  # pos3 C->A, pos6 outside... need 2 functional: pos3,pos4
    mutant = "AAGAAACA"  # pos3 C->A (func), pos4 U->A (func) => functional_hamming=2
    profile = {
        "index": 7,
        "profile_name": "SL5_0G-A_5pad6",
        "profile_sequence": mutant,
    }
    result = classify_functional_candidate(profile, _wt_profile(wt), 2, 4)
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"].startswith("functional_hamming_not_1_got_2")


def test_classify_excluded_name_sequence_mismatch_position() -> None:
    # Name encodes func pos 0, but the actual edit is at func pos 1
    wt = "AAGCUAAA"
    mutant = "AAGAUAAA"  # edit at func pos 1 (C->A), but name says pos 0
    profile = {
        "index": 8,
        "profile_name": "SL5_0C-A_5pad6",  # claims func pos 0
        "profile_sequence": mutant,
    }
    result = classify_functional_candidate(profile, _wt_profile(wt), 2, 4)
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"].startswith("name_sequence_mismatch_")
    assert "position" in result["exclusion_reason"]


def test_classify_excluded_name_sequence_mismatch_ref() -> None:
    # Position matches but ref base differs
    wt = "AAGCUAAA"
    mutant = "AAGAUAAA"  # func pos 1, ref C, alt A
    profile = {
        "index": 9,
        "profile_name": "SL5_1G-A_5pad6",  # claims ref G at func pos 1 (actual C)
        "profile_sequence": mutant,
    }
    result = classify_functional_candidate(profile, _wt_profile(wt), 2, 4)
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"].startswith("name_sequence_mismatch_")
    assert "ref" in result["exclusion_reason"]


def test_classify_excluded_missing_sequence() -> None:
    profile = {"index": 10, "profile_name": "SL5_0G-A_5pad6", "profile_sequence": None}
    result = classify_functional_candidate(profile, _wt_profile("AAGCUAAA"), 2, 4)
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"] == "missing_mutant_or_wt_sequence"


# ---------------------------------------------------------------------------
# find_wt_anchor_by_sequence
# ---------------------------------------------------------------------------


def test_find_wt_anchor_unique() -> None:
    full = "AAGCUAAA"
    profiles = [
        {"index": 1, "profile_name": "SL5_wt", "profile_sequence": full},
        {"index": 2, "profile_name": "SL5_0C-A", "profile_sequence": "AAGAUAAA"},
    ]
    wt = find_wt_anchor_by_sequence(profiles, full)
    assert wt is not None
    assert wt["index"] == 1


def test_find_wt_anchor_none() -> None:
    profiles = [
        {"index": 1, "profile_name": "SL5_0C-A", "profile_sequence": "AAGAUAAA"},
    ]
    assert find_wt_anchor_by_sequence(profiles, "AAGCUAAA") is None


def test_find_wt_anchor_ambiguous_returns_none() -> None:
    full = "AAGCUAAA"
    profiles = [
        {"index": 1, "profile_name": "SL5_wt_a", "profile_sequence": full},
        {"index": 2, "profile_name": "SL5_wt_b", "profile_sequence": full},
    ]
    # Two exact-match no-mutation profiles -> ambiguous -> None
    assert find_wt_anchor_by_sequence(profiles, full) is None


def test_find_wt_anchor_skips_exact_match_with_mutation_name() -> None:
    full = "AAGCUAAA"
    profiles = [
        {"index": 1, "profile_name": "SL5_0C-A", "profile_sequence": full},
        {"index": 2, "profile_name": "SL5_wt", "profile_sequence": full},
    ]
    wt = find_wt_anchor_by_sequence(profiles, full)
    assert wt is not None
    assert wt["index"] == 2


# ---------------------------------------------------------------------------
# verify_functional_anchor
# ---------------------------------------------------------------------------


def test_verify_functional_anchor_valid() -> None:
    full = "AA" + "GCUA" + "CCCC"  # functional at offset 2
    result = verify_functional_anchor(full, "GCUA", expected_offset=2)
    assert result["valid"] is True
    assert result["offset"] == 2
    assert result["occurrences"] == [2]
    assert result["reason"] is None


def test_verify_functional_anchor_default_offset_31() -> None:
    full = "A" * 31 + "GCUA" + "C" * 10
    result = verify_functional_anchor(full, "GCUA")
    assert result["valid"] is True
    assert result["offset"] == 31


def test_verify_functional_anchor_not_found() -> None:
    result = verify_functional_anchor("AAAACCCC", "GGGG", expected_offset=0)
    assert result["valid"] is False
    assert result["offset"] is None
    assert result["occurrences"] == []
    assert result["reason"] == "functional_anchor_not_found_in_full_anchor"


def test_verify_functional_anchor_not_unique() -> None:
    full = "GCUA" + "AAAA" + "GCUA"  # two occurrences
    result = verify_functional_anchor(full, "GCUA", expected_offset=0)
    assert result["valid"] is False
    assert result["reason"] == "functional_anchor_not_unique_in_full_anchor"
    assert len(result["occurrences"]) == 2


def test_verify_functional_anchor_wrong_offset() -> None:
    full = "A" * 5 + "GCUA" + "C" * 3  # at offset 5, expected 2
    result = verify_functional_anchor(full, "GCUA", expected_offset=2)
    assert result["valid"] is False
    assert result["offset"] == 5
    assert result["reason"] == "offset_5_expected_2"
