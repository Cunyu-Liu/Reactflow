"""D0-R tests: per-profile sequence, name mutation encoding, WT anchor, edit-set."""

from __future__ import annotations

from pathlib import Path

import pytest

from reactflow.delta.rdat import (
    RdatParseError,
    classify_profile_edit,
    compute_edit_set,
    find_wt_anchor,
    is_wt_profile,
    parse_mutations_from_name,
    parse_rdat,
    seqpos_to_indices,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# Per-profile SEQUENCE:N indexed line support
# ---------------------------------------------------------------------------


def test_parse_per_profile_sequence_indexed_line(tmp_path: Path) -> None:
    """SEQUENCE:N per-profile indexed lines are parsed alongside the global header."""
    path = _write(
        tmp_path / "indexed.rdat",
        "\n".join(
            [
                "RDAT_VERSION\t0.34",
                "NAME\tindexed",
                "SEQUENCE\tXXXX",
                "OFFSET\t0",
                "ANNOTATION\texperimentType:MutateAndMap",
                "ANNOTATION_DATA:1\tmutation:WT",
                "ANNOTATION_DATA:2\tmutation:A1G",
                "SEQPOS\tX1\tX2\tX3\tX4",
                "SEQUENCE:1\tACGU",
                "SEQUENCE:2\tGCGU",
                "REACTIVITY:1\t0.1\t0.2\t0.3\t0.4",
                "REACTIVITY:2\t0.5\t0.6\t0.7\t0.8",
            ]
        ),
    )
    document = parse_rdat(path)
    assert document["profiles"][0]["profile_sequence"] == "ACGU"
    assert document["profiles"][0]["profile_sequence_source"] == "sequence_indexed_line"
    assert document["profiles"][1]["profile_sequence"] == "GCGU"
    assert document["profiles"][1]["profile_sequence_source"] == "sequence_indexed_line"


def test_parse_per_profile_sequence_from_annotation_token(tmp_path: Path) -> None:
    """M2-seq style: per-profile sequence in ANNOTATION_DATA sequence: token."""
    path = _write(
        tmp_path / "m2seq.rdat",
        "\n".join(
            [
                "RDAT_VERSION\t0.34",
                "NAME\tSL5_M2seq",
                "SEQUENCE\tXXXXXXXXXXXX",
                "OFFSET\t0",
                "ANNOTATION\tmodifier:2A3",
                "ANNOTATION_DATA:1\tsequence:ACGUACGUACGU\tname:SL5_wt",
                "ANNOTATION_DATA:2\tsequence:GCGUACGUACGU\tname:SL5_0G-A",
                "SEQPOS\tX1\tX2\tX3\tX4\tX5\tX6\tX7\tX8\tX9\tX10\tX11\tX12",
                "REACTIVITY:1\t0.1\t0.2\t0.3\t0.4\t0.1\t0.2\t0.3\t0.4\t0.1\t0.2\t0.3\t0.4",
                "REACTIVITY:2\t0.5\t0.6\t0.7\t0.8\t0.1\t0.2\t0.3\t0.4\t0.1\t0.2\t0.3\t0.4",
            ]
        ),
    )
    document = parse_rdat(path)
    assert document["profiles"][0]["profile_sequence"] == "ACGUACGUACGU"
    assert document["profiles"][0]["profile_sequence_source"] == "annotation_sequence_token"
    assert document["profiles"][0]["profile_name"] == "SL5_wt"
    assert document["profiles"][1]["profile_name"] == "SL5_0G-A"


def test_backward_compat_d0_fixture_still_parses(tmp_path: Path) -> None:
    """Old D0 fixtures (no per-profile sequence) still parse without error."""
    path = _write(
        tmp_path / "d0style.rdat",
        "\n".join(
            [
                "RDAT_VERSION\t0.34",
                "NAME\tfixture",
                "SEQUENCE\tACGU",
                "OFFSET\t0",
                "ANNOTATION\texperimentType:MutateAndMap\tmodifier:DMS",
                "ANNOTATION_DATA:1\tmutation:WT",
                "ANNOTATION_DATA:2\tmutation:A1X",
                "SEQPOS\tA1\tC2\tG3\tU4",
                "REACTIVITY:1\t0.1\tNaN\t0.3\t0.4",
                "REACTIVITY:2\t0.2\t0.3\t0.4\t0.5",
                "REACTIVITY_ERROR:2\t0.01\t0.01\t0.01\t0.01",
            ]
        ),
    )
    document = parse_rdat(path)
    assert document["profiles"][0]["profile_sequence"] is None
    assert document["profiles"][0]["profile_sequence_source"] is None
    assert document["profiles"][0]["reactivity"][1] is None
    assert document["profiles"][0]["missing_reactivity_count"] == 1
    assert document["profiles"][1]["annotation"]["mutation"] == ["A1X"]


# ---------------------------------------------------------------------------
# Mutation encoding from name
# ---------------------------------------------------------------------------


def test_parse_single_mutation_from_name() -> None:
    mutations = parse_mutations_from_name("SL5_SARS_CoV_2_0G-A_5pad6_w53barcode")
    assert len(mutations) == 1
    assert mutations[0] == {"position": 0, "ref": "G", "mut": "A", "encoding": "0G-A"}


def test_parse_multi_mutation_from_name() -> None:
    mutations = parse_mutations_from_name("SL5_MERS_GCadded_78C-T_86G-C_0pad0_w53barcode")
    assert len(mutations) == 2
    assert mutations[0]["position"] == 78
    assert mutations[0]["ref"] == "C"
    assert mutations[0]["mut"] == "T"
    assert mutations[1]["position"] == 86
    assert mutations[1]["ref"] == "G"
    assert mutations[1]["mut"] == "C"


def test_parse_no_mutation_from_wt_name() -> None:
    assert parse_mutations_from_name("SL5_SARS_CoV_2_5pad6_w53barcode") == []
    assert parse_mutations_from_name(None) == []


def test_parse_mutation_does_not_match_year_or_version() -> None:
    # Year-like and version-like numbers must not be parsed as mutations.
    assert parse_mutations_from_name("Eterna_R78_0000") == []
    assert parse_mutations_from_name("dataset_2024-07-30") == []


# ---------------------------------------------------------------------------
# WT anchor identification
# ---------------------------------------------------------------------------


def test_is_wt_profile_by_mutation_annotation() -> None:
    profile = {"annotation": {"mutation": ["WT"]}, "profile_name": None}
    assert is_wt_profile(profile) is True


def test_is_wt_profile_by_name_without_mutation_suffix() -> None:
    profile = {"annotation": {}, "profile_name": "SL5_SARS_CoV_2_5pad6_w53barcode"}
    assert is_wt_profile(profile) is True


def test_is_not_wt_profile_with_mutation_suffix() -> None:
    profile = {"annotation": {}, "profile_name": "SL5_SARS_CoV_2_0G-A_5pad6_w53barcode"}
    assert is_wt_profile(profile) is False


def test_find_wt_anchor_prefers_explicit_wt_annotation() -> None:
    profiles = [
        {"index": 1, "annotation": {}, "profile_name": "SL5_wt", "profile_sequence": "ACGU"},
        {"index": 2, "annotation": {"mutation": ["WT"]}, "profile_name": "explicit_wt", "profile_sequence": "ACGU"},
        {"index": 3, "annotation": {}, "profile_name": "SL5_0G-A", "profile_sequence": "GCGU"},
    ]
    anchor = find_wt_anchor(profiles)
    assert anchor is not None
    assert anchor["index"] == 2


def test_find_wt_anchor_falls_back_to_name_without_suffix() -> None:
    profiles = [
        {"index": 1, "annotation": {}, "profile_name": "SL5_SARS_CoV_2_5pad6_w53barcode", "profile_sequence": "ACGU"},
        {"index": 2, "annotation": {}, "profile_name": "SL5_0G-A", "profile_sequence": "GCGU"},
    ]
    anchor = find_wt_anchor(profiles)
    assert anchor is not None
    assert anchor["index"] == 1


# ---------------------------------------------------------------------------
# Edit-set computation
# ---------------------------------------------------------------------------


def test_compute_edit_set_single_mutation() -> None:
    result = compute_edit_set("GCGU", "ACGU", seqpos_indices=[1, 2, 3, 4])
    assert result["status"] == "computed"
    assert result["edit_count"] == 1
    assert result["functional_edit_count"] == 1
    assert result["edits"][0] == {"position_1indexed": 1, "wt_base": "A", "mutant_base": "G"}


def test_compute_edit_set_separates_functional_and_flanking() -> None:
    """Barcode/adapter edits outside SEQPOS are reported as flanking, not functional."""
    # Full construct: 6nt adapter + 4nt functional + 6nt barcode
    wt = "GGGGGG" + "ACGU" + "AAAAAA"
    mut = "GGGGGG" + "GCGU" + "TTTTTT"  # 1 functional edit + 6 barcode edits
    result = compute_edit_set(mut, wt, seqpos_indices=[7, 8, 9, 10])
    assert result["edit_count"] == 7
    assert result["functional_edit_count"] == 1
    assert result["flanking_edit_count"] == 6
    assert result["functional_edits"][0]["position_1indexed"] == 7


def test_compute_edit_set_skips_missing_sequence() -> None:
    result = compute_edit_set(None, "ACGU")
    assert result["status"] == "skipped"
    assert result["edit_count"] == 0


def test_compute_edit_set_skips_length_mismatch() -> None:
    result = compute_edit_set("ACG", "ACGU")
    assert result["status"] == "skipped"
    assert result["reason"] == "length_mismatch"


# ---------------------------------------------------------------------------
# Full classification with M2SL5-style fixture
# ---------------------------------------------------------------------------


def test_classify_m2sl5_style_candidate_single(tmp_path: Path) -> None:
    """M2SL5-style: WT anchor + single-mutant with barcode variation.

    The functional edit is 1 (matching the name encoding), but the barcode
    region also varies. The classifier should report candidate_single_from_name
    with lineage_status = candidate_only_unverified.
    """
    path = _write(
        tmp_path / "m2sl5_mini.rdat",
        "\n".join(
            [
                "RDAT_VERSION\t0.34",
                "NAME\tSL5_M2seq",
                "SEQUENCE\tXXXXXXXXXXXXXXXXXXXX",
                "OFFSET\t0",
                "ANNOTATION\tmodifier:2A3",
                # 6nt adapter + 4nt functional + 6nt barcode + 4nt adapter = 20
                # RNA sequences use U (not T); name encoding may use T (DNA convention)
                "ANNOTATION_DATA:1\tsequence:GGGGGGACGUCCCCCCAAAA\tname:SL5_wt",
                "ANNOTATION_DATA:2\tsequence:GGGGGGGCGUGGGGGGAAAA\tname:SL5_0G-A",
                "SEQPOS\tX7\tX8\tX9\tX10",
                "REACTIVITY:1\t0.1\t0.2\t0.3\t0.4",
                "REACTIVITY:2\t0.5\t0.6\t0.7\t0.8",
            ]
        ),
    )
    document = parse_rdat(path)
    profiles = document["profiles"]
    wt = find_wt_anchor(profiles)
    assert wt is not None
    assert wt["index"] == 1
    seqpos_indices = seqpos_to_indices(document["seqpos"])
    assert seqpos_indices == [7, 8, 9, 10]

    mutant = profiles[1]
    classification = classify_profile_edit(mutant, wt, seqpos_indices)
    assert classification["name_encoded_mutation_count"] == 1
    assert classification["edit_set"]["edit_count"] == 7  # 1 functional + 6 barcode
    assert classification["edit_set"]["functional_edit_count"] == 1
    assert classification["edit_class"] == "candidate_single_from_name"
    assert classification["lineage_status"] == "candidate_only_unverified"


def test_classify_multi_edit_without_name_encoding() -> None:
    """Full-length multi-diff without name encoding is NOT a single mutation."""
    wt_profile = {
        "index": 1,
        "annotation": {},
        "profile_name": "wt",
        "profile_sequence": "ACGUACGU",
    }
    mutant_profile = {
        "index": 2,
        "annotation": {},
        "profile_name": "no_encoding",
        "profile_sequence": "GCAUACGU",  # 2 edits, no name encoding
    }
    classification = classify_profile_edit(mutant_profile, wt_profile, [1, 2, 3, 4, 5, 6, 7, 8])
    assert classification["edit_class"] == "multi_edit_no_name_encoding"
    assert classification["edit_set"]["functional_edit_count"] == 2


def test_classify_double_mutation_from_name() -> None:
    """Name encoding with 2 mutations -> candidate_multi_from_name."""
    wt_profile = {
        "index": 1,
        "annotation": {},
        "profile_name": "construct_wt",
        "profile_sequence": "ACGUACGU",
    }
    mutant_profile = {
        "index": 2,
        "annotation": {},
        "profile_name": "construct_0A-G_3U-A_w53",
        "profile_sequence": "GCGAACGU",  # pos1 A->G, pos4 U->A (exactly 2 edits)
    }
    classification = classify_profile_edit(mutant_profile, wt_profile, [1, 2, 3, 4, 5, 6, 7, 8])
    assert classification["name_encoded_mutation_count"] == 2
    assert classification["edit_class"] == "candidate_multi_from_name"


# ---------------------------------------------------------------------------
# seqpos_to_indices
# ---------------------------------------------------------------------------


def test_seqpos_to_indices_strips_nucleotide_prefix() -> None:
    assert seqpos_to_indices(["A1", "C2", "G3", "U4"]) == [1, 2, 3, 4]
    assert seqpos_to_indices(["X27", "X28", "X29"]) == [27, 28, 29]


def test_seqpos_to_indices_ignores_malformed_tokens() -> None:
    # Malformed tokens (e.g. "X87X88" from file corruption) are silently skipped.
    result = seqpos_to_indices(["X27", "X87X88", "X30"])
    assert result == [27, 30]
