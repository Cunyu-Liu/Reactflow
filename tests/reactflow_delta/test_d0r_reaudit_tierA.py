"""D0-R v2 re-audit tests: Tier A selection, annotation parsing, annotation-only and
general single-mutant candidate classification.

Tests the pure functions in scripts/reactflow_delta/d0r_reaudit_tierA.py directly
with small synthetic inputs (no network, no real RDAT fixtures).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "reactflow_delta"
        / "d0r_reaudit_tierA.py"
    )
    spec = importlib.util.spec_from_file_location("d0r_reaudit_tierA", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_module()


# ---------------------------------------------------------------------------
# parse_annotation_mutations
# ---------------------------------------------------------------------------


def test_parse_annotation_single_mutation_g159c() -> None:
    out = mod.parse_annotation_mutations({"mutation": ["G159C"]})
    assert len(out) == 1
    m = out[0]
    assert m["position_1indexed"] == 159
    assert m["position_0indexed"] == 158
    assert m["ref"] == "G"
    assert m["alt"] == "C"
    assert m["encoding"] == "G159C"
    assert m["source"] == "annotation"


def test_parse_annotation_alt_x_preserved_as_variable() -> None:
    out = mod.parse_annotation_mutations({"mutation": ["A100X"]})
    assert len(out) == 1
    assert out[0]["position_1indexed"] == 100
    assert out[0]["ref"] == "A"
    assert out[0]["alt"] == "X"


def test_parse_annotation_wt_is_excluded() -> None:
    out = mod.parse_annotation_mutations({"mutation": ["WT", "G1C"]})
    assert len(out) == 1
    assert out[0]["encoding"] == "G1C"


def test_parse_annotation_wt_case_insensitive() -> None:
    assert mod.parse_annotation_mutations({"mutation": ["wt"]}) == []
    assert mod.parse_annotation_mutations({"mutation": ["Wt"]}) == []


def test_parse_annotation_multiple_mutations() -> None:
    out = mod.parse_annotation_mutations({"mutation": ["G10C", "A20U"]})
    assert len(out) == 2
    assert out[0]["position_1indexed"] == 10
    assert out[1]["position_1indexed"] == 20


def test_parse_annotation_ignores_non_matching_tokens() -> None:
    out = mod.parse_annotation_mutations({"mutation": ["G159C", "del5", "WT", "1G-A"]})
    assert len(out) == 1
    assert out[0]["encoding"] == "G159C"


def test_parse_annotation_empty_or_none() -> None:
    assert mod.parse_annotation_mutations({}) == []
    assert mod.parse_annotation_mutations({"mutation": []}) == []
    assert mod.parse_annotation_mutations(None) == []


# ---------------------------------------------------------------------------
# classify_profile_annotation_only
# ---------------------------------------------------------------------------


def _seq_with_base(at_1indexed: int, base: str, length: int = 200) -> str:
    arr = ["A"] * length
    arr[at_1indexed - 1] = base
    return "".join(arr)


def test_classify_annotation_only_candidate_ref_match() -> None:
    # header SEQUENCE has G at position 159 (1-indexed); annotation G159C.
    header = _seq_with_base(159, "G")
    profile = {
        "index": 2,
        "profile_name": "RNASEP_159",
        "annotation": {"mutation": ["G159C"]},
    }
    result = mod.classify_profile_annotation_only(profile, header, offset=0)
    assert result["classification"] == "candidate_single_annotation_only"
    assert result["true_pair"] is False
    assert "candidate_only_pending" in result["lineage_status"]
    mm = result["matched_mutation"]
    assert mm["encoded_position_1indexed"] == 159
    assert mm["encoded_ref"] == "G"
    assert mm["encoded_alt"] == "C"
    assert mm["ref_verified_against"] == "header_SEQUENCE"
    assert mm["ref_match_index"] == "construct_local_1indexed"
    assert mm["alt_not_verified"] is False


def test_classify_annotation_only_alt_x_not_verified() -> None:
    header = _seq_with_base(100, "A")
    profile = {
        "index": 3,
        "profile_name": "L21_100",
        "annotation": {"mutation": ["A100X"]},
    }
    result = mod.classify_profile_annotation_only(profile, header, offset=0)
    assert result["classification"] == "candidate_single_annotation_only"
    assert result["matched_mutation"]["alt_not_verified"] is True
    assert result["matched_mutation"]["encoded_alt"] == "X"


def test_classify_annotation_only_ref_mismatch_excluded() -> None:
    # annotation claims G at pos 159 but header has A there.
    header = _seq_with_base(159, "A")
    profile = {
        "index": 4,
        "profile_name": "RNASEP_159",
        "annotation": {"mutation": ["G159C"]},
    }
    result = mod.classify_profile_annotation_only(profile, header, offset=0)
    assert result["classification"] == "excluded"
    assert result["true_pair"] is False
    assert result["lineage_status"] == "excluded"
    assert result["exclusion_reason"].startswith("annotation_ref_mismatch")
    assert "G" in result["exclusion_reason"]
    assert "A" in result["exclusion_reason"]
    assert "159" in result["exclusion_reason"]


def test_classify_annotation_only_no_annotation_mutation() -> None:
    profile = {"index": 5, "profile_name": "plain", "annotation": {"mutation": []}}
    result = mod.classify_profile_annotation_only(profile, "ACGU", offset=0)
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"] == "no_annotation_mutation"


def test_classify_annotation_only_multiple_mutations_excluded() -> None:
    header = _seq_with_base(10, "G")
    profile = {
        "index": 6,
        "profile_name": "multi",
        "annotation": {"mutation": ["G10C", "A20U"]},
    }
    result = mod.classify_profile_annotation_only(profile, header, offset=0)
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"] == "multiple_annotation_mutations"


def test_classify_annotation_only_offset_adjusted_match() -> None:
    # annotation G5C; without offset the local index 4 has A (mismatch), but
    # with offset=10 the adjusted index 14 has G -> match via offset_adjusted.
    arr = ["A"] * 30
    arr[14] = "G"  # 1-indexed pos 15 = offset(10) + pos(5)
    header = "".join(arr)
    profile = {
        "index": 7,
        "profile_name": "offset_test",
        "annotation": {"mutation": ["G5C"]},
    }
    result = mod.classify_profile_annotation_only(profile, header, offset=10)
    assert result["classification"] == "candidate_single_annotation_only"
    assert result["matched_mutation"]["ref_match_index"] == "offset_adjusted"


def test_classify_annotation_only_no_header_sequence() -> None:
    profile = {
        "index": 8,
        "profile_name": "nohdr",
        "annotation": {"mutation": ["G159C"]},
    }
    result = mod.classify_profile_annotation_only(profile, None, offset=0)
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"].startswith("annotation_ref_mismatch")


# ---------------------------------------------------------------------------
# find_wt_anchor_general
# ---------------------------------------------------------------------------


def test_find_wt_anchor_by_mutation_wt_annotation() -> None:
    profiles = [
        {"index": 1, "profile_name": "mut", "annotation": {"mutation": ["G10C"]}},
        {"index": 2, "profile_name": "wt", "annotation": {"mutation": ["WT"]}},
    ]
    anchor, method = mod.find_wt_anchor_general(profiles, None)
    assert anchor is not None
    assert anchor["index"] == 2
    assert method == "mutation_wt_annotation"


def test_find_wt_anchor_by_header_sequence_match() -> None:
    header = "ACGUACGU"
    profiles = [
        {"index": 1, "profile_name": "mut", "profile_sequence": "ACGUACGA"},
        {"index": 2, "profile_name": "wt", "profile_sequence": "ACGUACGU"},
    ]
    anchor, method = mod.find_wt_anchor_general(profiles, header)
    assert anchor is not None
    assert anchor["index"] == 2
    assert method == "header_sequence_match"


def test_find_wt_anchor_none_when_no_signal() -> None:
    # No mutation:WT annotation, no per-profile sequence, no header -> all
    # three strategies fail (find_wt_anchor returns None when no profile has
    # a profile_sequence).
    profiles = [{"index": 1, "profile_name": "SL5_0G-A"}]
    anchor, method = mod.find_wt_anchor_general(profiles, None)
    assert anchor is None
    assert method == "none"


# ---------------------------------------------------------------------------
# classify_profile_general (sequence + name/annotation encoding)
# ---------------------------------------------------------------------------


def test_classify_general_candidate_name_encoded_match() -> None:
    # No padding: name 0-indexed pos 1 == absolute 0-indexed pos 1 (1-indexed pos 2).
    # WT has C at index 1; mutant has G; SEQPOS marks pos 2 as functional.
    wt = "ACGU"
    mutant = "AGGU"
    wt_profile = {"index": 1, "profile_name": "wt", "profile_sequence": wt}
    profile = {
        "index": 2,
        "profile_name": "SL5_1C-G_pad",  # name encodes pos 1 (0-indexed) C->G
        "profile_sequence": mutant,
    }
    result = mod.classify_profile_general(profile, wt_profile, [2])
    assert result["classification"] == "candidate_single_functional_anchor"
    assert result["true_pair"] is False
    assert result["functional_edit_count"] == 1
    assert result["matched_mutation"]["actual_ref"] == "C"
    assert result["matched_mutation"]["actual_alt"] == "G"


def test_classify_general_excluded_no_encoded_mutation() -> None:
    wt = "AAGCUAAA"
    mutant = "AAGAUAAA"
    wt_profile = {"index": 1, "profile_name": "wt", "profile_sequence": wt}
    profile = {"index": 2, "profile_name": "plain", "profile_sequence": mutant}
    result = mod.classify_profile_general(profile, wt_profile, [3])
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"] == "no_encoded_mutation"


def test_classify_general_excluded_functional_count_not_1() -> None:
    wt = "AAGCUAAA"  # 1-indexed: pos4=C, pos5=U
    mutant = "AAGAAACA"  # pos4=A, pos5=A -> two functional edits
    wt_profile = {"index": 1, "profile_name": "wt", "profile_sequence": wt}
    profile = {
        "index": 2,
        "profile_name": "SL5_1C-A_pad",  # claims single mutation
        "profile_sequence": mutant,
    }
    result = mod.classify_profile_general(profile, wt_profile, [4, 5])
    assert result["classification"] == "excluded"
    assert result["exclusion_reason"].startswith("functional_edit_count_2_not_1")


# ---------------------------------------------------------------------------
# select_tierA_non_ribo
# ---------------------------------------------------------------------------


def _reg_entry(rmdb_id: str, comments: str, *, owner: str = "Test") -> dict:
    return {
        "rmdb_id": rmdb_id,
        "name": f"Test_{rmdb_id}",
        "description": comments,
        "comments": comments,
        "category": "General",
        "rdat_url": f"https://example.invalid/{rmdb_id}.rdat",
        "construct_count": "100",
        "owner": owner,
        "citation": {"doi": "10.0/0", "authors": "Test", "year": "2024", "title": "", "journal": "", "pubmed": ""},
    }


def test_select_tierA_filters_ribonanza_and_v1_and_non_tierA(tmp_path) -> None:
    registry = tmp_path / "registry.jsonl"
    entries = [
        # Ribonanza/Kaggle library -> excluded by RIBO_RE
        _reg_entry("RIBO_TEST_0000", "Ribonanza kaggle 15klib library"),
        # Tier A mutate-and-map m2-seq, not in v1 -> selected
        _reg_entry("MYM2SEQ_0000", "mutate-and-map-seq m2-seq construct"),
        # Tier A signal but already downloaded in D0-R v1 -> skipped
        _reg_entry("M2SL5_DMS_0000", "mutate-map rescue m2r"),
        # No Tier A signal -> skipped
        _reg_entry("PLAIN_0000", "plain structure no mutation signal"),
        # Another Tier A (SNP/riboSNitch) -> selected
        _reg_entry("SNP_TEST_0000", "riboSNitch single-nucleotide variant SNP"),
    ]
    registry.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    selected = mod.select_tierA_non_ribo(registry)
    ids = [s["rmdb_id"] for s in selected]
    assert "MYM2SEQ_0000" in ids
    assert "SNP_TEST_0000" in ids
    assert "RIBO_TEST_0000" not in ids
    assert "M2SL5_DMS_0000" not in ids  # in DOWNLOADED_V1
    assert "PLAIN_0000" not in ids
    assert all("_tierA_signals" in s for s in selected)


def test_select_tierA_signals_assigned(tmp_path) -> None:
    registry = tmp_path / "registry.jsonl"
    e = _reg_entry("M2SEQ_0001", "mutate-and-map m2-seq")
    registry.write_text(json.dumps(e), encoding="utf-8")
    selected = mod.select_tierA_non_ribo(registry)
    assert len(selected) == 1
    sigs = selected[0]["_tierA_signals"]
    assert "mutate_and_map" in sigs
    assert "m2_seq" in sigs
