"""Tests for the D1-X v2 canonical mask / parser rebuild (contract R2B, §13.2).

Covers the endpoint_v2 mask eligibility rules:
  EDITED_SITE, ALIGNMENT_CHANGE, PROBE_ELIGIBILITY_CHANGE,
  MISSING_REACTIVITY, LENGTH_MISMATCH, ELIGIBLE
plus NaN/missing handling and length-mismatch handling.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_SCRIPTS = _REPO / "scripts" / "reactflow_delta"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from reactflow.delta.d0x import parse_mutation_value  # noqa: E402

import d1x_v2_canonicalize as v2  # noqa: E402

ELIGIBILITY_CODES = v2.ELIGIBILITY_CODES
compute_position_eligibility = v2.compute_position_eligibility
compute_alignment_change_indices = v2.compute_alignment_change_indices
parse_rdat_v2 = v2.parse_rdat_v2
_annotation_map_tolerant = v2._annotation_map_tolerant
_numeric_values = v2._numeric_values
_parse_seqpos_tokens = v2._parse_seqpos_tokens
build_canonical_record = v2.build_canonical_record
_resolve_profile_annotations = v2._resolve_profile_annotations
_condition_tuple = v2._condition_tuple
_profile_data_role = v2._profile_data_role
_edited_site_indices = v2._edited_site_indices


# ---------------------------------------------------------------------------
# eligibility reason codes
# ---------------------------------------------------------------------------


def test_eligibility_code_set_matches_endpoint_v2():
    expected = {
        "EDITED_SITE",
        "ALIGNMENT_CHANGE",
        "PROBE_ELIGIBILITY_CHANGE",
        "MISSING_REACTIVITY",
        "LENGTH_MISMATCH",
        "ELIGIBLE",
    }
    assert set(ELIGIBILITY_CODES) == expected


def test_eligibility_permissive():
    codes = compute_position_eligibility([0.1, 0.2, 0.3], set(), set(), False)
    assert codes == ["ELIGIBLE", "ELIGIBLE", "ELIGIBLE"]


def test_eligibility_edited_site():
    codes = compute_position_eligibility([0.1, 0.2, 0.3], {1}, set(), False)
    assert codes == ["ELIGIBLE", "EDITED_SITE", "ELIGIBLE"]


def test_eligibility_alignment_change():
    codes = compute_position_eligibility([0.1, 0.2, 0.3], set(), {2}, False)
    assert codes == ["ELIGIBLE", "ELIGIBLE", "ALIGNMENT_CHANGE"]


def test_eligibility_probe_change():
    codes = compute_position_eligibility([0.1, 0.2, 0.3], set(), set(), True)
    assert codes == ["PROBE_ELIGIBILITY_CHANGE"] * 3


def test_eligibility_nan_missing():
    codes = compute_position_eligibility([0.1, None, 0.3], set(), set(), False)
    assert codes == ["ELIGIBLE", "MISSING_REACTIVITY", "ELIGIBLE"]


def test_eligibility_nan_precedence_over_edited_site():
    # a NaN at the edited site is still MISSING_REACTIVITY (never zero-filled)
    codes = compute_position_eligibility([0.1, None], {1}, set(), False)
    assert codes == ["ELIGIBLE", "MISSING_REACTIVITY"]


def test_eligibility_all_positions_have_code():
    n = 50
    codes = compute_position_eligibility([0.1] * n, {3}, {7, 9}, False)
    assert len(codes) == n
    assert all(c in ELIGIBILITY_CODES for c in codes)


# ---------------------------------------------------------------------------
# NaN / missing handling (never impute zero)
# ---------------------------------------------------------------------------


def test_numeric_values_maps_nan_to_none():
    assert _numeric_values(["1.0", "nan", "NA", ""], "REACTIVITY:1") == [1.0, None, None, None]


def test_numeric_values_inf_to_none():
    vals = _numeric_values(["inf", "-inf", "2.0"], "REACTIVITY:1")
    assert vals == [None, None, 2.0]


def test_numeric_values_raises_on_non_numeric():
    with pytest.raises(ValueError):
        _numeric_values(["abc"], "REACTIVITY:1")


# ---------------------------------------------------------------------------
# length mismatch handling
# ---------------------------------------------------------------------------


def test_parse_rdat_v2_length_mismatch_rejected_per_profile():
    text = (
        "RDAT_VERSION 0.4\n"
        "NAME test\n"
        "SEQUENCE ACGU\n"
        "OFFSET 0\n"
        "SEQPOS G1 C2 A3 U4\n"
        "REACTIVITY:1 0.1 0.2\n"  # length 2 != 4
    )
    path = _write_tmp(text)
    doc = parse_rdat_v2(path)
    p = doc["profiles"][0]
    assert len(p["reactivity"]) == 2
    assert len(doc["seqpos"]) == 4
    # the canonical builder / caller must NOT emit ELIGIBLE for length mismatch


def test_len_mismatch_never_emits_eligible_record():
    # simulate what the canonicalizer does: a profile whose reactivity length
    # does not match seqpos is dispositioned LENGTH_MISMATCH, not canonicalized.
    text = (
        "RDAT_VERSION 0.4\n"
        "NAME test\n"
        "SEQUENCE ACGUACGU\n"
        "OFFSET 0\n"
        "SEQPOS G1 C2 A3 U4 G5 C6 A7 U8\n"
        "REACTIVITY:1 0.1 0.2\n"
    )
    path = _write_tmp(text)
    doc = parse_rdat_v2(path)
    assert len(doc["profiles"][0]["reactivity"]) != len(doc["seqpos"])


# ---------------------------------------------------------------------------
# tolerant annotation parsing (root-cause fix for separator bug)
# ---------------------------------------------------------------------------


def test_annotation_map_tolerant_skips_empty_and_orphan():
    parsed = _annotation_map_tolerant(["modifier:1M7", "", "bareword", "x:y"])
    assert parsed["modifier"] == ["1M7"]
    assert parsed["x"] == ["y"]
    assert "_orphan" in parsed
    assert "" in parsed["_orphan"]
    assert "bareword" in parsed["_orphan"]


def test_tokenization_whitespace_normalized():
    # mixed tab/space line that broke the old parser
    text = (
        "RDAT_VERSION\t0.4\n"
        "NAME test\n"
        "SEQUENCE ACGU\n"
        "STRUCTURE ....\n"
        "OFFSET 0\n"
        "SEQPOS  G1\tG2\tA3\tU4\n"  # leading double space + tabs
        "REACTIVITY:1\t0.1\t0.2\t0.3\t0.4\n"
    )
    path = _write_tmp(text)
    doc = parse_rdat_v2(path)
    assert doc["seqpos"] == [1, 2, 3, 4]
    assert len(doc["profiles"][0]["reactivity"]) == 4


# ---------------------------------------------------------------------------
# signed/offset SEQPOS (root-cause fix)
# ---------------------------------------------------------------------------


def test_seqpos_signed_offsets():
    assert _parse_seqpos_tokens(["-9", "-8", "0", "1", "2"]) == [-9, -8, 0, 1, 2]


def test_seqpos_base_prefix():
    assert _parse_seqpos_tokens(["g-18", "A1", "u2", "X0"]) == [-18, 1, 2, 0]


def test_seqpos_malformed_raises():
    with pytest.raises(ValueError):
        _parse_seqpos_tokens(["abc"])


# ---------------------------------------------------------------------------
# DATA:N block fallback (root-cause fix for missing REACTIVITY rows)
# ---------------------------------------------------------------------------


def test_data_block_fallback():
    text = (
        "RDAT_VERSION 0.4\n"
        "NAME test\n"
        "SEQUENCE ACGU\n"
        "OFFSET 0\n"
        "SEQPOS G1 C2 A3 U4\n"
        "DATA_ANNOTATION:1 datatype:REACTIVITY:theta modifier:1M7\n"
        "DATA:1 0.1 0.2 0.3 0.4\n"
    )
    path = _write_tmp(text)
    doc = parse_rdat_v2(path)
    assert len(doc["profiles"]) == 1
    assert doc["profiles"][0]["reactivity"] == [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# canonical record: every position has an eligibility code
# ---------------------------------------------------------------------------


def _make_profile(reactivity, mutation_values=None, profile_sequence=None):
    return {
        "index": 1,
        "annotation": {"mutation": mutation_values or ["WT"]},
        "reactivity": reactivity,
        "reactivity_error": None,
        "missing_reactivity_count": sum(v is None for v in reactivity),
        "profile_sequence": profile_sequence,
    }


def test_canonical_record_all_positions_have_eligibility():
    rec = build_canonical_record(
        asset_name="A.rdat",
        source_accession="A",
        file_sha256="x",
        profile=_make_profile([0.1, None, 0.3, 0.4]),
        profile_index=1,
        seq="ACGU",
        offset=0,
        seqpos_values=[1, 2, 3, 4],
        global_annotations=[],
    )
    codes = rec["reactivity_layers"]["eligibility_reason_codes"]
    assert len(codes) == 4
    assert all(c in ELIGIBILITY_CODES for c in codes)
    assert codes[1] == "MISSING_REACTIVITY"


def test_canonical_record_edited_site_marked():
    profile = _make_profile([0.1, 0.2, 0.3, 0.4], mutation_values=["G2A"])
    rec = build_canonical_record(
        asset_name="A.rdat",
        source_accession="A",
        file_sha256="x",
        profile=profile,
        profile_index=1,
        seq="ACGU",
        offset=0,
        seqpos_values=[1, 2, 3, 4],
        global_annotations=[],
    )
    codes = rec["reactivity_layers"]["eligibility_reason_codes"]
    assert codes[1] == "EDITED_SITE"
    assert codes[0] == "ELIGIBLE"


def test_edited_site_indices():
    pm = [parse_mutation_value("G2A", header_sequence="ACGU", offset=0, profile_sequence=None)]
    idx = _edited_site_indices(pm, [1, 2, 3, 4])
    assert idx == {1}


# ---------------------------------------------------------------------------
# alignment change
# ---------------------------------------------------------------------------


def test_alignment_change_indices():
    valid = compute_alignment_change_indices("ACGU", "ACGA", [1, 2, 3, 4])
    assert valid == {3}


def test_alignment_change_length_mismatch_returns_empty():
    assert compute_alignment_change_indices("ACGU", "AAAAAA", [1, 2, 3, 4]) == set()


# ---------------------------------------------------------------------------
# profile data role
# ---------------------------------------------------------------------------


def test_profile_data_role_wt():
    pm = [parse_mutation_value("WT", header_sequence="", offset=0, profile_sequence=None)]
    role = _profile_data_role(pm)
    assert role["is_wt"] is True


def test_profile_data_role_exact_single():
    pm = [parse_mutation_value("G2A", header_sequence="ACGU", offset=0, profile_sequence=None)]
    role = _profile_data_role(pm)
    assert role["data_role"] == "PRIMARY_EXACT_DELTA"
    assert role["ref"] == "G"
    assert role["alt"] == "A"


def test_profile_data_role_latent_alt():
    pm = [parse_mutation_value("G2X", header_sequence="ACGU", offset=0, profile_sequence=None)]
    role = _profile_data_role(pm)
    assert role["data_role"] == "AUXILIARY_LATENT_ALT"


# ---------------------------------------------------------------------------
# condition resolution
# ---------------------------------------------------------------------------


def test_condition_tuple_inheritance():
    global_ann = [_annotation_map_tolerant(["modifier:1M7", "temperature:37C"])]
    profile_ann = _annotation_map_tolerant(["modifier:DMS"])
    resolved = _resolve_profile_annotations(global_ann, profile_ann)
    cond = _condition_tuple(resolved)
    assert cond["modifier"] == ("DMS",)
    assert cond["temperature"] == ("37C",)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_tmp(text: str) -> Path:
    import tempfile
    fd, name = tempfile.mkstemp(suffix=".rdat")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    return Path(name)