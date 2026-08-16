#!/usr/bin/env python3
"""Unit tests for build_asset_disposition_v3.py (Task 1A)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "scripts/reactflow_delta"))

import pytest

import build_asset_disposition_v3 as ad


def _asset(asset_id, accession, disposition="PARSE_SUCCESS", license_status="VERIFIED_CC0_RMDB"):
    return {
        "asset_id": asset_id,
        "source_accession": accession,
        "asset_name": f"{accession}.rdat",
        "source_group": "data-general",
        "release_tag": "data-general",
        "license_status": license_status,
        "file_present": True,
        "file_verified_hash": True,
        "disposition": disposition,
        "parse_error": None if disposition == "PARSE_SUCCESS" else "boom",
    }


def _profile(accession, reason):
    return {
        "asset_name": f"{accession}.rdat",
        "source_accession": accession,
        "disposition_reason": reason,
        "error": "e",
        "status": reason,
    }


def test_one_row_per_asset_uniqueness_no_silent_drop():
    assets = [
        _asset(1, "AA_1M7_0001", "PARSE_SUCCESS"),
        _asset(2, "BB_DMS_0001", "PARSE_FAIL_LENGTH_MISMATCH_REACTIVITY_VS_SEQPOS"),
        _asset(3, "CC_SHP_0001", "PARSE_SUCCESS"),
    ]
    rows = ad.build_asset_disposition_rows(assets, [])
    assert len(rows) == 3
    ids = [r["asset_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 3
    # every asset present, order-preserving
    assert {r["asset_id"] for r in rows} == {1, 2, 3}


def test_length_mismatch_profile_rows_counted_separately_from_asset_count():
    # 1 asset, but many LENGTH_MISMATCH *profile* rows -> profile count is
    # separate and larger than the asset count (11309 is a profile count).
    assets = [_asset(1, "AA_1M7_0001", "PARSE_FAIL_LENGTH_MISMATCH_REACTIVITY_VS_SEQPOS")]
    profiles = [_profile("AA_1M7_0001", "LENGTH_MISMATCH") for _ in range(7)]
    rows = ad.build_asset_disposition_rows(assets, profiles)
    assert len(rows) == 1  # exactly one asset row
    assert rows[0]["profile_failure_rows"] == 7  # profile count, not asset count
    assert rows[0]["profile_failure_rows"] != len(rows)


def test_parse_failure_is_explicit_status_never_zero_reactivity():
    assets = [_asset(1, "AA_1M7_0001", "PARSE_FAIL_MISSING_REACTIVITY_ROWS")]
    rows = ad.build_asset_disposition_rows(assets, [])
    assert rows[0]["asset_parse_status"] == "PARSE_FAIL_MISSING_REACTIVITY_ROWS"
    # never written as empty or 0 reactivity
    assert rows[0]["asset_parse_status"] != ""
    assert rows[0]["profile_failure_rows"] == 0
    assert rows[0]["unique_failure_reason"] == "PARSE_FAIL_MISSING_REACTIVITY_ROWS"


def test_missing_asset_info_still_gets_row_with_explicit_unknown():
    # An asset with no canonical count, no registry entry, no raw file must
    # still be present with UNKNOWN/NOT_RUN values, never dropped.
    assets = [_asset(1, "ZZ_1M7_0001", "PARSE_SUCCESS")]
    rows = ad.build_asset_disposition_rows(assets, [], registry={}, sha256_by_accession={})
    assert len(rows) == 1
    assert rows[0]["n_profiles"] == "NOT_RUN"
    assert rows[0]["n_records"] == "NOT_RUN"
    assert rows[0]["n_exact_pairs"] == "NOT_RUN"
    assert rows[0]["asset_sha256"] == "UNKNOWN_NOT_ASSERTED"
    assert rows[0]["citation_resolution_status"] == "UNRESOLVED_PUBLICATION"


def test_counts_and_sha_filled_from_caches():
    assets = [_asset(1, "AA_1M7_0001", "PARSE_SUCCESS")]
    record_counts = {"AA_1M7_0001": {"n_records": 150, "n_profiles": 3}}
    pair_counts = {"AA_1M7_0001": 12}
    rows = ad.build_asset_disposition_rows(
        assets, [],
        record_counts=record_counts,
        pair_counts=pair_counts,
        sha256_by_accession={"AA_1M7_0001": "a" * 64},
    )
    assert rows[0]["n_records"] == "150"
    assert rows[0]["n_profiles"] == "3"
    assert rows[0]["n_exact_pairs"] == "12"
    assert rows[0]["asset_sha256"] == "a" * 64
    assert rows[0]["failure_recoverability"] == "NOT_APPLICABLE"


def test_citation_resolution_status_from_registry():
    assets = [_asset(1, "AA_1M7_0001", "PARSE_SUCCESS")]
    registry = {
        "AA_1M7_0001": {
            "rmdb_id": "AA_1M7_0001",
            "citation": {"pubmed": "38427602", "doi": "10.1/x", "title": "t", "year": "2024"},
        },
    }
    rows = ad.build_asset_disposition_rows(assets, [], registry=registry)
    assert rows[0]["citation_resolution_status"] == "RESOLVED"

    # entry with neither pubmed nor doi -> unresolved (never invented)
    registry2 = {"AA_1M7_0001": {"rmdb_id": "AA_1M7_0001",
                                  "citation": {"pubmed": "", "doi": "", "title": "t"}}}
    rows2 = ad.build_asset_disposition_rows(assets, [], registry=registry2)
    assert rows2[0]["citation_resolution_status"] == "UNRESOLVED_PUBLICATION"


def test_unique_failure_reason_most_common_profile_reason():
    assets = [_asset(1, "AA_1M7_0001", "PARSE_FAIL_OTHER")]
    profiles = [
        _profile("AA_1M7_0001", "LENGTH_MISMATCH"),
        _profile("AA_1M7_0001", "LENGTH_MISMATCH"),
        _profile("AA_1M7_0001", "UNPARSEABLE"),
    ]
    rows = ad.build_asset_disposition_rows(assets, profiles)
    assert rows[0]["unique_failure_reason"] == "LENGTH_MISMATCH"
    assert rows[0]["profile_failure_rows"] == 2


def test_schema_validates_sample_row():
    schema = ad.asset_disposition_v3_schema()
    assert schema["row_key"] == "asset_id"
    field_names = [f["name"] for f in schema["fields"]]
    assert field_names == ad.ASSET_KEYS
    # every required field appears in a sample row
    sample = {
        "asset_id": 1, "source_url_or_accession": "AA_1M7_0001",
        "source_accession": "AA_1M7_0001", "asset_name": "AA_1M7_0001.rdat",
        "asset_sha256": "UNKNOWN_NOT_ASSERTED", "parser_version": "v2",
        "asset_parse_status": "PARSE_SUCCESS", "n_profiles": "NOT_RUN",
        "n_records": "NOT_RUN", "n_exact_pairs": "NOT_RUN",
        "profile_failure_rows": 0, "unique_failure_reason": "NONE",
        "failure_recoverability": "NOT_APPLICABLE", "license_status": "VERIFIED_CC0_RMDB",
        "citation_resolution_status": "RESOLVED", "notes": "",
    }
    for f in schema["fields"]:
        if f["required"]:
            assert f["name"] in sample


def test_duplicate_asset_id_raises():
    with pytest.raises(ValueError, match="uniqueness"):
        ad.build_asset_disposition_rows(
            [_asset(1, "AA_1M7_0001"), _asset(1, "BB_DMS_0001")], [])


def test_hash_never_fabricated_when_file_missing():
    raw_root = Path("/nonexistent/raw/rmdb")
    assert ad.find_and_hash_rdat("DOESNOTEXIST_0000", raw_root) == "UNKNOWN_NOT_ASSERTED"