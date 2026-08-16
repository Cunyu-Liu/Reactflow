#!/usr/bin/env python3
"""Unit tests for build_pair_publication_registry_v1.py (Task 1B)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "scripts/reactflow_delta"))

import build_pair_publication_registry_v1 as bp


def _pair(accession, wt_idx=1, mut_idx=2, offset=5, ref="G", alt="C",
          modifier=("1M7",), chemical=("MgCl2:10mM",)):
    return {
        "source_accession": accession,
        "wt_profile_index": wt_idx,
        "mutant_profile_index": mut_idx,
        "ref_allele": ref,
        "alt_allele": alt,
        "coordinate": {"offset": offset},
        "wt_reuse_group": f"{accession}:1",
        "condition": {"all_required_known": True, "chemical": list(chemical),
                      "modifier": list(modifier), "temperature": ["24C"]},
    }


def test_pair_id_is_deterministic():
    p = _pair("AA_1M7_0001")
    a = bp.build_pair_publication_registry([p], asset_meta={"AA_1M7_0001": {"asset_id": 1}})
    b = bp.build_pair_publication_registry([p], asset_meta={"AA_1M7_0001": {"asset_id": 1}})
    assert a[0]["pair_id"] == b[0]["pair_id"]


def test_pair_id_hex_is_64_chars():
    p = _pair("AA_1M7_0001")
    rows = bp.build_pair_publication_registry([p], asset_meta={"AA_1M7_0001": {"asset_id": 1}})
    pid = rows[0]["pair_id"]
    assert len(pid) == 64
    assert all(c in "0123456789abcdef" for c in pid)


def test_publication_normalized_prefers_pmid_over_doi_over_unresolved():
    base = {"rmdb_id": "AA_1M7_0001",
            "citation": {"title": "t", "authors": "a", "year": "2024"}}
    # PMID + DOI -> PMID
    reg_pmid = {"AA_1M7_0001": {**base, "citation": {**base["citation"], "pubmed": "38427602", "doi": "10.1/x"}}}
    rows = bp.build_pair_publication_registry([_pair("AA_1M7_0001")], registry=reg_pmid,
                                              asset_meta={"AA_1M7_0001": {"asset_id": 1}})
    assert rows[0]["publication_id_normalized"] == "pmid_38427602"
    assert rows[0]["citation_resolution_status"] == "RESOLVED"

    # DOI only -> DOI
    reg_doi = {"AA_1M7_0001": {**base, "citation": {**base["citation"], "doi": "10.1/y"}}}
    rows2 = bp.build_pair_publication_registry([_pair("AA_1M7_0001")], registry=reg_doi,
                                               asset_meta={"AA_1M7_0001": {"asset_id": 1}})
    assert rows2[0]["publication_id_normalized"] == "doi_10.1/y"

    # no pub/doi -> UNRESOLVED
    reg_none = {"AA_1M7_0001": base}
    rows3 = bp.build_pair_publication_registry([_pair("AA_1M7_0001")], registry=reg_none,
                                               asset_meta={"AA_1M7_0001": {"asset_id": 1}})
    assert rows3[0]["publication_id_normalized"] == "UNRESOLVED_PUBLICATION:AA"
    assert rows3[0]["citation_resolution_status"] == "UNRESOLVED_PUBLICATION"


def test_same_pmid_across_studies_flagged():
    reg = {}
    for acc, pmid in [("AA_1M7_0001", "12345"), ("BB_DMS_0001", "12345")]:
        reg[acc] = {"rmdb_id": acc, "citation": {"pubmed": pmid, "doi": "", "title": "t"}}
    rows = bp.build_pair_publication_registry(
        [_pair("AA_1M7_0001"), _pair("BB_DMS_0001")],
        registry=reg,
        asset_meta={"AA_1M7_0001": {"asset_id": 1}, "BB_DMS_0001": {"asset_id": 2}})
    anomalies = bp.publication_anomalies(rows)
    assert len(anomalies["same_pmid_across_studies"]) == 1
    assert anomalies["same_pmid_across_studies"][0]["pmid"] == "12345"
    assert anomalies["same_pmid_across_studies"][0]["studies"] == ["AA", "BB"]


def test_multi_pmid_within_study_flagged():
    # two profiles in the same study resolve to different PMIDs
    reg = {
        "CC_1M7_0001": {"rmdb_id": "CC_1M7_0001", "citation": {"pubmed": "111", "title": "t"}},
        "CC_DMS_0001": {"rmdb_id": "CC_DMS_0001", "citation": {"pubmed": "222", "title": "t"}},
    }
    rows = bp.build_pair_publication_registry(
        [_pair("CC_1M7_0001"), _pair("CC_DMS_0001")],
        registry=reg,
        asset_meta={"CC_1M7_0001": {"asset_id": 1}, "CC_DMS_0001": {"asset_id": 2}})
    anomalies = bp.publication_anomalies(rows)
    assert len(anomalies["multi_pmid_within_study"]) == 1
    assert anomalies["multi_pmid_within_study"][0]["study_id"] == "CC"
    assert anomalies["multi_pmid_within_study"][0]["pmids"] == ["111", "222"]


def test_unresolved_publications_do_not_count_toward_confirmed_n():
    reg = {
        "AA_1M7_0001": {"rmdb_id": "AA_1M7_0001", "citation": {"pubmed": "999", "title": "t"}},
        "ZZ_1M7_0001": {"rmdb_id": "ZZ_1M7_0001", "citation": {"pubmed": "", "doi": "", "title": "t"}},
    }
    rows = bp.build_pair_publication_registry(
        [_pair("AA_1M7_0001"), _pair("ZZ_1M7_0001")],
        registry=reg,
        asset_meta={"AA_1M7_0001": {"asset_id": 1}, "ZZ_1M7_0001": {"asset_id": 2}})
    # only the AA pmid counts; ZZ unresolved does not
    assert bp.confirmed_publication_n(rows) == 1
    unresolved = [r for r in rows if r["citation_resolution_status"] == "UNRESOLVED_PUBLICATION"]
    assert len(unresolved) == 1


def test_ledger_reports_counts_and_anomalies():
    reg = {"AA_1M7_0001": {"rmdb_id": "AA_1M7_0001", "citation": {"pubmed": "1", "title": "t"}},
           "BB_DMS_0001": {"rmdb_id": "BB_DMS_0001", "citation": {"pubmed": "1", "title": "t"}}}
    rows = bp.build_pair_publication_registry(
        [_pair("AA_1M7_0001"), _pair("BB_DMS_0001")],
        registry=reg,
        asset_meta={"AA_1M7_0001": {"asset_id": 1}, "BB_DMS_0001": {"asset_id": 2}})
    ledger = bp.build_ledger(rows, bp.publication_anomalies(rows))
    assert ledger["total_pairs"] == 2
    assert ledger["confirmed_publication_n"] == 1
    assert len(ledger["anomalies"]["same_pmid_across_studies"]) == 1