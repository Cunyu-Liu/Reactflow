#!/usr/bin/env python3
"""Unit tests for sequence_lineage_overlap_v1.py (Task 1C)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "scripts/reactflow_delta"))

import sequence_lineage_overlap_v1 as sl


def _row(pair_id, pub, seq, parent="P1", lineage="L1", family="F1", acc="AA_1M7_0001"):
    return {
        "pair_id": pair_id,
        "source_accession": acc,
        "study_id": acc.split("_")[0],
        "publication_id_normalized": pub,
        "sequence": seq,
        "lineage_id": lineage,
        "parent_id": parent,
        "rna_family": family,
    }


def test_shared_parent_detection():
    # two pairs from different publications sharing the same WT parent (design_group)
    rows = [
        _row("p1", "pmid_1", "AAAA", parent="design_group_1"),
        _row("p2", "pmid_2", "CCCC", parent="design_group_1"),
    ]
    per_pair, summary = sl.compute_sequence_lineage_metrics(rows)
    assert summary["shared_wt_parent_groups"] == 1
    assert summary["shared_wt_parent_pairs"] == 2
    flags = {r["pair_id"]: r["shared_wt_parent"] for r in per_pair}
    assert flags == {"p1": 1, "p2": 1}


def test_exact_sequence_duplicate_detection():
    rows = [
        _row("p1", "pmid_1", "AAAA"),
        _row("p2", "pmid_2", "AAAA"),
        _row("p3", "pmid_3", "GGGG"),
    ]
    _, summary = sl.compute_sequence_lineage_metrics(rows)
    assert summary["exact_sequence_duplicate_sequences"] == 1
    assert summary["exact_sequence_duplicate_pairs"] == 2


def test_homology_threshold_sensitivity_monotonic():
    # strongly similar sequences: flagged at 70 and 80, but not at 90
    # identity 14/16 = 0.875 -> >= 0.7 and >= 0.8, but < 0.9
    seq_a = "ACGTACGTACGTACGT"
    seq_b = "ACGTACGTACGTACAC"  # 2 mismatches of 16 -> identity 14/16 = 0.875
    rows = [
        _row("p1", "pmid_1", seq_a),
        _row("p2", "pmid_2", seq_b),
    ]
    _, summary = sl.compute_sequence_lineage_metrics(rows, thresholds=(70, 80, 90))
    flagged = {h["identity_coverage_threshold"]: h["flagged_pairs"] for h in summary["homology"]}
    # monotonic non-increasing
    vals = [flagged[t] for t in (70, 80, 90)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    # 2 pairs flagged at 70/80 (identity 0.9375 >= 0.7/0.8), none at 90
    assert flagged[70] == 2
    assert flagged[80] == 2
    assert flagged[90] == 0


def test_identical_sequence_flagged_at_all_thresholds():
    rows = [
        _row("p1", "pmid_1", "ACGTACGT"),
        _row("p2", "pmid_2", "ACGTACGT"),
    ]
    _, summary = sl.compute_sequence_lineage_metrics(rows)
    for h in summary["homology"]:
        assert h["flagged_pairs"] == 2


def test_no_leakage_flags_zero():
    rows = [
        _row("p1", "pmid_1", "ACGTACGT", parent="P1"),
        _row("p2", "pmid_2", "TTTTGGGG", parent="P2"),  # fully dissimilar, distinct parent
    ]
    _, summary = sl.compute_sequence_lineage_metrics(rows, thresholds=(70, 80, 90))
    for h in summary["homology"]:
        assert h["flagged_pairs"] == 0
    assert summary["exact_sequence_duplicate_pairs"] == 0
    assert summary["shared_wt_parent_pairs"] == 0