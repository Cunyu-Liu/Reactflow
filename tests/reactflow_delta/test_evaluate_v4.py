#!/usr/bin/env python3
"""Unit tests for evaluate_v4 (endpoint_v4 non-degenerate macro AUPRC)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "scripts/reactflow_delta"))

import evaluate_v4 as v4
from evaluate_v4 import (
    UNIDENTIFIABLE,
    publication_macro_auprc_non_degenerate,
    non_degenerate_publications,
    per_publication_ap,
    permutation_test_non_degenerate,
    bootstrap_ci_non_degenerate,
    paired_bootstrap_delta_ci,
    is_unidentifiable,
)


def _mk(pubs, labels, scores):
    return list(pubs), [int(l) for l in labels], [float(s) for s in scores]


def test_mixed_only_numeric():
    pubs, labs, scos = _mk(
        ["A", "A", "A", "B", "B", "B"],
        [1, 0, 1, 0, 1, 1],
        [0.9, 0.1, 0.8, 0.7, 0.6, 0.2],
    )
    stat, nondeg, deg = publication_macro_auprc_non_degenerate(pubs, labs, scos)
    assert not is_unidentifiable(stat)
    assert isinstance(stat, float)
    assert nondeg == ["A", "B"]
    assert deg == []


def test_constant_pub_excluded_not_blocking():
    # publication C is all-positive (constant) -> excluded; macro over A,B numeric
    pubs, labs, scos = _mk(
        ["A", "A", "B", "B", "C", "C"],
        [1, 0, 0, 1, 1, 1],
        [0.9, 0.1, 0.3, 0.7, 0.5, 0.5],
    )
    stat, nondeg, deg = publication_macro_auprc_non_degenerate(pubs, labs, scos)
    assert not is_unidentifiable(stat)
    assert isinstance(stat, float)
    assert nondeg == ["A", "B"]
    assert deg == ["C"]


def test_all_constant_unidentifiable():
    pubs, labs, scos = _mk(
        ["A", "A", "B", "B"],
        [1, 1, 0, 0],
        [0.9, 0.8, 0.2, 0.1],
    )
    stat, nondeg, deg = publication_macro_auprc_non_degenerate(pubs, labs, scos)
    assert is_unidentifiable(stat)
    assert len(nondeg) == 0
    assert sorted(deg) == ["A", "B"]


def test_non_degenerate_publications_partition():
    pubs, labs, _ = _mk(
        ["A", "A", "B", "B", "C", "C"],
        [1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0],
    )
    nondeg, deg = non_degenerate_publications(pubs, labs)
    assert nondeg == ["A"]
    assert sorted(deg) == ["B", "C"]


def test_per_publication_ap_none_for_constant():
    pubs, labs, scos = _mk(
        ["A", "A", "B", "B"],
        [1, 0, 1, 1],
        [0.9, 0.1, 0.8, 0.2],
    )
    ap = per_publication_ap(pubs, labs, scos)
    assert ap["A"] is not None
    assert ap["B"] is None


def test_tied_ap_row_order_invariant():
    # all tied scores: AP must be identical regardless of row order
    rows = [
        (["A", "A", "A"], [1, 0, 1], [0.5, 0.5, 0.5]),
        (["A", "A", "A"], [0, 1, 1], [0.5, 0.5, 0.5]),
    ]
    stats = []
    for pubs, labs, scos in rows:
        s, _, _ = publication_macro_auprc_non_degenerate(pubs, labs, scos)
        stats.append(s)
    assert stats[0] == stats[1]
    assert isinstance(stats[0], float)


def test_macro_matches_endpoint_v3_when_no_degenerate():
    # with no constant pubs, v4 non-degenerate macro == v2 macro
    import evaluate_v2 as e2
    pubs, labs, scos = _mk(
        ["A", "A", "A", "B", "B", "B"],
        [1, 0, 1, 0, 1, 0],
        [0.9, 0.1, 0.8, 0.7, 0.6, 0.2],
    )
    v4_stat, _, _ = publication_macro_auprc_non_degenerate(pubs, labs, scos)
    v2_stat = e2.publication_macro_auprc(pubs, labs, scos)
    assert abs(float(v4_stat) - float(v2_stat)) < 1e-12


def test_permutation_deterministic_and_bound():
    rng = np.random.RandomState(0)
    pubs = []
    labs = []
    scos = []
    for p in range(8):
        for _ in range(40):
            pubs.append(f"pub_{p}")
            labs.append(int(rng.rand() > 0.4))
            scos.append(float(rng.rand()))
    r1 = permutation_test_non_degenerate(pubs, labs, scos, seed=7, n_perm=200)
    r2 = permutation_test_non_degenerate(pubs, labs, scos, seed=7, n_perm=200)
    assert r1["p_value"] == r2["p_value"]
    assert r1["statistic"] == r2["statistic"]
    assert 0.0 <= r1["p_value"] <= 1.0
    assert r1["b"] >= 0
    assert len(r1["non_degenerate"]) == 8
    assert r1["degenerate"] == []


def test_bootstrap_ci_none_when_lt3_nondeg():
    pubs, labs, scos = _mk(
        ["A", "A", "B", "B"],
        [1, 0, 0, 1],
        [0.9, 0.1, 0.2, 0.8],
    )
    res = bootstrap_ci_non_degenerate(pubs, labs, scos, n_boot=100)
    assert res["ci_low"] is None
    assert res["ci_high"] is None
    assert "PUBLICATION_LT_3" in res.get("note", "")


def test_paired_bootstrap_delta_positive_when_model_better():
    # 3 non-degenerate pubs; model scores rank positives higher than baseline
    pubs = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    labs = ([1, 1, 0, 0, 1, 0] * 2) + ([1, 0, 0, 1, 0, 1] * 2)
    model_s = ([0.99, 0.95, 0.3, 0.2, 0.97, 0.1] * 2) + ([0.98, 0.2, 0.3, 0.9, 0.1, 0.8] * 2)
    base_s = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] * 3
    res = paired_bootstrap_delta_ci(pubs, labs, model_s, base_s, seed=1, n_boot=200)
    assert res["ci_low"] is not None
    assert res["ci_low"] > 0.0


def test_all_constant_permutation_unidentifiable():
    pubs, labs, scos = _mk(
        ["A", "A", "B", "B"],
        [1, 1, 1, 1],
        [0.9, 0.8, 0.2, 0.1],
    )
    r = permutation_test_non_degenerate(pubs, labs, scos, n_perm=50)
    assert is_unidentifiable(r["statistic"])
    assert r["p_value"] is None
