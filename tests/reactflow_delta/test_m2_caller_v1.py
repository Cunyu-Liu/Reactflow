#!/usr/bin/env python3
"""Tests for m2_caller_v1 — per-position-error changer caller (M2, no replicates)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import m2_caller_v1 as m2c


def test_per_position_z_formula():
    wt = [1.0, 2.0, 3.0]
    mut = [1.4, 2.0, 4.0]
    werr = [0.2, 0.2, 0.2]
    merr = [0.2, 0.2, 0.2]
    mask = [1, 1, 1]
    z = m2c.per_position_z(wt, mut, werr, merr, mask)
    # z0 = 0.4/sqrt(0.04+0.04)=0.4/0.2828=1.414 ; z1=0 ; z2=1/0.2828=3.535
    assert z[0] == pytest.approx(0.4 / math.sqrt(0.08), rel=1e-9)
    assert z[1] == pytest.approx(0.0, abs=1e-12)
    assert z[2] == pytest.approx(1.0 / math.sqrt(0.08), rel=1e-9)


def test_per_position_z_masked_and_nonfinite():
    wt = [1.0, float("nan"), 3.0]
    mut = [2.0, 3.0, 4.0]
    werr = [0.1, 0.1, 0.1]
    merr = [0.1, 0.1, 0.1]
    mask = [0, 1, 1]
    z = m2c.per_position_z(wt, mut, werr, merr, mask)
    assert z[0] is None          # masked
    assert z[1] is None          # non-finite wt
    assert z[2] is not None


def test_max_cluster_rms_single_peak():
    # all eligible; a strong single z at one position -> RMS of that window
    n = 21
    mask = [1] * n
    z = [0.0] * n
    z[10] = 4.0
    # max over windows: a 1-position window at 10 gives RMS=4.0 (larger than diluted)
    stat = m2c.max_cluster_rms(z, mask, cluster_window=5)
    assert stat == pytest.approx(4.0, rel=1e-9)


def test_max_cluster_rms_ignores_excluded():
    n = 9
    mask = [0, 1, 1, 0, 0, 0, 1, 1, 1]
    z = [99.0, 1.0, 2.0, 99.0, 99.0, 99.0, 3.0, 4.0, 5.0]
    # eligible positions: idx 1,2 and 6,7,8. Big 99s are masked out.
    stat = m2c.max_cluster_rms(z, mask, cluster_window=5)
    # max window = single position idx 8 with z=5 -> RMS=5.0 (99s excluded)
    assert stat == pytest.approx(5.0, rel=1e-9)


def test_gaussian_null_deterministic():
    mask = [1] * 50
    n1 = m2c.gaussian_null(mask, cluster_window=5, n_null=300, seed=123)
    n2 = m2c.gaussian_null(mask, cluster_window=5, n_null=300, seed=123)
    assert np.array_equal(n1, n2)
    # null stats are positive and roughly centered around expected max RMS
    assert np.all(n1 > 0)


def test_gaussian_null_respects_mask_sparsity():
    dense = m2c.gaussian_null([1] * 100, n_null=300, seed=1)
    sparse = m2c.gaussian_null([1] * 50 + [0] * 50, n_null=300, seed=1)
    # fewer eligible positions -> smaller max RMS in expectation
    assert np.median(sparse) < np.median(dense)


def test_call_mutant_changer_and_nonchanger():
    n = 30
    wt = [1.0] * n
    werr = [0.1] * n
    merr = [0.1] * n
    mask = [1] * n
    # a large, spatially coherent response -> changer
    mut_strong = [3.0] * n
    r_strong = m2c.call_mutant("s1", wt, mut_strong, werr, merr, mask, n_null=500, seed=7)
    assert r_strong.label == "1"
    # a tiny response -> nonchanger
    mut_weak = [1.05] * n
    r_weak = m2c.call_mutant("s2", wt, mut_weak, werr, merr, mask, n_null=500, seed=7)
    assert r_weak.label == "0"


def test_call_mutant_cached_null_matches():
    n = 30
    wt = [1.0] * n
    werr = [0.1] * n
    merr = [0.1] * n
    mask = [1] * n
    mut = [3.0] * n
    null = m2c.gaussian_null(mask, n_null=400, seed=9)
    a = m2c.call_mutant("a", wt, mut, werr, merr, mask, n_null=400, seed=9)
    b = m2c.call_mutant("b", wt, mut, werr, merr, mask, n_null=400, seed=9, null=null)
    assert a.label == b.label
    assert a.statistic == pytest.approx(b.statistic, rel=1e-12)
