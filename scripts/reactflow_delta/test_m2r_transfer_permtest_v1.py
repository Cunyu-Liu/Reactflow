#!/usr/bin/env python3
"""Tests for m2r_transfer_permtest_v1 (full-stack permutation test).

The permtest operates on SAVED OOF predictions, so the test needs a tiny
synthetic npz with the same keys as m2r_transfer_oof.npz.  It checks:
  * report contains all three models (existing / +transfer / full blend)
  * full-blend skill is highest (sanity: blend >= single in the real run)
  * permutation p is in (0, 1]
  * bootstrap CI is a valid interval
  * per-design pct positive and LOO-exclusion range are present
"""
from __future__ import annotations

import json, sys, tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_transfer_permtest_v1 as perm


@pytest.fixture()
def tiny_npz():
    """Synthetic 3-design dataset; predictions have a real signal so the
    permutation test and CI are well-defined."""
    rng = np.random.default_rng(20260816)
    keys = np.concatenate([
        np.full(40, f"D{i}") for i in range(3)]).astype("U36")  # match real <U36
    y = rng.normal(0.5, 1.0, size=120)
    # predictions correlated with y within each design (real signal)
    base = np.array([0.3, 0.6, 0.9])[np.array([int(k[1:]) for k in keys])]
    pred = 0.7 * y + 0.3 * base + rng.normal(0, 0.2, size=120)
    pred_ex = 0.6 * pred + 0.4 * rng.normal(0, 0.5, size=120)
    blend = pred  # full blend = the strong pred
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "o.npz"
        np.savez(p, pred_ex=pred_ex, pred_comb=pred, ridge_comb=pred_ex,
                 blend_comb=blend, y=y, keys=keys)
        yield str(p)


def test_permtest_report_structure(tiny_npz, tmp_path):
    out = tmp_path / "out"
    report = perm.run_permtest(tiny_npz, str(out), n_perm=50, n_boot=50)
    report = json.loads((out / "m2r_transfer_permtest.json").read_text(encoding="utf-8"))
    assert set(report["models"]) == {
        "existing_230_gbdt", "plus_transfer_gbdt", "full_stack_blend"}
    assert report["n_designs"] == 3
    assert report["baseline_mae"] > 0


@pytest.mark.parametrize("key", ["existing_230_gbdt", "plus_transfer_gbdt", "full_stack_blend"])
def test_permtest_fields(tiny_npz, tmp_path, key):
    out = tmp_path / "out"
    perm.run_permtest(tiny_npz, str(out), n_perm=50, n_boot=50)
    report = json.loads((out / "m2r_transfer_permtest.json").read_text(encoding="utf-8"))
    v = report["models"][key]
    assert 0 < v["permutation_p"] <= 1.0
    assert v["ci_low"] <= v["skill"] <= v["ci_high"]
    assert 0.0 <= v["per_design_skill_pct_positive"] <= 1.0
    assert v["loo_exclusion_min"] <= v["loo_exclusion_max"]


def test_full_blend_best(tiny_npz, tmp_path):
    """In this synthetic set the full blend (strong pred) must beat the
    degraded existing model, so the monotonic ordering is verified."""
    out = tmp_path / "out"
    perm.run_permtest(tiny_npz, str(out), n_perm=50, n_boot=50)
    report = json.loads((out / "m2r_transfer_permtest.json").read_text(encoding="utf-8"))
    s_ex = report["models"]["existing_230_gbdt"]["skill"]
    s_bl = report["models"]["full_stack_blend"]["skill"]
    assert s_bl > s_ex


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
