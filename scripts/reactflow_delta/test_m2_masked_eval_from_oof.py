#!/usr/bin/env python3
"""test_m2_masked_eval_from_oof.py — tests for the OOF-based masked evaluator."""
import json
from pathlib import Path

import numpy as np

import m2_masked_eval_from_oof as me


def _make_oof(tmp_path):
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.normal(0.0, 1.0, n)
    w = np.ones(n)
    keys = np.array([f"d{i % 20}" for i in range(n)])
    const = float(np.median(y))
    # 950 matched rows: deep/prior real; 50 unmatched: median placeholder
    deep = rng.normal(0.2, 1.0, n)
    prior = rng.normal(0.1, 1.0, n)
    deep[950:] = const
    prior[950:] = const
    gbdt = rng.normal(0.3, 1.0, n)
    blend = 0.5 * gbdt + 0.5 * deep
    p = Path(tmp_path) / "oof.npz"
    np.savez(p, gbdt=gbdt, deep=deep, prior=prior, blend=blend, y=y, w=w, keys=keys)
    return str(p)


def test_mask_detects_placeholder(tmp_path):
    p = _make_oof(tmp_path)
    z = np.load(p, allow_pickle=True)
    matched = me._mask_from_placeholder(z["deep"], z["prior"], z["y"])
    assert int(matched.sum()) == 950
    assert int((~matched).sum()) == 50


def test_masked_eval_runs(tmp_path):
    p = _make_oof(tmp_path)
    out = str(tmp_path / "out")
    rc = me.main.__wrapped__ if hasattr(me.main, "__wrapped__") else None
    import argparse
    # run via a small driver mimicking main()
    z = np.load(p, allow_pickle=True)
    y = z["y"]; w = z["w"]; keys = z["keys"]
    gbdt_all = z["gbdt"]; deep_all = z["deep"]; prior_all = z["prior"]
    matched = me._mask_from_placeholder(deep_all, prior_all, y)
    ym, wm = y[matched], w[matched]
    gbdt_m, deep_m, prior_m = gbdt_all[matched], deep_all[matched], prior_all[matched]
    keys_m = keys[matched]
    mae_prior = me._wmae(ym, wm, prior_m)
    mae_gbdt = me._wmae(ym, wm, gbdt_m)
    mae_attn = me._wmae(ym, wm, deep_m)
    assert me._skill(mae_gbdt, mae_prior) is not None
    assert np.isfinite(me._skill(mae_attn, mae_prior))
    assert len(set(keys_m.tolist())) == 20


def test_analyze_block_structure(tmp_path):
    p = _make_oof(tmp_path)
    z = np.load(p, allow_pickle=True)
    y = z["y"]; w = z["w"]; keys = z["keys"]
    matched = me._mask_from_placeholder(z["deep"], z["prior"], y)
    ym, wm = y[matched], w[matched]
    keys_m = keys[matched]
    pred = ym * 0.9
    base = z["prior"][matched]
    sig = me.analyze(ym, wm, keys_m, pred, base, n_perm=30, n_boot=30, seed=1)
    assert sig["n_designs"] == 20
    assert 0.0 <= sig["permutation_p"] <= 1.0
    assert sig["ci_low"] <= sig["ci_high"]
