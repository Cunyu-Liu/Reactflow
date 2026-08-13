#!/usr/bin/env python3
"""Unit tests for m2_deviation_report — verifies baseline/model unrolling, that the
mu-ensemble deviation score (|pred-prior|) recovers a real deviation signal with a
significant design-block permutation p, that a null (random) model gives a large p,
and that per-position Spearman peaks at the central edit site."""
from __future__ import annotations

import json
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import m2_deviation_report as mdr

SEEDS = mdr.SEEDS
MODEL = mdr.MODEL
BASELINE = mdr.BASELINE


def _mk_rows(n_pairs=40, W=21, seed=0):
    """Synthetic rows: prior differs from y by a position-dependent deviation, and
    the model's |pred-prior| tracks the TRUE deviation |y-prior|."""
    rng = np.random.default_rng(seed)
    base_prior = rng.uniform(-0.4, 0.4, W)
    rows = []
    for i in range(n_pairs):
        pair = f"design_{i // 4}:mut_{i}"
        # true deviation varies by position: random per-position offsets
        true_dev = np.abs(rng.normal(0, 0.3, W))          # which positions deviate
        y = base_prior + np.sign(rng.normal(0, 1, W)) * true_dev
        w = np.ones(W)
        # baseline prior is close to base_prior (a bit noisy, no deviation signal)
        base_pred = base_prior + rng.normal(0, 0.05, W)
        rows.append({"pair_id": pair, "task": "magnitude_spectrum",
                     "seed": 0, "model_variant": BASELINE,
                     "coverage_status": "CALLED",
                     "y": y.tolist(), "weight": w.tolist(),
                     "raw_prediction": base_pred.tolist()})
        for s in SEEDS:
            # model predicts prior + sign * approx(true_dev): score ~ |y-prior|
            pred = base_prior + np.sign(rng.normal(0, 1, W)) * (0.8 * true_dev + rng.normal(0, 0.04, W))
            rows.append({"pair_id": pair, "task": "magnitude_spectrum",
                         "seed": s, "model_variant": MODEL,
                         "coverage_status": "CALLED",
                         "y": y.tolist(), "weight": w.tolist(),
                         "raw_prediction": pred.tolist()})
    return rows


def _tmpfile(tmp_path, rows):
    p = tmp_path / "pred.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(p)


def test_unroll_separates_baseline_and_model_seeds(tmp_path):
    rows = _mk_rows(n_pairs=8)
    base, model = mdr._unroll(rows)
    assert len(base) == 8
    assert all(len(model[k]) == len(SEEDS) for k in base)
    for k in base:
        assert base[k]["y"].shape[0] == 21
        assert "prior" in base[k]
        for s in SEEDS:
            assert model[k][s].shape[0] == 21


def test_unroll_accepts_alternative_model_variant(tmp_path):
    """--model-variant parameterization: an alternate variant (e.g. position-aware)
    must be unrolled as the model when requested, and ignored otherwise."""
    rows = _mk_rows(n_pairs=8)
    alt = "wmae_resid_posaware_spectrum"
    rows2 = []
    for r in rows:
        rows2.append(dict(r))
        if r["model_variant"] == MODEL:
            rows2[-1]["model_variant"] = alt
    base, model = mdr._unroll(rows2, alt)
    assert len(base) == 8
    assert all(len(model[k]) == len(SEEDS) for k in base)
    # default variant must find nothing (all rows moved to alt)
    base2, model2 = mdr._unroll(rows2)
    assert not model2 or all(len(model2[k]) == 0 for k in model2)


def test_analyze_detects_real_deviation_signal(tmp_path):
    rows = _mk_rows(n_pairs=120, seed=7)
    base, model = mdr._unroll(rows)
    common = [k for k in base if len(model[k]) == len(SEEDS)]
    ens = {k: np.mean([model[k][s] for s in SEEDS], axis=0) for k in common}
    rep = mdr.analyze(base, model, ens, n_perm=80, perm_seed=1)
    assert rep["n_designs"] > 3
    assert rep["spearman_abs"] > 0.3
    assert rep["auroc_abs"] > 0.7
    assert rep["permutation_p"] < 0.05
    assert rep["per_design"]["pct_positive"] > 0.7


def test_analyze_null_random_predictions_gives_high_p(tmp_path):
    """Under a true null (pred = prior + random delta, so score=|delta| is
    independent of adt=|y-prior|) the permutation p should be large."""
    rng = np.random.default_rng(11)
    base = {}
    model = {}
    W = 21
    for i in range(60):
        pair = f"d_{i // 3}:m_{i}"
        prior = rng.uniform(-0.3, 0.3, W)
        base[pair] = {"y": prior + rng.normal(0, 0.2, W),
                      "prior": prior}
        # residual model: pred = prior + delta, delta independent of y/prior
        delta = rng.normal(0, 0.15, W)
        model[pair] = {s: prior + delta + rng.normal(0, 0.05, W) for s in SEEDS}
    ens = {k: np.mean([model[k][s] for s in SEEDS], axis=0) for k in base}
    rep = mdr.analyze(base, model, ens, n_perm=80, perm_seed=2)
    # null model should not give a significant (small) permutation p at 5%
    assert rep["permutation_p"] > 0.05


def test_per_position_signal_peaks_at_center(tmp_path):
    """Per-position Spearman of the deviation score should peak at the central edit
    site when the deviation signal is strongest there."""
    W = 21
    rng = np.random.default_rng(5)
    base = {}
    model = {}
    center = W // 2
    for i in range(80):
        pair = f"d_{i // 4}:m_{i}"
        prior = np.zeros(W)
        y = np.zeros(W)
        for j in range(W):
            # true deviation is a bump peaked at the center, with per-pair noise so
            # adt has rank variance (real data is not degenerate per position)
            y[j] = 0.5 * np.exp(-((j - center) ** 2) / 8.0) + rng.normal(0, 0.05)
        base[pair] = {"y": y, "prior": prior}
        # model deviation score |pred-prior| tracks the bump best at center
        dev_score = 0.9 * y + rng.normal(0, 0.02, W)
        model[pair] = {s: np.where(y > 0.05, dev_score + rng.normal(0, 0.01, W),
                                   rng.normal(0, 0.01, W)) for s in SEEDS}
    ens = {k: np.mean([model[k][s] for s in SEEDS], axis=0) for k in base}
    pp = mdr.per_position(base, model, ens, list(base), W=W)
    rhos = {p["position"]: p["spearman_abs"] for p in pp if p["spearman_abs"] == p["spearman_abs"]}
    assert max(rhos.values()) == rhos[center]
    assert rhos[center] > 0.7


def test_loo_robustness_signal_survives_exclusion(tmp_path):
    """The deviation signal must survive removing any single design."""
    rows = _mk_rows(n_pairs=160, seed=9)
    base, model = mdr._unroll(rows)
    common = [k for k in base if len(model[k]) == len(SEEDS)]
    ens = {k: np.mean([model[k][s] for s in SEEDS], axis=0) for k in common}
    rep = mdr.analyze(base, model, ens, n_perm=40, perm_seed=3)
    rob = rep["robustness"]
    assert rob["pooled_rho_min_over_loo"] is not None
    # pooled rho stays positive under every single-design removal
    assert rob["pooled_rho_min_over_loo"] > 0.3
    # excluding the strongest-rho design keeps the signal significant
    assert rob["exclude_strongest"]["spearman_abs"] > 0.3
    assert rob["exclude_strongest"]["permutation_p"] < 0.05


def test_main_writes_report(tmp_path):
    rows = _mk_rows(n_pairs=60, seed=2)
    pred = _tmpfile(tmp_path, rows)
    out = tmp_path / "out"
    import m2_deviation_report
    sys.argv = ["m2_deviation_report", "--pred", pred, "--out", str(out),
                "--n-perm", "30", "--perm-seed", "20260812"]
    m2_deviation_report.main()
    rep = json.loads((out / "m2_deviation_report.json").read_text(encoding="utf-8"))
    assert rep["mu_ensemble"]["n_designs"] > 3
    assert rep["mu_ensemble"]["permutation_p"] is not None
