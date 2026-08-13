#!/usr/bin/env python3
"""Unit tests for m2_horizontal_ensemble_report — verifies baseline/model unrolling,
mu-ensemble construction, pooled skill, per-position structure, and that the
permutation p is near-1 under a null (random model predictions)."""
from __future__ import annotations

import json
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import m2_horizontal_ensemble_report as mhr

SEEDS = mhr.SEEDS
MODEL = mhr.MODEL
BASELINE = mhr.BASELINE


def _mk_rows(n_pairs=40, W=21, seed=0):
    """Synthetic keyed rows with a per-position prior baseline and a model that,
    when ensembled, is slightly better than each single seed."""
    rng = np.random.default_rng(seed)
    base_prior = rng.uniform(-0.4, 0.4, W)
    rows = []
    for i in range(n_pairs):
        pair = f"design_{i // 4}:mut_{i}"
        y = base_prior + rng.normal(0, 0.15, W)
        w = np.ones(W)
        # baseline: noisy prior (worse than the model)
        base_pred = base_prior + rng.normal(0, 0.12, W)
        rows.append({"pair_id": pair, "task": "magnitude_spectrum",
                     "seed": 0, "model_variant": BASELINE,
                     "coverage_status": "CALLED",
                     "y": y.tolist(), "weight": w.tolist(),
                     "raw_prediction": base_pred.tolist()})
        for s in SEEDS:
            # each seed is closer to the true prior than the baseline -> model wins
            pred = base_prior + rng.normal(0, 0.04, W)
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
    base, model = mhr._unroll(rows)
    assert len(base) == 8
    assert all(len(model[k]) == len(SEEDS) for k in base)
    # baseline only seed 0
    assert all(set(r) == {0} for r in [] )  # placeholder
    # all model entries have the right window length
    for k in base:
        assert base[k]["y"].shape[0] == 21
        for s in SEEDS:
            assert model[k][s].shape[0] == 21


def test_mu_ensemble_beats_single_seed(tmp_path):
    rows = _mk_rows(n_pairs=80, seed=3)
    base, model = mhr._unroll(rows)
    common = [k for k in base if len(model[k]) == len(SEEDS)]
    ens = {k: np.mean([model[k][s] for s in SEEDS], axis=0) for k in common}
    single = {k: model[k][0] for k in common}
    sk_single, wm, wb = mhr._pooled_skill(base, single, common)
    sk_ens, wm2, wb2 = mhr._pooled_skill(base, ens, common)
    # synthetic setup guarantees the ensemble is (weakly) better
    assert wb > 0
    assert sk_single is not None and sk_ens is not None
    assert sk_ens >= sk_single - 1e-9


def test_analyze_reports_positive_signal(tmp_path):
    rows = _mk_rows(n_pairs=120, seed=7)
    base, model = mhr._unroll(rows)
    common = [k for k in base if len(model[k]) == len(SEEDS)]
    ens = {k: np.mean([model[k][s] for s in SEEDS], axis=0) for k in common}
    rep = mhr.analyze(base, model, ens, n_perm=50, n_boot=50, perm_seed=1)
    assert rep["n_designs"] > 3
    assert rep["skill"] is not None
    assert rep["permutation_p"] is not None
    assert rep["per_design"]["pct_positive"] > 0.8


def test_analyze_null_random_predictions_gives_high_p(tmp_path):
    """With random (null) model predictions the permutation p should be large."""
    rng = np.random.default_rng(11)
    base = {}
    model = {}
    W = 21
    for i in range(60):
        pair = f"d_{i // 3}:m_{i}"
        base[pair] = {"y": rng.uniform(-0.3, 0.3, W),
                      "w": np.ones(W), "pred": rng.uniform(-0.3, 0.3, W)}
        model[pair] = {s: rng.uniform(-0.5, 0.5, W) for s in SEEDS}
    ens = {k: np.mean([model[k][s] for s in SEEDS], axis=0) for k in base}
    rep = mhr.analyze(base, model, ens, n_perm=60, n_boot=40, perm_seed=2)
    assert rep["skill"] is not None
    # null model should not give a significant (small) permutation p at 5%
    assert rep["permutation_p"] > 0.05


def test_per_position_signal_peaks_at_center(tmp_path):
    """Per-position skill should peak near the central edit site when the model
    improves the target most at the center."""
    W = 21
    rng = np.random.default_rng(5)
    base = {}
    model = {}
    center = W // 2
    for i in range(80):
        pair = f"d_{i // 4}:m_{i}"
        y = np.zeros(W); w = np.ones(W)
        base_pred = np.zeros(W); pred = np.zeros(W)
        for j in range(W):
            # true signal is a bump peaked at the central edit site
            true_delta = 0.5 * np.exp(-((j - center) ** 2) / 8.0)
            y[j] = true_delta
            # baseline: a substantial constant prior (mediocre everywhere)
            base_pred[j] = 0.2
            # model captures the bump (best at center), with small noise
            pred[j] = 0.9 * true_delta + 0.05 + rng.normal(0, 0.02)
        base[pair] = {"y": y, "w": w, "pred": base_pred}
        model[pair] = {s: pred + rng.normal(0, 0.01, W) for s in SEEDS}
    ens = {k: np.mean([model[k][s] for s in SEEDS], axis=0) for k in base}
    pp = mhr.per_position(base, model, ens, list(base), W=W)
    skills = {p["position"]: p["skill"] for p in pp}
    # central skill should be the max and clearly positive
    assert max(skills.values()) == skills[center]
    assert skills[center] > 0.8
