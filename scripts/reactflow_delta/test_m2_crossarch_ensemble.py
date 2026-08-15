#!/usr/bin/env python3
"""test_m2_crossarch_ensemble — unit tests for the cross-architecture ensemble report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_crossarch_ensemble_report as mce  # noqa: E402

W = 21
SEEDS = [0, 1, 2, 3, 4]


def _synth_rows(designs, variant, n_pairs=8, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for d in designs:
        for pi in range(n_pairs):
            pid = f"{d}:A{d}_{pi}"
            y = rng.normal(size=W).tolist()
            w = [1.0] * W
            prior = [0.0] * W
            if variant == "wmed_spectrum":
                rows.append({"pair_id": pid, "task": "magnitude_spectrum",
                             "fold_id": d, "seed": 0, "model_variant": "wmed_spectrum",
                             "model_id": "x", "publication_id": d,
                             "source_accession": d, "split_role": "development",
                             "endpoint_version": "m2", "caller_version": "m2_caller_v1",
                             "caller_mode": "PER_POS_ERROR", "y": y, "weight": w,
                             "raw_prediction": prior, "transformed_prediction": prior,
                             "coverage_status": "CALLED"})
            else:
                for s in SEEDS:
                    rows.append({"pair_id": pid, "task": "magnitude_spectrum",
                                 "fold_id": d, "seed": s, "model_variant": variant,
                                 "model_id": "x", "publication_id": d,
                                 "source_accession": d, "split_role": "development",
                                 "endpoint_version": "m2", "caller_version": "m2_caller_v1",
                                 "caller_mode": "PER_POS_ERROR", "y": y, "weight": w,
                                 "raw_prediction": [float(rng.normal(0, 0.1)) for _ in range(W)],
                                 "transformed_prediction": [0.0] * W,
                                 "coverage_status": "CALLED"})
    return rows


def test_unroll_builds_base_and_seeds(tmp_path):
    rows = _synth_rows(["D1", "D2"], "wmae_resid_attn_spectrum", n_pairs=4)
    base, model = mce._unroll(rows, "wmae_resid_attn_spectrum")
    # no baseline rows in this synthetic set -> base empty
    assert len(model) == 2 * 4  # 2 designs x 4 pairs
    pid = next(iter(model))
    assert set(model[pid].keys()) == set(SEEDS)


def test_analyze_matched_subset():
    designs = [f"D{i}" for i in range(6)]
    rows_pa = _synth_rows(designs, "wmae_resid_posaware_spectrum", n_pairs=5, rng_seed=1)
    rows_attn = _synth_rows(designs, "wmae_resid_attn_spectrum", n_pairs=5, rng_seed=2)
    rows_base = _synth_rows(designs, "wmed_spectrum", n_pairs=5, rng_seed=3)
    # merge baseline into both
    rows_pa += rows_base
    rows_attn += rows_base
    base_pa, model_pa = mce._unroll(rows_pa, "wmae_resid_posaware_spectrum")
    base_attn, model_attn = mce._unroll(rows_attn, "wmae_resid_attn_spectrum")
    common = [k for k in base_pa if len(model_pa.get(k, {})) == len(SEEDS)
              and len(model_attn.get(k, {})) == len(SEEDS)]
    assert len(common) == 6 * 5
    base = {k: base_pa[k] for k in common}
    ens_pa = {k: np.mean([model_pa[k][s] for s in SEEDS], axis=0) for k in common}
    ens_attn = {k: np.mean([model_attn[k][s] for s in SEEDS], axis=0) for k in common}
    ens = {k: 0.5 * ens_pa[k] + 0.5 * ens_attn[k] for k in common}
    r = mce.analyze(base, ens, common, n_perm=20, n_boot=20, perm_seed=7)
    assert r["n_designs"] == 6
    assert r["n_positions"] > 0
    assert r["skill"] is not None
    assert r["ci_low"] is not None and r["ci_high"] is not None


def test_alpha_limits_recover_components():
    """alpha=1.0 must give exactly the position-aware component skill."""
    designs = [f"D{i}" for i in range(5)]
    rows_pa = _synth_rows(designs, "wmae_resid_posaware_spectrum", n_pairs=4, rng_seed=1)
    rows_attn = _synth_rows(designs, "wmae_resid_attn_spectrum", n_pairs=4, rng_seed=2)
    rows_base = _synth_rows(designs, "wmed_spectrum", n_pairs=4, rng_seed=3)
    rows_pa += rows_base
    rows_attn += rows_base
    base_pa, model_pa = mce._unroll(rows_pa, "wmae_resid_posaware_spectrum")
    _, model_attn = mce._unroll(rows_attn, "wmae_resid_attn_spectrum")
    common = [k for k in base_pa if len(model_pa.get(k, {})) == len(SEEDS)
              and len(model_attn.get(k, {})) == len(SEEDS)]
    base = {k: base_pa[k] for k in common}
    ens_pa = {k: np.mean([model_pa[k][s] for s in SEEDS], axis=0) for k in common}
    ens_attn = {k: np.mean([model_attn[k][s] for s in SEEDS], axis=0) for k in common}
    ens_a1 = {k: 1.0 * ens_pa[k] + 0.0 * ens_attn[k] for k in common}
    r = mce.analyze(base, ens_a1, common, n_perm=20, n_boot=20, perm_seed=7)
    r_pa = mce.analyze(base, ens_pa, common, n_perm=20, n_boot=20, perm_seed=7)
    assert abs(r["skill"] - r_pa["skill"]) < 1e-12
