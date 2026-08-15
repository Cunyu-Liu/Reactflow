#!/usr/bin/env python3
"""test_m2_three_way_ensemble — unit tests for the 3-way cross-architecture
ensemble report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_three_way_ensemble_report as mte  # noqa: E402

W = 21
SEEDS = [0, 1, 2, 3, 4]


def _synth_rows(designs, variant, n_pairs=6, rng_seed=0):
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


def _build_common(designs):
    rows_v3 = _synth_rows(designs, "wmae_resid_posaware_spectrum", rng_seed=1)
    rows_v4 = _synth_rows(designs, "wmae_resid_attn_spectrum", rng_seed=2)
    rows_v5 = _synth_rows(designs, "wmae_resid_attn_spectrum", rng_seed=3)
    rows_base = _synth_rows(designs, "wmed_spectrum", rng_seed=4)
    rows_v3 += rows_base
    rows_v4 += rows_base
    rows_v5 += rows_base
    b3, m3 = mte._unroll(rows_v3, "wmae_resid_posaware_spectrum")
    b4, m4 = mte._unroll(rows_v4, "wmae_resid_attn_spectrum")
    b5, m5 = mte._unroll(rows_v5, "wmae_resid_attn_spectrum")
    common = [k for k in b3 if len(m3.get(k, {})) == len(SEEDS)
              and len(m4.get(k, {})) == len(SEEDS)
              and len(m5.get(k, {})) == len(SEEDS)]
    base = {k: b3[k] for k in common}
    e3 = {k: np.mean([m3[k][s] for s in SEEDS], axis=0) for k in common}
    e4 = {k: np.mean([m4[k][s] for s in SEEDS], axis=0) for k in common}
    e5 = {k: np.mean([m5[k][s] for s in SEEDS], axis=0) for k in common}
    return base, common, e3, e4, e5


def test_three_way_ensemble_matches_component_when_isolated():
    designs = [f"D{i}" for i in range(5)]
    base, common, e3, e4, e5 = _build_common(designs)
    # weights (1,0,0) must recover v3 exactly
    ens = {k: 1.0 * e3[k] + 0.0 * e4[k] + 0.0 * e5[k] for k in common}
    r = mte.analyze(base, ens, common, n_perm=20, n_boot=20, perm_seed=7)
    r3 = mte.analyze(base, e3, common, n_perm=20, n_boot=20, perm_seed=7)
    assert abs(r["skill"] - r3["skill"]) < 1e-12
    # weights (0,0,1) must recover v5 exactly
    ens5 = {k: 0.0 * e3[k] + 0.0 * e4[k] + 1.0 * e5[k] for k in common}
    r5 = mte.analyze(base, ens5, common, n_perm=20, n_boot=20, perm_seed=7)
    r5c = mte.analyze(base, e5, common, n_perm=20, n_boot=20, perm_seed=7)
    assert abs(r5["skill"] - r5c["skill"]) < 1e-12


def test_three_way_ensemble_shape_and_finite():
    designs = [f"D{i}" for i in range(6)]
    base, common, e3, e4, e5 = _build_common(designs)
    ens = {k: 0.15 * e3[k] + 0.2 * e4[k] + 0.65 * e5[k] for k in common}
    r = mte.analyze(base, ens, common, n_perm=20, n_boot=20, perm_seed=9)
    assert r["n_designs"] == 6
    assert r["n_positions"] > 0
    assert r["skill"] is not None
    assert np.isfinite(r["skill"])
    assert r["ci_low"] is not None and r["ci_high"] is not None
    assert r["permutation_p"] > 0.0
