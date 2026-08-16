#!/usr/bin/env python3
"""test_run_response_spectrum_m2_seq_v1 — unit tests for the M2 residual-MLP
+ global-sequence full-spectrum runner.

Covers the fold-invariant feature-assembly helper (build_seq_features): local-window
+ pair-tail dimension, global-seq dimension/dedup, and that the concatenated feature
feeds the residual MLP (reducing train MAE vs prior without delta explosion) on CPU.
Also verifies the fail-closed N>=100 guard.

The full 159-fold LOO requires CUDA + the real M2 CSV (not run here); this test
verifies the deterministic, unit-testable feature/model plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reactflow_delta"))
import m2_data_v1 as m2d  # noqa: E402
import residual_spectrum_v2 as rsm  # noqa: E402
import global_seq_features_v1 as gsf  # noqa: E402
import run_response_spectrum_m2_seq_v1 as rms  # noqa: E402
from run_baselines_v6 import WINDOW  # noqa: E402


def _mk_samples(n_designs=2, n_mut=3, seq_len=40, sub_start=11, sub_end=25):
    """Build synthetic M2 samples (designs x mutants)."""
    samples = []
    for di in range(n_designs):
        seq = list("ACGU" * (seq_len // 4) + "ACGU"[:seq_len % 4])
        design = {
            "puzzle": f"P{di:02d}", "method": "M1",
            "source_accession": f"OK7a_M2_P{di:02d}_M1",
            "sequence": "".join(seq), "sub_start": sub_start, "sub_end": sub_end,
            "wt_reactivity": [1.0] * seq_len, "wt_error": [0.1] * seq_len,
            "mutants": [], "usable": True,
        }
        for p in range(1, n_mut + 1):
            pos = sub_start - 1 + (p - 1)
            mut = list(design["sequence"])
            b = mut[pos]
            alt = "C" if b == "A" else "A"
            mut[pos] = alt
            design["mutants"].append({
                "mutA": p, "edit_seq_pos": pos,
                "sequence": "".join(mut),
                "reactivity": [1.2] * seq_len, "error": [0.1] * seq_len,
            })
        samples.extend(m2d.build_samples(design))
    return samples


def test_build_seq_features_dimensions_and_keys():
    samples = _mk_samples(n_designs=2, n_mut=3)
    fx = rms.build_seq_features(samples)
    assert len(fx) == len(samples)
    pid = next(iter(fx))
    # feature = local window + pair tail | global sequence
    assert fx[pid].shape[0] > WINDOW * 7           # local part (>= window*POS_DIM)
    assert fx[pid].shape[0] >= gsf.GLOBAL_SEQ_DIM
    # two mutants of the same design must share the global part exactly
    pids = list(fx.keys())
    assert np.allclose(fx[pids[0]][-gsf.GLOBAL_SEQ_DIM:], fx[pids[1]][-gsf.GLOBAL_SEQ_DIM:])


def test_global_seq_features_fold_invariant_for_same_wt():
    # two mutants of the SAME design (same WT seq) must share the same global-seq
    # suffix, while the local feature may differ (ref/alt alleles differ).
    samples = _mk_samples(n_designs=1, n_mut=2)
    fx = rms.build_seq_features(samples)
    pids = list(fx.keys())
    assert np.allclose(fx[pids[0]][-gsf.GLOBAL_SEQ_DIM:], fx[pids[1]][-gsf.GLOBAL_SEQ_DIM:])
    assert not np.allclose(fx[pids[0]], fx[pids[1]])  # local parts differ


def test_residual_mlp_trains_on_seq_features():
    samples = _mk_samples(n_designs=6, n_mut=5)
    fx = rms.build_seq_features(samples)
    pids = list(fx.keys())
    X = np.stack([fx[p] for p in pids]).astype(np.float32)
    n = len(pids)
    prior_true = np.linspace(-0.4, 0.4, WINDOW).astype(np.float32)
    Y = np.tile(prior_true, (n, 1)).astype(np.float32)
    Y += np.linspace(0.0, 0.3, n)[:, None].astype(np.float32)
    Wm = np.ones((n, WINDOW), dtype=np.float32)
    prior, _ = rsm.per_position_prior(Y, Wm)
    device = torch.device("cpu")
    model, log = rsm.train_residual(
        X, Y, Wm, prior, epochs=40, bs=8, lr=1e-3,
        resid_pen=1e-3, hidden=16, seed=0, device=device)
    fin = log["final"]
    assert fin["mae_model_train"] <= fin["mae_prior_train"]
    assert fin["delta_abs_mean"] < 1.0
    assert len(log["learning_curve"]) >= 30


def test_main_guard_small_n_in_source():
    """The runner must STOP (fail-closed) if usable designs < 100."""
    src = Path(rms.__file__).read_text(encoding="utf-8")
    assert "len(resolved) < 100" in src
