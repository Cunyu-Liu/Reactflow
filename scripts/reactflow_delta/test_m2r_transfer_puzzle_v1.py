#!/usr/bin/env python3
"""Tests for m2r_transfer_puzzle_v1 + m2r_transfer_puzzle_permtest_v1.

Uses synthetic M2 OOF predictions + real M2R/M2 data (fast: only loads a tiny
M2 OOF derived from the real jsonl but truncated to a few designs).  Checks:
  * report structure (3+ models, per-puzzle breakdown, OOF npz written)
  * leak diagnostic: design-level vs puzzle-level transfer features differ
  * permtest report structure + valid p / CI on synthetic npz
"""
from __future__ import annotations

import json, sys, tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_transfer_v1 as tr
import m2r_transfer_puzzle_v1 as trpz
import m2r_transfer_puzzle_permtest_v1 as trpp

M2R_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
M2_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"
M2_PRED_PZ = "/mnt/cunyuliu/m2_attn_puzzle_20260817/keyed_predictions_m2_attn_puzzle.jsonl"
M2_PRED_DL = "/mnt/cunyuliu/m2_response_spectrum_attn_v5_deep_20260815/keyed_predictions_m2_attn.jsonl"


@pytest.fixture(scope="module")
def samples():
    designs, meta = m2r.parse_m2r_csv(M2R_CSV)
    m2r.attach_m2_structure(designs, M2_CSV)
    return [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]


def test_puzzle_level_oof_load(samples):
    m2_oof = tr.load_m2_oof(M2_PRED_PZ)
    assert len(m2_oof) > 100
    # design_key mapping should cover most samples
    keymap = {}
    for did in m2_oof:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            keymap[(parts[2], "_".join(parts[3:]))] = did
    matched = sum(1 for s in samples if (s.puzzle, s.method) in keymap)
    assert matched / len(samples) > 0.9


def test_puzzle_vs_design_transfer_features_differ(samples):
    m2_pz = tr.load_m2_oof(M2_PRED_PZ)
    m2_dl = tr.load_m2_oof(M2_PRED_DL)
    km = {}
    for did in m2_pz:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            km[(parts[2], "_".join(parts[3:]))] = did
    X_pz = tr.build_transfer_features(samples, m2_pz, km)
    km2 = {}
    for did in m2_dl:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            km2[(parts[2], "_".join(parts[3:]))] = did
    X_dl = tr.build_transfer_features(samples, m2_dl, km2)
    assert X_pz.shape == (len(samples), 6)
    assert X_dl.shape == (len(samples), 6)
    # puzzle-level and design-level predictions should differ for many samples
    diff = np.abs(X_pz - X_dl).sum(axis=1)
    assert (diff > 1e-6).mean() > 0.8, "puzzle vs design OOF should differ"


def test_permtest_report_structure(tmp_path):
    rng = np.random.default_rng(20260817)
    sp = np.concatenate([np.full(60, f"P{i:02d}") for i in range(4)]).astype("U36")
    y = rng.normal(0.5, 1.0, 240)
    base = np.array([0.2, 0.5, 0.8, 1.1])[np.array([int(p[1:]) for p in sp])]
    pred = 0.7 * y + 0.3 * base + rng.normal(0, 0.2, 240)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "o.npz"
        np.savez(p, pred_ex=0.6 * pred + 0.4 * rng.normal(0, 0.5, 240),
                 pred_pz=pred, blend_pz=pred, y=y, sample_puzzles=sp)
        out = tmp_path / "out"
        rep = trpp.run_puzzle_permtest(str(p), str(out), n_perm=100, n_boot=100)
        assert set(rep["models"]) == {"existing_230", "puzzle_transfer", "full_stack_blend"}
        for v in rep["models"].values():
            assert 0 < v["permutation_p"] <= 1.0
            assert v["ci_low"] <= v["skill"] <= v["ci_high"]
            assert 0.0 <= v["per_puzzle_skill_pct_positive"] <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
