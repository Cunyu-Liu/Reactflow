#!/usr/bin/env python3
"""test_run_m2_attn_finalize — unit tests for the attn finalize driver."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m2_attn_finalize as rf  # noqa: E402


def _mk_report(seed_val):
    return {
        "mu_ensemble": {
            "skill": 0.10, "ci_low": 0.05, "ci_high": 0.15, "permutation_p": 0.01,
            "n_designs": 5, "n_positions": 100, "wmae_model": 0.7, "wmae_baseline": 0.75,
            "per_design": {"mean": 0.1, "median": 0.1, "pct_positive": 1.0},
            "robustness": {"pooled_rho_min_over_loo": 0.08},
            "spearman_abs": seed_val, "auroc_abs": 0.7,
        },
        "n_seed_single_skill": {"seed_0": {"skill": 0.08}},
        "per_position": [{"position": 10, "spearman_abs": 0.44}],
    }


def test_progress_count(tmp_path):
    p = tmp_path / "fold_progress.json"
    p.write_text(json.dumps({"completed_folds": ["A", "B", "C"]}), encoding="utf-8")
    assert rf._progress(str(p)) == 3
    assert rf._progress(str(tmp_path / "missing.json")) == 0


def test_hrow_and_drow():
    h = rf._hrow(_mk_report(0.1))
    assert h["skill"] == 0.10
    assert h["pct_positive"] == 1.0
    d = rf._drow(_mk_report(0.41))
    assert d["spearman_rho"] == 0.41
    assert d["pos10_rho"] == 0.44
    assert d["loo_rho_min"] == 0.08
