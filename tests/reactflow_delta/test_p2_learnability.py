#!/usr/bin/env python3
"""Fixtures for P2 direct learnability statistical machinery."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.reactflow_delta.p2_learnability import (
    DIRECT_METHODS, TRIVIAL_METHODS, d_p_p2, leave_one_puzzle_influence,
    puzzle_level_ci20, select_inner_direct_star, select_inner_t_star,
    studentized_sign_flip,
)


def test_t_star_picks_lower_crps():
    ic = {"zero": 1.0, "train_median": 0.7}
    assert select_inner_t_star(ic, ["zero", "train_median"]) == "train_median"


def test_t_star_deterministic_tie_break():
    ic = {"zero": 0.5, "train_median": 0.5}
    # frozen precedence order zero first
    assert select_inner_t_star(ic, ["zero", "train_median"]) == "zero"
    assert select_inner_t_star(ic, ["train_median", "zero"]) == "train_median"


def test_direct_star_restricted_to_direct_methods():
    ic = {m: 1.0 for m in DIRECT_METHODS}
    ic["flat_mlp"] = 0.4
    assert select_inner_direct_star(ic, sorted(DIRECT_METHODS)) == "flat_mlp"


def test_d_p_direction_positive_when_candidate_better():
    # Direct* lower L (better) than T* => D positive
    assert d_p_p2(1.0, 0.8) == pytest.approx(0.2)


def test_ci20_and_planned_n_not_met():
    r = puzzle_level_ci20([0.2] * 20)
    assert r["planned_n_not_met"] is False
    assert r["ci_low"] > 0.0  # all positive, tiny SE
    r2 = puzzle_level_ci20([0.2] * 19)
    assert r2["planned_n_not_met"] is True


def test_sign_flip_strong_signal_small_p():
    rng = np.random.RandomState(0)
    effects = rng.normal(0.3, 0.05, 20)
    r = studentized_sign_flip(effects)
    assert r["status"] == "OK"
    assert r["p_value"] < 0.05


def test_sign_flip_identifiable_requires_sd():
    r = studentized_sign_flip([2] * 20)  # exactly zero sd -> unidentifiable
    assert r["status"] == "UNIDENTIFIABLE_SIGN_FLIP"


def test_leave_one_puzzle_influence():
    effects = [0.3] * 19 + [0.0]
    puzzles = [f"P{i:02d}" for i in range(1, 21)]
    r = leave_one_puzzle_influence(effects, puzzles)
    assert len(r["rows"]) == 20
    assert r["max_abs_shift"] > 0.01
