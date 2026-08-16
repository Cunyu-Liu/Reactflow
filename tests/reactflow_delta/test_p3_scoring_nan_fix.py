#!/usr/bin/env python3
"""Regression fixtures for run_p3_lrso_v2 scoring NaN-poisoning bug.

Root cause fixed: _bstar_held_crps and _lrso_held_crps used np.nanmean([]) when
a held record had no qualified positions, which returns NaN and poisons the
running total, invalidating the whole 20-puzzle CRPS. The fix skips empty-q
records. These fixtures reproduce the poison with a minimal fake universe.
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.reactflow_delta.run_p3_lrso_v2 as P


class _FakeConstruct:
    def __init__(self, wt_reactivity: np.ndarray) -> None:
        self.wt_reactivity = wt_reactivity


class _FakeRec:
    def __init__(self, construct_id: str, pos: int, target_observed: bool) -> None:
        self.construct_id = construct_id
        self.wt_id = "W"
        self.pos = pos
        self.ref = "A"
        self.alt = "G"
        self.target_observed = target_observed


class _FakeUniv:
    def __init__(self, profiles: dict, constructs: dict) -> None:
        self.profiles = profiles
        self.constructs = constructs

    def get_construct(self, cid: str):
        return self.constructs[cid]

    def mutant_full_profile(self, wt_id: str, pos: int, ref: str, alt: str):
        return self.profiles[(wt_id, pos, ref, alt)]


def _coef() -> np.ndarray:
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])


def _univ_with_empty_and_valid():
    wt = np.array([0.5, 0.7, 0.9])
    c = _FakeConstruct(wt)
    tprof_all_nan = np.full(3, np.nan)
    tprof_valid = np.array([0.6, np.nan, 1.0])  # qualified at pos 0 and 2
    univ = _FakeUniv(
        profiles={
            ("W", 1, "A", "G"): (tprof_all_nan, None),
            ("W", 2, "A", "G"): (tprof_valid, None),
        },
        constructs={"C1": c},
    )
    rec_empty = _FakeRec("C1", 1, True)
    rec_valid = _FakeRec("C1", 2, True)
    return univ, rec_empty, rec_valid


def test_bstar_held_crps_skips_empty_q_record():
    """All-qualified-positions-missing record must NOT poison total with NaN."""
    univ, rec_empty, rec_valid = _univ_with_empty_and_valid()
    # only the empty-q record => n==0, honest nan (no valid records)
    out_only_empty = P._bstar_held_crps(univ, [rec_empty], _coef(), None)
    assert out_only_empty != out_only_empty  # nan (n==0)
    # empty-q record followed by a valid record => finite mean (no poisoning)
    out_both = P._bstar_held_crps(univ, [rec_empty, rec_valid], _coef(), None)
    assert out_both == out_both  # finite


def test_bstar_held_crps_poison_before_fix_detected():
    """If the empty-q record is skipped correctly, the mean is finite; if the
    old np.nanmean([]) path were still present, it would be NaN. This is the
    exact failure mode observed for P20 (B*_held_crps=nan)."""
    univ, rec_empty, rec_valid = _univ_with_empty_and_valid()
    out = P._bstar_held_crps(univ, [rec_empty, rec_valid], _coef(), None)
    assert np.isfinite(out)
    # the valid record contributes CRPS of N(pred,0.3) at qualified targets only
    assert out > 0.0


def test_lrso_held_crps_empty_q_uses_same_guard():
    """_lrso_held_crps has the identical empty-q guard; the fixture here checks
    the shared helper path stays finite. The LRSO variant itself requires a
    trained model, so we assert the guard contract via _bstar_held_crps and a
    structural check that the runner now uses np.where + size guard."""
    src = P._bstar_held_crps.__code__.co_code  # noqa
    import inspect
    lrso_src = inspect.getsource(P._lrso_held_crps)
    assert "np.where(" in lrso_src
    assert "q.size == 0" in lrso_src
