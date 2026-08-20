#!/usr/bin/env python3
"""test_estimand_v2: hand-computed fixtures for the frozen method-balanced estimand.

Audit P0-2 acceptance:
  * method-balanced puzzle loss != pooled mutant loss on an imbalanced fixture.
  * adding duplicate mutants of one method does not change the method-balanced puzzle weight.
  * candidate/baseline biological-scoring-key exact pairing (mismatch raises, no zero-fill).
  * missing target positions are excluded, never treated as 0.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.reactflow_delta import evaluator_v2 as E


def _key(puzzle, method, construct, mutation, pos):
    return f"openknot_m2|{puzzle}|{method}|{construct}|{pos}|{mutation}|{pos}"


def _pred(bio_key, loc, scale, model="m", seed=0):
    return E.PredPoint(biological_scoring_key=bio_key, model_id=model,
                       seed_or_component_id=seed, outer_fold=0,
                       family="gaussian", location=loc, scale=scale)


def _tgt(bio_key, y, qualified=True):
    return E.TargetPoint(biological_scoring_key=bio_key, target=y, qualified=qualified)


def _crps_gaussian_manual(y: float, m: float = 0.0, s: float = 1.0) -> float:
    """Hand-computed Gaussian CRPS via the energy form."""
    from scipy import stats as st
    e_xy = s * st.norm.pdf((y - m) / s) * 2 + (y - m) * (2 * st.norm.cdf((y - m) / s) - 1)
    return e_xy - s / math.sqrt(math.pi)


def test_method_balanced_not_pooled_on_imbalanced_fixture():
    """P0-2 key test: A has 1 mutant, B has 9 mutants; method-balanced L=(L_A+L_B)/2
    must differ from pooled (1*L_A+9*L_B)/10."""
    puzzle = "P1"
    rows = []
    tgt = []
    # method A: 1 mutant, prediction loc=0.9, target=0.9 (tiny CRPS)
    rows.append(_pred(_key(puzzle, "A", "c", "m1", 0), 0.9, 1.0))
    tgt.append(_tgt(_key(puzzle, "A", "c", "m1", 0), 0.9))
    # method B: 9 mutants, prediction loc=0.9, target=-0.9 (large CRPS)
    for i in range(9):
        rows.append(_pred(_key(puzzle, "B", "c", f"m{i}", 0), 0.9, 1.0))
        tgt.append(_tgt(_key(puzzle, "B", "c", f"m{i}", 0), -0.9))

    cr_A = _crps_gaussian_manual(0.9, m=0.9)   # near 0
    cr_B = _crps_gaussian_manual(-0.9, m=0.9)  # large

    res_mb = E.score_ledger(rows, tgt, method_balanced=True)
    L_mb = res_mb["puzzles"][puzzle]["L"]
    assert math.isclose(L_mb, (cr_A + cr_B) / 2, rel_tol=1e-9)

    res_pool = E.score_ledger(rows, tgt, method_balanced=False)
    L_pool = res_pool["puzzles"][puzzle]["L"]
    # pooled = (1*cr_A + 9*cr_B)/10 — pulled toward B's poor method
    assert math.isclose(L_pool, (cr_A + 9 * cr_B) / 10, rel_tol=1e-9)
    assert abs(L_mb - L_pool) > 1e-6  # method-balanced != pooled on imbalanced fixture


def test_method_balanced_weight_invariant_to_duplicated_mutants():
    """P0-2 acceptance: adding duplicate mutants of method B must NOT change the
    method-balanced puzzle weight, but DOES change the pooled weight."""
    puzzle = "P2"

    def build(n_b):
        rows = []; tgt = []
        # method A: 1 mutant with target 0.9 (good prediction => low CRPS)
        rows.append(_pred(_key(puzzle, "A", "c", "mA", 0), 0.9, 1.0))
        tgt.append(_tgt(_key(puzzle, "A", "c", "mA", 0), 0.9))
        # method B: n_b mutants with target -0.9 (bad prediction => high CRPS)
        for i in range(n_b):
            rows.append(_pred(_key(puzzle, "B", "c", f"mB{i}", 0), 0.9, 1.0))
            tgt.append(_tgt(_key(puzzle, "B", "c", f"mB{i}", 0), -0.9))
        return rows, tgt

    from scipy import stats as st
    def crps(y, m=0.9, s=1.0):
        return _crps_gaussian_manual(y, m=m, s=s)

    L_A = crps(0.9)   # tiny
    L_B = crps(-0.9)  # large

    rows1, tgt1 = build(1)
    rows9, tgt9 = build(9)
    mb1 = E.score_ledger(rows1, tgt1, method_balanced=True)["puzzles"][puzzle]["L"]
    mb9 = E.score_ledger(rows9, tgt9, method_balanced=True)["puzzles"][puzzle]["L"]
    # method-balanced: (L_A + L_B)/2 regardless of B mutant count
    assert mb1 == pytest.approx((L_A + L_B) / 2, abs=1e-9)
    assert mb9 == pytest.approx((L_A + L_B) / 2, abs=1e-9)
    # pooled: (1*L_A + n*L_B)/(n+1) — differs and changes with n
    pooled1 = E.score_ledger(rows1, tgt1, method_balanced=False)["puzzles"][puzzle]["L"]
    pooled9 = E.score_ledger(rows9, tgt9, method_balanced=False)["puzzles"][puzzle]["L"]
    assert pooled1 == pytest.approx((L_A + L_B) / 2, abs=1e-9)
    assert pooled9 == pytest.approx((L_A + 9 * L_B) / 10, abs=1e-9)
    assert abs(pooled9 - mb9) > 1e-6  # pooled biased toward B's poor method


def test_exact_paired_effects_mismatch_raises():
    """P0-2: candidate/baseline key sets must match exactly; missing keys raise,
    never zero-filled."""
    rows = [_pred(_key("P1", "A", "c", "m1", 0), 0.0, 1.0)]
    tgt = [_tgt(_key("P1", "A", "c", "m1", 0), 0.5)]
    with pytest.raises(ValueError, match="mismatch"):
        E.score_ledger(rows, [], method_balanced=True)


def test_missing_target_excluded_never_zero():
    """P0-2: a missing (unqualified) target position is excluded from scoring and
    never treated as 0."""
    puzzle = "P3"
    rows = [
        _pred(_key(puzzle, "A", "c", "mA", 0), 0.0, 1.0),
        _pred(_key(puzzle, "A", "c", "mA", 1), 0.0, 1.0),
    ]
    tgt = [
        _tgt(_key(puzzle, "A", "c", "mA", 0), 0.5, qualified=True),
        _tgt(_key(puzzle, "A", "c", "mA", 1), None, qualified=False),  # missing
    ]
    res = E.score_ledger(rows, tgt, method_balanced=True)
    # only position 0 scored; a single position contributes a single CRPS value
    cr = _crps_gaussian_manual(0.5, m=0.0)
    assert res["puzzles"][puzzle]["L"] == pytest.approx(cr, abs=1e-9)


def test_method_balanced_equals_simple_mean_on_balanced_fixture():
    """Consistency check: when methods have equal mutant counts, method-balanced
    equals the cell-pooled mean (sanity that we did not over-rotate)."""
    puzzle = "P4"
    rows = []; tgt = []
    for meth in ("A", "B"):
        for i in range(2):
            rows.append(_pred(_key(puzzle, meth, "c", f"m{i}", 0), 0.3, 1.0))
            tgt.append(_tgt(_key(puzzle, meth, "c", f"m{i}", 0), 0.1))
    res = E.score_ledger(rows, tgt, method_balanced=True)
    res_pool = E.score_ledger(rows, tgt, method_balanced=False)
    assert res["puzzles"][puzzle]["L"] == pytest.approx(
        res_pool["puzzles"][puzzle]["L"], abs=1e-9)
