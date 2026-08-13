#!/usr/bin/env python3
"""Unit fixtures for run_p5_mechanism_v1 locked mechanism contrasts.

Covers:
  - _band_of: |dist| band assignment (edit site, near, mid, far, very-far).
  - _ci_one_sided / _pval_one_sided: one-sided 95% t-CI and t-test p-value.
  - _collect_d real vs permuted: with a coef that maps features to targets, the
    real direct model shows skill at the edit site while the permuted negative
    control destroys it (feature-dependence of the effect).
"""

from __future__ import annotations

import numpy as np

import scripts.reactflow_delta.run_p5_mechanism_v1 as P


class TestBandOf:
    def test_edges(self):
        assert P._band_of(0) == 0          # edit_site
        assert P._band_of(1) == 1 and P._band_of(3) == 1      # near_1_3
        assert P._band_of(4) == 2 and P._band_of(10) == 2     # mid_4_10
        assert P._band_of(11) == 3 and P._band_of(25) == 3    # far_11_25
        assert P._band_of(26) == 4 and P._band_of(1000) == 4  # very_far_26p

    def test_signed_distance_uses_abs(self):
        assert P._band_of(-2) == P._band_of(2)


class TestStats:
    def test_ci_one_sided(self):
        ci = P._ci_one_sided([0.1] * 10)
        assert ci["ci_low"] > 0.0 and ci["n"] == 10

    def test_pval_one_sided_positive(self):
        # non-degenerate positive data -> one-sided p < family alpha
        assert P._pval_one_sided(list(range(1, 11))) < 0.025

    def test_pval_zero_variance_is_one(self):
        assert P._pval_one_sided([0.0, 0.0]) == 1.0


class TestCollectD:
    def _coef_wt_delta(self, delta: float) -> np.ndarray:
        # prediction = delta + wt_r  (feat: [intercept, we, wt_r, ...])
        coef = np.zeros(13)
        coef[0] = delta
        coef[2] = 1.0
        return coef

    def _profiles(self, wt_vals: list[float], mut_delta: float, n: int = 25):
        mut_vals = [v + mut_delta for v in wt_vals]
        wt = {"profile_name": "WT", "profile_sequence": "A" * 40,
              "reactivity": wt_vals}
        profs = {"WT": wt}
        for k in range(n):
            profs[f"C1_{k}A-C"] = {
                "profile_name": f"C1_{k}A-C", "profile_sequence": "A" * 40,
                "reactivity": mut_vals}
        return profs

    def _comp(self) -> dict:
        shared = list(range(25))
        return {"wt_name": "WT", "dataset": "M2SL5_2A3_0000",
                "n_snv_mutants": 25,
                "mutants": [{"name": f"C1_{k}A-C", "edit_pos": 10,
                             "shared_region": shared} for k in range(25)]}

    def test_real_edit_site_skill_but_permuted_no_skill(self):
        # WT reactivity varies per position; mutant target = wt_r + 0.3.
        # coef predicts delta + wt_r => direct matches every target exactly in
        # the REAL case, while the WT-anchor (zero) baseline predicts wt_r (wrong).
        wt_vals = [0.1 + 0.02 * i for i in range(25)]
        coef = self._coef_wt_delta(0.3)
        comps = [self._comp()]
        profs = self._profiles(wt_vals, mut_delta=0.3)
        real = P._collect_d(coef, comps, profs, permute=False)
        perm = P._collect_d(coef, comps, profs, permute=True)
        assert len(real) == 1 and len(perm) == 1
        # real: direct matches target exactly -> positive D in every band
        assert real[0]["band_D"]["edit_site"] > 0.0
        assert real[0]["band_D"]["far_11_25"] > 0.0
        # permuted: features shuffled away from their targets -> direct no longer
        # matches -> feature-dependent skill destroyed (D collapses below real)
        assert perm[0]["band_D"]["edit_site"] < real[0]["band_D"]["edit_site"]
        # and the real skill is not a small residual: real D is clearly positive
        assert real[0]["band_D"]["edit_site"] > 0.05

    def test_permutation_seed_deterministic(self):
        wt_vals = [0.1 + 0.02 * i for i in range(25)]
        coef = self._coef_wt_delta(0.3)
        comps = [self._comp()]
        profs = self._profiles(wt_vals, mut_delta=0.3)
        a = P._collect_d(coef, comps, profs, permute=True)
        b = P._collect_d(coef, comps, profs, permute=True)
        assert a[0]["band_D"]["edit_site"] == b[0]["band_D"]["edit_site"]

    def test_rule3_filter_applied(self):
        # shared region too small -> no mutant scored -> component excluded
        comp = {"wt_name": "WT", "dataset": "M2SL5_2A3_0000",
                "n_snv_mutants": 25,
                "mutants": [{"name": f"C1_{k}A-C", "edit_pos": 2,
                             "shared_region": list(range(5))} for k in range(25)]}
        wt_vals = [0.1 + 0.02 * i for i in range(25)]
        profs = self._profiles(wt_vals, mut_delta=0.3, n=25)
        out = P._collect_d(self._coef_wt_delta(0.3), [comp], profs)
        assert len(out) == 0
