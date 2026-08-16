#!/usr/bin/env python3
"""Tests for m2r_noise_floor_v1 (rescue_factor formula + measurement-noise floor).

The rescue_factor formula (design-region RMSD quadrature) is verified against
the data, and the Monte-Carlo noise floor must be finite and bounded.  These
tests run on the real M2R data (fast: formula eval is cheap; MC draws reduced
via the module's per-call n_mc parameter through run_permtest-style helper).
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_noise_floor_v1 as nf

M2R_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"


def test_formula_exact():
    """The design-region quadrature RMSD formula must reproduce rescue_factor."""
    designs, meta = m2r.parse_m2r_csv(M2R_CSV)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    errs = []
    for s in samples[:200]:
        n = len(s.wt_reactivity)
        mask = nf._design_mask(n, s.sub_start, s.sub_end)
        pred = nf.rescue_from_profiles(
            nf._prof(s.wt_reactivity), nf._prof(s.singleA_reactivity),
            nf._prof(s.singleB_reactivity), nf._prof(s.double_reactivity), mask)
        if np.isfinite(pred):
            errs.append(abs(s.rescue_factor - pred))
    errs = np.array(errs)
    assert len(errs) > 150
    assert np.median(errs) < 0.001
    assert np.percentile(errs, 99) < 0.01


def test_design_mask_bounds():
    m = nf._design_mask(50, 5, 10)
    assert m[:4].sum() == 0
    assert m[4:10].sum() == 6
    assert m[10:].sum() == 0


def test_noise_floor_finite_and_ordered():
    """sigma_noise must be positive; median noise < mean noise (heavy tail);
    learnable fraction must be in (0.5, 1.0)."""
    designs, meta = m2r.parse_m2r_csv(M2R_CSV)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    rng = np.random.default_rng(nf.RNG_SEED)
    sigmas = []
    for s in samples[:60]:
        n = len(s.wt_reactivity)
        mask = nf._design_mask(n, s.sub_start, s.sub_end)
        wt = nf._prof(s.wt_reactivity); ra = nf._prof(s.singleA_reactivity)
        rb = nf._prof(s.singleB_reactivity); rd = nf._prof(s.double_reactivity)
        we = nf._prof(s.wt_error); ae = nf._prof(s.singleA_error)
        be = nf._prof(s.singleB_error); de = nf._prof(s.double_error)
        draws = []
        for _ in range(60):
            wtp = np.where(np.isfinite(wt), wt + rng.normal(0, 1, n) * np.where(np.isfinite(we), we, 0.0), np.nan)
            rap = np.where(np.isfinite(ra), ra + rng.normal(0, 1, n) * np.where(np.isfinite(ae), ae, 0.0), np.nan)
            rbp = np.where(np.isfinite(rb), rb + rng.normal(0, 1, n) * np.where(np.isfinite(be), be, 0.0), np.nan)
            rdp = np.where(np.isfinite(rd), rd + rng.normal(0, 1, n) * np.where(np.isfinite(de), de, 0.0), np.nan)
            v = nf.rescue_from_profiles(wtp, rap, rbp, rdp, mask)
            if np.isfinite(v):
                draws.append(v)
        if len(draws) >= 30:
            sigmas.append(float(np.std(draws)))
    sigmas = np.array(sigmas)
    assert len(sigmas) > 40
    assert np.all(sigmas > 0)
    # heavy tail: mean > median typically
    assert np.median(sigmas) < np.mean(sigmas) + 1e-9
    learn = 1.0 - (np.median(sigmas) / 0.2534) ** 2
    assert 0.5 < learn <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
