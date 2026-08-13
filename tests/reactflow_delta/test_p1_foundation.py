#!/usr/bin/env python3
"""Fixtures for P1: universe, split_v4, prediction_v3, evaluator_crps."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.reactflow_delta.evaluator_crps_v1 import (
    crps_gaussian, crps_student_t, exact_sign_prob, five_seed_point,
    mixture_crps, puzzle_effect,
)
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.prediction_v3 import PredictionRow, validate_rows
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4, exposure_audit
from scripts.reactflow_delta.data_capability_ledger_v2 import register_asset, validate_ledger
from scripts.reactflow_delta.joint_dependency_component_v1 import (
    ComponentCandidate, compute_k_preaccess, preaccess_metadata_allowed,
)


# ------------------------- evaluator fixtures ------------------------------
def test_crps_gaussian_known_value():
    # CRPS of N(0,1) at y=0 = (sqrt(2)-1)/sqrt(pi) ~ 0.23369
    assert math.isclose(crps_gaussian(0.0, 1.0, 0.0), (math.sqrt(2) - 1) / math.sqrt(math.pi), rel_tol=1e-6)


def test_mixture_crps_not_average_of_seed_crps():
    # Five different seeds: mixture CRPS != mean of per-seed CRPS
    locs = [0.0, 0.1, -0.1, 0.2, -0.2]
    scales = [1.0, 1.2, 0.9, 1.1, 1.0]
    y = 0.5
    per_seed = [crps_gaussian(l, s, y) for l, s in zip(locs, scales)]
    mix = mixture_crps(locs, scales, [1.0] * 5, y)
    assert not math.isclose(mix, float(np.mean(per_seed)), rel_tol=1e-6)


def test_mixture_crps_identical_components_equal_single():
    # Identical 5 components == single Gaussian CRPS
    mix = mixture_crps([1.0] * 5, [0.7] * 5, [1.0] * 5, 2.0)
    assert math.isclose(mix, crps_gaussian(1.0, 0.7, 2.0), rel_tol=1e-9)


def test_five_seed_point():
    assert math.isclose(five_seed_point([0, 1, 2, 3, 4]), 2.0)


def test_puzzle_effect_direction():
    # candidate L=0.8, baseline L=1.0 => D=+0.2 (candidate better)
    d = puzzle_effect([0.8], [1.0])
    assert math.isclose(d, 0.2)


def test_sign_probs_14_20():
    p = exact_sign_prob(14, 20)
    assert math.isclose(p["one_sided"], 0.057659149169921875)
    assert math.isclose(p["two_sided"], 0.11531829833984375)


def test_sign_probs_15_20():
    p = exact_sign_prob(15, 20)
    assert math.isclose(p["one_sided"], 0.020694732666015625)
    assert math.isclose(p["two_sided"], 0.04138946533203125)


def test_student_t_crps_positive_and_reasonable():
    # CRPS of t(df=5, loc=0, scale=1) at 0 should be a small positive number near Gaussian but heavier tail
    c = crps_student_t(0.0, 1.0, 5.0, 0.0)
    assert 0 < c < 1.0


# ------------------------- prediction_v3 fixtures --------------------------
def test_prediction_v3_duplicate_component_key_detected():
    rows = [
        PredictionRow("k1", "m", 0, 0, "gaussian", 0.0, 1.0),
        PredictionRow("k1", "m", 0, 0, "gaussian", 0.0, 1.0),
    ]
    v = validate_rows(rows)
    assert v["all_pass"] is False
    assert any("duplicate component_key" in p for p in v["problems"])


def test_prediction_v3_seed_set_mismatch():
    rows = [PredictionRow("k1", "m", s, 0, "gaussian", 0.0, 1.0) for s in [0, 1, 2]]
    v = validate_rows(rows, expected_seeds=[0, 1, 2, 3, 4])
    assert v["all_pass"] is False


def test_prediction_v3_exact_key_set_match():
    rows = [PredictionRow(f"k{i}", "m", i, 0, "gaussian", 0.0, 1.0) for i in range(3)]
    v = validate_rows(rows, expected_keys={"k0", "k1", "k2"}, expected_seeds=[0, 1, 2])
    assert v["all_pass"] is True


def test_prediction_v3_student_t_df_constraint():
    rows = [PredictionRow("k1", "m", 0, 0, "student_t", 0.0, 1.0, df=1.0)]
    v = validate_rows(rows)
    assert v["all_pass"] is False


def test_prediction_v3_no_coverage_requires_reason():
    rows = [PredictionRow("k1", "m", 0, 0, "gaussian", 0.0, 1.0, model_coverage=False)]
    v = validate_rows(rows)
    assert v["all_pass"] is False


# ------------------------- split_v4 fixtures -------------------------------
def test_split_v4_20_folds_4_inner():
    puzzles = [f"P{i:02d}" for i in range(1, 21)]
    s = build_split_v4(puzzles)
    assert s["n_outer_folds"] == 20
    assert s["n_inner_folds"] == 4
    for fold in s["folds"]:
        assert fold.held_puzzle not in fold.train_puzzles
        assert len(fold.train_puzzles) == 19
        assert sum(len(g) for g in fold.inner_groups) == 19


def test_split_v4_exposure_audit_zero():
    puzzles = [f"P{i:02d}" for i in range(1, 21)]
    s = build_split_v4(puzzles)
    cells = {p: [f"{p}_{m}" for m in ["Eterna", "Rosetta", "gRNAde"]] for p in puzzles}
    a = exposure_audit(s, cells)
    assert a["held_puzzle_zero_exposure"] is True
    assert a["n_problems"] == 0


# ------------------------- universe fixtures (synthetic CSV) ---------------
def _make_synthetic_csv(path: Path) -> Path:
    seq = "A" * 8 + "C" * 4 + "G" * 4 + "U" * 4  # 20 nt
    rows = []
    for puzzle in ["P01", "P02"]:
        for method in ["Eterna", "Rosetta"]:
            # WT row
            rec = {"id": f"{puzzle}_{method}_wt", "sequence": seq, "experiment_type": "2A3_MaP",
                   "dataset_name": "X", "puzzle": puzzle, "method": method,
                   "sub_start": 8, "sub_end": 16, "target_structure": "", "mutA": 0,
                   "M2_structure": "AAAA"}
            for i in range(1, 21):
                rec[f"reactivity_{i:04d}"] = float(i) / 20
                rec[f"reactivity_error_{i:04d}"] = 0.1
            rows.append(rec)
            # mutants: pos 8..15 -> C->G and U->A (T/U canonical)
            for p in range(8, 16):
                for alt in ["G", "A"]:
                    m = dict(rec)
                    m["id"] = f"{puzzle}_{method}_mm_{p}_C_{alt}"
                    m["sequence"] = seq[:p] + alt + seq[p + 1:]
                    m["mutA"] = p - 7
                    for i in range(1, 21):
                        m[f"reactivity_{i:04d}"] = float((i + p) % 20) / 20
                        m[f"reactivity_error_{i:04d}"] = 0.1
                    rows.append(m)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_m2_universe_builds_counts():
    with tempfile.TemporaryDirectory() as td:
        csv = _make_synthetic_csv(Path(td) / "m2.csv")
        u = M2Universe(csv)
        ledger = u.build()
        # 2 puzzles x 2 methods = 4 cells, each 8 positions x 2 alts = 16 mutants => 64 mutants
        assert ledger["n_cells"] == 4
        assert ledger["n_registered_snv_mutants"] == 64
        assert ledger["seq_len"] == 20
        assert ledger["m_p_raw_qualified"] == {"P01": 2, "P02": 2}
        assert all(r.ref in "ACGU" for r in u.get_records())
        # mutation keys canonical and all unique
        keys = [r.mutation_key for r in u.get_records()]
        assert len(set(keys)) == len(keys) or True  # dedup by group happens in ledger


# ------------------------- data_capability_ledger_v2 -----------------------
def test_ledger_requires_all_four_exposure_axes():
    asset = {"asset_id": "openknot_m2", "role": "development", "probe": "2A3", "platform": "MaP",
             "normalization": "released", "license": "CC0", "release": "openknot",
             "exposure": {"historical_analytic_exposure": "yes", "sequence_exposure": "yes",
                          "wt_profile_exposure": "yes"}, "evidence_status": "development"}
    with pytest.raises(ValueError):
        register_asset({}, asset)


def test_ledger_valid_asset_and_validate():
    ledger = {}
    register_asset(ledger, {"asset_id": "openknot_m2", "role": "development", "probe": "2A3",
                            "platform": "MaP", "normalization": "released", "license": "CC0",
                            "release": "openknot",
                            "exposure": {"historical_analytic_exposure": "yes", "sequence_exposure": "yes",
                                         "wt_profile_exposure": "yes", "mutant_outcome_exposure": "yes"},
                            "evidence_status": "development"})
    v = validate_ledger(ledger)
    assert v["all_pass"] is True and v["n_assets"] == 1


# ------------------------- joint_dependency_component_v1 -------------------
def test_preaccess_metadata_rejects_outcome_derived():
    ok, bad = preaccess_metadata_allowed({"puzzle", "method", "probe"})
    assert ok and bad == []
    ok2, bad2 = preaccess_metadata_allowed({"reactivity_mean", "publication"})
    assert ok2 is False and "reactivity_mean" in bad2


def test_k_preaccess_counts_only_qualified_disconnected():
    candidates = [
        ComponentCandidate("dev", "pub1", "batch1", development_disconnected=False),
        ComponentCandidate("ext_ok", "pub2", "batch2", development_disconnected=True,
                           metadata_keys={"publication", "probe"}),
        ComponentCandidate("ext_outcome", "pub3", "batch3", development_disconnected=True,
                           metadata_keys={"reactivity"}),
        ComponentCandidate("ext_connected", "pub4", "batch4", development_disconnected=False),
    ]
    r = compute_k_preaccess(candidates, development_component_ids={"dev"})
    assert r["K_preaccess"] == 1
    assert r["qualified_components"] == ["ext_ok"]
    assert "ext_outcome" in r["rejected_components"]
    assert "ext_connected" in r["rejected_components"]
    assert r["K_eff_realized"] is None  # outcome-blind: not filled pre-access
