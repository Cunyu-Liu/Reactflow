#!/usr/bin/env python3
"""Tests for Phase 2 baselines_v6 (run_baselines_v6) + learnability gate helpers."""
import json, math, os, sys, tempfile
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/reactflow_delta"))
import run_baselines_v6 as rb
import run_learnability_gate_v1 as lg


def test_weighted_median():
    y = [1.0, 2.0, 3.0, 100.0]
    w = [1, 1, 1, 1]
    # lower-median convention: element where cumulative weight first reaches half-total
    assert rb._weighted_median(y, w) == pytest.approx(2.0)
    w2 = [1, 1, 1, 100]
    assert rb._weighted_median(y, w2) == pytest.approx(100.0)
    # constant
    assert rb._weighted_median([3.0, 3.0], [1, 1]) == pytest.approx(3.0)


def test_weighted_lad_linear_recovers_linear():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    beta = np.array([2.0, -1.0, 0.5])
    y = X @ beta + 0.1 * rng.normal(size=200)
    w = np.abs(rng.normal(size=200)) + 0.5
    coef = rb._weighted_lad_linear_coef(X, y, w)
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    pred = Xb @ coef
    assert len(pred) == 200
    # correlation with true linear signal high
    r = np.corrcoef(pred, y)[0, 1]
    assert r > 0.9


def test_spline_linear_fit():
    rng = np.random.default_rng(1)
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = np.sin(8 * X[:, 0]) + 0.05 * rng.normal(size=100)
    w = np.ones(100)
    pred_fn = rb._spline_linear(X, y, w, n_knots=4)
    pred = pred_fn(X)
    r = np.corrcoef(pred, y)[0, 1]
    assert r > 0.9


def test_magnitude_trivial_wmedian():
    Xtr = np.zeros((5, 4), dtype=np.float32)
    Xte = np.zeros((3, 4), dtype=np.float32)
    ytr = np.array([1.0, 2.0, 3.0, 4.0, 100.0], dtype=np.float32)
    wtr = np.array([1, 1, 1, 1, 100], dtype=np.float32)
    pred, _ = rb.fit_predict_magnitude("wmedian", Xtr, ytr, wtr, Xte, 0, None)
    assert float(pred[0]) == pytest.approx(100.0)


def test_primary_prevalence_trivial():
    Xtr = np.zeros((4, 4), dtype=np.float32)
    Xte = np.zeros((2, 4), dtype=np.float32)
    ytr = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    pred, _ = rb.fit_predict_primary("prevalence", Xtr, ytr, None, Xte, 0, None)
    assert float(pred[0]) == pytest.approx(0.5)


def test_lad_lm_predicts_on_test_length():
    # regression: lad_lm previously returned training-length predictions,
    # causing IndexError when len(train) != len(test).
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(50, 4)).astype(np.float32)
    ytr = (Xtr[:, 0] * 2.0 + 1.0).astype(np.float32)
    wtr = np.ones(50, dtype=np.float32)
    Xte = rng.normal(size=(7, 4)).astype(np.float32)
    pred, _ = rb.fit_predict_magnitude("lad_lm", Xtr, ytr, wtr, Xte, 0, None)
    assert len(pred) == 7  # must match test length, not training length


def test_seeds_for():
    assert rb.seeds_for("p2_mlp") == [0, 1, 2, 3, 4]
    assert rb.seeds_for("gbm") == [0]


def _synthetic_rows():
    rows = []
    # primary: 3 pubs
    for pub, n, pos in [("pmid_a", 20, 10), ("pmid_b", 20, 10), ("pmid_c", 20, 10)]:
        for i in range(n):
            y = 1.0 if i < pos else 0.0
            rows.append({"task": "primary", "fold_id": pub, "seed": 0,
                         "model_variant": "gbm", "raw_prediction": 0.9 if y else 0.1,
                         "y": y, "weight": 1.0, "coverage_status": "CALLED"})
            rows.append({"task": "primary", "fold_id": pub, "seed": 0,
                         "model_variant": "prevalence", "raw_prediction": 0.5,
                         "y": y, "weight": 1.0, "coverage_status": "CALLED"})
    # magnitude: 3 pubs with changers
    for pub in ["pmid_a", "pmid_b", "pmid_c"]:
        for i in range(10):
            y = 1.0 + i * 0.1
            rows.append({"task": "magnitude", "fold_id": pub, "seed": 0,
                         "model_variant": "wmae_mlp", "raw_prediction": y,
                         "y": y, "weight": 1.0, "coverage_status": "CALLED"})
            rows.append({"task": "magnitude", "fold_id": pub, "seed": 0,
                         "model_variant": "wmedian", "raw_prediction": 1.5,
                         "y": y, "weight": 1.0, "coverage_status": "CALLED"})
    return rows


def test_primary_analysis_perfect():
    rows = _synthetic_rows()
    a = lg.primary_analysis(rows, "gbm", 0)
    # perfect separation -> macro AUPRC ~ 1
    assert a["macro_auprc_model"] is not None
    assert a["macro_auprc_model"] > 0.99
    assert a["mean_delta_ap"] > 0
    assert a["n_publications"] == 3


def test_primary_analysis_loo():
    rows = _synthetic_rows()
    a_loo = lg.primary_analysis(rows, "gbm", 0, exclude_pub="pmid_a")
    assert a_loo["n_publications"] == 2


def test_magnitude_analysis_perfect():
    rows = _synthetic_rows()
    a = lg.magnitude_analysis(rows, "wmae_mlp", 0)
    assert a["skill"] == pytest.approx(1.0)
    assert a["n_publications"] == 3


def test_magnitude_analysis_loo():
    rows = _synthetic_rows()
    a = lg.magnitude_analysis(rows, "wmae_mlp", 0, exclude_pub="pmid_a")
    assert a["n_publications"] == 2


def test_build_pair_recs_missing_pub_fallback(tmp_path):
    # build a tiny cache + registry to exercise the join fallback
    cache = {"rec_index": {}, "pairs": []}
    # no pairs -> empty recs, no malformed access
    pub_map = {}
    recs, missing = rb.build_pair_recs(cache, pub_map)
    assert recs == {}
    assert missing == 0


def _rec(react, err, seq="ACGUACGUACGUACGUACGU", probe=("1M7",), temp=("37C",)):
    return {
        "is_wt": True, "canonical_sequence": seq, "probe": probe, "temperature": temp,
        "reactivity_layers": {"train_frozen": {"reactivity": react, "error": err},
                              "raw": {"reactivity": react, "error": err}},
    }


def test_robust_builder_preserves_mask_when_error_empty():
    react = [0.1 * i for i in range(20)]
    wt = _rec(react, [])          # empty error (the SRPDIV case)
    mut = _rec([0.1 * i for i in range(20)], [])
    codes = ["ELIGIBLE"] * 19 + ["EDITED_SITE"]
    pair = {"source_accession": "SRPDIV_DMS_0001", "wt_profile_index": 1,
            "mutant_profile_index": 17, "eligibility_reason_codes": codes}
    pf = rb.build_pair_features_aligned_robust(pair, wt, mut)
    assert len(pf.eligibility_mask) == 20
    assert sum(pf.eligibility_mask) == 19
    assert len(pf.wt_error) == 20 and all(x == 0.0 for x in pf.wt_error)
    assert len(pf.mutant_error) == 20


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))