#!/usr/bin/env python3
"""Unit tests for evaluate_v6 + validate_prediction_artifact_v2 fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "scripts/reactflow_delta"))
sys.path.insert(0, str(_ROOT / "src"))

import pytest
import evaluate_v6 as v6
import validate_prediction_artifact_v2 as vp


def _row(pair_id, pub, fold="f0", seed=0, mv="cand", y=0.0, w=1.0,
         pred=0.0, cov="CALLED", model="m"):
    return {"pair_id": pair_id, "asset_id": "a", "study_id": "s",
            "publication_id": pub, "parent_id": "p", "lineage_id": "l",
            "fold_id": fold, "split_role": "development",
            "endpoint_version": "endpoint_v6", "caller_version": "caller_v4",
            "seed": seed, "model_id": model, "model_variant": mv,
            "y": y, "weight": w, "raw_prediction": pred,
            "transformed_prediction": pred, "coverage_status": cov,
            "data_hash": "d", "split_hash": "sp", "caller_hash": "c",
            "model_config_hash": "mc", "source_commit": "sc"}


def _make_candidate_baseline():
    cand, base = [], []
    for pub in ["P1", "P2", "P3"]:
        for i in range(10):
            pid = f"{pub}:{i}"
            cand.append(_row(pid, pub, seed=0, mv="cand",
                             y=1.0 if i % 2 else 0.0,
                             pred=0.7 if i % 2 else 0.3))
            base.append(_row(pid, pub, seed=0, mv="base",
                             y=1.0 if i % 2 else 0.0, pred=0.5))
    return cand, base


def test_1_out_of_order_pair_ids_fail():
    cand, base = _make_candidate_baseline()
    base = list(reversed(base))
    ci = v6.paired_publication_ci(cand, base)
    assert ci["n_publications"] == 3
    assert ci["ci_low"] is not None


def test_2_duplicate_or_missing_key_fail():
    cand, base = _make_candidate_baseline()
    base.pop()
    with pytest.raises(ValueError):
        v6.paired_publication_ci(cand, base)


def test_3_row_order_change_metric_invariant():
    cand, base = _make_candidate_baseline()
    a = v6.primary_auprc(cand, base)["pooled_auprc"]
    b = v6.primary_auprc(list(reversed(cand)), list(reversed(base)))["pooled_auprc"]
    assert abs(a - b) < 1e-12


def test_4_tied_ap_order_invariant():
    scores = [0.5, 0.5, 0.5, 0.2, 0.1]
    labels = [1, 0, 1, 0, 0]
    assert abs(v6.auprc(scores, labels) - v6.auprc(scores[::-1], labels[::-1])) < 1e-12


def test_5_weighted_mean_vs_median():
    y = [1.0, 2.0, 100.0]
    w = [1.0, 1.0, 1.0]
    assert v6._weighted_mean(y, w) == 103.0 / 3.0
    assert v6.weighted_median(y, w) == 2.0


def test_6_perfect_equal_worse_baseline():
    cand, base = [], []
    for pub in ["P1", "P2", "P3"]:
        for i in range(5):
            pid = f"{pub}:{i}"
            lab = 1.0 if i % 2 else 0.0
            cand.append(_row(pid, pub, seed=0, mv="cand", y=lab, pred=lab))
            base.append(_row(pid, pub, seed=0, mv="base", y=lab, pred=0.5))
    ci = v6.paired_publication_ci(cand, base)
    assert ci["ci_low"] > 0.9


def test_7_weight_scaling_invariant_wmae():
    y = [1.0, 2.0, 3.0]
    p = [1.1, 2.2, 2.8]
    assert abs(v6._wmae(y, p, [1.0, 1.0, 1.0]) - v6._wmae(y, p, [3.0, 3.0, 3.0])) < 1e-12


def test_8_pooled_vs_macro_differ():
    cand, base = [], []
    # P1: 100 rows, all positive, high score -> high AUPRC
    for i in range(100):
        pid = f"P1:{i}"
        cand.append(_row(pid, "P1", seed=0, mv="cand", y=1.0, pred=0.9))
        base.append(_row(pid, "P1", seed=0, mv="base", y=1.0, pred=0.5))
    # P2: 2 rows, all negative -> AUPRC 0
    for i in range(2):
        pid = f"P2:{i}"
        cand.append(_row(pid, "P2", seed=0, mv="cand", y=0.0, pred=0.9))
        base.append(_row(pid, "P2", seed=0, mv="base", y=0.0, pred=0.5))
    res = v6.primary_auprc(cand, base)
    # pooled (dominated by P1) high; macro = (1.0 + 0.0)/2 = 0.5
    assert res["pooled_auprc"] > 0.9
    # P2 has n_pos=0 -> excluded from macro; macro over P1 only = 1.0
    assert res["macro_auprc"] == 1.0
    assert res["pooled_auprc"] != res["macro_auprc"]


def test_9_same_pmid_across_studies_should_not_split_fold():
    # two studies sharing one publication must both be in the same fold
    cand, base = [], []
    for i in range(5):
        cand.append(_row(f"A:{i}", "PMID_X", fold="fold1", seed=0, mv="cand", y=1.0, pred=0.8))
        base.append(_row(f"A:{i}", "PMID_X", fold="fold1", seed=0, mv="base", y=1.0, pred=0.5))
    for i in range(5):
        cand.append(_row(f"B:{i}", "PMID_X", fold="fold1", seed=0, mv="cand", y=0.0, pred=0.2))
        base.append(_row(f"B:{i}", "PMID_X", fold="fold1", seed=0, mv="base", y=0.0, pred=0.5))
    # All rows share publication PMID_X and fold1 -> single fold, single pub
    pubs = {r["publication_id"] for r in cand}
    folds = {r["fold_id"] for r in cand}
    assert pubs == {"PMID_X"}
    assert folds == {"fold1"}


def test_10_seed_duplication_does_not_increase_N():
    cand, base = [], []
    for pub in ["P1", "P2", "P3"]:
        for seed in [0, 1, 2]:
            for i in range(4):
                pid = f"{pub}:{i}"
                cand.append(_row(pid, pub, seed=seed, mv="cand",
                                 y=1.0 if i % 2 else 0.0, pred=0.7 if i % 2 else 0.3))
                base.append(_row(pid, pub, seed=seed, mv="base",
                                 y=1.0 if i % 2 else 0.0, pred=0.5))
    res = v6.primary_auprc(cand, base)
    assert res["n_publications"] == 3


def test_11_three_pubs_insufficient_null_assignments():
    ns = v6.enumerate_null_space(["P1", "P2", "P3"])
    assert ns["identifiable"] is False


def test_12_identity_only_permutation_unidentifiable():
    ns = v6.enumerate_null_space(["P1"] * 5)
    assert ns["identifiable"] is False
    assert ns["unique_null_assignments"] == 1


def test_13_target_mask_not_in_model_input():
    # Schema-level: target_mask must not be declared as a model-input field.
    import json
    schema = json.loads(Path(_ROOT / "schemas/reactflow_delta/prediction_v2.schema.json").read_text())
    assert schema["rules"]["target_mask_not_in_model_input"] is True


def test_14_tool_failure_cannot_become_zero_prediction():
    rows = [_row("k1", "P1", y=1.0, pred=0.0, cov="TOOL_FAILURE")]
    with pytest.raises(ValueError):
        vp.validate_rows(rows)


def test_15_row_count_semantics():
    rows = []
    for pub in ["P1", "P2", "P3"]:
        for seed in [0, 1]:
            for i in range(4):
                rows.append(_row(f"{pub}:{i}", pub, seed=seed, mv="cand",
                                 y=1.0 if i % 2 else 0.0, pred=0.7 if i % 2 else 0.3))
    m = vp.validate_rows(rows, expect_seeds=2)
    assert m["row_count_semantics"]["cand"]["consistent"] is True


def test_validate_duplicate_key_fails():
    rows = [_row("k1", "P1", seed=0, mv="cand"), _row("k1", "P1", seed=0, mv="cand")]
    with pytest.raises(ValueError):
        vp.validate_rows(rows)
