"""Tests for ``reactflow.delta.evaluate`` (B0 unified evaluator).

Covers:
  * Endpoint mask construction (§12.1: unedited + aligned + probe-eligible).
  * Skill metric: 1 - WMAE(pred)/WMAE(0); edge cases (zero reference, empty mask).
  * Aggregation: pair -> parent macro -> study macro.
  * ``load_split_pairs`` against a tiny synthetic registry.
  * Secondary metrics (Pearson, Spearman, sign acc, AUPRC, local/mid/remote).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.evaluate import (
    EVALUATOR_SCHEMA_VERSION,
    PairRecord,
    PairMetrics,
    aggregate_metrics,
    build_endpoint_mask,
    compute_pair_metrics,
    evaluate_predictions,
    load_split_pairs,
    to_float_array,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    pair_id: str = "p1",
    parent: str = "parentA",
    study: str = "doiA",
    delta: list | None = None,
    wt_r: list | None = None,
    mut_r: list | None = None,
    edit_arr_idx: int = 2,
    edit_pos_1idx: int = 3,
    ref: str = "G",
    weight: float = 1.0,
    seq_positions: list | None = None,
) -> PairRecord:
    n = len(delta) if delta is not None else 5
    if delta is None:
        delta = [0.1, -0.2, 0.5, 0.0, 0.3]
    if wt_r is None:
        wt_r = [0.1] * n
    if mut_r is None:
        mut_r = [0.2] * n
    if seq_positions is None:
        seq_positions = [float(i + 1) for i in range(n)]
    d = to_float_array(delta)
    w = to_float_array(wt_r)
    m = to_float_array(mut_r)
    mask = build_endpoint_mask(d, w, m, edit_arr_idx)
    return PairRecord(
        pair_id=pair_id,
        parent=parent,
        study=study,
        rdat_path=f"/fake/{pair_id}.rdat",
        wt_profile_index=1,
        mutant_profile_index=2,
        edit_arr_idx=edit_arr_idx,
        edit_pos_1indexed=edit_pos_1idx,
        encoded_ref=ref,
        aligned_length=n,
        delta_true=d,
        endpoint_mask=mask,
        pair_quality_weight=weight,
        seq_positions=np.array(seq_positions, dtype=float),
    )


# ---------------------------------------------------------------------------
# to_float_array / build_endpoint_mask
# ---------------------------------------------------------------------------


class TestToFloatArray:
    def test_basic_with_none(self):
        arr = to_float_array([0.1, None, 0.3, None])
        assert arr.shape == (4,)
        assert math.isnan(arr[1])
        assert arr[0] == 0.1
        assert math.isnan(arr[3])

    def test_empty(self):
        arr = to_float_array([])
        assert arr.shape == (0,)


class TestEndpointMask:
    def test_excludes_edit_position(self):
        delta = to_float_array([0.1, 0.2, 0.3, 0.4])
        wt = to_float_array([0.0, 0.0, 0.0, 0.0])
        mut = to_float_array([0.0, 0.0, 0.0, 0.0])
        mask = build_endpoint_mask(delta, wt, mut, edit_arr_idx=1)
        assert mask.tolist() == [True, False, True, True]

    def test_excludes_missing(self):
        delta = to_float_array([0.1, None, 0.3, 0.4])
        wt = to_float_array([0.0, 0.0, None, 0.0])
        mut = to_float_array([0.0, 0.0, 0.0, None])
        mask = build_endpoint_mask(delta, wt, mut, edit_arr_idx=None)
        # Position 0: ok; 1: delta None; 2: wt None; 3: mut None
        assert mask.tolist() == [True, False, False, False]

    def test_edit_out_of_range_ignored(self):
        delta = to_float_array([0.1, 0.2])
        wt = to_float_array([0.0, 0.0])
        mut = to_float_array([0.0, 0.0])
        mask = build_endpoint_mask(delta, wt, mut, edit_arr_idx=99)
        assert mask.tolist() == [True, True]

    def test_edit_none(self):
        delta = to_float_array([0.1, 0.2])
        wt = to_float_array([0.0, 0.0])
        mut = to_float_array([0.0, 0.0])
        mask = build_endpoint_mask(delta, wt, mut, edit_arr_idx=None)
        assert mask.all()


# ---------------------------------------------------------------------------
# Skill metric
# ---------------------------------------------------------------------------


class TestSkillMetric:
    def test_perfect_prediction_skill_one(self):
        rec = _make_record(delta=[0.1, -0.2, 0.5, 0.0], edit_arr_idx=3)
        # Mask = [T, T, T, F] (edit at idx 3)
        pred = np.array([0.1, -0.2, 0.5, 0.0])
        m = compute_pair_metrics(rec, pred)
        assert m.wmae_pred == pytest.approx(0.0, abs=1e-9)
        assert m.skill == pytest.approx(1.0, abs=1e-9)

    def test_zero_prediction_skill_zero(self):
        # Predicting 0 gives WMAE_pred = WMAE_zero -> Skill = 0.
        rec = _make_record(delta=[0.1, -0.2, 0.5, 0.0], edit_arr_idx=3)
        pred = np.zeros(4)
        m = compute_pair_metrics(rec, pred)
        assert m.wmae_pred == pytest.approx(m.wmae_zero)
        assert m.skill == pytest.approx(0.0, abs=1e-9)

    def test_worse_than_zero_negative_skill(self):
        rec = _make_record(delta=[0.1, -0.2, 0.5, 0.0], edit_arr_idx=3)
        # Predict the negation: |pred - true| = 2|true| -> Skill = 1 - 2 = -1
        pred = np.array([-0.1, 0.2, -0.5, 0.0])
        m = compute_pair_metrics(rec, pred)
        assert m.wmae_pred == pytest.approx(2 * m.wmae_zero)
        assert m.skill == pytest.approx(-1.0, abs=1e-9)

    def test_zero_reference_skill_nan(self):
        # If true delta is all zeros on the mask, WMAE_zero = 0 -> Skill NaN.
        rec = _make_record(delta=[0.0, 0.0, 0.0, 0.5], edit_arr_idx=3)
        # Mask = [T, T, T, F]; true on mask = [0, 0, 0]
        pred = np.array([0.1, 0.2, 0.3, 0.0])
        m = compute_pair_metrics(rec, pred)
        assert m.wmae_zero == pytest.approx(0.0, abs=1e-12)
        assert math.isnan(m.skill)

    def test_quality_weight_recorded(self):
        rec = _make_record(weight=0.42)
        m = compute_pair_metrics(rec, np.zeros(5))
        assert m.weight == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Secondary metrics
# ---------------------------------------------------------------------------


class TestSecondaryMetrics:
    def test_pearson_perfect(self):
        rec = _make_record(delta=[0.1, 0.2, 0.3, 0.4, 0.5], edit_arr_idx=0)
        # Mask = [F, T, T, T, T]
        pred = np.array([0.0, 0.2, 0.3, 0.4, 0.5])
        m = compute_pair_metrics(rec, pred)
        assert m.pearson_r == pytest.approx(1.0, abs=1e-9)

    def test_pearson_undefined_constant(self):
        rec = _make_record(delta=[0.1, 0.2, 0.3, 0.4, 0.5], edit_arr_idx=0)
        pred = np.array([0.0, 0.5, 0.5, 0.5, 0.5])  # constant on mask
        m = compute_pair_metrics(rec, pred)
        assert math.isnan(m.pearson_r)

    def test_sign_accuracy(self):
        rec = _make_record(delta=[0.1, -0.2, 0.3, 0.4, 0.5], edit_arr_idx=0)
        # Mask = [F, T, T, T, T]; true on mask = [-0.2, 0.3, 0.4, 0.5]
        pred = np.array([0.0, -0.5, 0.1, -0.4, 0.5])
        # signs: true=[-, +, +, +], pred=[-, +, -, +] -> 3/4 match
        m = compute_pair_metrics(rec, pred)
        assert m.sign_accuracy == pytest.approx(0.75, abs=1e-9)

    def test_distance_bands(self):
        # 5 positions, edit at seq pos 3 (arr idx 2). seq_positions = [1,2,3,4,5]
        rec = _make_record(
            delta=[0.1, 0.2, 0.5, 0.4, 0.3],
            edit_arr_idx=2,
            edit_pos_1idx=3,
            seq_positions=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        # Mask = [T, T, F, T, T]; distances on mask = [|1-3|, |2-3|, |4-3|, |5-3|] = [2, 1, 1, 2]
        # local (<=10): all 4 positions
        pred = np.array([0.1, 0.2, 0.5, 0.4, 0.3])
        m = compute_pair_metrics(rec, pred)
        assert m.local_n == 4
        assert m.mid_n == 0
        assert m.remote_n == 0
        assert m.local_wmae == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_parent_macro_average(self):
        # 2 pairs in parentA, 1 pair in parentB; parentA and parentB in study1.
        records_metrics = [
            PairMetrics(
                pair_id="a1", parent="parentA", study="s1", n_positions=3, weight=1.0,
                wmae_pred=0.1, wmae_zero=0.2, skill=0.5, rmse=0.1,
                pearson_r=0.9, spearman_rho=0.8, sign_accuracy=0.7,
                affected_auprc=0.6, local_wmae=0.1, mid_wmae=0.1, remote_wmae=0.1,
                local_n=1, mid_n=1, remote_n=1,
            ),
            PairMetrics(
                pair_id="a2", parent="parentA", study="s1", n_positions=3, weight=1.0,
                wmae_pred=0.3, wmae_zero=0.2, skill=-0.5, rmse=0.3,
                pearson_r=0.5, spearman_rho=0.4, sign_accuracy=0.6,
                affected_auprc=0.5, local_wmae=0.3, mid_wmae=0.3, remote_wmae=0.3,
                local_n=1, mid_n=1, remote_n=1,
            ),
            PairMetrics(
                pair_id="b1", parent="parentB", study="s1", n_positions=3, weight=1.0,
                wmae_pred=0.2, wmae_zero=0.4, skill=0.5, rmse=0.2,
                pearson_r=0.7, spearman_rho=0.6, sign_accuracy=0.8,
                affected_auprc=0.4, local_wmae=0.2, mid_wmae=0.2, remote_wmae=0.2,
                local_n=1, mid_n=1, remote_n=1,
            ),
        ]
        agg = aggregate_metrics(records_metrics)
        # parentA skill = mean(0.5, -0.5) = 0.0; parentB skill = 0.5
        assert agg["per_parent"]["parentA"]["skill"] == pytest.approx(0.0)
        assert agg["per_parent"]["parentB"]["skill"] == pytest.approx(0.5)
        # study s1 = mean(parentA, parentB) = mean(0.0, 0.5) = 0.25
        assert agg["per_study"]["s1"]["skill"] == pytest.approx(0.25)
        # final = mean over studies = 0.25 (only one study)
        assert agg["final"]["skill"] == pytest.approx(0.25)

    def test_multi_study_macro(self):
        records_metrics = [
            PairMetrics(
                pair_id="a1", parent="pA", study="s1", n_positions=1, weight=1.0,
                wmae_pred=0.1, wmae_zero=0.2, skill=0.5, rmse=0.1,
                pearson_r=0.9, spearman_rho=0.8, sign_accuracy=0.7,
                affected_auprc=0.6, local_wmae=0.1, mid_wmae=0.1, remote_wmae=0.1,
                local_n=1, mid_n=0, remote_n=0,
            ),
            PairMetrics(
                pair_id="b1", parent="pB", study="s2", n_positions=1, weight=1.0,
                wmae_pred=0.2, wmae_zero=0.4, skill=0.5, rmse=0.2,
                pearson_r=0.7, spearman_rho=0.6, sign_accuracy=0.8,
                affected_auprc=0.4, local_wmae=0.2, mid_wmae=0.2, remote_wmae=0.2,
                local_n=1, mid_n=0, remote_n=0,
            ),
        ]
        agg = aggregate_metrics(records_metrics)
        assert agg["per_study"]["s1"]["skill"] == pytest.approx(0.5)
        assert agg["per_study"]["s2"]["skill"] == pytest.approx(0.5)
        assert agg["final"]["skill"] == pytest.approx(0.5)
        assert agg["final"]["n_studies"] == 2

    def test_nan_skill_excluded_from_aggregation(self):
        # One pair with NaN skill should not corrupt the parent macro.
        records_metrics = [
            PairMetrics(
                pair_id="a1", parent="pA", study="s1", n_positions=1, weight=1.0,
                wmae_pred=0.1, wmae_zero=0.2, skill=0.5, rmse=0.1,
                pearson_r=0.9, spearman_rho=0.8, sign_accuracy=0.7,
                affected_auprc=0.6, local_wmae=0.1, mid_wmae=0.1, remote_wmae=0.1,
                local_n=1, mid_n=0, remote_n=0,
            ),
            PairMetrics(
                pair_id="a2", parent="pA", study="s1", n_positions=0, weight=1.0,
                wmae_pred=float("nan"), wmae_zero=float("nan"), skill=float("nan"),
                rmse=float("nan"), pearson_r=float("nan"), spearman_rho=float("nan"),
                sign_accuracy=float("nan"), affected_auprc=float("nan"),
                local_wmae=float("nan"), mid_wmae=float("nan"), remote_wmae=float("nan"),
                local_n=0, mid_n=0, remote_n=0,
            ),
        ]
        agg = aggregate_metrics(records_metrics)
        # parentA skill = mean over non-NaN = 0.5
        assert agg["per_parent"]["pA"]["skill"] == pytest.approx(0.5)
        assert agg["per_parent"]["pA"]["n_pairs"] == 2
        assert agg["final"]["n_pairs_skill"] == 1


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------


class TestEvaluatePredictions:
    def test_missing_and_shape_errors(self):
        recs = [
            _make_record(pair_id="p1", delta=[0.1, 0.2, 0.3, 0.4], edit_arr_idx=3),
            _make_record(pair_id="p2", delta=[0.1, 0.2, 0.3, 0.4], edit_arr_idx=3),
        ]
        preds = {
            "p1": np.array([0.1, 0.2, 0.3, 0.0]),  # ok
            "p2": np.array([0.1, 0.2]),  # shape error
        }
        result = evaluate_predictions(recs, preds, baseline_name="test")
        assert result["n_pairs_evaluated"] == 1
        assert "p1" in [m["pair_id"] for m in result["aggregation"]["per_pair"]]
        assert len(result["shape_errors"]) == 1
        assert result["shape_errors"][0]["pair_id"] == "p2"

    def test_runtime_and_params_recorded(self):
        recs = [_make_record(pair_id="p1", delta=[0.1, 0.2, 0.3, 0.4], edit_arr_idx=3)]
        preds = {"p1": np.array([0.1, 0.2, 0.3, 0.0])}
        result = evaluate_predictions(
            recs, preds, baseline_name="test", runtime_seconds=1.23,
            peak_gpu_mb=None, param_count=42,
        )
        assert result["runtime_seconds"] == pytest.approx(1.23)
        assert result["param_count"] == 42
        assert result["baseline_name"] == "test"


# ---------------------------------------------------------------------------
# load_split_pairs with a synthetic registry
# ---------------------------------------------------------------------------


def _write_synthetic_registry(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a minimal registry + split_members + thermo manifest."""

    registry = {
        "schema_version": "reactflow-delta-d1-true-pair-registry-v1",
        "stage": "test",
        "candidate_total": 2,
        "true_pair_count": 2,
        "primary_eligible_count": 2,
        "registry": [
            {
                "rdat_path": "/fake/A.rdat",
                "rmdb_id": "A",
                "owner": "tester",
                "parent_prefix": "parentA",
                "citation_doi": "doiA",
                "wt_profile_index": 1,
                "mutant_profile_index": 2,
                "matched_mutation": {
                    "encoded_position_1indexed": 3,
                    "encoded_ref": "G",
                    "encoded_alt": "X",
                },
                "aligned_length": 4,
                "both_nonmissing": 4,
                "wt_reactivity_project": [0.1, 0.2, 0.3, 0.4],
                "mut_reactivity_project": [0.15, 0.25, 0.35, 0.45],
                "delta_reactivity_normalized": [0.05, 0.05, 0.05, 0.05],
                "primary_eligible": True,
                "true_pair": True,
                "pair_quality_weight": 1.0,
            },
            {
                "rdat_path": "/fake/B.rdat",
                "rmdb_id": "B",
                "owner": "tester",
                "parent_prefix": "parentB",
                "citation_doi": "doiB",
                "wt_profile_index": 1,
                "mutant_profile_index": 2,
                "matched_mutation": {
                    "encoded_position_1indexed": 2,
                    "encoded_ref": "A",
                    "encoded_alt": "X",
                },
                "aligned_length": 4,
                "both_nonmissing": 4,
                "wt_reactivity_project": [0.1, 0.2, 0.3, 0.4],
                "mut_reactivity_project": [0.1, 0.3, 0.3, 0.4],
                "delta_reactivity_normalized": [0.0, 0.1, 0.0, 0.0],
                "primary_eligible": True,
                "true_pair": True,
                "pair_quality_weight": 0.5,
            },
        ],
    }
    splits = {
        "schema_version": "reactflow-delta-ph0-split-members-v1",
        "split_method": "test",
        "study_assignment": {},
        "cross_contamination_check": {
            "all_disjoint": True, "test_in_train": 0, "test_in_val": 0, "val_in_train": 0,
        },
        "train": {"n_pairs": 1, "pair_ids": ["A.rdat:1:2:3"], "parents": ["parentA"], "sha256": "x"},
        "validation": {"n_pairs": 0, "pair_ids": [], "parents": [], "sha256": "y"},
        "test": {"n_pairs": 1, "pair_ids": ["B.rdat:1:2:2"], "parents": ["parentB"],
                  "sha256": "z", "frozen": True, "used_in_ph0_audit": False},
    }
    thermo = {
        "schema_version": "reactflow-delta-ph0-thermo-features-manifest-v1",
        "stage": "PH0",
        "per_pair": [
            {
                "pair_id": "A.rdat:1:2:3",
                "parent_prefix": "parentA",
                "citation_doi": "doiA",
                "rdat_file": "A.rdat",
                "edit_arr_idx": 2,
                "encoded_position_1indexed": 3,
                "encoded_ref": "G",
                "aligned_length": 4,
                "wt_features": {"bpp_paired_prob": 0.9, "n_contacts": 1},
            },
            {
                "pair_id": "B.rdat:1:2:2",
                "parent_prefix": "parentB",
                "citation_doi": "doiB",
                "rdat_file": "B.rdat",
                "edit_arr_idx": 1,
                "encoded_position_1indexed": 2,
                "encoded_ref": "A",
                "aligned_length": 4,
                "wt_features": {"bpp_paired_prob": 0.1, "n_contacts": 0},
            },
        ],
    }
    rp = tmp_path / "registry.json"
    sp = tmp_path / "splits.json"
    tp = tmp_path / "thermo.json"
    rp.write_text(json.dumps(registry))
    sp.write_text(json.dumps(splits))
    tp.write_text(json.dumps(thermo))
    return rp, sp, tp


class TestLoadSplitPairs:
    def test_load_test_split(self, tmp_path):
        rp, sp, tp = _write_synthetic_registry(tmp_path)
        records = load_split_pairs(
            "test",
            registry_path=rp,
            split_members_path=sp,
            thermo_manifest_path=tp,
        )
        assert len(records) == 1
        rec = records[0]
        assert rec.pair_id == "B.rdat:1:2:2"
        assert rec.parent == "parentB"
        assert rec.study == "doiB"
        assert rec.edit_arr_idx == 1
        assert rec.edit_pos_1indexed == 2
        assert rec.encoded_ref == "A"
        assert rec.aligned_length == 4
        assert rec.pair_quality_weight == pytest.approx(0.5)
        # Endpoint mask: exclude edit position (idx 1), all non-missing.
        assert rec.endpoint_mask.tolist() == [True, False, True, True]
        assert rec.wt_features is not None
        assert rec.wt_features["bpp_paired_prob"] == pytest.approx(0.1)

    def test_load_train_split(self, tmp_path):
        rp, sp, tp = _write_synthetic_registry(tmp_path)
        records = load_split_pairs(
            "train",
            registry_path=rp,
            split_members_path=sp,
            thermo_manifest_path=tp,
        )
        assert len(records) == 1
        assert records[0].pair_id == "A.rdat:1:2:3"

    def test_load_with_missing_in_delta(self, tmp_path):
        rp, sp, tp = _write_synthetic_registry(tmp_path)
        # Patch the registry so one delta value is None.
        doc = json.loads(rp.read_text())
        doc["registry"][1]["delta_reactivity_normalized"] = [0.0, None, 0.0, 0.0]
        doc["registry"][1]["wt_reactivity_project"] = [0.1, 0.2, 0.3, 0.4]
        doc["registry"][1]["mut_reactivity_project"] = [0.1, None, 0.3, 0.4]
        rp.write_text(json.dumps(doc))
        records = load_split_pairs(
            "test", registry_path=rp, split_members_path=sp, thermo_manifest_path=tp,
        )
        rec = records[0]
        # edit at idx 1 (excluded), idx 1 has None delta and None mut -> excluded
        assert rec.endpoint_mask.tolist() == [True, False, True, True]

    def test_invalid_split_name(self, tmp_path):
        rp, sp, tp = _write_synthetic_registry(tmp_path)
        with pytest.raises(ValueError):
            load_split_pairs("bogus", registry_path=rp, split_members_path=sp)
