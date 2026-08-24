import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

from scripts.reactflow_delta.merge_model_rescue_v7_probe import merge_folds
from scripts.reactflow_delta.model_rescue_v7_probe import (
    DependencyFeatureCache,
    accumulate_candidate_train_stats,
)
from scripts.reactflow_delta.model_rescue_v7_schema import CACHE_SCHEMA, FEATURE_NAMES
from scripts.reactflow_delta.qualify_model_rescue_v7_probe import (
    ABSOLUTE_RELATIVE_GAIN_MIN,
    SIGNED_POSITIVE_PUZZLES_MIN,
    SIGNED_RELATIVE_GAIN_MIN,
    qualify,
)
from scripts.reactflow_delta.run_model_rescue_v7_probe import (
    CORRECTED_REFERENCE_SCHEMA,
    FOLD_SCHEMA,
    PREDICTION_SCHEMA,
    assert_probe_authority,
    assert_corrected_baseline_replay,
    predict_registered_held,
)
from scripts.reactflow_delta.score_model_rescue_v7_probe import (
    SCHEMA as SCORE_SCHEMA,
    assert_score_authority,
)


def _dependency_cache(path: Path, *, full_pos: int = 1) -> None:
    string = h5py.string_dtype(encoding="utf-8")
    features = np.ones((1, 3, len(FEATURE_NAMES)), dtype=np.float32)
    features[0, full_pos] = 0.0
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = CACHE_SCHEMA
        handle.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
        for name, value in (
            ("puzzle", "P01"),
            ("method", "Starting sequence"),
            ("ref", "T"),
            ("alt", "A"),
        ):
            handle.create_dataset(name, data=np.asarray([value], dtype=object), dtype=string)
        handle.create_dataset("design_pos", data=np.asarray([0], dtype=np.int16))
        handle.create_dataset("full_pos", data=np.asarray([full_pos], dtype=np.int16))
        handle.create_dataset("features", data=features)


def test_dependency_cache_uses_canonical_biological_key_and_source_coordinate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dependency.h5"
    _dependency_cache(path)
    cache = DependencyFeatureCache(path)
    try:
        record = SimpleNamespace(
            puzzle="P01",
            method="Starting sequence",
            design_pos=0,
            ref="U",
            alt="A",
            full_pos=1,
        )
        value = cache.get(record)
        assert value.shape == (3, 6)
        assert np.array_equal(value[1], np.zeros(6, dtype=np.float32))
        record.full_pos = 2
        with pytest.raises(ValueError, match="source coordinate differs"):
            cache.get(record)
    finally:
        cache.close()


class _ArrayCache:
    def __init__(self, width: int) -> None:
        self.width = width

    def get(self, _record: SimpleNamespace) -> np.ndarray:
        return np.zeros((3, self.width), dtype=np.float32)


class _TrainingUniverse:
    def __init__(self, targets: dict[str, np.ndarray | None]) -> None:
        self.targets = targets
        self.constructs = {
            "cell_a": SimpleNamespace(
                sequence="AUG",
                wt_observed=np.asarray([True, True, True]),
                wt_reactivity=np.asarray([0.1, 0.2, 0.3]),
                wt_error=np.asarray([0.1, 0.1, 0.1]),
                region_map=np.asarray(["design_region", "other", "other"]),
            ),
            "cell_b": SimpleNamespace(
                sequence="AUG",
                wt_observed=np.asarray([True, True, True]),
                wt_reactivity=np.asarray([0.1, 0.2, 0.3]),
                wt_error=np.asarray([0.1, 0.1, 0.1]),
                region_map=np.asarray(["design_region", "other", "other"]),
            ),
        }

    def get_construct(self, construct_id: str) -> SimpleNamespace:
        return self.constructs[construct_id]

    def mutant_full_profile(
        self, wt_id: str, _design_pos: int, _ref: str, _alt: str
    ) -> tuple[np.ndarray | None, None]:
        return self.targets[wt_id], None


def _record(cell: str, wt_id: str, *, design_pos: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        construct_id=cell,
        wt_id=wt_id,
        puzzle="P01",
        method=cell,
        design_pos=design_pos,
        full_pos=design_pos,
        ref="A",
        alt="U",
    )


def test_v7_candidate_stats_preserve_cell_weight_and_exclude_missing_target() -> None:
    targets = {
        "a1": np.asarray([0.2, 0.3, 0.4]),
        "a2": np.asarray([0.3, 0.4, 0.5]),
        "a3": np.asarray([0.2, 0.3, 0.4]),
        "b1": np.asarray([0.15, 0.25, 0.35]),
        "missing": None,
    }
    univ = _TrainingUniverse(targets)
    unconstrained = _ArrayCache(12)
    constrained = _ArrayCache(11)
    dependency = _ArrayCache(6)

    records = [_record("cell_a", "a1"), _record("cell_a", "a2"), _record("cell_b", "b1")]
    stats, counts = accumulate_candidate_train_stats(
        univ, records, unconstrained, constrained, dependency
    )
    assert stats.sum_weight == pytest.approx(2.0)
    assert counts["n_train_cells"] == 2
    assert counts["n_train_valid_mutants"] == 3

    duplicated, duplicated_counts = accumulate_candidate_train_stats(
        univ,
        records + [_record("cell_a", "a3"), _record("cell_b", "missing")],
        unconstrained,
        constrained,
        dependency,
    )
    assert duplicated.sum_weight == pytest.approx(2.0)
    assert duplicated_counts["n_train_cells"] == 2
    assert duplicated_counts["n_train_valid_mutants"] == 4


def _constant_model(width: int, signed: float, absolute: float) -> dict:
    return {
        "mean_x": np.zeros(width, dtype=np.float64),
        "scale_x": np.ones(width, dtype=np.float64),
        "mean_y": np.asarray([signed, absolute], dtype=np.float64),
        "coefficient": np.zeros((width, 2), dtype=np.float64),
        "alpha": 1.0,
    }


class _PredictionUniverse:
    def __init__(self) -> None:
        self.construct = SimpleNamespace(
            sequence="AUG",
            wt_observed=np.asarray([True, True, True]),
            wt_reactivity=np.asarray([0.1, 0.2, 0.3]),
            wt_error=np.asarray([0.1, 0.1, 0.1]),
            region_map=np.asarray(["design_region", "other", "other"]),
        )

    def get_construct(self, _construct_id: str) -> SimpleNamespace:
        return self.construct

    def mutant_full_profile(self, *_args: object) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("held target access entered the V7M2 prediction path")


def test_v7_registered_prediction_path_is_held_target_invariant() -> None:
    univ = _PredictionUniverse()
    record = _record("cell_a", "held")
    prediction = predict_registered_held(
        univ,
        [record],
        _ArrayCache(12),
        _ArrayCache(11),
        _ArrayCache(6),
        _constant_model(41, signed=0.1, absolute=0.2),
        _constant_model(47, signed=0.3, absolute=0.4),
        outer_fold=0,
    )
    assert len(prediction["keys"]) == 3
    assert np.array_equal(prediction["baseline_signed_delta"], np.full(3, 0.1))
    assert np.array_equal(prediction["baseline_absolute_delta"], np.full(3, 0.2))
    assert np.array_equal(prediction["candidate_signed_delta"], np.full(3, 0.3))
    assert np.array_equal(prediction["candidate_absolute_delta"], np.full(3, 0.4))


def test_corrected_feature41_replay_requires_exact_key_order_and_predictions(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.npz"
    keys = np.asarray(["P01|method|mutant|0", "P01|method|mutant|1"], dtype=object)
    np.savez_compressed(
        reference,
        schema_version=np.asarray(CORRECTED_REFERENCE_SCHEMA),
        keys=keys,
        v6_feature41_signed_delta=np.asarray([0.1, -0.2]),
        v6_feature41_absolute_delta=np.asarray([0.1, 0.2]),
    )
    prediction = {
        "keys": keys.copy(),
        "baseline_signed_delta": np.asarray([0.1, -0.2]),
        "baseline_absolute_delta": np.asarray([0.1, 0.2]),
    }
    assert_corrected_baseline_replay(prediction, reference)
    prediction["baseline_signed_delta"][1] += 1e-8
    with pytest.raises(ValueError, match="corrected baseline replay failed"):
        assert_corrected_baseline_replay(prediction, reference)


def _write_fold(root: Path, fold: int, *, include_target: bool = False) -> None:
    prediction_path = root / f"v7_probe_predictions_fold{fold}.npz"
    model_path = root / f"v7_probe_models_fold{fold}.json"
    reference_path = root / f"tic2a_corrected_predictions_fold{fold}.npz"
    key = np.asarray([f"P{fold + 1:02d}|method|mutant|0"], dtype=object)
    values = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": key,
        "biological_scoring_key": key.copy(),
        "outer_fold": np.asarray([fold], dtype=np.int64),
        "registered_status": np.asarray(["covered"], dtype=object),
        "baseline_signed_delta": np.asarray([0.0]),
        "baseline_absolute_delta": np.asarray([0.0]),
        "candidate_signed_delta": np.asarray([0.0]),
        "candidate_absolute_delta": np.asarray([0.0]),
    }
    if include_target:
        values["target"] = np.asarray([0.0])
    np.savez_compressed(prediction_path, **values)
    model_path.write_text("{}\n")
    reference_path.write_text("reference\n")
    row = {
        "schema_version": FOLD_SCHEMA,
        "phase": "V7M2",
        "outer_fold": fold,
        "held_puzzle": f"P{fold + 1:02d}",
        "prediction_artifact": str(prediction_path),
        "model_artifact": str(model_path),
        "corrected_baseline_reference": str(reference_path),
        "n_registered_prediction_rows": 1,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "corrected_feature41_replay_pass": True,
        "held_target_used_for_prediction": False,
        "held_score_computed": False,
        "partial_score_inspected": False,
        "model_selection_performed": False,
        "legacy_target_dependent_prediction_reused": False,
        "external_outcome_accessed": False,
    }
    (root / f"v7_probe_fold_result_fold{fold}.json").write_text(
        json.dumps(row) + "\n"
    )


def test_v7_probe_merge_requires_complete_prediction_only_universe(tmp_path: Path) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold)
    merged = merge_folds(tmp_path)
    assert merged["status"] == "V7M2_COMPLETE_UNSCORED_MERGE_PASS"
    assert merged["merge_integrity"]["corrected_feature41_replay_all_folds"] is True

    (tmp_path / "v7_probe_fold_result_fold19.json").unlink()
    with pytest.raises(ValueError, match="fold universe incomplete"):
        merge_folds(tmp_path)


def test_v7_probe_merge_rejects_target_side_prediction_fields(tmp_path: Path) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold, include_target=fold == 0)
    with pytest.raises(ValueError, match="target-side fields"):
        merge_folds(tmp_path)


def _complete_scores(*, signed_gain: float, absolute_gain: float) -> dict:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "baseline_signed_delta_mae": 1.0,
                "candidate_signed_delta_mae": 1.0 - signed_gain,
                "baseline_absolute_delta_mae": 1.0,
                "candidate_absolute_delta_mae": 1.0 - absolute_gain,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
            }
        )
    return {
        "schema_version": SCORE_SCHEMA,
        "status": "V7M2_COMPLETE_CORRECTED_SCORE_PASS",
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "partial_fold_scores_inspected": False,
        "model_selection_performed": False,
        "scores": rows,
    }


def test_v7_probe_qualifier_applies_frozen_eligibility_gate() -> None:
    passed = qualify(_complete_scores(signed_gain=0.02, absolute_gain=-0.004))
    assert passed["status"] == "V7M2_RINALMO_DEPENDENCY_SIGNAL_ELIGIBLE"
    assert all(passed["checks"].values())
    assert passed["candidate_model_trained"] is False
    assert passed["sota"] == "NOT_ESTABLISHED"

    failed = qualify(_complete_scores(signed_gain=0.009, absolute_gain=0.0))
    assert failed["status"] == "V7M2_RINALMO_DEPENDENCY_SIGNAL_NOT_ELIGIBLE"
    assert failed["checks"]["signed_delta_relative_gain_at_least_one_percent"] is False


def test_v7_probe_constants_match_the_machine_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    contract = yaml.safe_load(
        (root / "configs/reactflow_delta/model_rescue_v7_amendment.yaml").read_text()
    )
    gate = contract["eligibility_probe"]["gate"]
    assert SIGNED_RELATIVE_GAIN_MIN == gate["signed_delta_relative_mae_gain_min"]
    assert SIGNED_POSITIVE_PUZZLES_MIN == gate["signed_delta_positive_puzzles_min"]
    assert ABSOLUTE_RELATIVE_GAIN_MIN == gate[
        "absolute_delta_relative_mae_guardrail_min"
    ]


def _write_v7m2_authority(
    root: Path,
    *,
    held_score: bool,
    partial_score: bool = False,
    external_outcome: bool = False,
) -> None:
    path = root / "configs/reactflow_delta/active_contract.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "authority": {"current_phase": "V7M2"},
                "training_allowed": "FIXED_CORRECTED_WEIGHTED_RIDGE_ELIGIBILITY_ONLY",
                "held_score_read_allowed": held_score,
                "partial_fold_score_read_allowed": partial_score,
                "new_external_outcome_access_allowed": external_outcome,
            }
        )
    )


def test_v7m2_prediction_and_complete_score_authorities_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    _write_v7m2_authority(tmp_path, held_score=False)
    assert_probe_authority(tmp_path)
    with pytest.raises(RuntimeError, match="score access is closed"):
        assert_score_authority(tmp_path)

    _write_v7m2_authority(tmp_path, held_score=True)
    assert_score_authority(tmp_path)
    with pytest.raises(RuntimeError, match="held scores closed"):
        assert_probe_authority(tmp_path)

    _write_v7m2_authority(tmp_path, held_score=True, partial_score=True)
    with pytest.raises(RuntimeError, match="partial V7M2 scores"):
        assert_score_authority(tmp_path)

    _write_v7m2_authority(tmp_path, held_score=True, external_outcome=True)
    with pytest.raises(RuntimeError, match="external outcomes locked"):
        assert_score_authority(tmp_path)
