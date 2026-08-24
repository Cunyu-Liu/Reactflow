from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from scripts.reactflow_delta.merge_model_rescue_v6_probe import merge_folds
from scripts.reactflow_delta.model_rescue_v5_probe import (
    BASELINE_FEATURE_NAMES as DIRECT_FEATURE_NAMES,
)
from scripts.reactflow_delta.model_rescue_v5_schema import (
    CACHE_SCHEMA as V5_CACHE_SCHEMA,
    FEATURE_NAMES as UNCONSTRAINED_FEATURE_NAMES,
)
from scripts.reactflow_delta.model_rescue_v6_probe import (
    BASELINE_PROBE_FEATURE_NAMES,
    CANDIDATE_PROBE_FEATURE_NAMES,
    ConstrainedFeatureCache,
    validate_cache_alignment,
)
from scripts.reactflow_delta.model_rescue_v6_schema import (
    CACHE_SCHEMA as V6_CACHE_SCHEMA,
    FEATURE_NAMES as CONSTRAINED_CACHE_FEATURE_NAMES,
    PROBE_FEATURE_INDICES,
    PROBE_FEATURE_NAMES,
)
from scripts.reactflow_delta.qualify_model_rescue_v6_probe import qualify
from scripts.reactflow_delta.run_model_rescue_v5_probe import (
    PREDICTION_SCHEMA as V5_PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.run_model_rescue_v6_probe import (
    FOLD_SCHEMA,
    PREDICTION_SCHEMA,
    assert_probe_authority,
    assert_v5_baseline_replay,
    predict_registered_held,
)
from scripts.reactflow_delta.score_model_rescue_v6_probe import (
    _puzzle_macro,
    assert_score_authority,
)


@dataclass
class _Record:
    puzzle: str = "P1"
    method: str = "M"
    construct_id: str = "P1_M"
    design_pos: int = 1
    full_pos: int = 1
    ref: str = "C"
    alt: str = "U"


def _write_cache(
    path: Path,
    *,
    schema: str,
    feature_names: tuple[str, ...],
    features: np.ndarray,
) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = schema
        handle.attrs["feature_names"] = json.dumps(feature_names)
        for name, value in (
            ("row_id", "raw_mm_1_C_U"),
            ("puzzle", "P1"),
            ("method", "M"),
            ("ref", "C"),
            ("alt", "U"),
        ):
            handle.create_dataset(
                name,
                data=np.asarray([value], dtype=object),
                dtype=string_dtype,
            )
        handle.create_dataset("design_pos", data=np.asarray([1], dtype=np.int64))
        handle.create_dataset("features", data=features.astype(np.float32))


def test_constrained_cache_exposes_only_the_frozen_independent_basis(
    tmp_path: Path,
) -> None:
    path = tmp_path / "constrained.h5"
    full = np.arange(3 * 12, dtype=np.float32).reshape(1, 3, 12)
    _write_cache(
        path,
        schema=V6_CACHE_SCHEMA,
        feature_names=CONSTRAINED_CACHE_FEATURE_NAMES,
        features=full,
    )
    cache = ConstrainedFeatureCache(path)
    try:
        selected = cache.get(_Record())
        assert selected.shape == (3, 11)
        assert np.array_equal(selected, full[0][:, PROBE_FEATURE_INDICES])
        assert len(PROBE_FEATURE_NAMES) == 11
    finally:
        cache.close()


def test_cache_alignment_requires_identical_biological_universe(tmp_path: Path) -> None:
    from scripts.reactflow_delta.model_rescue_v5_probe import EnsembleFeatureCache

    unconstrained_path = tmp_path / "unconstrained.h5"
    constrained_path = tmp_path / "constrained.h5"
    _write_cache(
        unconstrained_path,
        schema=V5_CACHE_SCHEMA,
        feature_names=UNCONSTRAINED_FEATURE_NAMES,
        features=np.zeros((1, 3, 12), dtype=np.float32),
    )
    _write_cache(
        constrained_path,
        schema=V6_CACHE_SCHEMA,
        feature_names=CONSTRAINED_CACHE_FEATURE_NAMES,
        features=np.zeros((1, 3, 12), dtype=np.float32),
    )
    unconstrained = EnsembleFeatureCache(unconstrained_path)
    constrained = ConstrainedFeatureCache(constrained_path)
    try:
        result = validate_cache_alignment(unconstrained, constrained)
        assert result == {
            "biological_key_universe_equal": True,
            "registered_mutants": 1,
            "receiver_length": 3,
            "unconstrained_width": 12,
            "constrained_cache_width": 12,
            "constrained_probe_width": 11,
        }
    finally:
        unconstrained.close()
        constrained.close()


def _write_authority(repo_root: Path, *, phase: str, held_score: bool) -> None:
    path = repo_root / "configs" / "reactflow_delta" / "active_contract.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "authority": {"current_phase": phase},
                "training_allowed": "FIXED_WEIGHTED_RIDGE_ELIGIBILITY_ONLY",
                "held_score_read_allowed": held_score,
                "partial_fold_score_read_allowed": False,
                "new_external_outcome_access_allowed": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_authority_blocks_probe_and_score_until_their_exact_substates(
    tmp_path: Path,
) -> None:
    _write_authority(tmp_path, phase="V6M1", held_score=False)
    with pytest.raises(RuntimeError, match="closed outside active V6M2"):
        assert_probe_authority(tmp_path)

    _write_authority(tmp_path, phase="V6M2", held_score=False)
    assert_probe_authority(tmp_path)
    with pytest.raises(RuntimeError, match="score access has not been opened"):
        assert_score_authority(tmp_path)

    _write_authority(tmp_path, phase="V6M2", held_score=True)
    with pytest.raises(RuntimeError, match="requires held scores closed"):
        assert_probe_authority(tmp_path)
    assert_score_authority(tmp_path)


def test_held_prediction_is_full_output_and_never_calls_target_accessor() -> None:
    class Construct:
        sequence = "ACG"
        wt_reactivity = np.asarray([0.1, 0.2, 0.3])
        wt_error = np.asarray([0.1, 0.1, 0.1])
        wt_observed = np.asarray([True, False, True])
        region_map = np.asarray(["design_region"] * 3)

    class Universe:
        def get_construct(self, _construct_id):
            return Construct()

        def mutant_full_profile(self, *_args, **_kwargs):
            raise AssertionError("held target accessor entered v6 prediction path")

    class Unconstrained:
        def get(self, _record):
            return np.zeros((3, 12), dtype=np.float32)

    class Constrained:
        def get(self, _record):
            return np.zeros((3, 11), dtype=np.float32)

    def zero_model(width: int) -> dict:
        return {
            "mean_x": np.zeros(width),
            "scale_x": np.ones(width),
            "mean_y": np.zeros(2),
            "coefficient": np.zeros((width, 2)),
            "alpha": 1.0,
        }

    result = predict_registered_held(
        Universe(),
        [_Record()],
        Unconstrained(),
        Constrained(),
        zero_model(len(BASELINE_PROBE_FEATURE_NAMES)),
        zero_model(len(CANDIDATE_PROBE_FEATURE_NAMES)),
        outer_fold=0,
    )
    assert len(DIRECT_FEATURE_NAMES) == 18
    assert len(BASELINE_PROBE_FEATURE_NAMES) == 30
    assert len(CANDIDATE_PROBE_FEATURE_NAMES) == 41
    assert len(result["keys"]) == len(Construct.sequence)
    assert set(result["registered_status"]) == {"covered"}
    assert {
        "target",
        "target_error",
        "target_mask",
        "score",
        "mae",
        "crps",
    }.isdisjoint(result)


def test_v6_baseline_must_replay_v5_candidate(tmp_path: Path) -> None:
    keys = np.asarray(["k1", "k2"], dtype=object)
    prediction = {
        "keys": keys,
        "baseline_signed_delta": np.asarray([0.1, -0.2]),
        "baseline_absolute_delta": np.asarray([0.1, 0.2]),
    }
    reference = tmp_path / "v5.npz"
    np.savez_compressed(
        reference,
        schema_version=np.asarray(V5_PREDICTION_SCHEMA),
        keys=keys,
        candidate_signed_delta=np.asarray([0.1, -0.2]),
        candidate_absolute_delta=np.asarray([0.1, 0.2]),
    )
    assert_v5_baseline_replay(prediction, reference)
    prediction["baseline_signed_delta"][0] += 1e-6
    with pytest.raises(ValueError, match="failed v5 candidate replay"):
        assert_v5_baseline_replay(prediction, reference)


def _key(method: str, design_pos: int, mutation: str, position: int) -> str:
    return f"openknot_m2|P1|{method}|P1_{method}|{design_pos}|{mutation}|{position}"


def test_score_hierarchy_keeps_same_allele_at_different_positions_distinct() -> None:
    losses = {
        _key("A", 3, "A>G", 0): 0.0,
        _key("A", 3, "A>G", 1): 0.0,
        _key("A", 7, "A>G", 0): 2.0,
        _key("B", 2, "G>A", 0): 1.0,
    }
    assert _puzzle_macro(losses) == pytest.approx(1.0)
    assert np.mean(list(losses.values())) == pytest.approx(0.75)


def _write_prediction(path: Path, fold: int, *, include_target: bool = False) -> None:
    key = f"openknot_m2|P{fold:02d}|M|P{fold:02d}_M|0|A>G|0"
    fields = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray([key], dtype=object),
        "biological_scoring_key": np.asarray([key], dtype=object),
        "outer_fold": np.asarray([fold], dtype=np.int64),
        "baseline_signed_delta": np.asarray([0.0]),
        "baseline_absolute_delta": np.asarray([0.0]),
        "candidate_signed_delta": np.asarray([0.1]),
        "candidate_absolute_delta": np.asarray([0.1]),
        "registered_status": np.asarray(["covered"], dtype=object),
    }
    if include_target:
        fields["target"] = np.asarray([0.2])
    np.savez_compressed(path, **fields)


def _write_fold(tmp_path: Path, fold: int, *, include_target: bool = False) -> None:
    prediction = tmp_path / f"v6_probe_predictions_fold{fold}.npz"
    model = tmp_path / f"v6_probe_models_fold{fold}.json"
    reference = tmp_path / f"v5_probe_predictions_fold{fold}.npz"
    _write_prediction(prediction, fold, include_target=include_target)
    model.write_text("{}\n", encoding="utf-8")
    reference.write_text("reference\n", encoding="utf-8")
    result = {
        "schema_version": FOLD_SCHEMA,
        "phase": "V6M2",
        "outer_fold": fold,
        "held_puzzle": f"P{fold:02d}",
        "prediction_artifact": str(prediction),
        "model_artifact": str(model),
        "v5_reference_prediction": str(reference),
        "n_registered_prediction_rows": 1,
        "v5_baseline_replay_pass": True,
        "held_score_computed": False,
        "partial_score_inspected": False,
        "external_outcome_accessed": False,
    }
    (tmp_path / f"v6_probe_fold_result_fold{fold}.json").write_text(
        json.dumps(result), encoding="utf-8"
    )


def test_merge_requires_complete_prediction_only_replayed_universe(
    tmp_path: Path,
) -> None:
    for fold in range(20):
        _write_fold(tmp_path, fold)
    merged = merge_folds(tmp_path)
    assert merged["status"] == "V6M2_COMPLETE_UNSCORED_MERGE_PASS"
    assert merged["merge_integrity"]["v5_baseline_replay_all_folds"] is True


def test_merge_rejects_incomplete_or_target_contaminated_artifacts(
    tmp_path: Path,
) -> None:
    for fold in range(19):
        _write_fold(tmp_path, fold)
    with pytest.raises(ValueError, match="incomplete"):
        merge_folds(tmp_path)
    _write_fold(tmp_path, 19, include_target=True)
    with pytest.raises(ValueError, match="target-side"):
        merge_folds(tmp_path)


def _score_rows(signed_gain: float, absolute_gain: float) -> dict:
    return {
        "status": "V6M2_COMPLETE_SCORE_PASS",
        "v5_baseline_replay_all_folds": True,
        "scores": [
            {
                "outer_fold": fold,
                "baseline_signed_delta_mae": 0.2,
                "candidate_signed_delta_mae": 0.2 - signed_gain,
                "baseline_absolute_delta_mae": 0.1,
                "candidate_absolute_delta_mae": 0.1 - absolute_gain,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
            }
            for fold in range(20)
        ],
    }


def test_qualifier_opens_only_the_frozen_v6m2_gate() -> None:
    passed = qualify(_score_rows(signed_gain=0.003, absolute_gain=0.0))
    assert passed["overall_status"] == "V6M2_CONSTRAINED_SIGNAL_ELIGIBLE"
    assert passed["v6m3_authorized"] is True
    failed = qualify(_score_rows(signed_gain=0.001, absolute_gain=0.0))
    assert failed["overall_status"] == "MODEL_RESCUE_V6_FAIL"
    assert failed["v6m3_authorized"] is False


def test_qualifier_rejects_incomplete_or_nonreplayed_score() -> None:
    scores = _score_rows(signed_gain=0.003, absolute_gain=0.0)
    scores["scores"] = scores["scores"][:-1]
    with pytest.raises(ValueError, match="complete"):
        qualify(scores)
    scores = _score_rows(signed_gain=0.003, absolute_gain=0.0)
    scores["v5_baseline_replay_all_folds"] = False
    with pytest.raises(ValueError, match="baseline replay"):
        qualify(scores)
