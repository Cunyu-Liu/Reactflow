from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.reactflow_delta.merge_puzzle_set_meta_context_probe as merger

from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import (
    MERGED_SCHEMA,
    PRODUCTION_MERGE_UNIVERSES,
    assert_production_merge_universe,
    merge_complete_universe,
)
from scripts.reactflow_delta.puzzle_set_meta_context import (
    FULL_CROSS_CONSTRUCT,
    POINT_CONTEXT_LR,
    POINT_GRADIENT_CLIP,
    POINT_HEAD_LR,
    POINT_HEAD_WARMUP_EPOCHS,
    POSITION_ALIGNED_OPERATOR,
    POSITION_DERANGEMENT_SHIFT,
    POSITION_DERANGED_NULL,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import PREDICTION_SCHEMA
from scripts.reactflow_delta.puzzle_set_meta_context_retention import (
    RETENTION_DIAGNOSTIC_EPOCH,
    RETENTION_SCHEMA,
)
from scripts.reactflow_delta.puzzle_set_safe_sources import FROZEN_INPUT_SOURCE_SPEC
from scripts.reactflow_delta.run_puzzle_set_meta_context_probe import (
    FOLD_SCHEMA,
    frozen_input_sources_for_fold,
)


def _retention_diagnostic(
    *,
    arm: str,
    held_puzzle: str,
    puzzle_ids: list[str],
    training_epochs: int,
) -> dict[str, object]:
    metric_values = (1.0, 0.6, 0.7) if arm == "candidate" else (0.95, 0.65, 0.72)
    metric_names = (
        "initial_context_l1",
        "post_pretraining_l1",
        "post_point_l1",
    )
    per_puzzle = [
        {
            "puzzle": puzzle,
            "eligible_constructs": 8,
            **dict(zip(metric_names, metric_values)),
        }
        for puzzle in sorted(puzzle_ids)
    ]
    means = {
        metric: float(np.mean([row[metric] for row in per_puzzle]))
        for metric in metric_names
    }
    pretraining_gain = means["initial_context_l1"] - means["post_pretraining_l1"]
    retained_fraction = (
        means["initial_context_l1"] - means["post_point_l1"]
    ) / pretraining_gain
    return {
        "schema_version": RETENTION_SCHEMA,
        "arm": arm,
        "evidence_status": "OUTER_TRAIN_WT_RETENTION_DIAGNOSTIC_ONLY",
        "diagnostic_epoch": RETENTION_DIAGNOSTIC_EPOCH,
        "training_mask_epochs": [0, training_epochs - 1],
        "held_puzzle": held_puzzle,
        "outer_train_puzzle_ids": sorted(puzzle_ids),
        "per_puzzle": per_puzzle,
        "mean": means,
        "retained_fraction": retained_fraction,
        "pretraining_established": True,
        "retention_positive": True,
        "same_final_frozen_decoder": True,
        "diagnostic_mask_disjoint_from_training": True,
        "mutant_outcome_used": False,
        "held_puzzle_accessed": False,
        "checkpoint_selection_performed": False,
        "learning_rate_selection_performed": False,
    }


def _write_fold(
    directory: Path,
    *,
    fold: int,
    seed: int = 0,
    epochs: int = 1,
    target_field: bool = False,
    short_candidate: bool = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    keys = np.asarray([f"k{fold}-0", f"k{fold}-1"], dtype=object)
    prediction = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": keys,
        "biological_scoring_key": keys.copy(),
        "outer_fold": np.full(2, fold, dtype=np.int64),
        "seed": np.full(2, seed, dtype=np.int64),
        "registered_status": np.full(2, "covered", dtype=object),
        "feature41_point": np.zeros(2),
        "parent_point": np.zeros(2),
        "candidate_point": np.zeros(2),
        "null_point": np.zeros(2),
    }
    for name in ("candidate", "null"):
        prediction[f"{name}_weights"] = np.full((2, 2), 0.5)
        prediction[f"{name}_locations"] = np.zeros((2, 2))
        prediction[f"{name}_scales"] = np.tile([0.1, 0.2], (2, 1))
        prediction[f"{name}_expected_absolute_delta"] = np.zeros(2)
    if short_candidate:
        prediction["candidate_point"] = np.zeros(1)
    if target_field:
        prediction["target"] = np.zeros(2)
    prediction_path = directory / f"prediction{fold}_{seed}.npz"
    np.savez_compressed(prediction_path, **prediction)
    checkpoints = {}
    for stage in ("point", "decoder", "residual"):
        checkpoints[stage] = {}
        for name in ("candidate", "null"):
            path = directory / f"{name}_{stage}{fold}_{seed}.pt"
            path.write_bytes(f"{name}-{stage}".encode())
            checkpoints[stage][name] = str(path)
    frozen_parents = {
        "v13_point": str(directory / f"v13_candidate_point_fold{fold}_seed0.pt"),
        "v14_encoder": str(directory / f"v14_candidate_point_fold{fold}_seed0.pt"),
    }
    for name, raw_path in frozen_parents.items():
        Path(raw_path).write_bytes(name.encode())
    source_paths = {
        "v8_meanaligned_checkpoint": (
            directory / f"v8_corrected_mean_fold{fold}_seed0.pt"
        ),
        "tic2a_feature41_model_artifact": (
            directory / f"tic2a_corrected_models_fold{fold}.json"
        ),
        "tic2a_merged_registry": directory / "tic2a_merged.json",
        "unconstrained_feature_cache": directory / "unconstrained.h5",
        "constrained_feature_cache": directory / "constrained.h5",
    }
    for path in source_paths.values():
        path.touch()
    frozen_input_sources = frozen_input_sources_for_fold(
        outer_fold=fold,
        v13_point_checkpoint=Path(frozen_parents["v13_point"]),
        v14_encoder_checkpoint=Path(frozen_parents["v14_encoder"]),
        **source_paths,
    )
    held_puzzle = f"P{fold + 1:02d}"
    outer_train_puzzle_ids = [f"train{fold}_{index}" for index in range(19)]
    point_training_summary = {
        "optimizer_steps": 19 * epochs,
        "head_update_steps": 19 * epochs,
        "context_update_steps": 19 * max(epochs - POINT_HEAD_WARMUP_EPOCHS, 0),
        "target_exposures_per_available_cell": epochs,
        "head_only_warmup_epochs": POINT_HEAD_WARMUP_EPOCHS,
        "head_learning_rate": POINT_HEAD_LR,
        "context_learning_rate": POINT_CONTEXT_LR,
        "gradient_clip": POINT_GRADIENT_CLIP,
        "warmup_context_unchanged": True,
        "best_epoch_selection_performed": False,
    }
    row = {
        "schema_version": FOLD_SCHEMA,
        "phase": "P1M3",
        "outer_fold": fold,
        "held_puzzle": held_puzzle,
        "seed": seed,
        "pretraining_epochs": epochs,
        "point_epochs": epochs,
        "calibration_epochs": epochs,
        "candidate_connectivity": FULL_CROSS_CONSTRUCT,
        "null_connectivity": POSITION_DERANGED_NULL,
        "cross_construct_operator": POSITION_ALIGNED_OPERATOR,
        "position_derangement_shift": POSITION_DERANGEMENT_SHIFT,
        "candidate_parameter_count": 100,
        "null_parameter_count": 100,
        "candidate_trainable_parameter_count": 50,
        "null_trainable_parameter_count": 50,
        "frozen_parent_seed": 0,
        "initial_parent_replay_max_abs_difference": {
            "candidate": 0.0,
            "null": 0.0,
        },
        "post_pretraining_parent_replay_max_abs_difference": {
            "candidate": 0.0,
            "null": 0.0,
        },
        "frozen_parent_checkpoints": frozen_parents,
        "frozen_input_sources": frozen_input_sources,
        "n_validated_puzzle_coordinate_frames": 20,
        "n_outer_train_puzzles": 19,
        "n_pretraining_puzzles": 19,
        "outer_train_puzzle_ids": outer_train_puzzle_ids,
        "pretraining_puzzle_ids": outer_train_puzzle_ids,
        "expected_pretraining_eligible_construct_counts": [8],
        "pretraining_optimizer_steps_each": 19 * epochs,
        "point_optimizer_steps_each": 19 * epochs,
        "residual_optimizer_steps_each": 19 * epochs,
        "training_histories": {
            "candidate_pretraining": [0.7] * epochs,
            "null_pretraining": [0.8] * epochs,
            "candidate_point": [0.5] * epochs,
            "null_point": [0.6] * epochs,
            "candidate_residual": [0.4] * epochs,
            "null_residual": [0.45] * epochs,
        },
        "point_checkpoints": checkpoints["point"],
        "pretraining_decoder_checkpoints": checkpoints["decoder"],
        "pretraining_decoder_parameter_counts": {
            "candidate": 769,
            "null": 769,
        },
        "pretraining_summaries": {
            name: {
                "optimizer_steps": 19 * epochs,
                "trainable_parameter_count": 858369,
                "eligible_construct_counts": [8],
                "mask_fraction": 0.4,
                "context_layers_changed": True,
                "encoder_changed": False,
                "point_head_changed": False,
                "decoder_frozen_downstream": True,
                "mutant_outcome_used": False,
            }
            for name in ("candidate", "null")
        },
        "point_training_summaries": {
            name: dict(point_training_summary) for name in ("candidate", "null")
        },
        "context_retention_diagnostics": {
            name: _retention_diagnostic(
                arm=name,
                held_puzzle=held_puzzle,
                puzzle_ids=outer_train_puzzle_ids,
                training_epochs=epochs,
            )
            for name in ("candidate", "null")
        },
        "residual_checkpoints": checkpoints["residual"],
        "residual_parameter_counts": {"candidate": 63748, "null": 63748},
        "candidate_specific_trainable_parameter_counts": {
            "candidate": 50 + 63748,
            "null": 50 + 63748,
        },
        "prediction_artifact": str(prediction_path),
        "n_registered_prediction_rows": 2,
        "invariants": {
            "outcome_blind_puzzle_set_inputs": True,
            "exact_parameter_and_initialization_match": True,
            "candidate_nonfocal_only_cross_attention": True,
            "null_position_deranged_nonfocal_cross_attention": True,
            "candidate_null_equal_attention_support": True,
            "attention_weight_dropout_disabled": True,
            "puzzle_balanced_training": True,
            "position_aligned_nonfocal_cross_values": True,
            "nonfocal_summary_alignment_statistics": True,
            "matched_null_position_deranged_summary_statistics": True,
            "nonfocal_only_cross_values": True,
            "focal_excluded_from_cross_kv": True,
            "eight_token_cross_support": True,
            "paired_cross_block_reference_cancellation": True,
            "zero_nonfocal_exact_cross_replay": True,
            "paired_point_head_reference_cancellation": True,
            "zero_cross_exact_parent_replay": True,
            "fixed_position_derangement_shift_17": True,
            "outer_train_wt_only_puzzle_set_pretraining": True,
            "held_puzzle_excluded_from_pretraining": True,
            "mutant_outcome_excluded_from_pretraining": True,
            "candidate_null_equal_pretraining_budget": True,
            "pretraining_decoder_frozen_downstream": True,
            "encoder_and_point_unchanged_during_pretraining": True,
            "puzzle_coordinate_frames_validated": True,
            "frozen_v13_point_parent": True,
            "frozen_v14_context_encoder": True,
            "zero_initialized_parent_replay_at_1e_7": True,
            "point_head_only_warmup": True,
            "point_discriminative_learning_rates": True,
            "pretraining_capability_retention_diagnostic_complete": True,
            "point_frozen_during_calibration": True,
            "v10_residual_family_reused": True,
            "puzzle_balanced_residual_calibration": True,
            "median_constraint_all_held_rows": True,
            "prediction_target_free": True,
            "held_score_computed": False,
            "external_outcome_accessed": False,
        },
    }
    path = directory / f"puzzle_set_fold_result_fold{fold}_seed{seed}.json"
    path.write_text(json.dumps(row), encoding="utf-8")


def _canonicalize_fold_artifacts(
    directory: Path, *, fold: int, seed: int = 0
) -> dict[str, object]:
    row_path = directory / f"puzzle_set_fold_result_fold{fold}_seed{seed}.json"
    row = json.loads(row_path.read_text(encoding="utf-8"))
    observed = merger._fold_artifact_paths(row)
    expected = merger._canonical_fold_artifact_paths(
        directory.resolve(), fold=fold, seed=seed
    )
    for name, source in observed.items():
        source.replace(expected[name])
    row["prediction_artifact"] = str(expected["prediction_artifact"])
    row["point_checkpoints"] = {
        "candidate": str(expected["point_candidate"]),
        "null": str(expected["point_null"]),
    }
    row["pretraining_decoder_checkpoints"] = {
        "candidate": str(expected["decoder_candidate"]),
        "null": str(expected["decoder_null"]),
    }
    row["residual_checkpoints"] = {
        "candidate": str(expected["residual_candidate"]),
        "null": str(expected["residual_null"]),
    }
    row_path.write_text(json.dumps(row), encoding="utf-8")
    return row


def _manifest_rows_from_fold(row: dict[str, object]) -> dict[int, dict[str, object]]:
    fold = int(row["outer_fold"])
    sources = {}
    for source_id, runtime_source in row["frozen_input_sources"].items():
        spec = FROZEN_INPUT_SOURCE_SPEC[source_id]
        sources[source_id] = {
            **runtime_source,
            "realized_parameter_count": spec["realized_parameter_count"],
            "trainable_in_p1": spec["trainable_in_p1"],
        }
    return {
        fold: {
            "outer_fold": fold,
            "held_puzzle": row["held_puzzle"],
            "seed": 0,
            "sources": sources,
        }
    }


def test_merger_accepts_only_the_exact_complete_prediction_universe(
    tmp_path: Path,
) -> None:
    _write_fold(tmp_path, fold=0)
    _write_fold(tmp_path, fold=1)
    result = merge_complete_universe(
        tmp_path,
        expected_phase="P1M3",
        expected_folds=[0, 1],
        expected_seeds=[0],
        expected_pretraining_epochs=1,
        expected_point_epochs=1,
        expected_calibration_epochs=1,
        expected_parameter_count=100,
        expected_trainable_parameter_count=50,
    )
    assert result["schema_version"] == MERGED_SCHEMA
    assert result["status"] == "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"
    assert [row["outer_fold"] for row in result["folds"]] == [0, 1]
    assert result["merge_integrity"]["partial_scores_inspected"] is False
    assert (
        result["context_retention_summary"][
            "candidate_pretraining_established_all_runs"
        ]
        is True
    )
    assert (
        result["context_retention_summary"]["candidate_retention_positive_all_runs"]
        is True
    )


def test_production_binding_accepts_only_current_canonical_fold_artifacts(
    tmp_path: Path,
) -> None:
    _write_fold(tmp_path, fold=0)
    row = _canonicalize_fold_artifacts(tmp_path, fold=0)
    result = merge_complete_universe(
        tmp_path,
        expected_phase="P1M3",
        expected_folds=[0],
        expected_seeds=[0],
        expected_pretraining_epochs=1,
        expected_point_epochs=1,
        expected_calibration_epochs=1,
        expected_parameter_count=100,
        expected_trainable_parameter_count=50,
        expected_artifact_dir=tmp_path.resolve(),
        expected_source_manifest_rows=_manifest_rows_from_fold(row),
    )
    assert result["status"] == "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"


@pytest.mark.parametrize(
    "artifact_name",
    [
        "prediction_artifact",
        "point_candidate",
        "point_null",
        "decoder_candidate",
        "decoder_null",
        "residual_candidate",
        "residual_null",
    ],
)
def test_production_binding_rejects_compatible_artifact_outside_current_universe(
    tmp_path: Path, artifact_name: str
) -> None:
    _write_fold(tmp_path, fold=0)
    row = _canonicalize_fold_artifacts(tmp_path, fold=0)
    manifest_rows = _manifest_rows_from_fold(row)
    canonical = merger._canonical_fold_artifact_paths(
        tmp_path.resolve(), fold=0, seed=0
    )
    stale_path = tmp_path / "stale_compatible_attempt" / canonical[artifact_name].name
    stale_path.parent.mkdir()
    stale_path.write_bytes(canonical[artifact_name].read_bytes())
    if artifact_name == "prediction_artifact":
        row["prediction_artifact"] = str(stale_path)
    else:
        stage, arm = artifact_name.split("_", maxsplit=1)
        field = {
            "point": "point_checkpoints",
            "decoder": "pretraining_decoder_checkpoints",
            "residual": "residual_checkpoints",
        }[stage]
        row[field][arm] = str(stale_path)
    row_path = tmp_path / "puzzle_set_fold_result_fold0_seed0.json"
    row_path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match="authority-bound prediction directory"):
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
            expected_artifact_dir=tmp_path.resolve(),
            expected_source_manifest_rows=manifest_rows,
        )


def test_production_binding_rejects_fold_source_outside_active_manifest(
    tmp_path: Path,
) -> None:
    _write_fold(tmp_path, fold=0)
    row = _canonicalize_fold_artifacts(tmp_path, fold=0)
    manifest_rows = _manifest_rows_from_fold(row)
    runtime_source = row["frozen_input_sources"]["v8_meanaligned_checkpoint"]
    current_path = Path(runtime_source["path"])
    stale_path = tmp_path / "stale_sources" / current_path.name
    stale_path.parent.mkdir()
    stale_path.write_bytes(current_path.read_bytes())
    runtime_source["path"] = str(stale_path)
    row_path = tmp_path / "puzzle_set_fold_result_fold0_seed0.json"
    row_path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the active source manifest"):
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
            expected_artifact_dir=tmp_path.resolve(),
            expected_source_manifest_rows=manifest_rows,
        )


@pytest.mark.parametrize("phase", ["P1M2", "P1M3", "P1M4"])
def test_production_merge_universe_is_frozen_per_phase(phase: str) -> None:
    expected = PRODUCTION_MERGE_UNIVERSES[phase]
    observed = assert_production_merge_universe(
        phase=phase,
        folds=list(expected["folds"]),
        seeds=list(expected["seeds"]),
        pretraining_epochs=expected["pretraining_epochs"],
        point_epochs=expected["point_epochs"],
        calibration_epochs=expected["calibration_epochs"],
        parameter_count=expected["parameter_count"],
        trainable_parameter_count=expected["trainable_parameter_count"],
    )
    assert observed == expected


@pytest.mark.parametrize(
    ("phase", "changed_field", "changed_value"),
    [
        ("P1M2", "seeds", [0, 1]),
        ("P1M3", "folds", list(range(19))),
        ("P1M4", "point_epochs", 39),
    ],
)
def test_production_main_rejects_redefined_universe_before_canonical_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
    changed_field: str,
    changed_value: object,
) -> None:
    monkeypatch.setattr(merger, "assert_merge_authority", lambda *_args, **_kw: {})
    frozen = dict(PRODUCTION_MERGE_UNIVERSES[phase])
    frozen[changed_field] = changed_value
    out_json = tmp_path / "canonical_complete_unscored_merge.json"
    argv = [
        "--repo-root",
        str(tmp_path),
        "--input-dir",
        str(tmp_path),
        "--phase",
        phase,
        "--folds",
        ",".join(map(str, frozen["folds"])),
        "--seeds",
        ",".join(map(str, frozen["seeds"])),
        "--pretraining-epochs",
        str(frozen["pretraining_epochs"]),
        "--point-epochs",
        str(frozen["point_epochs"]),
        "--calibration-epochs",
        str(frozen["calibration_epochs"]),
        "--parameter-count",
        str(frozen["parameter_count"]),
        "--trainable-parameter-count",
        str(frozen["trainable_parameter_count"]),
        "--out-json",
        str(out_json),
    ]
    with pytest.raises(ValueError, match="production merge universe differs"):
        merger.main(argv)
    assert not out_json.exists()


def test_production_main_cannot_omit_artifact_and_manifest_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "source_manifest.json"
    manifest_rows = {"validated": "manifest rows"}
    monkeypatch.setattr(
        merger,
        "assert_merge_authority",
        lambda *_args, **_kwargs: {
            "authority": {"source_manifest_path": str(manifest_path)}
        },
    )
    monkeypatch.setattr(
        merger,
        "validate_source_manifest",
        lambda path: manifest_rows if path == manifest_path else None,
    )
    captured = {}

    def fake_merge(input_dir: Path, **kwargs: object) -> dict[str, object]:
        captured["input_dir"] = input_dir
        captured.update(kwargs)
        return {
            "schema_version": MERGED_SCHEMA,
            "status": "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS",
        }

    monkeypatch.setattr(merger, "merge_complete_universe", fake_merge)
    frozen = PRODUCTION_MERGE_UNIVERSES["P1M2"]
    out_json = tmp_path / "p1m2_complete_unscored_merge.json"
    result = merger.main(
        [
            "--repo-root",
            str(tmp_path),
            "--input-dir",
            str(tmp_path),
            "--phase",
            "P1M2",
            "--folds",
            ",".join(map(str, frozen["folds"])),
            "--seeds",
            ",".join(map(str, frozen["seeds"])),
            "--pretraining-epochs",
            str(frozen["pretraining_epochs"]),
            "--point-epochs",
            str(frozen["point_epochs"]),
            "--calibration-epochs",
            str(frozen["calibration_epochs"]),
            "--parameter-count",
            str(frozen["parameter_count"]),
            "--trainable-parameter-count",
            str(frozen["trainable_parameter_count"]),
            "--out-json",
            str(out_json),
        ]
    )

    assert result == 0
    assert captured["input_dir"] == tmp_path.resolve()
    assert captured["expected_artifact_dir"] == tmp_path.resolve()
    assert captured["expected_source_manifest_rows"] is manifest_rows


def test_merger_rejects_missing_fold(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0)
    try:
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0, 1],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "incomplete or unexpected" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted a missing fold")


def test_merger_rejects_target_bearing_prediction(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0, target_field=True)
    try:
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "target_free" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted target-bearing prediction")


def test_merger_rejects_wrong_epoch_or_parameter_count(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0, epochs=2)
    try:
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "epoch freeze" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted an epoch mismatch")


def test_merger_rejects_changed_position_derangement(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0)
    path = tmp_path / "puzzle_set_fold_result_fold0_seed0.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["position_derangement_shift"] = POSITION_DERANGEMENT_SHIFT - 1
    path.write_text(json.dumps(row), encoding="utf-8")
    try:
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "changed connectivity" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted a changed derangement")


def test_merger_rejects_changed_frozen_input_role(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0)
    path = tmp_path / "puzzle_set_fold_result_fold0_seed0.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["frozen_input_sources"]["v8_meanaligned_checkpoint"]["role"] = "COMPARATOR_ONLY"
    path.write_text(json.dumps(row), encoding="utf-8")
    try:
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except RuntimeError as error:
        assert "frozen input source changed" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted a changed input role")


def test_merger_requires_both_arms_in_parameter_provenance(tmp_path: Path) -> None:
    for field, message in (
        ("residual_parameter_counts", "changed residual family"),
        (
            "candidate_specific_trainable_parameter_counts",
            "changed candidate-specific trainable count",
        ),
    ):
        directory = tmp_path / field
        _write_fold(directory, fold=0)
        path = directory / "puzzle_set_fold_result_fold0_seed0.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        del row[field]["null"]
        path.write_text(json.dumps(row), encoding="utf-8")
        try:
            merge_complete_universe(
                directory,
                expected_phase="P1M3",
                expected_folds=[0],
                expected_seeds=[0],
                expected_pretraining_epochs=1,
                expected_point_epochs=1,
                expected_calibration_epochs=1,
                expected_parameter_count=100,
                expected_trainable_parameter_count=50,
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"puzzle-set merger accepted missing null {field}")


def test_merger_requires_source_paths_to_be_fixed_within_scope(tmp_path: Path) -> None:
    global_dir = tmp_path / "global"
    _write_fold(global_dir, fold=0)
    _write_fold(global_dir, fold=1)
    global_row_path = global_dir / "puzzle_set_fold_result_fold1_seed0.json"
    global_row = json.loads(global_row_path.read_text(encoding="utf-8"))
    alternate_cache = global_dir / "alternate_unconstrained.h5"
    alternate_cache.touch()
    global_row["frozen_input_sources"]["unconstrained_feature_cache"]["path"] = str(
        alternate_cache
    )
    global_row_path.write_text(json.dumps(global_row), encoding="utf-8")
    try:
        merge_complete_universe(
            global_dir,
            expected_phase="P1M3",
            expected_folds=[0, 1],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except RuntimeError as error:
        assert "path changed within its registered scope" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted fold-varying global cache")

    fold_dir = tmp_path / "fold"
    _write_fold(fold_dir, fold=0, seed=0)
    _write_fold(fold_dir, fold=0, seed=1)
    fold_row_path = fold_dir / "puzzle_set_fold_result_fold0_seed1.json"
    fold_row = json.loads(fold_row_path.read_text(encoding="utf-8"))
    alternate_v8 = fold_dir / "alternate" / "v8_corrected_mean_fold0_seed0.pt"
    alternate_v8.parent.mkdir()
    alternate_v8.touch()
    fold_row["frozen_input_sources"]["v8_meanaligned_checkpoint"]["path"] = str(
        alternate_v8
    )
    fold_row_path.write_text(json.dumps(fold_row), encoding="utf-8")
    try:
        merge_complete_universe(
            fold_dir,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0, 1],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except RuntimeError as error:
        assert "path changed within its registered scope" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted seed-varying fold source")


def test_merger_rejects_misaligned_prediction_rows(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0, short_candidate=True)
    try:
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "aligned_rows" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted misaligned prediction rows")


def test_merger_rejects_biological_key_overlap_across_folds(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0)
    _write_fold(tmp_path, fold=1)
    first = tmp_path / "prediction0_0.npz"
    second = tmp_path / "prediction1_0.npz"
    with np.load(first, allow_pickle=True) as handle:
        payload = {name: handle[name] for name in handle.files}
    payload["outer_fold"] = np.full(2, 1, dtype=np.int64)
    np.savez_compressed(second, **payload)
    try:
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0, 1],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "repeats biological keys" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted cross-fold key overlap")


def test_merger_rejects_distribution_that_moves_the_point_median(
    tmp_path: Path,
) -> None:
    _write_fold(tmp_path, fold=0)
    path = tmp_path / "prediction0_0.npz"
    with np.load(path, allow_pickle=True) as handle:
        payload = {name: handle[name] for name in handle.files}
    payload["candidate_locations"] = np.full((2, 2), 0.5)
    np.savez_compressed(path, **payload)
    try:
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "median_preserved" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted a shifted distribution median")


def test_merger_rejects_pretraining_protocol_drift(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0)
    path = tmp_path / "puzzle_set_fold_result_fold0_seed0.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["pretraining_summaries"]["candidate"]["mask_fraction"] = 0.5
    path.write_text(json.dumps(row), encoding="utf-8")
    try:
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "candidate pretraining summary" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted changed pretraining protocol")


def test_merger_rejects_post_pretraining_parent_shift(tmp_path: Path) -> None:
    _write_fold(tmp_path, fold=0)
    path = tmp_path / "puzzle_set_fold_result_fold0_seed0.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["post_pretraining_parent_replay_max_abs_difference"]["candidate"] = 1e-4
    path.write_text(json.dumps(row), encoding="utf-8")
    try:
        merge_complete_universe(
            tmp_path,
            expected_phase="P1M3",
            expected_folds=[0],
            expected_seeds=[0],
            expected_pretraining_epochs=1,
            expected_point_epochs=1,
            expected_calibration_epochs=1,
            expected_parameter_count=100,
            expected_trainable_parameter_count=50,
        )
    except ValueError as error:
        assert "post_pretraining_parent_replay" in str(error)
    else:
        raise AssertionError("puzzle-set merger accepted a shifted parent point")


def test_p1m3_merger_preserves_artifacts_but_closes_scoring_on_negative_retention(
    tmp_path: Path,
) -> None:
    _write_fold(tmp_path, fold=0)
    path = tmp_path / "puzzle_set_fold_result_fold0_seed0.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    candidate = row["context_retention_diagnostics"]["candidate"]
    for puzzle in candidate["per_puzzle"]:
        puzzle["post_point_l1"] = 1.1
    candidate["mean"]["post_point_l1"] = 1.1
    candidate["retained_fraction"] = -0.25
    candidate["retention_positive"] = False
    path.write_text(json.dumps(row), encoding="utf-8")

    result = merge_complete_universe(
        tmp_path,
        expected_phase="P1M3",
        expected_folds=[0],
        expected_seeds=[0],
        expected_pretraining_epochs=1,
        expected_point_epochs=1,
        expected_calibration_epochs=1,
        expected_parameter_count=100,
        expected_trainable_parameter_count=50,
    )

    assert result["status"] == "PUZZLE_SET_TRAIN_ONLY_RETENTION_GATE_FAIL"
    assert (
        result["context_retention_summary"][
            "candidate_pretraining_established_all_runs"
        ]
        is True
    )
    assert (
        result["context_retention_summary"]["candidate_retention_positive_all_runs"]
        is False
    )
    assert len(result["folds"]) == 1
