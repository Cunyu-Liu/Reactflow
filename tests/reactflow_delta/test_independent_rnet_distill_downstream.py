from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts.reactflow_delta.independent_rnet_distill import (
    IndependentRNetDistillStudent,
)
from scripts.reactflow_delta.run_independent_rnet_distill_downstream import (
    EVIDENCE_STATUS,
    EXPECTED_EXPERIMENT_ID,
    EXPECTED_FOLDS,
    EXPECTED_PREDICTION_FIELDS,
    EXPECTED_SCHEDULE,
    EXPECTED_SEEDS,
    PREDICTION_SCHEMA,
    PRETRAIN_CHECKPOINT_SCHEMA,
    _artifact_paths,
    _assert_tensor_cuda,
    _canonicalize_fold_result_paths,
    _downstream_epoch_order,
    _publish_fold_artifacts,
    _refuse_fold_overwrite,
    _rename_v11_prediction,
    _requires_authoritative_feature41_replay,
    _reset_downstream_rng,
    _validate_phase_request,
    validate_downstream_cli_binding,
    validate_pretrained_pair,
)
from scripts.reactflow_delta.run_model_rescue_v11 import _held_prediction


def _checkpoint_payload(
    *, condition: str, encoder_value: float, residual_value: float = 0.0
) -> dict:
    return {
        "schema_version": PRETRAIN_CHECKPOINT_SCHEMA,
        "experiment_id": "RND1_TEST",
        "condition": condition,
        "seed": 20260828,
        "data_order_seed": 20260828,
        "epochs": 1,
        "training_device": "cuda:0",
        "precision": "float32",
        "source": {"model_name": "RibonanzaNet2", "record_count": 208905},
        "model": {"width": 256, "context_blocks": 6},
        "point_model_state_dict": {
            "input_projection.weight": torch.tensor([encoder_value]),
            "residual_head.7.weight": torch.tensor([residual_value]),
        },
        "distill_head_excluded_from_downstream": True,
    }


def test_pretrained_pair_requires_equal_residual_but_different_encoder() -> None:
    candidate = _checkpoint_payload(condition="aligned_candidate", encoder_value=1.0)
    null = _checkpoint_payload(condition="cyclic_shift_17_null", encoder_value=2.0)
    audit = validate_pretrained_pair(candidate, null)
    assert audit["residual_heads_identical"] is True
    assert audit["pretrained_encoders_different"] is True
    assert audit["changed_encoder_tensor_count"] == 1

    null["point_model_state_dict"]["residual_head.7.weight"] = torch.tensor([3.0])
    with pytest.raises(RuntimeError, match="residual initialization differs"):
        validate_pretrained_pair(candidate, null)


def test_pretrained_pair_rejects_identical_encoder_and_distill_head() -> None:
    candidate = _checkpoint_payload(condition="aligned_candidate", encoder_value=1.0)
    null = _checkpoint_payload(condition="cyclic_shift_17_null", encoder_value=1.0)
    with pytest.raises(RuntimeError, match="pretrained encoders are identical"):
        validate_pretrained_pair(candidate, null)

    null["point_model_state_dict"]["distill_head.1.weight"] = torch.tensor([1.0])
    with pytest.raises(RuntimeError, match="contains distill_head"):
        validate_pretrained_pair(candidate, null)


def test_each_arm_replays_the_same_rng_and_epoch_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("cpu", seed)))
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: calls.append(("cuda", seed)),
    )
    assert _reset_downstream_rng(0) == 2_800_000
    assert _reset_downstream_rng(0) == 2_800_000
    assert calls == [
        ("cpu", 2_800_000),
        ("cuda", 2_800_000),
        ("cpu", 2_800_000),
        ("cuda", 2_800_000),
    ]
    assert _downstream_epoch_order(8, seed=0, epoch=3) == _downstream_epoch_order(
        8, seed=0, epoch=3
    )


def test_prediction_rename_is_exact_and_target_free() -> None:
    source: dict[str, np.ndarray] = {}
    for name in EXPECTED_PREDICTION_FIELDS:
        source_name = name
        if name.startswith("candidate_"):
            source_name = f"anchored_{name.removeprefix('candidate_')}"
        elif name.startswith("null_"):
            source_name = f"unanchored_{name.removeprefix('null_')}"
        if name == "schema_version":
            source[source_name] = np.asarray("old.v11.schema")
        elif name in {"keys", "biological_scoring_key"}:
            source[source_name] = np.asarray(["key"], dtype=object)
        elif name == "registered_status":
            source[source_name] = np.asarray(["covered"], dtype=object)
        else:
            source[source_name] = np.ones(1)
    output = _rename_v11_prediction(source)
    assert frozenset(output) == EXPECTED_PREDICTION_FIELDS
    assert str(output["schema_version"].item()) == PREDICTION_SCHEMA
    assert "anchored_point" not in output
    assert "unanchored_point" not in output


def test_cuda_fail_closed_and_fold_overwrite_refusal(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="CUDA_REQUIRED"):
        _assert_tensor_cuda(torch.ones(1), label="test tensor")
    paths = _artifact_paths(tmp_path, fold=0, seed=0)
    paths["result"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _refuse_fold_overwrite(paths)


def test_downstream_encoder_preserves_unobserved_edit_query_state() -> None:
    model = IndependentRNetDistillStudent().eval()
    length = 4
    sequence = torch.eye(4, dtype=torch.float32)
    reactivity = torch.zeros(length)
    precision = torch.zeros(length)
    observed = torch.tensor([1.0, 0.0, 1.0, 1.0])
    position = torch.arange(length, dtype=torch.float32)
    region = torch.zeros(length, 2)
    with torch.no_grad():
        hidden = model.encode(
            (sequence, reactivity, precision, observed, position, region)
        )
    assert hidden.shape == (length, 256)
    assert not torch.equal(hidden[1], torch.zeros_like(hidden[1]))


def _write_active_binding(repo_root: Path, *, phase: str = "RND2") -> dict[str, Path]:
    output_leaf = {
        "RND2": "rnd2",
        "RND3": "rnd3",
        "RND6P": "rnd6_formal",
    }[phase]
    paths = {
        "m2_csv": repo_root / "data" / "m2.csv",
        "pretrain_dir": repo_root / "artifacts" / "rnd1",
        "v8_dir": repo_root / "artifacts" / "v8",
        "v10_dir": repo_root / "artifacts" / "v10",
        "tic2a_merged_json": repo_root / "artifacts" / "tic2a.json",
        "unconstrained_cache": repo_root / "artifacts" / "unconstrained.h5",
        "constrained_cache": repo_root / "artifacts" / "constrained.h5",
        "out_dir": repo_root / "artifacts" / output_leaf,
    }
    authority = {
        "current_phase": phase,
        "m2_csv_path": str(paths["m2_csv"]),
        "pretraining_dir": str(paths["pretrain_dir"]),
        "historical_v8_dir": str(paths["v8_dir"]),
        "historical_v10_dir": str(paths["v10_dir"]),
        "tic2a_merged_registry_path": str(paths["tic2a_merged_json"]),
        "unconstrained_feature_cache_path": str(paths["unconstrained_cache"]),
        "constrained_feature_cache_path": str(paths["constrained_cache"]),
        "smoke_prediction_dir": str(repo_root / "artifacts" / "rnd2"),
        "screen_prediction_dir": str(repo_root / "artifacts" / "rnd3"),
        "formal_prediction_dir": str(repo_root / "artifacts" / "rnd6_formal"),
    }
    active_path = repo_root / "configs/reactflow_delta/active_contract.yaml"
    active_path.parent.mkdir(parents=True)
    active_path.write_text(
        yaml.safe_dump({"authority": authority}, sort_keys=False), encoding="utf-8"
    )
    return paths


def test_direct_runner_binds_every_path_and_experiment_to_active(tmp_path: Path) -> None:
    paths = _write_active_binding(tmp_path)
    args = SimpleNamespace(
        phase="RND2",
        experiment_id="RND2_RNET_DISTILL_TWO_FOLD_GPU_ENGINEERING_SMOKE",
        **paths,
    )
    binding = validate_downstream_cli_binding(tmp_path, args)
    assert binding["out_dir"] == str(paths["out_dir"].resolve())
    args.m2_csv = tmp_path / "wrong.csv"
    with pytest.raises(RuntimeError, match="m2_csv path differs"):
        validate_downstream_cli_binding(tmp_path, args)
    args.m2_csv = paths["m2_csv"]
    args.experiment_id = "RND2_WRONG"
    with pytest.raises(RuntimeError, match="experiment_id differs"):
        validate_downstream_cli_binding(tmp_path, args)

    rnd3_root = tmp_path / "rnd3_repo"
    rnd3_paths = _write_active_binding(rnd3_root, phase="RND3")
    rnd3_args = SimpleNamespace(
        phase="RND3",
        experiment_id="RND3_RNET_DISTILL_COMPLETE_SEED0_PREDICTION_ONLY",
        **rnd3_paths,
    )
    rnd3_binding = validate_downstream_cli_binding(rnd3_root, rnd3_args)
    assert rnd3_binding["out_dir"] == str(rnd3_paths["out_dir"].resolve())

    rnd6_root = tmp_path / "rnd6_repo"
    rnd6_paths = _write_active_binding(rnd6_root, phase="RND6P")
    rnd6_args = SimpleNamespace(
        phase="RND6P",
        experiment_id=(
            "RND6P_RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_PREDICTION_ONLY"
        ),
        **rnd6_paths,
    )
    rnd6_binding = validate_downstream_cli_binding(rnd6_root, rnd6_args)
    assert rnd6_binding["out_dir"] == str(rnd6_paths["out_dir"].resolve())


def test_rnd6p_freezes_formal_universe_and_replays_feature41_for_every_seed() -> None:
    assert EXPECTED_FOLDS["RND6P"] == tuple(range(20))
    assert EXPECTED_SEEDS["RND6P"] == tuple(range(5))
    assert EXPECTED_SCHEDULE["RND6P"] == (40, 40)
    assert EXPECTED_EXPERIMENT_ID["RND6P"] == (
        "RND6P_RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_PREDICTION_ONLY"
    )
    assert EVIDENCE_STATUS["RND6P"] == (
        "EXPOSURE_DISCLOSED_DEVELOPMENT_FORMAL_PREDICTION_ONLY"
    )
    for seed in range(5):
        _validate_phase_request(
            phase="RND6P",
            folds=(0, 19),
            point_epochs=40,
            calibration_epochs=40,
            seed=seed,
        )
        assert _requires_authoritative_feature41_replay("RND6P", seed) is True
    with pytest.raises(ValueError, match="seed or epoch schedule changed"):
        _validate_phase_request(
            phase="RND6P",
            folds=(0,),
            point_epochs=40,
            calibration_epochs=40,
            seed=5,
        )
    with pytest.raises(ValueError, match="outside the frozen universe"):
        _validate_phase_request(
            phase="RND6P",
            folds=(20,),
            point_epochs=40,
            calibration_epochs=40,
            seed=0,
        )
    assert _requires_authoritative_feature41_replay("RND3", 0) is True
    assert _requires_authoritative_feature41_replay("RND2", 0) is False
    assert "defined only for seed0" not in inspect.getsource(_held_prediction)


def _write_staged_fold(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))


def test_complete_fold_publishes_result_last_without_overwrite(tmp_path: Path) -> None:
    staging_paths = _artifact_paths(tmp_path / "staging", fold=0, seed=0)
    canonical_paths = _artifact_paths(tmp_path / "canonical", fold=0, seed=0)
    _write_staged_fold(staging_paths)
    canonical_paths["result"].parent.mkdir(parents=True)
    _publish_fold_artifacts(staging_paths, canonical_paths)
    assert all(path.is_file() for path in canonical_paths.values())
    assert not any(path.exists() for path in staging_paths.values())
    assert canonical_paths["result"].read_bytes() == b"result"

    staged_again = _artifact_paths(tmp_path / "staging_again", fold=0, seed=0)
    _write_staged_fold(staged_again)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _publish_fold_artifacts(staged_again, canonical_paths)
    assert canonical_paths["candidate_point"].read_bytes() == b"candidate_point"


def test_publication_failure_rolls_back_every_canonical_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_paths = _artifact_paths(tmp_path / "staging", fold=0, seed=0)
    canonical_paths = _artifact_paths(tmp_path / "canonical", fold=0, seed=0)
    _write_staged_fold(staging_paths)
    canonical_paths["result"].parent.mkdir(parents=True)
    original_replace = Path.replace
    staging_root = staging_paths["result"].parent
    forward_moves = 0
    failed = False

    def fail_once_during_publish(source: Path, target: Path) -> Path:
        nonlocal forward_moves, failed
        if source.parent == staging_root:
            forward_moves += 1
            if forward_moves == 3 and not failed:
                failed = True
                raise OSError("injected publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_once_during_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        _publish_fold_artifacts(staging_paths, canonical_paths)
    assert not any(path.exists() for path in canonical_paths.values())
    assert all(path.is_file() for path in staging_paths.values())


def test_completed_row_records_only_canonical_paths(tmp_path: Path) -> None:
    canonical = _artifact_paths(tmp_path / "canonical", fold=1, seed=0)
    result = {
        "point_checkpoints": {"candidate": "temp", "null": "temp"},
        "residual_checkpoints": {
            "feature41": "temp",
            "candidate": "temp",
            "null": "temp",
        },
        "prediction_artifact": "temp",
    }
    bound = _canonicalize_fold_result_paths(result, canonical)
    assert bound["point_checkpoints"]["candidate"] == str(
        canonical["candidate_point"].resolve()
    )
    assert bound["prediction_artifact"] == str(canonical["prediction"].resolve())
    assert result["prediction_artifact"] == "temp"
