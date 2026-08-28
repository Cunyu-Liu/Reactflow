from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.reactflow_delta.independent_rnet_distill import (
    IndependentRNetDistillStudent,
    downstream_point_state_dict,
    encoder_parameter_count,
    total_parameter_count,
)
from scripts.reactflow_delta import run_independent_rnet_distill_pretrain as pretrain
from scripts.reactflow_delta import (
    validate_independent_rnet_distill_pretrain_artifacts as validator,
)


def _contract(tmp_path: Path, source_manifest: Path) -> dict:
    source_cache = tmp_path / "teacher_cache"
    return {
        "project_task_id": validator.PROJECT_TASK_ID,
        "scope": {"result_class": "EXPOSURE_DISCLOSED_DEVELOPMENT_CANDIDATE"},
        "source_binding": {
            "source_cache": str(source_cache),
            "source_manifest": str(source_manifest),
            "expected_layout": "reactflow-sharded-frozen-v1",
            "expected_model_name": "RibonanzaNet2",
            "expected_model_version": "alpha-v1",
            "expected_weights_sha256": "frozen-weights",
            "expected_record_count": 208905,
            "expected_shard_count": 409,
            "expected_full_shard_size": 512,
            "expected_last_shard_records": 9,
            "expected_single_feature_dim": 384,
            "exact_sequence_overlap_with_registered_openknot_mutants": 0,
            "registered_openknot_mutant_sequence_count": 14136,
            "overlap_audit_interpretation": (
                "EXACT_SEQUENCE_ONLY_NEAR_NEIGHBOR_EXPOSURE_NOT_EXCLUDED"
            ),
            "source_integrity_repair": {
                "expected_parent_child_content_binding_mismatches": 388,
                "verified_child_content_sha256": "verified-child",
                "binding_basis": (
                    "CHILD_PROVENANCE_SCHEMA_COUNT_WEIGHTS_PLUS_ACTUAL_INDEX_"
                    "AND_NPZ_SHAPE_READ"
                ),
            },
        },
        "student": {
            "input_channels": 11,
            "context_width": 256,
            "attention_heads": 8,
            "context_blocks": 6,
            "ffn_width": 1024,
            "relative_distance_window": 256,
            "dropout": 0.1,
            "distillation_head": "LAYERNORM_256_LINEAR_384",
            "downstream_head": "FEATURE41_ANCHORED_V14_RESIDUAL_HEAD",
            "pretraining_inputs": {
                "sequence": "ONE_HOT_A_C_G_U",
                "reactivity": "ZERO",
                "precision": "ZERO",
                "observed": "TRUE_FOR_VALID_TOKENS",
                "position": "REGISTERED_ZERO_BASED_INDEX",
                "region": "ZERO",
                "corruption": False,
            },
            "strict_length_identity": (
                "INDEX_LENGTH_EQUALS_SEQUENCE_LENGTH_EQUALS_TEACHER_L_"
                "EQUALS_STUDENT_TOKEN_LENGTH"
            ),
            "silent_resize_or_padding_allowed": False,
        },
        "frozen_schedule": {
            "distillation_seed": 20260828,
            "data_order_seed": 20260828,
            "distillation_epochs": 1,
            "optimizer": "ADAMW",
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "batch_size": 16,
            "gradient_clip_norm": 1.0,
            "mixed_precision": "BFLOAT16_ON_SUPPORTED_CUDA_ELSE_FLOAT32_CUDA",
        },
        "gpu_policy": {
            "training_and_gpu_validation_device_class": "CUDA_ONLY",
            "cpu_model_or_loss_fallback_allowed": False,
            "actual_model_input_target_output_loss_and_optimizer_state_must_be_cuda": True,
        },
        "outcome_policy": {
            "pretraining_may_read_openknot_mutant_outcome": False,
            "new_external_outcome_access_allowed": False,
        },
        "phase_contract": {
            "RND1": {
                "outputs": [
                    "candidate_checkpoint",
                    "null_checkpoint",
                    "pretraining_audit",
                ]
            },
            "RND2": {"required_predecessor": validator.PASS},
        },
    }


def _source_manifest(contract: dict) -> dict:
    source = contract["source_binding"]
    repair = source["source_integrity_repair"]
    return {
        "schema_version": validator.SOURCE_SCHEMA,
        "status": pretrain.SOURCE_MANIFEST_STATUS,
        "project_task_id": validator.PROJECT_TASK_ID,
        "source_cache": str(Path(source["source_cache"]).resolve()),
        "layout": source["expected_layout"],
        "model_name": source["expected_model_name"],
        "model_version": source["expected_model_version"],
        "weights_sha256": source["expected_weights_sha256"],
        "record_count": source["expected_record_count"],
        "shard_count": source["expected_shard_count"],
        "single_feature_dim": source["expected_single_feature_dim"],
        "legacy_parent_content_hashes_authoritative": False,
        "legacy_parent_content_binding_mismatches": 388,
        "verified_recovered_last_shard_content_sha256": repair[
            "verified_child_content_sha256"
        ],
        "payload_hash_verification_scope": "SHARD_00408_ONLY",
        "full_cache_rehash_performed": False,
        "structural_binding_basis": repair["binding_basis"],
        "index_contains_outcome_fields": False,
        "teacher_pair_features_used": False,
        "live_teacher_used": False,
        "openknot_mutant_outcome_accessed": False,
        "new_external_outcome_accessed": False,
        "exact_sequence_overlap_prior_audit": {
            "observed_overlap": 0,
            "registered_sequence_count": 14136,
            "interpretation": source["overlap_audit_interpretation"],
        },
        "scientific_evidence_ceiling": contract["scope"]["result_class"],
    }


def _checkpoint_payloads(source: dict) -> tuple[dict, dict]:
    candidate = IndependentRNetDistillStudent()
    null = copy.deepcopy(candidate)
    with torch.no_grad():
        null.input_projection.weight[0, 0].add_(0.25)
    model_contract = {
        "input_channels": 11,
        "width": 256,
        "context_blocks": 6,
        "ffn_width": 1024,
        "teacher_width": 384,
        "dropout": 0.1,
        "encoder_parameters": encoder_parameter_count(candidate),
        "total_parameters": total_parameter_count(candidate),
    }

    def payload(condition: str, model: IndependentRNetDistillStudent) -> dict:
        return {
            "schema_version": pretrain.CHECKPOINT_SCHEMA,
            "experiment_id": "RND1_TEST",
            "condition": condition,
            "seed": 20260828,
            "data_order_seed": 20260828,
            "epochs": 1,
            "training_device": "cuda:0",
            "precision": "float32",
            "source": source,
            "model": model_contract,
            "point_model_state_dict": downstream_point_state_dict(model),
            "distill_head_excluded_from_downstream": True,
        }

    return payload("aligned_candidate", candidate), payload(
        "cyclic_shift_17_null", null
    )


def _audit(
    *,
    authority: dict,
    source: dict,
    source_manifest: Path,
    pretrain_dir: Path,
    source_cache: Path,
) -> dict:
    return {
        "schema_version": pretrain.SCHEMA,
        "experiment_id": "RND1_TEST",
        "started_at_utc": "2026-08-28T00:00:00+00:00",
        "finished_at_utc": "2026-08-28T01:00:00+00:00",
        "command": ["run_independent_rnet_distill_pretrain.py"],
        "source_root": str(source_cache.resolve()),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_status": pretrain.SOURCE_MANIFEST_STATUS,
        "source": source,
        "authority": authority,
        "seed": 20260828,
        "data_order_seed": 20260828,
        "batch_size": 16,
        "device_requested": "cuda:0",
        "device_actual": "cuda:0",
        "gpu_name": "fixture GPU",
        "torch_cuda": "12.4",
        "candidate_target": "teacher_single_same_position",
        "null_target": "teacher_single_positive_cyclic_shift_min_17_L_minus_1",
        "student_input": (
            "sequence_one_hot+zero_reactivity+zero_precision+observed_token+"
            "normalized_position+zero_region+zero_corruption"
        ),
        "loss": "smooth_l1_mean_over_valid_positions_and_384_features",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "capturable": True,
        },
        "training": {
            "epochs": 1,
            "steps_each": 13057,
            "records_each": 208905,
            "same_batch_order": True,
            "same_optimizer_hyperparameters": True,
            "same_dropout_rng_per_step": True,
            "batch_padding_used": False,
            "precision": "float32",
            "candidate_null_residual_head_exact_equal": True,
            "candidate_null_encoder_exact_different": True,
            "candidate_mean_training_loss": "SECRET_LOSS_MUST_NOT_BE_READ",
            "null_mean_training_loss": "SECRET_LOSS_MUST_NOT_BE_READ",
            "first_batch_record_ids": ["first"],
            "last_batch_record_ids": ["last"],
        },
        "metric_eligibility": "engineering_pretraining_only_not_scientific",
        "outcome_accessed": False,
        "cpu_fallback": False,
        "exit_code": 0,
        "artifacts": {
            "candidate_checkpoint": str(
                pretrain_dir / pretrain.CANDIDATE_CHECKPOINT
            ),
            "null_checkpoint": str(pretrain_dir / pretrain.NULL_CHECKPOINT),
            "audit": str(pretrain_dir / pretrain.AUDIT_NAME),
        },
    }


def _write_valid_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    pretrain_dir = tmp_path / "rnd1_pretrain"
    pretrain_dir.mkdir()
    source_manifest_path = tmp_path / "source_manifest.json"
    contract = _contract(tmp_path, source_manifest_path)
    authority = {
        "schema_version": "reactflow_delta.independent_rnet_distill_authority_validation.v1",
        "status": "INDEPENDENT_RNET_DISTILL_AUTHORITY_EXACT_PASS",
        "project_task_id": validator.PROJECT_TASK_ID,
        "phase": "RND1",
        "branch": "codex/reactflow-delta-independent-rnet-distill-20260828",
        "training_allowed": True,
        "held_score_read_allowed": False,
        "partial_score_accessed": False,
        "new_external_outcome_accessed": False,
    }
    active = {
        "authority": {
            "pretraining_dir": str(pretrain_dir),
            "teacher_source_manifest_path": str(source_manifest_path),
            "teacher_source_manifest_status": pretrain.SOURCE_MANIFEST_STATUS,
        }
    }
    monkeypatch.setattr(
        validator,
        "assert_run_authority",
        lambda _repo_root, phase: authority if phase == "RND1" else None,
    )

    def fake_load_yaml(path: Path) -> dict:
        if path.name == validator.ACTIVE_PATH.name:
            return active
        if path.name == validator.CONTRACT_PATH.name:
            return contract
        raise AssertionError(path)

    monkeypatch.setattr(validator, "_load_yaml", fake_load_yaml)
    source_manifest_path.write_text(
        json.dumps(_source_manifest(contract)), encoding="utf-8"
    )
    source = validator._expected_source(contract)
    candidate, null = _checkpoint_payloads(source)
    torch.save(candidate, pretrain_dir / pretrain.CANDIDATE_CHECKPOINT)
    torch.save(null, pretrain_dir / pretrain.NULL_CHECKPOINT)
    audit = _audit(
        authority=authority,
        source=source,
        source_manifest=source_manifest_path,
        pretrain_dir=pretrain_dir,
        source_cache=Path(contract["source_binding"]["source_cache"]),
    )
    (pretrain_dir / pretrain.AUDIT_NAME).write_text(
        json.dumps(audit), encoding="utf-8"
    )
    return {
        "repo_root": repo_root,
        "pretrain_dir": pretrain_dir,
        "source_manifest": source_manifest_path,
        "contract": contract,
    }


def _validate(fixture: dict) -> dict:
    return validator.validate_pretraining_artifacts(
        repo_root=fixture["repo_root"],
        pretrain_dir=fixture["pretrain_dir"],
        source_manifest_path=fixture["source_manifest"],
    )


def test_terminal_validator_is_exact_and_loss_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_valid_fixture(tmp_path, monkeypatch)
    result = _validate(fixture)
    assert result["status"] == validator.PASS
    assert result["artifact_count"] == 3
    assert result["residual_heads_identical"] is True
    assert result["pretrained_encoders_different"] is True
    assert result["distill_head_excluded_from_downstream"] is True
    assert result["training_loss_accessed"] is False
    assert "SECRET_LOSS_MUST_NOT_BE_READ" not in json.dumps(result)


def test_terminal_validator_rejects_unexpected_and_boundary_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_valid_fixture(tmp_path, monkeypatch)
    unexpected = fixture["pretrain_dir"] / "unexpected.txt"
    unexpected.write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact universe is not exact"):
        _validate(fixture)
    unexpected.unlink()

    audit_path = fixture["pretrain_dir"] / pretrain.AUDIT_NAME
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["cpu_fallback"] = True
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CPU fallback"):
        _validate(fixture)

    audit["cpu_fallback"] = False
    audit["device_actual"] = "cpu"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="actual CUDA device"):
        _validate(fixture)

    audit["device_actual"] = "cuda:0"
    audit["seed"] = 7
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="audit seed changed"):
        _validate(fixture)

    audit["seed"] = 20260828
    audit["training"]["steps_each"] = 13_056
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="step count is outside"):
        _validate(fixture)

    audit["training"]["steps_each"] = 13_057
    audit["training"]["first_batch_record_ids"] = ["duplicate", "duplicate"]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="first batch identities are not unique"):
        _validate(fixture)

    audit["training"]["first_batch_record_ids"] = ["first"]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    source_path = fixture["source_manifest"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["new_external_outcome_accessed"] = True
    source_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source manifest binding differs"):
        _validate(fixture)

    source["new_external_outcome_accessed"] = False
    source_path.write_text(json.dumps(source), encoding="utf-8")
    fixture["contract"]["frozen_schedule"]["learning_rate"] = 1e-3
    with pytest.raises(RuntimeError, match="frozen schedule differs"):
        _validate(fixture)


def test_terminal_validator_rejects_attribution_or_point_state_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_valid_fixture(tmp_path, monkeypatch)
    candidate_path = fixture["pretrain_dir"] / pretrain.CANDIDATE_CHECKPOINT
    null_path = fixture["pretrain_dir"] / pretrain.NULL_CHECKPOINT
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    original_null = torch.load(null_path, map_location="cpu", weights_only=False)

    null = copy.deepcopy(original_null)
    null["experiment_id"] = "RND1_OTHER"
    torch.save(null, null_path)
    with pytest.raises(RuntimeError, match="null experiment id differs"):
        _validate(fixture)

    null = copy.deepcopy(original_null)
    encoder_name = next(
        name
        for name, value in null["point_model_state_dict"].items()
        if not name.startswith("residual_head.")
        and value.is_floating_point()
        and value.numel()
    )
    null["point_model_state_dict"][encoder_name].view(-1)[0] = float("nan")
    torch.save(null, null_path)
    with pytest.raises(RuntimeError, match="null state is nonfinite"):
        _validate(fixture)

    null = copy.deepcopy(original_null)
    residual_name = next(
        name
        for name in null["point_model_state_dict"]
        if name.startswith("residual_head.")
    )
    null["point_model_state_dict"][residual_name].view(-1)[0] += 1.0
    torch.save(null, null_path)
    with pytest.raises(RuntimeError, match="residual heads differ"):
        _validate(fixture)

    null = copy.deepcopy(original_null)
    for name in null["point_model_state_dict"]:
        if not name.startswith("residual_head."):
            null["point_model_state_dict"][name] = candidate[
                "point_model_state_dict"
            ][name].clone()
    torch.save(null, null_path)
    with pytest.raises(RuntimeError, match="pretrained encoders are identical"):
        _validate(fixture)

    null = copy.deepcopy(original_null)
    null["point_model_state_dict"]["distill_head.1.weight"] = torch.zeros(1)
    torch.save(null, null_path)
    with pytest.raises(RuntimeError, match="point state key universe differs"):
        _validate(fixture)


def test_atomic_publish_is_exact_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "rnd1_pretrain"
    pretrain._atomic_publish_artifacts(
        output,
        candidate_bytes=b"candidate",
        null_bytes=b"null",
        audit_bytes=b"audit",
    )
    assert {path.name for path in output.iterdir()} == {
        pretrain.CANDIDATE_CHECKPOINT,
        pretrain.NULL_CHECKPOINT,
        pretrain.AUDIT_NAME,
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pretrain._atomic_publish_artifacts(
            output,
            candidate_bytes=b"candidate-2",
            null_bytes=b"null-2",
            audit_bytes=b"audit-2",
        )


def test_atomic_publish_failure_leaves_no_canonical_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "rnd1_pretrain"
    original = pretrain._write_exclusive
    calls = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging write failure")
        original(path, payload)

    monkeypatch.setattr(pretrain, "_write_exclusive", fail_second)
    with pytest.raises(OSError, match="injected staging write failure"):
        pretrain._atomic_publish_artifacts(
            output,
            candidate_bytes=b"candidate",
            null_bytes=b"null",
            audit_bytes=b"audit",
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".rnd1_pretrain.publishing-*"))
