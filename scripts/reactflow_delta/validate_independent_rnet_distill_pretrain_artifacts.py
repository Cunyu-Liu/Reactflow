#!/usr/bin/env python3
"""Loss-blind terminal validator for the exact RND1 pretraining trio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from scripts.reactflow_delta.independent_rnet_distill import (
    ATTENTION_HEADS,
    CONTEXT_BLOCKS,
    DROPOUT,
    FFN_WIDTH,
    RELATIVE_DISTANCE_WINDOW,
    STUDENT_INPUT_CHANNELS,
    STUDENT_WIDTH,
    TEACHER_WIDTH,
    IndependentRNetDistillStudent,
    downstream_point_state_dict,
    encoder_parameter_count,
    load_downstream_point_state_dict,
    total_parameter_count,
)
from scripts.reactflow_delta.run_independent_rnet_distill_pretrain import (
    AUDIT_NAME,
    CANDIDATE_CHECKPOINT,
    CHECKPOINT_SCHEMA,
    DATA_ORDER_SEED,
    DISTILLATION_SEED,
    EPOCHS,
    FROZEN_BATCH_SIZE,
    GRADIENT_CLIP,
    LEARNING_RATE,
    NULL_CHECKPOINT,
    SCHEMA as AUDIT_SCHEMA,
    SOURCE_MANIFEST_STATUS,
    WEIGHT_DECAY,
)
from scripts.reactflow_delta.validate_independent_rnet_distill_contract import (
    ACTIVE_PATH,
    CONTRACT_PATH,
    PROJECT_TASK_ID,
    _load_yaml,
    assert_run_authority,
)


SCHEMA = "reactflow_delta.independent_rnet_distill_pretrain_validation.v1"
PASS = "RND1_PAIRED_PRETRAIN_EXACT_PASS"
SOURCE_SCHEMA = "reactflow_delta.independent_rnet_distill_teacher_source.v1"

EXPECTED_FILENAMES = {
    CANDIDATE_CHECKPOINT,
    NULL_CHECKPOINT,
    AUDIT_NAME,
}
CHECKPOINT_FIELDS = {
    "schema_version",
    "experiment_id",
    "condition",
    "seed",
    "data_order_seed",
    "epochs",
    "training_device",
    "precision",
    "source",
    "model",
    "point_model_state_dict",
    "distill_head_excluded_from_downstream",
}
AUDIT_FIELDS = {
    "schema_version",
    "experiment_id",
    "started_at_utc",
    "finished_at_utc",
    "command",
    "source_root",
    "source_manifest",
    "source_manifest_status",
    "source",
    "authority",
    "seed",
    "data_order_seed",
    "batch_size",
    "device_requested",
    "device_actual",
    "gpu_name",
    "torch_cuda",
    "candidate_target",
    "null_target",
    "student_input",
    "loss",
    "optimizer",
    "training",
    "metric_eligibility",
    "outcome_accessed",
    "cpu_fallback",
    "exit_code",
    "artifacts",
}
TRAINING_FIELDS = {
    "epochs",
    "steps_each",
    "records_each",
    "same_batch_order",
    "same_optimizer_hyperparameters",
    "same_dropout_rng_per_step",
    "batch_padding_used",
    "precision",
    "candidate_null_residual_head_exact_equal",
    "candidate_null_encoder_exact_different",
    "candidate_mean_training_loss",
    "null_mean_training_loss",
    "first_batch_record_ids",
    "last_batch_record_ids",
}
SOURCE_FIELDS = {
    "schema_version",
    "status",
    "project_task_id",
    "source_cache",
    "layout",
    "model_name",
    "model_version",
    "weights_sha256",
    "record_count",
    "shard_count",
    "single_feature_dim",
    "legacy_parent_content_hashes_authoritative",
    "legacy_parent_content_binding_mismatches",
    "verified_recovered_last_shard_content_sha256",
    "payload_hash_verification_scope",
    "full_cache_rehash_performed",
    "structural_binding_basis",
    "index_contains_outcome_fields",
    "teacher_pair_features_used",
    "live_teacher_used",
    "openknot_mutant_outcome_accessed",
    "new_external_outcome_accessed",
    "exact_sequence_overlap_prior_audit",
    "scientific_evidence_ceiling",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"expected JSON object: {path}")
    return payload


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), f"checkpoint is not a mapping: {path}")
    return payload


def _expected_source(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["source_binding"]
    mismatches = int(
        source["source_integrity_repair"][
            "expected_parent_child_content_binding_mismatches"
        ]
    )
    return {
        "layout": source["expected_layout"],
        "model_name": source["expected_model_name"],
        "model_version": source["expected_model_version"],
        "record_count": int(source["expected_record_count"]),
        "shard_count": int(source["expected_shard_count"]),
        "shard_size": int(source["expected_full_shard_size"]),
        "weights_sha256": source["expected_weights_sha256"],
        "single_schema": {
            "axes": ["L", int(source["expected_single_feature_dim"])],
            "dtype": "<f4",
        },
        "legacy_parent_content_binding_mismatch_count": mismatches,
        "runtime_parent_content_binding_mismatch_count": mismatches,
        "root_content_hashes_are_authority": False,
        "full_cache_rehash_performed": False,
    }


def _validate_contract_rnd1(contract: dict[str, Any]) -> dict[str, Any]:
    """Bind the terminal evidence to the frozen RND1 schedule and student."""

    _require(contract["project_task_id"] == PROJECT_TASK_ID, "wrong RND1 project")
    schedule = contract["frozen_schedule"]
    _require(
        {
            "distillation_seed": schedule["distillation_seed"],
            "data_order_seed": schedule["data_order_seed"],
            "distillation_epochs": schedule["distillation_epochs"],
            "optimizer": schedule["optimizer"],
            "learning_rate": schedule["learning_rate"],
            "weight_decay": schedule["weight_decay"],
            "batch_size": schedule["batch_size"],
            "gradient_clip_norm": schedule["gradient_clip_norm"],
            "mixed_precision": schedule["mixed_precision"],
        }
        == {
            "distillation_seed": DISTILLATION_SEED,
            "data_order_seed": DATA_ORDER_SEED,
            "distillation_epochs": EPOCHS,
            "optimizer": "ADAMW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": FROZEN_BATCH_SIZE,
            "gradient_clip_norm": GRADIENT_CLIP,
            "mixed_precision": "BFLOAT16_ON_SUPPORTED_CUDA_ELSE_FLOAT32_CUDA",
        },
        "RND1 frozen schedule differs from the pretraining implementation",
    )
    student = contract["student"]
    _require(
        {
            "input_channels": student["input_channels"],
            "context_width": student["context_width"],
            "attention_heads": student["attention_heads"],
            "context_blocks": student["context_blocks"],
            "ffn_width": student["ffn_width"],
            "relative_distance_window": student["relative_distance_window"],
            "dropout": student["dropout"],
            "distillation_head": student["distillation_head"],
            "downstream_head": student["downstream_head"],
            "strict_length_identity": student["strict_length_identity"],
            "silent_resize_or_padding_allowed": student[
                "silent_resize_or_padding_allowed"
            ],
        }
        == {
            "input_channels": STUDENT_INPUT_CHANNELS,
            "context_width": STUDENT_WIDTH,
            "attention_heads": ATTENTION_HEADS,
            "context_blocks": CONTEXT_BLOCKS,
            "ffn_width": FFN_WIDTH,
            "relative_distance_window": RELATIVE_DISTANCE_WINDOW,
            "dropout": DROPOUT,
            "distillation_head": "LAYERNORM_256_LINEAR_384",
            "downstream_head": "FEATURE41_ANCHORED_V14_RESIDUAL_HEAD",
            "strict_length_identity": (
                "INDEX_LENGTH_EQUALS_SEQUENCE_LENGTH_EQUALS_TEACHER_L_"
                "EQUALS_STUDENT_TOKEN_LENGTH"
            ),
            "silent_resize_or_padding_allowed": False,
        },
        "RND1 student contract differs from the point-model implementation",
    )
    _require(
        student["pretraining_inputs"]
        == {
            "sequence": "ONE_HOT_A_C_G_U",
            "reactivity": "ZERO",
            "precision": "ZERO",
            "observed": "TRUE_FOR_VALID_TOKENS",
            "position": "REGISTERED_ZERO_BASED_INDEX",
            "region": "ZERO",
            "corruption": False,
        },
        "RND1 student input contract changed",
    )
    _require(
        contract["phase_contract"]["RND1"]["outputs"]
        == ["candidate_checkpoint", "null_checkpoint", "pretraining_audit"],
        "RND1 output universe changed",
    )
    _require(
        contract["phase_contract"]["RND2"]["required_predecessor"] == PASS,
        "RND2 predecessor does not name the canonical RND1 terminal PASS",
    )
    gpu = contract["gpu_policy"]
    _require(
        gpu["training_and_gpu_validation_device_class"] == "CUDA_ONLY"
        and gpu["cpu_model_or_loss_fallback_allowed"] is False
        and gpu["actual_model_input_target_output_loss_and_optimizer_state_must_be_cuda"]
        is True,
        "RND1 CUDA-only contract changed",
    )
    outcome = contract["outcome_policy"]
    _require(
        outcome["pretraining_may_read_openknot_mutant_outcome"] is False
        and outcome["new_external_outcome_access_allowed"] is False,
        "RND1 outcome boundary changed",
    )
    return {
        "input_channels": STUDENT_INPUT_CHANNELS,
        "width": STUDENT_WIDTH,
        "context_blocks": CONTEXT_BLOCKS,
        "ffn_width": FFN_WIDTH,
        "teacher_width": TEACHER_WIDTH,
        "dropout": DROPOUT,
    }


def _validate_source_manifest(
    manifest: dict[str, Any],
    *,
    contract: dict[str, Any],
    source_manifest_path: Path,
) -> None:
    source = contract["source_binding"]
    repair = source["source_integrity_repair"]
    _require(set(manifest) == SOURCE_FIELDS, "teacher source field universe changed")
    expected = {
        "schema_version": SOURCE_SCHEMA,
        "status": SOURCE_MANIFEST_STATUS,
        "project_task_id": PROJECT_TASK_ID,
        "source_cache": str(Path(source["source_cache"]).resolve()),
        "layout": source["expected_layout"],
        "model_name": source["expected_model_name"],
        "model_version": source["expected_model_version"],
        "weights_sha256": source["expected_weights_sha256"],
        "record_count": int(source["expected_record_count"]),
        "shard_count": int(source["expected_shard_count"]),
        "single_feature_dim": int(source["expected_single_feature_dim"]),
        "legacy_parent_content_hashes_authoritative": False,
        "legacy_parent_content_binding_mismatches": int(
            repair["expected_parent_child_content_binding_mismatches"]
        ),
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
        "scientific_evidence_ceiling": contract["scope"]["result_class"],
    }
    observed = {name: manifest.get(name) for name in expected}
    _require(observed == expected, "teacher source manifest binding differs")
    overlap = manifest.get("exact_sequence_overlap_prior_audit")
    _require(
        overlap
        == {
            "observed_overlap": source[
                "exact_sequence_overlap_with_registered_openknot_mutants"
            ],
            "registered_sequence_count": source[
                "registered_openknot_mutant_sequence_count"
            ],
            "interpretation": source["overlap_audit_interpretation"],
        },
        "teacher source overlap disclosure differs",
    )
    _require(source_manifest_path.is_file(), "teacher source manifest is missing")


def _validate_audit(
    audit: dict[str, Any],
    *,
    authority: dict[str, Any],
    expected_source: dict[str, Any],
    expected_source_root: Path,
    pretrain_dir: Path,
    source_manifest_path: Path,
) -> None:
    _require(set(audit) == AUDIT_FIELDS, "pretraining audit field universe changed")
    _require(audit["schema_version"] == AUDIT_SCHEMA, "pretraining audit schema changed")
    _require(isinstance(audit["experiment_id"], str) and audit["experiment_id"], "experiment id is missing")
    _require(isinstance(audit["started_at_utc"], str) and audit["started_at_utc"], "start time is missing")
    _require(isinstance(audit["finished_at_utc"], str) and audit["finished_at_utc"], "finish time is missing")
    _require(isinstance(audit["command"], list) and audit["command"], "training command is missing")
    _require(audit["source_root"] == str(expected_source_root), "teacher cache path differs")
    _require(audit["source_manifest"] == str(source_manifest_path), "source manifest path differs")
    _require(audit["source_manifest_status"] == SOURCE_MANIFEST_STATUS, "source manifest status differs")
    _require(audit["source"] == expected_source, "audit source binding differs")
    _require(audit["authority"] == authority, "audit authority binding differs")
    _require(int(audit["seed"]) == DISTILLATION_SEED, "audit seed changed")
    _require(int(audit["data_order_seed"]) == DATA_ORDER_SEED, "audit data-order seed changed")
    _require(int(audit["batch_size"]) == FROZEN_BATCH_SIZE, "audit batch size changed")
    _require(str(audit["device_requested"]).startswith("cuda:"), "requested CUDA device is absent")
    _require(str(audit["device_actual"]).startswith("cuda:"), "actual CUDA device is absent")
    try:
        requested_device = torch.device(str(audit["device_requested"]))
        actual_device = torch.device(str(audit["device_actual"]))
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError("recorded CUDA device is invalid") from error
    _require(
        requested_device.type == "cuda"
        and requested_device.index is not None
        and actual_device.type == "cuda"
        and actual_device.index is not None
        and requested_device == actual_device,
        "requested and actual CUDA devices differ",
    )
    _require(isinstance(audit["gpu_name"], str) and audit["gpu_name"], "GPU identity is absent")
    _require(isinstance(audit["torch_cuda"], str) and audit["torch_cuda"], "torch CUDA version is absent")
    _require(audit["candidate_target"] == "teacher_single_same_position", "candidate target changed")
    _require(audit["null_target"] == "teacher_single_positive_cyclic_shift_min_17_L_minus_1", "null target changed")
    _require(
        audit["student_input"]
        == (
            "sequence_one_hot+zero_reactivity+zero_precision+observed_token+"
            "normalized_position+zero_region+zero_corruption"
        ),
        "student pretraining input contract changed",
    )
    _require(
        audit["loss"] == "smooth_l1_mean_over_valid_positions_and_384_features",
        "distillation loss contract changed",
    )
    _require(
        audit["optimizer"]
        == {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "capturable": True,
        },
        "optimizer contract changed",
    )
    _require(audit["metric_eligibility"] == "engineering_pretraining_only_not_scientific", "metric eligibility changed")
    _require(audit["outcome_accessed"] is False, "pretraining outcome boundary changed")
    _require(audit["cpu_fallback"] is False, "CPU fallback was recorded")
    _require(int(audit["exit_code"]) == 0, "pretraining exit code is nonzero")
    expected_artifacts = {
        "candidate_checkpoint": str(pretrain_dir / CANDIDATE_CHECKPOINT),
        "null_checkpoint": str(pretrain_dir / NULL_CHECKPOINT),
        "audit": str(pretrain_dir / AUDIT_NAME),
    }
    _require(audit["artifacts"] == expected_artifacts, "audit artifact paths differ")

    training = audit["training"]
    _require(isinstance(training, dict), "training invariant block is missing")
    _require(set(training) == TRAINING_FIELDS, "training invariant field universe changed")
    # Deliberately do not access either *_training_loss value.
    expected_training = {
        "epochs": EPOCHS,
        "records_each": int(expected_source["record_count"]),
        "same_batch_order": True,
        "same_optimizer_hyperparameters": True,
        "same_dropout_rng_per_step": True,
        "batch_padding_used": False,
        "candidate_null_residual_head_exact_equal": True,
        "candidate_null_encoder_exact_different": True,
    }
    observed_training = {name: training.get(name) for name in expected_training}
    _require(observed_training == expected_training, "training invariants differ")
    record_count = int(expected_source["record_count"])
    minimum_steps = (record_count + FROZEN_BATCH_SIZE - 1) // FROZEN_BATCH_SIZE
    steps = int(training["steps_each"])
    _require(
        minimum_steps <= steps <= record_count,
        "paired step count is outside exact-length batching bounds",
    )
    _require(training["precision"] in {"bfloat16", "float32"}, "training precision changed")
    for label, field in (
        ("first", "first_batch_record_ids"),
        ("last", "last_batch_record_ids"),
    ):
        record_ids = training[field]
        _require(
            isinstance(record_ids, list)
            and 1 <= len(record_ids) <= FROZEN_BATCH_SIZE
            and all(isinstance(record_id, str) and record_id for record_id in record_ids),
            f"{label} batch identity count or type is invalid",
        )
        _require(
            len(set(record_ids)) == len(record_ids),
            f"{label} batch identities are not unique",
        )


def _validate_checkpoint(
    payload: dict[str, Any],
    *,
    label: str,
    condition: str,
    audit: dict[str, Any],
    expected_source: dict[str, Any],
    expected_model: dict[str, Any],
    expected_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _require(set(payload) == CHECKPOINT_FIELDS, f"{label} checkpoint field universe changed")
    _require(payload["schema_version"] == CHECKPOINT_SCHEMA, f"{label} checkpoint schema changed")
    _require(payload["experiment_id"] == audit["experiment_id"], f"{label} experiment id differs")
    _require(payload["condition"] == condition, f"{label} condition changed")
    _require(int(payload["seed"]) == DISTILLATION_SEED, f"{label} seed changed")
    _require(int(payload["data_order_seed"]) == DATA_ORDER_SEED, f"{label} data-order seed changed")
    _require(int(payload["epochs"]) == EPOCHS, f"{label} epoch count changed")
    _require(payload["training_device"] == audit["device_actual"], f"{label} actual CUDA binding differs")
    _require(payload["precision"] == audit["training"]["precision"], f"{label} precision differs")
    _require(payload["source"] == expected_source, f"{label} source binding differs")
    _require(payload["model"] == expected_model, f"{label} model contract differs")
    _require(payload["distill_head_excluded_from_downstream"] is True, f"{label} distill-head flag differs")
    state = payload["point_model_state_dict"]
    _require(isinstance(state, dict) and state, f"{label} point state is missing")
    _require(state.keys() == expected_state.keys(), f"{label} point state key universe differs")
    _require(not any(str(name).startswith("distill_head.") for name in state), f"{label} point state contains distill head")
    for name, expected_tensor in expected_state.items():
        value = state[name]
        _require(isinstance(value, torch.Tensor), f"{label} state is not tensor at {name}")
        _require(value.shape == expected_tensor.shape, f"{label} state shape differs at {name}")
        _require(value.dtype == expected_tensor.dtype, f"{label} state dtype differs at {name}")
        _require(
            bool(torch.isfinite(value).all()),
            f"{label} state is nonfinite at {name}",
        )
    model = IndependentRNetDistillStudent()
    load_downstream_point_state_dict(model, state)
    return state


def validate_pretraining_artifacts(
    *,
    repo_root: Path,
    pretrain_dir: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Validate RND1 structurally without accessing training-loss values."""

    repo_root = repo_root.resolve()
    pretrain_dir = pretrain_dir.resolve()
    source_manifest_path = source_manifest_path.resolve()
    authority = assert_run_authority(repo_root, "RND1")
    active = _load_yaml(repo_root / ACTIVE_PATH)
    contract = _load_yaml(repo_root / CONTRACT_PATH)
    expected_model = _validate_contract_rnd1(contract)
    active_authority = active["authority"]
    _require(pretrain_dir == Path(active_authority["pretraining_dir"]).resolve(), "pretraining directory is not authority-bound")
    _require(source_manifest_path == Path(active_authority["teacher_source_manifest_path"]).resolve(), "source manifest path is not authority-bound")
    _require(source_manifest_path == Path(contract["source_binding"]["source_manifest"]).resolve(), "source manifest path is not contract-bound")
    _require(active_authority["teacher_source_manifest_status"] == SOURCE_MANIFEST_STATUS, "active source status differs")
    _require(pretrain_dir.is_dir(), "pretraining directory is missing")
    observed_names = {path.name for path in pretrain_dir.iterdir()}
    _require(observed_names == EXPECTED_FILENAMES, "RND1 artifact universe is not exact")
    for name in EXPECTED_FILENAMES:
        path = pretrain_dir / name
        _require(path.is_file() and path.stat().st_size > 0, f"RND1 artifact is empty: {name}")

    source_manifest = _read_json(source_manifest_path)
    _validate_source_manifest(
        source_manifest,
        contract=contract,
        source_manifest_path=source_manifest_path,
    )
    expected_source = _expected_source(contract)
    audit = _read_json(pretrain_dir / AUDIT_NAME)
    _validate_audit(
        audit,
        authority=authority,
        expected_source=expected_source,
        expected_source_root=Path(
            contract["source_binding"]["source_cache"]
        ).resolve(),
        pretrain_dir=pretrain_dir,
        source_manifest_path=source_manifest_path,
    )

    reference = IndependentRNetDistillStudent()
    expected_state = downstream_point_state_dict(reference)
    expected_model = {
        **expected_model,
        "encoder_parameters": encoder_parameter_count(reference),
        "total_parameters": total_parameter_count(reference),
    }
    candidate_state = _validate_checkpoint(
        _load_checkpoint(pretrain_dir / CANDIDATE_CHECKPOINT),
        label="candidate",
        condition="aligned_candidate",
        audit=audit,
        expected_source=expected_source,
        expected_model=expected_model,
        expected_state=expected_state,
    )
    null_state = _validate_checkpoint(
        _load_checkpoint(pretrain_dir / NULL_CHECKPOINT),
        label="null",
        condition="cyclic_shift_17_null",
        audit=audit,
        expected_source=expected_source,
        expected_model=expected_model,
        expected_state=expected_state,
    )
    residual_names = sorted(
        name for name in expected_state if name.startswith("residual_head.")
    )
    encoder_names = sorted(
        name for name in expected_state if not name.startswith("residual_head.")
    )
    _require(residual_names and encoder_names, "point-model state partition is empty")
    for name in residual_names:
        _require(
            torch.equal(candidate_state[name], null_state[name]),
            f"candidate/null residual heads differ at {name}",
        )
    changed_encoder_names = [
        name
        for name in encoder_names
        if not torch.equal(candidate_state[name], null_state[name])
    ]
    _require(changed_encoder_names, "candidate/null pretrained encoders are identical")

    return {
        "schema_version": SCHEMA,
        "status": PASS,
        "project_task_id": PROJECT_TASK_ID,
        "experiment_id": audit["experiment_id"],
        "artifact_count": len(EXPECTED_FILENAMES),
        "source_manifest_status": SOURCE_MANIFEST_STATUS,
        "device_actual": audit["device_actual"],
        "precision": audit["training"]["precision"],
        "cpu_fallback": False,
        "outcome_accessed": False,
        "residual_heads_identical": True,
        "pretrained_encoders_different": True,
        "changed_encoder_tensor_count": len(changed_encoder_names),
        "distill_head_excluded_from_downstream": True,
        "training_loss_accessed": False,
        "scientific_metric_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_pretraining_artifacts(
        repo_root=args.repo_root,
        pretrain_dir=args.pretrain_dir,
        source_manifest_path=args.source_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
