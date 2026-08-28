#!/usr/bin/env python3
"""Run the one-epoch GPU-only paired RNet2 representation distillation."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.reactflow_delta.gpu_runtime import require_cuda_device
from scripts.reactflow_delta.independent_rnet_distill import (
    CONTEXT_BLOCKS,
    DROPOUT,
    FFN_WIDTH,
    NULL_SHIFT,
    STUDENT_INPUT_CHANNELS,
    STUDENT_WIDTH,
    TEACHER_WIDTH,
    DistillBatch,
    IndependentRNetDistillStudent,
    RNet2SingleShardStream,
    assert_encoder_states_differ,
    assert_exact_student_initial_match,
    assert_exact_residual_head_match,
    downstream_point_state_dict,
    encoder_parameter_count,
    make_exact_student_pair,
    paired_teacher_targets,
    pretraining_parameters,
    total_parameter_count,
)
from scripts.reactflow_delta.validate_independent_rnet_distill_contract import (
    ACTIVE_PATH,
    CONTRACT_PATH,
    _load_yaml,
    assert_run_authority,
)


SCHEMA = "reactflow_delta.independent_rnet_distill_pretrain.v1"
CHECKPOINT_SCHEMA = "reactflow_delta.independent_rnet_distill_checkpoint.v1"
EPOCHS = 1
DISTILLATION_SEED = 20_260_828
DATA_ORDER_SEED = 20_260_828
FROZEN_BATCH_SIZE = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
CANDIDATE_CHECKPOINT = "independent_rnet_distill_candidate.pt"
NULL_CHECKPOINT = "independent_rnet_distill_null_shift17.pt"
AUDIT_NAME = "independent_rnet_distill_pretrain_audit.json"
SOURCE_MANIFEST_STATUS = "RNET2_TEACHER_STRUCTURAL_SOURCE_BINDING_EXACT_PASS"


def require_cuda_training_device(requested_device: str) -> torch.device:
    """Resolve CUDA and independently verify an actual CUDA tensor."""

    actual = torch.device(require_cuda_device(requested_device))
    if actual.type != "cuda":
        raise RuntimeError(f"CUDA_REQUIRED: resolved non-CUDA device {actual}")
    probe = torch.ones(1, device=actual)
    if probe.device.type != "cuda":
        raise RuntimeError("CUDA_REQUIRED: pretraining probe silently left CUDA")
    return actual


def _assert_module_cuda(module: torch.nn.Module, *, label: str) -> None:
    for name, parameter in module.named_parameters():
        if parameter.device.type != "cuda":
            raise RuntimeError(f"CUDA_REQUIRED: {label} parameter on CPU: {name}")
    for name, buffer in module.named_buffers():
        if buffer.device.type != "cuda":
            raise RuntimeError(f"CUDA_REQUIRED: {label} buffer on CPU: {name}")


def _assert_optimizer_cuda(optimizer: torch.optim.Optimizer, *, label: str) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.device.type != "cuda":
                raise RuntimeError(
                    f"CUDA_REQUIRED: {label} optimizer state on CPU: {name}"
                )


def _finite_cuda_gradients(module: torch.nn.Module, *, label: str) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        if parameter.grad.device.type != "cuda":
            raise RuntimeError(f"CUDA_REQUIRED: {label} gradient on CPU: {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"nonfinite {label} gradient: {name}")


def _masked_distill_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if prediction.device.type != "cuda" or target.device.type != "cuda":
        raise RuntimeError("CUDA_REQUIRED: prediction or teacher target is not CUDA")
    if mask.device.type != "cuda":
        raise RuntimeError("CUDA_REQUIRED: distillation mask is not CUDA")
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
        raise RuntimeError("distillation prediction/target/mask shapes differ")
    selected = mask.unsqueeze(-1).expand_as(prediction)
    loss = F.smooth_l1_loss(prediction[selected], target[selected], reduction="mean")
    if loss.device.type != "cuda":
        raise RuntimeError("CUDA_REQUIRED: distillation loss is not CUDA")
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite distillation loss")
    return loss


def capture_paired_rng_states(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Capture the stochastic stream immediately before candidate forward."""

    cuda_state = (
        torch.cuda.get_rng_state(device=device) if device.type == "cuda" else None
    )
    return torch.get_rng_state(), cuda_state


def restore_paired_rng_states(
    *,
    cpu_state: torch.Tensor,
    cuda_state: torch.Tensor | None,
    device: torch.device,
) -> None:
    torch.set_rng_state(cpu_state)
    if device.type == "cuda":
        if cuda_state is None:
            raise RuntimeError("CUDA paired RNG snapshot is missing")
        torch.cuda.set_rng_state(cuda_state, device=device)
    elif cuda_state is not None:
        raise RuntimeError("CPU paired RNG snapshot unexpectedly contains CUDA state")


def _uses_bfloat16(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return bool(torch.cuda.is_bf16_supported())


def _autocast_context(*, use_bfloat16: bool):
    if use_bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _move_batch_to_cuda(
    batch: DistillBatch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        batch.student_inputs.device.type != "cpu"
        or batch.teacher_targets.device.type != "cpu"
        or batch.mask.device.type != "cpu"
    ):
        raise RuntimeError("RNet2 teacher reader must hand off a CPU batch")
    local = batch.student_inputs.to(device=device, non_blocking=False)
    teacher = batch.teacher_targets.to(device=device, non_blocking=False)
    mask = batch.mask.to(device=device, non_blocking=False)
    if {local.device.type, teacher.device.type, mask.device.type} != {"cuda"}:
        raise RuntimeError("CUDA_REQUIRED: batch transfer silently left CUDA")
    return local, teacher, mask


def train_exact_paired_students(
    *,
    stream: RNet2SingleShardStream,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[
    IndependentRNetDistillStudent,
    IndependentRNetDistillStudent,
    dict[str, object],
]:
    """Train candidate/null in one shared batch loop with matched dropout RNG."""

    if device.type != "cuda":
        raise RuntimeError("CUDA_REQUIRED: paired pretraining received non-CUDA device")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    candidate, null = make_exact_student_pair(seed=seed, device=device)
    assert_exact_student_initial_match(candidate, null)
    _assert_module_cuda(candidate, label="candidate")
    _assert_module_cuda(null, label="null")
    candidate.train()
    null.train()
    candidate_residual_before = {
        name: value.detach().clone()
        for name, value in candidate.residual_head.state_dict().items()
    }
    null_residual_before = {
        name: value.detach().clone()
        for name, value in null.residual_head.state_dict().items()
    }
    use_bfloat16 = _uses_bfloat16(device)
    candidate_optimizer = torch.optim.AdamW(
        pretraining_parameters(candidate),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        capturable=True,
    )
    null_optimizer = torch.optim.AdamW(
        pretraining_parameters(null),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        capturable=True,
    )

    candidate_loss_sum = 0.0
    null_loss_sum = 0.0
    steps = 0
    records = 0
    first_record_ids: tuple[str, ...] = ()
    last_record_ids: tuple[str, ...] = ()
    for batch in stream.iter_batches(batch_size=batch_size, seed=seed):
        if not first_record_ids:
            first_record_ids = batch.record_ids
        last_record_ids = batch.record_ids
        local, teacher, mask = _move_batch_to_cuda(batch, device)
        candidate_target, null_target = paired_teacher_targets(teacher, mask)
        if candidate_target.device.type != "cuda" or null_target.device.type != "cuda":
            raise RuntimeError("CUDA_REQUIRED: paired teacher targets are not CUDA")

        cpu_rng, cuda_rng = capture_paired_rng_states(device)
        candidate_optimizer.zero_grad(set_to_none=True)
        with _autocast_context(use_bfloat16=use_bfloat16):
            candidate_prediction = candidate(local, mask)
            candidate_loss = _masked_distill_loss(
                candidate_prediction, candidate_target, mask
            )
        candidate_loss.backward()
        _finite_cuda_gradients(candidate, label="candidate")
        torch.nn.utils.clip_grad_norm_(
            pretraining_parameters(candidate), GRADIENT_CLIP
        )
        candidate_optimizer.step()

        # Restore the exact pre-forward RNG so the matched null sees the same
        # dropout masks.  Its only frozen intervention is teacher alignment.
        restore_paired_rng_states(
            cpu_state=cpu_rng, cuda_state=cuda_rng, device=device
        )
        null_optimizer.zero_grad(set_to_none=True)
        with _autocast_context(use_bfloat16=use_bfloat16):
            null_prediction = null(local, mask)
            null_loss = _masked_distill_loss(null_prediction, null_target, mask)
        null_loss.backward()
        _finite_cuda_gradients(null, label="null")
        torch.nn.utils.clip_grad_norm_(pretraining_parameters(null), GRADIENT_CLIP)
        null_optimizer.step()

        _assert_optimizer_cuda(candidate_optimizer, label="candidate")
        _assert_optimizer_cuda(null_optimizer, label="null")
        candidate_loss_sum += float(candidate_loss.detach().item())
        null_loss_sum += float(null_loss.detach().item())
        steps += 1
        records += len(batch.record_ids)

    if records != stream.record_count:
        raise RuntimeError("paired distillation did not consume the exact one-epoch universe")
    if steps < 1:
        raise RuntimeError("paired distillation completed without an optimizer step")
    _assert_module_cuda(candidate, label="candidate_final")
    _assert_module_cuda(null, label="null_final")
    for name, value in candidate_residual_before.items():
        if not torch.equal(value, candidate.residual_head.state_dict()[name]):
            raise RuntimeError(f"candidate residual head changed in distillation: {name}")
    for name, value in null_residual_before.items():
        if not torch.equal(value, null.residual_head.state_dict()[name]):
            raise RuntimeError(f"null residual head changed in distillation: {name}")
    assert_exact_residual_head_match(candidate, null)
    assert_encoder_states_differ(candidate, null)
    return candidate, null, {
        "epochs": EPOCHS,
        "steps_each": steps,
        "records_each": records,
        "same_batch_order": True,
        "same_optimizer_hyperparameters": True,
        "same_dropout_rng_per_step": True,
        "batch_padding_used": False,
        "precision": "bfloat16" if use_bfloat16 else "float32",
        "candidate_null_residual_head_exact_equal": True,
        "candidate_null_encoder_exact_different": True,
        "candidate_mean_training_loss": candidate_loss_sum / steps,
        "null_mean_training_loss": null_loss_sum / steps,
        "first_batch_record_ids": list(first_record_ids),
        "last_batch_record_ids": list(last_record_ids),
    }


def _checkpoint_bytes(
    *,
    model: IndependentRNetDistillStudent,
    experiment_id: str,
    condition: str,
    seed: int,
    source: dict[str, object],
    device: torch.device,
    precision: str,
) -> bytes:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "experiment_id": experiment_id,
        "condition": condition,
        "seed": int(seed),
        "data_order_seed": DATA_ORDER_SEED,
        "epochs": EPOCHS,
        "training_device": str(device),
        "precision": precision,
        "source": source,
        "model": {
            "input_channels": STUDENT_INPUT_CHANNELS,
            "width": STUDENT_WIDTH,
            "context_blocks": CONTEXT_BLOCKS,
            "ffn_width": FFN_WIDTH,
            "teacher_width": TEACHER_WIDTH,
            "dropout": DROPOUT,
            "encoder_parameters": encoder_parameter_count(model),
            "total_parameters": total_parameter_count(model),
        },
        "point_model_state_dict": downstream_point_state_dict(model),
        "distill_head_excluded_from_downstream": True,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _target_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "candidate_checkpoint": output_dir / CANDIDATE_CHECKPOINT,
        "null_checkpoint": output_dir / NULL_CHECKPOINT,
        "audit": output_dir / AUDIT_NAME,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def _atomic_publish_artifacts(
    output_dir: Path,
    *,
    candidate_bytes: bytes,
    null_bytes: bytes,
    audit_bytes: bytes,
) -> None:
    """Publish the exact RND1 trio with one same-filesystem directory rename."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite distillation output {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.publishing-",
            dir=output_dir.parent,
        )
    )
    payloads = {
        CANDIDATE_CHECKPOINT: candidate_bytes,
        NULL_CHECKPOINT: null_bytes,
        AUDIT_NAME: audit_bytes,
    }
    try:
        for name, payload in payloads.items():
            _write_exclusive(staging_dir / name, payload)
        observed = {path.name for path in staging_dir.iterdir()}
        if observed != set(payloads) or any(
            not (staging_dir / name).is_file()
            or (staging_dir / name).stat().st_size != len(payload)
            for name, payload in payloads.items()
        ):
            raise RuntimeError("staged RND1 artifact trio is incomplete")
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite distillation output {output_dir}"
            )
        os.replace(staging_dir, output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _require_mnt_output(output_dir: Path) -> None:
    absolute = output_dir.expanduser().resolve()
    if absolute == Path("/mnt") or Path("/mnt") not in absolute.parents:
        raise RuntimeError("independent RNet distillation artifacts must be under /mnt")


def validate_rnd1_source_authority(
    *,
    repo_root: Path,
    source_manifest_path: Path,
    shard_root: Path,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Bind RND1 to the single active authority and structural source PASS."""

    repo_root = repo_root.resolve()
    authority = assert_run_authority(repo_root, "RND1")
    active = _load_yaml(repo_root / ACTIVE_PATH)
    contract = _load_yaml(repo_root / CONTRACT_PATH)
    active_authority = active["authority"]
    source = contract["source_binding"]
    expected_manifest = Path(source["source_manifest"]).resolve()
    expected_cache = Path(source["source_cache"]).resolve()
    if source_manifest_path.resolve() != expected_manifest or Path(
        active_authority["teacher_source_manifest_path"]
    ).resolve() != expected_manifest:
        raise RuntimeError("RND1 teacher source manifest path is not contract-bound")
    if shard_root.resolve() != expected_cache or Path(
        active_authority["teacher_cache_path"]
    ).resolve() != expected_cache:
        raise RuntimeError("RND1 teacher cache path is not contract-bound")
    if output_dir.resolve() != Path(active_authority["pretraining_dir"]).resolve():
        raise RuntimeError("RND1 pretraining output path is not authority-bound")
    if active_authority["teacher_source_manifest_status"] != SOURCE_MANIFEST_STATUS:
        raise RuntimeError("RND1 active source-manifest status is not exact PASS")
    if not source_manifest_path.is_file():
        raise RuntimeError("RND1 structural source manifest is missing")
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("RND1 structural source manifest must be a JSON object")
    expected = {
        "status": SOURCE_MANIFEST_STATUS,
        "source_cache": str(expected_cache),
        "model_name": source["expected_model_name"],
        "model_version": source["expected_model_version"],
        "weights_sha256": source["expected_weights_sha256"],
        "record_count": 208_905,
        "shard_count": 409,
        "single_feature_dim": 384,
        "index_contains_outcome_fields": False,
        "teacher_pair_features_used": False,
        "live_teacher_used": False,
        "openknot_mutant_outcome_accessed": False,
        "new_external_outcome_accessed": False,
    }
    observed = {name: manifest.get(name) for name in expected}
    if observed != expected:
        raise RuntimeError(
            f"RND1 structural source manifest differs from contract: {observed}"
        )
    return authority, manifest


def run_pretraining(args: argparse.Namespace) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    repo_root = Path(args.repo_root)
    shard_root = Path(args.shard_root)
    source_manifest_path = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    _require_mnt_output(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite distillation output {output_dir}")
    authority, source_manifest = validate_rnd1_source_authority(
        repo_root=repo_root,
        source_manifest_path=source_manifest_path,
        shard_root=shard_root,
        output_dir=output_dir,
    )
    paths = _target_paths(output_dir)
    if int(args.seed) != DISTILLATION_SEED:
        raise RuntimeError("distillation seed differs from frozen contract")
    if int(args.batch_size) != FROZEN_BATCH_SIZE:
        raise RuntimeError("distillation batch size differs from frozen contract")

    # CUDA and source validation precede model construction and every artifact.
    device = require_cuda_training_device(str(args.device))
    stream = RNet2SingleShardStream(shard_root)
    candidate, null, training = train_exact_paired_students(
        stream=stream,
        device=device,
        seed=int(args.seed),
        batch_size=int(args.batch_size),
    )
    source = stream.source_summary()
    finished = datetime.now(timezone.utc)
    audit: dict[str, object] = {
        "schema_version": SCHEMA,
        "experiment_id": str(args.experiment_id),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "command": list(sys.argv),
        "source_root": str(Path(args.shard_root).resolve()),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_status": source_manifest["status"],
        "source": source,
        "authority": authority,
        "seed": int(args.seed),
        "data_order_seed": DATA_ORDER_SEED,
        "batch_size": int(args.batch_size),
        "device_requested": str(args.device),
        "device_actual": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_cuda": torch.version.cuda,
        "candidate_target": "teacher_single_same_position",
        "null_target": f"teacher_single_positive_cyclic_shift_min_{NULL_SHIFT}_L_minus_1",
        "student_input": (
            "sequence_one_hot+zero_reactivity+zero_precision+observed_token+"
            "normalized_position+zero_region+zero_corruption"
        ),
        "loss": "smooth_l1_mean_over_valid_positions_and_384_features",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "capturable": True,
        },
        "training": training,
        "metric_eligibility": "engineering_pretraining_only_not_scientific",
        "outcome_accessed": False,
        "cpu_fallback": False,
        "exit_code": 0,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }

    candidate_bytes = _checkpoint_bytes(
        model=candidate,
        experiment_id=str(args.experiment_id),
        condition="aligned_candidate",
        seed=int(args.seed),
        source=source,
        device=device,
        precision=str(training["precision"]),
    )
    null_bytes = _checkpoint_bytes(
        model=null,
        experiment_id=str(args.experiment_id),
        condition=f"cyclic_shift_{NULL_SHIFT}_null",
        seed=int(args.seed),
        source=source,
        device=device,
        precision=str(training["precision"]),
    )
    audit_bytes = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_publish_artifacts(
        output_dir,
        candidate_bytes=candidate_bytes,
        null_bytes=null_bytes,
        audit_bytes=audit_bytes,
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--device", required=True, help="Explicit cuda:<index>")
    parser.add_argument(
        "--seed", type=int, choices=[DISTILLATION_SEED], default=DISTILLATION_SEED
    )
    parser.add_argument(
        "--batch-size", type=int, choices=[FROZEN_BATCH_SIZE], default=FROZEN_BATCH_SIZE
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if int(args.batch_size) < 1:
        raise SystemExit("--batch-size must be positive")
    audit = run_pretraining(args)
    print(json.dumps(audit, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
