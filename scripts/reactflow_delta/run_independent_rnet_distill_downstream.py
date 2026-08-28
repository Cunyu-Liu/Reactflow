#!/usr/bin/env python3
"""Run GPU-only, score-blind downstream folds for independent RNet2 distillation."""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.gpu_runtime import require_cuda_device
from scripts.reactflow_delta.independent_rnet_distill import (
    IndependentRNetDistillStudent,
    downstream_point_state_dict,
    load_downstream_point_state_dict,
)
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v10 import parameter_count as residual_count
from scripts.reactflow_delta.model_rescue_v11 import (
    freeze_point_model,
    method_cell_balanced_l1,
)
from scripts.reactflow_delta.model_rescue_v2 import freeze_mean_model
from scripts.reactflow_delta.model_rescue_v5_probe import EnsembleFeatureCache
from scripts.reactflow_delta.model_rescue_v6_probe import (
    ConstrainedFeatureCache,
    validate_cache_alignment,
)
from scripts.reactflow_delta.run_model_rescue_v10 import _mutant_balanced_crps
from scripts.reactflow_delta.run_model_rescue_v11 import (
    _assert_unchanged,
    _calibration_cells,
    _feature41_replay_max_difference,
    _fold_sources,
    _held_prediction,
    _load_authoritative_v10_feature41,
    _load_v8_mean,
    _new_residual_heads,
    _parse_folds,
    _point_cells,
    _prepare_calibration_inputs,
    _read_json,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta.validate_independent_rnet_distill_contract import (
    ACTIVE_PATH,
    CONTRACT_PATH,
    LEDGER_PATH,
    assert_run_authority,
)


FOLD_SCHEMA = "reactflow_delta.independent_rnet_distill_fold.v1"
PREDICTION_SCHEMA = "reactflow_delta.independent_rnet_distill_prediction.v1"
POINT_CHECKPOINT_SCHEMA = (
    "reactflow_delta.independent_rnet_distill_downstream_point_checkpoint.v1"
)
PRETRAIN_CHECKPOINT_SCHEMA = "reactflow_delta.independent_rnet_distill_checkpoint.v1"
EXPECTED_SEED = 0
EXPECTED_FOLDS = {
    "RND2": (0, 1),
    "RND3": tuple(range(20)),
}
EXPECTED_SCHEDULE = {
    "RND2": (3, 3),
    "RND3": (40, 40),
}
EXPECTED_EXPERIMENT_ID = {
    "RND2": "RND2_RNET_DISTILL_TWO_FOLD_GPU_ENGINEERING_SMOKE",
    "RND3": "RND3_RNET_DISTILL_COMPLETE_SEED0_PREDICTION_ONLY",
}
EVIDENCE_STATUS = {
    "RND2": "ENGINEERING_SMOKE_ONLY_NOT_SCIENTIFIC",
    "RND3": "EXPOSURE_DISCLOSED_DEVELOPMENT_PREDICTION_ONLY",
}
PRETRAIN_FILENAMES = {
    "candidate": "independent_rnet_distill_candidate.pt",
    "null": "independent_rnet_distill_null_shift17.pt",
    "audit": "independent_rnet_distill_pretrain_audit.json",
}
POINT_NAMES = ("feature41", "candidate", "null")
V11_POINT_NAMES = ("feature41", "anchored", "unanchored")
EXPECTED_PREDICTION_FIELDS = frozenset(
    {
        "schema_version",
        "keys",
        "biological_scoring_key",
        "outer_fold",
        "seed",
        "registered_status",
        "feature41_point",
        "v8_point",
        "candidate_point",
        "null_point",
        "feature41_weights",
        "feature41_locations",
        "feature41_scales",
        "feature41_expected_absolute_delta",
        "candidate_weights",
        "candidate_locations",
        "candidate_scales",
        "candidate_expected_absolute_delta",
        "null_weights",
        "null_locations",
        "null_scales",
        "null_expected_absolute_delta",
        "historical_v10_weights",
        "historical_v10_locations",
        "historical_v10_scales",
        "historical_v10_expected_absolute_delta",
    }
)
FORBIDDEN_PREDICTION_FIELDS = frozenset(
    {
        "target",
        "targets",
        "target_error",
        "qualified_target_mask",
        "qualified_mask",
        "loss",
        "score",
        "crps",
        "mae",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_mnt_artifact_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == Path("/mnt") or Path("/mnt") not in resolved.parents:
        raise RuntimeError("independent RNet downstream artifacts must be under /mnt")
    return resolved


def _read_active_contract(repo_root: Path) -> dict[str, Any]:
    active = yaml.safe_load((repo_root / ACTIVE_PATH).read_text(encoding="utf-8"))
    if not isinstance(active, dict) or not isinstance(active.get("authority"), dict):
        raise RuntimeError("independent RNet active contract is not a mapping")
    return active


def canonical_downstream_paths(repo_root: Path, phase: str) -> dict[str, Path]:
    """Return the active contract's exact downstream paths for one phase."""

    if phase not in EXPECTED_FOLDS:
        raise ValueError(f"unsupported downstream phase: {phase}")
    active = _read_active_contract(repo_root.resolve())
    authority = active["authority"]
    if authority.get("current_phase") != phase:
        raise RuntimeError(
            f"active phase {authority.get('current_phase')!r} does not bind {phase}"
        )
    output_key = "smoke_prediction_dir" if phase == "RND2" else "screen_prediction_dir"
    keys = {
        "m2_csv": "m2_csv_path",
        "pretrain_dir": "pretraining_dir",
        "v8_dir": "historical_v8_dir",
        "v10_dir": "historical_v10_dir",
        "tic2a_merged_json": "tic2a_merged_registry_path",
        "unconstrained_cache": "unconstrained_feature_cache_path",
        "constrained_cache": "constrained_feature_cache_path",
        "out_dir": output_key,
    }
    missing = [active_key for active_key in keys.values() if not authority.get(active_key)]
    if missing:
        raise RuntimeError(f"active downstream path binding is incomplete: {missing}")
    return {
        cli_name: Path(str(authority[active_key])).expanduser().resolve()
        for cli_name, active_key in keys.items()
    }


def validate_downstream_cli_binding(
    repo_root: Path, args: argparse.Namespace
) -> dict[str, str]:
    """Bind every direct-runner input and output to the active authority."""

    phase = str(args.phase)
    expected_experiment = EXPECTED_EXPERIMENT_ID[phase]
    if str(args.experiment_id) != expected_experiment:
        raise RuntimeError(
            "downstream experiment_id differs: "
            f"observed={args.experiment_id!r} expected={expected_experiment!r}"
        )
    canonical = canonical_downstream_paths(repo_root, phase)
    for cli_name, expected in canonical.items():
        observed = Path(getattr(args, cli_name)).expanduser().resolve()
        if observed != expected:
            raise RuntimeError(
                f"downstream {cli_name} path differs: observed={observed} expected={expected}"
            )
    return {
        "phase": phase,
        "experiment_id": expected_experiment,
        **{name: str(path) for name, path in canonical.items()},
    }


def _assert_tensor_cuda(value: torch.Tensor, *, label: str) -> None:
    if value.device.type != "cuda":
        raise RuntimeError(f"CUDA_REQUIRED: {label} is on {value.device}")


def _assert_module_cuda(module: torch.nn.Module, *, label: str) -> None:
    for name, parameter in module.named_parameters():
        _assert_tensor_cuda(parameter, label=f"{label} parameter {name}")
    for name, buffer in module.named_buffers():
        _assert_tensor_cuda(buffer, label=f"{label} buffer {name}")


def _assert_optimizer_cuda(
    optimizer: torch.optim.Optimizer, *, label: str
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                _assert_tensor_cuda(value, label=f"{label} optimizer state {name}")


def _assert_finite_cuda_gradients(module: torch.nn.Module, *, label: str) -> None:
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        _assert_tensor_cuda(parameter.grad, label=f"{label} gradient {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"nonfinite {label} gradient: {name}")


def _assert_context_cuda(context: tuple[torch.Tensor, ...], *, label: str) -> None:
    if len(context) != 6:
        raise RuntimeError(f"{label} context must contain six tensors")
    for index, tensor in enumerate(context):
        _assert_tensor_cuda(tensor, label=f"{label} context[{index}]")


def _point_parameters(
    model: IndependentRNetDistillStudent,
) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith("distill_head."):
            parameter.requires_grad_(False)
            parameter.grad = None
        else:
            parameter.requires_grad_(True)
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError("independent RNet point parameter universe is empty")
    return parameters


def _reset_downstream_rng(seed: int) -> int:
    """Reset both RNG domains so each arm receives the same dropout stream."""

    stream_seed = int(seed) + 2_800_000
    torch.manual_seed(stream_seed)
    torch.cuda.manual_seed_all(stream_seed)
    return stream_seed


def _downstream_epoch_order(n_cells: int, *, seed: int, epoch: int) -> list[int]:
    order = list(range(int(n_cells)))
    random.Random(int(seed) * 100_003 + int(epoch)).shuffle(order)
    return order


def _state_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _assert_state_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], *, label: str
) -> None:
    if left.keys() != right.keys():
        raise RuntimeError(f"{label} key universe differs")
    for name in left:
        if left[name].shape != right[name].shape or not torch.equal(
            left[name], right[name]
        ):
            raise RuntimeError(f"{label} differs at {name}")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen RNet distillation checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"RNet distillation checkpoint is not a mapping: {path}")
    return payload


def validate_pretrained_pair(
    candidate_payload: dict[str, Any], null_payload: dict[str, Any]
) -> dict[str, Any]:
    """Validate the only scientific intervention before downstream training."""

    required_checkpoint_fields = {
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
    for label, payload in (("candidate", candidate_payload), ("null", null_payload)):
        if set(payload) != required_checkpoint_fields:
            raise RuntimeError(f"{label} pretraining checkpoint field universe changed")
        if payload.get("schema_version") != PRETRAIN_CHECKPOINT_SCHEMA:
            raise RuntimeError(f"{label} pretraining checkpoint schema changed")
        if int(payload.get("seed", -1)) != 20260828:
            raise RuntimeError(f"{label} pretraining checkpoint seed changed")
        if int(payload.get("data_order_seed", -1)) != 20260828:
            raise RuntimeError(f"{label} pretraining data-order seed changed")
        if int(payload.get("epochs", -1)) != 1:
            raise RuntimeError(f"{label} pretraining epoch count changed")
        if not str(payload.get("training_device", "")).startswith("cuda:"):
            raise RuntimeError(f"{label} pretraining checkpoint lacks CUDA evidence")
        if payload.get("precision") not in {"bfloat16", "float32"}:
            raise RuntimeError(f"{label} pretraining precision changed")
        if payload.get("distill_head_excluded_from_downstream") is not True:
            raise RuntimeError(f"{label} checkpoint did not exclude distill_head")
        state = payload.get("point_model_state_dict")
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"{label} checkpoint lacks point_model_state_dict")
        if any(str(name).startswith("distill_head.") for name in state):
            raise RuntimeError(f"{label} downstream state contains distill_head")

    if candidate_payload.get("condition") != "aligned_candidate":
        raise RuntimeError("candidate checkpoint condition changed")
    if null_payload.get("condition") != "cyclic_shift_17_null":
        raise RuntimeError("null checkpoint condition changed")
    if candidate_payload.get("source") != null_payload.get("source"):
        raise RuntimeError("candidate/null teacher source binding differs")
    if candidate_payload.get("model") != null_payload.get("model"):
        raise RuntimeError("candidate/null pretraining model contract differs")
    if candidate_payload.get("experiment_id") != null_payload.get("experiment_id"):
        raise RuntimeError("candidate/null pretraining experiment binding differs")

    candidate_state = candidate_payload["point_model_state_dict"]
    null_state = null_payload["point_model_state_dict"]
    if candidate_state.keys() != null_state.keys():
        raise RuntimeError("candidate/null downstream state key universe differs")
    residual_names = sorted(
        name for name in candidate_state if str(name).startswith("residual_head.")
    )
    encoder_names = sorted(
        name for name in candidate_state if not str(name).startswith("residual_head.")
    )
    if not residual_names or not encoder_names:
        raise RuntimeError("checkpoint lacks encoder or residual-head state")
    _assert_state_equal(
        {name: candidate_state[name] for name in residual_names},
        {name: null_state[name] for name in residual_names},
        label="candidate/null residual initialization",
    )
    changed_encoder_names = [
        name
        for name in encoder_names
        if not torch.equal(candidate_state[name], null_state[name])
    ]
    if not changed_encoder_names:
        raise RuntimeError("candidate/null pretrained encoders are identical")
    return {
        "same_source_binding": True,
        "residual_heads_identical": True,
        "pretrained_encoders_different": True,
        "changed_encoder_tensor_count": len(changed_encoder_names),
        "point_state_tensor_count": len(candidate_state),
    }


def load_pretrained_pair(
    *, pretrain_dir: Path, device: str
) -> tuple[
    IndependentRNetDistillStudent,
    IndependentRNetDistillStudent,
    dict[str, Any],
]:
    candidate_path = pretrain_dir / PRETRAIN_FILENAMES["candidate"]
    null_path = pretrain_dir / PRETRAIN_FILENAMES["null"]
    audit_path = pretrain_dir / PRETRAIN_FILENAMES["audit"]
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing paired pretraining audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or audit.get("outcome_accessed") is not False:
        raise RuntimeError("paired pretraining audit outcome boundary is invalid")

    candidate_payload = _load_checkpoint(candidate_path)
    null_payload = _load_checkpoint(null_path)
    pair_audit = validate_pretrained_pair(candidate_payload, null_payload)
    candidate = IndependentRNetDistillStudent().to(device)
    null = copy.deepcopy(candidate)
    load_downstream_point_state_dict(
        candidate, candidate_payload["point_model_state_dict"]
    )
    load_downstream_point_state_dict(null, null_payload["point_model_state_dict"])
    _point_parameters(candidate)
    _point_parameters(null)
    _assert_module_cuda(candidate, label="candidate pretrained point model")
    _assert_module_cuda(null, label="null pretrained point model")
    return candidate, null, {
        **pair_audit,
        "candidate_checkpoint": str(candidate_path.resolve()),
        "null_checkpoint": str(null_path.resolve()),
        "pretraining_audit": str(audit_path.resolve()),
    }


def fit_point_model_cuda(
    model: IndependentRNetDistillStudent,
    cells: list[dict[str, Any]],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    *,
    epochs: int,
    seed: int,
    label: str,
) -> list[float]:
    """Run the frozen V14 point objective with CUDA-only optimizer state."""

    if not cells:
        raise ValueError("downstream point training requires outer-train cells")
    _reset_downstream_rng(seed)
    point_parameters = _point_parameters(model)
    distill_before = _state_snapshot(model.distill_head)
    model.train()
    model.distill_head.eval()
    optimizer = torch.optim.Adam(
        point_parameters,
        lr=1e-3,
        weight_decay=0.0,
        capturable=True,
    )
    history: list[float] = []
    for epoch in range(int(epochs)):
        order = _downstream_epoch_order(len(cells), seed=seed, epoch=epoch)
        losses: list[float] = []
        for index in order:
            cell = cells[index]
            context = context_cache[cell["construct_id"]]
            _assert_context_cuda(context, label=label)
            for key in (
                "edit",
                "distance",
                "target",
                "prediction_mask",
                "qualified_mask",
                "wt",
                "feature41_point",
            ):
                _assert_tensor_cuda(cell[key], label=f"{label} cell {key}")
            hidden = model.encode(context)
            _assert_tensor_cuda(hidden, label=f"{label} hidden")
            point = model.forward_point(
                hidden,
                cell["edit"],
                cell["distance"],
                cell["refs"],
                cell["alts"],
                cell["prediction_mask"],
                cell["feature41_point"],
            )
            _assert_tensor_cuda(point, label=f"{label} point")
            loss = method_cell_balanced_l1(
                point,
                cell["target"],
                cell["qualified_mask"],
                cell["wt"],
            )
            _assert_tensor_cuda(loss, label=f"{label} loss")
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite {label} downstream loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _assert_finite_cuda_gradients(model, label=label)
            if any(parameter.grad is not None for parameter in model.distill_head.parameters()):
                raise RuntimeError(f"{label} downstream training reached distill_head")
            torch.nn.utils.clip_grad_norm_(point_parameters, 5.0)
            optimizer.step()
            _assert_optimizer_cuda(optimizer, label=label)
            losses.append(float(loss.detach().item()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    if len(history) != int(epochs) or not bool(np.isfinite(history).all()):
        raise RuntimeError(f"{label} downstream history is incomplete or nonfinite")
    _assert_state_equal(
        distill_before,
        _state_snapshot(model.distill_head),
        label=f"{label} frozen distill_head",
    )
    _assert_module_cuda(model, label=f"{label} trained point model")
    return history


def fit_calibration_head_cuda(
    head: torch.nn.Module,
    cells: list[dict[str, Any]],
    inputs: list[np.ndarray],
    point_field: str,
    device: str,
    epochs: int,
    seed: int,
    *,
    label: str,
) -> list[float]:
    """Use the V10 objective while forcing all training state onto CUDA."""

    _assert_module_cuda(head, label=label)
    torch.manual_seed(int(seed) + 2_810_000)
    torch.cuda.manual_seed_all(int(seed) + 2_810_000)
    head.train()
    optimizer = torch.optim.Adam(
        head.parameters(), lr=1e-3, weight_decay=0.0, capturable=True
    )
    history: list[float] = []
    for epoch in range(int(epochs)):
        order = list(range(len(cells)))
        random.Random(int(seed) * 100_003 + epoch).shuffle(order)
        losses: list[float] = []
        for index in order:
            cell = cells[index]
            x = torch.tensor(inputs[index], device=device)
            point = torch.tensor(cell[point_field], device=device)
            target = torch.tensor(cell["target_delta"], device=device)
            mutant_index = torch.tensor(cell["mutant_index"], device=device)
            for value_name, value in (
                ("input", x),
                ("point", point),
                ("target", target),
                ("mutant_index", mutant_index),
            ):
                _assert_tensor_cuda(value, label=f"{label} {value_name}")
            weights, locations, scales = head(point, x)
            for value_name, value in (
                ("weights", weights),
                ("locations", locations),
                ("scales", scales),
            ):
                _assert_tensor_cuda(value, label=f"{label} {value_name}")
            loss = _mutant_balanced_crps(
                weights,
                locations,
                scales,
                target,
                mutant_index,
                int(cell["n_mutants"]),
            )
            _assert_tensor_cuda(loss, label=f"{label} loss")
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite {label} calibration loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _assert_finite_cuda_gradients(head, label=label)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            _assert_optimizer_cuda(optimizer, label=label)
            losses.append(float(loss.detach().item()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    if len(history) != int(epochs) or not bool(np.isfinite(history).all()):
        raise RuntimeError(f"{label} calibration history is incomplete or nonfinite")
    _assert_module_cuda(head, label=f"{label} calibrated head")
    return history


def _rename_v11_prediction(
    prediction: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, value in prediction.items():
        if name == "schema_version":
            output[name] = np.asarray(PREDICTION_SCHEMA)
        elif name.startswith("anchored_"):
            output[f"candidate_{name.removeprefix('anchored_')}"] = value
        elif name.startswith("unanchored_"):
            output[f"null_{name.removeprefix('unanchored_')}"] = value
        else:
            output[name] = value
    names = frozenset(output)
    if names != EXPECTED_PREDICTION_FIELDS:
        missing = sorted(EXPECTED_PREDICTION_FIELDS - names)
        unexpected = sorted(names - EXPECTED_PREDICTION_FIELDS)
        raise RuntimeError(
            f"independent RNet prediction schema differs: missing={missing} unexpected={unexpected}"
        )
    if names & FORBIDDEN_PREDICTION_FIELDS:
        raise RuntimeError("independent RNet prediction contains a forbidden outcome field")
    return output


def _artifact_paths(out_dir: Path, fold: int, seed: int) -> dict[str, Path]:
    stem = f"fold{fold}_seed{seed}"
    return {
        "result": out_dir / f"rnet_distill_fold_result_{stem}.json",
        "prediction": out_dir / f"rnet_distill_predictions_{stem}.npz",
        "candidate_point": out_dir / f"rnet_distill_candidate_point_{stem}.pt",
        "null_point": out_dir / f"rnet_distill_null_point_{stem}.pt",
        "feature41_residual": out_dir / f"rnet_distill_feature41_asymmetric_{stem}.pt",
        "candidate_residual": out_dir / f"rnet_distill_candidate_asymmetric_{stem}.pt",
        "null_residual": out_dir / f"rnet_distill_null_asymmetric_{stem}.pt",
    }


def _refuse_fold_overwrite(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite downstream fold artifacts: {existing}")


def _canonicalize_fold_result_paths(
    result: dict[str, Any], canonical_paths: dict[str, Path]
) -> dict[str, Any]:
    """Replace staging paths in a completed row with its canonical destinations."""

    output = copy.deepcopy(result)
    output["point_checkpoints"] = {
        "candidate": str(canonical_paths["candidate_point"].resolve()),
        "null": str(canonical_paths["null_point"].resolve()),
    }
    output["residual_checkpoints"] = {
        "feature41": str(canonical_paths["feature41_residual"].resolve()),
        "candidate": str(canonical_paths["candidate_residual"].resolve()),
        "null": str(canonical_paths["null_residual"].resolve()),
    }
    output["prediction_artifact"] = str(canonical_paths["prediction"].resolve())
    return output


def _publish_fold_artifacts(
    staging_paths: dict[str, Path], canonical_paths: dict[str, Path]
) -> None:
    """Publish a complete seven-file fold, with the result marker last."""

    if set(staging_paths) != set(canonical_paths):
        raise RuntimeError("staging/canonical fold artifact universes differ")
    missing = [str(path) for path in staging_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"staged fold payload is incomplete: {missing}")
    _refuse_fold_overwrite(canonical_paths)
    publish_order = (
        "candidate_point",
        "null_point",
        "feature41_residual",
        "candidate_residual",
        "null_residual",
        "prediction",
        "result",
    )
    published: list[str] = []
    try:
        for name in publish_order:
            staging_paths[name].replace(canonical_paths[name])
            published.append(name)
    except BaseException:
        # Every published path was absent before this function. Move only those
        # files back into this run's private staging directory before cleanup.
        for name in reversed(published):
            if canonical_paths[name].is_file() and not staging_paths[name].exists():
                canonical_paths[name].replace(staging_paths[name])
        raise


def _save_point_checkpoint(
    path: Path,
    *,
    model: IndependentRNetDistillStudent,
    phase: str,
    arm: str,
    fold: int,
    seed: int,
    pretrain_checkpoint: str,
) -> None:
    torch.save(
        {
            "schema_version": POINT_CHECKPOINT_SCHEMA,
            "phase": phase,
            "arm": arm,
            "outer_fold": int(fold),
            "seed": int(seed),
            "pretrain_checkpoint": pretrain_checkpoint,
            "point_model_state_dict": downstream_point_state_dict(model),
            "distill_head_excluded_from_downstream": True,
        },
        path,
    )


def run_fold(
    *,
    univ: M2Universe,
    records: list[Any],
    fold: Any,
    device: str,
    out_dir: Path,
    pretrain_dir: Path,
    v8_dir: Path,
    v10_dir: Path,
    tic2a_merged: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    point_epochs: int,
    calibration_epochs: int,
    seed: int,
    phase: str,
    experiment_id: str,
    repo_root: Path,
    git_commit: str,
) -> dict[str, Any]:
    started_at = _utc_now()
    fold_id = int(fold.outer_fold)
    paths = _artifact_paths(out_dir, fold_id, seed)
    _refuse_fold_overwrite(paths)
    v8_row, tic_row, v10_row, feature41_model = _fold_sources(
        fold_id,
        v8_dir=v8_dir,
        v10_dir=v10_dir,
        tic2a_merged=tic2a_merged,
    )
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    construct_ids = sorted(
        {record.construct_id for record in train_records + held_records}
    )
    context_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in construct_ids
    }
    for construct_id, context in context_cache.items():
        _assert_context_cuda(context, label=f"construct {construct_id}")
    replay = _feature41_replay_max_difference(
        univ,
        held_records,
        feature41_model,
        unconstrained,
        constrained,
        Path(tic_row["prediction_artifact"]),
        fold_id,
    )
    if replay > 1e-7:
        raise RuntimeError("independent RNet feature41 replay exceeds 1e-7")
    cells = _point_cells(
        univ,
        train_records,
        feature41_model,
        unconstrained,
        constrained,
        device,
    )

    candidate, null, pretrain_audit = load_pretrained_pair(
        pretrain_dir=pretrain_dir, device=device
    )
    candidate_residual_before = _state_snapshot(candidate.residual_head)
    null_residual_before = _state_snapshot(null.residual_head)
    _assert_state_equal(
        candidate_residual_before,
        null_residual_before,
        label="candidate/null residual heads before downstream step one",
    )
    candidate_history = fit_point_model_cuda(
        candidate,
        cells,
        context_cache,
        epochs=point_epochs,
        seed=seed,
        label="candidate",
    )
    print(
        f"[{phase}] fold={fold_id} seed={seed} candidate_point_complete",
        flush=True,
    )
    null_history = fit_point_model_cuda(
        null,
        cells,
        context_cache,
        epochs=point_epochs,
        seed=seed,
        label="null",
    )
    print(f"[{phase}] fold={fold_id} seed={seed} null_point_complete", flush=True)
    candidate_point_count = sum(
        parameter.numel()
        for name, parameter in candidate.named_parameters()
        if not name.startswith("distill_head.")
    )
    null_point_count = sum(
        parameter.numel()
        for name, parameter in null.named_parameters()
        if not name.startswith("distill_head.")
    )
    if candidate_point_count != null_point_count or candidate_point_count < 1:
        raise RuntimeError("candidate/null downstream parameter counts differ")
    _save_point_checkpoint(
        paths["candidate_point"],
        model=candidate,
        phase=phase,
        arm="candidate",
        fold=fold_id,
        seed=seed,
        pretrain_checkpoint=pretrain_audit["candidate_checkpoint"],
    )
    _save_point_checkpoint(
        paths["null_point"],
        model=null,
        phase=phase,
        arm="null",
        fold=fold_id,
        seed=seed,
        pretrain_checkpoint=pretrain_audit["null_checkpoint"],
    )

    freeze_point_model(candidate)
    freeze_point_model(null)
    candidate_snapshot = _state_snapshot(candidate)
    null_snapshot = _state_snapshot(null)
    v8_model = _load_v8_mean(Path(v8_row["meanaligned_checkpoint"]), device)
    freeze_mean_model(v8_model)
    _assert_module_cuda(v8_model, label="frozen V8 comparator")
    calibration_cells = _calibration_cells(
        cells,
        anchored=candidate,
        unanchored=null,
        v8_model=v8_model,
        v11_context_cache=context_cache,
        v8_context_cache=context_cache,
    )
    heads = _new_residual_heads(seed, device)
    _assert_state_equal(
        _state_snapshot(heads["anchored"]),
        _state_snapshot(heads["unanchored"]),
        label="candidate/null calibration-head initialization",
    )
    standardizers: dict[str, Any] = {}
    calibration_inputs: dict[str, list[np.ndarray]] = {}
    histories: dict[str, list[float]] = {}
    for name in V11_POINT_NAMES:
        standardizers[name], calibration_inputs[name] = _prepare_calibration_inputs(
            calibration_cells, name
        )
        if name == "feature41" and phase == "RND3" and seed == 0:
            head, standardizer, history = _load_authoritative_v10_feature41(
                v10_row, device
            )
            if not np.array_equal(
                standardizers[name].mean, standardizer.mean
            ) or not np.array_equal(standardizers[name].scale, standardizer.scale):
                raise RuntimeError(
                    "independent RNet feature41 standardizer does not replay V10"
                )
            heads[name] = head
            standardizers[name] = standardizer
            histories[name] = history
            _assert_module_cuda(head, label="authoritative feature41 head")
        else:
            histories[name] = fit_calibration_head_cuda(
                heads[name],
                calibration_cells,
                calibration_inputs[name],
                f"{name}_point",
                device,
                calibration_epochs,
                seed,
                label=f"{name} calibration",
            )
    _assert_unchanged(candidate_snapshot, candidate, "independent RNet candidate point")
    _assert_unchanged(null_snapshot, null, "independent RNet null point")
    if any(parameter.grad is not None for parameter in candidate.parameters()):
        raise RuntimeError("calibration produced candidate point gradients")
    if any(parameter.grad is not None for parameter in null.parameters()):
        raise RuntimeError("calibration produced null point gradients")

    residual_checkpoints: dict[str, str] = {}
    residual_path_keys = {
        "feature41": "feature41_residual",
        "anchored": "candidate_residual",
        "unanchored": "null_residual",
    }
    for name, head in heads.items():
        mapped = {"anchored": "candidate", "unanchored": "null"}.get(name, name)
        path = paths[residual_path_keys[name]]
        torch.save(
            {
                "state_dict": head.state_dict(),
                "standardizer_mean": standardizers[name].mean,
                "standardizer_scale": standardizers[name].scale,
                "point_name": mapped,
            },
            path,
        )
        residual_checkpoints[mapped] = str(path.resolve())

    prediction = _rename_v11_prediction(
        _held_prediction(
            univ=univ,
            held_records=held_records,
            feature41_model=feature41_model,
            anchored=candidate,
            unanchored=null,
            v8_model=v8_model,
            heads=heads,
            standardizers=standardizers,
            v11_context_cache=context_cache,
            v8_context_cache=context_cache,
            unconstrained=unconstrained,
            constrained=constrained,
            fold_id=fold_id,
            seed=seed,
            v8_prediction_path=Path(v8_row["expert_prediction_artifact"]),
            tic2a_prediction_path=Path(tic_row["prediction_artifact"]),
            historical_v10_path=Path(v10_row["prediction_artifact"]),
            require_v10_feature41_replay=(phase == "RND3" and seed == 0),
        )
    )
    np.savez_compressed(paths["prediction"], **prediction)
    print(f"[{phase}] fold={fold_id} seed={seed} prediction_complete", flush=True)

    mapped_histories = {
        {"anchored": "candidate", "unanchored": "null"}.get(name, name): values
        for name, values in histories.items()
    }
    return {
        "schema_version": FOLD_SCHEMA,
        "experiment_id": experiment_id,
        "phase": phase,
        "evidence_status": EVIDENCE_STATUS[phase],
        "metric_eligibility": EVIDENCE_STATUS[phase],
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "git_commit": git_commit,
        "command": list(sys.argv),
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "seed": seed,
        "point_epochs": point_epochs,
        "calibration_epochs": calibration_epochs,
        "training_device": device,
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "contract_paths": {
            "active": str((repo_root / "configs/reactflow_delta/active_contract.yaml").resolve()),
            "machine": str((repo_root / CONTRACT_PATH).resolve()),
            "ledger": str((repo_root / LEDGER_PATH).resolve()),
        },
        "split": {
            "name": "split_v4_lopo_puzzle",
            "seed": 20260813,
            "fold_universe": list(range(20)),
        },
        "pretraining_checkpoints": {
            "candidate": pretrain_audit["candidate_checkpoint"],
            "null": pretrain_audit["null_checkpoint"],
            "audit": pretrain_audit["pretraining_audit"],
        },
        "point_checkpoints": {
            "candidate": str(paths["candidate_point"].resolve()),
            "null": str(paths["null_point"].resolve()),
        },
        "residual_checkpoints": residual_checkpoints,
        "prediction_artifact": str(paths["prediction"].resolve()),
        "training_histories": {
            "candidate_point": candidate_history,
            "null_point": null_history,
            **{f"{name}_residual": values for name, values in mapped_histories.items()},
        },
        "n_train_cells": len(cells),
        "n_registered_prediction_rows": int(len(prediction["keys"])),
        "feature41_replay_max_abs_difference": replay,
        "point_parameter_counts": {
            "candidate": candidate_point_count,
            "null": null_point_count,
        },
        "residual_parameter_counts": {
            {"anchored": "candidate", "unanchored": "null"}.get(name, name): residual_count(head)
            for name, head in heads.items()
        },
        "invariants": {
            "target_profile_identity_exact": True,
            "pretrained_source_pair_bound": True,
            "residual_heads_identical_before_downstream_step_one": True,
            "pretrained_encoders_different_before_downstream": True,
            "same_downstream_training_order_and_dropout_stream": True,
            "distillation_head_excluded_and_frozen_downstream": True,
            "point_frozen_during_calibration": True,
            "v10_residual_family_reused": True,
            "feature41_replay_at_1e_7": True,
            "authoritative_feature41_seed0_comparator_or_smoke_not_applicable": True,
            "median_constraint_all_held_rows": True,
            "cuda_only_training": True,
            "held_target_read": False,
            "held_score_computed": False,
            "partial_score_inspected": False,
            "prediction_contains_target_fields": False,
            "external_outcome_accessed": False,
        },
        "exit_code": 0,
    }


def _phase_schedule(phase: str) -> tuple[tuple[int, ...], int, int]:
    if phase not in EXPECTED_FOLDS:
        raise ValueError(f"unsupported downstream phase: {phase}")
    point_epochs, calibration_epochs = EXPECTED_SCHEDULE[phase]
    return EXPECTED_FOLDS[phase], point_epochs, calibration_epochs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=tuple(EXPECTED_FOLDS), required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--pretrain-dir", type=Path, required=True)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--v10-dir", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--point-epochs", type=int, required=True)
    parser.add_argument("--calibration-epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    assert_run_authority(repo_root, args.phase)
    validate_downstream_cli_binding(repo_root, args)
    expected_folds, expected_point_epochs, expected_calibration_epochs = (
        _phase_schedule(args.phase)
    )
    folds = _parse_folds(args.folds)
    if not set(folds) <= set(expected_folds):
        raise ValueError(f"{args.phase} requested folds outside the frozen universe")
    if args.seed != EXPECTED_SEED or (
        args.point_epochs,
        args.calibration_epochs,
    ) != (expected_point_epochs, expected_calibration_epochs):
        raise ValueError(f"{args.phase} seed or epoch schedule changed")

    # CUDA validation deliberately occurs before creating any fold artifact.
    device = require_cuda_device(args.device)
    if torch.device(device).type != "cuda":
        raise RuntimeError("CUDA_REQUIRED: downstream device resolved off CUDA")
    probe_loss = torch.ones((), device=device)
    _assert_tensor_cuda(probe_loss, label="downstream preflight loss probe")
    out_dir = _require_mnt_artifact_dir(args.out_dir)
    pretrain_dir = _require_mnt_artifact_dir(args.pretrain_dir)
    for fold_id in folds:
        _refuse_fold_overwrite(_artifact_paths(out_dir, fold_id, args.seed))

    univ = M2Universe(args.m2_csv)
    identity = univ.build()
    if identity.get("n_canonical_mutant_full_profiles") != 13976 or identity.get(
        "canonical_mutant_full_profile_identity"
    ) != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("independent RNet requires exact canonical target identity")
    records = univ.get_records()
    split = build_split_v4(
        sorted({record.puzzle for record in records}), seed=20260813
    )
    selected = [
        fold for fold in split["folds"] if int(fold.outer_fold) in set(folds)
    ]
    if len(selected) != len(folds):
        raise ValueError("one or more requested independent RNet folds are absent")

    tic2a_merged = _read_json(args.tic2a_merged_json)
    unconstrained = EnsembleFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    validate_cache_alignment(unconstrained, constrained)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit(repo_root)
    try:
        for fold in selected:
            fold_id = int(fold.outer_fold)
            print(
                f"[{args.phase}] fold={fold_id} held={fold.held_puzzle} seed={args.seed} start",
                flush=True,
            )
            canonical_paths = _artifact_paths(out_dir, fold_id, args.seed)
            with tempfile.TemporaryDirectory(
                prefix=f".rnet_distill_fold{fold_id}_seed{args.seed}_",
                dir=out_dir,
            ) as staging_name:
                staging_dir = Path(staging_name)
                result = run_fold(
                    univ=univ,
                    records=records,
                    fold=fold,
                    device=device,
                    out_dir=staging_dir,
                    pretrain_dir=pretrain_dir,
                    v8_dir=args.v8_dir,
                    v10_dir=args.v10_dir,
                    tic2a_merged=tic2a_merged,
                    unconstrained=unconstrained,
                    constrained=constrained,
                    point_epochs=args.point_epochs,
                    calibration_epochs=args.calibration_epochs,
                    seed=args.seed,
                    phase=args.phase,
                    experiment_id=args.experiment_id,
                    repo_root=repo_root,
                    git_commit=commit,
                )
                staging_paths = _artifact_paths(staging_dir, fold_id, args.seed)
                result = _canonicalize_fold_result_paths(result, canonical_paths)
                staging_paths["result"].write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                _publish_fold_artifacts(staging_paths, canonical_paths)
            print(f"[{args.phase}] fold={fold_id} complete", flush=True)
            torch.cuda.empty_cache()
    finally:
        unconstrained.close()
        constrained.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
