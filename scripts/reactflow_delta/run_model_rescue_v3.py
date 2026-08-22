#!/usr/bin/env python3
"""Run the frozen Model Rescue v3 disagreement-gated procedure."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import AlignedDeltaModel, aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v2 import (
    ConditionalScaleMixtureCalibrator,
    MeanAlignedModel,
    cell_balanced_crps,
)
from scripts.reactflow_delta.model_rescue_v3 import (
    CANDIDATE,
    INNER_PREDICTION_SCHEMA,
    PREDICTION_SCHEMA,
    DisagreementGate,
    apply_disagreement_gate_torch,
    build_inner_crossfit_ledger,
    fit_disagreement_gate,
    hierarchy_position_weights,
)
from scripts.reactflow_delta.run_model_rescue_m2_v1 import (
    fit_candidate,
    predict_held,
    score_predictions,
)
from scripts.reactflow_delta.run_model_rescue_v2 import (
    BASELINE,
    _make_cells,
    _require_finite_gradients,
    fit_mean,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.run_p3_lrso_v3 import _wt_filled
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v3_run.v1"


def assert_run_authority(repo_root: Path, phase: str, *, smoke: bool) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != phase:
        raise RuntimeError(
            f"v3 runner phase {phase} is closed; active phase is "
            f"{active['authority']['current_phase']}"
        )
    if active.get("runnable_phases") != [phase]:
        raise RuntimeError("v3 runner requires a single matching runnable phase")
    if phase == "R3M2":
        if not smoke or active.get("training_allowed") != "ENGINEERING_SMOKE_ONLY":
            raise RuntimeError("R3M2 authorizes engineering smoke only")
    else:
        if smoke or active.get("training_allowed") is not True:
            raise RuntimeError(f"{phase} requires full active training authority")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("v3 runner requires external outcomes to remain locked")


def validate_outer_expert_reuse(
    *,
    seed: int,
    smoke: bool,
    b1_result_dir: Path | None,
    mean_result_dir: Path | None,
) -> None:
    if (b1_result_dir is None) != (mean_result_dir is None):
        raise ValueError("outer expert reuse requires both B1 and MeanAligned result dirs")
    if seed != 0 and b1_result_dir is not None:
        raise ValueError("existing v1/v2 outer expert reuse is authorized only for seed 0")
    if smoke and b1_result_dir is not None:
        raise ValueError("R3M2 smoke must train its outer experts for at most three epochs")


def _module_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _assert_module_unchanged(
    snapshot: dict[str, torch.Tensor], module: torch.nn.Module, label: str
) -> None:
    current = module.state_dict()
    for name, expected in snapshot.items():
        if not torch.equal(expected, current[name].detach().cpu()):
            raise RuntimeError(f"residual calibration changed frozen {label} parameter {name}")


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _b1_delta(
    model: AlignedDeltaModel,
    H: torch.Tensor,
    edit: torch.Tensor,
    distance: torch.Tensor,
    refs: list[str],
    alts: list[str],
    prediction_mask: torch.Tensor,
) -> torch.Tensor:
    weights, locations, _scales = model.forward_distribution(
        H, edit, distance, refs, alts, prediction_mask
    )
    return (weights * locations).sum(-1) / weights.sum(-1).clamp(min=1e-12)


def predict_expert_means(
    b1_model: AlignedDeltaModel,
    mean_model: MeanAlignedModel,
    univ: M2Universe,
    records: list[Any],
    ctx_cache: dict[str, Any],
    device: str,
) -> dict[str, np.ndarray]:
    """Target-blind registered predictions for inner gate fitting."""
    by_construct: dict[str, list[Any]] = {}
    for record in records:
        by_construct.setdefault(record.construct_id, []).append(record)
    keys: list[str] = []
    b1_values: list[float] = []
    mean_values: list[float] = []
    b1_model.eval()
    mean_model.eval()
    with torch.no_grad():
        for construct_id, recs in sorted(by_construct.items()):
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            edit = torch.tensor([record.full_pos for record in recs], device=device)
            distance = (torch.arange(length, device=device)[None, :] - edit[:, None]).float()
            prediction_mask = torch.tensor(
                np.tile(construct.wt_observed.astype(bool), (len(recs), 1)),
                device=device,
            )
            b1_H = b1_model.encode(ctx_cache[construct_id])
            mean_H = mean_model.encode(ctx_cache[construct_id])
            b1_delta = _b1_delta(
                b1_model,
                b1_H,
                edit,
                distance,
                [record.ref for record in recs],
                [record.alt for record in recs],
                prediction_mask,
            )
            mean_delta = mean_model.forward_mean(
                mean_H,
                edit,
                distance,
                [record.ref for record in recs],
                [record.alt for record in recs],
                prediction_mask,
            )
            for row, record in enumerate(recs):
                for position in range(length):
                    keys.append(_bio_key(univ, record, position))
                    b1_values.append(float(b1_delta[row, position].cpu()))
                    mean_values.append(float(mean_delta[row, position].cpu()))
    if len(keys) != len(set(keys)):
        raise RuntimeError("inner expert prediction contains duplicate biological keys")
    return {
        "schema_version": np.asarray(INNER_PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "b1_delta_mean": np.asarray(b1_values, dtype=np.float64),
        "meanaligned_delta_mean": np.asarray(mean_values, dtype=np.float64),
    }


def _gate_training_rows(
    univ: M2Universe,
    records: list[Any],
    prediction_artifacts: list[Path],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prediction: dict[str, tuple[float, float]] = {}
    for artifact in prediction_artifacts:
        with np.load(artifact, allow_pickle=True) as handle:
            if str(handle["schema_version"]) != INNER_PREDICTION_SCHEMA:
                raise ValueError(f"wrong inner prediction schema in {artifact}")
            for key, b1, mean in zip(
                handle["keys"],
                handle["b1_delta_mean"],
                handle["meanaligned_delta_mean"],
            ):
                key = str(key)
                if key in prediction:
                    raise ValueError(f"duplicate inner prediction key {key}")
                prediction[key] = (float(b1), float(mean))

    rows: dict[str, list[Any]] = {
        "target": [],
        "b1": [],
        "mean": [],
        "puzzle": [],
        "method": [],
        "mutant": [],
    }
    expected_puzzles = sorted({record.puzzle for record in records})
    observed_puzzles: set[str] = set()
    for record in records:
        construct = univ.get_construct(record.construct_id)
        target, _error = univ.mutant_full_profile(
            record.wt_id, record.design_pos, record.ref, record.alt
        )
        if target is None:
            continue
        qualified = construct.wt_observed & np.isfinite(target)
        positions = np.flatnonzero(qualified)
        if len(positions) == 0:
            continue
        biological_keys = [
            _bio_key(univ, record, int(position)) for position in positions
        ]
        missing = [key for key in biological_keys if key not in prediction]
        if missing:
            raise ValueError(
                f"inner OOF prediction missing {len(missing)} qualified keys for {record.puzzle}"
            )
        values = [prediction[key] for key in biological_keys]
        mutant_id = (
            f"{record.construct_id}|{record.design_pos}|{record.ref}>{record.alt}"
        )
        rows["target"].extend(
            (target[positions] - construct.wt_reactivity[positions]).tolist()
        )
        rows["b1"].extend(value[0] for value in values)
        rows["mean"].extend(value[1] for value in values)
        rows["puzzle"].extend([record.puzzle] * len(positions))
        rows["method"].extend([record.method] * len(positions))
        rows["mutant"].extend([mutant_id] * len(positions))
        observed_puzzles.add(record.puzzle)
    if sorted(observed_puzzles) != expected_puzzles:
        raise ValueError("inner OOF gate rows do not cover every outer-train puzzle")
    arrays = {
        name: np.asarray(values, dtype=float if name in {"target", "b1", "mean"} else object)
        for name, values in rows.items()
    }
    arrays["weight"] = hierarchy_position_weights(
        arrays["puzzle"], arrays["method"], arrays["mutant"]
    )
    return arrays, {
        "n_prediction_keys": len(prediction),
        "n_qualified_gate_rows": len(arrays["target"]),
        "covered_puzzles": expected_puzzles,
        "hierarchy_weight_sum": float(arrays["weight"].sum()),
    }


def run_inner_crossfit(
    *,
    univ: M2Universe,
    fold: Any,
    outer_train_records: list[Any],
    ctx_cache: dict[str, Any],
    device: str,
    out_dir: Path,
    expert_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[DisagreementGate, Path]:
    ledger_rows = build_inner_crossfit_ledger(fold.train_puzzles, fold.inner_groups)
    prediction_paths: list[Path] = []
    execution_rows = []
    for row in ledger_rows:
        inner_fold = int(row["inner_fold"])
        train_set = set(row["train_puzzles"])
        held_set = set(row["held_puzzles"])
        train_records = [r for r in outer_train_records if r.puzzle in train_set]
        held_records = [r for r in outer_train_records if r.puzzle in held_set]
        if set(r.puzzle for r in train_records) & set(r.puzzle for r in held_records):
            raise RuntimeError("inner held puzzle leaked into expert training records")
        inner_seed = seed * 100_000 + int(fold.outer_fold) * 100 + inner_fold
        torch.manual_seed(inner_seed)
        b1_model = AlignedDeltaModel(k_rank=0, sparse=False).to(device)
        b1_history = fit_candidate(
            b1_model,
            univ,
            train_records,
            ctx_cache,
            device,
            expert_epochs,
            learning_rate,
            weight_decay,
            0.0,
        )
        torch.manual_seed(inner_seed)
        mean_model = MeanAlignedModel().to(device)
        mean_history = fit_mean(
            mean_model,
            _make_cells(univ, train_records, device),
            ctx_cache,
            expert_epochs,
            learning_rate,
            weight_decay,
            inner_seed,
        )
        b1_checkpoint = out_dir / (
            f"v3_inner_b1_outer{fold.outer_fold}_inner{inner_fold}_seed{seed}.pt"
        )
        mean_checkpoint = out_dir / (
            f"v3_inner_mean_outer{fold.outer_fold}_inner{inner_fold}_seed{seed}.pt"
        )
        torch.save(b1_model.state_dict(), b1_checkpoint)
        torch.save(mean_model.state_dict(), mean_checkpoint)
        inner_prediction = predict_expert_means(
            b1_model, mean_model, univ, held_records, ctx_cache, device
        )
        inner_prediction.update(
            {
                "outer_fold": np.full(
                    len(inner_prediction["keys"]), int(fold.outer_fold), dtype=np.int64
                ),
                "inner_fold": np.full(
                    len(inner_prediction["keys"]), inner_fold, dtype=np.int64
                ),
                "seed": np.full(len(inner_prediction["keys"]), seed, dtype=np.int64),
            }
        )
        prediction_path = out_dir / (
            f"v3_inner_predictions_outer{fold.outer_fold}_inner{inner_fold}_seed{seed}.npz"
        )
        np.savez_compressed(prediction_path, **inner_prediction)
        prediction_paths.append(prediction_path)
        execution_rows.append(
            {
                **row,
                "inner_seed": inner_seed,
                "b1_checkpoint": str(b1_checkpoint),
                "meanaligned_checkpoint": str(mean_checkpoint),
                "prediction_artifact": str(prediction_path),
                "n_registered_prediction_keys": len(inner_prediction["keys"]),
                "b1_train_loss": b1_history,
                "meanaligned_train_loss": mean_history,
            }
        )
        del b1_model, mean_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    arrays, coverage = _gate_training_rows(
        univ, outer_train_records, prediction_paths
    )
    gate = fit_disagreement_gate(
        arrays["target"], arrays["b1"], arrays["mean"], arrays["weight"]
    )
    ledger = {
        "schema_version": "reactflow_delta.model_rescue_v3_inner_crossfit_ledger.v1",
        "outer_fold": int(fold.outer_fold),
        "outer_held_puzzle": fold.held_puzzle,
        "seed": seed,
        "inner_folds": execution_rows,
        "coverage": coverage,
        "gate": gate.to_dict(),
        "target_values_stored": False,
        "method_used_as_gate_input": False,
    }
    ledger_path = out_dir / f"v3_inner_ledger_outer{fold.outer_fold}_seed{seed}.json"
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return gate, ledger_path


def fit_blended_calibrator(
    calibrator: ConditionalScaleMixtureCalibrator,
    b1_model: AlignedDeltaModel,
    mean_model: MeanAlignedModel,
    gate: DisagreementGate,
    cells: list[dict[str, Any]],
    ctx_cache: dict[str, Any],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> list[float]:
    _freeze(b1_model)
    _freeze(mean_model)
    b1_snapshot = _module_snapshot(b1_model)
    mean_snapshot = _module_snapshot(mean_model)
    calibrator.train()
    optimizer = torch.optim.Adam(
        calibrator.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    for epoch in range(epochs):
        rng = np.random.RandomState(seed * 100_003 + 20_000 + epoch)
        order = rng.permutation(len(cells)).tolist()
        losses = []
        for index in order:
            cell = cells[index]
            with torch.no_grad():
                b1_H = b1_model.encode(ctx_cache[cell["construct_id"]])
                mean_H = mean_model.encode(ctx_cache[cell["construct_id"]])
                b1_delta = _b1_delta(
                    b1_model,
                    b1_H,
                    cell["edit"],
                    cell["distance"],
                    cell["refs"],
                    cell["alts"],
                    cell["prediction_mask"],
                )
                mean_delta, features = mean_model.forward_mean_and_features(
                    mean_H,
                    cell["edit"],
                    cell["distance"],
                    cell["refs"],
                    cell["alts"],
                    cell["prediction_mask"],
                )
                blend, _alpha, _disagreement = apply_disagreement_gate_torch(
                    b1_delta, mean_delta, gate
                )
            weights, locations, scales = calibrator(blend.detach(), features.detach())
            loss = cell_balanced_crps(
                weights,
                locations,
                scales,
                cell["target"],
                cell["qualified_mask"],
                cell["wt"],
            )
            optimizer.zero_grad()
            loss.backward()
            _require_finite_gradients(calibrator, "v3 residual calibration")
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    _assert_module_unchanged(b1_snapshot, b1_model, "B1 expert")
    _assert_module_unchanged(mean_snapshot, mean_model, "MeanAligned expert")
    if any(parameter.grad is not None for parameter in b1_model.parameters()):
        raise RuntimeError("calibration produced gradient on B1 expert")
    if any(parameter.grad is not None for parameter in mean_model.parameters()):
        raise RuntimeError("calibration produced gradient on MeanAligned expert")
    return history


def _prediction_arrays(
    *,
    keys: list[str],
    b1_delta: list[float],
    mean_delta: list[float],
    disagreement: list[float],
    alpha: list[float],
    blend_delta: list[float],
    point_mean: list[float],
    locations: list[np.ndarray],
    scales: list[np.ndarray],
    weights: list[np.ndarray],
    gate: DisagreementGate,
    outer_fold: int,
    seed: int,
    b1_checkpoint: Path,
    mean_checkpoint: Path,
    calibration_checkpoint: Path,
    inner_ledger: Path,
) -> dict[str, np.ndarray]:
    n = len(keys)
    result = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "candidate_id": np.full(n, CANDIDATE, dtype=object),
        "outer_fold": np.full(n, outer_fold, dtype=np.int64),
        "seed": np.full(n, seed, dtype=np.int64),
        "b1_delta_mean": np.asarray(b1_delta, dtype=np.float64),
        "meanaligned_delta_mean": np.asarray(mean_delta, dtype=np.float64),
        "expert_disagreement": np.asarray(disagreement, dtype=np.float64),
        "gate_threshold": np.full(n, gate.threshold, dtype=np.float64),
        "gate_alpha_low": np.full(n, gate.alpha_low, dtype=np.float64),
        "gate_alpha_high": np.full(n, gate.alpha_high, dtype=np.float64),
        "gate_alpha_applied": np.asarray(alpha, dtype=np.float64),
        "delta_mean": np.asarray(blend_delta, dtype=np.float64),
        "point_mean": np.asarray(point_mean, dtype=np.float64),
        "locations": np.asarray(locations, dtype=np.float64),
        "scales": np.asarray(scales, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "registered_status": np.full(n, "covered", dtype=object),
        "b1_checkpoint_path": np.full(n, str(b1_checkpoint), dtype=object),
        "meanaligned_checkpoint_path": np.full(n, str(mean_checkpoint), dtype=object),
        "calibration_checkpoint_path": np.full(
            n, str(calibration_checkpoint), dtype=object
        ),
        "inner_crossfit_ledger_path": np.full(n, str(inner_ledger), dtype=object),
    }
    if len(set(result["keys"].tolist())) != n:
        raise RuntimeError("candidate prediction contains duplicate keys")
    if not np.allclose(result["weights"].sum(1), 1.0, atol=1e-7, rtol=0.0):
        raise RuntimeError("candidate residual weights do not sum to one")
    if not np.all(np.isfinite(result["scales"])) or not np.all(result["scales"] > 0):
        raise RuntimeError("candidate residual scales are invalid")
    mixture_mean = np.sum(result["weights"] * result["locations"], axis=1)
    mixture_mean /= result["weights"].sum(axis=1)
    maximum = float(np.max(np.abs(mixture_mean - result["point_mean"])))
    if maximum > 1e-7:
        raise RuntimeError(f"residual distribution changed point mean by {maximum}")
    prohibited = {"target", "target_error", "target_mask", "qualified_mask", "score"}
    if prohibited & set(result):
        raise RuntimeError("prediction-only artifact contains a prohibited target field")
    return result


def predict_candidate(
    *,
    b1_model: AlignedDeltaModel,
    mean_model: MeanAlignedModel,
    calibrator: ConditionalScaleMixtureCalibrator,
    gate: DisagreementGate,
    univ: M2Universe,
    records: list[Any],
    ctx_cache: dict[str, Any],
    device: str,
    outer_fold: int,
    seed: int,
    b1_checkpoint: Path,
    mean_checkpoint: Path,
    calibration_checkpoint: Path,
    inner_ledger: Path,
) -> dict[str, np.ndarray]:
    by_construct: dict[str, list[Any]] = {}
    for record in records:
        by_construct.setdefault(record.construct_id, []).append(record)
    keys: list[str] = []
    b1_values: list[float] = []
    mean_values: list[float] = []
    disagreements: list[float] = []
    alphas: list[float] = []
    blends: list[float] = []
    points: list[float] = []
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    b1_model.eval()
    mean_model.eval()
    calibrator.eval()
    with torch.no_grad():
        for construct_id, recs in sorted(by_construct.items()):
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            edit = torch.tensor([record.full_pos for record in recs], device=device)
            distance = (torch.arange(length, device=device)[None, :] - edit[:, None]).float()
            prediction_mask = torch.tensor(
                np.tile(construct.wt_observed.astype(bool), (len(recs), 1)),
                device=device,
            )
            b1_H = b1_model.encode(ctx_cache[construct_id])
            mean_H = mean_model.encode(ctx_cache[construct_id])
            b1_delta = _b1_delta(
                b1_model,
                b1_H,
                edit,
                distance,
                [record.ref for record in recs],
                [record.alt for record in recs],
                prediction_mask,
            )
            mean_delta, features = mean_model.forward_mean_and_features(
                mean_H,
                edit,
                distance,
                [record.ref for record in recs],
                [record.alt for record in recs],
                prediction_mask,
            )
            blend, alpha, disagreement = apply_disagreement_gate_torch(
                b1_delta, mean_delta, gate
            )
            component_weights, delta_locations, component_scales = calibrator(
                blend, features
            )
            wt = torch.tensor(_wt_filled(univ, construct_id), device=device)
            point = blend + wt[None, :]
            mutant_locations = delta_locations + wt[None, :, None]
            for row, record in enumerate(recs):
                for position in range(length):
                    keys.append(_bio_key(univ, record, position))
                    b1_values.append(float(b1_delta[row, position].cpu()))
                    mean_values.append(float(mean_delta[row, position].cpu()))
                    disagreements.append(float(disagreement[row, position].cpu()))
                    alphas.append(float(alpha[row, position].cpu()))
                    blends.append(float(blend[row, position].cpu()))
                    points.append(float(point[row, position].cpu()))
                    locations.append(mutant_locations[row, position].cpu().numpy())
                    scales.append(component_scales[row, position].cpu().numpy())
                    weights.append(component_weights[row, position].cpu().numpy())
    return _prediction_arrays(
        keys=keys,
        b1_delta=b1_values,
        mean_delta=mean_values,
        disagreement=disagreements,
        alpha=alphas,
        blend_delta=blends,
        point_mean=points,
        locations=locations,
        scales=scales,
        weights=weights,
        gate=gate,
        outer_fold=outer_fold,
        seed=seed,
        b1_checkpoint=b1_checkpoint,
        mean_checkpoint=mean_checkpoint,
        calibration_checkpoint=calibration_checkpoint,
        inner_ledger=inner_ledger,
    )


def assert_target_invariance(expected: dict[str, np.ndarray], **kwargs: Any) -> None:
    perturbed = copy.deepcopy(kwargs["records"])
    for record in perturbed:
        record.target_reactivity = -12345.0
        record.target_error = 98765.0
        record.target_observed = not bool(record.target_observed)
        record.target_mask = np.logical_not(record.target_mask)
    actual = predict_candidate(**{**kwargs, "records": perturbed})
    for field in expected:
        if not np.array_equal(expected[field], actual[field]):
            raise RuntimeError(f"held target/error/mask changed v3 prediction field {field}")


def _load_reused_outer_experts(
    *,
    fold: int,
    seed: int,
    b1_result_dir: Path,
    mean_result_dir: Path,
    device: str,
) -> tuple[AlignedDeltaModel, MeanAlignedModel, Path, Path, dict[str, Any]]:
    b1_result_path = b1_result_dir / f"m2_fold_result_fold{fold}.json"
    b1_row = json.loads(b1_result_path.read_text(encoding="utf-8"))["candidates"][BASELINE]
    mean_result_path = mean_result_dir / f"v2_fold_result_fold{fold}_seed{seed}.json"
    mean_result = json.loads(mean_result_path.read_text(encoding="utf-8"))
    if int(mean_result["seed"]) != seed:
        raise ValueError("reused MeanAligned result has wrong seed")
    mean_row = mean_result["candidates"]["b1_mean_aligned"]
    b1_checkpoint = Path(b1_row["checkpoint"])
    mean_checkpoint = Path(mean_row["mean_checkpoint"])
    b1_model = AlignedDeltaModel(k_rank=0, sparse=False).to(device)
    mean_model = MeanAlignedModel().to(device)
    b1_model.load_state_dict(
        torch.load(b1_checkpoint, map_location=device, weights_only=True)
    )
    mean_model.load_state_dict(
        torch.load(mean_checkpoint, map_location=device, weights_only=True)
    )
    baseline = {
        "model_id": BASELINE,
        "score": b1_row["score"],
        "prediction_artifact": b1_row["prediction_artifact"],
        "checkpoint": str(b1_checkpoint),
        "source_fold_artifact": str(b1_result_path),
        "reused_exact_seed0_outer_expert": True,
    }
    return b1_model, mean_model, b1_checkpoint, mean_checkpoint, baseline


def _train_outer_experts(
    *,
    univ: M2Universe,
    train_records: list[Any],
    held_records: list[Any],
    ctx_cache: dict[str, Any],
    device: str,
    out_dir: Path,
    fold: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[AlignedDeltaModel, MeanAlignedModel, Path, Path, dict[str, Any]]:
    torch.manual_seed(seed)
    b1_model = AlignedDeltaModel(k_rank=0, sparse=False).to(device)
    b1_history = fit_candidate(
        b1_model,
        univ,
        train_records,
        ctx_cache,
        device,
        epochs,
        learning_rate,
        weight_decay,
        0.0,
    )
    torch.manual_seed(seed)
    mean_model = MeanAlignedModel().to(device)
    mean_history = fit_mean(
        mean_model,
        _make_cells(univ, train_records, device),
        ctx_cache,
        epochs,
        learning_rate,
        weight_decay,
        seed,
    )
    b1_checkpoint = out_dir / f"v3_outer_b1_fold{fold}_seed{seed}.pt"
    mean_checkpoint = out_dir / f"v3_outer_mean_fold{fold}_seed{seed}.pt"
    torch.save(b1_model.state_dict(), b1_checkpoint)
    torch.save(mean_model.state_dict(), mean_checkpoint)
    baseline_prediction = predict_held(
        b1_model, univ, held_records, ctx_cache, device
    )
    baseline_path = out_dir / f"v3_baseline_predictions_fold{fold}_seed{seed}.npz"
    np.savez_compressed(baseline_path, **baseline_prediction)
    baseline = {
        "model_id": BASELINE,
        "score": score_predictions(baseline_prediction, univ, held_records),
        "prediction_artifact": str(baseline_path),
        "checkpoint": str(b1_checkpoint),
        "b1_train_loss": b1_history,
        "meanaligned_train_loss": mean_history,
        "reused_exact_seed0_outer_expert": False,
    }
    return b1_model, mean_model, b1_checkpoint, mean_checkpoint, baseline


def run_fold(
    *,
    univ: M2Universe,
    fold: Any,
    records: list[Any],
    device: str,
    out_dir: Path,
    expert_epochs: int,
    residual_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    reuse_b1_result_dir: Path | None,
    reuse_mean_result_dir: Path | None,
) -> dict[str, Any]:
    train_set = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_set]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    construct_ids = sorted({record.construct_id for record in train_records + held_records})
    ctx_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in construct_ids
    }
    gate, inner_ledger = run_inner_crossfit(
        univ=univ,
        fold=fold,
        outer_train_records=train_records,
        ctx_cache=ctx_cache,
        device=device,
        out_dir=out_dir,
        expert_epochs=expert_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    validate_outer_expert_reuse(
        seed=seed,
        smoke=False,
        b1_result_dir=reuse_b1_result_dir,
        mean_result_dir=reuse_mean_result_dir,
    )
    if reuse_b1_result_dir is not None and reuse_mean_result_dir is not None:
        b1_model, mean_model, b1_checkpoint, mean_checkpoint, baseline = (
            _load_reused_outer_experts(
                fold=int(fold.outer_fold),
                seed=seed,
                b1_result_dir=reuse_b1_result_dir,
                mean_result_dir=reuse_mean_result_dir,
                device=device,
            )
        )
    else:
        b1_model, mean_model, b1_checkpoint, mean_checkpoint, baseline = (
            _train_outer_experts(
                univ=univ,
                train_records=train_records,
                held_records=held_records,
                ctx_cache=ctx_cache,
                device=device,
                out_dir=out_dir,
                fold=int(fold.outer_fold),
                seed=seed,
                epochs=expert_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        )
    feature_dim = int(mean_model.bdirect[0].in_features)
    torch.manual_seed(seed + 200_000)
    calibrator = ConditionalScaleMixtureCalibrator(feature_dim=feature_dim).to(device)
    residual_history = fit_blended_calibrator(
        calibrator,
        b1_model,
        mean_model,
        gate,
        _make_cells(univ, train_records, device),
        ctx_cache,
        residual_epochs,
        learning_rate,
        weight_decay,
        seed,
    )
    calibration_checkpoint = out_dir / (
        f"v3_residual_calibration_fold{fold.outer_fold}_seed{seed}.pt"
    )
    torch.save(calibrator.state_dict(), calibration_checkpoint)
    prediction_kwargs = {
        "b1_model": b1_model,
        "mean_model": mean_model,
        "calibrator": calibrator,
        "gate": gate,
        "univ": univ,
        "records": held_records,
        "ctx_cache": ctx_cache,
        "device": device,
        "outer_fold": int(fold.outer_fold),
        "seed": seed,
        "b1_checkpoint": b1_checkpoint,
        "mean_checkpoint": mean_checkpoint,
        "calibration_checkpoint": calibration_checkpoint,
        "inner_ledger": inner_ledger,
    }
    prediction = predict_candidate(**prediction_kwargs)
    assert_target_invariance(prediction, **prediction_kwargs)
    prediction_path = out_dir / (
        f"v3_predictions_{CANDIDATE}_fold{fold.outer_fold}_seed{seed}.npz"
    )
    np.savez_compressed(prediction_path, **prediction)
    result = {
        "schema_version": SCHEMA,
        "outer_fold": int(fold.outer_fold),
        "held_puzzle": fold.held_puzzle,
        "seed": seed,
        "baseline": baseline,
        "candidate": {
            "candidate_id": CANDIDATE,
            "gate": gate.to_dict(),
            "inner_crossfit_ledger": str(inner_ledger),
            "b1_checkpoint": str(b1_checkpoint),
            "meanaligned_checkpoint": str(mean_checkpoint),
            "calibration_checkpoint": str(calibration_checkpoint),
            "prediction_artifact": str(prediction_path),
            "score": score_predictions(prediction, univ, held_records),
            "residual_calibration_loss": residual_history,
        },
        "invariants": {
            "held_target_error_mask_invariance": True,
            "inner_crossfit_complete": True,
            "method_used_as_gate_input": False,
            "residual_changed_point_mean": False,
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=["R3M2", "R3M3", "R3M4"], required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--expert-epochs", type=int, default=40)
    parser.add_argument("--residual-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse-b1-result-dir", type=Path)
    parser.add_argument("--reuse-mean-result-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    assert_run_authority(args.repo_root.resolve(), args.phase, smoke=args.smoke)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    expert_epochs = min(args.expert_epochs, 3) if args.smoke else args.expert_epochs
    residual_epochs = min(args.residual_epochs, 3) if args.smoke else args.residual_epochs
    validate_outer_expert_reuse(
        seed=args.seed,
        smoke=args.smoke,
        b1_result_dir=args.reuse_b1_result_dir,
        mean_result_dir=args.reuse_mean_result_dir,
    )
    universe = M2Universe(args.m2_csv)
    universe.build()
    records = universe.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    selected = {int(value) for value in args.folds.split(",") if value}
    folds = [fold for fold in split["folds"] if fold.outer_fold in selected]
    if not folds:
        raise ValueError("no requested outer folds")
    results = []
    for fold in folds:
        print(
            f"[R3] fold={fold.outer_fold} held={fold.held_puzzle} seed={args.seed} start",
            flush=True,
        )
        result = run_fold(
            univ=universe,
            fold=fold,
            records=records,
            device=device,
            out_dir=args.out_dir,
            expert_epochs=expert_epochs,
            residual_epochs=residual_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            reuse_b1_result_dir=args.reuse_b1_result_dir,
            reuse_mean_result_dir=args.reuse_mean_result_dir,
        )
        results.append(result)
        path = args.out_dir / f"v3_fold_result_fold{fold.outer_fold}_seed{args.seed}.json"
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[R3] fold={fold.outer_fold} artifact={path} complete", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = {
        "schema_version": SCHEMA,
        "evidence_status": (
            "ENGINEERING_SMOKE_ONLY" if args.smoke else "DEVELOPMENT_CONSUMED_SCREEN"
        ),
        "seed": args.seed,
        "expert_epochs": expert_epochs,
        "residual_epochs": residual_epochs,
        "folds": results,
        "qualification": {
            "external": "NOT_ACCESSED",
            "sota": "NOT_ESTABLISHED",
            "partial_results_must_not_change_configuration": True,
        },
    }
    output_path = args.out_dir / f"v3_result_seed{args.seed}.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "result": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
