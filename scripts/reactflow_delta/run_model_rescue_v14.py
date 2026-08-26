#!/usr/bin/env python3
"""Run score-blind masked-WT-pretraining folds for Model Rescue v14."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import aligned_wt_ctx_tensors
from scripts.reactflow_delta.model_rescue_v10 import parameter_count as residual_count
from scripts.reactflow_delta.model_rescue_v14 import (
    EXPECTED_DOWNSTREAM_PARAMETERS,
    EXPECTED_TOTAL_PARAMETERS,
    FOLD_SCHEMA,
    FROM_SCRATCH_NULL,
    PREDICTION_SCHEMA,
    PRETRAINED_CANDIDATE,
    assert_exact_initial_match,
    assert_snapshot_equal,
    fit_point_model,
    freeze_point_model,
    module_snapshot,
    parameter_count,
    pretrain_wt_encoder,
    make_exact_matched_pair,
)
from scripts.reactflow_delta.model_rescue_v2 import freeze_mean_model
from scripts.reactflow_delta.model_rescue_v5_probe import EnsembleFeatureCache
from scripts.reactflow_delta.model_rescue_v6_probe import (
    ConstrainedFeatureCache,
    validate_cache_alignment,
)
from scripts.reactflow_delta.run_model_rescue_v11 import (
    POINT_NAMES,
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
    fit_v10_residual_head,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta.validate_model_rescue_v14_contract import (
    assert_run_authority,
)


def _v14_prediction(v11_prediction: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rename the reused V11 prediction mechanics into the V14 schema."""
    output: dict[str, np.ndarray] = {}
    for name, value in v11_prediction.items():
        if name == "schema_version":
            output[name] = np.asarray(PREDICTION_SCHEMA)
        elif name.startswith("anchored_"):
            output[f"candidate_{name.removeprefix('anchored_')}"] = value
        elif name.startswith("unanchored_"):
            output[f"null_{name.removeprefix('unanchored_')}"] = value
        else:
            output[name] = value
    forbidden_prefixes = ("anchored_", "unanchored_")
    if any(name.startswith(forbidden_prefixes) for name in output):
        raise RuntimeError("V14 prediction retained a V11 candidate name")
    return output


def _pretraining_contexts(
    all_contexts: dict[str, tuple[torch.Tensor, ...]],
    train_construct_ids: set[str],
    held_construct_ids: set[str],
) -> dict[str, tuple[torch.Tensor, ...]]:
    if train_construct_ids & held_construct_ids:
        raise RuntimeError("V14 held construct entered the outer-train WT universe")
    if len(train_construct_ids) != 152:
        raise RuntimeError(
            f"V14 expected 152 outer-train WT constructs, got {len(train_construct_ids)}"
        )
    return {construct_id: all_contexts[construct_id] for construct_id in sorted(train_construct_ids)}


def run_fold(
    *,
    univ: M2Universe,
    records: list[Any],
    fold: Any,
    device: str,
    out_dir: Path,
    v8_dir: Path,
    v10_dir: Path,
    tic2a_merged: dict[str, Any],
    unconstrained: EnsembleFeatureCache,
    constrained: ConstrainedFeatureCache,
    pretraining_epochs: int,
    point_epochs: int,
    calibration_epochs: int,
    seed: int,
    phase: str,
) -> dict[str, Any]:
    fold_id = int(fold.outer_fold)
    v8_row, tic_row, v10_row, feature41_model = _fold_sources(
        fold_id,
        v8_dir=v8_dir,
        v10_dir=v10_dir,
        tic2a_merged=tic2a_merged,
    )
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    train_construct_ids = {record.construct_id for record in train_records}
    held_construct_ids = {record.construct_id for record in held_records}
    all_construct_ids = sorted(train_construct_ids | held_construct_ids)
    context_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in all_construct_ids
    }
    pretraining_contexts = _pretraining_contexts(
        context_cache, train_construct_ids, held_construct_ids
    )
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
        raise RuntimeError("V14 feature41 replay exceeds 1e-7")
    cells = _point_cells(
        univ,
        train_records,
        feature41_model,
        unconstrained,
        constrained,
        device,
    )

    candidate, null = make_exact_matched_pair(seed=seed, device=device)
    null_common_initialization = module_snapshot(null)
    candidate_encoder_before = module_snapshot(candidate.input_projection)
    candidate_residual_before = module_snapshot(candidate.residual_head)
    null_residual_before = module_snapshot(null.residual_head)
    pretraining_history = pretrain_wt_encoder(
        candidate,
        pretraining_contexts,
        epochs=pretraining_epochs,
        seed=seed,
    )
    assert_snapshot_equal(null_common_initialization, null, "from-scratch null before supervised training")
    assert_snapshot_equal(candidate_residual_before, candidate.residual_head, "candidate residual before supervised training")
    assert_snapshot_equal(null_residual_before, null.residual_head, "null residual before supervised training")
    if all(
        torch.equal(value, candidate.input_projection.state_dict()[name].detach().cpu())
        for name, value in candidate_encoder_before.items()
    ):
        raise RuntimeError("V14 candidate encoder did not change during pretraining")
    for name, value in candidate.residual_head.state_dict().items():
        if not torch.equal(value.detach().cpu(), null.residual_head.state_dict()[name].detach().cpu()):
            raise RuntimeError("V14 residual heads differ before supervised step one")

    candidate_history = fit_point_model(
        candidate,
        cells,
        context_cache,
        epochs=point_epochs,
        seed=seed,
    )
    null_history = fit_point_model(
        null,
        cells,
        context_cache,
        epochs=point_epochs,
        seed=seed,
    )
    point_parameter_counts = {
        PRETRAINED_CANDIDATE: parameter_count(candidate, trainable_only=True),
        FROM_SCRATCH_NULL: parameter_count(null, trainable_only=True),
    }
    if set(point_parameter_counts.values()) != {EXPECTED_DOWNSTREAM_PARAMETERS}:
        raise RuntimeError("V14 downstream parameter count changed")
    if parameter_count(candidate) != EXPECTED_TOTAL_PARAMETERS or parameter_count(null) != EXPECTED_TOTAL_PARAMETERS:
        raise RuntimeError("V14 total parameter count changed")

    candidate_checkpoint = out_dir / f"v14_candidate_point_fold{fold_id}_seed{seed}.pt"
    null_checkpoint = out_dir / f"v14_null_point_fold{fold_id}_seed{seed}.pt"
    torch.save(candidate.state_dict(), candidate_checkpoint)
    torch.save(null.state_dict(), null_checkpoint)
    freeze_point_model(candidate)
    freeze_point_model(null)
    candidate_snapshot = module_snapshot(candidate)
    null_snapshot = module_snapshot(null)

    v8_model = _load_v8_mean(Path(v8_row["meanaligned_checkpoint"]), device)
    freeze_mean_model(v8_model)
    calibration_cells = _calibration_cells(
        cells,
        anchored=candidate,
        unanchored=null,
        v8_model=v8_model,
        v11_context_cache=context_cache,
        v8_context_cache=context_cache,
    )
    heads = _new_residual_heads(seed, device)
    standardizers = {}
    calibration_inputs = {}
    histories = {}
    for name in POINT_NAMES:
        standardizers[name], calibration_inputs[name] = _prepare_calibration_inputs(
            calibration_cells, name
        )
        if name == "feature41" and phase != "V14M2" and seed == 0:
            head, standardizer, history = _load_authoritative_v10_feature41(
                v10_row, device
            )
            if not np.array_equal(standardizers[name].mean, standardizer.mean) or not np.array_equal(standardizers[name].scale, standardizer.scale):
                raise RuntimeError("V14 feature41 standardizer does not replay V10")
            heads[name] = head
            standardizers[name] = standardizer
            histories[name] = history
        else:
            histories[name] = fit_v10_residual_head(
                heads[name],
                calibration_cells,
                calibration_inputs[name],
                f"{name}_point",
                device,
                calibration_epochs,
                seed,
            )
    _assert_unchanged(candidate_snapshot, candidate, "V14 candidate point")
    _assert_unchanged(null_snapshot, null, "V14 null point")
    if any(parameter.grad is not None for parameter in candidate.parameters()) or any(
        parameter.grad is not None for parameter in null.parameters()
    ):
        raise RuntimeError("V14 calibration produced point gradients")

    residual_checkpoints = {}
    for name, head in heads.items():
        mapped = {"anchored": "candidate", "unanchored": "null"}.get(name, name)
        path = out_dir / f"v14_{mapped}_asymmetric_fold{fold_id}_seed{seed}.pt"
        torch.save(
            {
                "state_dict": head.state_dict(),
                "standardizer_mean": standardizers[name].mean,
                "standardizer_scale": standardizers[name].scale,
                "point_name": mapped,
            },
            path,
        )
        residual_checkpoints[mapped] = str(path)

    prediction = _v14_prediction(
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
            require_v10_feature41_replay=(phase != "V14M2" and seed == 0),
        )
    )
    prediction_path = out_dir / f"v14_predictions_fold{fold_id}_seed{seed}.npz"
    np.savez_compressed(prediction_path, **prediction)
    return {
        "schema_version": FOLD_SCHEMA,
        "phase": phase,
        "evidence_status": (
            "ENGINEERING_SMOKE_ONLY"
            if phase == "V14M2"
            else "DEVELOPMENT_CONSUMED_PREDICTION_ONLY"
        ),
        "outer_fold": fold_id,
        "held_puzzle": str(fold.held_puzzle),
        "seed": seed,
        "pretraining_epochs": pretraining_epochs,
        "point_epochs": point_epochs,
        "calibration_epochs": calibration_epochs,
        "point_checkpoints": {
            "candidate": str(candidate_checkpoint),
            "null": str(null_checkpoint),
        },
        "residual_checkpoints": residual_checkpoints,
        "prediction_artifact": str(prediction_path),
        "training_histories": {
            "candidate_pretraining": pretraining_history,
            "candidate_point": candidate_history,
            "null_point": null_history,
            **{
                f"{ {'anchored': 'candidate', 'unanchored': 'null'}.get(name, name) }_residual": values
                for name, values in histories.items()
            },
        },
        "n_pretraining_constructs": len(pretraining_contexts),
        "n_train_cells": len(cells),
        "n_registered_prediction_rows": int(len(prediction["keys"])),
        "feature41_replay_max_abs_difference": replay,
        "total_parameter_counts": {
            PRETRAINED_CANDIDATE: EXPECTED_TOTAL_PARAMETERS,
            FROM_SCRATCH_NULL: EXPECTED_TOTAL_PARAMETERS,
        },
        "point_parameter_counts": point_parameter_counts,
        "residual_parameter_counts": {
            {"anchored": "candidate", "unanchored": "null"}.get(name, name): residual_count(head)
            for name, head in heads.items()
        },
        "invariants": {
            "target_profile_identity_exact": True,
            "outer_train_wt_only_pretraining": True,
            "held_puzzle_wt_excluded_from_pretraining": True,
            "mutant_outcome_excluded_from_pretraining": True,
            "exact_initial_parameter_match": True,
            "exact_total_and_downstream_parameter_match": True,
            "candidate_encoder_changed_during_pretraining": True,
            "null_state_unchanged_before_supervised_training": True,
            "residual_heads_identical_before_supervised_step_one": True,
            "pretraining_decoder_frozen_downstream": True,
            "same_point_training_order_and_dropout_stream": True,
            "point_frozen_during_calibration": True,
            "v10_residual_family_reused": True,
            "feature41_replay_at_1e_7": True,
            "median_constraint_all_held_rows": True,
            "held_score_computed": False,
            "prediction_contains_target_fields": False,
            "external_outcome_accessed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("V14M2", "V14M3", "V14M4"), required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--v10-dir", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--pretraining-epochs", type=int, required=True)
    parser.add_argument("--point-epochs", type=int, required=True)
    parser.add_argument("--calibration-epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    assert_run_authority(repo_root, args.phase)
    folds = _parse_folds(args.folds)
    schedule = (
        args.pretraining_epochs,
        args.point_epochs,
        args.calibration_epochs,
    )
    if args.phase == "V14M2":
        if args.seed != 0 or not set(folds) <= {0, 1} or schedule != (3, 3, 3):
            raise ValueError("V14M2 is frozen to seed0 folds0/1 and 3+3+3 epochs")
    elif args.phase == "V14M3":
        if args.seed != 0 or schedule != (200, 40, 40):
            raise ValueError("V14M3 is frozen to seed0 and 200+40+40 epochs")
    elif args.seed not in range(5) or schedule != (200, 40, 40):
        raise ValueError("V14M4 is frozen to seeds0-4 and 200+40+40 epochs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold_id in folds:
        result_path = args.out_dir / f"v14_fold_result_fold{fold_id}_seed{args.seed}.json"
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite V14 fold {fold_id}")
    device = args.device if torch.cuda.is_available() else "cpu"
    univ = M2Universe(args.m2_csv)
    identity = univ.build()
    if identity.get("n_canonical_mutant_full_profiles") != 13976 or identity.get(
        "canonical_mutant_full_profile_identity"
    ) != "EXACT_PUZZLE_METHOD_MUTATION":
        raise RuntimeError("V14 requires exact canonical target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    selected = [fold for fold in split["folds"] if int(fold.outer_fold) in set(folds)]
    if len(selected) != len(folds):
        raise ValueError("one or more requested V14 folds are absent")
    tic2a_merged = _read_json(args.tic2a_merged_json)
    unconstrained = EnsembleFeatureCache(args.unconstrained_cache)
    constrained = ConstrainedFeatureCache(args.constrained_cache)
    validate_cache_alignment(unconstrained, constrained)
    try:
        for fold in selected:
            fold_id = int(fold.outer_fold)
            print(f"[{args.phase}] fold={fold_id} seed={args.seed} start", flush=True)
            result = run_fold(
                univ=univ,
                records=records,
                fold=fold,
                device=device,
                out_dir=args.out_dir,
                v8_dir=args.v8_dir,
                v10_dir=args.v10_dir,
                tic2a_merged=tic2a_merged,
                unconstrained=unconstrained,
                constrained=constrained,
                pretraining_epochs=args.pretraining_epochs,
                point_epochs=args.point_epochs,
                calibration_epochs=args.calibration_epochs,
                seed=args.seed,
                phase=args.phase,
            )
            result_path = args.out_dir / f"v14_fold_result_fold{fold_id}_seed{args.seed}.json"
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"[{args.phase}] fold={fold_id} complete", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        unconstrained.close()
        constrained.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
