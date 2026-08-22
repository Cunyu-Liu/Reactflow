#!/usr/bin/env python3
"""Rebuild corrected-coordinate B1 and MeanAligned outer experts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import (
    AlignedDeltaModel,
    aligned_wt_ctx_tensors,
)
from scripts.reactflow_delta.model_rescue_v2 import MeanAlignedModel
from scripts.reactflow_delta.run_model_rescue_m2_v1 import fit_candidate
from scripts.reactflow_delta.run_model_rescue_v2 import _make_cells, fit_mean
from scripts.reactflow_delta.run_model_rescue_v3 import predict_expert_means
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v3_corrected_expert_rebuild.v1"
PREDICTION_SCHEMA = "reactflow_delta.model_rescue_v3_corrected_expert_prediction.v1"


def assert_expert_rebuild_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "R3C3":
        raise RuntimeError("corrected expert rebuild is closed outside R3C3")
    if active.get("runnable_phases") != ["R3C3"]:
        raise RuntimeError("R3C3 must be the only runnable phase")
    if active.get("training_allowed") != (
        "CORRECTED_B1_AND_MEANALIGNED_REBUILD_ONLY"
    ):
        raise RuntimeError("R3C3 corrected expert rebuild authority is absent")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("R3C3 requires external outcomes to remain locked")


def _save_expert_prediction(
    path: Path,
    prediction: dict[str, np.ndarray],
    *,
    fold: int,
    seed: int,
) -> None:
    keys = np.asarray(prediction["keys"], dtype=object)
    if len(keys) != len(set(keys.tolist())):
        raise RuntimeError("corrected expert prediction contains duplicate keys")
    np.savez_compressed(
        path,
        schema_version=np.asarray(PREDICTION_SCHEMA),
        keys=keys,
        b1_delta_mean=np.asarray(prediction["b1_delta_mean"], dtype=np.float64),
        meanaligned_delta_mean=np.asarray(
            prediction["meanaligned_delta_mean"], dtype=np.float64
        ),
        outer_fold=np.full(len(keys), fold, dtype=np.int64),
        seed=np.full(len(keys), seed, dtype=np.int64),
    )


def run_expert_fold(
    *,
    univ: M2Universe,
    fold: Any,
    records: list[Any],
    device: str,
    out_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    train_puzzles = set(fold.train_puzzles)
    train_records = [record for record in records if record.puzzle in train_puzzles]
    held_records = [record for record in records if record.puzzle == fold.held_puzzle]
    construct_ids = sorted(
        {record.construct_id for record in train_records + held_records}
    )
    ctx_cache = {
        construct_id: aligned_wt_ctx_tensors(univ, construct_id, device)
        for construct_id in construct_ids
    }

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
    if len(b1_history) != epochs or len(mean_history) != epochs:
        raise RuntimeError("corrected expert histories do not match frozen epochs")
    if not np.isfinite(np.asarray(b1_history + mean_history, dtype=float)).all():
        raise RuntimeError("corrected expert training produced a non-finite loss")

    fold_id = int(fold.outer_fold)
    b1_checkpoint = out_dir / f"v3_corrected_b1_fold{fold_id}_seed{seed}.pt"
    mean_checkpoint = out_dir / f"v3_corrected_mean_fold{fold_id}_seed{seed}.pt"
    torch.save(b1_model.state_dict(), b1_checkpoint)
    torch.save(mean_model.state_dict(), mean_checkpoint)

    prediction = predict_expert_means(
        b1_model,
        mean_model,
        univ,
        held_records,
        ctx_cache,
        device,
    )
    prediction_path = out_dir / (
        f"v3_corrected_expert_predictions_fold{fold_id}_seed{seed}.npz"
    )
    _save_expert_prediction(
        prediction_path, prediction, fold=fold_id, seed=seed
    )
    return {
        "schema_version": SCHEMA,
        "evidence_status": "DEVELOPMENT_CONSUMED_EXPERT_REBUILD_ONLY",
        "outer_fold": fold_id,
        "held_puzzle": fold.held_puzzle,
        "seed": seed,
        "epochs": epochs,
        "b1_checkpoint": str(b1_checkpoint),
        "meanaligned_checkpoint": str(mean_checkpoint),
        "expert_prediction_artifact": str(prediction_path),
        "b1_train_loss": b1_history,
        "meanaligned_train_loss": mean_history,
        "held_score_computed": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    assert_expert_rebuild_authority(args.repo_root.resolve())
    if args.seed != 0:
        raise ValueError("R3C3 rebuild is frozen to seed 0")
    if args.epochs != 40:
        raise ValueError("R3C3 rebuild is frozen to 40 epochs")
    if args.learning_rate != 1e-3 or args.weight_decay != 0.0:
        raise ValueError("R3C3 optimizer configuration is frozen")

    selected = [int(value) for value in args.folds.split(",") if value]
    if len(selected) != len(set(selected)) or not set(selected) <= set(range(20)):
        raise ValueError("R3C3 folds must be unique members of 0 through 19")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    universe = M2Universe(args.m2_csv)
    universe.build()
    records = universe.get_records()
    split = build_split_v4(
        sorted({record.puzzle for record in records}), seed=20260813
    )
    folds = [fold for fold in split["folds"] if fold.outer_fold in set(selected)]
    if len(folds) != len(selected):
        raise ValueError("one or more requested R3C3 folds are absent")

    for fold in folds:
        print(
            f"[R3C3] fold={fold.outer_fold} held={fold.held_puzzle} seed=0 start",
            flush=True,
        )
        result = run_expert_fold(
            univ=universe,
            fold=fold,
            records=records,
            device=device,
            out_dir=args.out_dir,
            epochs=40,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=0,
        )
        path = args.out_dir / (
            f"v3_corrected_expert_fold_result_fold{fold.outer_fold}_seed0.json"
        )
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[R3C3] fold={fold.outer_fold} artifact={path} complete", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
