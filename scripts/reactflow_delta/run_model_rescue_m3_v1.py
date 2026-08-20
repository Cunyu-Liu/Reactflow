#!/usr/bin/env python3
"""M3 nested adaptive model-rescue evaluation on consumed development data.

Family, SparseDelta lambda, and comparator choices are made from outer-train
inner puzzles only. The outer-held puzzle is predicted once after selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta import evaluator_v2 as E
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import AlignedDeltaModel, aligned_wt_ctx_tensors
from scripts.reactflow_delta.run_model_rescue_m2_v1 import (
    CandidateSpec,
    fit_candidate,
    predict_held,
    score_predictions,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_m3_nested.v1"
SEEDS = [0, 1, 2, 3, 4]
SPECS = {
    "b1_rfd_direct_aligned": CandidateSpec("b1_rfd_direct_aligned", 0, False, 0.0),
    "l2_aligned_rank2": CandidateSpec("l2_aligned_rank2", 2, False, 0.0),
    "sparse_delta_mdn_h0": CandidateSpec("sparse_delta_mdn_h0", 0, True, 0.0),
    "sparse_delta_mdn_h01": CandidateSpec("sparse_delta_mdn_h01", 0, True, 0.1),
}
TIE_ORDER = {
    "b1_rfd_direct_aligned": 0,
    "l2_aligned_rank2": 1,
    "sparse_delta_mdn_h0": 2,
    "sparse_delta_mdn_h01": 3,
}


def eligible_specs(qualification: dict[str, Any]) -> list[CandidateSpec]:
    if qualification.get("overall_status") != "M2_SCREEN_PASS":
        raise ValueError("M3 is closed unless M2 qualification is M2_SCREEN_PASS")
    families = set(qualification.get("m3_eligible_families", []))
    ids = ["b1_rfd_direct_aligned"]
    if "l2_aligned_rank2" in families:
        ids.append("l2_aligned_rank2")
    if "sparse_delta_mdn_inner_selected_lambda" in families:
        ids.extend(["sparse_delta_mdn_h0", "sparse_delta_mdn_h01"])
    return [SPECS[x] for x in ids]


def combine_seed_predictions(predictions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if len(predictions) != 5:
        raise ValueError("formal M3 deployment requires exactly five seeds")
    base_keys = predictions[0]["keys"]
    for row in predictions[1:]:
        if not np.array_equal(row["keys"], base_keys):
            raise ValueError("seed prediction key universes differ")
    locations = np.concatenate([row["locations"] for row in predictions], axis=1)
    scales = np.concatenate([row["scales"] for row in predictions], axis=1)
    weights = np.concatenate([row["weights"] / 5.0 for row in predictions], axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    return {"keys": base_keys, "locations": locations, "scales": scales, "weights": weights}


def choose_from_inner(inner: dict[str, dict[str, float]], candidate_ids: list[str]) -> dict[str, Any]:
    baseline = inner["b1_rfd_direct_aligned"]
    rows = []
    for candidate in candidate_ids:
        score = inner[candidate]
        crps_ratio = score["crps"] / baseline["crps"]
        delta_ratio = score["signed_delta_mae"] / baseline["signed_delta_mae"]
        feasible = score["crps"] <= baseline["crps"] and score["signed_delta_mae"] <= baseline["signed_delta_mae"]
        rows.append(
            {
                "candidate": candidate,
                "crps_ratio_vs_b1": crps_ratio,
                "delta_mae_ratio_vs_b1": delta_ratio,
                "aggregate_inner_score": 0.5 * crps_ratio + 0.5 * delta_ratio,
                "feasible": feasible,
            }
        )
    feasible_rows = [row for row in rows if row["feasible"]]
    if not feasible_rows:
        raise RuntimeError("B1 must always be feasible against itself")
    selected = min(
        feasible_rows,
        key=lambda row: (row["aggregate_inner_score"], TIE_ORDER[row["candidate"]]),
    )
    comparator_ids = [x for x in ["b1_rfd_direct_aligned", "l2_aligned_rank2"] if x in candidate_ids]
    comparator = min(
        [row for row in rows if row["candidate"] in comparator_ids and row["feasible"]],
        key=lambda row: (row["aggregate_inner_score"], TIE_ORDER[row["candidate"]]),
    )
    return {
        "selected_candidate": selected["candidate"],
        "selected_comparator": comparator["candidate"],
        "candidate_rows": rows,
    }


def _score_validation_puzzles(
    model: AlignedDeltaModel,
    univ: M2Universe,
    records: list[Any],
    ctx_cache: dict[str, Any],
    device: str,
) -> list[dict[str, Any]]:
    rows = []
    for puzzle in sorted({r.puzzle for r in records}):
        puzzle_records = [r for r in records if r.puzzle == puzzle]
        prediction = predict_held(model, univ, puzzle_records, ctx_cache, device)
        score = score_predictions(prediction, univ, puzzle_records)
        rows.append({"puzzle": puzzle, **score})
    return rows


def score_wt_anchor_signed_delta_mae(univ: M2Universe, held_records: list[Any]) -> float:
    """Method-balanced signed-delta MAE for the no-change WT anchor."""
    losses: dict[str, float] = {}
    for record in held_records:
        construct = univ.get_construct(record.construct_id)
        target, _ = univ.mutant_full_profile(record.wt_id, record.pos, record.ref, record.alt)
        if target is None:
            continue
        for pos in range(len(construct.sequence)):
            if construct.wt_observed[pos] and np.isfinite(target[pos]):
                losses[_bio_key(univ, record, pos)] = abs(
                    float(target[pos]) - float(construct.wt_reactivity[pos])
                )
    result = E.score_position_losses(losses, method_balanced=True)
    if len(result["puzzles"]) != 1:
        raise ValueError("WT anchor scoring expects exactly one outer-held puzzle")
    return float(next(iter(result["puzzles"].values()))["L"])


def inner_evaluate(
    spec: CandidateSpec,
    univ: M2Universe,
    outer_train_records: list[Any],
    inner_groups: list[list[str]],
    ctx_cache: dict[str, Any],
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    puzzle_rows = []
    for inner_fold, validation_puzzles in enumerate(inner_groups):
        validation_set = set(validation_puzzles)
        train_records = [r for r in outer_train_records if r.puzzle not in validation_set]
        validation_records = [r for r in outer_train_records if r.puzzle in validation_set]
        torch.manual_seed(0)
        model = AlignedDeltaModel(k_rank=spec.k_rank, sparse=spec.sparse).to(device)
        fit_candidate(
            model,
            univ,
            train_records,
            ctx_cache,
            device,
            epochs,
            learning_rate,
            weight_decay,
            spec.huber_lambda,
        )
        scored = _score_validation_puzzles(model, univ, validation_records, ctx_cache, device)
        for row in scored:
            row["inner_fold"] = inner_fold
        puzzle_rows.extend(scored)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(puzzle_rows) != len({r.puzzle for r in outer_train_records}):
        raise ValueError("inner OOF must score every outer-train puzzle exactly once")
    return {
        "crps": float(np.mean([row["crps"] for row in puzzle_rows])),
        "signed_delta_mae": float(np.mean([row["signed_delta_mae"] for row in puzzle_rows])),
        "puzzles": puzzle_rows,
    }


def fit_five_seed_prediction(
    spec: CandidateSpec,
    univ: M2Universe,
    train_records: list[Any],
    held_records: list[Any],
    ctx_cache: dict[str, Any],
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, np.ndarray]:
    predictions = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = AlignedDeltaModel(k_rank=spec.k_rank, sparse=spec.sparse).to(device)
        fit_candidate(
            model,
            univ,
            train_records,
            ctx_cache,
            device,
            epochs,
            learning_rate,
            weight_decay,
            spec.huber_lambda,
        )
        predictions.append(predict_held(model, univ, held_records, ctx_cache, device))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return combine_seed_predictions(predictions)


def run_outer_fold(
    univ: M2Universe,
    all_records: list[Any],
    fold: Any,
    specs: list[CandidateSpec],
    device: str,
    out_dir: Path,
    inner_epochs: int,
    final_epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    train_set = set(fold.train_puzzles)
    outer_train = [r for r in all_records if r.puzzle in train_set]
    held = [r for r in all_records if r.puzzle == fold.held_puzzle]
    construct_ids = sorted({r.construct_id for r in outer_train + held})
    ctx_cache = {cid: aligned_wt_ctx_tensors(univ, cid, device) for cid in construct_ids}
    inner: dict[str, Any] = {}
    for spec in specs:
        print(f"[M3] outer={fold.outer_fold} inner candidate={spec.model_id} start", flush=True)
        inner[spec.model_id] = inner_evaluate(
            spec,
            univ,
            outer_train,
            fold.inner_groups,
            ctx_cache,
            device,
            inner_epochs,
            learning_rate,
            weight_decay,
        )
        print(
            f"[M3] outer={fold.outer_fold} inner candidate={spec.model_id} "
            f"crps={inner[spec.model_id]['crps']:.8f} "
            f"delta={inner[spec.model_id]['signed_delta_mae']:.8f}",
            flush=True,
        )
    selection = choose_from_inner(
        {key: {"crps": value["crps"], "signed_delta_mae": value["signed_delta_mae"]} for key, value in inner.items()},
        [spec.model_id for spec in specs],
    )
    required_ids = sorted(
        {selection["selected_candidate"], selection["selected_comparator"]},
        key=lambda x: TIE_ORDER[x],
    )
    predictions: dict[str, dict[str, np.ndarray]] = {}
    scores: dict[str, Any] = {}
    for model_id in required_ids:
        print(f"[M3] outer={fold.outer_fold} final model={model_id} five-seed start", flush=True)
        prediction = fit_five_seed_prediction(
            SPECS[model_id],
            univ,
            outer_train,
            held,
            ctx_cache,
            device,
            final_epochs,
            learning_rate,
            weight_decay,
        )
        predictions[model_id] = prediction
        scores[model_id] = score_predictions(prediction, univ, held)
        np.savez_compressed(out_dir / f"m3_oof_{model_id}_fold{fold.outer_fold}.npz", **prediction)
    selected = selection["selected_candidate"]
    comparator = selection["selected_comparator"]
    candidate_score = scores[selected]
    comparator_score = scores[comparator]
    wt_anchor_delta_mae = score_wt_anchor_signed_delta_mae(univ, held)
    return {
        "outer_fold": int(fold.outer_fold),
        "held_puzzle": fold.held_puzzle,
        "inner_results": inner,
        "selection": selection,
        "outer_scores": scores,
        "wt_anchor_signed_delta_mae": wt_anchor_delta_mae,
        "effects": {
            "crps_gain": float(comparator_score["crps"] - candidate_score["crps"]),
            "signed_delta_mae_gain": float(
                comparator_score["signed_delta_mae"] - candidate_score["signed_delta_mae"]
            ),
            "signed_delta_mae_gain_vs_wt_anchor": float(
                wt_anchor_delta_mae - candidate_score["signed_delta_mae"]
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--m2-qualification", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--inner-epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args(argv)
    qualification = json.loads(args.m2_qualification.read_text(encoding="utf-8"))
    specs = eligible_specs(qualification)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    univ = M2Universe(args.m2_csv)
    univ.build()
    records = univ.get_records()
    split = build_split_v4(sorted({r.puzzle for r in records}), seed=20260813)
    requested = {int(x) for x in args.folds.split(",") if x}
    folds = [fold for fold in split["folds"] if fold.outer_fold in requested]
    results = []
    for fold in folds:
        row = run_outer_fold(
            univ,
            records,
            fold,
            specs,
            device,
            args.out_dir,
            args.inner_epochs,
            args.final_epochs,
            args.learning_rate,
            args.weight_decay,
        )
        results.append(row)
        path = args.out_dir / f"m3_fold_result_fold{fold.outer_fold}.json"
        path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[M3] outer={fold.outer_fold} artifact={path} complete", flush=True)
    final = {
        "schema_version": SCHEMA,
        "evidence_status": "POST_HOC_DEVELOPMENT_NESTED",
        "m2_qualification": str(args.m2_qualification),
        "eligible_specs": [spec.model_id for spec in specs],
        "inner_epochs": args.inner_epochs,
        "final_epochs": args.final_epochs,
        "seeds": SEEDS,
        "folds": results,
        "new_external_outcomes_accessed": False,
    }
    out = args.out_dir / "m3_model_rescue_nested_result.json"
    out.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "result": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
