#!/usr/bin/env python3
"""M2 bounded candidate screen for aligned direct, rank-2, and SparseDelta."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta import evaluator_v2 as E
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.model_rescue_v1 import (
    AlignedDeltaModel,
    aligned_mixture_loss,
    aligned_wt_ctx_tensors,
    weighted_gaussian_mixture_crps,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.run_p3_lrso_v3 import (
    _qualified_mask,
    _target_matrix,
    _wt_filled,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_m2.v1"


@dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    k_rank: int
    sparse: bool
    huber_lambda: float


CANDIDATES = [
    CandidateSpec("b1_rfd_direct_aligned", 0, False, 0.0),
    CandidateSpec("l2_aligned_rank2", 2, False, 0.0),
    CandidateSpec("sparse_delta_mdn_h0", 0, True, 0.0),
    CandidateSpec("sparse_delta_mdn_h01", 0, True, 0.1),
]


def _make_batches(univ: M2Universe, records: list[Any], device: str):
    by: dict[str, list[Any]] = {}
    for record in records:
        by.setdefault(record.construct_id, []).append(record)
    batches = []
    for construct_id, recs in sorted(by.items()):
        target, wt_obs = _target_matrix(univ, recs)
        qualified = _qualified_mask(target, wt_obs)
        if qualified.sum() == 0:
            continue
        length = target.shape[1]
        edit = torch.tensor([r.pos for r in recs], device=device)
        distance = (torch.arange(length, device=device)[None, :] - edit[:, None]).float()
        batches.append(
            {
                "construct_id": construct_id,
                "edit": edit,
                "distance": distance,
                "refs": [r.ref for r in recs],
                "alts": [r.alt for r in recs],
                "target": torch.tensor(target, device=device),
                "prediction_mask": torch.tensor(wt_obs, device=device),
                "qualified_mask": torch.tensor(qualified, device=device),
                "wt": torch.tensor(_wt_filled(univ, construct_id), device=device),
            }
        )
    return batches


def fit_candidate(
    model: AlignedDeltaModel,
    univ: M2Universe,
    train_records: list[Any],
    ctx_cache: dict[str, Any],
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    huber_lambda: float,
) -> list[float]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    batches = _make_batches(univ, train_records, device)
    history = []
    for _ in range(epochs):
        losses = []
        for batch in batches:
            H = model.encode(ctx_cache[batch["construct_id"]])
            loss = aligned_mixture_loss(
                model,
                H,
                batch["edit"],
                batch["distance"],
                batch["refs"],
                batch["alts"],
                batch["target"],
                batch["prediction_mask"],
                batch["qualified_mask"],
                batch["wt"],
                huber_lambda=huber_lambda,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
    return history


def predict_held(
    model: AlignedDeltaModel,
    univ: M2Universe,
    held_records: list[Any],
    ctx_cache: dict[str, Any],
    device: str,
) -> dict[str, np.ndarray]:
    """Prediction-only arrays. Mutant targets, errors, and target masks are not read."""
    keys: list[str] = []
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    by: dict[str, list[Any]] = {}
    for record in held_records:
        by.setdefault(record.construct_id, []).append(record)
    model.eval()
    with torch.no_grad():
        for construct_id, recs in sorted(by.items()):
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            edit = torch.tensor([r.pos for r in recs], device=device)
            distance = (torch.arange(length, device=device)[None, :] - edit[:, None]).float()
            prediction_mask = torch.tensor(
                np.tile(construct.wt_observed.astype(bool), (len(recs), 1)), device=device
            )
            H = model.encode(ctx_cache[construct_id])
            component_weights, delta_locations, component_scales = model.forward_distribution(
                H,
                edit,
                distance,
                [r.ref for r in recs],
                [r.alt for r in recs],
                prediction_mask,
            )
            wt = torch.tensor(_wt_filled(univ, construct_id), device=device)
            mutant_locations = delta_locations + wt[None, :, None]
            w_np = component_weights.cpu().numpy()
            loc_np = mutant_locations.cpu().numpy()
            scale_np = component_scales.cpu().numpy()
            for row, record in enumerate(recs):
                for pos in range(length):
                    keys.append(_bio_key(univ, record, pos))
                    locations.append(loc_np[row, pos])
                    scales.append(scale_np[row, pos])
                    weights.append(w_np[row, pos])
    return {
        "keys": np.asarray(keys, dtype=object),
        "locations": np.asarray(locations, dtype=np.float64),
        "scales": np.asarray(scales, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
    }


def score_predictions(
    prediction: dict[str, np.ndarray],
    univ: M2Universe,
    held_records: list[Any],
) -> dict[str, Any]:
    """Evaluator-side target join and method-balanced CRPS/signed-delta MAE."""
    target_map: dict[str, tuple[float, float]] = {}
    expected_keys: set[str] = set()
    for record in held_records:
        construct = univ.get_construct(record.construct_id)
        expected_keys.update(
            _bio_key(univ, record, pos) for pos in range(len(construct.sequence))
        )
        target, _ = univ.mutant_full_profile(record.wt_id, record.pos, record.ref, record.alt)
        if target is None:
            continue
        for pos in range(len(construct.sequence)):
            if construct.wt_observed[pos] and np.isfinite(target[pos]):
                target_map[_bio_key(univ, record, pos)] = (
                    float(target[pos]), float(construct.wt_reactivity[pos])
                )

    crps_losses: dict[str, float] = {}
    delta_losses: dict[str, float] = {}
    coverage68: dict[str, float] = {}
    coverage95: dict[str, float] = {}
    key_to_index = {str(key): i for i, key in enumerate(prediction["keys"])}
    predicted_keys = set(key_to_index)
    valid_prediction_rows = np.isfinite(prediction["locations"]).all(axis=1)
    valid_prediction_rows &= np.isfinite(prediction["scales"]).all(axis=1)
    valid_prediction_rows &= np.isfinite(prediction["weights"]).all(axis=1)
    valid_prediction_rows &= (prediction["scales"] > 0).all(axis=1)
    valid_prediction_rows &= prediction["weights"].sum(axis=1) > 0
    valid_by_key = {
        str(key): bool(valid_prediction_rows[i])
        for i, key in enumerate(prediction["keys"])
    }
    covered_expected = expected_keys & predicted_keys
    finite_expected = sum(valid_by_key.get(key, False) for key in expected_keys)
    registered_coverage = len(covered_expected) / max(len(expected_keys), 1)
    failure_rate = 1.0 - finite_expected / max(len(expected_keys), 1)
    if set(target_map) - set(key_to_index):
        raise ValueError("qualified target key missing from prediction-only ledger")
    for key, (target, wt) in target_map.items():
        i = key_to_index[key]
        loc = prediction["locations"][i]
        scale = prediction["scales"][i]
        weight = prediction["weights"][i]
        crps_losses[key] = float(
            weighted_gaussian_mixture_crps(
                loc[None, :], scale[None, :], weight[None, :], np.array([target])
            )[0]
        )
        mean = float(np.sum(weight * loc) / np.sum(weight))
        second = float(np.sum(weight * (scale**2 + loc**2)) / np.sum(weight))
        sd = float(np.sqrt(max(second - mean**2, 1e-12)))
        delta_losses[key] = abs((target - wt) - (mean - wt))
        coverage68[key] = float(abs(target - mean) <= sd)
        coverage95[key] = float(abs(target - mean) <= 1.96 * sd)

    def puzzle_macro(losses: dict[str, float]) -> tuple[float, dict[str, Any]]:
        score = E.score_position_losses(losses, method_balanced=True)
        if len(score["puzzles"]) != 1:
            raise ValueError("one outer fold must contain exactly one held puzzle")
        row = next(iter(score["puzzles"].values()))
        return float(row["L"]), row

    crps, crps_detail = puzzle_macro(crps_losses)
    delta_mae, delta_detail = puzzle_macro(delta_losses)
    cov68, _ = puzzle_macro(coverage68)
    cov95, _ = puzzle_macro(coverage95)
    return {
        "crps": crps,
        "signed_delta_mae": delta_mae,
        "coverage68": cov68,
        "coverage95": cov95,
        "n_qualified_positions": len(crps_losses),
        "n_registered_prediction_keys_expected": len(expected_keys),
        "n_registered_prediction_keys_observed": len(predicted_keys),
        "registered_prediction_coverage": registered_coverage,
        "failure_rate": failure_rate,
        "n_unexpected_prediction_keys": len(predicted_keys - expected_keys),
        "crps_methods": crps_detail["methods"],
        "delta_mae_methods": delta_detail["methods"],
    }


def run_fold(
    univ: M2Universe,
    fold: Any,
    all_records: list[Any],
    device: str,
    out_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    train_set = set(fold.train_puzzles)
    train_records = [r for r in all_records if r.puzzle in train_set]
    held_records = [r for r in all_records if r.puzzle == fold.held_puzzle]
    construct_ids = sorted({r.construct_id for r in train_records + held_records})
    ctx_cache = {cid: aligned_wt_ctx_tensors(univ, cid, device) for cid in construct_ids}
    result: dict[str, Any] = {
        "outer_fold": int(fold.outer_fold),
        "held_puzzle": fold.held_puzzle,
        "candidates": {},
    }
    for spec in CANDIDATES:
        print(
            f"[M2] fold={fold.outer_fold} held={fold.held_puzzle} "
            f"candidate={spec.model_id} start epochs={epochs}",
            flush=True,
        )
        torch.manual_seed(seed)
        model = AlignedDeltaModel(k_rank=spec.k_rank, sparse=spec.sparse).to(device)
        history = fit_candidate(
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
        prediction = predict_held(model, univ, held_records, ctx_cache, device)
        prediction_path = out_dir / f"m2_predictions_{spec.model_id}_fold{fold.outer_fold}.npz"
        np.savez_compressed(prediction_path, **prediction)
        checkpoint_path = out_dir / f"m2_checkpoint_{spec.model_id}_fold{fold.outer_fold}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        score = score_predictions(prediction, univ, held_records)
        result["candidates"][spec.model_id] = {
            "spec": asdict(spec),
            "epochs": epochs,
            "train_loss": history,
            "score": score,
            "prediction_artifact": str(prediction_path),
            "checkpoint": str(checkpoint_path),
        }
        print(
            f"[M2] fold={fold.outer_fold} candidate={spec.model_id} "
            f"crps={score['crps']:.8f} delta_mae={score['signed_delta_mae']:.8f} done",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def summarize(folds: list[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    candidate_ids = [spec.model_id for spec in CANDIDATES]
    summary: dict[str, Any] = {}
    for candidate in candidate_ids:
        crps = [row["candidates"][candidate]["score"]["crps"] for row in folds]
        delta = [row["candidates"][candidate]["score"]["signed_delta_mae"] for row in folds]
        summary[candidate] = {
            "mean_crps": float(np.mean(crps)),
            "mean_signed_delta_mae": float(np.mean(delta)),
            "per_fold_crps": crps,
            "per_fold_signed_delta_mae": delta,
        }
    b1 = summary["b1_rfd_direct_aligned"]
    for candidate, row in summary.items():
        row["crps_gain_vs_b1"] = b1["mean_crps"] - row["mean_crps"]
        row["signed_delta_mae_gain_vs_b1"] = b1["mean_signed_delta_mae"] - row["mean_signed_delta_mae"]
        row["directional_both_primary"] = (
            row["crps_gain_vs_b1"] >= 0 and row["signed_delta_mae_gain_vs_b1"] >= 0
        )
    return {
        "status": "ENGINEERING_SMOKE_ONLY" if smoke else "DEVELOPMENT_SCREEN",
        "n_folds": len(folds),
        "candidates": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    epochs = min(args.epochs, 3) if args.smoke else args.epochs

    univ = M2Universe(args.m2_csv)
    univ.build()
    all_records = univ.get_records()
    splits = build_split_v4(sorted({r.puzzle for r in all_records}), seed=20260813)
    selected_folds = {int(x) for x in args.folds.split(",") if x}
    folds = [fold for fold in splits["folds"] if fold.outer_fold in selected_folds]
    if not folds:
        raise ValueError("no requested outer folds")
    fold_results = []
    for fold in folds:
        fold_result = run_fold(
            univ,
            fold,
            all_records,
            device,
            args.out_dir,
            epochs,
            args.learning_rate,
            args.weight_decay,
            args.seed,
        )
        fold_results.append(fold_result)
        fold_path = args.out_dir / f"m2_fold_result_fold{fold.outer_fold}.json"
        fold_path.write_text(
            json.dumps(fold_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[M2] fold={fold.outer_fold} artifact={fold_path} complete", flush=True)
    result = {
        "schema_version": SCHEMA,
        "evidence_status": "ENGINEERING_SMOKE_ONLY" if args.smoke else "DEVELOPMENT_CONSUMED_SCREEN",
        "device": device,
        "seed": args.seed,
        "epochs": epochs,
        "folds": fold_results,
        "summary": summarize(fold_results, args.smoke),
        "qualification": {
            "external": "NOT_ACCESSED",
            "sota": "NOT_ESTABLISHED",
            "selection_warning": "two-fold smoke is not candidate selection",
        },
    }
    out = args.out_dir / "m2_model_rescue_result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "result": str(out), "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
