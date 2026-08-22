#!/usr/bin/env python3
"""Build the M1 failure atlas from the frozen P2-v3 prediction-only OOF ledgers.

This is a diagnostic over already-consumed development outcomes.  It does not
train a model and does not read any external outcome.  The script joins targets
only after loading target-free OOF predictions, then reports the frozen
method-balanced puzzle-macro estimand for CRPS and signed mutation-effect error.

The CRPS difference between rank zero and the selected positive rank is split
into mean and scale contributions with a two-order Shapley decomposition.  This
avoids attributing uncertainty calibration gains to the low-rank mean operator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.run_p3_lrso_v3 import _mixture_crps_vec


SCHEMA = "reactflow_delta.model_rescue_failure_atlas.v1"
METRICS = [
    "rank0_crps",
    "rankpos_crps",
    "low_rank_crps_gain",
    "mean_crps_gain_shapley",
    "scale_crps_gain_shapley",
    "wt_signed_delta_abs_error",
    "rank0_signed_delta_abs_error",
    "rankpos_signed_delta_abs_error",
    "rank0_delta_mae_gain_vs_wt",
    "rankpos_delta_mae_gain_vs_wt",
    "low_rank_delta_mae_gain",
    "mutant_mae_rank0",
    "mutant_mae_rankpos",
    "low_rank_residual_abs",
    "low_rank_residual_sq",
    "rank0_predicted_delta_sq",
    "true_delta_abs",
    "coverage68_rank0",
    "coverage68_rankpos",
    "coverage95_rank0",
    "coverage95_rankpos",
]
WMAE_LOSS_COLUMNS = [
    "wt_signed_delta_abs_error",
    "rank0_signed_delta_abs_error",
    "rankpos_signed_delta_abs_error",
]


def magnitude_bin(abs_delta: np.ndarray) -> np.ndarray:
    """Fixed, interpretable response bins; thresholds are not data quantiles."""
    return np.select(
        [abs_delta <= 0.05, abs_delta <= 0.20, abs_delta <= 0.50],
        ["near_zero_le_0.05", "small_0.05_0.20", "medium_0.20_0.50"],
        default="tail_gt_0.50",
    )


def distance_bin(signed_distance: np.ndarray) -> np.ndarray:
    d = np.abs(signed_distance)
    return np.select(
        [d == 0, d <= 5, d <= 20, d <= 50],
        ["edit_site_0", "near_1_5", "mid_6_20", "far_21_50"],
        default="distal_gt_50",
    )


def error_bin(error: np.ndarray) -> np.ndarray:
    return np.select(
        [~np.isfinite(error), error <= 0.10, error <= 0.25, error <= 0.50],
        ["missing", "le_0.10", "0.10_0.25", "0.25_0.50"],
        default="gt_0.50",
    )


def missingness_bin(frac: np.ndarray) -> np.ndarray:
    return np.select(
        [frac <= 0.01, frac <= 0.05, frac <= 0.15],
        ["le_1pct", "1_5pct", "5_15pct"],
        default="gt_15pct",
    )


def mixture_moments(locs: list[np.ndarray], scales: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    loc_stack = np.stack(locs)
    scale_stack = np.stack(scales)
    mean = loc_stack.mean(axis=0)
    second = (scale_stack**2 + loc_stack**2).mean(axis=0)
    sd = np.sqrt(np.maximum(second - mean**2, 1e-12))
    return mean, sd


def shapley_mean_scale_crps(
    loc0: list[np.ndarray],
    scale0: list[np.ndarray],
    loc1: list[np.ndarray],
    scale1: list[np.ndarray],
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    """Two-factor Shapley split of CRPS(rank0)-CRPS(rank-positive)."""
    c00 = _mixture_crps_vec(loc0, scale0, y)
    c11 = _mixture_crps_vec(loc1, scale1, y)
    c10 = _mixture_crps_vec(loc1, scale0, y)
    c01 = _mixture_crps_vec(loc0, scale1, y)
    mean_gain = 0.5 * ((c00 - c10) + (c01 - c11))
    scale_gain = 0.5 * ((c00 - c01) + (c10 - c11))
    return {
        "rank0": c00,
        "rankpos": c11,
        "total_gain": c00 - c11,
        "mean_gain": mean_gain,
        "scale_gain": scale_gain,
    }


def _fold_from_name(path: Path) -> int:
    match = re.search(r"_fold(\d+)\.npz$", path.name)
    if not match:
        raise ValueError(f"cannot parse fold from {path}")
    return int(match.group(1))


def discover_oof_pairs(root: Path) -> list[tuple[int, Path, Path]]:
    rank0 = {_fold_from_name(p): p for p in root.glob("shard_gpu*/p2_v3_oof_predictions_rank0_fold*.npz")}
    rankpos = {_fold_from_name(p): p for p in root.glob("shard_gpu*/p2_v3_oof_predictions_rankpos_fold*.npz")}
    if rank0.keys() != rankpos.keys():
        raise ValueError("rank0/rank-positive OOF fold sets differ")
    if not rank0:
        raise FileNotFoundError(f"no OOF ledgers found below {root}")
    return [(fold, rank0[fold], rankpos[fold]) for fold in sorted(rank0)]


def load_seed_mixture(path: Path) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    z = np.load(path, allow_pickle=True)
    keys = np.asarray(z["keys"], dtype=object)
    loc = np.asarray(z["loc"], dtype=float)
    scale = np.asarray(z["scale"], dtype=float)
    seed = np.asarray(z["seed"], dtype=int)
    seeds = sorted(np.unique(seed).tolist())
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError(f"{path}: expected seeds 0..4, got {seeds}")
    base = keys[seed == seeds[0]]
    locs: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for value in seeds:
        mask = seed == value
        if not np.array_equal(keys[mask], base):
            raise ValueError(f"{path}: biological key order differs across seeds")
        lv = loc[mask]
        sv = scale[mask]
        if not np.all(np.isfinite(lv)) or not np.all(np.isfinite(sv)) or np.any(sv <= 0):
            raise ValueError(f"{path}: non-finite location or non-positive scale")
        locs.append(lv)
        scales.append(sv)
    return base, locs, scales


def _target_metadata(
    univ: M2Universe,
    keys: np.ndarray,
    record_index: dict[tuple[str, int, str, str], Any],
    profile_cache: dict[tuple[str, int, str, str], tuple[np.ndarray | None, np.ndarray | None]],
) -> pd.DataFrame:
    n = len(keys)
    target = np.full(n, np.nan)
    target_error = np.full(n, np.nan)
    wt = np.full(n, np.nan)
    wt_error = np.full(n, np.nan)
    distance = np.zeros(n, dtype=int)
    receiver_region = np.empty(n, dtype=object)
    puzzle = np.empty(n, dtype=object)
    method = np.empty(n, dtype=object)
    construct = np.empty(n, dtype=object)
    construct_missing_fraction = np.zeros(n, dtype=float)

    for i, key in enumerate(keys):
        parts = str(key).split("|")
        if len(parts) < 7:
            raise ValueError(f"malformed biological key: {key}")
        pz, meth, cid = parts[1], parts[2], parts[3]
        edit_pos, receiver_pos = int(parts[4]), int(parts[6])
        ref, alt = parts[5].split(">", 1)
        rec_key = (cid, edit_pos, ref, alt)
        rec = record_index.get(rec_key)
        if rec is None:
            raise KeyError(f"registered mutation absent for {key}")
        if rec_key not in profile_cache:
            profile_cache[rec_key] = univ.mutant_full_profile(
                rec.wt_id, rec.design_pos, rec.ref, rec.alt
            )
        tprof, eprof = profile_cache[rec_key]
        c = univ.get_construct(cid)
        target[i] = np.nan if tprof is None else float(tprof[receiver_pos])
        if eprof is not None:
            target_error[i] = float(eprof[receiver_pos])
        wt[i] = float(c.wt_reactivity[receiver_pos])
        wt_error[i] = float(c.wt_error[receiver_pos])
        distance[i] = receiver_pos - edit_pos
        receiver_region[i] = str(c.region_map[receiver_pos])
        puzzle[i] = pz
        method[i] = meth
        construct[i] = cid
        construct_missing_fraction[i] = 1.0 - float(c.wt_observed.mean())

    qualified = np.isfinite(target) & np.isfinite(wt)
    return pd.DataFrame(
        {
            "biological_scoring_key": keys,
            "qualified": qualified,
            "target": target,
            "target_error": target_error,
            "wt": wt,
            "wt_error": wt_error,
            "signed_distance": distance,
            "receiver_region": receiver_region,
            "puzzle": puzzle,
            "method": method,
            "construct": construct,
            "construct_missing_fraction": construct_missing_fraction,
        }
    )


def _cell_table(frame: pd.DataFrame, stratum: np.ndarray | pd.Series) -> pd.DataFrame:
    work = frame.copy()
    work["stratum"] = np.asarray(stratum, dtype=object)
    group_cols = ["stratum", "puzzle", "method"]
    cell = work.groupby(group_cols, observed=True)[METRICS].mean().reset_index()

    valid_weight = np.isfinite(work["target_error"]) & np.isfinite(work["wt_error"])
    variance = work["target_error"] ** 2 + work["wt_error"] ** 2 + 0.05**2
    work["reliability_weight"] = np.where(valid_weight, 1.0 / variance, 0.0)
    weighted = work[group_cols + ["reliability_weight"]].copy()
    for loss in WMAE_LOSS_COLUMNS:
        weighted[f"weighted_{loss}"] = work[loss] * work["reliability_weight"]
    sums = weighted.groupby(group_cols, observed=True).sum().reset_index()
    for loss in WMAE_LOSS_COLUMNS:
        sums[f"wmae_{loss}"] = np.where(
            sums["reliability_weight"] > 0,
            sums[f"weighted_{loss}"] / sums["reliability_weight"],
            np.nan,
        )
    keep = group_cols + [f"wmae_{loss}" for loss in WMAE_LOSS_COLUMNS]
    return cell.merge(sums[keep], on=group_cols, how="left")


def _summarize_cells(cells: pd.DataFrame, counts: dict[str, int]) -> dict[str, Any]:
    numeric = [c for c in cells.columns if c not in {"stratum", "puzzle", "method"}]
    puzzle = cells.groupby(["stratum", "puzzle"], observed=True)[numeric].mean().reset_index()
    out: dict[str, Any] = {}
    for stratum, rows in puzzle.groupby("stratum", observed=True):
        metric_means = rows[numeric].mean()
        out[str(stratum)] = {
            "n_positions": int(counts.get(str(stratum), 0)),
            "n_puzzles": int(rows["puzzle"].nunique()),
            "metrics": {k: float(v) for k, v in metric_means.items()},
            "positive_puzzles": {
                "low_rank_crps_gain": int((rows["low_rank_crps_gain"] > 0).sum()),
                "low_rank_delta_mae_gain": int((rows["low_rank_delta_mae_gain"] > 0).sum()),
                "rank0_delta_mae_gain_vs_wt": int((rows["rank0_delta_mae_gain_vs_wt"] > 0).sum()),
                "rankpos_delta_mae_gain_vs_wt": int((rows["rankpos_delta_mae_gain_vs_wt"] > 0).sum()),
            },
        }
    return out


def _puzzle_rows(cells_all: pd.DataFrame, correlations: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    numeric = [c for c in cells_all.columns if c not in {"stratum", "puzzle", "method"}]
    puzzle = cells_all.groupby("puzzle", observed=True)[numeric].mean().reset_index()
    rows = []
    for row in puzzle.to_dict(orient="records"):
        pz = str(row.pop("puzzle"))
        row = {k: float(v) for k, v in row.items()}
        row["puzzle"] = pz
        row.update(correlations[pz])
        rows.append(row)
    return sorted(rows, key=lambda x: x["puzzle"])


def _safe_float(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_float(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_float(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def build_failure_atlas(m2_csv: Path, oof_root: Path, max_folds: int | None = None) -> dict[str, Any]:
    univ = M2Universe(m2_csv)
    universe = univ.build()
    record_index = {
        (r.construct_id, r.design_pos, r.ref, r.alt): r
        for r in univ.get_records()
    }
    profile_cache: dict[tuple[str, int, str, str], tuple[np.ndarray | None, np.ndarray | None]] = {}
    pairs = discover_oof_pairs(oof_root)
    if max_folds is not None:
        pairs = pairs[:max_folds]

    cells_by_stratifier: dict[str, list[pd.DataFrame]] = defaultdict(list)
    counts_by_stratifier: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    correlations: dict[str, dict[str, float]] = {}
    n_prediction_rows = 0
    n_qualified_rows = 0

    for fold, rank0_path, rankpos_path in pairs:
        keys0, loc0, scale0 = load_seed_mixture(rank0_path)
        keys1, loc1, scale1 = load_seed_mixture(rankpos_path)
        if not np.array_equal(keys0, keys1):
            raise ValueError(f"fold {fold}: rank0/rank-positive key universes differ")
        metadata = _target_metadata(univ, keys0, record_index, profile_cache)
        qualified = metadata["qualified"].to_numpy(dtype=bool)
        if qualified.sum() == 0:
            raise ValueError(f"fold {fold}: no qualified target positions")
        y = metadata.loc[qualified, "target"].to_numpy(dtype=float)
        wt = metadata.loc[qualified, "wt"].to_numpy(dtype=float)
        loc0q = [x[qualified] for x in loc0]
        scale0q = [x[qualified] for x in scale0]
        loc1q = [x[qualified] for x in loc1]
        scale1q = [x[qualified] for x in scale1]

        split = shapley_mean_scale_crps(loc0q, scale0q, loc1q, scale1q, y)
        mean0, sd0 = mixture_moments(loc0q, scale0q)
        mean1, sd1 = mixture_moments(loc1q, scale1q)
        true_delta = y - wt
        pred_delta0 = mean0 - wt
        pred_delta1 = mean1 - wt
        residual = mean1 - mean0

        frame = metadata.loc[qualified].reset_index(drop=True)
        frame["true_delta"] = true_delta
        frame["rank0_crps"] = split["rank0"]
        frame["rankpos_crps"] = split["rankpos"]
        frame["low_rank_crps_gain"] = split["total_gain"]
        frame["mean_crps_gain_shapley"] = split["mean_gain"]
        frame["scale_crps_gain_shapley"] = split["scale_gain"]
        frame["wt_signed_delta_abs_error"] = np.abs(true_delta)
        frame["rank0_signed_delta_abs_error"] = np.abs(pred_delta0 - true_delta)
        frame["rankpos_signed_delta_abs_error"] = np.abs(pred_delta1 - true_delta)
        frame["rank0_delta_mae_gain_vs_wt"] = (
            frame["wt_signed_delta_abs_error"] - frame["rank0_signed_delta_abs_error"]
        )
        frame["rankpos_delta_mae_gain_vs_wt"] = (
            frame["wt_signed_delta_abs_error"] - frame["rankpos_signed_delta_abs_error"]
        )
        frame["low_rank_delta_mae_gain"] = (
            frame["rank0_signed_delta_abs_error"] - frame["rankpos_signed_delta_abs_error"]
        )
        frame["mutant_mae_rank0"] = np.abs(mean0 - y)
        frame["mutant_mae_rankpos"] = np.abs(mean1 - y)
        frame["low_rank_residual_abs"] = np.abs(residual)
        frame["low_rank_residual_sq"] = residual**2
        frame["rank0_predicted_delta_sq"] = pred_delta0**2
        frame["true_delta_abs"] = np.abs(true_delta)
        frame["coverage68_rank0"] = (np.abs(y - mean0) <= sd0).astype(float)
        frame["coverage68_rankpos"] = (np.abs(y - mean1) <= sd1).astype(float)
        frame["coverage95_rank0"] = (np.abs(y - mean0) <= 1.96 * sd0).astype(float)
        frame["coverage95_rankpos"] = (np.abs(y - mean1) <= 1.96 * sd1).astype(float)

        pz = str(frame["puzzle"].iloc[0])
        correlations[pz] = {
            "corr_low_rank_residual_true_delta": float(np.corrcoef(residual, true_delta)[0, 1]),
            "corr_rank0_rankpos_predicted_delta": float(np.corrcoef(pred_delta0, pred_delta1)[0, 1]),
            "low_rank_residual_energy_ratio_pooled": float(
                np.mean(residual**2) / max(np.mean(pred_delta0**2), 1e-12)
            ),
        }

        strata = {
            "all": np.repeat("all", len(frame)),
            "response_magnitude": magnitude_bin(np.abs(true_delta)),
            "signed_distance": distance_bin(frame["signed_distance"].to_numpy(dtype=int)),
            "receiver_region": frame["receiver_region"].to_numpy(dtype=object),
            "method": frame["method"].to_numpy(dtype=object),
            "wt_error": error_bin(frame["wt_error"].to_numpy(dtype=float)),
            "construct_wt_missingness": missingness_bin(
                frame["construct_missing_fraction"].to_numpy(dtype=float)
            ),
        }
        for name, labels in strata.items():
            labels = np.asarray(labels, dtype=object)
            cells_by_stratifier[name].append(_cell_table(frame, labels))
            unique, count = np.unique(labels, return_counts=True)
            for label, value in zip(unique, count):
                counts_by_stratifier[name][str(label)] += int(value)

        n_prediction_rows += len(keys0) * 5 * 2
        n_qualified_rows += int(qualified.sum())

    summaries: dict[str, Any] = {}
    concatenated_cells: dict[str, pd.DataFrame] = {}
    for name, tables in cells_by_stratifier.items():
        cells = pd.concat(tables, ignore_index=True)
        concatenated_cells[name] = cells
        summaries[name] = _summarize_cells(cells, counts_by_stratifier[name])

    all_metrics = summaries["all"]["all"]["metrics"]
    residual_energy_ratio = all_metrics["low_rank_residual_sq"] / max(
        all_metrics["rank0_predicted_delta_sq"], 1e-12
    )
    crps_gain = all_metrics["low_rank_crps_gain"]
    delta_gain_vs_wt = all_metrics["rankpos_delta_mae_gain_vs_wt"]
    if delta_gain_vs_wt <= 0 and crps_gain < 0.003:
        route = "DO_NOT_SCALE_LRSO_PRIORITIZE_ALIGNED_DIRECT_AND_SPARSE_DELTA"
    elif delta_gain_vs_wt <= 0:
        route = "LRSO_CALIBRATION_ONLY_NOT_MUTATION_EFFECT_MODEL"
    else:
        route = "LRSO_REMAINS_ELIGIBLE_FOR_CONTROLLED_M2_SCREEN"

    result = {
        "schema_version": SCHEMA,
        "evidence_status": "DEVELOPMENT_CONSUMED_DIAGNOSTIC_ONLY",
        "inputs": {
            "m2_csv": str(m2_csv),
            "oof_root": str(oof_root),
            "folds": [fold for fold, _, _ in pairs],
            "models": ["rfd_direct_rank0", "inner_selected_rank_positive"],
            "seeds": [0, 1, 2, 3, 4],
        },
        "counts": {
            "n_folds": len(pairs),
            "n_prediction_rows_loaded": n_prediction_rows,
            "n_unique_qualified_positions": n_qualified_rows,
            "n_registered_snv_mutants": universe["n_registered_snv_mutants"],
            "n_constructs": universe["n_constructs"],
        },
        "definitions": {
            "estimand": "position -> mutant -> puzzle-method cell -> method-balanced puzzle -> puzzle macro",
            "crps_decomposition": "two-order Shapley split over mixture locations and scales",
            "signed_delta": "mutant_reactivity - WT_reactivity at each receiver position",
            "wmae_sensitivity": "inverse(target_error^2 + WT_error^2 + 0.05^2), normalized within cell",
            "response_bins": ["<=0.05", "(0.05,0.20]", "(0.20,0.50]", ">0.50"],
        },
        "summary": summaries,
        "per_puzzle": _puzzle_rows(concatenated_cells["all"], correlations),
        "low_rank_residual_energy_ratio": residual_energy_ratio,
        "route_recommendation": route,
        "qualification": {
            "current_low_rank_claim": "SMALL_BUT_SIGNIFICANT_DEVELOPMENT_CRPS_ONLY",
            "signed_mutation_effect_claim": (
                "SUPPORTED_VS_WT" if delta_gain_vs_wt > 0 else "FAIL_VS_WT_ANCHOR"
            ),
            "sota": "NOT_ESTABLISHED",
            "external": "NOT_ASSESSED_BY_THIS_SCRIPT",
        },
    }
    return _safe_float(result)


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def render_markdown(result: dict[str, Any]) -> str:
    all_row = result["summary"]["all"]["all"]
    m = all_row["metrics"]
    pp = all_row["positive_puzzles"]
    magnitude = result["summary"]["response_magnitude"]
    lines = [
        "# ReactFlow-Delta M1 failure atlas v1",
        "",
        f"Evidence status: `{result['evidence_status']}`. This is a diagnostic over consumed development outcomes, not confirmation.",
        "",
        "## Decision",
        "",
        f"- Route: `{result['route_recommendation']}`.",
        f"- Selected-rank CRPS gain over rank zero: {_fmt(m['low_rank_crps_gain'])} ({pp['low_rank_crps_gain']}/20 puzzles positive).",
        f"- Of that gain, Shapley mean contribution is {_fmt(m['mean_crps_gain_shapley'])}; scale contribution is {_fmt(m['scale_crps_gain_shapley'])}.",
        f"- Rank-zero signed-delta MAE gain vs WT: {_fmt(m['rank0_delta_mae_gain_vs_wt'])} ({pp['rank0_delta_mae_gain_vs_wt']}/20 puzzles positive).",
        f"- Selected-rank signed-delta MAE gain vs WT: {_fmt(m['rankpos_delta_mae_gain_vs_wt'])} ({pp['rankpos_delta_mae_gain_vs_wt']}/20 puzzles positive).",
        f"- Low-rank signed-delta MAE gain over rank zero: {_fmt(m['low_rank_delta_mae_gain'])} ({pp['low_rank_delta_mae_gain']}/20 puzzles positive).",
        f"- Low-rank residual energy / rank-zero predicted-delta energy: {result['low_rank_residual_energy_ratio']:.6f}.",
        "",
        "## Response-magnitude strata",
        "",
        "| stratum | positions | CRPS gain | mean part | scale part | signed-delta gain vs WT | low-rank delta-MAE gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in magnitude.items():
        mm = row["metrics"]
        lines.append(
            f"| {name} | {row['n_positions']} | {_fmt(mm['low_rank_crps_gain'])} | "
            f"{_fmt(mm['mean_crps_gain_shapley'])} | {_fmt(mm['scale_crps_gain_shapley'])} | "
            f"{_fmt(mm['rankpos_delta_mae_gain_vs_wt'])} | {_fmt(mm['low_rank_delta_mae_gain'])} |"
        )
    lines += [
        "",
        "## Per-puzzle heterogeneity",
        "",
        "| puzzle | CRPS gain | mean part | scale part | rankpos delta gain vs WT | low-rank delta gain | residual energy ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_puzzle"]:
        lines.append(
            f"| {row['puzzle']} | {_fmt(row['low_rank_crps_gain'])} | "
            f"{_fmt(row['mean_crps_gain_shapley'])} | {_fmt(row['scale_crps_gain_shapley'])} | "
            f"{_fmt(row['rankpos_delta_mae_gain_vs_wt'])} | {_fmt(row['low_rank_delta_mae_gain'])} | "
            f"{row['low_rank_residual_energy_ratio_pooled']:.6f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The atlas may identify where the existing model fails and which component contributes to CRPS. It cannot establish prospective performance, external replication, mechanism, or SOTA. The structure probe is a separate pre-frozen LOPO analysis.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--max-folds", type=int)
    args = parser.parse_args(argv)
    result = build_failure_atlas(args.m2_csv, args.oof_root, args.max_folds)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": "PASS", "out_json": str(args.out_json), "out_md": str(args.out_md), "route": result["route_recommendation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
