#!/usr/bin/env python3
"""Pre-frozen M1 LOPO probe for outcome-blind ViennaRNA structure features.

The probe compares a fixed sequence-distance ridge with the same ridge plus two
features computed from the WT sequence only: MFE graph distance and ensemble
base-pair probability.  Ridge standardization and fitting use outer-train
puzzles only.  Held-puzzle outcomes are used only for method-balanced scoring.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import RNA
from scipy.stats import t as student_t

from scripts.reactflow_delta.m2_universe_v1 import M2Universe


SCHEMA = "reactflow_delta.model_rescue_structure_probe.v1"
ALPHABET = "ACGU"


def _mfe_pairs(dotbracket: str) -> list[tuple[int, int]]:
    stacks: dict[str, list[int]] = {"(": [], "[": [], "{": [], "<": []}
    close = {")": "(", "]": "[", "}": "{", ">": "<"}
    pairs: list[tuple[int, int]] = []
    for i, char in enumerate(dotbracket):
        if char in stacks:
            stacks[char].append(i)
        elif char in close:
            opener = close[char]
            if not stacks[opener]:
                raise ValueError("unbalanced dot-bracket structure")
            pairs.append((stacks[opener].pop(), i))
    if any(stacks.values()):
        raise ValueError("unbalanced dot-bracket structure")
    return pairs


def _all_pair_graph_distance(length: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    adjacency = [set() for _ in range(length)]
    for i in range(length - 1):
        adjacency[i].add(i + 1)
        adjacency[i + 1].add(i)
    for i, j in pairs:
        adjacency[i].add(j)
        adjacency[j].add(i)
    out = np.full((length, length), length + 1, dtype=np.int16)
    for source in range(length):
        out[source, source] = 0
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            next_distance = int(out[source, node]) + 1
            for receiver in adjacency[node]:
                if next_distance < out[source, receiver]:
                    out[source, receiver] = next_distance
                    queue.append(receiver)
    return out


def vienna_pair_features(sequence: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return MFE graph distance and ensemble direct-pair probabilities."""
    fc = RNA.fold_compound(sequence)
    mfe, _ = fc.mfe()
    fc.pf()
    raw_bpp = fc.bpp()
    length = len(sequence)
    bpp = np.zeros((length, length), dtype=np.float32)
    for i in range(length):
        for j in range(i + 1, length):
            value = float(raw_bpp[i + 1][j + 1])
            bpp[i, j] = value
            bpp[j, i] = value
    graph_distance = _all_pair_graph_distance(length, _mfe_pairs(mfe))
    return graph_distance, bpp, mfe


def _sequence_features(
    length: int,
    edit_pos: int,
    receiver_pos: np.ndarray,
    ref: str,
    alt: str,
    design: np.ndarray,
) -> np.ndarray:
    distance = receiver_pos - edit_pos
    abs_distance = np.abs(distance)
    n_receivers = len(receiver_pos)
    cols = [
        distance / max(length - 1, 1),
        abs_distance / max(length - 1, 1),
        np.log1p(abs_distance) / math.log(max(length, 2)),
        np.full(n_receivers, edit_pos / max(length - 1, 1)),
        receiver_pos / max(length - 1, 1),
        (receiver_pos == edit_pos).astype(float),
        design.astype(float),
    ]
    cols.extend(np.full(n_receivers, float(ref == base)) for base in ALPHABET)
    cols.extend(np.full(n_receivers, float(alt == base)) for base in ALPHABET)
    return np.column_stack(cols).astype(np.float32)


def _fit_standardized_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict[str, np.ndarray | float]:
    mean_x = X.mean(axis=0, dtype=np.float64)
    scale_x = X.std(axis=0, dtype=np.float64)
    scale_x = np.where(scale_x < 1e-8, 1.0, scale_x)
    mean_y = float(np.mean(y))
    z = (X.astype(np.float64) - mean_x) / scale_x
    centered_y = y.astype(np.float64) - mean_y
    lhs = z.T @ z + alpha * np.eye(z.shape[1])
    coef = np.linalg.solve(lhs, z.T @ centered_y)
    return {"mean_x": mean_x, "scale_x": scale_x, "mean_y": mean_y, "coef": coef}


def _predict_standardized_ridge(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    z = (X.astype(np.float64) - model["mean_x"]) / model["scale_x"]
    return model["mean_y"] + z @ model["coef"]


def _ridge_sufficient_statistics(X: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    x = X.astype(np.float64)
    y = targets.astype(np.float64)
    return {
        "n": int(len(x)),
        "sum_x": x.sum(axis=0),
        "sum_x2": (x**2).sum(axis=0),
        "xtx": x.T @ x,
        "sum_y": y.sum(axis=0),
        "xty": x.T @ y,
    }


def _subtract_statistics(total: dict[str, Any], held: dict[str, Any]) -> dict[str, Any]:
    return {key: total[key] - held[key] for key in total}


def _sum_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot sum an empty statistics list")
    result: dict[str, Any] = {}
    for key in rows[0]:
        first = rows[0][key]
        total = first.copy() if hasattr(first, "copy") else first
        for row in rows[1:]:
            total = total + row[key]
        result[key] = total
    return result


def _fit_ridge_from_statistics(stats: dict[str, Any], alpha: float = 1.0) -> dict[str, Any]:
    n = int(stats["n"])
    mean_x = stats["sum_x"] / n
    variance_x = np.maximum(stats["sum_x2"] / n - mean_x**2, 0.0)
    scale_x = np.sqrt(variance_x)
    scale_x = np.where(scale_x < 1e-8, 1.0, scale_x)
    mean_y = stats["sum_y"] / n
    centered_xtx = stats["xtx"] - n * np.outer(mean_x, mean_x)
    centered_xty = stats["xty"] - np.outer(stats["sum_x"], mean_y)
    ztz = centered_xtx / np.outer(scale_x, scale_x)
    zty = centered_xty / scale_x[:, None]
    coef = np.linalg.solve(ztz + alpha * np.eye(ztz.shape[0]), zty)
    return {"mean_x": mean_x, "scale_x": scale_x, "mean_y": mean_y, "coef": coef}


def _method_balanced_mae(y: np.ndarray, pred: np.ndarray, methods: np.ndarray) -> float:
    losses = []
    for method in np.unique(methods):
        mask = methods == method
        losses.append(float(np.mean(np.abs(y[mask] - pred[mask]))))
    return float(np.mean(losses))


def _paired_summary(effects: list[float]) -> dict[str, Any]:
    arr = np.asarray(effects, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        lo = hi = float("nan")
    else:
        half = float(student_t.ppf(0.975, n - 1) * arr.std(ddof=1) / math.sqrt(n))
        lo, hi = mean - half, mean + half
    return {
        "n": n,
        "mean_gain": mean,
        "ci95": [lo, hi],
        "positive_puzzles": int((arr > 0).sum()),
    }


def resolve_structure_gate(
    signed: dict[str, Any],
    magnitude: dict[str, Any],
    signed_baseline: float,
    magnitude_baseline: float,
) -> dict[str, Any]:
    summaries = {"signed_delta": signed, "absolute_delta": magnitude}
    baselines = {"signed_delta": signed_baseline, "absolute_delta": magnitude_baseline}
    target_pass: dict[str, bool] = {}
    guardrails: dict[str, bool] = {}
    for target, row in summaries.items():
        other = "absolute_delta" if target == "signed_delta" else "signed_delta"
        target_pass[target] = row["ci95"][0] > 0 and row["positive_puzzles"] >= 12
        guardrails[target] = summaries[other]["mean_gain"] >= -0.005 * baselines[other]
    eligible = any(target_pass[t] and guardrails[t] for t in target_pass)
    return {
        "target_pass": target_pass,
        "other_target_guardrail": guardrails,
        "status": "STRUCT_DELTA_ELIGIBLE_M2" if eligible else "STRUCT_DELTA_EXCLUDED_M2",
    }


def _build_puzzle_data(univ: M2Universe) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    structure_cache: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    rows: dict[str, dict[str, list[Any]]] = {}
    methods = sorted({r.method for r in univ.get_records()})
    method_to_id = {method: i for i, method in enumerate(methods)}

    for rec in univ.get_records():
        construct = univ.get_construct(rec.construct_id)
        sequence = construct.sequence
        if sequence not in structure_cache:
            structure_cache[sequence] = vienna_pair_features(sequence)
        graph_distance, bpp, _ = structure_cache[sequence]
        target, _ = univ.mutant_full_profile(
            rec.wt_id, rec.design_pos, rec.ref, rec.alt
        )
        if target is None:
            continue
        qualified = construct.wt_observed & np.isfinite(target)
        receiver = np.flatnonzero(qualified)
        if receiver.size == 0:
            continue
        length = len(sequence)
        seq_x = _sequence_features(
            length,
            rec.full_pos,
            receiver,
            rec.ref,
            rec.alt,
            construct.region_map[receiver] == "design_region",
        )
        struct_x = np.column_stack(
            [
                graph_distance[rec.full_pos, receiver] / max(length - 1, 1),
                bpp[rec.full_pos, receiver],
            ]
        ).astype(np.float32)
        signed = target[receiver].astype(np.float32) - construct.wt_reactivity[receiver].astype(np.float32)
        pz = rec.puzzle
        if pz not in rows:
            rows[pz] = {"x_seq": [], "x_struct": [], "signed": [], "method": []}
        rows[pz]["x_seq"].append(seq_x)
        rows[pz]["x_struct"].append(struct_x)
        rows[pz]["signed"].append(signed)
        rows[pz]["method"].append(np.full(receiver.size, method_to_id[rec.method], dtype=np.int8))

    data: dict[str, dict[str, np.ndarray]] = {}
    for puzzle, value in rows.items():
        data[puzzle] = {
            "x_seq": np.concatenate(value["x_seq"], axis=0),
            "x_struct": np.concatenate(value["x_struct"], axis=0),
            "signed": np.concatenate(value["signed"], axis=0),
            "method": np.concatenate(value["method"], axis=0),
        }
    metadata = {
        "n_unique_sequences": len(structure_cache),
        "methods": methods,
        "sequence_feature_count": next(iter(data.values()))["x_seq"].shape[1],
        "structure_feature_count": 2,
    }
    return data, metadata


def run_probe(m2_csv: Path, max_folds: int | None = None) -> dict[str, Any]:
    univ = M2Universe(m2_csv)
    universe = univ.build()
    data, metadata = _build_puzzle_data(univ)
    puzzles = sorted(data)
    if max_folds is not None:
        held_puzzles = puzzles[:max_folds]
    else:
        held_puzzles = puzzles

    stats: dict[str, dict[str, dict[str, Any]]] = {}
    for puzzle, value in data.items():
        targets = np.column_stack([value["signed"], np.abs(value["signed"])])
        full = np.concatenate([value["x_seq"], value["x_struct"]], axis=1)
        stats[puzzle] = {
            "sequence": _ridge_sufficient_statistics(value["x_seq"], targets),
            "structure": _ridge_sufficient_statistics(full, targets),
        }
    total_stats = {
        family: _sum_statistics([stats[p][family] for p in puzzles])
        for family in ["sequence", "structure"]
    }

    rows = []
    for held in held_puzzles:
        held_data = data[held]
        x_seq_held = held_data["x_seq"]
        x_full_held = np.concatenate([held_data["x_seq"], held_data["x_struct"]], axis=1)
        signed_held = held_data["signed"]
        abs_held = np.abs(signed_held)
        methods = held_data["method"]

        seq_model = _fit_ridge_from_statistics(
            _subtract_statistics(total_stats["sequence"], stats[held]["sequence"])
        )
        full_model = _fit_ridge_from_statistics(
            _subtract_statistics(total_stats["structure"], stats[held]["structure"])
        )
        seq_predictions = _predict_standardized_ridge(seq_model, x_seq_held)
        full_predictions = _predict_standardized_ridge(full_model, x_full_held)

        fold: dict[str, Any] = {"puzzle": held, "n_positions": int(len(signed_held))}
        for target_index, (target_name, y_held) in enumerate([
            ("signed_delta", signed_held),
            ("absolute_delta", abs_held),
        ]):
            seq_pred = seq_predictions[:, target_index]
            full_pred = full_predictions[:, target_index]
            seq_mae = _method_balanced_mae(y_held, seq_pred, methods)
            full_mae = _method_balanced_mae(y_held, full_pred, methods)
            fold[target_name] = {
                "sequence_mae": seq_mae,
                "structure_mae": full_mae,
                "gain": seq_mae - full_mae,
            }
        rows.append(fold)

    signed_summary = _paired_summary([x["signed_delta"]["gain"] for x in rows])
    magnitude_summary = _paired_summary([x["absolute_delta"]["gain"] for x in rows])
    signed_baseline = float(np.mean([x["signed_delta"]["sequence_mae"] for x in rows]))
    magnitude_baseline = float(np.mean([x["absolute_delta"]["sequence_mae"] for x in rows]))
    gate = resolve_structure_gate(signed_summary, magnitude_summary, signed_baseline, magnitude_baseline)
    return {
        "schema_version": SCHEMA,
        "evidence_status": "DEVELOPMENT_CONSUMED_POST_HOC_PROBE",
        "input": str(m2_csv),
        "split": "split_v4_lopo_puzzle",
        "learner": "train-only standardized ridge alpha=1",
        "features": {
            "sequence_only": [
                "signed_sequence_distance",
                "absolute_sequence_distance",
                "log_absolute_sequence_distance",
                "edit_position",
                "receiver_position",
                "same_site",
                "receiver_design_region",
                "mutation_ref_one_hot",
                "mutation_alt_one_hot",
            ],
            "structure_additions": [
                "viennarna_mfe_graph_distance",
                "viennarna_ensemble_direct_base_pair_probability",
            ],
        },
        "counts": {
            "n_puzzles_total": len(puzzles),
            "n_puzzles_scored": len(rows),
            "n_registered_snv_mutants": universe["n_registered_snv_mutants"],
            **metadata,
        },
        "summary": {
            "signed_delta": {**signed_summary, "sequence_mae": signed_baseline},
            "absolute_delta": {**magnitude_summary, "sequence_mae": magnitude_baseline},
        },
        "gate": gate,
        "per_puzzle": rows,
        "qualification": {
            "mechanism": "NOT_ESTABLISHED",
            "external": "NOT_ASSESSED",
            "allowed_interpretation": "screening evidence for whether StructDelta may enter M2",
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ReactFlow-Delta M1 ViennaRNA structure probe v1",
        "",
        f"Evidence status: `{result['evidence_status']}`.",
        "",
        "## Decision",
        "",
        f"- Gate: `{result['gate']['status']}`.",
    ]
    for target in ["signed_delta", "absolute_delta"]:
        row = result["summary"][target]
        lines.append(
            f"- {target}: mean method-balanced MAE gain = {row['mean_gain']:+.8f}, "
            f"95% CI [{row['ci95'][0]:+.8f}, {row['ci95'][1]:+.8f}], "
            f"positive puzzles = {row['positive_puzzles']}/{row['n']}."
        )
    lines += [
        "",
        "## Per-puzzle results",
        "",
        "| puzzle | signed gain | magnitude gain | positions |",
        "|---|---:|---:|---:|",
    ]
    for row in result["per_puzzle"]:
        lines.append(
            f"| {row['puzzle']} | {row['signed_delta']['gain']:+.8f} | "
            f"{row['absolute_delta']['gain']:+.8f} | {row['n_positions']} |"
        )
    lines += [
        "",
        "This fixed probe can qualify a structure-aware candidate for screening. It cannot establish an RNA mechanism or external generalization.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--max-folds", type=int)
    args = parser.parse_args(argv)
    result = run_probe(args.m2_csv, args.max_folds)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": "PASS", "gate": result["gate"]["status"], "out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
