#!/usr/bin/env python3
"""DeltaFlow mechanism analysis (Task 9.3, interpretive, no gate).

Aligns the velocity network's learned pair representation with known
RNA base-pairing, using the frozen 28-channel pair semantics: the model
input encodes Watson-Crick/wobble pair identities only through sequence
one-hots, so any excess pair-state energy on canonical pairs reflects
what the denoiser's triangle/conv operators learned about base pairing.

Statistics per (fold, seed), all interpretive:
  - pair energy e_ij = squared channel norm of the final block's pair
    state at (i, j), restricted to WT-observed rows and columns and
    excluding the diagonal;
  - mean e on canonical Watson-Crick pairs (A-U, U-A, G-C, C-G), on
    wobble pairs (G-U, U-G), and on non-canonical pairs;
  - AUROC of e_ij discriminating canonical (WC + wobble) from
    non-canonical position pairs (rank-based, ties averaged);
  - the same three statistics computed on the raw input pair channels
    (the frozen base-pair one-hot block energy) as a reference floor.

The analysis reads model checkpoints and WT context only -- never held
targets or scores.  Output is a JSON artifact; nothing here feeds any
gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.deltaflow import (
    POINT_CHANNELS,
    POINT_REF_ONEHOT,
    PAIR_CHANNELS,
    TEACHER_WIDTH,
    DeltaFlowDenoiser,
    build_pair_channels,
)
from scripts.reactflow_delta.m2_universe_v1 import M2Universe


SCHEMA = "reactflow_delta.deltaflow_mechanism_analysis.v1"
WATSON_CRICK = {frozenset("AU"), frozenset("GC")}
Wobble = frozenset("GU")
DISTAL_K = 2


def _base_index(base: str) -> int:
    return {"A": 0, "C": 1, "G": 2, "U": 3}[base]


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(sorted_scores):
        stop = start
        while (
            stop + 1 < len(sorted_scores)
            and sorted_scores[stop + 1] == sorted_scores[start]
        ):
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    if positives == 0.0 or negatives == 0.0:
        return float("nan")
    return float((ranks[labels.astype(bool)].sum() - positives * (positives + 1) / 2.0)
                  / (positives * negatives))


def _pair_kind_matrix(sequence: str) -> np.ndarray:
    """K[i, j] in {0: non-canonical, 1: wobble, 2: Watson-Crick}."""
    length = len(sequence)
    kinds = np.zeros((length, length), dtype=np.int8)
    for i in range(length):
        for j in range(length):
            if i == j:
                continue
            pair = frozenset((sequence[i], sequence[j]))
            if pair in WATSON_CRICK:
                kinds[i, j] = 2
            elif pair == Wobble:
                kinds[i, j] = 1
    return kinds


def _construct_point_block(sequence: str, edit: int) -> torch.Tensor:
    length = len(sequence)
    point = np.zeros((1, length, POINT_CHANNELS), dtype=np.float32)
    positions = np.arange(length)
    distance = positions - edit
    point[0, :, 0] = distance / max(length - 1, 1)
    point[0, :, 1] = np.abs(distance) / max(length - 1, 1)
    point[0, :, 2] = np.log1p(np.abs(distance)) / math.log(max(length, 2))
    point[0, :, 3] = edit / max(length - 1, 1)
    point[0, :, 4] = positions / max(length - 1, 1)
    point[0, :, 5] = (positions == edit).astype(np.float32)
    for index, base in enumerate(sequence):
        point[0, index, POINT_REF_ONEHOT + _base_index(base)] = 1.0
        point[0, index, POINT_REF_ONEHOT + 4 + _base_index(base)] = 1.0
    point[0, :, 15] = 0.0
    point[0, :, 16] = 0.0
    point[0, :, 17] = 1.0
    return torch.from_numpy(point)


def _input_reference_energy(pair_input: torch.Tensor) -> np.ndarray:
    base_block = pair_input[0, :16].pow(2).sum(dim=0).cpu().numpy()
    return base_block


def analyze_checkpoint(
    checkpoint_path: Path,
    sequence: str,
    observed: np.ndarray,
    device: str,
) -> dict[str, float]:
    model = DeltaFlowDenoiser(diagonal=False).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["pair"] = output[1].detach()

    handle = model.blocks[-1].register_forward_hook(hook)
    length = len(sequence)
    edit = length // 2
    point = _construct_point_block(sequence, edit).to(device)
    mask = torch.from_numpy(observed.astype(bool)).unsqueeze(0).to(device)
    pair_input = build_pair_channels(point, mask)
    teacher = torch.zeros(1, length, TEACHER_WIDTH, device=device)
    stage1 = torch.zeros(1, length, device=device)
    state_r = torch.zeros(1, length, device=device)
    t = torch.full((1,), 0.5, device=device)
    with torch.no_grad():
        model(state_r, point, pair_input, teacher, stage1, mask, t)
    handle.remove()
    pair_state = captured["pair"][0]
    energy = pair_state.pow(2).sum(dim=0).cpu().numpy()

    kinds = _pair_kind_matrix(sequence)
    valid = np.zeros((length, length), dtype=bool)
    valid[np.ix_(observed.astype(bool), observed.astype(bool))] = True
    np.fill_diagonal(valid, False)
    np.fill_diagonal(kinds, 0)

    wc_mask = valid & (kinds == 2)
    wobble_mask = valid & (kinds == 1)
    non_mask = valid & (kinds == 0)
    canonical_mask = wc_mask | wobble_mask
    labels = canonical_mask[valid].astype(np.int8)
    energy_valid = energy[valid]
    auroc = _auroc(energy_valid, labels)

    reference = _input_reference_energy(pair_input)
    reference_valid = reference[valid]
    reference_auroc = _auroc(reference_valid, labels)

    return {
        "pair_energy_watson_crick_mean": float(energy[wc_mask].mean()) if wc_mask.any() else float("nan"),
        "pair_energy_wobble_mean": float(energy[wobble_mask].mean()) if wobble_mask.any() else float("nan"),
        "pair_energy_non_canonical_mean": float(energy[non_mask].mean()) if non_mask.any() else float("nan"),
        "pair_energy_canonical_over_non_ratio": (
            float(energy[canonical_mask].mean() / energy[non_mask].mean())
            if canonical_mask.any() and non_mask.any() and energy[non_mask].mean() > 0
            else float("nan")
        ),
        "auroc_canonical_vs_non": auroc,
        "reference_auroc_canonical_vs_non": reference_auroc,
        "n_valid_pairs": int(valid.sum()),
        "n_watson_crick": int(wc_mask.sum()),
        "n_wobble": int(wobble_mask.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", action="append", required=True,
                        help="seed=DIR containing deltaflow_candidate_flow_fold{f}_seed{s}.pt")
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-constructs", type=int, default=8,
                        help="number of WT constructs to analyze per fold")
    args = parser.parse_args(argv)

    experiment_dirs = {}
    for raw in args.experiment_dir:
        seed_text, _, path_text = raw.partition("=")
        experiment_dirs[int(seed_text)] = Path(path_text)
    univ = M2Universe(args.m2_csv)
    univ.build()
    construct_ids = sorted(univ.constructs)
    step = max(len(construct_ids) // max(args.n_constructs, 1), 1)
    selected = construct_ids[::step][: args.n_constructs]

    rows: list[dict[str, Any]] = []
    for seed, directory in sorted(experiment_dirs.items()):
        for fold in range(20):
            checkpoint = directory / f"deltaflow_candidate_flow_fold{fold}_seed{seed}.pt"
            if not checkpoint.is_file():
                continue
            for construct_id in selected:
                construct = univ.get_construct(construct_id)
                sequence = construct.sequence.replace("T", "U")
                observed = construct.wt_observed.astype(bool)
                if observed.sum() < DISTAL_K + 2:
                    continue
                stats = analyze_checkpoint(checkpoint, sequence, observed, args.device)
                stats["seed"] = seed
                stats["outer_fold"] = fold
                stats["construct_id"] = construct_id
                rows.append(stats)
    if not rows:
        raise RuntimeError("no checkpoints analyzed")

    def _median(field: str) -> float:
        values = [row[field] for row in rows if math.isfinite(row[field])]
        return float(np.median(values)) if values else float("nan")

    summary = {
        "schema_version": SCHEMA,
        "n_analyzed": len(rows),
        "median": {
            field: _median(field)
            for field in (
                "pair_energy_watson_crick_mean",
                "pair_energy_wobble_mean",
                "pair_energy_non_canonical_mean",
                "pair_energy_canonical_over_non_ratio",
                "auroc_canonical_vs_non",
                "reference_auroc_canonical_vs_non",
            )
        },
        "interpretation_note": (
            "Interpretive only; excluded from all gates. The reference AUROC "
            "reflects the frozen base-pair one-hot input channels; the model "
            "AUROC reflects the learned pair state after the final block."
        ),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_analyzed": summary["n_analyzed"],
                "median_auroc": summary["median"]["auroc_canonical_vs_non"],
                "median_reference_auroc": summary["median"]["reference_auroc_canonical_vs_non"],
                "result": str(args.out_json),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
