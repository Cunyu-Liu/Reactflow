#!/usr/bin/env python3
"""merge_p2_v3_shards: combine per-shard p2_v3_scores.json into ONE 20-fold result.

The 20 outer folds are independent, so run_p2_v3 can be sharded across GPUs
(4-5 folds per shard). Each shard writes its own p2_v3_scores.json containing
per-puzzle method-balanced L for all 6 models (puzzle -> L). This script:
  1. merges model_puzzle_L across shards (every puzzle appears exactly once),
  2. recomputes the 6 frozen paired contrasts (per-puzzle D_p, mean, 20-puzzle
     t-CI, exhaustive sign-flip, leave-one-puzzle influence),
  3. merges the per-fold rank-selection ledger,
  4. writes p2_v3_scores_merged.json.

This is a read-only aggregator; it does NOT rerun any model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.reactflow_delta.p2_learnability import (
    puzzle_level_ci20, studentized_sign_flip, leave_one_puzzle_influence,
)
from scripts.reactflow_delta.run_p2_v3 import (
    FAST_MODELS, RANK0_ID, RANKPOS_ID,
)

SCHEMA = "reactflow_delta.run_p2_v3.merged.v1"

CONTRASTS = [
    ("reg_direct", "zero", "Direct(ridge) vs WT-anchor"),
    ("reg_direct", "train_median", "Direct(ridge) vs train-median"),
    ("nonlinear", "zero", "Direct(MLP) vs WT-anchor"),
    ("nonlinear", "train_median", "Direct(MLP) vs train-median"),
    (RANK0_ID, "reg_direct", "RFD-Direct(K_rank=0) vs ridge"),
    (RANKPOS_ID, RANK0_ID, "selected-rank vs K_rank=0 (main null)"),
]

ALL_MODELS = FAST_MODELS + [RANK0_ID, RANKPOS_ID]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    model_l: dict[str, dict[str, float]] = {m: {} for m in ALL_MODELS}
    selection_ledger: dict = {}
    folds_run: list[int] = []
    held_to_fold: dict[str, int] = {}
    shard_meta = {}
    for d in args.shard_dirs:
        p = Path(d) / "p2_v3_scores.json"
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        doc = json.loads(p.read_text())
        for m in ALL_MODELS:
            if m in doc.get("model_puzzle_L", {}):
                model_l[m].update(doc["model_puzzle_L"][m])
        selection_ledger.update(doc.get("selection_ledger", {}))
        folds_run.extend(int(f) for f in doc.get("folds_run", []))
        shard_meta[str(p)] = {"folds": doc.get("folds_run", []),
                              "smoke": doc.get("smoke", False)}
    folds_run = sorted(folds_run)

    effects_out = {}
    for cand, base, label in CONTRASTS:
        # per-puzzle D_p = L_baseline - L_candidate (positive => candidate better)
        puzzle_effects = {}
        for fid in folds_run:
            held = _held_for_fold(fid)
            if held is None:
                continue
            lc = model_l[cand].get(held); lb = model_l[base].get(held)
            if lc is not None and lb is not None:
                puzzle_effects[held] = lb - lc
        eff_list = list(puzzle_effects.values())
        ci = puzzle_level_ci20(eff_list) if len(eff_list) >= 2 else {}
        effects_out[f"{cand}__vs__{base}"] = {
            "label": label,
            "per_puzzle": puzzle_effects,
            "mean": float(np.mean(eff_list)) if eff_list else None,
            "n": len(eff_list),
            "ci95": ci,
            "sign_flip": studentized_sign_flip(eff_list) if eff_list else None,
            "lop": leave_one_puzzle_influence(eff_list, list(puzzle_effects.keys()))
                   if eff_list else None,
        }

    result = {
        "schema_version": SCHEMA,
        "n_folds_total": len(folds_run),
        "folds_run": folds_run,
        "shard_sources": shard_meta,
        "model_puzzle_L": model_l,
        "effects": effects_out,
        "selection_ledger": selection_ledger,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    for k, v in effects_out.items():
        lo = v["ci95"].get("ci_low") if v["ci95"] else None
        hi = v["ci95"].get("ci_high") if v["ci95"] else None
        print(f"effect {k}: mean={v['mean']:.5f} n={v['n']} "
              f"ci95=[{lo}..{hi}] ci_low_gt_0={v['ci95'].get('ci_low_gt_0') if v['ci95'] else None}",
              flush=True)
    return 0


def _held_for_fold(fold_id: int) -> str | None:
    """Map outer fold index -> held puzzle (P01..P20) using build_split_v4 ordering."""
    from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
    puzzles = [f"P{i+1:02d}" for i in range(20)]
    split = build_split_v4(puzzles)
    for f in split["folds"]:
        if f.outer_fold == fold_id:
            return f.held_puzzle
    return None


if __name__ == "__main__":
    raise SystemExit(main())
