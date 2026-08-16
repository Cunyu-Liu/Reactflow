#!/usr/bin/env python3
"""Lightweight pair-level burden aggregation diagnosis for dev12.

Why position-level |pred| vs |true| is POSITIVE (+0.446) but pair-level burden
mean(|pred|) vs mean(|true|) is NEGATIVE (-0.28)?

Hypothesis: the MAE regression head SHRINKS all predictions toward the median/
mean of the target, so across pairs the predicted pair-burden is compressed to a
narrow range (low variance), destroying across-pair ordering while preserving
within-pair relative ordering.  This script checks:
  1. Distribution/spread of predicted pair-burden vs true pair-burden.
  2. Correlation of predicted pair-burden with true pair-burden.
  3. Whether a MONOTONIC per-pair rescale (rank-preserving) exists: if rank
     correlation of pred-burden vs true-burden is high but Pearson is low, then
     it is a pure scale problem solvable by calibration.
  4. Spearman rank of predicted position-magnitude WITHIN each pair (does the
     model rank positions correctly within a pair?).

Reads only the saved predictions.npz + split/canonical (no feature rebuild).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
if (_HERE.parents[2] / "src").exists():
    sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(Path.cwd() / "src"))

from b0x_baselines import _pair_scale  # noqa: E402
from b0x_data import load_pairs, split_groups  # noqa: E402


def _spear(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--predictions-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"validation"})
    groups = split_groups(pairs)
    val = groups.get("validation", [])

    preds_all = np.load(args.predictions_npz, allow_pickle=True)

    true_b, pred_b, ranks_within, n_elig = [], [], [], []
    pos_abs_pred, pos_abs_true = [], []
    for p in val:
        if p.pair_id not in preds_all.files:
            continue
        sc = np.asarray(preds_all[p.pair_id], dtype=np.float32)
        mask = np.asarray(p.mask, dtype=bool)
        d = np.asarray(p.delta, dtype=np.float64)
        scale = _pair_scale(p)
        elig = mask & np.isfinite(d) & (np.arange(len(mask)) < len(sc))
        if elig.sum() == 0:
            continue
        sc_e = np.abs(sc[elig])
        tr_e = np.abs(d[elig]) / scale
        true_b.append(float(tr_e.mean()))
        pred_b.append(float(sc_e.mean()))
        within = _spear(sc_e, tr_e)
        ranks_within.append(within)
        n_elig.append(int(elig.sum()))
        pos_abs_pred.extend(sc_e.tolist())
        pos_abs_true.extend(tr_e.tolist())

    true_b = np.array(true_b)
    pred_b = np.array(pred_b)
    ranks_within = np.array(ranks_within)
    n_elig = np.array(n_elig)

    report = {
        "n_pairs": int(len(true_b)),
        "true_burden": {"mean": float(true_b.mean()), "std": float(true_b.std()),
                        "median": float(np.median(true_b)),
                        "p10": float(np.percentile(true_b, 10)),
                        "p90": float(np.percentile(true_b, 90))},
        "pred_burden": {"mean": float(pred_b.mean()), "std": float(pred_b.std()),
                        "median": float(np.median(pred_b)),
                        "p10": float(np.percentile(pred_b, 10)),
                        "p90": float(np.percentile(pred_b, 90))},
        "pair_burden": {
            "spearman": _spear(true_b, pred_b),
            "pearson": (float(np.corrcoef(true_b, pred_b)[0, 1])
                        if len(true_b) > 2 else None),
            "pred_scaled_to_true_note": "if spear high but pearson low -> pure scale/calibration issue",
        },
        "within_pair_position_ranking": {
            "mean_spearman": float(np.nanmean(ranks_within)),
            "median_spearman": float(np.nanmedian(ranks_within)),
            "frac_positive": float(np.nanmean(ranks_within > 0)),
        },
        "position_level": {
            "n_pos": len(pos_abs_pred),
            "spearman_abs": _spear(pos_abs_pred, pos_abs_true),
        },
        "pred_compression_ratio": float(pred_b.std() / max(true_b.std(), 1e-9)),
    }
    (out_dir / "pair_aggregation_diagnosis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())