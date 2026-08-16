#!/usr/bin/env python3
"""dev12 pair-level magnitude calibration: find a burden proxy that flips sign.

Diagnosis (20260807): within-pair position ranking is GOOD (+0.49) but cross-pair
absolute magnitude is MISCALIBRATED (pred std 0.26 vs true 0.08, p90 0.75 vs 0.21),
so raw pair-burden mean(|pred|) is NEGATIVELY correlated with true burden.

This script compares multiple pair-level burden proxies (raw / rank / z-score /
quantile) and reports which one recovers a POSITIVE correlation with the true
burden mean(|delta|/scale).  Read-only, uses saved predictions.npz.

Proxies evaluated (per pair p, over eligible positions):
  raw                : mean(|m|)
  log                : mean(log1p(|m|))
  max                : max(|m|)
  global_rank        : mean(percentile_rank of |m| across ALL val positions)
  global_quantile    : same as global_rank (empirical CDF), alias
  within_pair_z_max  : max within-pair z-score of |m|
  within_pair_z_mean : mean within-pair z-score of |m| (top-20% only)
  within_rank_concentration : fraction of pair's positions in the top-half of
                              its own within-pair |m| ordering (captures "spread"
                              relative to pair's own scale) -- not a cross-pair
                              comparator alone, included for reference.
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


def _kendall(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n < 3:
        return float("nan")
    concord = 0.0
    discord = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if da * db > 0:
                concord += 1
            else:
                discord += 1
    denom = concord + discord
    return float((concord - discord) / denom) if denom > 0 else float("nan")


def _ndcg_at10(scores, labels):
    """NDCG@10 treating true burden (higher=better) as graded relevance."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    top = min(10, len(labels_sorted))
    if top == 0:
        return 0.0
    dcg = np.sum((2 ** labels_sorted[:top] - 1) / np.log2(np.arange(2, top + 2)))
    ideal = np.sort(labels)[::-1]
    idcg = np.sum((2 ** ideal[:top] - 1) / np.log2(np.arange(2, top + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _percentile_rank_col(values):
    """Global empirical CDF percentile of each element in a vector."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    order = np.argsort(values)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1) / n
    return ranks


# ---------------------------------------------------------------------------
# Generic pair-level magnitude proxies (model-agnostic).
#
# Input: dict pair_id -> np.array of per-position MAGNITUDE scores over eligible
#        positions (for regression models this is |delta_r_hat|; for folding
#        baselines the per-position score in whatever unit).
# Output: dict pair_id -> pair-level burden proxy scalar.
#
# The proxy of interest is `within_pair_z_max`: the max within-pair z-score of
# the per-position magnitudes.  It removes cross-pair absolute-scale drift
# (the root cause of the negative raw-burden correlation) while preserving the
# relative "hot-spot intensity" the model ranks well.  Exposed here so every
# model in the horizontal comparison can call the SAME proxy.
# ---------------------------------------------------------------------------
def _within_pair_z_max(magnitudes: np.ndarray) -> float:
    """Max within-pair z-score of position magnitudes (scale-invariant)."""
    m = np.asarray(magnitudes, dtype=float)
    if m.size == 0:
        return float("nan")
    mu, sd = m.mean(), m.std()
    if sd <= 1e-9:
        return 0.0
    return float(((m - mu) / sd).max())


def _within_pair_z_topk_mean(magnitudes: np.ndarray, frac: float = 0.2) -> float:
    """Mean within-pair z-score of the top `frac` magnitude positions."""
    m = np.asarray(magnitudes, dtype=float)
    if m.size == 0:
        return float("nan")
    mu, sd = m.mean(), m.std()
    if sd <= 1e-9:
        return 0.0
    z = (m - mu) / sd
    topk = max(1, int(np.ceil(frac * len(z))))
    return float(np.sort(z)[-topk:].mean())


def _within_rank_concentration(magnitudes: np.ndarray) -> float:
    """Fraction of positions in the top-half of within-pair magnitude."""
    m = np.asarray(magnitudes, dtype=float)
    if m.size <= 1:
        return float("nan")
    return float(np.mean(m >= np.median(m)))


def pair_magnitude_proxies(pair_magnitudes: dict,
                           topk_frac: float = 0.2) -> dict:
    """Compute all magnitude proxies for a dict of per-position magnitudes.

    Generic, model-agnostic: accepts any dict of pair_id -> per-position score
    array and returns a {proxy_name: {pair_id: scalar}} dict.  `within_pair_z_max`
    is the burden proxy of choice (see module docstring).
    """
    proxies = {name: {} for name in ("raw", "log", "max", "global_rank",
                                     "within_pair_z_max",
                                     "within_pair_z_top20",
                                     "within_rank_concentration")}
    pair_ids = list(pair_magnitudes.keys())

    # global percentile ranks over ALL pooled positions (for global_rank proxy)
    all_mag = np.concatenate([np.asarray(pair_magnitudes[pid], dtype=float)
                              for pid in pair_ids])
    glob_rank = _percentile_rank_col(all_mag)
    offset = 0
    grank_by_pair = {}
    for pid in pair_ids:
        n = len(pair_magnitudes[pid])
        grank_by_pair[pid] = glob_rank[offset:offset + n]
        offset += n

    for pid in pair_ids:
        m = np.asarray(pair_magnitudes[pid], dtype=float)
        proxies["raw"][pid] = float(m.mean())
        proxies["log"][pid] = float(np.log1p(m).mean())
        proxies["max"][pid] = float(m.max())
        proxies["global_rank"][pid] = float(grank_by_pair[pid].mean())
        proxies["within_pair_z_max"][pid] = _within_pair_z_max(m)
        proxies["within_pair_z_top20"][pid] = _within_pair_z_topk_mean(
            m, frac=topk_frac)
        proxies["within_rank_concentration"][pid] = \
            _within_rank_concentration(m)
    return proxies


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

    # Collect per-pair eligible |pred| magnitudes and true |delta|/scale
    true_b = []
    pair_magnitudes = {}  # pair_id -> np.array of |pred| over eligible
    pair_true_pos = {}    # pair_id -> np.array of |delta|/scale over eligible
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
        mag = np.abs(sc[elig])
        tr = np.abs(d[elig]) / scale
        pair_magnitudes[p.pair_id] = mag
        pair_true_pos[p.pair_id] = tr
        true_b.append(float(tr.mean()))

    pair_ids = list(pair_magnitudes.keys())
    true_b = np.array(true_b)

    # Use the generic, model-agnostic proxy suite.
    proxies = pair_magnitude_proxies(pair_magnitudes)

    # evaluate
    results = {}
    for name, proxy in proxies.items():
        pid_list = list(proxy.keys())
        vals = np.array([proxy[pid] for pid in pid_list])
        # align with true_b (same order as pair_ids)
        idx = [pair_ids.index(pid) for pid in pid_list]
        true_aligned = true_b[idx]
        spearman = _spear(vals, true_aligned)
        kendall = _kendall(vals, true_aligned)
        ndcg = _ndcg_at10(vals, true_aligned)
        results[name] = {
            "spearman": spearman,
            "kendall": kendall,
            "ndcg_at_10": ndcg,
            "n": int(len(vals)),
            "sign_fixed": bool(spearman > 0),
        }

    report = {
        "report": "m0x_dev12_pair_magnitude_calibration.v1",
        "model_dir": str(args.predictions_npz),
        "n_pairs": len(pair_ids),
        "true_burden": {"mean": float(true_b.mean()), "std": float(true_b.std())},
        "proxies": results,
        "note": "sign_fixed=True means the proxy's Spearman vs true burden > 0",
    }
    (out_dir / "calibration_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] -> {out_dir/'calibration_comparison.json'}", flush=True)
    print(f"{'proxy':32s} {'spear':>8s} {'kend':>8s} {'ndcg@10':>8s}  fixed")
    for name in ["raw", "log", "max", "global_rank",
                 "within_pair_z_max", "within_pair_z_top20",
                 "within_rank_concentration"]:
        r = results[name]
        print(f"{name:32s} {r['spearman']:8.4f} {r['kendall']:8.4f} "
              f"{r['ndcg_at_10']:8.4f}  {'YES' if r['sign_fixed'] else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())