#!/usr/bin/env python3
"""evaluate_deviation_v1.py — publication-block deviation-detection evaluation for the
full-spectrum response model.

Motivation (see docs/audits/reactflow_delta_method_reflection_20260812.md):
pooled WMAE skill vs the per-position median prior is *structurally saturated*: the
median prior already sits at the noise ceiling, so no architecture can show positive
pooled WMAE skill.  The model's real, non-saturated signal is *deviation detection* —
ranking which positions deviate from the per-position median prior.

This module measures that signal with the same statistical discipline as evaluate_v2:
the *publication* is the exchangeable block unit, and significance is a
publication-block permutation test.

Metrics:
  * spearman_signed : Spearman rank corr between predicted deviation (model - prior)
                      and true deviation (y - prior).
  * spearman_abs    : Spearman rank corr between |pred deviation| and |true deviation|.
  * auroc_abs       : AUROC of |pred deviation| for detecting |true deviation| above
                      the pooled median (>=0.5 = better than chance).
Significance for spearman_signed is a publication-block permutation: predicted
deviation blocks are permuted across publications (true y and baseline prior stay
fixed), breaking the model<->truth coupling under the null.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_rows(pred_path):
    rows = []
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _unroll(rows, model_variant="wmae_resid_deepsets_seq"):
    """Return (base, model) where base: (pair_id, fold) -> (y, prior);
    model: seed -> list of ((pair_id, fold), fold, y, pred)."""
    base = {}
    model = defaultdict(list)
    for r in rows:
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        yv = r.get("y") or []
        wv = r.get("weight") or []
        pv = r.get("raw_prediction") or []
        if not all(isinstance(x, list) for x in (yv, wv, pv)):
            continue
        y = [float(a) for a, w in zip(yv, wv) if w]
        p = [float(a) for a, w in zip(pv, wv) if w]
        if r["model_variant"] == "wmed_spectrum" and r["seed"] == 0:
            base[(r["pair_id"], r["fold_id"])] = (np.array(y), np.array(p))
        elif r["model_variant"] == model_variant:
            model[r["seed"]].append(
                ((r["pair_id"], r["fold_id"]), r["fold_id"], np.array(y), np.array(p)))
    return base, model


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    n = len(a)
    if n < 2:
        return float("nan")
    d = ra - rb
    return float(1.0 - 6.0 * np.sum(d * d) / (n * (n * n - 1)))


def _auroc(label, score):
    label = np.asarray(label, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    pos = score[label == 1]
    neg = score[label == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    allsc = np.concatenate([pos, neg])
    order = np.argsort(allsc)
    ranks = np.empty_like(allsc, dtype=np.float64)
    ranks[order] = np.arange(1, len(allsc) + 1)
    # average ranks for ties
    usort, inv, counts = np.unique(allsc, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for v in usort:
            m = allsc == v
            ranks[m] = ranks[m].mean()
    s = ranks[:n1].sum()
    return float((s - n1 * (n1 + 1) / 2) / (n1 * n0))


def pub_blocks(base, model, seed, exclude_pub=None):
    blocks = defaultdict(lambda: {"dt": [], "dp": [], "adt": [], "adp": []})
    for (pid, fold), fold_id, y, p in model[seed]:
        if exclude_pub is not None and fold_id == exclude_pub:
            continue
        b = base.get((pid, fold))
        if b is None or len(b[0]) != len(y):
            continue
        bp = b[1]
        blocks[fold_id]["dt"].extend(y - bp)
        blocks[fold_id]["dp"].extend(p - bp)
        blocks[fold_id]["adt"].extend(np.abs(y - bp))
        blocks[fold_id]["adp"].extend(np.abs(p - bp))
    return blocks


def pooled_metrics(blocks):
    dt = np.concatenate([b["dt"] for b in blocks.values()])
    dp = np.concatenate([b["dp"] for b in blocks.values()])
    adt = np.concatenate([b["adt"] for b in blocks.values()])
    adp = np.concatenate([b["adp"] for b in blocks.values()])
    th = float(np.median(adt))
    lab = (adt > th).astype(int)
    return {
        "n_positions": len(dt),
        "spearman_signed": _spearman(dt, dp),
        "spearman_abs": _spearman(adt, adp),
        "auroc_abs": _auroc(lab, adp),
        "median_abs_dev_threshold": th,
    }


def perm_test(blocks, metric="spearman_signed", n_perm=300, seed=20260812):
    rng = np.random.default_rng(seed)
    pub_ids = list(blocks.keys())
    dt_all = np.concatenate([blocks[p]["dt"] for p in pub_ids])
    dp_all = np.concatenate([blocks[p]["dp"] for p in pub_ids])
    real = _spearman(dt_all, dp_all)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(pub_ids)
        dp_perm = np.concatenate([blocks[p]["dp"] for p in perm])
        r = _spearman(dt_all, dp_perm)
        if not np.isnan(r) and r >= real:
            cnt += 1
    return float(real), (cnt + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-variant", default="wmae_resid_deepsets_seq")
    ap.add_argument("--dominant-pub", default="pmid_29446752")
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--perm-seed", type=int, default=20260812)
    args = ap.parse_args()

    rows = _load_rows(args.pred)
    base, model = _unroll(rows, args.model_variant)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "reactflow_delta.deviation_detection.v1",
        "endpoint": "endpoint_v6", "caller_version": "caller_v4",
        "model_variant": args.model_variant,
        "models": {},
    }
    for seed in [0, 1, 2, 3, 4]:
        blocks = pub_blocks(base, model, seed)
        m = pooled_metrics(blocks)
        rho, p = perm_test(blocks, n_perm=args.n_perm, seed=args.perm_seed + seed)
        m["spearman_signed_perm_p"] = p
        m["n_publications"] = len(blocks)
        report["models"][f"{args.model_variant}:{seed}"] = m

        # leave-one-out on dominant publication
        blocks_loo = pub_blocks(base, model, seed, exclude_pub=args.dominant_pub)
        m_loo = pooled_metrics(blocks_loo)
        rho_loo, p_loo = perm_test(blocks_loo, n_perm=args.n_perm,
                                   seed=args.perm_seed + seed + 100)
        m_loo["spearman_signed_perm_p"] = p_loo
        m_loo["n_publications"] = len(blocks_loo)
        report["models"][f"{args.model_variant}:{seed}"]["without_dominant"] = m_loo

    (out / "deviation_detection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE -> {out / 'deviation_detection.json'}")
    for seed in [0, 1, 2, 3, 4]:
        m = report["models"][f"{args.model_variant}:{seed}"]
        print(f"SEED {seed} spearman_signed={m['spearman_signed']:.4f} "
              f"perm_p={m['spearman_signed_perm_p']:.4f} "
              f"auroc_abs={m['auroc_abs']:.4f} n_pub={m['n_publications']}")


if __name__ == "__main__":
    main()
