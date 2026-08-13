#!/usr/bin/env python3
"""m2_deviation_report — deviation-detection on OpenKnot M2 for the mu-ensemble
residual-MLP full-spectrum model.

Complementary, independent capability to the pooled-WMAE horizontal comparison
(m2_horizontal_ensemble_report).  The residual deltas learned by the model are
themselves a *deviation score*: after mu-ensembling seeds, we ask whether the
model ranks positions by TRUE deviation from the sequence-free per-position
median prior:

    adt  = |y - prior|            (true deviation magnitude)
    score= |mu_ens_pred - prior|  (model-predicted deviation magnitude, implicit)

Reported, mirroring evaluate_deviation_v2 on the original data:
  * pooled Spearman rank corr rho(adt, score)
  * AUROC of score for detecting |y-prior| above the pooled median
  * design-block permutation p (exchangeable unit = puzzle x method design:
    shuffle score blocks across designs while keeping each design's y/prior fixed,
    breaking the model<->truth coupling under the null)
  * per-design deviation-skill distribution and per-window-position curve

Statistical discipline matches the M2 WMAE report: window positions pooled within
a design, but the design is the exchangeable unit for CI/permutation.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = [0, 1, 2, 3, 4]
BASELINE = "wmed_spectrum"
MODEL = "wmae_resid_spectrum"


def _load_rows(pred_path):
    rows = []
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _as_vec(v, w):
    return np.array([float(a) for a, ww in zip(v, w) if ww], dtype=np.float64)


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    n = len(a)
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
    for v in np.unique(allsc):
        m = allsc == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    s = ranks[:n1].sum()
    return float((s - n1 * (n1 + 1) / 2) / (n1 * n0))


def _unroll(rows):
    """Return (base, model): base[pair_id] -> {y, prior}; model[pair_id][seed] -> pred."""
    base = {}
    model = defaultdict(dict)
    for r in rows:
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        yv = r.get("y") or []
        wv = r.get("weight") or []
        pv = r.get("raw_prediction")
        if not (isinstance(yv, list) and isinstance(wv, list) and isinstance(pv, list)):
            continue
        y = _as_vec(yv, wv)
        p = _as_vec(pv, wv)
        if len(y) != len(p):
            continue
        if r["model_variant"] == BASELINE and r["seed"] == 0:
            base[r["pair_id"]] = {"y": y, "prior": p}
        elif r["model_variant"] == MODEL:
            model[r["pair_id"]][r["seed"]] = p
    return base, model


def _design_blocks(base, model, ens_pred, key, W=21):
    """Group per-position (adt, score) into design blocks (exchangeable unit)."""
    groups = defaultdict(lambda: {"adt": [], "score": []})
    for k in key:
        b = base[k]
        m = ens_pred[k]
        d = k.split(":")[0]
        for j in range(min(W, len(b["y"]))):
            if b["y"][j] != b["y"][j]:  # NaN guard
                continue
            adt = float(abs(b["y"][j] - b["prior"][j]))
            sc = float(abs(m[j] - b["prior"][j]))
            groups[d]["adt"].append(adt)
            groups[d]["score"].append(sc)
    return {d: {kk: np.array(v, dtype=np.float64) for kk, v in g.items()}
            for d, g in groups.items()}


def _pooled(blocks):
    adt = np.concatenate([b["adt"] for b in blocks.values()])
    score = np.concatenate([b["score"] for b in blocks.values()])
    th = float(np.median(adt))
    lab = (adt > th).astype(int)
    return {
        "n_positions": int(len(adt)),
        "spearman_abs": _spearman(adt, score),
        "auroc_abs": _auroc(lab, score),
        "median_abs_dev_threshold": th,
    }


def _perm(blocks, n_perm, seed):
    """Design-block permutation p for pooled Spearman (swap score blocks across
    designs, keep each design's adt fixed)."""
    rng = np.random.default_rng(seed)
    ids = list(blocks.keys())
    real = _spearman(np.concatenate([blocks[p]["adt"] for p in ids]),
                     np.concatenate([blocks[p]["score"] for p in ids]))
    if not np.isfinite(real):
        return real, 1.0
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(ids)
        score_perm = np.concatenate([blocks[perm[j]]["score"] for j in range(len(ids))])
        r = _spearman(np.concatenate([blocks[p]["adt"] for p in ids]), score_perm)
        if not np.isnan(r) and r >= real:
            cnt += 1
    return float(real), (cnt + 1) / (n_perm + 1)


def _design_devskill(blocks):
    """Per-design deviation-skill = rho within a single design block."""
    out = []
    for d, g in blocks.items():
        r = _spearman(g["adt"], g["score"])
        if not np.isnan(r):
            out.append((d, r))
    return out


def _loo_sensitivity(blocks, n_perm, seed):
    """Robustness: recompute pooled rho when each single design is excluded.

    Returns dict with the pooled rho/perm_p excluding the single strongest-rho
    design and the single weakest-rho design, plus the min/max rho over ALL
    single-design exclusions.  If the overall signal survives every single-design
    removal, the result is not driven by one design.
    """
    ids = list(blocks.keys())
    rng = np.random.default_rng(seed)
    # strongest/weakest by per-design rho
    ds = sorted(_design_devskill(blocks), key=lambda x: x[1])
    weakest = ds[0][0]
    strongest = ds[-1][0]

    def pooled_rho_perm(excl):
        sub = {d: g for d, g in blocks.items() if d != excl}
        r = _pooled(sub)["spearman_abs"]
        p = _perm(sub, n_perm, seed)
        return {"excluded": excl, "spearman_abs": r, "permutation_p": p[1]}

    excl_strong = pooled_rho_perm(strongest)
    excl_weak = pooled_rho_perm(weakest)

    # sweep over all single exclusions (rho only; n_perm re-computation is heavy)
    lo, hi = float("inf"), float("-inf")
    for d in ids:
        r = _pooled({x: g for x, g in blocks.items() if x != d})["spearman_abs"]
        if r == r and not np.isnan(r):
            lo, hi = min(lo, r), max(hi, r)
    return {
        "n_designs": len(ids),
        "exclude_strongest": excl_strong,
        "exclude_weakest": excl_weak,
        "pooled_rho_min_over_loo": None if lo == float("inf") else float(lo),
        "pooled_rho_max_over_loo": None if hi == float("-inf") else float(hi),
    }


def analyze(base, model, ens_pred, W=21, n_perm=300, perm_seed=20260812):
    """Full deviation-detection report for the mu-ensemble; returns dict."""
    common = [k for k in base if len(model.get(k, {})) == len(SEEDS)]
    blocks = _design_blocks(base, model, ens_pred, common, W=W)
    if not blocks:
        return {"n_designs": 0, "n_positions": 0}
    pooled = _pooled(blocks)
    rho, p = _perm(blocks, n_perm, perm_seed)
    dskills = sorted(_design_devskill(blocks), key=lambda x: x[1])
    arr = np.array([s for _, s in dskills])
    return {
        "n_designs": len(blocks),
        "n_positions": int(pooled["n_positions"]),
        "spearman_abs": pooled["spearman_abs"],
        "auroc_abs": pooled["auroc_abs"],
        "permutation_p": p,
        "median_abs_dev_threshold": pooled["median_abs_dev_threshold"],
        "n_perm": n_perm,
        "per_design": {
            "mean": float(arr.mean()) if len(arr) else None,
            "median": float(np.median(arr)) if len(arr) else None,
            "pct_positive": float((arr > 0).mean()) if len(arr) else None,
            "best": [{"design": d, "rho": float(s)} for d, s in dskills[-5:]],
            "worst": [{"design": d, "rho": float(s)} for d, s in dskills[:5]],
        },
        "robustness": _loo_sensitivity(blocks, n_perm, perm_seed),
    }


def per_position(base, model, ens_pred, common, W=21):
    """Per-window-position Spearman of the mu-ensemble deviation score."""
    pos = [{"adt": [], "score": []} for _ in range(W)]
    for k in common:
        b = base[k]
        m = ens_pred[k]
        for j in range(min(W, len(b["y"]))):
            pos[j]["adt"].append(float(abs(b["y"][j] - b["prior"][j])))
            pos[j]["score"].append(float(abs(m[j] - b["prior"][j])))
    out = []
    for j in range(W):
        if len(pos[j]["adt"]) < 2:
            continue
        out.append({"position": j, "n_positions": len(pos[j]["adt"]),
                    "spearman_abs": _spearman(pos[j]["adt"], pos[j]["score"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--perm-seed", type=int, default=20260812)
    args = ap.parse_args()

    rows = _load_rows(args.pred)
    base, model = _unroll(rows)
    common = [k for k in base if len(model.get(k, {})) == len(SEEDS)]
    ens = {k: np.mean([model[k][s] for s in SEEDS], axis=0) for k in common}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "reactflow_delta.response_spectrum.m2_deviation_report.v1",
        "dataset": "OpenKnot_M2", "exchangeable_unit": "puzzle_x_method_design",
        "baseline": BASELINE, "model": MODEL, "seeds": SEEDS,
        "n_pairs": len(common),
        "mu_ensemble": analyze(base, model, ens, n_perm=args.n_perm, perm_seed=args.perm_seed),
        "per_seed": {},
        "per_position": per_position(base, model, ens, common),
    }
    for s in SEEDS:
        single = {k: model[k][s] for k in common}
        report["per_seed"][f"seed_{s}"] = analyze(
            base, model, single, n_perm=args.n_perm, perm_seed=args.perm_seed + s)

    (out / "m2_deviation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    m = report["mu_ensemble"]
    print("=== M2 deviation-detection (mu-ensemble, implicit |pred-prior|) ===")
    print(f"  Spearman rho = {m['spearman_abs']:.4f}  perm_p = {m['permutation_p']:.4f}")
    print(f"  AUROC        = {m['auroc_abs']:.4f}  n_designs={m['n_designs']} "
          f"n_pos={m['n_positions']}")
    pd = m["per_design"]
    if pd["mean"] is not None:
        print(f"  per-design rho mean={pd['mean']:.4f} median={pd['median']:.4f} "
              f"%positive={pd['pct_positive']:.3f}")
    print("  per-position rho:", ", ".join(
        f"{p['position']}:{p['spearman_abs']:+.3f}" for p in report["per_position"]))
    print(f"DONE -> {out / 'm2_deviation_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
