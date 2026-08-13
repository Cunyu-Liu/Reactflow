#!/usr/bin/env python3
"""m2_horizontal_ensemble_report — publication-level horizontal comparison for the
OpenKnot M2 full-spectrum response experiment, adding the mu-ensemble (seed-averaged)
row to the single-seed residual-MLP rows and the wmed prior baseline.

Reads the KEYED per-position-spectrum predictions produced by
run_response_spectrum_m2_v1.py (plain residual MLP) and reports, for the magnitude
FULL-SPECTRUM task (independent of primary), the per-seed and mu-ensemble rows:

  * pooled WMAE skill vs the sequence-free per-position weighted-median prior
  * per-window-position skill (the edit-site signal is expected to peak at the
    central position and decay symmetrically outward)
  * per-design skill distribution (mean / median / % positive / best / worst)
  * publication-block bootstrap CI + permutation p for the mu-ensemble row

Statistical unit = design (block).  Window positions are pooled within a design but
the design is the exchangeable unit for CI/permutation, mirroring the M2 run's
exchangeable_unit = puzzle_x_method_design.  This is the horizontal comparison that
contrasts the sequence-free prior, the plain residual MLP, and its mu-ensemble.
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


def _wmae(y, w, pred):
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    num = float(np.sum(w * np.abs(y - pred)))
    den = float(np.sum(w))
    return num / den if den > 0 else float("nan")


def _unroll(rows, model_variant=MODEL):
    """Return (base, model) dicts: key = pair_id -> {y, w, pred} arrays.

    base   : wmed_spectrum seed 0 (sequence-free prior).
    model  : ``model_variant`` keyed by (pair_id, seed).
    """
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
        y = np.array([float(a) for a, ww in zip(yv, wv) if ww], dtype=np.float64)
        w = np.ones(len(y), dtype=np.float64)
        p = np.array([float(a) for a, ww in zip(pv, wv) if ww], dtype=np.float64)
        if r["model_variant"] == BASELINE and r["seed"] == 0:
            base[r["pair_id"]] = {"y": y, "w": w, "pred": p}
        elif r["model_variant"] == model_variant:
            model[r["pair_id"]][r["seed"]] = p
    return base, model


def _pooled_skill(base, model_preds, key):
    """Skill = 1 - WMAE(y, ens_pred) / WMAE(y, base_pred) over pairs in ``key``."""
    yb, wb, bb, mb = [], [], [], []
    for k in key:
        b = base[k]
        yb.append(b["y"]); wb.append(b["w"]); bb.append(b["pred"])
        mb.append(model_preds[k])
    y = np.concatenate(yb); w = np.concatenate(wb)
    b = np.concatenate(bb); m = np.concatenate(mb)
    wmae_b = _wmae(y, w, b)
    wmae_m = _wmae(y, w, m)
    if not np.isfinite(wmae_b) or wmae_b <= 0.0:
        return None, wmae_m, wmae_b
    return 1.0 - wmae_m / wmae_b, wmae_m, wmae_b


def _pub_blocks(base, model_preds, key, W=21):
    """Group per-position arrays into design blocks (the exchangeable unit)."""
    groups = defaultdict(lambda: {"y": [], "w": [], "m": [], "b": []})
    for k in key:
        b = base[k]
        d = k.split(":")[0]
        if len(b["y"]) != len(model_preds[k]):
            continue
        for j in range(min(W, len(b["y"]))):
            if b["w"][j] <= 0:
                continue
            groups[d]["y"].append(b["y"][j]); groups[d]["w"].append(1.0)
            groups[d]["b"].append(b["pred"][j]); groups[d]["m"].append(model_preds[k][j])
    return {d: {kk: np.array(v, dtype=np.float64) for kk, v in g.items()}
            for d, g in groups.items()}


def _block_skill(blocks):
    y = np.concatenate([g["y"] for g in blocks.values()])
    w = np.concatenate([g["w"] for g in blocks.values()])
    b = np.concatenate([g["b"] for g in blocks.values()])
    m = np.concatenate([g["m"] for g in blocks.values()])
    wmae_b = _wmae(y, w, b)
    wmae_m = _wmae(y, w, m)
    if not np.isfinite(wmae_b) or wmae_b <= 0.0:
        return None
    return 1.0 - wmae_m / wmae_b


def _design_blocks(blocks):
    """(design_id, skill) for every design block, used for per-design distribution."""
    out = []
    for d, g in blocks.items():
        m = _block_skill({d: g})
        if m is not None:
            out.append((d, m))
    return out


def analyze(base, model, ens_pred, W=21, n_perm=300, n_boot=300, perm_seed=20260812):
    """Full horizontal analysis of the mu-ensemble vs prior; returns report dict."""
    common = [k for k in base if len(model.get(k, {})) == len(SEEDS)]
    ens = {k: ens_pred[k] for k in common}
    blocks = _pub_blocks(base, ens, common, W=W)
    n_pos = int(sum(len(g["y"]) for g in blocks.values()))

    rng = np.random.default_rng(perm_seed)
    real, wmae_m, wmae_b = _pooled_skill(base, ens, common)
    if real is None or not blocks:
        return {"n_designs": len(blocks), "n_positions": n_pos, "skill": None}

    # publication/design-block bootstrap CI (sample integer indices into ids)
    ids = list(blocks.keys())
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), size=len(ids))
        sk = _block_skill({ids[i]: blocks[ids[i]] for i in idx})
        if sk is not None:
            boot.append(sk)
    lo = float(np.percentile(boot, 2.5)) if boot else None
    hi = float(np.percentile(boot, 97.5)) if boot else None

    # design-block permutation p: swap ONLY the model block across designs while
    # keeping each design's y / w / baseline fixed (breaking the model-target pairing).
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(ids))
        perm_blocks = {}
        for j in range(len(ids)):
            b = blocks[ids[j]]                 # keep this design's y/w/b
            src = blocks[ids[perm[j]]]         # borrow another design's model
            perm_blocks[ids[j]] = {"y": b["y"], "w": b["w"], "b": b["b"], "m": src["m"]}
        sk = _block_skill(perm_blocks)
        if sk is not None and sk >= real:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)

    dskills = sorted(_design_blocks(blocks), key=lambda x: x[1])
    arr = np.array([s for _, s in dskills])
    return {
        "skill": float(real), "wmae_model": float(wmae_m), "wmae_baseline": float(wmae_b),
        "ci_low": lo, "ci_high": hi, "permutation_p": float(p),
        "n_designs": len(blocks), "n_positions": n_pos, "n_perm": n_perm, "n_boot": n_boot,
        "per_design": {
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "pct_positive": float((arr > 0).mean()),
            "best": [{"design": d, "skill": float(s)} for d, s in dskills[-5:]],
            "worst": [{"design": d, "skill": float(s)} for d, s in dskills[:5]],
        },
    }


def per_position(base, model, ens_pred, common, W=21):
    """Per-window-position skill of the mu-ensemble vs prior."""
    pos = [{"y": [], "w": [], "m": [], "b": []} for _ in range(W)]
    for k in common:
        b = base[k]
        m = ens_pred[k]
        for j in range(min(W, len(b["y"]))):
            if b["w"][j] <= 0:
                continue
            pos[j]["y"].append(b["y"][j]); pos[j]["w"].append(1.0)
            pos[j]["b"].append(b["pred"][j]); pos[j]["m"].append(m[j])
    out = []
    for j in range(W):
        y = np.array(pos[j]["y"]); w = np.array(pos[j]["w"])
        b = np.array(pos[j]["b"]); m = np.array(pos[j]["m"])
        if len(y) == 0 or w.sum() <= 0:
            continue
        wmae_b = _wmae(y, w, b); wmae_m = _wmae(y, w, m)
        sk = 1.0 - wmae_m / wmae_b if wmae_b > 0 else None
        out.append({"position": j, "wmae_baseline": wmae_b, "wmae_model": wmae_m,
                    "skill": sk, "n_positions": int(len(y))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-variant", default=MODEL,
                    help="model_variant to analyze (default wmae_resid_spectrum)")
    ap.add_argument("--dominant-design", default="__NONE__")
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--perm-seed", type=int, default=20260812)
    args = ap.parse_args()

    rows = _load_rows(args.pred)
    base, model = _unroll(rows, args.model_variant)
    common = [k for k in base if len(model.get(k, {})) == len(SEEDS)]
    ens = {k: np.mean([model[k][s] for s in SEEDS], axis=0) for k in common}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "reactflow_delta.response_spectrum.m2_horizontal_ensemble.v1",
        "dataset": "OpenKnot_M2", "exchangeable_unit": "puzzle_x_method_design",
        "baseline": BASELINE, "model": args.model_variant, "seeds": SEEDS,
        "n_pairs": len(common), "n_seed_single_skill": {},
    }
    # per-seed single rows
    for s in SEEDS:
        single = {k: model[k][s] for k in common}
        sk, wm, wb = _pooled_skill(base, single, common)
        report["n_seed_single_skill"][f"seed_{s}"] = {
            "skill": sk, "wmae_model": wm, "wmae_baseline": wb}
    # mu-ensemble row + CI/permutation
    report["mu_ensemble"] = analyze(base, model, ens, n_perm=args.n_perm,
                                    n_boot=args.n_boot, perm_seed=args.perm_seed)
    report["per_position"] = per_position(base, model, ens, common)

    (out / "m2_horizontal_ensemble_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== M2 horizontal comparison (mu-ensemble) ===")
    print(f"model_variant={args.model_variant}")
    print(f"baseline wmed_prior WMAE = {report['n_seed_single_skill']['seed_0']['wmae_baseline']:.4f}")
    for s in SEEDS:
        e = report["n_seed_single_skill"][f"seed_{s}"]
        print(f"  {args.model_variant} seed {s}: skill={e['skill']:+.4f}")
    m = report["mu_ensemble"]
    print(f"  {args.model_variant} mu-ensemble: skill={m['skill']:+.4f} "
          f"ci=({m['ci_low']:.4f},{m['ci_high']:.4f}) perm_p={m['permutation_p']:.4f}")
    pd = m["per_design"]
    print(f"  per-design mean={pd['mean']:+.4f} median={pd['median']:+.4f} "
          f"%positive={pd['pct_positive']:.3f} n_designs={m['n_designs']}")
    print("  per-position skill (peaks at central edit site expected):")
    print("   " + ", ".join(f"{p['position']}:{p['skill']:+.3f}"
                            for p in report["per_position"]))
    print(f"DONE -> {out / 'm2_horizontal_ensemble_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
