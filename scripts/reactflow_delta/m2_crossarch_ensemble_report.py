#!/usr/bin/env python3
"""m2_crossarch_ensemble_report — cross-architecture ensemble of the M2
position-aware (v3) and self-attention (v4) mu-ensemble predictions.

Idea: v3 (per-position MLP heads) and v4 (self-attention over positions) capture
different inductive biases; averaging their mu-ensemble predictions may reduce
variance / combine strengths without any new training.  We evaluate:

  ens_pred = 0.5 * mu_ens(v3) + 0.5 * mu_ens(v4)

against the same sequence-free per-position median prior baseline, with the same
pooled-WMAE skill, design-block bootstrap CI, and design-block permutation p as
the single-architecture rows.

Reads the KEYED predictions from both runs; requires the same pair_ids in both.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = [0, 1, 2, 3, 4]
BASELINE = "wmed_spectrum"


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


def _unroll(rows, model_variant):
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


def _pub_blocks(base, ens, key, W=21):
    groups = defaultdict(lambda: {"y": [], "w": [], "m": [], "b": []})
    for k in key:
        b = base[k]
        d = k.split(":")[0]
        if len(b["y"]) != len(ens[k]):
            continue
        for j in range(min(W, len(b["y"]))):
            if b["w"][j] <= 0:
                continue
            groups[d]["y"].append(b["y"][j]); groups[d]["w"].append(1.0)
            groups[d]["b"].append(b["pred"][j]); groups[d]["m"].append(ens[k][j])
    return {d: {kk: np.array(v, dtype=np.float64) for kk, v in g.items()}
            for d, g in groups.items()}


def analyze(base, ens, key, W=21, n_perm=300, n_boot=300, perm_seed=20260815):
    blocks = _pub_blocks(base, ens, key, W=W)
    n_pos = int(sum(len(g["y"]) for g in blocks.values()))
    y = np.concatenate([g["y"] for g in blocks.values()])
    w = np.concatenate([g["w"] for g in blocks.values()])
    b = np.concatenate([g["b"] for g in blocks.values()])
    m = np.concatenate([g["m"] for g in blocks.values()])
    wmae_b = _wmae(y, w, b)
    wmae_m = _wmae(y, w, m)
    if not np.isfinite(wmae_b) or wmae_b <= 0.0 or not blocks:
        return {"n_designs": len(blocks), "n_positions": n_pos, "skill": None}
    real = 1.0 - wmae_m / wmae_b
    rng = np.random.default_rng(perm_seed)
    ids = list(blocks.keys())
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), size=len(ids))
        sk = _block_skill({ids[i]: blocks[ids[i]] for i in idx})
        if sk is not None:
            boot.append(sk)
    lo = float(np.percentile(boot, 2.5)) if boot else None
    hi = float(np.percentile(boot, 97.5)) if boot else None
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(ids))
        perm_blocks = {}
        for j in range(len(ids)):
            blk = blocks[ids[j]]
            src = blocks[ids[perm[j]]]
            perm_blocks[ids[j]] = {"y": blk["y"], "w": blk["w"], "b": blk["b"], "m": src["m"]}
        sk = _block_skill(perm_blocks)
        if sk is not None and sk >= real:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    dskills = sorted((d, _block_skill({d: blocks[d]})) for d in ids if _block_skill({d: blocks[d]}) is not None)
    arr = np.array([s for _, s in dskills]) if dskills else np.array([])
    return {
        "skill": float(real), "wmae_model": float(wmae_m), "wmae_baseline": float(wmae_b),
        "ci_low": lo, "ci_high": hi, "permutation_p": float(p),
        "n_designs": len(blocks), "n_positions": n_pos, "n_perm": n_perm, "n_boot": n_boot,
        "per_design": {
            "mean": float(arr.mean()) if len(arr) else None,
            "median": float(np.median(arr)) if len(arr) else None,
            "pct_positive": float((arr > 0).mean()) if len(arr) else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-pa", required=True)
    ap.add_argument("--pred-attn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pa-variant", default="wmae_resid_posaware_spectrum")
    ap.add_argument("--attn-variant", default="wmae_resid_attn_spectrum")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight on the position-aware prediction: "
                         "ens = alpha*pa + (1-alpha)*attn")
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--perm-seed", type=int, default=20260815)
    args = ap.parse_args()

    rows_pa = _load_rows(args.pred_pa)
    rows_attn = _load_rows(args.pred_attn)
    base_pa, model_pa = _unroll(rows_pa, args.pa_variant)
    base_attn, model_attn = _unroll(rows_attn, args.attn_variant)

    # require identical pair coverage in both runs for a matched cross-arch ensemble
    common = [k for k in base_pa
              if len(model_pa.get(k, {})) == len(SEEDS)
              and len(model_attn.get(k, {})) == len(SEEDS)]
    base = {k: base_pa[k] for k in common}
    ens_pa = {k: np.mean([model_pa[k][s] for s in SEEDS], axis=0) for k in common}
    ens_attn = {k: np.mean([model_attn[k][s] for s in SEEDS], axis=0) for k in common}
    alpha = args.alpha
    ens = {k: alpha * ens_pa[k] + (1.0 - alpha) * ens_attn[k] for k in common}

    report = analyze(base, ens, common, n_perm=args.n_perm,
                     n_boot=args.n_boot, perm_seed=args.perm_seed)
    # also report the two components on the same matched subset for context
    comp_pa = analyze(base, ens_pa, common, n_perm=args.n_perm,
                      n_boot=args.n_boot, perm_seed=args.perm_seed)
    comp_attn = analyze(base, ens_attn, common, n_perm=args.n_perm,
                        n_boot=args.n_boot, perm_seed=args.perm_seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "reactflow_delta.response_spectrum.m2_crossarch_ensemble.v1",
        "dataset": "OpenKnot_M2", "exchangeable_unit": "puzzle_x_method_design",
        "baseline": BASELINE, "alpha": alpha,
        "n_pairs": len(common), "n_designs": report["n_designs"],
        "ensemble": report,
        "component_position_aware": comp_pa,
        "component_attention": comp_attn,
    }
    (out / "m2_crossarch_ensemble_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("=== M2 cross-architecture ensemble (matched pairs) ===")
    print(f"alpha(pa)={alpha} n_pairs={len(common)} n_designs={report['n_designs']}")
    for name, r in (("position_aware", comp_pa), ("attention", comp_attn),
                    ("ENSEMBLE", report)):
        print(f"  {name:18s} skill={r['skill']:+.4f} ci=({r['ci_low']:.4f},{r['ci_high']:.4f}) "
              f"perm_p={r['permutation_p']:.4f}")
    print(f"DONE -> {out / 'm2_crossarch_ensemble_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
