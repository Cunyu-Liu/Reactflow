#!/usr/bin/env python3
"""m2_masked_eval_from_oof.py — recompute the M2 GBDT+3-way ensemble headline
on the MATCHED row set only, directly from the existing OOF npz of the
all-rows run (/mnt/cunyuliu/m2_gbdt_3way_ensemble_20260818).

Rationale:
  * The +14.08% run evaluated on ALL 277,451 rows; unmatched rows (4,463) got a
    median placeholder for deep AND prior.  On those rows deep is handicapped to
    the baseline median, so the blend-vs-deep comparison is NOT fair.
  * The mask version of m2_gbdt_ensemble_v1.py fixes this by evaluating only on
    the 272,988 matched rows.  Its GBDT is trained on all rows with the same
    design-LOO (_loo_lgb), so its per-row GBDT/deep predictions are IDENTICAL to
    the saved OOF arrays; only the evaluation row set changes.
  * Therefore the honest masked headline == recomputing all metrics on the
    matched subset of the saved OOF npz.  This avoids a ~30 min ViennaRNA-BPP
    feature rebuild.

Design-block bootstrap CI and permutation p use design as the exchangeable unit,
exactly like the ensemble script.
"""
from __future__ import annotations

import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260818
ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _wmae(y, w, pred):
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    num = float(np.sum(w * np.abs(y - pred)))
    den = float(np.sum(w))
    return num / den if den > 0 else float("nan")


def _skill(mae_model, mae_base):
    return 1.0 - mae_model / mae_base if mae_base > 0 else 0.0


def _mask_from_placeholder(deep, prior, y):
    """Rows whose deep AND prior were both set to the median placeholder are
    unmatched (had no keyed prediction); everything else is a matched row."""
    const = float(np.median(y))
    matched = ~((np.abs(deep - const) < 1e-9) & (np.abs(prior - const) < 1e-9))
    return matched


def analyze(y, w, keys, pred, base, n_perm=300, n_boot=300, seed=SEED):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(keys.tolist())))
    real = _skill(_wmae(y, w, pred), _wmae(y, w, base))

    blocks = {}
    for d in ids:
        m = keys == d
        blocks[d] = {"y": y[m], "w": w[m], "b": base[m], "p": pred[m]}

    # bootstrap CI over designs
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), size=len(ids))
        sub = {}
        for j in set(idx.tolist()):
            sub[j] = blocks[ids[j]]
        yv = np.concatenate([sub[j]["y"] for j in sub])
        wv = np.concatenate([sub[j]["w"] for j in sub])
        bv = np.concatenate([sub[j]["b"] for j in sub])
        pv = np.concatenate([sub[j]["p"] for j in sub])
        boots.append(_skill(_wmae(yv, wv, pv), _wmae(yv, wv, bv)))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # permutation test (block permutation of the model preds across designs)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(ids))
        perm_blocks = {}
        for j in range(len(ids)):
            b = blocks[ids[j]]
            src = blocks[ids[perm[j]]]
            perm_blocks[ids[j]] = {"y": b["y"], "w": b["w"], "b": b["b"], "p": src["p"]}
        yv = np.concatenate([perm_blocks[d]["y"] for d in perm_blocks])
        wv = np.concatenate([perm_blocks[d]["w"] for d in perm_blocks])
        bv = np.concatenate([perm_blocks[d]["b"] for d in perm_blocks])
        pv = np.concatenate([perm_blocks[d]["p"] for d in perm_blocks])
        sk = _skill(_wmae(yv, wv, pv), _wmae(yv, wv, bv))
        if np.isfinite(sk) and sk >= real:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return {"skill": float(real), "ci_low": float(lo), "ci_high": float(hi),
            "permutation_p": float(p), "n_designs": int(len(ids)),
            "n_perm": n_perm, "n_boot": n_boot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True,
                    help="m2_gbdt_ensemble_oof.npz from the all-rows run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    t0 = time.time()
    z = np.load(args.oof, allow_pickle=True)
    y = z["y"]; w = z["w"]; keys = z["keys"]
    gbdt_all = z["gbdt"]; deep_all = z["deep"]; prior_all = z["prior"]

    matched = _mask_from_placeholder(deep_all, prior_all, y)
    print(f"[m2m] rows={len(y)} matched={int(matched.sum())} "
          f"unmatched={int((~matched).sum())}", flush=True)
    assert int(matched.sum()) == 272988, "mask mismatch vs expected 272,988"

    ym, wm = y[matched], w[matched]
    gbdt_m, deep_m, prior_m = gbdt_all[matched], deep_all[matched], prior_all[matched]
    keys_m = keys[matched]

    mae_prior = _wmae(ym, wm, prior_m)
    mae_gbdt = _wmae(ym, wm, gbdt_m)
    mae_attn = _wmae(ym, wm, deep_m)
    skill_gbdt = _skill(mae_gbdt, mae_prior)
    skill_attn = _skill(mae_attn, mae_prior)

    blends = {}
    for a in ALPHA_GRID:
        b = a * gbdt_m + (1.0 - a) * deep_m
        blends[str(a)] = {"skill": float(_skill(_wmae(ym, wm, b), mae_prior)),
                          "mae": float(_wmae(ym, wm, b))}

    ALPHA_HL = args.alpha
    blend_hl = ALPHA_HL * gbdt_m + (1.0 - ALPHA_HL) * deep_m
    mae_hl = _wmae(ym, wm, blend_hl)
    skill_hl = _skill(mae_hl, mae_prior)
    sig = analyze(ym, wm, keys_m, blend_hl, prior_m,
                  n_perm=args.n_perm, n_boot=args.n_boot, seed=SEED)

    # LOO-exclusion gain: blend vs deep (the current M2 headline)
    des_list_m = sorted(set(keys_m.tolist()))
    gains = []
    for held in des_list_m:
        mm = keys_m != held
        if mm.sum() < 100:
            continue
        g = (_skill(_wmae(ym[mm], wm[mm], blend_hl[mm]), _wmae(ym[mm], wm[mm], prior_m[mm]))
             - _skill(_wmae(ym[mm], wm[mm], deep_m[mm]), _wmae(ym[mm], wm[mm], prior_m[mm])))
        gains.append(g)
    gains = np.array(gains)

    # per-design blend-vs-deep gain
    dskills = defaultdict(lambda: {"y": [], "w": [], "b": [], "a": [], "m": []})
    for i in range(len(ym)):
        d = keys_m[i]
        dskills[d]["y"].append(ym[i]); dskills[d]["w"].append(wm[i])
        dskills[d]["b"].append(prior_m[i]); dskills[d]["a"].append(deep_m[i])
        dskills[d]["m"].append(blend_hl[i])
    pd_gains = []
    for d, g in dskills.items():
        yv = np.array(g["y"]); wv = np.array(g["w"])
        bb = np.array(g["b"]); aa = np.array(g["a"]); mm = np.array(g["m"])
        sk_a = _skill(_wmae(yv, wv, aa), _wmae(yv, wv, bb))
        sk_m = _skill(_wmae(yv, wv, mm), _wmae(yv, wv, bb))
        pd_gains.append(sk_m - sk_a)
    pd_gains = np.array(pd_gains)

    report = {
        "schema": "reactflow_delta.response_spectrum.m2_gbdt_ensemble.masked_eval.v1",
        "source_oof": args.oof,
        "evaluation_rows": "matched_only",
        "n_rows_total": int(len(y)), "n_rows_matched": int(matched.sum()),
        "n_designs": len(des_list_m),
        "alpha_headline": ALPHA_HL,
        "baseline_wmae": mae_prior,
        "results": {
            "gbdt_3way": {"mae": mae_gbdt, "skill": skill_gbdt},
            "deep_mu": {"mae": mae_attn, "skill": skill_attn},
            "blend": {"mae": mae_hl, "skill": skill_hl, "sig": sig},
        },
        "blend_curve": blends,
        "blend_vs_deep": {
            "pooled_gain_pp": float((skill_hl - skill_attn) * 100),
            "per_design_mean_pp": float(pd_gains.mean() * 100),
            "per_design_pct_positive": float((pd_gains > 0).mean()),
            "loo_exclusion": {
                "gain_mean_pp": float(gains.mean() * 100),
                "gain_min_pp": float(gains.min() * 100),
                "gain_max_pp": float(gains.max() * 100),
                "pct_positive": float((gains > 0).mean()),
                "n_folds": int(len(gains)),
            },
        },
        "wall_seconds": round(time.time() - t0, 1),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2_masked_eval_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2_masked_eval_matched.npz",
             gbdt=gbdt_m, deep=deep_m, prior=prior_m, blend=blend_hl,
             y=ym, w=wm, keys=keys_m)

    print(f"\n=== M2 masked-eval (matched {int(matched.sum())} rows) ===")
    print(f"GBDT  : skill={skill_gbdt:+.4f} mae={mae_gbdt:.4f}")
    print(f"deep  : skill={skill_attn:+.4f} mae={mae_attn:.4f}")
    print(f"blend : skill={skill_hl:+.4f} mae={mae_hl:.4f} "
          f"ci=({sig['ci_low']:.4f},{sig['ci_high']:.4f}) p={sig['permutation_p']:.4f}")
    print("blend curve: " + ", ".join(f"a={a}:{v['skill']:+.4f}" for a, v in blends.items()))
    g = report["blend_vs_deep"]
    print(f"blend-vs-attn: pooled={g['pooled_gain_pp']:+.2f}pp "
          f"per-design={g['per_design_mean_pp']:+.2f}pp ({g['per_design_pct_positive']:.2f} pos)")
    loo = g["loo_exclusion"]
    print(f"LOO-exclusion: mean={loo['gain_mean_pp']:+.2f}pp "
          f"range=[{loo['gain_min_pp']:+.2f},{loo['gain_max_pp']:+.2f}]pp "
          f"pct_pos={loo['pct_positive']:.3f}")
    print(f"DONE -> {out / 'm2_masked_eval_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
