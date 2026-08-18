#!/usr/bin/env python3
"""m2_gbdt_ensemble_v1.py — cross-architecture GBDT + attention ensemble for the
OpenKnot M2 response-spectrum task.

Method-level lever (mirrors the M2R recipe that produced +30.3% skill):

  * M2R's single largest gain was a legal thermodynamic modality (ViennaRNA
    folding of WT + mutant sequences) plus a feature-based GBDT blended with a
    deep model for error decorrelation.
  * The M2 deep attention models consume only base+reactivity+error per
    position.  We add per-position MFE/partition-function structural features
    (m2_gbdt_features_v1) and train a design-LOO L1 GBDT whose error structure
    is decorrelated from the attention model, then blend.

Rows are (pair, window-position) with weight 1 (eligible).  The exchangeable
unit is the design (puzzle x method), same as every M2 audit.

Metric: pooled WMAE skill vs the sequence-free weighted-median prior
(``wmed_spectrum`` seed-0 rows of the attn prediction file), design-block
bootstrap CI and design-block permutation p.
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_gbdt_features_v1 as gf

SEED = 20260818
SEEDS = [0, 1, 2, 3, 4]
ATTN_VARIANT = "wmae_resid_attn_spectrum"
BASELINE = "wmed_spectrum"
CFG = dict(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05,
           min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
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


def _load_preds(pred_path):
    """Return (attn_mu, prior, design_of_pid) dicts keyed by "pair_id:k".

    attn_mu = mean over SEEDS of wmae_resid_attn_spectrum raw_prediction.
    prior   = wmed_spectrum seed-0 raw_prediction.
    """
    attn = defaultdict(list)
    prior = {}
    designs = {}
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        pid = r["pair_id"]
        designs[pid] = r["source_accession"] or r["pair_id"].split(":")[0]
        pv = r.get("raw_prediction") or []
        if not isinstance(pv, list):
            continue
        if r["model_variant"] == BASELINE and r["seed"] == 0:
            prior[pid] = np.array([float(x) for x in pv], dtype=np.float64)
        elif r["model_variant"] == ATTN_VARIANT:
            attn[pid].append(np.array([float(x) for x in pv], dtype=np.float64))
    attn_mu = {pid: np.mean(arrs, axis=0) for pid, arrs in attn.items()
               if len(arrs) == len(SEEDS)}
    return attn_mu, prior, designs


def _load_variant_mu(pred_path, variant):
    """Mean-over-seeds mu-ensemble for one model variant (keyed by pair_id)."""
    mu = {}
    preds = defaultdict(list)
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        pv = r.get("raw_prediction") or []
        if not isinstance(pv, list):
            continue
        if r["model_variant"] == variant:
            preds[r["pair_id"]].append(np.array([float(x) for x in pv], dtype=np.float64))
    for pid, arrs in preds.items():
        if len(arrs) == len(SEEDS):
            mu[pid] = np.mean(arrs, axis=0)
    return mu


def build_threeway(v3_file, v4_file, v5_file, w_v3=0.15, w_v4=0.2, w_v5=0.65):
    """Weighted 3-way mu-ensemble (posaware + attn-1L + attn-2L), keyed by pair_id."""
    v3 = _load_variant_mu(v3_file, "wmae_resid_posaware_spectrum")
    v4 = _load_variant_mu(v4_file, ATTN_VARIANT)
    v5 = _load_variant_mu(v5_file, ATTN_VARIANT)
    common = set(v3) & set(v4) & set(v5)
    return {pid: w_v3 * v3[pid] + w_v4 * v4[pid] + w_v5 * v5[pid] for pid in common}


def _loo_lgb(X, y, keys, des_list):
    import lightgbm as lgb
    preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 100:
            preds[~m] = np.median(y)
            continue
        g = lgb.LGBMRegressor(objective="l1", random_state=SEED, verbose=-1,
                              n_jobs=2, **CFG)
        g.fit(X[m], y[m])
        preds[~m] = g.predict(X[~m])
    return preds


def _block_skill(y, w, pred, keys):
    """Pooled WMAE skill over the given rows."""
    mae_b = _wmae(y, w, pred["prior"])
    mae_m = _wmae(y, w, pred["model"])
    return _skill(mae_m, mae_b), mae_m, mae_b


def analyze(y, w, keys, model_pred, prior_pred, n_perm=300, n_boot=300,
            perm_seed=20260818):
    """Design-block CI + permutation p for model_pred vs prior_pred."""
    blocks = defaultdict(lambda: {"y": [], "w": [], "m": [], "b": []})
    for i in range(len(y)):
        d = keys[i]
        blocks[d]["y"].append(y[i])
        blocks[d]["w"].append(w[i])
        blocks[d]["m"].append(model_pred[i])
        blocks[d]["b"].append(prior_pred[i])
    ids = list(blocks.keys())

    def pooled():
        yv = np.concatenate([blocks[d]["y"] for d in ids])
        wv = np.concatenate([blocks[d]["w"] for d in ids])
        mv = np.concatenate([blocks[d]["m"] for d in ids])
        bv = np.concatenate([blocks[d]["b"] for d in ids])
        return _skill(_wmae(yv, wv, mv), _wmae(yv, wv, bv))

    real = pooled()
    rng = np.random.default_rng(perm_seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), size=len(ids))
        sub = {}
        for j in set(idx.tolist()):
            sub[j] = blocks[ids[j]]
        yv = np.concatenate([sub[j]["y"] for j in sub])
        wv = np.concatenate([sub[j]["w"] for j in sub])
        mv = np.concatenate([sub[j]["m"] for j in sub])
        bv = np.concatenate([sub[j]["b"] for j in sub])
        sk = _skill(_wmae(yv, wv, mv), _wmae(yv, wv, bv))
        if np.isfinite(sk):
            boot.append(sk)
    lo = float(np.percentile(boot, 2.5)) if boot else None
    hi = float(np.percentile(boot, 97.5)) if boot else None

    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(ids))
        perm_blocks = {}
        for j in range(len(ids)):
            b = blocks[ids[j]]
            src = blocks[ids[perm[j]]]
            perm_blocks[ids[j]] = {"y": b["y"], "w": b["w"], "b": b["b"], "m": src["m"]}
        yv = np.concatenate([perm_blocks[d]["y"] for d in perm_blocks])
        wv = np.concatenate([perm_blocks[d]["w"] for d in perm_blocks])
        mv = np.concatenate([perm_blocks[d]["m"] for d in perm_blocks])
        bv = np.concatenate([perm_blocks[d]["b"] for d in perm_blocks])
        sk = _skill(_wmae(yv, wv, mv), _wmae(yv, wv, bv))
        if np.isfinite(sk) and sk >= real:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return {"skill": float(real), "ci_low": lo, "ci_high": hi,
            "permutation_p": float(p), "n_designs": len(ids),
            "n_perm": n_perm, "n_boot": n_boot}


def run_m2_gbdt_ensemble(X, y, w, keys, pids, deep_pred, prior_pred, args) -> dict:
    """Blend the leak-free per-position GBDT with a deep mu-ensemble.

    ``deep_pred`` is keyed by pair_id -> 21-vector (either the attn-v5
    mu-ensemble or the weighted 3-way posaware+attn ensemble).
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    des_list = sorted(set(keys.tolist()))

    # align deep + prior per (pair_id, k)
    deep_align = np.zeros(len(y))
    prior_align = np.zeros(len(y))
    mask = np.zeros(len(y), dtype=bool)
    for i, p in enumerate(pids):
        pid, k = p.rsplit(":", 1)
        k = int(k)
        if pid in deep_pred and pid in prior_pred and k < len(deep_pred[pid]):
            deep_align[i] = deep_pred[pid][k]
            prior_align[i] = prior_pred[pid][k]
            mask[i] = True
    print(f"[m2g] aligned deep/prior rows={int(mask.sum())}/{len(y)}", flush=True)

    # evaluate ONLY on rows with a real model + baseline prediction (matched
    # coverage, identical to the published 3-way report's 272,988-position set)
    ym = y[mask]; wm = w[mask]
    prior_m = prior_align[mask]; deep_m = deep_align[mask]
    keys_m = keys[mask]
    des_list_m = sorted(set(keys_m.tolist()))

    # design-LOO per-position L1 GBDT (trained on all rows, evaluated on matched)
    t0 = time.time()
    gbdt_full = _loo_lgb(X, y, keys, des_list)
    gbdt_m = gbdt_full[mask]
    print(f"[m2g] GBDT LOO done wall={time.time()-t0:.0f}s", flush=True)

    mae_prior = _wmae(ym, wm, prior_m)
    mae_gbdt = _wmae(ym, wm, gbdt_m)
    mae_attn = _wmae(ym, wm, deep_m)
    skill_gbdt = _skill(mae_gbdt, mae_prior)
    skill_attn = _skill(mae_attn, mae_prior)

    # blend curve over a-priori alpha (gbdt weight)
    blends = {}
    for a in ALPHA_GRID:
        b = a * gbdt_m + (1.0 - a) * deep_m
        sk = _skill(_wmae(ym, wm, b), mae_prior)
        blends[str(a)] = {"skill": float(sk), "mae": float(_wmae(ym, wm, b))}

    # headline blend: fixed a-priori alpha = 0.5 (no tuning on the outcome)
    ALPHA_HL = getattr(args, "alpha", 0.5)
    blend_hl = ALPHA_HL * gbdt_m + (1.0 - ALPHA_HL) * deep_m
    mae_hl = _wmae(ym, wm, blend_hl)
    skill_hl = _skill(mae_hl, mae_prior)

    # significance of the headline blend (design-block)
    sig = analyze(ym, wm, keys_m, blend_hl, prior_m,
                  n_perm=args.n_perm, n_boot=args.n_boot, perm_seed=SEED)

    # LOO-exclusion gain: blend vs deep (the current M2 headline)
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
        "schema": "reactflow_delta.response_spectrum.m2_gbdt_ensemble.v1",
        "dataset": "OpenKnot_M2", "exchangeable_unit": "puzzle_x_method_design",
        "baseline": BASELINE, "attn_variant": ATTN_VARIANT,
        "deep_component": getattr(args, "deep_component", "attn_v5"),
        "n_rows_total": int(len(y)), "n_rows_matched": int(mask.sum()),
        "n_designs": len(des_list_m),
        "n_features": int(X.shape[1]), "alpha_headline": ALPHA_HL,
        "gbdt_cfg": CFG,
        "results": {
            "gbdt_3way": {"mae": mae_gbdt, "skill": skill_gbdt},
            "deep_mu": {"mae": mae_attn, "skill": skill_attn},
            "blend": {"mae": mae_hl, "skill": skill_hl,
                      "sig": sig},
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
    (out / "m2_gbdt_ensemble_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2_gbdt_ensemble_oof.npz",
             gbdt=gbdt_m, deep=deep_m, prior=prior_m,
             blend=blend_hl, y=ym, w=wm, keys=keys_m)

    print(f"\n=== M2 GBDT + deep ensemble ({report['deep_component']}) ===")
    print(f"GBDT  : skill={skill_gbdt:+.4f} mae={mae_gbdt:.4f}")
    print(f"deep  : skill={skill_attn:+.4f} mae={mae_attn:.4f}")
    print(f"blend : skill={skill_hl:+.4f} mae={mae_hl:.4f} "
          f"ci=({sig['ci_low']:.4f},{sig['ci_high']:.4f}) p={sig['permutation_p']:.4f}")
    print(f"blend curve: " + ", ".join(f"a={a}:{v['skill']:+.4f}" for a, v in blends.items()))
    g = report["blend_vs_deep"]
    print(f"blend-vs-attn: pooled={g['pooled_gain_pp']:+.2f}pp "
          f"per-design={g['per_design_mean_pp']:+.2f}pp ({g['per_design_pct_positive']:.2f} pos)")
    loo = g["loo_exclusion"]
    print(f"LOO-exclusion: mean={loo['gain_mean_pp']:+.2f}pp "
          f"range=[{loo['gain_min_pp']:+.2f},{loo['gain_max_pp']:+.2f}]pp "
          f"pct_pos={loo['pct_positive']:.3f}")
    print(f"wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2_gbdt_ensemble_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--attn-pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-v3", default=None,
                    help="posaware keyed predictions (for the 3-way deep component)")
    ap.add_argument("--pred-v4", default=None,
                    help="attn 1-layer keyed predictions (for the 3-way deep component)")
    ap.add_argument("--pred-v5", default=None,
                    help="attn 2-layer keyed predictions (for the 3-way deep component)")
    ap.add_argument("--w-v3", type=float, default=0.15)
    ap.add_argument("--w-v4", type=float, default=0.20)
    ap.add_argument("--w-v5", type=float, default=0.65)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="fixed a-priori GBDT weight in the blend (no tuning)")
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    import m2_data_v1 as m2d
    import response_spectrum_scinv_v1 as rss
    from run_baselines_v6 import WINDOW as W

    designs, dmeta = m2d.parse_m2_csv(args.m2_csv)
    samples = m2d.build_all_samples(designs)
    spectra = {}
    for s in samples:
        pid = f"{s.design_id}:{s.mutA}"
        yv, wv, sc = rss.pair_response_spectrum(
            s.wt_reactivity, s.mut_reactivity, s.eligibility_mask,
            s.edit_seq_pos, window=W)
        if sc is None or sum(wv) <= 0:
            continue
        spectra[pid] = {"y": yv, "w": wv, "design_id": s.design_id}
    print(f"[m2g] n_samples={len(samples)} spectra={len(spectra)} "
          f"designs={dmeta['n_designs']}", flush=True)

    X, y, w, keys, pids = gf.build_all(samples, spectra)
    attn_pred, prior_pred, _ = _load_preds(args.attn_pred)
    if args.pred_v3 and args.pred_v4 and args.pred_v5:
        deep = build_threeway(args.pred_v3, args.pred_v4, args.pred_v5,
                              args.w_v3, args.w_v4, args.w_v5)
        args.deep_component = f"3way(w3={args.w_v3},w4={args.w_v4},w5={args.w_v5})"
        print(f"[m2g] using 3-way deep component: {len(deep)} pairs", flush=True)
    else:
        deep = attn_pred
        args.deep_component = "attn_v5"
    print(f"[m2g] X={X.shape} rows={len(y)}", flush=True)

    run_m2_gbdt_ensemble(X, y, w, keys, pids, deep, prior_pred, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
