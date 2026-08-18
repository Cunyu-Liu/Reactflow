#!/usr/bin/env python3
"""m2_gbdt_puzzle_ensemble_v1.py — PUZZLE-LEVEL generalization audit of the M2
response-spectrum GBDT + deep ensemble.

Mirrors m2_gbdt_ensemble_v1.py (design-LOO) EXCEPT the fold unit is the PUZZLE:
for each held-out puzzle P, the leak-free per-position GBDT is trained on all
rows from the OTHER 19 puzzles and predicts all rows of P.  This is the
strongest leak-free generalization claim (a completely unseen puzzle).

Components (ALL puzzle-level leak-free):
  * GBDT   : m2_gbdt_features_v1 leak-free 31-dim features, L1 LightGBM,
             retrained per held puzzle (train 19 -> predict 1)
  * deep   : PUZZLE-LEVEL OOF mu-ensemble of the attn model
             (/mnt/cunyuliu/m2_attn_puzzle_20260817/keyed_predictions_m2_attn_puzzle.jsonl,
             trained on 19 puzzles -> predict the held puzzle, 5-seed mean)
  * prior  : PUZZLE-LEVEL per-position median baseline (seed 0 of the same file)

NOTE on the deep component: only the attn model (1-layer, nlayers=1) has
puzzle-level OOF so far; the design-level 3-way (v3+v4+v5) is NOT used here.
A full puzzle-level 3-way would require puzzle-level OOF for v3/v4/v5 (GPU
retraining).  The deep component is therefore labelled "attn_1layer_puzzle".

Metric: pooled WMAE skill vs the puzzle-level median prior, with the PUZZLE as
the exchangeable unit (puzzle-block bootstrap CI + puzzle-block permutation p).
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


def puzzle_of(design_id: str) -> str:
    """'OK7a_M2_P01_Eterna' -> 'P01'."""
    parts = design_id.split("_")
    return parts[2] if len(parts) >= 3 else design_id


def _wmae(y, w, pred):
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    num = float(np.sum(w * np.abs(y - pred)))
    den = float(np.sum(w))
    return num / den if den > 0 else float("nan")


def _skill(mae_model, mae_base):
    return 1.0 - mae_model / mae_base if mae_base > 0 else 0.0


def _load_puzzle_preds(pred_path):
    """(attn_mu, prior, designs) keyed by "design_id:mutA" from the PUZZLE OOF.

    attn_mu = mean over SEEDS of the attn raw_prediction (21-vector).
    prior   = wmed_spectrum seed-0 raw_prediction (21-vector).

    Pairs with INCOMPLETE seed coverage (len(arrs) != len(SEEDS)) are dropped
    and counted in the third element (excluded_by_seed) so the caller can
    document coverage honestly (e.g. the P01 duplicate-write artifact in
    m2_attn_puzzle_20260817).
    """
    attn = defaultdict(list)
    prior = {}
    designs = {}
    n_incomplete = 0
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        pid = r["pair_id"]
        designs[pid] = r["source_accession"] or pid.split(":")[0]
        pv = r.get("raw_prediction") or []
        if not isinstance(pv, list):
            continue
        if r["model_variant"] == BASELINE and r["seed"] == 0:
            prior[pid] = np.array([float(x) for x in pv], dtype=np.float64)
        elif r["model_variant"] == ATTN_VARIANT:
            attn[pid].append(np.array([float(x) for x in pv], dtype=np.float64))
    attn_mu = {}
    for pid, arrs in attn.items():
        if len(arrs) == len(SEEDS):
            attn_mu[pid] = np.mean(arrs, axis=0)
        else:
            n_incomplete += 1
    if n_incomplete:
        print(f"[m2gpz] WARNING: {n_incomplete} attn pairs have incomplete "
              f"seed coverage (excluded from the mu-ensemble)", flush=True)
    return attn_mu, prior, designs, n_incomplete


def analyze_puzzle(y, w, pz_keys, pred, base, n_perm=300, n_boot=300, seed=SEED):
    """Puzzle-block bootstrap CI + puzzle-block permutation p.

    Exchangeable unit = PUZZLE (block permutation of the model preds across
    puzzle blocks).
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(pz_keys.tolist())))
    real = _skill(_wmae(y, w, pred), _wmae(y, w, base))

    blocks = {}
    for d in ids:
        m = pz_keys == d
        blocks[d] = {"y": y[m], "w": w[m], "b": base[m], "p": pred[m]}

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
            "permutation_p": float(p), "n_puzzles": int(len(ids)),
            "n_perm": n_perm, "n_boot": n_boot}


def _loo_puzzle_lgb(X, y, sample_puzzles, puzzles, seed=SEED):
    """Puzzle-level leave-one-puzzle-out L1 LightGBM: train on 19 puzzles,
    predict the held-out puzzle.  Returns per-row OOF predictions."""
    import lightgbm as lgb
    preds = np.zeros(len(y))
    for held in puzzles:
        m = sample_puzzles != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        g = lgb.LGBMRegressor(objective="l1", random_state=seed, verbose=-1,
                              n_jobs=8, **CFG)
        g.fit(X[m], y[m])
        preds[~m] = g.predict(X[~m])
    return preds


def run_m2_gbdt_puzzle_ensemble(X, y, w, keys, pids, deep_pred, prior_pred, args) -> dict:
    """Puzzle-level LOPO GBDT + puzzle-level deep blend."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sample_puzzles = np.array([puzzle_of(k) for k in keys])
    puzzles = sorted(set(sample_puzzles.tolist()))
    print(f"[m2gpz] n_rows={len(y)} n_puzzles={len(puzzles)}", flush=True)

    # ---- puzzle-level LOPO L1 GBDT (leak-free, 19 train -> 1 held) ----
    t0 = time.time()
    gbdt_pred = _loo_puzzle_lgb(X, y, sample_puzzles, puzzles)
    print(f"[m2gpz] puzzle-LOPO GBDT done wall={time.time()-t0:.0f}s", flush=True)

    # ---- align deep + prior per (pair_id, k) ----
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
    print(f"[m2gpz] aligned deep/prior rows={int(mask.sum())}/{len(y)}", flush=True)

    ym = y[mask]; wm = w[mask]
    prior_m = prior_align[mask]; deep_m = deep_align[mask]
    gbdt_m = gbdt_pred[mask]
    keys_m = keys[mask]
    pz_m = sample_puzzles[mask]
    puzzles_m = sorted(set(pz_m.tolist()))

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

    ALPHA_HL = getattr(args, "alpha", 0.5)
    blend_hl = ALPHA_HL * gbdt_m + (1.0 - ALPHA_HL) * deep_m
    mae_hl = _wmae(ym, wm, blend_hl)
    skill_hl = _skill(mae_hl, mae_prior)
    sig = analyze_puzzle(ym, wm, pz_m, blend_hl, prior_m,
                         n_perm=args.n_perm, n_boot=args.n_boot, seed=SEED)

    # LOO-exclusion (leave one puzzle out) blend-vs-deep gain
    gains = []
    for held in puzzles_m:
        mm = pz_m != held
        if mm.sum() < 100:
            continue
        g = (_skill(_wmae(ym[mm], wm[mm], blend_hl[mm]), _wmae(ym[mm], wm[mm], prior_m[mm]))
             - _skill(_wmae(ym[mm], wm[mm], deep_m[mm]), _wmae(ym[mm], wm[mm], prior_m[mm])))
        gains.append(g)
    gains = np.array(gains)

    # per-puzzle blend-vs-deep gain
    pzskills = defaultdict(lambda: {"y": [], "w": [], "b": [], "a": [], "m": []})
    for i in range(len(ym)):
        d = pz_m[i]
        pzskills[d]["y"].append(ym[i]); pzskills[d]["w"].append(wm[i])
        pzskills[d]["b"].append(prior_m[i]); pzskills[d]["a"].append(deep_m[i])
        pzskills[d]["m"].append(blend_hl[i])
    pd_gains = []
    for d, g in pzskills.items():
        yv = np.array(g["y"]); wv = np.array(g["w"])
        bb = np.array(g["b"]); aa = np.array(g["a"]); mm = np.array(g["m"])
        sk_a = _skill(_wmae(yv, wv, aa), _wmae(yv, wv, bb))
        sk_m = _skill(_wmae(yv, wv, mm), _wmae(yv, wv, bb))
        pd_gains.append(sk_m - sk_a)
    pd_gains = np.array(pd_gains)

    report = {
        "schema": "reactflow_delta.response_spectrum.m2_gbdt_puzzle_ensemble.v1",
        "dataset": "OpenKnot_M2", "fold_unit": "puzzle",
        "exchangeable_unit": "puzzle",
        "baseline": BASELINE, "attn_variant": ATTN_VARIANT,
        "deep_component": "attn_1layer_puzzle_oof (5-seed mu; full 3-way needs puzzle-level v3/v4/v5 OOF)",
        "n_rows_total": int(len(y)), "n_rows_matched": int(mask.sum()),
        "n_puzzles": len(puzzles_m),
        "n_features": int(X.shape[1]), "alpha_headline": ALPHA_HL,
        "gbdt_cfg": CFG,
        "results": {
            "gbdt_puzzle": {"mae": mae_gbdt, "skill": skill_gbdt},
            "deep_attn_puzzle": {"mae": mae_attn, "skill": skill_attn},
            "blend": {"mae": mae_hl, "skill": skill_hl, "sig": sig},
        },
        "blend_curve": blends,
        "blend_vs_deep": {
            "pooled_gain_pp": float((skill_hl - skill_attn) * 100),
            "per_puzzle_mean_pp": float(pd_gains.mean() * 100),
            "per_puzzle_pct_positive": float((pd_gains > 0).mean()),
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
    (out / "m2_gbdt_puzzle_ensemble_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2_gbdt_puzzle_ensemble_oof.npz",
             gbdt=gbdt_m, deep=deep_m, prior=prior_m, blend=blend_hl,
             y=ym, w=wm, keys=keys_m, puzzles=pz_m)

    print(f"\n=== M2 GBDT + deep ensemble (PUZZLE-level LOPO) ===")
    print(f"GBDT  : skill={skill_gbdt:+.4f} mae={mae_gbdt:.4f}")
    print(f"deep  : skill={skill_attn:+.4f} mae={mae_attn:.4f}")
    print(f"blend : skill={skill_hl:+.4f} mae={mae_hl:.4f} "
          f"ci=({sig['ci_low']:.4f},{sig['ci_high']:.4f}) p={sig['permutation_p']:.4f}")
    print("blend curve: " + ", ".join(f"a={a}:{v['skill']:+.4f}" for a, v in blends.items()))
    g = report["blend_vs_deep"]
    print(f"blend-vs-attn: pooled={g['pooled_gain_pp']:+.2f}pp "
          f"per-puzzle={g['per_puzzle_mean_pp']:+.2f}pp ({g['per_puzzle_pct_positive']:.2f} pos)")
    loo = g["loo_exclusion"]
    print(f"LOO-exclusion: mean={loo['gain_mean_pp']:+.2f}pp "
          f"range=[{loo['gain_min_pp']:+.2f},{loo['gain_max_pp']:+.2f}]pp "
          f"pct_pos={loo['pct_positive']:.3f}")
    print(f"wall={report['wall_seconds']:.0f}s DONE -> "
          f"{out / 'm2_gbdt_puzzle_ensemble_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--puzzle-pred", required=True,
                    help="PUZZLE-level keyed predictions jsonl "
                         "(m2_attn_puzzle_20260817)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
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
    print(f"[m2gpz] n_samples={len(samples)} spectra={len(spectra)} "
          f"designs={dmeta['n_designs']}", flush=True)

    X, y, w, keys, pids = gf.build_all(samples, spectra)
    deep, prior, _, n_incomplete = _load_puzzle_preds(args.puzzle_pred)
    print(f"[m2gpz] X={X.shape} rows={len(y)} "
          f"deep_pairs={len(deep)} excluded_by_seed={n_incomplete}", flush=True)

    rep = run_m2_gbdt_puzzle_ensemble(X, y, w, keys, pids, deep, prior, args)
    if n_incomplete:
        # append the honest coverage note (also self-documented in the script)
        rp = Path(args.out) / "m2_gbdt_puzzle_ensemble_report.json"
        d = json.loads(rp.read_text(encoding="utf-8"))
        d["n_incomplete_seed_pairs_excluded"] = n_incomplete
        d.setdefault("coverage_note",
                     "Pairs with incomplete seed coverage in the upstream "
                     "puzzle-level attn OOF are excluded from the mu-ensemble; "
                     "the affected puzzle is dropped from the matched set.")
        rp.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
