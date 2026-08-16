#!/usr/bin/env python3
"""m2r_transfer_puzzle_v1.py — puzzle-level M2R LOO with leak-free transfer.

The design-level M2 OOF preds (used by m2r_transfer_v1.py) are leak-free for a
design-level M2R claim, but for a PUZZLE-level M2R claim they would leak: each
design's M2 prediction came from an M2 model trained on its sibling designs of
the SAME puzzle.  This script uses the PUZZLE-LEVEL M2 OOF predictions
(m2_attn_puzzle_20260817, produced by run_response_spectrum_m2_attn_puzzle_v1:
train 19 puzzles -> predict all 8 designs of the held-out puzzle), so the
transfer features for a held-out puzzle carry NO information from that puzzle.

Evaluation: puzzle-level leave-one-out over 20 puzzles.
  * existing features (230 dims incl. M2_structure)
  * existing + 6 M2-transfer features (puzzle-level OOF)
  * + GBDT+Ridge blend (a=0.80)

Also reports the "leak diagnostic": puzzle-level M2R evaluated with the
DESIGN-level M2 OOF transfer features, which quantifies how much the design-
level transfer would overstate the puzzle-level gain if used carelessly.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_transfer_v1 as tr

SEED = 20260816


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def puzzle_of(keys, samples):
    """Map each sample index to its puzzle id."""
    return np.array([s.puzzle for s in samples])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--m2-pred-puzzle", required=True,
                    help="PUZZLE-level M2 keyed_predictions jsonl (leak-free)")
    ap.add_argument("--m2-pred-design", default=None,
                    help="DESIGN-level M2 keyed_predictions jsonl (leak diagnostic)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load M2R ----
    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    m2r.attach_m2_structure(designs, args.m2_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs) if s.rescue_factor is not None]
    X, y, keys, names = m2rf.build_all(samples)
    keys = np.array(keys)
    sample_puzzles = np.array([s.puzzle for s in samples])
    puzzles = sorted(set(sample_puzzles.tolist()))
    print(f"[m2r_trpz] n_samples={len(y)} n_puzzles={len(puzzles)} X={X.shape}",
          flush=True)

    # ---- load M2 puzzle-level OOF ----
    m2_oof_pz = tr.load_m2_oof(args.m2_pred_puzzle)
    m2_design_key = {}
    for did in m2_oof_pz:
        parts = did.split("_")
        if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
            m2_design_key[(parts[2], "_".join(parts[3:]))] = did
    print(f"[m2r_trpz] puzzle-level M2 designs: {len(m2_oof_pz)} "
          f"mapped {len(m2_design_key)}", flush=True)

    X_tr_pz = tr.build_transfer_features(samples, m2_oof_pz, m2_design_key)
    nz = (np.abs(X_tr_pz).sum(axis=1) > 0).mean()
    print(f"[m2r_trpz] X_tr(puzzle)={X_tr_pz.shape} nonzero_frac={nz:.3f}", flush=True)

    # ---- optional design-level OOF (leak diagnostic) ----
    X_tr_dl = None
    if args.m2_pred_design:
        m2_oof_dl = tr.load_m2_oof(args.m2_pred_design)
        m2_design_key2 = {}
        for did in m2_oof_dl:
            parts = did.split("_")
            if len(parts) >= 4 and parts[0] == "OK7a" and parts[1] == "M2":
                m2_design_key2[(parts[2], "_".join(parts[3:]))] = did
        X_tr_dl = tr.build_transfer_features(samples, m2_oof_dl, m2_design_key2)
        print(f"[m2r_trpz] X_tr(design)={X_tr_dl.shape}", flush=True)

    # ---- baseline ----
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    print(f"[m2r_trpz] baseline MAE={mae_bl:.4f}", flush=True)

    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    # ---- puzzle-level LOO helper ----
    def loo_puzzle(X_use, label):
        preds = np.zeros(len(y))
        for held in puzzles:
            m = sample_puzzles != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            g = lgb.LGBMRegressor(
                n_estimators=args.trees, max_depth=args.depth,
                random_state=SEED, verbose=-1, n_jobs=2)
            g.fit(X_use[m], y[m])
            preds[~m] = g.predict(X_use[~m])
        return preds

    def loo_ridge(X_use):
        preds = np.zeros(len(y))
        for held in puzzles:
            m = sample_puzzles != held
            if m.sum() <= 10:
                preds[~m] = y_med
                continue
            r = Ridge(alpha=1.0).fit(X_use[m], y[m])
            preds[~m] = r.predict(X_use[~m])
        return preds

    results = {}
    t0 = time.time()

    # existing (230 dims)
    pred_ex = loo_puzzle(X, "existing")
    results["existing_230"] = {
        "mae": _mae(y, pred_ex), "skill": _skill(_mae(y, pred_ex), mae_bl),
        "r2": _r2(y, pred_ex)}

    # existing + puzzle-level transfer
    X_comb_pz = np.concatenate([X, X_tr_pz], axis=1)
    pred_pz = loo_puzzle(X_comb_pz, "puzzle_transfer")
    results["existing_plus_puzzle_transfer"] = {
        "mae": _mae(y, pred_pz), "skill": _skill(_mae(y, pred_pz), mae_bl),
        "r2": _r2(y, pred_pz)}

    # + blend (a=0.80) with puzzle-level transfer
    ridge_pz = loo_ridge(X_comb_pz)
    blend_pz = 0.80 * pred_pz + 0.20 * ridge_pz
    results["puzzle_transfer_blend_a80"] = {
        "mae": _mae(y, blend_pz), "skill": _skill(_mae(y, blend_pz), mae_bl),
        "r2": _r2(y, blend_pz)}

    # leak diagnostic: existing + DESIGN-level transfer evaluated at puzzle level
    if X_tr_dl is not None:
        X_comb_dl = np.concatenate([X, X_tr_dl], axis=1)
        pred_dl = loo_puzzle(X_comb_dl, "design_transfer")
        results["existing_plus_design_transfer_LEAKY"] = {
            "mae": _mae(y, pred_dl), "skill": _skill(_mae(y, pred_dl), mae_bl),
            "r2": _r2(y, pred_dl)}

    wall = round(time.time() - t0, 1)

    # ---- per-puzzle breakdown for the headline (puzzle-level transfer) ----
    per_puzzle = {}
    for held in puzzles:
        m = sample_puzzles == held
        if m.sum() == 0:
            continue
        s_ex = _skill(_mae(y[m], pred_ex[m]), mae_bl)
        s_pz = _skill(_mae(y[m], pred_pz[m]), mae_bl)
        per_puzzle[held] = {"n": int(m.sum()),
                            "existing_skill": float(s_ex),
                            "puzzle_transfer_skill": float(s_pz),
                            "transfer_gain_pp": float(s_pz - s_ex)}

    report = {
        "schema": "reactflow_delta.m2r_transfer_puzzle.v1",
        "dataset": "OpenKnot_M2R",
        "source_task": "M2 response-spectrum (attn v5), PUZZLE-level OOF",
        "n_samples": len(y), "n_puzzles": len(puzzles),
        "n_features_existing": int(X.shape[1]),
        "n_features_transfer": int(X_tr_pz.shape[1]),
        "trees": args.trees, "depth": args.depth,
        "baseline_mae": mae_bl,
        "transfer_nonzero_frac": float(nz),
        "results": results,
        "per_puzzle": per_puzzle,
        "wall_seconds": wall,
    }
    # ---- save OOF preds for the puzzle-block perm test ----
    save = {"pred_ex": pred_ex, "pred_pz": pred_pz, "blend_pz": blend_pz,
            "y": y, "sample_puzzles": sample_puzzles}
    if X_tr_dl is not None:
        save["pred_dl"] = pred_dl
    np.savez(out / "m2r_transfer_puzzle_oof.npz", **save)
    (out / "m2r_transfer_puzzle_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== M2R puzzle-level transfer ===")
    for k, v in results.items():
        print(f"  {k:42s} skill={v['skill']:+.4f} R2={v['r2']:.4f} MAE={v['mae']:.4f}")
    gains = [p["transfer_gain_pp"] for p in per_puzzle.values()]
    print(f"\n  per-puzzle transfer gain: mean={np.mean(gains):+.4f}pp "
          f"pct_pos={(np.array(gains)>0).mean():.3f} "
          f"min={np.min(gains):+.4f} max={np.max(gains):+.4f}")
    print(f"  wall={wall}s DONE -> {out / 'm2r_transfer_puzzle_report.json'}")


if __name__ == "__main__":
    sys.exit(main())
