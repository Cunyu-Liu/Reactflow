#!/usr/bin/env python3
"""smoke_v7_capacity.py — quick capacity smoke test for the v7 hypothesis.

Question: the current v5 (hidden=128, nlayers=2, epochs=30) has train MAE 0.619
vs held-out 0.659 — no overfitting, and the noise floor is only 8.8% of baseline.
This suggests the model is UNDER-CAPACITY / UNDER-TRAINED, not noise-limited.
v7 tests: hidden=256, epochs=60 (and optionally nlayers=3), on a SUBSET of folds,
compared against the SAME folds' v5 held-out WMAE (read from v5 predictions).

This is a GPU smoke test (fast): run K folds with the candidate config, report
per-fold held-out WMAE and skill vs the v5 baseline on those exact folds.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines_v6 import WINDOW, sha256_file  # noqa: E402
from run_p2_v3 import build_feature  # noqa: E402
import m2_data_v1 as m2d  # noqa: E402
import m2_caller_v1 as m2c  # noqa: E402
import response_spectrum_scinv_v1 as rss  # noqa: E402
import residual_spectrum_v2 as rsv2  # noqa: E402
import residual_spectrum_v4 as rv4  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
BASELINE = "wmed_spectrum"
POS_DIM = 7


def _load_rows(pred_path):
    rows = []
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _v5_fold_wmae(pred_path, held):
    """Return (baseline_wmae, model_wmae, skill) for a given held fold from v5 preds."""
    base_num = base_den = 0.0
    mod_num = mod_den = 0.0
    mod_seeds = defaultdict(list)  # pair_id -> list of seed preds
    for r in _load_rows(pred_path):
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        if r["fold_id"] != held:
            continue
        yv = r.get("y") or []
        wv = r.get("weight") or []
        pv = r.get("raw_prediction")
        if not (isinstance(yv, list) and isinstance(wv, list) and isinstance(pv, list)):
            continue
        y = np.array([float(a) for a, ww in zip(yv, wv) if ww], dtype=np.float64)
        p = np.array([float(a) for a, ww in zip(pv, wv) if ww], dtype=np.float64)
        if r["model_variant"] == BASELINE and r["seed"] == 0:
            base_num += float(np.abs(y - p).sum()); base_den += len(y)
        elif r["model_variant"] == "wmae_resid_attn_spectrum":
            mod_seeds[r["pair_id"]].append(p)
    bw = base_num / base_den if base_den else None
    if bw is None or not mod_seeds:
        return None
    num = den = 0.0
    for pid, ps in mod_seeds.items():
        if len(ps) < len(SEEDS):
            continue
        ens = np.mean(ps, axis=0)
        # need y again; re-read
    # recompute cleanly: store y per pair
    ys = {}
    for r in _load_rows(pred_path):
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        if r["fold_id"] != held or r["model_variant"] != BASELINE or r["seed"] != 0:
            continue
        yv = r.get("y") or []
        wv = r.get("weight") or []
        y = np.array([float(a) for a, ww in zip(yv, wv) if ww], dtype=np.float64)
        ys[r["pair_id"]] = y
    for pid, ps in mod_seeds.items():
        if len(ps) < len(SEEDS) or pid not in ys:
            continue
        ens = np.mean(ps, axis=0)
        if len(ys[pid]) != len(ens):
            continue
        num += float(np.abs(ys[pid] - ens).sum()); den += len(ys[pid])
    mw = num / den if den else None
    if mw is None:
        return None
    return {"baseline_wmae": float(bw), "model_wmae": float(mw),
            "skill": 1.0 - mw / bw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--v5-pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cuda-device", default="2")
    ap.add_argument("--folds", default="OK7a_M2_P01_Eterna,OK7a_M2_P01_MPNN-RFdiff,OK7a_M2_P05_Eterna")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resid-pen", type=float, default=1e-2)
    ap.add_argument("--nhead", type=int, default=8)
    args = ap.parse_args()

    os_env_ok = True
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Contract: STOP, no silent CPU fallback.")
    device = torch.device("cuda")
    print(f"[smoke] GPU OK: cuda_visible={args.cuda_device} name="
          f"{torch.cuda.get_device_name(0)}", flush=True)

    designs, dmeta = m2d.parse_m2_csv(args.m2_csv)
    samples = m2d.build_all_samples(designs)
    by_design = {}
    for s in samples:
        by_design.setdefault(s.design_id, []).append(s)

    fx = {f"{s.design_id}:{s.mutA}": build_feature(s.pair, s.wt_rec, True, True, True)
          for s in samples}
    null_rng = np.random.default_rng(m2c.RNG_SEED)
    null_cache = {}
    spectra = {}
    for s in samples:
        pid = f"{s.design_id}:{s.mutA}"
        mkey = tuple(s.eligibility_mask)
        null = null_cache.get(mkey)
        if null is None:
            null = m2c.gaussian_null(s.eligibility_mask, rng=null_rng)
            null_cache[mkey] = null
        res = m2c.call_mutant(pid, s.wt_reactivity, s.mut_reactivity,
                              s.wt_error, s.mut_error, s.eligibility_mask, null=null)
        if res.label != "1":
            continue
        y_vec, w_vec, scale = rss.pair_response_spectrum(
            s.wt_reactivity, s.mut_reactivity, s.eligibility_mask,
            s.edit_seq_pos, window=WINDOW, scale_mode="mean_level")
        if scale is None or sum(w_vec) <= 0:
            continue
        spectra[pid] = {"y": y_vec, "w": w_vec, "design_id": s.design_id}

    folds = [f for f in args.folds.split(",") if f]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for held in folds:
        train_pids = [pid for pid in spectra if spectra[pid]["design_id"] != held]
        he_pids = [pid for pid in spectra if spectra[pid]["design_id"] == held]
        Ytr = np.stack([spectra[pid]["y"] for pid in train_pids]).astype(np.float32)
        Wtr = np.stack([spectra[pid]["w"] for pid in train_pids]).astype(np.float32)
        prior, _ = rsv2.per_position_prior(Ytr, Wtr)
        Yte = np.stack([spectra[pid]["y"] for pid in he_pids]).astype(np.float32)
        Wte = np.stack([spectra[pid]["w"] for pid in he_pids]).astype(np.float32)
        Xtr = np.stack([fx[pid] for pid in train_pids]).astype(np.float32)
        Xte = np.stack([fx[pid] for pid in he_pids]).astype(np.float32)
        pos_tr, glob_tr = rv4.split_pos_glob(Xtr, WINDOW, POS_DIM)
        pos_te, glob_te = rv4.split_pos_glob(Xte, WINDOW, POS_DIM)
        tail_dim = glob_tr.shape[1]

        preds = []
        t0 = __import__("time").time()
        for seed in SEEDS:
            model, tlog = rv4.train_posaware_attn2(
                pos_tr, glob_tr, Ytr, Wtr, prior, POS_DIM, tail_dim, WINDOW,
                epochs=args.epochs, bs=args.bs, lr=args.lr, resid_pen=args.resid_pen,
                hidden=args.hidden, nhead=args.nhead, nlayers=args.nlayers,
                dropout=0.1, seed=seed, device=device, fast=True)
            delta = rv4.predict_posaware_attn(model, pos_te, glob_te, device)
            preds.append(np.asarray(prior) + delta)
        ens = np.mean(preds, axis=0)
        num = den = 0.0
        for j in range(len(he_pids)):
            m = Wte[j] > 0
            num += float(np.abs(Yte[j][m] - ens[j][m]).sum())
            den += float(m.sum())
        mw = num / den
        bn = bd = 0.0
        for j in range(len(he_pids)):
            m = Wte[j] > 0
            bn += float(np.abs(Yte[j][m] - prior[m]).sum())
            bd += float(m.sum())
        bw = bn / bd
        v5 = _v5_fold_wmae(args.v5_pred, held)
        results[held] = {
            "v7_model_wmae": float(mw), "v7_baseline_wmae": float(bw),
            "v7_skill": 1.0 - mw / bw,
            "v5_model_wmae": v5["model_wmae"] if v5 else None,
            "v5_skill": v5["skill"] if v5 else None,
            "wall_secs": round(__import__("time").time() - t0, 1),
        }
        print(f"[smoke] {held}: v7_skill={results[held]['v7_skill']:+.4f} "
              f"v5_skill={results[held]['v5_skill']:+.4f} "
              f"v7_wmae={results[held]['v7_model_wmae']:.4f} "
              f"v5_wmae={results[held]['v5_model_wmae']:.4f}", flush=True)

    report = {
        "schema": "reactflow_delta.response_spectrum.smoke_v7_capacity.v1",
        "config": {"hidden": args.hidden, "nlayers": args.nlayers, "epochs": args.epochs,
                   "bs": args.bs, "lr": args.lr, "resid_pen": args.resid_pen,
                   "nhead": args.nhead},
        "dataset": "OpenKnot_M2", "folds": folds, "results": results,
    }
    (out / "smoke_v7_capacity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE -> {out / 'smoke_v7_capacity.json'}")


if __name__ == "__main__":
    sys.exit(main())
