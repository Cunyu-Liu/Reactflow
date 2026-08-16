#!/usr/bin/env python3
"""run_response_spectrum_m2_attn_puzzle_v1 — PUZZLE-LEVEL OOF for the M2 attn
response spectrum, mirroring run_response_spectrum_m2_attn_v1 EXACTLY except
that the fold unit is the PUZZLE instead of the design.

Why this exists (leak-free audit of M2R cross-task transfer):
  The M2R transfer features use M2 model OOF predictions.  The design-level M2
  OOF gives, for each design D, a prediction from an M2 model trained on all
  OTHER designs — including D's sibling designs of the SAME puzzle.  For an
  M2R design-level claim this is leak-free.  But if we want to test whether the
  transfer gain GENERALIZES at the M2R puzzle level (held-out puzzle), the
  design-level M2 OOF would leak puzzle information (the M2 model saw other
  designs of that puzzle).

  This runner produces PUZZLE-LEVEL M2 OOF predictions: for each held-out
  puzzle P, train an M2 model on all designs of the OTHER 19 puzzles and
  predict all designs of P.  These predictions are leak-free at puzzle level.

Fold semantics: fold_id = puzzle id (e.g. "P01").  Same training config as the
design-level runner (wmae_resid_attn_spectrum, seeds 0..4, baseline
wmed_spectrum seed 0, WINDOW=21, POS_DIM=7, position-aware self-attention v4).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines_v6 import SEEDS, WINDOW, sha256_file  # noqa: E402
from run_p2_v3 import build_feature  # noqa: E402
import m2_data_v1 as m2d  # noqa: E402
import m2_caller_v1 as m2c  # noqa: E402
import response_spectrum_scinv_v1 as rss  # noqa: E402
import residual_spectrum_v2 as rsv2  # noqa: E402
import residual_spectrum_v4 as rv4  # noqa: E402

MODEL_ID = "response_spectrum_m2_attn_v1"
CALLER_VERSION = "m2_caller_v1"
MODEL_VARIANT = "wmae_resid_attn_spectrum"
BASELINE = "wmed_spectrum"
POS_DIM = 7


def puzzle_of(design_id: str) -> str:
    """'OK7a_M2_P01_Eterna' -> 'P01'."""
    parts = design_id.split("_")
    return parts[2] if len(parts) >= 3 else design_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cuda-device", default="6")
    ap.add_argument("--scale", choices=("mean_level", "mad"), default="mean_level")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--resid-pen", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--head-hidden", type=int, default=32)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Contract: STOP, no silent CPU fallback.")
    device = torch.device("cuda")
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[m2attn_pz] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "fold_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    # ---- load + build M2 samples (identical to design-level runner) ----
    designs, dmeta = m2d.parse_m2_csv(args.m2_csv)
    samples = m2d.build_all_samples(designs)
    by_design: dict[str, list] = {}
    for s in samples:
        by_design.setdefault(s.design_id, []).append(s)
    resolved = sorted(by_design.keys())
    print(f"[m2attn_pz] rows={dmeta['n_rows']} designs={dmeta['n_designs']} "
          f"usable_designs={len(resolved)} samples={len(samples)}", flush=True)
    if len(resolved) < 100:
        raise RuntimeError(f"N (usable designs) = {len(resolved)} < 100; STOP.")

    # puzzle set
    puzzles = sorted(set(puzzle_of(d) for d in resolved))
    print(f"[m2attn_pz] puzzles={len(puzzles)}: {puzzles}", flush=True)

    # ---- fold-invariant features ----
    fx = {f"{s.design_id}:{s.mutA}": build_feature(s.pair, s.wt_rec, True, True, True)
          for s in samples}

    # ---- fold-invariant changer labels ----
    null_cache: dict[tuple, np.ndarray] = {}
    null_rng = np.random.default_rng(m2c.RNG_SEED)
    label = {}
    n_changer = 0
    for s in samples:
        pid = f"{s.design_id}:{s.mutA}"
        mkey = tuple(s.eligibility_mask)
        null = null_cache.get(mkey)
        if null is None:
            null = m2c.gaussian_null(s.eligibility_mask, rng=null_rng)
            null_cache[mkey] = null
        res = m2c.call_mutant(pid, s.wt_reactivity, s.mut_reactivity,
                              s.wt_error, s.mut_error, s.eligibility_mask, null=null)
        label[pid] = res.label
        if res.label == "1":
            n_changer += 1

    # ---- spectra for changers ----
    spectra = {}
    for s in samples:
        pid = f"{s.design_id}:{s.mutA}"
        if label[pid] != "1":
            continue
        y_vec, w_vec, scale = rss.pair_response_spectrum(
            s.wt_reactivity, s.mut_reactivity, s.eligibility_mask,
            s.edit_seq_pos, window=WINDOW, scale_mode=args.scale)
        if scale is None or sum(w_vec) <= 0:
            continue
        spectra[pid] = {"y": y_vec, "w": w_vec, "design_id": s.design_id,
                        "source_accession": s.design_id}
    print(f"[m2attn_pz] changers with usable spectrum={len(spectra)}", flush=True)
    if not spectra:
        raise RuntimeError("no changers with usable spectra; STOP.")

    # ---- puzzle-level LOO ----
    pred_path = out / "keyed_predictions_m2_attn_puzzle.jsonl"
    ckpt_path = out / "fold_progress.json"
    done_folds = set()
    if args.resume and ckpt_path.exists():
        done_folds = set(json.loads(ckpt_path.read_text(encoding="utf-8")).get("completed_folds", []))
    fp = pred_path.open("a", encoding="utf-8")
    n_rows_total = (sum(1 for _ in pred_path.open("r", encoding="utf-8"))
                    if pred_path.exists() else 0)
    t_start = time.time()
    global_log = {}

    for fold, held_puzzle in enumerate(puzzles):
        t0 = time.time()
        if held_puzzle in done_folds:
            print(f"[fold] held={held_puzzle} RESUME SKIP", flush=True)
            continue
        # train on ALL designs whose puzzle != held_puzzle
        train_pids = [pid for pid in spectra if puzzle_of(spectra[pid]["design_id"]) != held_puzzle]
        he_pids = [pid for pid in spectra if puzzle_of(spectra[pid]["design_id"]) == held_puzzle]
        rows = []
        if not he_pids:
            print(f"[fold] held={held_puzzle} no held changers; skip", flush=True)
            done_folds.add(held_puzzle)
            ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                                 encoding="utf-8")
            continue

        Ytr = np.stack([spectra[pid]["y"] for pid in train_pids]).astype(np.float32)
        Wtr = np.stack([spectra[pid]["w"] for pid in train_pids]).astype(np.float32)
        prior, prior_cnt = rsv2.per_position_prior(Ytr, Wtr)
        prior_dist = rsv2.prior_distribution(Ytr, Wtr, prior)
        Yte = np.stack([spectra[pid]["y"] for pid in he_pids]).astype(np.float32)
        Wte = np.stack([spectra[pid]["w"] for pid in he_pids]).astype(np.float32)

        Xtr_base = np.stack([fx[pid] for pid in train_pids]).astype(np.float32)
        Xte_base = np.stack([fx[pid] for pid in he_pids]).astype(np.float32)
        pos_tr, glob_tr = rv4.split_pos_glob(Xtr_base, WINDOW, POS_DIM)
        pos_te, glob_te = rv4.split_pos_glob(Xte_base, WINDOW, POS_DIM)
        tail_dim = glob_tr.shape[1]

        # baseline: per-position prior (seed 0)
        for j, pid in enumerate(he_pids):
            rows.append({"pair_id": pid, "task": "magnitude_spectrum", "fold_id": held_puzzle,
                         "seed": 0, "model_variant": BASELINE,
                         "model_id": MODEL_ID, "publication_id": held_puzzle,
                         "source_accession": spectra[pid]["source_accession"],
                         "split_role": "development", "endpoint_version": "m2",
                         "caller_version": CALLER_VERSION, "caller_mode": "PER_POS_ERROR",
                         "y": Yte[j].tolist(), "weight": Wte[j].tolist(),
                         "raw_prediction": prior.tolist(),
                         "transformed_prediction": prior.tolist(),
                         "coverage_status": "CALLED"})

        fold_log = {"held": held_puzzle, "n_train_changers": len(train_pids),
                    "n_held_changers": len(he_pids),
                    "prior": prior.tolist(), "prior_count": prior_cnt.tolist(),
                    "prior_distribution": prior_dist, "seeds": {}}
        for seed in SEEDS:
            model, tlog = rv4.train_posaware_attn2(
                pos_tr, glob_tr, Ytr, Wtr, prior, POS_DIM, tail_dim, WINDOW,
                epochs=args.epochs, bs=args.bs, lr=args.lr, resid_pen=args.resid_pen,
                hidden=args.hidden, head_hidden=args.head_hidden, nhead=args.nhead,
                nlayers=args.nlayers, dropout=args.dropout, seed=seed, device=device,
                fast=args.fast)
            delta = rv4.predict_posaware_attn(model, pos_te, glob_te, device)
            pred = (np.asarray(prior) + delta).astype(np.float32)
            fold_log["seeds"][str(seed)] = tlog
            for j, pid in enumerate(he_pids):
                rows.append({"pair_id": pid, "task": "magnitude_spectrum", "fold_id": held_puzzle,
                             "seed": seed, "model_variant": MODEL_VARIANT,
                             "model_id": MODEL_ID, "publication_id": held_puzzle,
                             "source_accession": spectra[pid]["source_accession"],
                             "split_role": "development", "endpoint_version": "m2",
                             "caller_version": CALLER_VERSION, "caller_mode": "PER_POS_ERROR",
                             "y": Yte[j].tolist(), "weight": Wte[j].tolist(),
                             "raw_prediction": pred[j].tolist(),
                             "transformed_prediction": pred[j].tolist(),
                             "coverage_status": "CALLED"})

        (logdir / f"{held_puzzle}.json").write_text(json.dumps(fold_log, sort_keys=True),
                                                    encoding="utf-8")
        global_log[held_puzzle] = {
            "n_train_changers": len(train_pids), "n_held_changers": len(he_pids),
            "prior": prior.tolist(), "prior_count": prior_cnt.tolist(),
            "seeds": {s: {"final_delta_abs_mean": fold_log["seeds"][s]["final"]["delta_abs_mean"],
                          "final_frac_delta_gt_0": fold_log["seeds"][s]["final"]["frac_delta_gt_0"],
                          "final_mae_model_train": fold_log["seeds"][s]["final"]["mae_model_train"],
                          "final_mae_prior_train": fold_log["seeds"][s]["final"]["mae_prior_train"]}
                      for s in fold_log["seeds"]},
        }

        for r in rows:
            fp.write(json.dumps(r, sort_keys=True) + "\n")
        n_rows_total += len(rows)
        fp.flush()
        done_folds.add(held_puzzle)
        ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                             encoding="utf-8")
        print(f"[fold] held={held_puzzle} train_ch={len(train_pids)} "
              f"held_ch={len(he_pids)} t={time.time()-t0:.1f}s", flush=True)
    fp.close()

    # ---- manifest ----
    pred_hash = sha256_file(pred_path)
    n_rows_manifest = sum(1 for _ in pred_path.open("r", encoding="utf-8"))
    manifest = {
        "schema": "reactflow_delta.response_spectrum.m2_attn_puzzle.v1",
        "dataset": "OpenKnot_M2",
        "run_id": out.name,
        "fold_unit": "puzzle",
        "exchangeable_unit": "puzzle",
        "n_usable_designs": len(resolved),
        "n_puzzles": len(puzzles),
        "model_variants": [BASELINE, MODEL_VARIANT],
        "seeds": SEEDS,
        "n_rows": n_rows_manifest,
        "keyed_predictions": str(pred_path.name),
        "keyed_predictions_sha256": pred_hash,
        "window": WINDOW,
        "scale": args.scale,
        "model_config": {"hidden": args.hidden, "head_hidden": args.head_hidden,
                         "nhead": args.nhead, "nlayers": args.nlayers,
                         "dropout": args.dropout, "epochs": args.epochs,
                         "bs": args.bs, "lr": args.lr, "resid_pen": args.resid_pen},
        "wall_seconds": round(time.time() - t_start, 1),
        "completed_folds": sorted(done_folds),
        "gpu": {"cuda_available": torch.cuda.is_available(),
                "device_index": gpu_index, "device_name": gpu_name},
    }
    (out / "response_spectrum_m2_attn_puzzle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[m2attn_pz] DONE -> {out}")
    print(f"  puzzles={len(puzzles)} done_folds={len(done_folds)} "
          f"rows={n_rows_manifest} wall={manifest['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
