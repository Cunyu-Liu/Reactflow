#!/usr/bin/env python3
"""run_response_spectrum_m2_v1 — residual-learning response spectrum on OpenKnot M2.

Goal
----
Re-run the residual-learning + per-position median prior model on an EXPANDED
exchangeable-unit set (N >= 100).  The original development pool had only ~13
resolved publications; OpenKnot M2 provides 159 usable (puzzle x method) DESIGNS
as exchangeable units (see m2_data_v1).  Each design contributes a WT and
52-100 single-nt mutants with per-position 2A3 reactivity + error, so the changer
pool grows from hundreds to ~13.8k candidate mutants across 159 designs.

Protocol
--------
* Exchangeable unit = (puzzle, method) design.  LOO: hold out one design, train on
  the other designs' CHANGERS, predict the held design's changers.
* Changer labels come from m2_caller_v1 (per-position error, theoretical Gaussian
  null).  The M2 caller is inherently FOLD-INVARIANT (it uses only the mutant's own
  per-position errors + a theoretical null, no cross-design replicate info), so
  labels are computed once and there is no train/test information leak.
* Target = scale-invariant per-position response spectrum (response_spectrum_scinv_v1)
  aligned to a WINDOW=21 local window around each edit site.
* Model = residual learning around the per-position train-changer median prior
  (residual_spectrum_v2): pred[k] = prior[k] + delta[k], delta init 0, L2 shrink.
  Variants: wmed_spectrum (prior baseline, seed 0) + wmae_resid_spectrum (seeds 0..4).
* Neural model MUST run on CUDA (STOP if unavailable; no silent CPU fallback).

Output is keyed predictions (prediction_v2-compatible subset) + a manifest, matching
the schema used by run_response_spectrum_v2 so the same evaluator can consume it.
Development-only; fail-closed.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines_v6 import SEEDS, WINDOW, sha256_file  # noqa: E402
from run_p2_v3 import build_feature  # noqa: E402
import m2_data_v1 as m2d  # noqa: E402
import m2_caller_v1 as m2c  # noqa: E402
import response_spectrum_scinv_v1 as rss  # noqa: E402
import residual_spectrum_v2 as rsm  # noqa: E402

MODEL_ID = "response_spectrum_m2_v1"
CALLER_VERSION = "m2_caller_v1"


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
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Contract: STOP, no silent CPU fallback.")
    device = torch.device("cuda")
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[m2] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "fold_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    # ---- load + build M2 samples ----
    designs, dmeta = m2d.parse_m2_csv(args.m2_csv)
    samples = m2d.build_all_samples(designs)
    by_design: dict[str, list] = {}
    for s in samples:
        by_design.setdefault(s.design_id, []).append(s)
    resolved = sorted(by_design.keys())
    print(f"[m2] rows={dmeta['n_rows']} designs={dmeta['n_designs']} "
          f"usable_designs={len(resolved)} samples={len(samples)}", flush=True)
    if len(resolved) < 100:
        raise RuntimeError(f"N (usable designs) = {len(resolved)} < 100; "
                           "M2 expansion target not met. STOP.")
    for d in resolved:
        print(f"    {d}: {len(by_design[d])} mutants", flush=True)

    # ---- fold-invariant features (allowed inputs only) ----
    fx = {f"{s.design_id}:{s.mutA}": build_feature(s.pair, s.wt_rec, True, True, True)
          for s in samples}
    print(f"[m2] precomputed features for {len(fx)} samples", flush=True)

    # ---- fold-invariant changer labels (M2 caller: no replicate info) ----
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
        res = m2c.call_mutant(
            pid, s.wt_reactivity, s.mut_reactivity, s.wt_error, s.mut_error,
            s.eligibility_mask, null=null)
        label[pid] = res.label
        if res.label == "1":
            n_changer += 1
    print(f"[m2] total changers={n_changer} / {len(samples)} ({n_changer/len(samples):.3f}) "
          f"null_masks={len(null_cache)}", flush=True)

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
    print(f"[m2] changers with usable spectrum={len(spectra)}", flush=True)
    if not spectra:
        raise RuntimeError("no changers with usable spectra; STOP.")

    # ---- LOO over designs ----
    pred_path = out / "keyed_predictions_m2_spectrum.jsonl"
    ckpt_path = out / "fold_progress.json"
    done_folds = set()
    if args.resume and ckpt_path.exists():
        done_folds = set(json.loads(ckpt_path.read_text(encoding="utf-8")).get("completed_folds", []))
    fp = pred_path.open("a", encoding="utf-8")
    n_rows_total = (sum(1 for _ in pred_path.open("r", encoding="utf-8"))
                    if pred_path.exists() else 0)
    t_start = time.time()
    global_log = {}

    for fold, held in enumerate(resolved):
        t0 = time.time()
        if held in done_folds:
            print(f"[fold] held={held} RESUME SKIP", flush=True)
            continue
        train_pids = [pid for pid in spectra if spectra[pid]["design_id"] != held]
        he_pids = [pid for pid in spectra if spectra[pid]["design_id"] == held]
        rows = []
        if not he_pids:
            for pid in he_pids:
                for m in ("wmed_spectrum", "wmae_resid_spectrum"):
                    for seed in ([0] if m == "wmed_spectrum" else SEEDS):
                        rows.append({"pair_id": pid, "task": "magnitude_spectrum",
                                     "fold_id": held, "seed": seed, "model_variant": m,
                                     "model_id": MODEL_ID, "publication_id": held,
                                     "source_accession": spectra[pid]["source_accession"],
                                     "split_role": "development", "endpoint_version": "m2",
                                     "caller_version": CALLER_VERSION, "caller_mode": "PER_POS_ERROR",
                                     "y": None, "weight": 0.0, "raw_prediction": None,
                                     "transformed_prediction": None, "coverage_status": "NO_CALL"})
            done_folds.add(held)
            ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                                 encoding="utf-8")
            continue

        Xtr = np.stack([fx[pid] for pid in train_pids]).astype(np.float32)
        Xte = np.stack([fx[pid] for pid in he_pids]).astype(np.float32)
        Ytr = np.stack([spectra[pid]["y"] for pid in train_pids]).astype(np.float32)
        Wtr = np.stack([spectra[pid]["w"] for pid in train_pids]).astype(np.float32)
        Yte = np.stack([spectra[pid]["y"] for pid in he_pids]).astype(np.float32)
        Wte = np.stack([spectra[pid]["w"] for pid in he_pids]).astype(np.float32)

        prior, prior_cnt = rsm.per_position_prior(Ytr, Wtr)
        prior_dist = rsm.prior_distribution(Ytr, Wtr, prior)

        # baseline: per-position prior (seed 0)
        for j, pid in enumerate(he_pids):
            rows.append({"pair_id": pid, "task": "magnitude_spectrum", "fold_id": held,
                         "seed": 0, "model_variant": "wmed_spectrum",
                         "model_id": MODEL_ID, "publication_id": held,
                         "source_accession": spectra[pid]["source_accession"],
                         "split_role": "development", "endpoint_version": "m2",
                         "caller_version": CALLER_VERSION, "caller_mode": "PER_POS_ERROR",
                         "y": Yte[j].tolist(), "weight": Wte[j].tolist(),
                         "raw_prediction": prior.tolist(),
                         "transformed_prediction": prior.tolist(),
                         "coverage_status": "CALLED"})

        # residual-learning model (seeds 0..4)
        in_dim = Xtr.shape[1]
        fold_log = {"held": held, "n_train_changers": len(train_pids),
                    "n_held_changers": len(he_pids),
                    "prior": prior.tolist(), "prior_count": prior_cnt.tolist(),
                    "prior_distribution": prior_dist, "seeds": {}}
        for seed in SEEDS:
            model, tlog = rsm.train_residual(
                Xtr, Ytr, Wtr, prior, epochs=args.epochs, bs=args.bs, lr=args.lr,
                resid_pen=args.resid_pen, hidden=args.hidden, seed=seed, device=device)
            delta = rsm.predict_delta(model, Xte, device)
            pred = (np.asarray(prior) + delta).astype(np.float32)
            fold_log["seeds"][str(seed)] = tlog
            for j, pid in enumerate(he_pids):
                rows.append({"pair_id": pid, "task": "magnitude_spectrum", "fold_id": held,
                             "seed": seed, "model_variant": "wmae_resid_spectrum",
                             "model_id": MODEL_ID, "publication_id": held,
                             "source_accession": spectra[pid]["source_accession"],
                             "split_role": "development", "endpoint_version": "m2",
                             "caller_version": CALLER_VERSION, "caller_mode": "PER_POS_ERROR",
                             "y": Yte[j].tolist(), "weight": Wte[j].tolist(),
                             "raw_prediction": pred[j].tolist(),
                             "transformed_prediction": pred[j].tolist(),
                             "coverage_status": "CALLED"})

        (logdir / f"{held}.json").write_text(json.dumps(fold_log, sort_keys=True),
                                             encoding="utf-8")
        global_log[held] = {
            "n_train_changers": len(train_pids), "n_held_changers": len(he_pids),
            "prior": prior.tolist(), "prior_count": prior_cnt.tolist(),
            "seeds": {s: {"final_delta_abs_mean": fold_log["seeds"][s]["final"]["delta_abs_mean"],
                          "final_frac_delta_gt_0": fold_log["seeds"][s]["final"]["frac_delta_gt_0"],
                          "final_mae_model_train": fold_log["seeds"][s]["final"]["mae_model_train"],
                          "final_mae_prior_train": fold_log["seeds"][s]["final"]["mae_prior_train"],
                          "epochs_loss": fold_log["seeds"][s]["learning_curve"]}
                      for s in fold_log["seeds"]},
        }

        for r in rows:
            fp.write(json.dumps(r, sort_keys=True) + "\n")
        n_rows_total += len(rows)
        fp.flush()
        done_folds.add(held)
        ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                             encoding="utf-8")
        print(f"[fold] held={held} changers={len(he_pids)} "
              f"t={time.time()-t0:.1f}s", flush=True)
    fp.close()

    (out / "residual_training_log.json").write_text(
        json.dumps(global_log, indent=2, sort_keys=True), encoding="utf-8")

    dhash = sha256_file(pred_path)
    manifest = {
        "schema": "reactflow_delta.response_spectrum.m2.v1",
        "run_id": out.name, "dataset": "OpenKnot_M2", "source_url": m2d.M2_SOURCE_URL,
        "caller_version": CALLER_VERSION, "caller_mode": "PER_POS_ERROR",
        "caller_null": "theoretical Gaussian (iid N(0,1) over eligibility mask)",
        "experiment": "residual_learning_full_spectrum_response_m2",
        "scale": args.scale, "window": WINDOW, "epochs": args.epochs,
        "resid_pen": args.resid_pen, "bs": args.bs, "lr": args.lr, "hidden": args.hidden,
        "model_variants": ["wmed_spectrum", "wmae_resid_spectrum"],
        "exchangeable_unit": "puzzle_x_method_design",
        "n_usable_designs": len(resolved), "n_samples": len(samples),
        "n_changers": n_changer, "n_changers_with_spectrum": len(spectra),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {"cuda_available": True, "device_index": gpu_index, "device_name": gpu_name,
                "free_bytes": int(free_mem), "total_bytes": int(tot_mem)},
        "n_rows": n_rows_total, "keyed_predictions_sha256": dhash,
        "source_hashes": {
            "run_response_spectrum_m2_v1.py": sha256_file(Path(__file__)),
            "m2_data_v1.py": sha256_file(Path(__file__).resolve().parent / "m2_data_v1.py"),
            "m2_caller_v1.py": sha256_file(Path(__file__).resolve().parent / "m2_caller_v1.py"),
            "residual_spectrum_v2.py": sha256_file(Path(__file__).resolve().parent / "residual_spectrum_v2.py"),
            "response_spectrum_scinv_v1.py": sha256_file(Path(__file__).resolve().parent / "response_spectrum_scinv_v1.py"),
            "run_p2_v3.py": sha256_file(Path(__file__).resolve().parent / "run_p2_v3.py"),
        },
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "response_spectrum_m2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDONE -> {out}  n_rows={n_rows_total}  sha256={dhash}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
