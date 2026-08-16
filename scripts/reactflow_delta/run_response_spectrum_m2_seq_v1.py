#!/usr/bin/env python3
"""run_response_spectrum_m2_seq_v1 — residual MLP + STRICT-legal global-sequence
features on the expanded OpenKnot M2 dataset.

Method-level follow-up to run_response_spectrum_m2_v1 (plain residual MLP).
That run showed REAL, non-saturated WMAE skill of ~+7% per seed / +8.88% mu-ensemble
(baseline 0.754 -> model 0.699, perm_p=0.0033) over the per-position median prior
across 159 exchangeable (puzzle x method) designs.  On the ORIGINAL data the strongest
cross-publication transfer came from adding STRICT-legal FULL-SEQUENCE features
(k-mer composition + ViennaRNA WT folding, global_seq_features_v1) to the residual
model.  This variant tests whether that same representation improves the *fast*
residual MLP on M2 — the computationally feasible alternative to the (infeasible,
~33+ day) DeepSets+global-seq run.

Everything else is identical to run_response_spectrum_m2_v1 (M2 caller PER_POS_ERROR,
LOO over designs, WINDOW=21 scale-invariant response target, residual penalty,
CUDA-only).  The only change is that each sample's input feature vector is
[local window + pair tail | global sequence features], deduplicated by WT sequence
so the ViennaRNA fold is computed once per unique WT.

Output: keyed predictions with model_variant
  * wmed_spectrum              (per-position median prior baseline, seed 0)
  * wmae_resid_seq_spectrum    (residual MLP + global seq, seeds 0..4)
consumed by the same horizontal evaluators (compare_response_spectrum_v1 /
m2_horizontal_ensemble_report) so it can be compared head-to-head against
wmae_resid_spectrum on the same M2 folds.

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
import global_seq_features_v1 as gsf  # noqa: E402

MODEL_ID = "response_spectrum_m2_seq_v1"
CALLER_VERSION = "m2_caller_v1"
MODEL_VARIANT = "wmae_resid_seq_spectrum"


def build_seq_features(samples):
    """Fold-invariant feature assembly for M2 samples: local window + pair tail,
    concatenated with STRICT-legal global-sequence features deduplicated by WT seq.

    Returns fx: dict keyed by "{design_id}:{mutA}" -> np.ndarray (local_dim + GLOBAL_SEQ_DIM).
    Deterministic, fold-invariant, no held-out target leakage.  Unit-testable.
    """
    seq_cache: dict = {}
    fx: dict = {}
    for s in samples:
        pid = f"{s.design_id}:{s.mutA}"
        local = build_feature(s.pair, s.wt_rec, True, True, True)
        seq = (s.wt_rec.get("canonical_sequence") or "").upper()
        if seq not in seq_cache:
            seq_cache[seq] = gsf.build_global_seq_feature(s.wt_rec, s.pair)
        fx[pid] = np.concatenate([local, seq_cache[seq]])
    return fx


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
    print(f"[m2_seq] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "fold_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    # ---- load + build M2 samples (identical to run_response_spectrum_m2_v1) ----
    designs, dmeta = m2d.parse_m2_csv(args.m2_csv)
    samples = m2d.build_all_samples(designs)
    by_design: dict[str, list] = {}
    for s in samples:
        by_design.setdefault(s.design_id, []).append(s)
    resolved = sorted(by_design.keys())
    print(f"[m2_seq] rows={dmeta['n_rows']} designs={dmeta['n_designs']} "
          f"usable_designs={len(resolved)} samples={len(samples)}", flush=True)
    if len(resolved) < 100:
        raise RuntimeError(f"N (usable designs) = {len(resolved)} < 100; "
                           "M2 expansion target not met. STOP.")

    # ---- fold-invariant features: local window + pair tail | global sequence ----
    fx = build_seq_features(samples)
    n_unique_seq = len({s.wt_rec.get("canonical_sequence") for s in samples})
    print(f"[m2_seq] precomputed features for {len(fx)} samples "
          f"(unique_wt_seqs={n_unique_seq} feature_dim={next(iter(fx.values())).shape[0]} "
          f"local_dim={next(iter(fx.values())).shape[0] - gsf.GLOBAL_SEQ_DIM} "
          f"global_seq_dim={gsf.GLOBAL_SEQ_DIM})", flush=True)

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
    print(f"[m2_seq] total changers={n_changer} / {len(samples)} "
          f"({n_changer/len(samples):.3f}) null_masks={len(null_cache)}", flush=True)

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
    print(f"[m2_seq] changers with usable spectrum={len(spectra)}", flush=True)
    if not spectra:
        raise RuntimeError("no changers with usable spectra; STOP.")

    # ---- LOO over designs ----
    pred_path = out / "keyed_predictions_m2_seq.jsonl"
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
                for m in ("wmed_spectrum", MODEL_VARIANT):
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

        # residual-learning model + global seq features (seeds 0..4)
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
                             "seed": seed, "model_variant": MODEL_VARIANT,
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
        "schema": "reactflow_delta.response_spectrum.m2_seq.v1",
        "run_id": out.name, "dataset": "OpenKnot_M2", "source_url": m2d.M2_SOURCE_URL,
        "caller_version": CALLER_VERSION, "caller_mode": "PER_POS_ERROR",
        "caller_null": "theoretical Gaussian (iid N(0,1) over eligibility mask)",
        "experiment": "residual_mlp_globalseq_full_spectrum_response_m2",
        "scale": args.scale, "window": WINDOW, "epochs": args.epochs,
        "resid_pen": args.resid_pen, "bs": args.bs, "lr": args.lr, "hidden": args.hidden,
        "model_variants": ["wmed_spectrum", MODEL_VARIANT],
        "global_seq_dim": gsf.GLOBAL_SEQ_DIM,
        "exchangeable_unit": "puzzle_x_method_design",
        "n_usable_designs": len(resolved), "n_samples": len(samples),
        "n_changers": n_changer, "n_changers_with_spectrum": len(spectra),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {"cuda_available": True, "device_index": gpu_index, "device_name": gpu_name,
                "free_bytes": int(free_mem), "total_bytes": int(tot_mem)},
        "n_rows": n_rows_total, "keyed_predictions_sha256": dhash,
        "source_hashes": {
            "run_response_spectrum_m2_seq_v1.py": sha256_file(Path(__file__)),
            "global_seq_features_v1.py": sha256_file(Path(__file__).resolve().parent / "global_seq_features_v1.py"),
            "residual_spectrum_v2.py": sha256_file(Path(__file__).resolve().parent / "residual_spectrum_v2.py"),
            "m2_data_v1.py": sha256_file(Path(__file__).resolve().parent / "m2_data_v1.py"),
            "m2_caller_v1.py": sha256_file(Path(__file__).resolve().parent / "m2_caller_v1.py"),
            "response_spectrum_scinv_v1.py": sha256_file(Path(__file__).resolve().parent / "response_spectrum_scinv_v1.py"),
            "run_p2_v3.py": sha256_file(Path(__file__).resolve().parent / "run_p2_v3.py"),
        },
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "response_spectrum_m2_seq_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDONE -> {out}  n_rows={n_rows_total}  sha256={dhash}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
