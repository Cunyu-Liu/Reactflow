#!/usr/bin/env python3
"""run_response_spectrum_v2 — residual-learning FULL-SPECTRUM response experiment.

Builds directly on the v1 spectrum experiment (scale-invariant full per-position
response target) which broke the p=1.0 degeneracy (permutation p = 0.005) but whose
position-aware MLP came out significantly WORSE than the sequence-free per-position
median baseline (skill ~ -0.18).

This runner replaces the direct MLP with RESIDUAL LEARNING around the median prior:
    pred[k] = prior[k] + delta[k]
where prior[k] = train-changer weighted median at window position k (the sequence-free
baseline), and delta[k] is learned from sequence/condition features.  The model is
initialized to delta = 0 (== baseline) and a residual-regularization term shrinks
learned residuals toward 0, so the model can only beat the baseline if the features
carry transferable signal.

Keyed schema is IDENTICAL to v1 so the comparison evaluator can consume both.  The
new model variant is `wmae_resid_spectrum` (seeds 0..4).  Diagnostic logs per fold
(written to <out>/fold_logs/<held_pub>.json) capture:
  * prior distribution per window position (value, support count, train |Y-prior|)
  * the full per-epoch residual learning curve (loss, |delta| mean/p90/max)
  * final residual stats (|delta|, frac moved off prior, train model-vs-prior MAE)

Same caller (CallerV4 STRICT), fold protocol (LOO held pub), and CUDA-only rule as
run_response_spectrum_v1.  Development-only; fail-closed.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines_v6 import (  # noqa: E402
    SEEDS, WINDOW, load_cache, load_publication_map, build_pair_recs,
    build_rep_groups, rep_groups_for_train, build_pair_features_aligned_robust,
    build_feature, sha256_file,
)
from run_p2_v3 import edited_index  # noqa: E402
from caller_v4 import CallerV4, MODE_STRICT  # noqa: E402
import response_spectrum_scinv_v1 as rss  # noqa: E402
import residual_spectrum_v2 as rsm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--registry", required=True)
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
    print(f"[resid] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "fold_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache)
    pub_map = load_publication_map(args.registry)
    pair_recs, missing = build_pair_recs(cache, pub_map)
    print(f"[resid] n_pair_recs={len(pair_recs)} registry_missing={missing}", flush=True)
    resolved = sorted({v["pub"] for v in pair_recs.values() if (v["pub"] or "").startswith("pmid_")})

    all_rep_groups = build_rep_groups(cache["rec_index"])
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])

    fx_base = {pid: build_feature(v["pair"], v["wt"], True, True, True)
               for pid, v in pair_recs.items()}
    pf_all = {pid: build_pair_features_aligned_robust(v["pair"], v["wt"], v["mut"])
              for pid, v in pair_recs.items()}
    print(f"[resid] precomputed base features for {len(pair_recs)} pairs", flush=True)

    caller_seed = 20260809
    pred_path = out / "keyed_predictions_resid_spectrum.jsonl"
    ckpt_path = out / "fold_progress.json"
    done_folds = set()
    if args.resume and ckpt_path.exists():
        done_folds = set(json.loads(ckpt_path.read_text(encoding="utf-8")).get("completed_folds", []))

    fp = pred_path.open("a", encoding="utf-8")
    n_rows_total = (sum(1 for _ in pred_path.open("r", encoding="utf-8"))
                    if pred_path.exists() else 0)
    t_start = time.time()
    global_log = {}

    for fold, held_pub in enumerate(resolved):
        t0 = time.time()
        if held_pub in done_folds:
            print(f"[fold] held={held_pub} RESUME SKIP", flush=True)
            continue
        rows = []
        train_studies = set()
        for p_ in resolved:
            if p_ != held_pub:
                train_studies |= pub_study[p_]
        train_groups = rep_groups_for_train(all_rep_groups, train_studies)
        caller = CallerV4(mode=MODE_STRICT, seed=caller_seed).fit(train_groups, [])

        spectra = {}
        for pid, v in pair_recs.items():
            lab = caller.call(pf_all[pid]).label
            if lab != "1":
                continue
            pf = pf_all[pid]
            ei = edited_index(v["pair"])
            y_vec, w_vec, scale = rss.pair_response_spectrum(
                pf.wt_reactivity, pf.mutant_reactivity, pf.eligibility_mask,
                ei, window=WINDOW, scale_mode=args.scale)
            if scale is None or sum(w_vec) <= 0:
                continue
            spectra[pid] = {"y": y_vec, "w": w_vec, "scale": scale, "edit_index": ei}

        train_pids = [pid for pid in spectra if pair_recs[pid]["pub"] != held_pub]
        he_pids = [pid for pid in spectra if pair_recs[pid]["pub"] == held_pub]

        if not he_pids:
            for pid in he_pids:
                for m in ("wmed_spectrum", "wmae_resid_spectrum"):
                    for seed in ([0] if m == "wmed_spectrum" else SEEDS):
                        rows.append({"pair_id": pid, "task": "magnitude_spectrum",
                                     "fold_id": held_pub, "seed": seed, "model_variant": m,
                                     "model_id": "response_spectrum_v2", "publication_id": held_pub,
                                     "source_accession": pair_recs[pid]["pair"]["source_accession"],
                                     "split_role": "development", "endpoint_version": "endpoint_v6",
                                     "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                                     "y": None, "weight": 0.0, "raw_prediction": None,
                                     "transformed_prediction": None, "coverage_status": "NO_CALL"})
            done_folds.add(held_pub)
            ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                                 encoding="utf-8")
            continue

        Xtr = np.stack([fx_base[pid] for pid in train_pids]).astype(np.float32)
        Xte = np.stack([fx_base[pid] for pid in he_pids]).astype(np.float32)
        Ytr = np.stack([spectra[pid]["y"] for pid in train_pids]).astype(np.float32)
        Wtr = np.stack([spectra[pid]["w"] for pid in train_pids]).astype(np.float32)
        Yte = np.stack([spectra[pid]["y"] for pid in he_pids]).astype(np.float32)
        Wte = np.stack([spectra[pid]["w"] for pid in he_pids]).astype(np.float32)

        # sequence-free prior = per-position train-weighted median
        prior, prior_cnt = rsm.per_position_prior(Ytr, Wtr)
        prior_dist = rsm.prior_distribution(Ytr, Wtr, prior)

        # baseline rows (seed 0): per-position prior (sequence-free)
        for j, pid in enumerate(he_pids):
            rows.append({"pair_id": pid, "task": "magnitude_spectrum", "fold_id": held_pub,
                         "seed": 0, "model_variant": "wmed_spectrum",
                         "model_id": "response_spectrum_v2", "publication_id": held_pub,
                         "source_accession": pair_recs[pid]["pair"]["source_accession"],
                         "split_role": "development", "endpoint_version": "endpoint_v6",
                         "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                         "y": Yte[j].tolist(), "weight": Wte[j].tolist(),
                         "raw_prediction": prior.tolist(),
                         "transformed_prediction": prior.tolist(),
                         "coverage_status": "CALLED"})

        # residual-learning model (seeds 0..4)
        in_dim = Xtr.shape[1]
        fold_log = {"held_pub": held_pub, "n_train_changers": len(train_pids),
                    "n_held_changers": len(he_pids),
                    "prior": prior.tolist(), "prior_count": prior_cnt.tolist(),
                    "prior_distribution": prior_dist, "seeds": {}}
        for seed in SEEDS:
            model, tlog = rsm.train_residual(
                Xtr, Ytr, Wtr, prior,
                epochs=args.epochs, bs=args.bs, lr=args.lr,
                resid_pen=args.resid_pen, hidden=args.hidden, seed=seed, device=device)
            delta = rsm.predict_delta(model, Xte, device)
            pred = (np.asarray(prior) + delta).astype(np.float32)
            fold_log["seeds"][str(seed)] = tlog
            for j, pid in enumerate(he_pids):
                rows.append({"pair_id": pid, "task": "magnitude_spectrum", "fold_id": held_pub,
                             "seed": seed, "model_variant": "wmae_resid_spectrum",
                             "model_id": "response_spectrum_v2", "publication_id": held_pub,
                             "source_accession": pair_recs[pid]["pair"]["source_accession"],
                             "split_role": "development", "endpoint_version": "endpoint_v6",
                             "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                             "y": Yte[j].tolist(), "weight": Wte[j].tolist(),
                             "raw_prediction": pred[j].tolist(),
                             "transformed_prediction": pred[j].tolist(),
                             "coverage_status": "CALLED"})

        # drop bulky learning curve from the fold summary to keep JSON small, but keep
        # a pointer to the full per-seed log file
        (logdir / f"{held_pub}.json").write_text(json.dumps(fold_log, sort_keys=True),
                                                 encoding="utf-8")
        global_log[held_pub] = {
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
        done_folds.add(held_pub)
        ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                             encoding="utf-8")
        print(f"[fold] held={held_pub} changers={len(he_pids)} t={time.time()-t0:.1f}s", flush=True)
    fp.close()

    (out / "residual_training_log.json").write_text(
        json.dumps(global_log, indent=2, sort_keys=True), encoding="utf-8")

    dhash = sha256_file(pred_path)
    manifest = {
        "schema": "reactflow_delta.response_spectrum.v2",
        "run_id": out.name, "authority_epoch": 20, "endpoint": "endpoint_v6",
        "caller_version": "caller_v4", "caller_mode_primary": MODE_STRICT,
        "experiment": "residual_learning_full_spectrum_response",
        "scale": args.scale, "window": WINDOW, "epochs": args.epochs,
        "resid_pen": args.resid_pen, "bs": args.bs, "lr": args.lr, "hidden": args.hidden,
        "model_variants": ["wmed_spectrum", "wmae_resid_spectrum"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {"cuda_available": True, "device_index": gpu_index, "device_name": gpu_name,
                "free_bytes": int(free_mem), "total_bytes": int(tot_mem)},
        "n_pair_recs": len(pair_recs), "n_resolved_publications": len(resolved),
        "n_rows": n_rows_total, "keyed_predictions_sha256": dhash,
        "source_hashes": {
            "run_response_spectrum_v2.py": sha256_file(Path(__file__)),
            "residual_spectrum_v2.py": sha256_file(Path(__file__).resolve().parent / "residual_spectrum_v2.py"),
            "response_spectrum_scinv_v1.py": sha256_file(Path(__file__).resolve().parent / "response_spectrum_scinv_v1.py"),
            "caller_v4.py": sha256_file(Path(__file__).resolve().parent / "caller_v4.py"),
            "run_baselines_v6.py": sha256_file(Path(__file__).resolve().parent / "run_baselines_v6.py"),
        },
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "response_spectrum_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDONE -> {out}  n_rows={n_rows_total}  sha256={dhash}", flush=True)


if __name__ == "__main__":
    sys.exit(main())