#!/usr/bin/env python3
"""run_resid_deepsets_seq_v1 — DeepSets residual full-spectrum response experiment
with STRICT-legal global sequence features.

Builds on run_response_spectrum_v2 (residual learning around the per-position
median prior) but replaces the flat MLP with a POSITION-AWARE DeepSets whose global
branch is augmented by the global full-sequence representation (k-mer + ViennaRNA WT
folding) from global_seq_features_v1.  This supplies the full-sequence context the
stock local-only features lacked.

  pred[k] = prior[k] + delta[k]          (residual around per-position median prior)
  delta   = DeepSets( local_window_pos, [pair_tail | global_seq] )

Keyed schema is IDENTICAL to v1/v2 (schema reactflow_delta.response_spectrum.v3),
so compare_response_spectrum_v1.py can consume it with
  --model-variant wmae_resid_deepsets_seq
The new model variant is `wmae_resid_deepsets_seq` (seeds 0..4).  Same caller
(CallerV4 STRICT), fold protocol (LOO held pub), and CUDA-only rule (STOP if CUDA
unavailable, no CPU fallback).  Development-only; fail-closed.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines_v6 import (  # noqa: E402
    SEEDS, WINDOW, POS_DIM,
    load_cache, load_publication_map, build_pair_recs,
    build_rep_groups, rep_groups_for_train, build_pair_features_aligned_robust,
    build_feature, sha256_file,
)
from run_p2_v3 import edited_index  # noqa: E402
from caller_v4 import CallerV4, MODE_STRICT  # noqa: E402
import response_spectrum_scinv_v1 as rss  # noqa: E402
import global_seq_features_v1 as gsf  # noqa: E402
import resid_deepsets_seq_v1 as rds  # noqa: E402


MODEL_VARIANT = "wmae_resid_deepsets_seq"
HIDDEN = 64
EPOCHS = 30
BS = 128
LR = 1e-3
RESID_PEN = 1e-2


def _he_pids_rows(he_pids, held_pub, seed, m, pair_recs, Yte, Wte, pred):
    out = []
    for j, pid in enumerate(he_pids):
        out.append({
            "pair_id": pid, "task": "magnitude_spectrum", "fold_id": held_pub,
            "seed": seed, "model_variant": m,
            "model_id": "response_spectrum_v3", "publication_id": held_pub,
            "source_accession": pair_recs[pid]["pair"]["source_accession"],
            "split_role": "development", "endpoint_version": "endpoint_v6",
            "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
            "y": Yte[j].tolist(), "weight": Wte[j].tolist(),
            "raw_prediction": pred[j].tolist(),
            "transformed_prediction": pred[j].tolist(),
            "coverage_status": "CALLED",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cuda-device", default="6")
    ap.add_argument("--scale", choices=("mean_level", "mad"), default="mean_level")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--resid-pen", type=float, default=RESID_PEN)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Contract: STOP, no silent CPU fallback.")
    device = torch.device("cuda")
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[resid_deepsets_seq] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "fold_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache)
    pub_map = load_publication_map(args.registry)
    pair_recs, missing = build_pair_recs(cache, pub_map)
    print(f"[resid_deepsets_seq] n_pair_recs={len(pair_recs)} registry_missing={missing}",
          flush=True)
    resolved = sorted({v["pub"] for v in pair_recs.values() if (v["pub"] or "").startswith("pmid_")})

    all_rep_groups = build_rep_groups(cache["rec_index"])
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])

    # fold-invariant features
    fx_base = {pid: build_feature(v["pair"], v["wt"], True, True, True)
               for pid, v in pair_recs.items()}
    # global seq features: ViennaRNA folding is CPU-bound, so dedupe by WT sequence
    seq_cache: dict = {}
    fx_seq = {}
    for pid, v in pair_recs.items():
        seq = (v["wt"].get("canonical_sequence") or "").upper()
        if seq not in seq_cache:
            seq_cache[seq] = gsf.build_global_seq_feature(v["wt"], v["pair"])
        fx_seq[pid] = seq_cache[seq]
    pf_all = {pid: build_pair_features_aligned_robust(v["pair"], v["wt"], v["mut"])
              for pid, v in pair_recs.items()}
    print(f"[resid_deepsets_seq] precomputed features for {len(pair_recs)} pairs "
          f"(unique_wt_seqs={len(seq_cache)} base_dim={next(iter(fx_base.values())).shape[0]} "
          f"global_seq_dim={gsf.GLOBAL_SEQ_DIM})", flush=True)

    caller_seed = 20260809
    pred_path = out / "keyed_predictions_resid_deepsets_seq.jsonl"
    ckpt_path = out / "fold_progress.json"
    done_folds = set()
    if args.resume and ckpt_path.exists():
        done_folds = set(json.loads(ckpt_path.read_text(encoding="utf-8")).get("completed_folds", []))

    fp = pred_path.open("a", encoding="utf-8")
    n_rows_total = (sum(1 for _ in pred_path.open("r", encoding="utf-8"))
                    if pred_path.exists() else 0)
    global_log = {}
    t_start = time.time()

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
                rows.append({"pair_id": pid, "task": "magnitude_spectrum",
                             "fold_id": held_pub, "seed": 0, "model_variant": MODEL_VARIANT,
                             "model_id": "response_spectrum_v3", "publication_id": held_pub,
                             "source_accession": pair_recs[pid]["pair"]["source_accession"],
                             "split_role": "development", "endpoint_version": "endpoint_v6",
                             "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                             "y": None, "weight": 0.0, "raw_prediction": None,
                             "transformed_prediction": None, "coverage_status": "NO_CALL"})
            done_folds.add(held_pub)
            ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                                 encoding="utf-8")
            continue

        # ---- assemble per-position prior from TRAIN changers ----
        Ytr = np.stack([spectra[pid]["y"] for pid in train_pids]).astype(np.float32)
        Wtr = np.stack([spectra[pid]["w"] for pid in train_pids]).astype(np.float32)
        prior, prior_cnt = rds.per_position_prior(Ytr, Wtr)
        Yte = np.stack([spectra[pid]["y"] for pid in he_pids]).astype(np.float32)
        Wte = np.stack([spectra[pid]["w"] for pid in he_pids]).astype(np.float32)

        # ---- features: local window (pos) + [pair tail | global seq] (glob) ----
        Xtr_base = np.stack([fx_base[pid] for pid in train_pids]).astype(np.float32)
        Xte_base = np.stack([fx_base[pid] for pid in he_pids]).astype(np.float32)
        S_tr = np.stack([fx_seq[pid] for pid in train_pids]).astype(np.float32)
        S_te = np.stack([fx_seq[pid] for pid in he_pids]).astype(np.float32)

        pos_tr, glob_tr_base = rds.split_pos_glob(Xtr_base, WINDOW, POS_DIM)
        pos_te, glob_te_base = rds.split_pos_glob(Xte_base, WINDOW, POS_DIM)
        glob_tr = rds.concat_glob_seq(glob_tr_base, S_tr)
        glob_te = rds.concat_glob_seq(glob_te_base, S_te)
        glob_dim = glob_tr.shape[1]

        fold_log = {"held_pub": held_pub, "n_train_changers": len(train_pids),
                    "n_held_changers": len(he_pids),
                    "prior": prior.tolist(), "prior_count": prior_cnt.tolist(),
                    "glob_dim": glob_dim, "seeds": {}}
        for seed in SEEDS:
            model = rds.DeepSetsResidSpectrum(POS_DIM, args.hidden, glob_dim, WINDOW, seed=seed)
            model, tlog = rds.train_resid_sets(
                model, pos_tr, glob_tr, Ytr, Wtr, prior, device, seed=seed,
                epochs=args.epochs, bs=args.bs, lr=args.lr, resid_pen=args.resid_pen)
            pred = rds.predict_resid_sets(model, pos_te, glob_te, prior, device)
            fold_log["seeds"][str(seed)] = tlog
            rows.extend(_he_pids_rows(he_pids, held_pub, seed, MODEL_VARIANT,
                                      pair_recs, Yte, Wte, pred))

        (logdir / f"{held_pub}.json").write_text(json.dumps(fold_log, sort_keys=True),
                                                 encoding="utf-8")
        global_log[held_pub] = {
            "n_train_changers": len(train_pids), "n_held_changers": len(he_pids),
            "prior": prior.tolist(), "prior_count": prior_cnt.tolist(),
            "seeds": {s: fold_log["seeds"][s]["final"] for s in fold_log["seeds"]},
        }

        for r in rows:
            fp.write(json.dumps(r, sort_keys=True) + "\n")
        n_rows_total += len(rows)
        fp.flush()
        done_folds.add(held_pub)
        ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                             encoding="utf-8")
        print(f"[fold] held={held_pub} changers={len(he_pids)} t={time.time()-t0:.1f}s",
              flush=True)
    fp.close()

    (out / "residual_deepsets_seq_training_log.json").write_text(
        json.dumps(global_log, indent=2, sort_keys=True), encoding="utf-8")

    dhash = sha256_file(pred_path)
    manifest = {
        "schema": "reactflow_delta.response_spectrum.v3",
        "run_id": out.name, "authority_epoch": 20, "endpoint": "endpoint_v6",
        "caller_version": "caller_v4", "caller_mode_primary": MODE_STRICT,
        "experiment": "residual_deepsets_globalseq_full_spectrum_response",
        "scale": args.scale, "window": WINDOW, "epochs": args.epochs,
        "resid_pen": args.resid_pen, "bs": args.bs, "lr": args.lr, "hidden": args.hidden,
        "global_seq_dim": gsf.GLOBAL_SEQ_DIM,
        "model_variants": [MODEL_VARIANT],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {"cuda_available": True, "device_index": gpu_index, "device_name": gpu_name,
                "free_bytes": int(free_mem), "total_bytes": int(tot_mem)},
        "n_pair_recs": len(pair_recs), "n_resolved_publications": len(resolved),
        "n_rows": n_rows_total, "keyed_predictions_sha256": dhash,
        "source_hashes": {
            "run_resid_deepsets_seq_v1.py": sha256_file(Path(__file__)),
            "resid_deepsets_seq_v1.py": sha256_file(Path(__file__).resolve().parent / "resid_deepsets_seq_v1.py"),
            "global_seq_features_v1.py": sha256_file(Path(__file__).resolve().parent / "global_seq_features_v1.py"),
            "response_spectrum_scinv_v1.py": sha256_file(Path(__file__).resolve().parent / "response_spectrum_scinv_v1.py"),
            "caller_v4.py": sha256_file(Path(__file__).resolve().parent / "caller_v4.py"),
            "run_baselines_v6.py": sha256_file(Path(__file__).resolve().parent / "run_baselines_v6.py"),
        },
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "response_spectrum_v3_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDONE -> {out}  n_rows={n_rows_total}  sha256={dhash}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
