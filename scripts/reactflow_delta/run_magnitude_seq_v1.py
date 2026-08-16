#!/usr/bin/env python3
"""run_magnitude_seq_v1 — magnitude-task experiment for the RNA-representation upgrade.

Adds a STRICT-legal global sequence representation to the DeepSets global branch
and compares the resulting `wmae_deepsets_seq` against the Phase 2 `wmae_deepsets`
baseline on the SECONDARY conditional-magnitude task only (the task with a real
22-25% signal).  Same architecture, params, seeds, caller, and fold protocol as
run_baselines_v6; the ONLY difference is that the feature vector is augmented by
appending the global seq features (k-mer + ViennaRNA WT folding) to the global
branch (the tail of X), which DeepSets already routes to `glob`.

Input permission is identical to Phase 2 (endpoint_v6 STRICT_INDUCTIVE_WT_ALLOWED):
WT sequence + exact mutation + condition + allowed WT reactivity anchor.  The
global seq features are derived ONLY from the WT sequence (allowed) + edit
position, so they are STRICT-legal and fold-invariant.

Output: keyed_predictions_seq.jsonl (schema-subset of prediction_v2) with rows
for `wmae_deepsets_seq` on the magnitude task for all 13 LOOCV folds / 5 seeds.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pickle, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines_v6 import (  # noqa: E402
    SEEDS, EPOCHS, BS, LR, WINDOW, POS_DIM,
    load_cache, load_publication_map, build_pair_recs,
    build_rep_groups, rep_groups_for_train, build_pair_features_aligned_robust,
    build_feature, pair_magnitude, DeepSets, _train_nn, _predict_nn,
    _weighted_median, sha256_file,
)
from caller_v4 import CallerV4, MODE_STRICT  # noqa: E402
import global_seq_features_v1 as gsf  # noqa: E402


def sha256_file_impl(path: Path) -> str:
    return sha256_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cuda-device", default="6")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Contract: STOP, no silent CPU fallback.")
    device = torch.device("cuda")
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[magnitude_seq] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache)
    pub_map = load_publication_map(args.registry)
    pair_recs, missing = build_pair_recs(cache, pub_map)
    print(f"[magnitude_seq] n_pair_recs={len(pair_recs)} registry_missing={missing}", flush=True)
    resolved = sorted({v["pub"] for v in pair_recs.values() if (v["pub"] or "").startswith("pmid_")})

    all_rep_groups = build_rep_groups(cache["rec_index"])
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])

    # fold-invariant features: base (local window) + global seq augmentation
    fx_base = {pid: build_feature(v["pair"], v["wt"], True, True, True)
               for pid, v in pair_recs.items()}
    fx_seq = {pid: gsf.build_global_seq_feature(v["wt"], v["pair"])
              for pid, v in pair_recs.items()}
    pf_all = {pid: build_pair_features_aligned_robust(v["pair"], v["wt"], v["mut"])
              for pid, v in pair_recs.items()}
    print(f"[magnitude_seq] precomputed base+seq features for {len(pair_recs)} pairs "
          f"(global_seq_dim={gsf.GLOBAL_SEQ_DIM})", flush=True)

    caller_seed = 20260809
    pred_path = out / "keyed_predictions_seq.jsonl"
    ckpt_path = out / "fold_progress.json"
    done_folds = set()
    if args.resume and ckpt_path.exists():
        done_folds = set(json.loads(ckpt_path.read_text(encoding="utf-8")).get("completed_folds", []))

    fp = pred_path.open("a", encoding="utf-8")
    n_rows_total = (sum(1 for _ in pred_path.open("r", encoding="utf-8"))
                    if pred_path.exists() else 0)
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

        labels = {}
        mags = {}
        for pid, v in pair_recs.items():
            lab = caller.call(pf_all[pid]).label
            labels[pid] = lab
            if lab == "1":
                mval, wval = pair_magnitude(pf_all[pid])
                mags[pid] = (mval, wval) if mval is not None else (None, 0)

        train_pids = [pid for pid in pair_recs if pair_recs[pid]["pub"] != held_pub]
        held_pids = [pid for pid in pair_recs if pair_recs[pid]["pub"] == held_pub]

        # magnitude task: TRUE CHANGERS only
        tr_mag = [pid for pid in train_pids if labels[pid] == "1" and mags[pid][1] > 0]
        he_mag = [pid for pid in held_pids if labels[pid] == "1" and mags[pid][1] > 0]

        if not he_mag:
            for pid in he_mag:
                rows.append({"pair_id": pid, "task": "magnitude", "fold_id": held_pub,
                             "seed": 0, "model_variant": "wmae_deepsets_seq",
                             "model_id": "magnitude_seq_v1", "publication_id": held_pub,
                             "source_accession": pair_recs[pid]["pair"]["source_accession"],
                             "split_role": "development", "endpoint_version": "endpoint_v6",
                             "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                             "y": None, "weight": 0.0, "raw_prediction": None,
                             "transformed_prediction": None, "coverage_status": "NO_CALL"})
            done_folds.add(held_pub)
            ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                                 encoding="utf-8")
            continue

        # augmented features: base || global_seq
        Xmg_tr = np.stack([np.concatenate([fx_base[pid], fx_seq[pid]]) for pid in tr_mag]).astype(np.float32)
        Xmg_te = np.stack([np.concatenate([fx_base[pid], fx_seq[pid]]) for pid in he_mag]).astype(np.float32)
        ymg_tr = np.array([mags[pid][0] for pid in tr_mag], dtype=np.float32)
        wmg_tr = np.array([mags[pid][1] for pid in tr_mag], dtype=np.float32)
        ymg_te = np.array([mags[pid][0] for pid in he_mag], dtype=np.float32)
        wmg_te = np.array([mags[pid][1] for pid in he_mag], dtype=np.float32)

        for seed in SEEDS:
            glob_dim = Xmg_tr.shape[1] - WINDOW * POS_DIM
            net = DeepSets(POS_DIM, 64, glob_dim, seed, out_dim=1)
            net = _train_nn(net, Xmg_tr, ymg_tr, wmg_tr, device, seed, reg=True,
                            sets=True, pos_dim=POS_DIM, W=WINDOW, glob_dim=glob_dim)
            pred = np.clip(_predict_nn(net, Xmg_te, device, sets=True,
                                       pos_dim=POS_DIM, W=WINDOW), 0.0, None).astype(np.float32)
            for j, pid in enumerate(he_mag):
                rows.append({"pair_id": pid, "task": "magnitude", "fold_id": held_pub,
                             "seed": seed, "model_variant": "wmae_deepsets_seq",
                             "model_id": "magnitude_seq_v1", "publication_id": held_pub,
                             "source_accession": pair_recs[pid]["pair"]["source_accession"],
                             "split_role": "development", "endpoint_version": "endpoint_v6",
                             "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                             "y": float(ymg_te[j]), "weight": float(wmg_te[j]),
                             "raw_prediction": float(pred[j]),
                             "transformed_prediction": float(pred[j]),
                             "coverage_status": "CALLED"})
        for r in rows:
            fp.write(json.dumps(r, sort_keys=True) + "\n")
        n_rows_total += len(rows)
        fp.flush()
        done_folds.add(held_pub)
        ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                             encoding="utf-8")
        print(f"[fold] held={held_pub} changers={len(he_mag)} t={time.time()-t0:.1f}s", flush=True)
    fp.close()

    dhash = sha256_file_impl(pred_path)
    manifest = {
        "schema": "reactflow_delta.magnitude_seq.v1",
        "run_id": out.name, "authority_epoch": 20, "endpoint": "endpoint_v6",
        "caller_version": "caller_v4", "caller_mode_primary": MODE_STRICT,
        "experiment": "rna_representation_upgrade",
        "model_variant": "wmae_deepsets_seq",
        "global_seq_dim": gsf.GLOBAL_SEQ_DIM,
        "global_seq_module_hash": sha256_file_impl(Path(__file__).resolve().parent / "global_seq_features_v1.py"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {"cuda_available": True, "device_index": gpu_index, "device_name": gpu_name,
                "free_bytes": int(free_mem), "total_bytes": int(tot_mem)},
        "n_pair_recs": len(pair_recs), "n_resolved_publications": len(resolved),
        "n_rows": n_rows_total, "keyed_predictions_sha256": dhash,
        "source_hashes": {
            "run_magnitude_seq_v1.py": sha256_file_impl(Path(__file__)),
            "global_seq_features_v1.py": sha256_file_impl(Path(__file__).resolve().parent / "global_seq_features_v1.py"),
            "caller_v4.py": sha256_file_impl(Path(__file__).resolve().parent / "caller_v4.py"),
            "run_baselines_v6.py": sha256_file_impl(Path(__file__).resolve().parent / "run_baselines_v6.py"),
        },
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "magnitude_seq_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDONE -> {out}  n_rows={n_rows_total}  sha256={dhash}", flush=True)


if __name__ == "__main__":
    sys.exit(main())