#!/usr/bin/env python3
"""run_response_spectrum_v1 — scale-invariant FULL-SPECTRUM response experiment.

Counters the p=1.0 root cause from a different angle than scinv: instead of
collapsing each true-changer pair to a single scalar magnitude (which discards the
signed/spatial structure of the mutation response), it predicts the FULL per-position
response spectrum over the model's local window around the edit site.

Target (see response_spectrum_scinv_v1):
    for each window position k -> y[k] = (mut[idx] - wt[idx]) / scale   (SIGNED)
    weight w[k] = 1.0 if idx eligible AND finite WT+mut, else 0.0
    scale = mean WT reactivity over eligible positions (STRICT-legal WT anchor).

Model: a position-aware MLP that maps the SAME fold-invariant features as Phase 2
(build_feature local window + exact-alt + condition) to a WINDOW-dim output vector
(aligned to window positions).  Loss = masked weighted MAE over the spectrum.

Baselines:
  * wmed_spectrum  : per-window-position train-changer weighted median (WINDOW-dim
                     constant vector, seed 0).  Strongest sequence-free trivial.
  * wmae_mlp_spectrum : position-aware MLP regressing the full spectrum (seeds 0..4).

Same caller (CallerV4 STRICT), fold protocol (LOO held pub), and keyed schema as
run_baselines_v6 / run_magnitude_scinv_v1.  Neural models MUST run on CUDA (STOP if
unavailable, no CPU fallback).  Development-only; fail-closed.
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
    build_feature, _weighted_median, sha256_file,
)
from run_p2_v3 import edited_index  # noqa: E402
from caller_v4 import CallerV4, MODE_STRICT  # noqa: E402
import response_spectrum_scinv_v1 as rss  # noqa: E402


class SpectrumMLP(torch.nn.Module):
    """Maps flattened local-window+global features to a WINDOW-dim response vector.

    Output index k corresponds to window position k (edit-HALF+k), so the model is
    position-aware: each output coordinate is aligned to a specific sequence position.
    """
    def __init__(self, in_dim, out_dim, hidden=128, seed=0):
        super().__init__()
        import torch.nn as nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _train_spectrum(model, X, Y, W, device, seed):
    """Masked weighted-MAE training over a (B, WINDOW) response target.

    Y, W are (n, WINDOW).  Positions with W==0 contribute nothing to the loss.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    Yt = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(device)
    Wt = torch.from_numpy(np.asarray(W, dtype=np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = Xt.shape[0]
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i + BS].numpy()
            pred = model(Xt[idx])
            w = Wt[idx]
            wh = w.mean()
            if wh <= 0.0:
                continue
            loss = (w * (pred - Yt[idx]).abs()).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _predict_spectrum(model, X, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        return model(Xt).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cuda-device", default="6")
    ap.add_argument("--scale", choices=("mean_level", "mad"), default="mean_level")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Contract: STOP, no silent CPU fallback.")
    device = torch.device("cuda")
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[spectrum] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache)
    pub_map = load_publication_map(args.registry)
    pair_recs, missing = build_pair_recs(cache, pub_map)
    print(f"[spectrum] n_pair_recs={len(pair_recs)} registry_missing={missing}", flush=True)
    resolved = sorted({v["pub"] for v in pair_recs.values() if (v["pub"] or "").startswith("pmid_")})

    all_rep_groups = build_rep_groups(cache["rec_index"])
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])

    fx_base = {pid: build_feature(v["pair"], v["wt"], True, True, True)
               for pid, v in pair_recs.items()}
    pf_all = {pid: build_pair_features_aligned_robust(v["pair"], v["wt"], v["mut"])
              for pid, v in pair_recs.items()}
    print(f"[spectrum] precomputed base features for {len(pair_recs)} pairs", flush=True)

    caller_seed = 20260809
    pred_path = out / "keyed_predictions_spectrum.jsonl"
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

        # per true-changer pair: (y_vec, w_vec, scale, edit_index)
        spectra = {}  # pid -> dict
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
                for m in ("wmed_spectrum", "wmae_mlp_spectrum"):
                    for seed in ([0] if m == "wmed_spectrum" else SEEDS):
                        rows.append({"pair_id": pid, "task": "magnitude_spectrum",
                                     "fold_id": held_pub, "seed": seed, "model_variant": m,
                                     "model_id": "response_spectrum_v1", "publication_id": held_pub,
                                     "source_accession": pair_recs[pid]["pair"]["source_accession"],
                                     "split_role": "development", "endpoint_version": "endpoint_v6",
                                     "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                                     "y": None, "weight": 0.0, "raw_prediction": None,
                                     "transformed_prediction": None, "coverage_status": "NO_CALL"})
            done_folds.add(held_pub)
            ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                                 encoding="utf-8")
            continue

        # stack train/held spectra
        Xtr = np.stack([fx_base[pid] for pid in train_pids]).astype(np.float32)
        Xte = np.stack([fx_base[pid] for pid in he_pids]).astype(np.float32)
        Ytr = np.stack([spectra[pid]["y"] for pid in train_pids]).astype(np.float32)
        Wtr = np.stack([spectra[pid]["w"] for pid in train_pids]).astype(np.float32)
        Yte = np.stack([spectra[pid]["y"] for pid in he_pids]).astype(np.float32)
        Wte = np.stack([spectra[pid]["w"] for pid in he_pids]).astype(np.float32)

        # --- trivial baseline: per-window-position train-weighted median ---
        med_vec = np.array([_weighted_median(Ytr[:, k], Wtr[:, k]) for k in range(WINDOW)],
                           dtype=np.float32)
        for j, pid in enumerate(he_pids):
            rows.append({"pair_id": pid, "task": "magnitude_spectrum", "fold_id": held_pub,
                         "seed": 0, "model_variant": "wmed_spectrum",
                         "model_id": "response_spectrum_v1", "publication_id": held_pub,
                         "source_accession": pair_recs[pid]["pair"]["source_accession"],
                         "split_role": "development", "endpoint_version": "endpoint_v6",
                         "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                         "y": Yte[j].tolist(), "weight": Wte[j].tolist(),
                         "raw_prediction": med_vec.tolist(),
                         "transformed_prediction": med_vec.tolist(),
                         "coverage_status": "CALLED"})

        # --- position-aware MLP ---
        in_dim = Xtr.shape[1]
        for seed in SEEDS:
            model = SpectrumMLP(in_dim, WINDOW, hidden=128, seed=seed)
            model = _train_spectrum(model, Xtr, Ytr, Wtr, device, seed)
            pred = _predict_spectrum(model, Xte, device)
            for j, pid in enumerate(he_pids):
                rows.append({"pair_id": pid, "task": "magnitude_spectrum", "fold_id": held_pub,
                             "seed": seed, "model_variant": "wmae_mlp_spectrum",
                             "model_id": "response_spectrum_v1", "publication_id": held_pub,
                             "source_accession": pair_recs[pid]["pair"]["source_accession"],
                             "split_role": "development", "endpoint_version": "endpoint_v6",
                             "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                             "y": Yte[j].tolist(), "weight": Wte[j].tolist(),
                             "raw_prediction": pred[j].tolist(),
                             "transformed_prediction": pred[j].tolist(),
                             "coverage_status": "CALLED"})

        for r in rows:
            fp.write(json.dumps(r, sort_keys=True) + "\n")
        n_rows_total += len(rows)
        fp.flush()
        done_folds.add(held_pub)
        ckpt_path.write_text(json.dumps({"completed_folds": sorted(done_folds)}, sort_keys=True),
                             encoding="utf-8")
        print(f"[fold] held={held_pub} changers={len(he_pids)} t={time.time()-t0:.1f}s", flush=True)
    fp.close()

    dhash = sha256_file(pred_path)
    manifest = {
        "schema": "reactflow_delta.response_spectrum.v1",
        "run_id": out.name, "authority_epoch": 20, "endpoint": "endpoint_v6",
        "caller_version": "caller_v4", "caller_mode_primary": MODE_STRICT,
        "experiment": "scale_invariant_full_spectrum_response",
        "scale": args.scale, "window": WINDOW,
        "model_variants": ["wmed_spectrum", "wmae_mlp_spectrum"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {"cuda_available": True, "device_index": gpu_index, "device_name": gpu_name,
                "free_bytes": int(free_mem), "total_bytes": int(tot_mem)},
        "n_pair_recs": len(pair_recs), "n_resolved_publications": len(resolved),
        "n_rows": n_rows_total, "keyed_predictions_sha256": dhash,
        "source_hashes": {
            "run_response_spectrum_v1.py": sha256_file(Path(__file__)),
            "response_spectrum_scinv_v1.py": sha256_file(Path(__file__).resolve().parent / "response_spectrum_scinv_v1.py"),
            "caller_v4.py": sha256_file(Path(__file__).resolve().parent / "caller_v4.py"),
            "run_baselines_v6.py": sha256_file(Path(__file__).resolve().parent / "run_baselines_v6.py"),
        },
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "response_spectrum_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDONE -> {out}  n_rows={n_rows_total}  sha256={dhash}", flush=True)


if __name__ == "__main__":
    sys.exit(main())