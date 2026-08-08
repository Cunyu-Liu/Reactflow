#!/usr/bin/env python3
"""P2v5 — nested leave-one-publication-out CONDITIONAL MAGNITUDE regression gate.

Route B (authority epoch 17, endpoint_v5 conditional task).

For each outer fold (outer unit = publication):
  * hold out ONE publication;
  * fit the fold-local caller (caller_v3) on the REMAINING train publications and
    adjudicate binary changer labels C_i for ALL pool pairs;
  * TRUE CHANGERS = pairs with label C_i == "1";
  * for each true changer compute the profile-level magnitude target
        y_i = mean over ELIGIBLE positions of |mutant_react[i] - wt_react[i]|
    and weight w_i = number of ELIGIBLE positions used;
  * train regression models on TRAIN-publication changers (allowed inputs only:
    WT seq + exact single-nucleotide mutation + allowed WT reactivity + condition),
    predict magnitude on HELD-OUT-publication changers.

Metric (endpoint_v5):
  conditional WMAE skill = 1 - WMAE_model / WMAE_trivial, where trivial is the
  train-changer weighted-mean constant predictor (baseline, train-fold only).
  Plus paired publication-block bootstrap CI and publication-block permutation.

Every neural model MUST run on CUDA; if CUDA is unavailable the run STOPS.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch  # noqa: E402

# --- read-only reuse of run_p2_v3 data/feature helpers & caller_v3 ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_p2_v3 import (  # noqa: E402
    build_rep_groups, rep_groups_for_train, build_pair_features_aligned,
    build_feature, require_cuda,
)
from caller_v3 import CallerV3  # noqa: E402
from evaluate_v5 import (  # noqa: E402
    conditional_wmae_skill, paired_bootstrap_skill_ci,
    permutation_test_skill,
)

SEEDS = [0, 1, 2, 3, 4]
MODELS_DEFAULT = "trivial,linear,gbm,p2_mlp,deepsets"
EPOCHS = 30
BS = 128
LR = 1e-3


def _study_of(sa: str) -> str:
    return (sa or "").split("_")[0]


def pair_magnitude(pf):
    """(magnitude, weight) from PairFeatures over ELIGIBLE positions."""
    wt = pf.wt_reactivity
    mu = pf.mutant_reactivity
    mask = pf.eligibility_mask
    L = min(len(wt), len(mu), len(mask))
    vals = []
    for i in range(L):
        if not mask[i]:
            continue
        a, b = float(wt[i]), float(mu[i])
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        vals.append(abs(b - a))
    if not vals:
        return (None, 0)
    return (float(sum(vals)) / len(vals), len(vals))


# ---------------------------------------------------------------------------
# Regression models (PyTorch on CUDA)
# ---------------------------------------------------------------------------
class RegMLP(torch.nn.Module):
    def __init__(self, in_dim, seed):
        super().__init__()
        import torch.nn as nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DeepSetsReg(torch.nn.Module):
    def __init__(self, pos_dim, hidden, glob_dim, seed):
        super().__init__()
        import torch.nn as nn
        torch.manual_seed(seed)
        self.phi = nn.Sequential(nn.Linear(pos_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.rho = nn.Sequential(nn.Linear(hidden + glob_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, pos_set, glob):
        e = self.phi(pos_set).sum(dim=1)
        return self.rho(torch.cat([e, glob], dim=1)).squeeze(-1)


def train_reg(model, X, y, w, device, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    wt_ = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = Xt.shape[0]
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            pred = model(Xt[idx])
            loss = (wt_[idx] * (pred - yt[idx]).abs()).mean()  # weighted MAE
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def predict_reg(model, X, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        return model(Xt).cpu().numpy()


def deepsets_reg_split(Xflat, W, pos_dim):
    n = Xflat.shape[0]
    pos = Xflat[:, :W * pos_dim].reshape(n, W, pos_dim)
    glob = Xflat[:, W * pos_dim:]
    return pos, glob


def fit_predict(name, Xtr, ytr, wtr, Xte, seed, device, pos_dim):
    if name == "trivial":
        # train-changer weighted-mean constant predictor (the baseline)
        c = float(np.sum(np.asarray(wtr) * np.asarray(ytr)) / max(np.sum(wtr), 1e-12))
        return np.full(len(Xte), c, dtype=np.float32)
    if name == "linear":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=1.0, random_state=seed)
        m.fit(Xtr, np.asarray(ytr))
        return np.asarray(m.predict(Xte), dtype=np.float32)
    if name == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=100, max_depth=3,
                                          learning_rate=0.05, random_state=seed)
        m.fit(Xtr, np.asarray(ytr))
        return np.asarray(m.predict(Xte), dtype=np.float32)
    if name == "p2_mlp":
        model = RegMLP(Xtr.shape[1], seed)
        train_reg(model, Xtr, ytr, wtr, device, seed)
        return predict_reg(model, Xte, device).astype(np.float32)
    if name == "deepsets":
        W = 21
        pos_tr, gl_tr = deepsets_reg_split(Xtr, W, pos_dim)
        glob_dim = Xtr.shape[1] - W * pos_dim
        model = DeepSetsReg(pos_dim, 64, glob_dim, seed)
        train_reg_set(model, pos_tr, gl_tr, ytr, wtr, device, seed)
        pos_te, gl_te = deepsets_reg_split(Xte, W, pos_dim)
        return predict_reg_set(model, pos_te, gl_te, device).astype(np.float32)
    raise ValueError(name)


def train_reg_set(model, pos, glob, y, w, device, seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    wt_ = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = pos.shape[0]
    model.train()
    for ep in range(EPOCHS):
        perm = np.random.permutation(n)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            pb = torch.from_numpy(pos[idx]).to(device)
            gb = torch.from_numpy(glob[idx]).to(device)
            yb, wb = yt[idx], wt_[idx]
            pred = model(pb, gb)
            loss = (wb * (pred - yb).abs()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def predict_reg_set(model, pos, glob, device):
    model.eval()
    with torch.no_grad():
        pos_t = torch.from_numpy(np.asarray(pos, dtype=np.float32)).to(device)
        glob_t = torch.from_numpy(np.asarray(glob, dtype=np.float32)).to(device)
        return model(pos_t, glob_t).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split-yaml", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cuda-device", default="6")
    ap.add_argument("--models", default=MODELS_DEFAULT)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after CUDA_VISIBLE_DEVICES=" + args.cuda_device +
                           ". Contract: STOP, no silent CPU fallback.")
    device = require_cuda()
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[p2v5] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import yaml
    split = yaml.safe_load(Path(args.split_yaml).read_text(encoding="utf-8"))
    pub_map = split["publication_map"]
    study_roles = split["study_roles"]

    with open(args.cache, "rb") as fh:
        cache = __import__("pickle").load(fh)
    rec_index = cache["rec_index"]
    pairs = cache["pairs"]
    pool_studies = set(cache["pool"])

    test_studies = {s for s, r in study_roles.items() if r == "test"}
    pool_studies = pool_studies - test_studies

    pair_recs = {}
    missing = 0
    for p in pairs:
        if _study_of(p["source_accession"]) in test_studies:
            continue
        wt = rec_index.get((p["source_accession"], p["wt_profile_index"], p["asset_name"]))
        mu = rec_index.get((p["source_accession"], p["mutant_profile_index"], p["asset_name"]))
        if wt is None or mu is None:
            missing += 1
            continue
        pair_recs[p["source_accession"] + ":" + str(p["mutant_profile_index"])] = {
            "pair": p, "wt": wt, "mut": mu,
            "study": _study_of(p["source_accession"]),
            "pub": pub_map.get(_study_of(p["source_accession"]), "UNKNOWN"),
        }
    print(f"[p2v5] pool_studies={sorted(pool_studies)} n_pairs_usable={len(pair_recs)} missing={missing}", flush=True)

    all_rep_groups = build_rep_groups(rec_index, study_whitelist=pool_studies)
    pubs = sorted({v["pub"] for v in pair_recs.values()})
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])
    caller_seed = 20260807

    # fold-invariant features + pair magnitude inputs
    fx_full = {pid: build_feature(v["pair"], v["wt"], True, True, True)
               for pid, v in pair_recs.items()}
    pf_all = {pid: build_pair_features_aligned(v["pair"], v["wt"], v["mut"])
              for pid, v in pair_recs.items()}
    print(f"[p2v5] precomputed features for {len(pf_all)} pairs", flush=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    heldout = {m: {s: {"pub": [], "y": [], "w": [], "pred": []} for s in SEEDS}
               for m in models}

    fold_info = {}
    for fold, held_pub in enumerate(pubs):
        t0 = time.time()
        train_studies = set()
        for p_ in pubs:
            if p_ != held_pub:
                train_studies |= pub_study[p_]
        train_groups = rep_groups_for_train(all_rep_groups, train_studies)
        caller = CallerV3(seed=caller_seed).fit(train_groups, [], noise_replicate_groups=all_rep_groups)
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

        tr_ch = [pid for pid in train_pids if labels[pid] == "1" and mags[pid][1] > 0]
        he_ch = [pid for pid in held_pids if labels[pid] == "1" and mags[pid][1] > 0]

        fold_info[held_pub] = {"n_train_changers": len(tr_ch), "n_held_changers": len(he_ch)}
        if not tr_ch or not he_ch:
            print(f"[fold] held={held_pub} SKIP (train_ch={len(tr_ch)} held_ch={len(he_ch)})", flush=True)
            continue

        Xtr = np.stack([fx_full[pid] for pid in tr_ch])
        ytr = np.array([mags[pid][0] for pid in tr_ch], dtype=np.float32)
        wtr = np.array([mags[pid][1] for pid in tr_ch], dtype=np.float32)
        Xte = np.stack([fx_full[pid] for pid in he_ch])
        yte = np.array([mags[pid][0] for pid in he_ch], dtype=np.float32)
        wte = np.array([mags[pid][1] for pid in he_ch], dtype=np.float32)

        for m in models:
            for seed in SEEDS:
                pred = fit_predict(m, Xtr, ytr, wtr, Xte, seed, device, pos_dim=7)
                h = heldout[m][seed]
                h["pub"].extend([held_pub] * len(pred))
                h["y"].extend(yte.tolist())
                h["w"].extend(wte.tolist())
                h["pred"].extend(np.clip(pred, 0.0, None).tolist())
        fold_info[held_pub]["seconds"] = round(time.time() - t0, 1)
        print(f"[fold] held={held_pub} train_ch={len(tr_ch)} held_ch={len(he_ch)} "
              f"t={time.time()-t0:.1f}s", flush=True)

    # ---- evaluate conditional WMAE skill per model x seed ----
    table = {}
    for m in models:
        for seed in SEEDS:
            h = heldout[m][seed]
            # baseline = trivial model's prediction for the same held changers
            triv = heldout["trivial"][seed]
            skill_res = conditional_wmae_skill(h["pub"], h["y"], h["w"], h["pred"], triv["pred"])
            ci = paired_bootstrap_skill_ci(h["pub"], h["y"], h["w"], h["pred"], triv["pred"], seed=seed, n_boot=1000)
            perm = permutation_test_skill(h["pub"], h["y"], h["w"], h["pred"], triv["pred"], seed=seed, n_perm=1000)
            table[(m, seed)] = {
                "skill": skill_res.get("skill"),
                "wmae_model": skill_res.get("wmae_model"),
                "wmae_baseline": skill_res.get("wmae_baseline"),
                "n_changers": skill_res.get("n_changers"),
                "n_publications": skill_res.get("n_publications"),
                "ci": ci,
                "permutation": perm,
            }
    print("\n[p2v5] conditional WMAE skill by model x seed:", flush=True)
    for m in models:
        print(f"  {m}: {[table[(m,s)]['skill'] for s in SEEDS]}", flush=True)

    results = {
        "run_id": out.name,
        "endpoint": "endpoint_v5",
        "authority_epoch": 17,
        "metric": "conditional WMAE skill",
        "n_pool_studies": len(pool_studies),
        "n_pool_pairs": len(pair_recs),
        "n_distinct_publications": len(pubs),
        "publications": pubs,
        "publication_studies": {p: sorted(s) for p, s in pub_study.items()},
        "fold_info": fold_info,
        "models": models,
        "seeds": SEEDS,
        "table": {f"{m}:{s}": {
            "skill": table[(m, s)]["skill"],
            "wmae_model": table[(m, s)]["wmae_model"],
            "wmae_baseline": table[(m, s)]["wmae_baseline"],
            "ci_low": table[(m, s)]["ci"]["ci_low"],
            "ci_high": table[(m, s)]["ci"]["ci_high"],
            "permutation_p": table[(m, s)]["permutation"]["p_value"],
            "n_changers": table[(m, s)]["n_changers"],
            "n_publications": table[(m, s)]["n_publications"],
        } for m in models for s in SEEDS},
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    for m in models:
        for seed in SEEDS:
            h = heldout[m][seed]
            np.savez_compressed(out / f"heldout_{m}_seed{seed}.npz",
                                pub=np.array(h["pub"]), y=np.array(h["y"]),
                                w=np.array(h["w"]), pred=np.array(h["pred"]))

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proj = Path(__file__).resolve().parent.parent.parent
    source_hashes = {}
    for rel in ["configs/reactflow_delta/endpoint_v5.yaml",
                "configs/reactflow_delta/split_v2.yaml",
                "scripts/reactflow_delta/caller_v3.py",
                "scripts/reactflow_delta/evaluate_v2.py",
                "scripts/reactflow_delta/evaluate_v5.py",
                "scripts/reactflow_delta/run_p2_v3.py",
                "scripts/reactflow_delta/run_p2_v5.py"]:
        fp = proj / rel
        source_hashes[rel] = sha(fp) if fp.exists() else None

    manifest = {
        "schema": "reactflow_delta.p2v5_magnitude.v1",
        "run_id": out.name,
        "authority_epoch": 17,
        "endpoint": "endpoint_v5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {"cuda_available": bool(torch.cuda.is_available()),
                "device_index": gpu_index, "device_name": gpu_name,
                "free_bytes": int(free_mem), "total_bytes": int(tot_mem),
                "cuda_visible_devices": args.cuda_device},
        "source_hashes": source_hashes,
        "models": models,
        "seeds": SEEDS,
        "table": results["table"],
        "fold_info": fold_info,
    }
    (out / "P2v5_magnitude_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("\nDONE conditional magnitude LOOCV ->", out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
