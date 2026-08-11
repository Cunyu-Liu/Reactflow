#!/usr/bin/env python3
"""run_baselines_v6 — Phase 2 (AUTHORIZE_PHASE2_LEARNABILITY) frozen baseline runner.

Implements Batch 2A on endpoint_v6 / CallerV4 (STRICT_INDUCTIVE_WT_ALLOWED) /
split_v3 development pool, with nested leave-one-resolved-publication-out CV.

Two tasks (judged independently, endpoint_v6 degenerate/independent policy):

  PRIMARY prospective-changer probability :
    unit = pair (called subset); label C_i = CallerV4(STRICT) binary changer label.
    score = P(C_i=1). metric = publication-macro AUPRC; paired = per-pub AP delta
    vs prevalence trivial. Baselines: prevalence(trivial), wlogit, gam, gbm,
    p2_mlp(generic), deepsets(control).

  SECONDARY oracle-conditioned conditional magnitude :
    unit = TRUE CHANGER pair (C_i=1); y = mean over ELIGIBLE positions of
    |mut-wt|; w = n_eligible. metric = conditional WMAE skill; paired = pub-block
    bootstrap CI. Baselines: wmedian(trivial), lad_lm, wgam, wmse_gbm, wmae_mlp,
    wmae_deepsets.

Input permission is fold-invariant and identical across tasks (endpoint_v6
information_permission): WT sequence + exact single-nt mutation + condition +
allowed WT reactivity anchor. The target eligibility mask NEVER enters model
input. Neural models MUST run on CUDA (STOP if unavailable, no CPU fallback).

Everything is written as KEYED predictions (prediction_v2-compatible subset):
rows carry biological keys (pair_id, publication_id, source_accession, fold_id,
seed, model_variant) + provenance hashes. No position-zip; raw/transformed
separate; non-call explicit coverage_status (never 0-filled).

Determinism / seeds: prevalence/wlogit/gam/gbm/lad_lm/wgam/wmse_gbm are
deterministic (seed=0, single row). Neural (p2_mlp/deepsets/wmae_mlp/
wmae_deepsets) run 5 documented seeds; seed is NOT biological N.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, pickle, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

import torch  # noqa: E402

# --- read-only reuse of development data/feature helpers + CallerV4 ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_p2_v3 import (  # noqa: E402
    build_rep_groups, rep_groups_for_train, build_feature, require_cuda,
)
from caller_v4 import CallerV4, MODE_STRICT  # noqa: E402
from caller_v2 import PairFeatures, compute_eligible_mask  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 30
BS = 128
LR = 1e-3
WINDOW = 21
POS_DIM = 7  # base(5)+react(1)+err(1) per window position

# neural model families (run 5 seeds)
NEURAL = {"p2_mlp", "deepsets", "wmae_mlp", "wmae_deepsets"}
# deterministic sklearn/constant families (single seed=0)
DET = {"prevalence", "wlogit", "gam", "gbm", "wmedian", "lad_lm", "wgam", "wmse_gbm"}


def _study_of(sa: str) -> str:
    return (sa or "").split("_")[0]


def build_pair_features_aligned_robust(pair, wt_rec, mut_rec):
    """PairFeatures aligned to the reactivity length, with zero-padded error arrays.

    The stock run_p2_v3.build_pair_features_aligned truncates to the minimum of
    the mask AND the reactivity/error arrays; ~40% of cache records have empty
    error arrays, which collapses the mask to length 0 and makes every such pair
    NO_CALL. Here the mask is aligned to the minimum non-empty reactivity length
    and missing error arrays are zero-padded. This is safe for CallerV4 STRICT:
    the per-position sigma is taken from the train-only replicate groups (with a
    train-global median fallback); the pair error array is only a last-resort
    sigma source when no train sigma exists (absent here), so padding to 0 is
    never used to inflate callability.
    """
    def get(r, key):
        rl = r.get("reactivity_layers", {})
        return list(rl.get("train_frozen", {}).get(key)
                    or rl.get("raw", {}).get(key) or [])
    wt_react = get(wt_rec, "reactivity")
    mut_react = get(mut_rec, "reactivity")
    wt_err = get(wt_rec, "error")
    mut_err = get(mut_rec, "error")
    mask = compute_eligible_mask(pair.get("eligibility_reason_codes") or [])
    L = min(len(mask), len(wt_react), len(mut_react))
    L = max(L, 0)

    def pad(a):
        a = list(a[:L])
        if len(a) < L:
            a = a + [0.0] * (L - len(a))
        return a

    grp = (_study_of(pair.get("source_accession") or ""),
           wt_rec.get("canonical_sequence") or "",
           tuple(wt_rec.get("probe") or []),
           tuple(wt_rec.get("temperature") or []))
    return PairFeatures(
        pair_id=f"{pair.get('source_accession')}:{pair.get('mutant_profile_index')}",
        wt_reactivity=wt_react[:L], mutant_reactivity=mut_react[:L],
        wt_error=pad(wt_err), mutant_error=pad(mut_err),
        eligibility_mask=mask[:L], group_key=grp, role="train")


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data assembly: cache pairs joined to publication via registry parent_id
# ---------------------------------------------------------------------------
def load_cache(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_publication_map(registry_tsv: str) -> dict:
    """(source_accession, parent_id) -> publication_id_normalized."""
    import csv
    out: dict = {}
    with open(registry_tsv, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[(r["source_accession"], r["parent_id"])] = r["publication_id_normalized"]
    return out


def build_pair_recs(cache: dict, pub_map: dict) -> dict:
    """pair_recs: key = f"{sa}:{mut_idx}"; value has pair/wt/mut/study/pub."""
    rec_index = cache["rec_index"]
    pairs = cache["pairs"]
    out: dict = {}
    missing = 0
    for p in pairs:
        sa = p["source_accession"]
        wt = rec_index.get((sa, p["wt_profile_index"], p["asset_name"]))
        mu = rec_index.get((sa, p["mutant_profile_index"], p["asset_name"]))
        if wt is None or mu is None:
            missing += 1
            continue
        parent = f"{sa}.rdat:{p['wt_profile_index']}"
        pub = pub_map.get((sa, parent))
        if pub is None:
            # fallback: single-publication study
            st = _study_of(sa)
            singles = {v for k, v in pub_map.items() if k[0] == sa}
            pub = next(iter(singles)) if len(singles) == 1 else None
        out[f"{sa}:{p['mutant_profile_index']}"] = {
            "pair": p, "wt": wt, "mut": mu,
            "study": _study_of(sa), "pub": pub,
        }
    return out, missing


# ---------------------------------------------------------------------------
# Pair-level magnitude target (endpoint_v6 secondary)
# ---------------------------------------------------------------------------
def pair_magnitude(pf):
    """(magnitude, weight) over ELIGIBLE positions."""
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
# Models
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


class DeepSets(torch.nn.Module):
    def __init__(self, pos_dim, hidden, glob_dim, seed, out_dim=1):
        super().__init__()
        import torch.nn as nn
        torch.manual_seed(seed)
        self.phi = nn.Sequential(nn.Linear(pos_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.rho = nn.Sequential(nn.Linear(hidden + glob_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, pos_set, glob):
        e = self.phi(pos_set).sum(dim=1)
        return self.rho(torch.cat([e, glob], dim=1)).squeeze(-1)


def _train_nn(model, X, y, w, device, seed, reg=False, sets=False, pos_dim=None, W=None, glob_dim=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    yt = np.asarray(y, dtype=np.float32)
    wt_ = np.asarray(w, dtype=np.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = X.shape[0]
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i + BS].numpy()
            if sets:
                Xb = Xt[idx]
                pos = Xb[:, :W * pos_dim].reshape(-1, W, pos_dim)
                gl = Xb[:, W * pos_dim:]
                pred = model(pos, gl)
            else:
                pred = model(Xt[idx])
            yb = torch.from_numpy(yt[idx]).to(device)
            wb = torch.from_numpy(wt_[idx]).to(device)
            if reg:
                loss = (wb * (pred - yb).abs()).mean()
            else:
                p = torch.clamp(pred, 1e-6, 1 - 1e-6)
                loss = -(wb * (yb * torch.log(p) + (1 - yb) * torch.log(1 - p))).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _predict_nn(model, X, device, sets=False, W=None, pos_dim=None):
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        if sets:
            pos = Xt[:, :W * pos_dim].reshape(-1, W, pos_dim)
            gl = Xt[:, W * pos_dim:]
            out = model(pos, gl)
        else:
            out = model(Xt)
        return out.cpu().numpy()


def _weighted_median(y, w):
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    order = np.argsort(y)
    y, w = y[order], w[order]
    cw = np.cumsum(w)
    total = cw[-1] if len(cw) else 0.0
    if total <= 0:
        return float(np.median(y)) if len(y) else 0.0
    idx = int(np.searchsorted(cw, 0.5 * total))
    idx = min(idx, len(y) - 1)
    return float(y[idx])


def _weighted_lad_linear_coef(X, y, w):
    """Weighted L1 (LAD) linear fit; returns full coefficient vector [intercept, ...]."""
    from scipy.optimize import minimize
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    def obj(b):
        return float(np.sum(w * np.abs(Xb @ b - y)))
    b0 = np.zeros(Xb.shape[1])
    b0[0] = _weighted_median(y, w)
    res = minimize(obj, b0, method="BFGS")
    return res.x


def _spline_linear(X, y, w, n_knots=4):
    """Weighted least squares on a spline basis (sklearn SplineTransformer)."""
    from sklearn.preprocessing import SplineTransformer
    from sklearn.linear_model import Ridge
    X = np.asarray(X, dtype=np.float64)
    st = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
    Xs = st.fit_transform(X)
    m = Ridge(alpha=1.0)
    m.fit(Xs, y, sample_weight=w)
    return lambda Xt: m.predict(st.transform(np.asarray(Xt, dtype=np.float64)))


def _binary_auprc(labels, scores):
    from sklearn.metrics import average_precision_score
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(set(labels.tolist())) <= 1:
        return None
    return float(average_precision_score(labels, scores))


def fit_predict_primary(model, Xtr, ytr, wtr, Xte, seed, device, pos_dim=POS_DIM, W=WINDOW):
    """Return (pred_proba, meta)."""
    if model == "prevalence":
        c = float(np.mean(np.asarray(ytr)))
        return np.full(len(Xte), c, dtype=np.float32), {}
    if model == "wlogit":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        m.fit(Xtr, np.asarray(ytr))
        return m.predict_proba(Xte)[:, 1].astype(np.float32), {}
    if model == "gam":
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import LogisticRegression
        st = SplineTransformer(n_knots=4, degree=3, include_bias=False)
        Xs = st.fit_transform(Xtr)
        m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        m.fit(Xs, np.asarray(ytr))
        return m.predict_proba(st.transform(Xte))[:, 1].astype(np.float32), {}
    if model == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=100, max_depth=3,
                                           learning_rate=0.05, random_state=seed)
        m.fit(Xtr, np.asarray(ytr))
        return m.predict_proba(Xte)[:, 1].astype(np.float32), {}
    if model == "p2_mlp":
        net = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        net = _train_nn(net, Xtr, ytr, wtr, device, seed, reg=False)
        raw = _predict_nn(net, Xte, device)
        return torch.sigmoid(torch.from_numpy(raw)).numpy().astype(np.float32).reshape(-1), {}
    if model == "deepsets":
        glob_dim = Xtr.shape[1] - W * pos_dim
        net = DeepSets(pos_dim, 64, glob_dim, seed, out_dim=1)
        net = _train_nn(net, Xtr, ytr, wtr, device, seed, reg=False, sets=True,
                        pos_dim=pos_dim, W=W, glob_dim=glob_dim)
        raw = _predict_nn(net, Xte, device, sets=True, pos_dim=pos_dim, W=W)
        return torch.sigmoid(torch.from_numpy(raw)).numpy().astype(np.float32).reshape(-1), {}
    raise ValueError(model)


def fit_predict_magnitude(model, Xtr, ytr, wtr, Xte, seed, device, pos_dim=POS_DIM, W=WINDOW):
    """Return (pred, meta)."""
    if model == "wmedian":
        c = _weighted_median(ytr, wtr)
        return np.full(len(Xte), c, dtype=np.float32), {}
    if model == "lad_lm":
        beta = _weighted_lad_linear_coef(Xtr, ytr, wtr)
        Xte_b = np.column_stack([np.ones(Xte.shape[0]),
                                 np.asarray(Xte, dtype=np.float64)])
        pred = Xte_b @ beta
        return np.asarray(np.clip(pred, 0.0, None), dtype=np.float32).reshape(-1), {}
    if model == "wgam":
        pred_fn = _spline_linear(Xtr, ytr, wtr)
        return np.clip(pred_fn(Xte), 0.0, None).astype(np.float32), {}
    if model == "wmse_gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=100, max_depth=3,
                                          learning_rate=0.05, random_state=seed)
        m.fit(Xtr, np.asarray(ytr), sample_weight=np.asarray(wtr))
        return np.clip(m.predict(Xte), 0.0, None).astype(np.float32), {}
    if model == "wmae_mlp":
        in_dim = Xtr.shape[1]
        net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        net = _train_nn(net, Xtr, ytr, wtr, device, seed, reg=True)
        return np.clip(_predict_nn(net, Xte, device), 0.0, None).astype(np.float32), {}
    if model == "wmae_deepsets":
        glob_dim = Xtr.shape[1] - W * pos_dim
        net = DeepSets(pos_dim, 64, glob_dim, seed, out_dim=1)
        net = _train_nn(net, Xtr, ytr, wtr, device, seed, reg=True, sets=True,
                        pos_dim=pos_dim, W=W, glob_dim=glob_dim)
        return np.clip(_predict_nn(net, Xte, device, sets=True, pos_dim=pos_dim, W=W),
                       0.0, None).astype(np.float32), {}
    raise ValueError(model)


def seeds_for(model: str) -> list:
    return list(SEEDS) if model in NEURAL else [0]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cuda-device", default="6")
    ap.add_argument("--resume", action="store_true",
                    help="Resume: skip folds already recorded in the progress checkpoint.")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Contract: STOP, no silent CPU fallback for neural models.")
    device = require_cuda()
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[baselines_v6] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cache = load_cache(args.cache)
    pub_map = load_publication_map(args.registry)
    pair_recs, missing = build_pair_recs(cache, pub_map)
    print(f"[baselines_v6] n_pair_recs={len(pair_recs)} registry_missing={missing}", flush=True)

    # resolved publications = exchangeable units for LOOCV inference
    resolved = sorted({v["pub"] for v in pair_recs.values() if (v["pub"] or "").startswith("pmid_")})
    print(f"[baselines_v6] resolved publications: {len(resolved)}", flush=True)
    for p in resolved:
        n = sum(1 for v in pair_recs.values() if v["pub"] == p)
        print(f"    {p}: {n} pairs", flush=True)

    all_rep_groups = build_rep_groups(cache["rec_index"])
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])

    # fold-invariant features (allowed inputs only)
    fx_full = {pid: build_feature(v["pair"], v["wt"], True, True, True)
               for pid, v in pair_recs.items()}
    pf_all = {pid: build_pair_features_aligned_robust(v["pair"], v["wt"], v["mut"])
              for pid, v in pair_recs.items()}
    print(f"[baselines_v6] precomputed features for {len(pf_all)} pairs", flush=True)

    caller_seed = 20260809
    primary_models = list(cfg["models"]["primary_prospective_changer"].keys())
    magn_models = list(cfg["models"]["secondary_conditional_magnitude"].keys())

    # keyed prediction rows: incremental append + resume checkpoint
    rows = []
    fold_info = {}
    pred_path = out / "keyed_predictions.jsonl"
    ckpt_path = out / "fold_progress.json"
    done_folds = set()
    if args.resume and ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        done_folds = set(ckpt.get("completed_folds", []))
        print(f"[baselines_v6] resume: {len(done_folds)} folds already completed, "
              f"will skip them", flush=True)
    fp = pred_path.open("a", encoding="utf-8")  # append across resume
    n_rows_total = (sum(1 for _ in pred_path.open("r", encoding="utf-8"))
                    if pred_path.exists() else 0)
    t_start = time.time()
    for fold, held_pub in enumerate(resolved):
        t0 = time.time()
        if held_pub in done_folds:
            print(f"[fold] held={held_pub} RESUME SKIP (already completed)", flush=True)
            continue
        rows.clear()
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

        # ---- PRIMARY task (called subset, NO_CALL excluded) ----
        tr_prim = [pid for pid in train_pids if labels[pid] != "NO_CALL"]
        he_prim = [pid for pid in held_pids if labels[pid] != "NO_CALL"]
        if not he_prim:
            print(f"[fold] held={held_pub} PRIMARY SKIP (n_held_called=0)", flush=True)
            fold_info[held_pub] = {"n_train_called": len(tr_prim), "n_held_called": 0,
                                   "n_train_changers": 0, "n_held_changers": 0}
            done_folds.add(held_pub)
            ckpt_path.write_text(json.dumps(
                {"completed_folds": sorted(done_folds), "last": held_pub},
                indent=2, sort_keys=True), encoding="utf-8")
            continue
        Xpr_tr = np.stack([fx_full[pid] for pid in tr_prim]) if tr_prim else np.zeros((0, fx_full[he_prim[0]].shape[0]), dtype=np.float32)
        ypr_tr = np.array([1.0 if labels[pid] == "1" else 0.0 for pid in tr_prim], dtype=np.float32)
        wpr_tr = np.ones(len(tr_prim), dtype=np.float32)
        Xpr_te = np.stack([fx_full[pid] for pid in he_prim])
        ypr_te = np.array([1.0 if labels[pid] == "1" else 0.0 for pid in he_prim], dtype=np.float32)

        # ---- SECONDARY task (TRUE CHANGERS only) ----
        tr_mag = [pid for pid in tr_prim if labels[pid] == "1" and mags[pid][1] > 0]
        he_mag = [pid for pid in he_prim if labels[pid] == "1" and mags[pid][1] > 0]
        Xmg_tr = np.stack([fx_full[pid] for pid in tr_mag]) if tr_mag else np.zeros((0, fx_full[tr_prim[0]].shape[0]), dtype=np.float32)
        ymg_tr = np.array([mags[pid][0] for pid in tr_mag], dtype=np.float32)
        wmg_tr = np.array([mags[pid][1] for pid in tr_mag], dtype=np.float32)
        Xmg_te = np.stack([fx_full[pid] for pid in he_mag]) if he_mag else np.zeros((0, fx_full[he_prim[0]].shape[0]), dtype=np.float32)
        ymg_te = np.array([mags[pid][0] for pid in he_mag], dtype=np.float32)
        wmg_te = np.array([mags[pid][1] for pid in he_mag], dtype=np.float32)

        fold_info[held_pub] = {
            "n_train_called": len(tr_prim), "n_held_called": len(he_prim),
            "n_train_changers": len(tr_mag), "n_held_changers": len(he_mag),
        }

        # ---- predict primary baselines ----
        for m in primary_models:
            for seed in seeds_for(m):
                pred, _ = fit_predict_primary(m, Xpr_tr, ypr_tr, wpr_tr, Xpr_te, seed, device)
                for j, pid in enumerate(he_prim):
                    rows.append({
                        "pair_id": pid, "task": "primary", "fold_id": held_pub,
                        "seed": seed, "model_variant": m, "model_id": "baseline_v6",
                        "publication_id": held_pub,
                        "source_accession": pair_recs[pid]["pair"]["source_accession"],
                        "split_role": "development", "endpoint_version": "endpoint_v6",
                        "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                        "y": float(ypr_te[j]), "weight": 1.0,
                        "raw_prediction": float(pred[j]),
                        "transformed_prediction": float(pred[j]),
                        "coverage_status": "CALLED",
                    })

        # ---- predict magnitude baselines ----
        for m in magn_models:
            for seed in seeds_for(m):
                if not he_mag:
                    for j, pid in enumerate(he_mag):
                        rows.append({
                            "pair_id": pid, "task": "magnitude", "fold_id": held_pub,
                            "seed": seed, "model_variant": m, "model_id": "baseline_v6",
                            "publication_id": held_pub,
                            "source_accession": pair_recs[pid]["pair"]["source_accession"],
                            "split_role": "development", "endpoint_version": "endpoint_v6",
                            "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                            "y": None, "weight": 0.0,
                            "raw_prediction": None, "transformed_prediction": None,
                            "coverage_status": "NO_CALL",
                        })
                    continue
                pred, _ = fit_predict_magnitude(m, Xmg_tr, ymg_tr, wmg_tr, Xmg_te, seed, device)
                for j, pid in enumerate(he_mag):
                    rows.append({
                        "pair_id": pid, "task": "magnitude", "fold_id": held_pub,
                        "seed": seed, "model_variant": m, "model_id": "baseline_v6",
                        "publication_id": held_pub,
                        "source_accession": pair_recs[pid]["pair"]["source_accession"],
                        "split_role": "development", "endpoint_version": "endpoint_v6",
                        "caller_version": "caller_v4", "caller_mode": MODE_STRICT,
                        "y": float(ymg_te[j]), "weight": float(wmg_te[j]),
                        "raw_prediction": float(pred[j]),
                        "transformed_prediction": float(pred[j]),
                        "coverage_status": "CALLED",
                    })

        fold_info[held_pub]["seconds"] = round(time.time() - t0, 1)
        for r in rows:
            fp.write(json.dumps(r, sort_keys=True) + "\n")
        fp.flush()
        # record completion so a crash can resume past this fold
        done_folds.add(held_pub)
        ckpt_path.write_text(json.dumps(
            {"completed_folds": sorted(done_folds), "last": held_pub},
            indent=2, sort_keys=True), encoding="utf-8")
        print(f"[fold] held={held_pub} called={len(he_prim)} changers={len(he_mag)} "
              f"t={time.time()-t0:.1f}s", flush=True)
    fp.close()

    # ---- finalize manifest (predictions already streamed to keyed_predictions.jsonl) ----
    pred_path = out / "keyed_predictions.jsonl"

    dhash = sha256_file(pred_path)
    manifest = {
        "schema": "reactflow_delta.baselines_v6.keyed_predictions.v1",
        "run_id": out.name,
        "authority_epoch": 20,
        "endpoint": "endpoint_v6",
        "caller_version": "caller_v4",
        "caller_mode_primary": MODE_STRICT,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {"cuda_available": True, "device_index": gpu_index,
                "device_name": gpu_name, "free_bytes": int(free_mem),
                "total_bytes": int(tot_mem), "cuda_visible_devices": args.cuda_device},
        "n_pair_recs": len(pair_recs),
        "n_resolved_publications": len(resolved),
        "publications": resolved,
        "n_rows": n_rows_total,
        "keyed_predictions_sha256": dhash,
        "fold_info": fold_info,
        "primary_models": primary_models,
        "magnitude_models": magn_models,
        "source_hashes": {
            "baselines_v6.yaml": sha256_file(Path(args.config)),
            "run_baselines_v6.py": sha256_file(Path(__file__)),
            "caller_v4.py": sha256_file(Path(__file__).resolve().parent / "caller_v4.py"),
            "run_p2_v3.py": sha256_file(Path(__file__).resolve().parent / "run_p2_v3.py"),
        },
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "baselines_v6_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nDONE -> {out}  n_rows={n_rows_total}  sha256={dhash}", flush=True)


if __name__ == "__main__":
    sys.exit(main())