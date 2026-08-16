#!/usr/bin/env python3
"""P2 — nested leave-one-publication-out learnability gate (development diagnostic).

Rebuilds the direct baseline / P2 gate for the frozen endpoint_v2 primary task:
under ALLOWED inputs only (WT sequence + exact single-nucleotide mutation +
allowed WT reactivity state + condition, WITHOUT the mutant profile), is the
mutation-induced reactivity response cross-publication learnable, and which
simple/generic models give incremental skill over trivial baselines.

Design (per audit contract §13.2 R5 / Phase 2):
  * Outer unit = publication (same PMID = one publication).
  * For each outer fold: hold out ONE publication, fit the fold-local caller
    (caller_v2) on the REMAINING train publications, generate binary changer
    labels C_i for ALL pool pairs with that caller, train models on the train
    publications, predict scores on the held-out publication.
  * Score = pair-level P(C_i=1) (direct output); metric = publication-macro AUPRC
    (evaluate_v2.publication_macro_auprc / evaluate_primary).
  * NO_CALL units are excluded from the metric (never zero-filled).
  * Reuses caller_v2.py and evaluate_v2.py read-only (imported, not modified).

Models (Phase 2): trivial (constant), logistic (sklearn), GBM tree (sklearn),
P2 pair-level generic MLP (PyTorch, CUDA), DeepSets generic (PyTorch, CUDA).
Every neural model MUST run on CUDA; if CUDA is unavailable the run STOPS.

Ablations: sequence-only vs WT-anchor (and exact-alt / condition variants).
Learning curve: retrain from scratch at train-publication fractions.
Negative control: publication-level label permutation p = (b+1)/(B+1).
Single-study dominance: leave-one-study-out sensitivity.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, pickle, random, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import yaml

import torch  # noqa: E402  (needed for DeepSetsGeneric class definition / CUDA guard)

# --- read-only reuse of frozen caller & evaluator ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from caller_v2 import (
    CallerV2, ReplicateGroup, PairFeatures, compute_eligible_mask,
    build_pair_features as _caller_build_pair_features,
)
from evaluate_v2 import evaluate_primary, is_unidentifiable, publication_macro_auprc

# ---------------------------------------------------------------------------
# Frozen / documented constants
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]            # documented deterministic seeds
TRAIN_FRACTIONS = [0.25, 0.5, 1.0]
WINDOW = 21                        # local window around the edited site (positions)
HALF = WINDOW // 2
BASE_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
LEARNING_CURVE_MODELS = ["logistic", "p2_mlp"]
LEARNING_CURVE_SEEDS = [0, 1, 2, 3, 4]


def _study_of(sa: str) -> str:
    return (sa or "").split("_")[0]


def load_cache(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def sanitize_records(rec_index: dict) -> dict:
    """Truncate train_frozen reactivity/error arrays to a common per-record
    length so caller_v2 (which assumes aligned arrays) does not IndexError on
    records whose error array is shorter than its reactivity array."""
    n_mismatch = 0
    for r in rec_index.values():
        tf = r.get("reactivity_layers", {}).get("train_frozen")
        if not isinstance(tf, dict):
            continue
        react = tf.get("reactivity")
        err = tf.get("error")
        if isinstance(react, list) and isinstance(err, list) and len(react) != len(err):
            L = min(len(react), len(err))
            tf["reactivity"] = react[:L]
            tf["error"] = err[:L]
            n_mismatch += 1
    if n_mismatch:
        print(f"[p2] sanitized {n_mismatch} records with reactivity/error length mismatch", flush=True)
    return rec_index


# ---------------------------------------------------------------------------
# Replicate groups from cached WT records (whole pool, then filtered per fold)
# ---------------------------------------------------------------------------
def build_rep_groups(rec_index: dict, study_whitelist=None) -> list:
    wt_by_key: dict = {}
    for key, r in rec_index.items():
        if not r.get("is_wt"):
            continue
        st = _study_of(r.get("source_accession") or "")
        if study_whitelist is not None and st not in study_whitelist:
            continue
        gk = (st, r.get("canonical_sequence"), tuple(r.get("probe") or []),
              tuple(r.get("temperature") or []))
        wt_by_key.setdefault(gk, []).append(r)
    groups = []
    for (st, seq, probe, temp), recs in wt_by_key.items():
        if len(recs) < 2:
            continue
        rl0 = recs[0].get("reactivity_layers", {})
        mask = compute_eligible_mask(rl0.get("eligibility_reason_codes") or [])
        profs = []
        errs = []
        for r in recs:
            tf = r.get("reactivity_layers", {}).get("train_frozen", {}) or {}
            react = list(tf.get("reactivity") or [])
            err = list(tf.get("error") or [])
            L = min(len(react), len(err))   # sanitize length mismatch (robustness)
            profs.append(react[:L])
            errs.append(err[:L])
        groups.append(ReplicateGroup(
            group_key=(st, seq, tuple(probe), tuple(temp)),
            wt_profiles=profs, wt_errors=errs,
            eligibility_mask=mask, study=st))
    return groups


def rep_groups_for_train(all_groups, train_studies) -> list:
    return [g for g in all_groups if g.study in train_studies]


def build_pair_features_aligned(pair, wt_rec, mut_rec) -> PairFeatures:
    """Like caller_v2.build_pair_features but truncates all aligned arrays to a
    common length so caller_v2.per_position_z never IndexErrors when a pair's
    eligibility mask is longer than a record's reactivity/error arrays
    (robustness handling; keeps the same replicate-group identity)."""
    def get(r, key):
        rl = r.get("reactivity_layers", {})
        return list(rl.get("train_frozen", {}).get(key)
                    or rl.get("raw", {}).get(key) or [])
    wt_react = get(wt_rec, "reactivity"); wt_err = get(wt_rec, "error")
    mut_react = get(mut_rec, "reactivity"); mut_err = get(mut_rec, "error")
    mask = compute_eligible_mask(pair.get("eligibility_reason_codes") or [])
    L = min(len(mask), len(wt_react), len(wt_err), len(mut_react), len(mut_err))
    grp = (_study_of(pair.get("source_accession") or ""),
           wt_rec.get("canonical_sequence") or "",
           tuple(wt_rec.get("probe") or []),
           tuple(wt_rec.get("temperature") or []))
    return PairFeatures(
        pair_id=f"{pair.get('source_accession')}:{pair.get('mutant_profile_index')}",
        wt_reactivity=wt_react[:L], mutant_reactivity=mut_react[:L],
        wt_error=wt_err[:L], mutant_error=mut_err[:L],
        eligibility_mask=mask[:L], group_key=grp, role="train")


# ---------------------------------------------------------------------------
# Feature construction (ALLOWED inputs only)
# ---------------------------------------------------------------------------
PROBES = ["1M7", "DMS", "2A3", "SHAPE", "NMIA", "NOMe", "CMC", "R1J", "LCK", "RSQ", "SHP", "NMD"]
MODIFIERS = ["1M7", "DMS", "2A3", "NMIA", "NOMe", "CMC", "R1J", "LCK", "RSQ", "SHP", "NMD", "Lys", "CMCT", "glyoxal"]
EXPTYPES = ["MutateAndMap", "MapAll", "SingleHit", "MutateMap"]
BASE_OTHER = 4  # 'other' base slot


def _oh(val, index_of, size):
    """One-hot with an explicit 'other' slot at the end (deterministic size)."""
    v = np.zeros(size + 1, dtype=np.float32)
    i = index_of.get(str(val))
    if i is None:
        v[size] = 1.0
    else:
        v[i] = 1.0
    return v


def _base_oh(base):
    v = np.zeros(5, dtype=np.float32)
    i = BASE_MAP.get(base)
    if i is None:
        v[BASE_OTHER] = 1.0
    else:
        v[i] = 1.0
    return v


def _norm_react(v):
    return float(np.clip(v, 0.0, 3.0) / 3.0)


def _norm_err(v):
    return float(np.clip(v, 0.0, 1.0))


def edited_index(pair) -> int:
    codes = pair.get("eligibility_reason_codes") or []
    for i, c in enumerate(codes):
        if c == "EDITED_SITE":
            return i
    coord = pair.get("coordinate") or {}
    off = coord.get("offset")
    if isinstance(off, int):
        return off
    return 0


def build_feature(pair, wt_rec, use_wt_anchor=True, use_exact_alt=True,
                  use_condition=True) -> np.ndarray:
    seq = wt_rec.get("canonical_sequence") or ""
    rl = wt_rec.get("reactivity_layers", {})
    tf = rl.get("train_frozen", {}) or rl.get("raw", {})
    react = tf.get("reactivity") or []
    err = tf.get("error") or []
    react = np.nan_to_num(np.asarray(react, dtype=np.float32), nan=0.0)
    err = np.nan_to_num(np.asarray(err, dtype=np.float32), nan=0.0)
    n = len(seq)
    ei = edited_index(pair)
    parts = []
    for k in range(WINDOW):
        idx = ei - HALF + k
        if 0 <= idx < n:
            base = _base_oh(seq[idx])
        else:
            base = np.zeros(5, dtype=np.float32)
        if use_wt_anchor:
            r = _norm_react(react[idx]) if 0 <= idx < len(react) else 0.0
            e = _norm_err(err[idx]) if 0 <= idx < len(err) else 0.0
            base = np.concatenate([base, [r, e]])
        parts.append(base)
    feats = np.concatenate(parts)
    if use_exact_alt:
        feats = np.concatenate([feats, _base_oh(pair.get("ref_allele")),
                                _base_oh(pair.get("alt_allele"))])
    feats = np.concatenate([feats, [float(ei) / max(n, 1)]])
    if use_condition:
        cond = pair.get("condition") or {}
        probe = wt_rec.get("probe") or []
        feats = np.concatenate([feats, _oh(probe[0] if probe else "", _oh_index(PROBES), len(PROBES))])
        mod = cond.get("modifier") or []
        feats = np.concatenate([feats, _oh(mod[0] if mod else "", _oh_index(MODIFIERS), len(MODIFIERS))])
        et = cond.get("experimentType") or []
        feats = np.concatenate([feats, _oh(et[0] if et else "", _oh_index(EXPTYPES), len(EXPTYPES))])
        temps = [t for t in (cond.get("temperature") or []) if str(t).replace(".", "").replace("C", "").isdigit()]
        tval = float(str(temps[0]).replace("C", "")) if temps else 37.0
        feats = np.concatenate([feats, [tval / 100.0]])
    return feats.astype(np.float32)


def _oh_index(lst):
    return {str(v): i for i, v in enumerate(lst)}


# ---------------------------------------------------------------------------
# CUDA guard (contract: STOP, never silent CPU fallback for neural models)
# ---------------------------------------------------------------------------
def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is NOT available. Contract requires GPU for all neural P2 "
            "training; refusing to silently train on CPU. STOP.")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Neural models (PyTorch)
# ---------------------------------------------------------------------------
def _make_mlp(in_dim, hidden, out, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, out),
    )


class DeepSetsGeneric(torch.nn.Module):
    def __init__(self, pos_dim, hidden, glob_dim, seed):
        super().__init__()
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        self.phi = nn.Sequential(nn.Linear(pos_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.rho = nn.Sequential(nn.Linear(hidden + glob_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, pos_set, glob):
        e = self.phi(pos_set)              # (B, W, hidden)
        pooled = e.sum(dim=1)              # (B, hidden)
        x = torch.cat([pooled, glob], dim=1)
        return self.rho(x).squeeze(-1)


def train_torch(model, X, y, device, epochs=30, bs=256, lr=1e-3, seed=0):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = Xt.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = lossf(logits, yb)
            loss.backward()
            opt.step()
    return model


def predict_torch(model, X, device):
    import torch
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        return torch.sigmoid(model(Xt)).cpu().numpy().squeeze(-1)


def deepsets_from_flat(Xflat, W, pos_dim, glob_dim, seed):
    # split flat feature (W*pos_dim + glob_dim) into pos set + globals
    import torch
    n = Xflat.shape[0]
    pos = Xflat[:, :W * pos_dim].reshape(n, W, pos_dim)
    glob = Xflat[:, W * pos_dim:]
    return pos, glob


# ---------------------------------------------------------------------------
# Models dispatch
# ---------------------------------------------------------------------------
class ModelRunner:
    def __init__(self, device, use_wt_anchor=True):
        self.device = device
        self.use_wt_anchor = use_wt_anchor
        self.pos_dim = 7 if use_wt_anchor else 5
        self.glob_dim = None  # set after first feature

    def fit_predict(self, name, Xtr, ytr, Xte, seed):
        if name == "trivial":
            p = float(np.mean(ytr))
            return np.full(len(Xte), p, dtype=np.float32)
        if name == "logistic":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=2000, C=0.5, solver="liblinear", random_state=seed)
            m.fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1].astype(np.float32)
        if name == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=100, max_depth=3,
                                               learning_rate=0.05, early_stopping=True,
                                               validation_fraction=0.1, n_iter_no_change=20,
                                               random_state=seed)
            m.fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1].astype(np.float32)
        if name == "p2_mlp":
            import torch
            in_dim = Xtr.shape[1]
            model = _make_mlp(in_dim, 128, 1, seed)
            train_torch(model, Xtr, ytr, self.device, seed=seed)
            return predict_torch(model, Xte, self.device).astype(np.float32)
        if name == "deepsets":
            W, pos_dim = WINDOW, self.pos_dim
            glob_dim = Xtr.shape[1] - W * pos_dim
            model = DeepSetsGeneric(pos_dim, 64, glob_dim, seed)
            pos_tr, gl_tr = deepsets_from_flat(Xtr, W, pos_dim, glob_dim, seed)
            train_deepsets(model, pos_tr, gl_tr, ytr, self.device, seed=seed)
            pos_te, gl_te = deepsets_from_flat(Xte, W, pos_dim, glob_dim, seed)
            return predict_deepsets(model, pos_te, gl_te, self.device).astype(np.float32)
        raise ValueError(name)


def train_deepsets(model, pos, glob, y, device, epochs=30, bs=256, lr=1e-3, seed=0):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = model.to(device)
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = pos.shape[0]
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pb = torch.from_numpy(pos[idx]).to(device)
            gb = torch.from_numpy(glob[idx]).to(device)
            yb = yt[idx]
            opt.zero_grad()
            loss = lossf(model(pb, gb).squeeze(-1), yb)
            loss.backward()
            opt.step()
    return model


def predict_deepsets(model, pos, glob, device):
    import torch
    model.eval()
    with torch.no_grad():
        pos_t = torch.from_numpy(np.asarray(pos, dtype=np.float32)).to(device)
        glob_t = torch.from_numpy(np.asarray(glob, dtype=np.float32)).to(device)
        return torch.sigmoid(model(pos_t, glob_t)).cpu().numpy().squeeze(-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_split(split_path):
    split = yaml.safe_load(Path(split_path).read_text(encoding="utf-8"))
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="path to cache_p2.pkl")
    ap.add_argument("--split-yaml", required=True)
    ap.add_argument("--out-dir", required=True, help="results/p2_v1/<run_id>")
    ap.add_argument("--cuda-device", default="0", help="CUDA_VISIBLE_DEVICES index")
    ap.add_argument("--explore-only", action="store_true")
    ap.add_argument("--models", default="trivial,logistic,gbm,p2_mlp,deepsets")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after CUDA_VISIBLE_DEVICES=" + args.cuda_device +
                           ". Contract: STOP, no silent CPU fallback. Cannot run P2 neural models.")
    device = require_cuda()
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[p2] GPU OK: cuda_visible={args.cuda_device} current_idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    split = load_split(args.split_yaml)
    pub_map = split["publication_map"]          # study -> publication
    study_roles = split["study_roles"]          # study -> role

    cache = load_cache(args.cache)
    rec_index = cache["rec_index"]
    sanitize_records(rec_index)
    pairs = cache["pairs"]
    pool_studies = set(cache["pool"])

    # Exclude the sealed SL5 test family from the P2 pool entirely.
    test_studies = {s for s, r in study_roles.items() if r == "test"}
    pool_studies = pool_studies - test_studies

    # pair metadata: pair_id -> (pair, wt_rec, mut_rec, PairFeatures)
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
    print(f"[p2] pool_studies={sorted(pool_studies)} n_pairs_usable={len(pair_recs)} missing={missing}", flush=True)

    # all replicate groups (pool studies) for the caller
    all_rep_groups = build_rep_groups(rec_index, study_whitelist=pool_studies)

    # distinct publications in pool
    pubs = sorted({v["pub"] for v in pair_recs.values()})
    print(f"[p2] distinct publications in pool: {len(pubs)} -> {pubs}", flush=True)
    caller_seed = 20260807
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])
    print(f"[p2] publication->studies: { {p: sorted(s) for p, s in pub_study.items()} }", flush=True)

    # ---- explore mode ----
    if args.explore_only:
        t0 = time.time()
        caller = CallerV2(seed=caller_seed).fit(all_rep_groups, [])
        print(f"[explore] caller fit time={time.time()-t0:.1f}s null_median={caller.null_median}", flush=True)
        results = []
        for pid, v in pair_recs.items():
            pf = build_pair_features_aligned(v["pair"], v["wt"], v["mut"])
            results.append((v["pub"], v["study"], caller.call(pf)))
        lab_by_pub = defaultdict(Counter)
        for pub, study, r in results:
            lab_by_pub[pub].update([r.label])
        print("[explore] label distribution by publication:")
        for pub in pubs:
            c = lab_by_pub.get(pub, Counter())
            print(f"   {pub}: {dict(c)}", flush=True)
        n_ch = sum(1 for _, _, r in results if r.label == "1")
        n_nc = sum(1 for _, _, r in results if r.label == "NO_CALL")
        n0 = sum(1 for _, _, r in results if r.label == "0")
        print(f"[explore] total labels change={n_ch} non={n0} no_call={n_nc}", flush=True)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Precompute features once per ablation setting (allowed-input invariant).
    def featset(use_wt_anchor, use_exact_alt, use_condition):
        fx = {}
        for pid, v in pair_recs.items():
            fx[pid] = build_feature(v["pair"], v["wt"], use_wt_anchor, use_exact_alt, use_condition)
        return fx

    # ---- main LOOCV ----
    # For each outer fold, fit caller on train pubs, label all pairs, train
    # models on train pubs, predict held-out pub.  Record held-out predictions.
    heldout = {m: {s: {"pub": [], "label": [], "score": [], "study": []} for s in SEEDS}
               for m in models}

    caller_seed = 20260807
    fold_timing = {}
    fold_labels = {}
    fx_full = featset(True, True, True)   # WT-anchor features (fold-invariant)
    for fold, held_pub in enumerate(pubs):
        t0 = time.time()
        train_studies = set()
        for p_ in pubs:
            if p_ != held_pub:
                train_studies |= pub_study[p_]
        train_groups = rep_groups_for_train(all_rep_groups, train_studies)

        caller = CallerV2(seed=caller_seed).fit(train_groups, [])
        # label all pairs in pool with this fold-local caller
        labels = {}
        for pid, v in pair_recs.items():
            pf = build_pair_features_aligned(v["pair"], v["wt"], v["mut"])
            labels[pid] = caller.call(pf)
        fold_labels[held_pub] = {pid: labels[pid].label for pid in pair_recs}
        # training split (pairs in train publications, not NO_CALL)
        train_pids = [pid for pid, v in pair_recs.items() if v["pub"] != held_pub]
        held_pids = [pid for pid, v in pair_recs.items() if v["pub"] == held_pub]

        # feature set (WT-anchor default for main)
        Xtr = np.stack([fx_full[pid] for pid in train_pids])
        ytr = np.array([1.0 if labels[pid].label == "1" else 0.0 for pid in train_pids])
        tr_ok = np.array([labels[pid].label != "NO_CALL" for pid in train_pids])
        Xtr_ok = Xtr[tr_ok]; ytr_ok = ytr[tr_ok]

        Xte = np.stack([fx_full[pid] for pid in held_pids])
        te_ok = np.array([labels[pid].label != "NO_CALL" for pid in held_pids])
        Xte_ok = Xte[te_ok]
        te_labels = np.array([1.0 if labels[pid].label == "1" else 0.0 for pid in held_pids])[te_ok]

        if int(np.sum(te_ok)) == 0 or int(np.sum(tr_ok)) == 0:
            fold_timing[held_pub] = {"train_studies": len(train_studies),
                                     "n_train_pairs": int(np.sum(tr_ok)),
                                     "n_held_pairs": int(np.sum(te_ok)),
                                     "skipped": True,
                                     "seconds": round(time.time() - t0, 1)}
            print(f"[fold] held={held_pub} SKIPPED (empty eligible: train_ok={int(np.sum(tr_ok))} held_ok={int(np.sum(te_ok))})", flush=True)
            continue

        runner = ModelRunner(device, use_wt_anchor=True)
        for m in models:
            for seed in SEEDS:
                scores = runner.fit_predict(m, Xtr_ok, ytr_ok, Xte_ok, seed)
                h = heldout[m][seed]
                h["pub"].extend([held_pub] * len(scores))
                h["label"].extend(te_labels.tolist())
                h["score"].extend(scores.tolist())
                h["study"].extend([v["study"] for pid, v in pair_recs.items()
                                   if v["pub"] == held_pub and labels[pid].label != "NO_CALL"])

        fold_timing[held_pub] = {"train_studies": len(train_studies),
                                 "n_train_pairs": int(np.sum(tr_ok)),
                                 "n_held_pairs": int(np.sum(te_ok)),
                                 "seconds": round(time.time() - t0, 1)}
        print(f"[fold] held={held_pub} train_studies={len(train_studies)} "
              f"train_ok={int(np.sum(tr_ok))} held_ok={int(np.sum(te_ok))} "
              f"t={time.time()-t0:.1f}s", flush=True)

    # ---- evaluate each model x seed (pooled held-out) ----
    table = {}
    per_pub_aps = {}
    for m in models:
        for seed in SEEDS:
            h = heldout[m][seed]
            # drop NO_CALL (already excluded in te_ok), but keep for counting
            pubs_arr = h["pub"]; labs = h["label"]; scos = h["score"]
            res = evaluate_primary(pubs_arr, labs, scos, seed=seed, n_perm=1000, n_boot=1000)
            table[(m, seed)] = res
    print("\n[p2] publication-macro AUPRC by model x seed:", flush=True)
    for m in models:
        vals = [table[(m, s)]["metric"] for s in SEEDS]
        print(f"  {m}: {vals}", flush=True)

    # per-publication AP for reporting / degeneracy inspection (seed 0)
    ap_report = {}
    for m in models:
        h = heldout[m][SEEDS[0]]
        groups = defaultdict(list)
        for p, l, s in zip(h["pub"], h["label"], h["score"]):
            groups[p].append((int(bool(l)), float(s)))
        ap_report[m] = {}
        for p in sorted(groups, key=str):
            gl = [t for t, _ in groups[p]]; gs = [s for _, s in groups[p]]
            if len(set(gl)) <= 1:
                ap_report[m][p] = "DEGENERATE"
            else:
                ap_report[m][p] = publication_macro_auprc([p] * len(gl), gl, gs)
    print("\n[p2] per-publication AP (seed 0):", flush=True)
    for m in models:
        print(f"  {m}: {ap_report[m]}", flush=True)

    # ---- write results ----
    results = {
        "run_id": out.name,
        "n_pool_studies": len(pool_studies),
        "n_pool_pairs": len(pair_recs),
        "n_distinct_publications": len(pubs),
        "publications": pubs,
        "publication_studies": {p: sorted(s) for p, s in pub_study.items()},
        "fold_timing": fold_timing,
        "models": models,
        "seeds": SEEDS,
        "table": {f"{m}:{s}": {
            "metric": table[(m, s)]["metric"],
            "ci": table[(m, s)]["ci"],
            "permutation_p": table[(m, s)]["permutation"]["p_value"],
        } for m in models for s in SEEDS},
        "per_publication_ap_seed0": ap_report,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    # save held-out predictions (parquet-like via npz for each model/seed)
    for m in models:
        for seed in SEEDS:
            h = heldout[m][seed]
            np.savez_compressed(
                out / f"heldout_{m}_seed{seed}.npz",
                pub=np.array(h["pub"]), label=np.array(h["label"]),
                score=np.array(h["score"]), study=np.array(h["study"]))

    # ---- fold-local labels (reused by ablations / learning curve) ----
    (out / "fold_labels.json").write_text(
        json.dumps(fold_labels, sort_keys=True), encoding="utf-8")

    # ---- source hashes + manifest ----
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proj = Path(__file__).resolve().parent.parent.parent
    source_hashes = {}
    for rel in ["configs/reactflow_delta/endpoint_v2.yaml",
                "configs/reactflow_delta/split_v2.yaml",
                "scripts/reactflow_delta/caller_v2.py",
                "scripts/reactflow_delta/evaluate_v2.py",
                "scripts/reactflow_delta/run_p2_v1.py"]:
        fp = proj / rel
        source_hashes[rel] = sha(fp) if fp.exists() else None

    manifest = {
        "schema": "reactflow_delta.p2_learnability.v1",
        "run_id": out.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {"cuda_available": bool(torch.cuda.is_available()),
                "device_index": gpu_index, "device_name": gpu_name,
                "free_bytes": int(free_mem), "total_bytes": int(tot_mem),
                "cuda_visible_devices": args.cuda_device},
        "n_pool_studies": len(pool_studies),
        "n_pool_pairs": len(pair_recs),
        "n_distinct_publications": len(pubs),
        "publications": pubs,
        "publication_studies": {p: sorted(s) for p, s in pub_study.items()},
        "source_hashes": source_hashes,
        "models": models,
        "seeds": SEEDS,
        "table": results["table"],
        "per_publication_ap_seed0": ap_report,
        "fold_timing": fold_timing,
    }
    (out / "P2_learnability_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("\nDONE main LOOCV ->", out, flush=True)


if __name__ == "__main__":
    main()
