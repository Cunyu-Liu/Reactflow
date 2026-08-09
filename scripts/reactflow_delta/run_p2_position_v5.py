#!/usr/bin/env python3
"""P2v5 — position-granularity learnability gate (endpoint_v5, epoch 17).

Route C (user explicit grant, epoch 17). After the P2 primary pair-level
binary-changer publication-macro AUPRC was adjudicated STOP (fail-closed) under
endpoint_v4, this re-estimates the SAME core scientific question (is the
mutation-induced reactivity response cross-publication learnable) at the ELIGIBLE
POSITION unit.  Each publication contributes thousands of position units, which
repairs the statistical power that endpoint_v4 lacked (only 8-10 publication
units -> wide paired bootstrap delta CI and DEGENERATE_NO_POWER permutation).

Design (per endpoint_v5 / amendment Route C):
  * Outer unit = publication (same PMID = one publication), identical to
    run_p2_v3.  For each outer fold: hold out ONE publication, fit the
    fold-local caller (caller_v3) on the REMAINING train publications, compute
    per-position empirical-scatter z_i for ALL pool positions with that caller,
    define position labels y_i = 1[|z_i| > Z_CUT] (Z_CUT = 2.0 pre-registered),
    train models on train-publication positions, predict per-position scores on
    the held-out publication.
  * Score = per-position P(y_i=1) (direct position-level output; NO pair-level
    aggregation).
  * Metric = publication-macro position-AUPRC over NON-DEGENERATE publications
    (each publication's position-AP computed over its thousands of positions,
    then macro-averaged, each publication equal weight).  Constant-label
    publications are explicitly listed and excluded; exclusion set is written to
    the result (no silent drop).
  * Paired publication-block bootstrap delta CI (vs trivial) and publication-block
    permutation p = (b+1)/(B+1), as in evaluate_v2 but at position granularity.
  * Pre-registered thresholds (frozen before run): delta > 0, bootstrap CI low > 0
    (alpha=0.05, n_boot=1000), permutation p < 0.05 (n_perm=1000),
    n_non_degenerate_publications >= 5.  Any unmet -> STOP (fail-closed).

Information permission (endpoint_v5): allowed = WT sequence + exact single-nt
mutation + allowed WT reactivity state + per-group empirical WT scatter +
condition; forbidden = mutant reactivity profile / mutant noise / actual-alt
thermo / test-fold outcome / target mask as input.  Caller_v3 labels use
train-fold replicates only (outer outcome invisible).

Models: trivial (majority predictor), logistic, GBM tree (sklearn), position-level
MLP (PyTorch, CUDA).  Every neural model MUST run on CUDA; if CUDA is unavailable
the run STOPS (no silent CPU fallback).
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, pickle, random, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import yaml

import torch  # noqa: E402

# --- read-only reuse ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from caller_v3 import CallerV3  # noqa: E402
from caller_v2 import (  # noqa: E402
    compute_eligible_mask, _finite,
)
from run_p2_v3 import (  # noqa: E402  (reuse feature builder + helpers)
    _study_of, load_cache, sanitize_records, build_rep_groups, rep_groups_for_train,
    build_pair_features_aligned, build_feature, edited_index, BASE_MAP, WINDOW,
    HALF, _base_oh, _norm_react, _norm_err, _oh, _oh_index, PROBES, MODIFIERS,
    EXPTYPES,
)

# ---------------------------------------------------------------------------
# Frozen / pre-registered constants (endpoint_v5)
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]          # documented deterministic seeds
Z_CUT = 2.0                      # pre-registered per-position changer threshold
N_PERM = 1000
N_BOOT = 1000
ALPHA = 0.05
MIN_NON_DEGEN_PUBS = 5           # confirmatory GO premise
CALLER_SEED = 20260807


# ---------------------------------------------------------------------------
# AUPRC (numeric, position-level)
# ---------------------------------------------------------------------------
def average_precision_numeric(y_true, y_score):
    """Average precision for a list of binary labels + real-valued scores."""
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    tp = 0.0
    fp = 0.0
    prec_sum = 0.0
    n_pos = float(sum(1 for v in y_true if v))
    if n_pos == 0:
        return float("nan")
    for rank, i in enumerate(order, start=1):
        if y_true[i]:
            tp += 1.0
            prec_sum += tp / rank
    return prec_sum / n_pos


def is_unidentifiable(x):
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def publication_macro_position_auprc(publications, labels, scores):
    """Publication-macro position-AUPRC over NON-DEGENERATE publications.

    Each publication's position-AP is computed over its eligible positions, then
    macro-averaged.  Constant-label (all-changer or all-non-changer) publications
    are UNIDENTIFIABLE at the AP level and cause the whole metric to be
    UNIDENTIFIABLE unless excluded via the explicit Route-C exclusion policy.
    """
    groups = defaultdict(list)
    for pub, lab, sc in zip(publications, labels, scores):
        groups[pub].append((int(bool(lab)), float(sc)))
    if not groups:
        return None
    pub_aps = []
    for pub in sorted(groups, key=str):
        labs = [t for t, _ in groups[pub]]
        scos = [s for _, s in groups[pub]]
        if len(set(labs)) <= 1:
            return None
        ap = average_precision_numeric(labs, scos)
        if is_unidentifiable(ap):
            return None
        pub_aps.append(ap)
    return sum(pub_aps) / float(len(pub_aps))


def publication_macro_position_auprc_non_degenerate(publications, labels, scores):
    """Same but returns (metric, degenerate_publications, non_degenerate_pubs)."""
    groups = defaultdict(list)
    for pub, lab, sc in zip(publications, labels, scores):
        groups[pub].append((int(bool(lab)), float(sc)))
    if not groups:
        return None, list(groups.keys()), []
    degenerate = []
    non_deg = []
    pub_aps = []
    for pub in sorted(groups, key=str):
        labs = [t for t, _ in groups[pub]]
        scos = [s for _, s in groups[pub]]
        if len(set(labs)) <= 1 or is_unidentifiable(average_precision_numeric(labs, scos)):
            degenerate.append(pub)
        else:
            ap = average_precision_numeric(labs, scos)
            if is_unidentifiable(ap):
                degenerate.append(pub)
            else:
                pub_aps.append(ap)
                non_deg.append(pub)
    if not non_deg:
        return None, degenerate, []
    return sum(pub_aps) / float(len(pub_aps)), degenerate, non_deg


# ---------------------------------------------------------------------------
# Publication-block permutation (position granularity)
# ---------------------------------------------------------------------------
def publication_block_permutation(publications, labels, scores, seed=0, n_perm=N_PERM):
    """Permute score-blocks within equal-size publication classes; return p.

    Each publication's position block is kept intact (no mixed blocks).  Blocks
    are permuted among equal-size classes.  p = (b+1)/(B+1).
    """
    groups = defaultdict(list)
    for pub, lab, sc in zip(publications, labels, scores):
        groups[pub].append((int(bool(lab)), float(sc)))
    pub_ids = sorted(groups, key=str)
    if len(pub_ids) < 2:
        return {"statistic": None, "p_value": None, "null": []}
    label_blocks = [[t for t, _ in groups[p]] for p in pub_ids]
    score_blocks = [[s for _, s in groups[p]] for p in pub_ids]

    def metric_for(perm_labels, perm_scores, perm_pubs):
        m, deg, nondeg = publication_macro_position_auprc_non_degenerate(
            perm_pubs, perm_labels, perm_scores)
        return m, len(nondeg)

    real, real_deg, real_nondeg = publication_macro_position_auprc_non_degenerate(
        publications, labels, scores)
    if is_unidentifiable(real) or len(real_nondeg) < MIN_NON_DEGEN_PUBS:
        return {"statistic": real, "p_value": None, "null": [],
                "n_non_degenerate": len(real_nondeg), "degenerate": real_deg}

    size_classes = defaultdict(list)
    for idx, lb in enumerate(label_blocks):
        size_classes[len(lb)].append(idx)

    rng = random.Random(seed)
    null_stats = []
    b = 0
    for _ in range(n_perm):
        perm_score_blocks = [None] * len(score_blocks)
        for size, idxs in size_classes.items():
            perm_idxs = idxs[:]
            rng.shuffle(perm_idxs)
            for orig, dest in zip(idxs, perm_idxs):
                perm_score_blocks[dest] = score_blocks[orig]
        perm_labels = [v for blk in label_blocks for v in blk]
        perm_scores = [v for blk in perm_score_blocks for v in blk]
        perm_pubs = [p for p, lb in zip(pub_ids, label_blocks) for _ in lb]
        m, nnd = metric_for(perm_labels, perm_scores, perm_pubs)
        if is_unidentifiable(m) or nnd < MIN_NON_DEGEN_PUBS:
            null_stats.append(float("nan"))
            continue
        null_stats.append(m)
        if m >= real:
            b += 1
    nonnan = [s for s in null_stats if not math.isnan(s)]
    return {"statistic": real, "p_value": (b + 1) / (len(nonnan) + 1) if nonnan else None,
            "b": b, "null": sorted(nonnan), "n_null": len(nonnan),
            "n_non_degenerate": len(real_nondeg), "degenerate": real_deg}


# ---------------------------------------------------------------------------
# Paired publication-block bootstrap delta CI
# ---------------------------------------------------------------------------
def paired_publication_block_delta_ci(per_pub_model_ap, per_pub_trivial_ap,
                                      seed=0, n_boot=N_BOOT, alpha=ALPHA):
    """Cluster-bootstrap CI on the per-publication delta (model_ap - trivial_ap).

    Resamples publications with replacement, computes mean delta, percentile CI.
    """
    assert len(per_pub_model_ap) == len(per_pub_trivial_ap)
    deltas = [m - t for m, t in zip(per_pub_model_ap, per_pub_trivial_ap)]
    n = len(deltas)
    if n < 3:
        return {"lower": None, "upper": None, "point": float(np.mean(deltas)) if deltas else None,
                "n_pub": n, "n_boot": n_boot}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(float(np.mean(sample)))
    boots.sort()
    lo = boots[int(round((alpha / 2.0) * (n_boot - 1)))]
    hi = boots[int(round((1.0 - alpha / 2.0) * (n_boot - 1)))]
    return {"lower": lo, "upper": hi, "point": float(np.mean(deltas)),
            "n_pub": n, "n_boot": n_boot, "alpha": alpha}


# ---------------------------------------------------------------------------
# CUDA guard
# ---------------------------------------------------------------------------
def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Contract: STOP, no silent CPU fallback.")
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Position feature + label builder
# ---------------------------------------------------------------------------
def build_position_features_and_labels(pair, wt_rec, mut_rec, caller, z_cut=Z_CUT):
    """Build per-eligible-position feature vectors + binary labels.

    Feature: reuse run_p2_v3.build_feature's WT-window + exact-alt + condition
    (WT-only, fold-invariant) but append the POSITION COORDINATE (normalized) and
    a one-hot of the local base so the model can distinguish positions.
    Label: y_i = 1[|z_i| > z_cut] from caller_v3 per-position z.

    Returns (feats, labels, elig_indices) where feats is a list of numpy arrays
    aligned to the eligible positions.
    """
    pf = build_pair_features_aligned(pair, wt_rec, mut_rec)
    z, eligible = caller._z_for_pair(pf)
    base_feat = build_feature(pair, wt_rec, use_wt_anchor=True,
                              use_exact_alt=True, use_condition=True)
    n_base = base_feat.shape[0]
    seq = wt_rec.get("canonical_sequence") or ""
    n_seq = len(seq)
    ei = edited_index(pair)
    feats = []
    labels = []
    for i, el in enumerate(eligible):
        if not el:
            continue
        zval = z[i]
        if zval is None or not _finite(zval):
            continue
        # position feature: base_feat + normalized coordinate + local base one-hot
        coord = np.array([i / max(n_seq, 1)], dtype=np.float32)
        base_oh = _base_oh(seq[i]) if 0 <= i < n_seq else np.zeros(5, dtype=np.float32)
        f = np.concatenate([base_feat, coord, base_oh]).astype(np.float32)
        feats.append(f)
        labels.append(1.0 if abs(zval) > z_cut else 0.0)
    return feats, labels


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _make_mlp(in_dim, hidden, out, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, out),
    )


def train_torch(model, X, y, device, epochs=20, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X).float().to(device)
    yt = torch.from_numpy(y).float().to(device)
    model.to(device)
    model.train()
    n = Xt.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            loss = lossf(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
    return model


def predict_torch(model, X, device):
    model.eval()
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(Xt).squeeze(-1)).cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split-yaml", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cuda-device", default="0")
    ap.add_argument("--models", default="trivial,logistic,gbm,p2_mlp")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after CUDA_VISIBLE_DEVICES=" + args.cuda_device)
    device = require_cuda()
    gpu_index = torch.cuda.current_device()
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[p2v5] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    split = yaml.safe_load(open(args.split_yaml))
    pub_map = split["publication_map"]
    study_roles = split["study_roles"]

    cache = load_cache(args.cache)
    rec_index = cache["rec_index"]
    sanitize_records(rec_index)
    pairs = cache["pairs"]
    pool_studies = set(cache["pool"])
    test_studies = {s for s, r in study_roles.items() if r == "test"}
    pool_studies = pool_studies - test_studies

    pair_recs = {}
    for p in pairs:
        if _study_of(p["source_accession"]) in test_studies:
            continue
        wt = rec_index.get((p["source_accession"], p["wt_profile_index"], p["asset_name"]))
        mu = rec_index.get((p["source_accession"], p["mutant_profile_index"], p["asset_name"]))
        if wt is None or mu is None:
            continue
        pair_recs[p["source_accession"] + ":" + str(p["mutant_profile_index"])] = {
            "pair": p, "wt": wt, "mut": mu,
            "study": _study_of(p["source_accession"]),
            "pub": pub_map.get(_study_of(p["source_accession"]), "UNKNOWN"),
        }
    print(f"[p2v5] pool_studies={sorted(pool_studies)} n_pairs={len(pair_recs)}", flush=True)

    all_rep_groups = build_rep_groups(rec_index, study_whitelist=pool_studies)
    pubs = sorted({v["pub"] for v in pair_recs.values()})
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])
    print(f"[p2v5] distinct publications: {len(pubs)} -> {pubs}", flush=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Fold-invariant PairFeatures + base features (WT-only, allowed input)
    print("[p2v5] precomputing fold-invariant pair features...", flush=True)
    t0 = time.time()
    pf_all = {}
    base_feat_dim = None
    for pid, v in pair_recs.items():
        pf_all[pid] = build_pair_features_aligned(v["pair"], v["wt"], v["mut"])
        bf = build_feature(v["pair"], v["wt"], True, True, True)
        base_feat_dim = bf.shape[0]
    print(f"[p2v5] precomputed {len(pf_all)} pair features in {time.time()-t0:.1f}s "
          f"base_feat_dim={base_feat_dim}", flush=True)

    # feature dim = base + 1 (coord) + 5 (base one-hot)
    feat_dim = base_feat_dim + 6
    hidden = 64

    # ---- LOOCV ----
    heldout = {m: {s: {"pub": [], "label": [], "score": [], "study": []} for s in SEEDS}
               for m in models}
    fold_timing = {}
    fold_stats = {}

    for fold, held_pub in enumerate(pubs):
        t_fold = time.time()
        train_studies = set()
        for p_ in pubs:
            if p_ != held_pub:
                train_studies |= pub_study[p_]
        train_groups = rep_groups_for_train(all_rep_groups, train_studies)
        caller = CallerV3(seed=CALLER_SEED).fit(
            train_groups, [], noise_replicate_groups=all_rep_groups)

        # build position features + labels for train and held pairs
        tr_feats, tr_labels, tr_pubs, tr_studies = [], [], [], []
        te_feats, te_labels, te_pubs, te_studies = [], [], [], []
        for pid, v in pair_recs.items():
            feats, labs = build_position_features_and_labels(
                v["pair"], v["wt"], v["mut"], caller)
            if not feats:
                continue
            is_held = (v["pub"] == held_pub)
            if is_held:
                te_feats.extend(feats); te_labels.extend(labs)
                te_pubs.extend([v["pub"]] * len(feats)); te_studies.extend([v["study"]] * len(feats))
            else:
                tr_feats.extend(feats); tr_labels.extend(labs)
                tr_pubs.extend([v["pub"]] * len(feats)); tr_studies.extend([v["study"]] * len(feats))

        Xtr = np.stack(tr_feats) if tr_feats else np.zeros((0, feat_dim), dtype=np.float32)
        ytr = np.array(tr_labels, dtype=np.float32)
        Xte = np.stack(te_feats) if te_feats else np.zeros((0, feat_dim), dtype=np.float32)
        yte = np.array(te_labels, dtype=np.float32)
        te_pub_arr = np.array(te_pubs)
        te_study_arr = np.array(te_studies)

        if len(te_feats) == 0 or len(tr_feats) == 0:
            fold_timing[held_pub] = {"n_train_pos": len(tr_feats), "n_held_pos": len(te_feats),
                                     "skipped": True, "seconds": round(time.time()-t_fold, 1)}
            print(f"[fold] held={held_pub} SKIPPED (empty positions: train={len(tr_feats)} held={len(te_feats)})", flush=True)
            continue

        tr_pos_rate = float(np.mean(ytr))
        te_pos_rate = float(np.mean(yte))
        fold_stats[held_pub] = {"n_train_pos": int(len(tr_feats)), "n_held_pos": int(len(te_feats)),
                                "train_pos_rate": tr_pos_rate, "held_pos_rate": te_pos_rate,
                                "seconds": round(time.time()-t_fold, 1)}
        print(f"[fold] held={held_pub} train_pos={len(tr_feats)} held_pos={len(te_feats)} "
              f"tr_rate={tr_pos_rate:.3f} te_rate={te_pos_rate:.3f}", flush=True)

        # trivial predictor = train positive rate (constant)
        for seed in SEEDS:
            h = heldout["trivial"][seed]
            h["pub"].extend(te_pub_arr.tolist())
            h["label"].extend(yte.tolist())
            h["score"].extend([tr_pos_rate] * len(yte))
            h["study"].extend(te_study_arr.tolist())

        if "logistic" in models:
            from sklearn.linear_model import LogisticRegression
            for seed in SEEDS:
                clf = LogisticRegression(max_iter=2000, random_state=seed)
                clf.fit(Xtr, ytr)
                sc = clf.predict_proba(Xte)[:, 1]
                h = heldout["logistic"][seed]
                h["pub"].extend(te_pub_arr.tolist())
                h["label"].extend(yte.tolist())
                h["score"].extend(sc.tolist())
                h["study"].extend(te_study_arr.tolist())

        if "gbm" in models:
            from sklearn.ensemble import HistGradientBoostingClassifier
            for seed in SEEDS:
                clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
                clf.fit(Xtr, ytr)
                sc = clf.predict_proba(Xte)[:, 1]
                h = heldout["gbm"][seed]
                h["pub"].extend(te_pub_arr.tolist())
                h["label"].extend(yte.tolist())
                h["score"].extend(sc.tolist())
                h["study"].extend(te_study_arr.tolist())

        if "p2_mlp" in models:
            for seed in SEEDS:
                model = _make_mlp(feat_dim, hidden, 1, seed)
                model = train_torch(model, Xtr, ytr, device, epochs=20, bs=256, lr=1e-3, seed=seed)
                sc = predict_torch(model, Xte, device)
                h = heldout["p2_mlp"][seed]
                h["pub"].extend(te_pub_arr.tolist())
                h["label"].extend(yte.tolist())
                h["score"].extend(sc.tolist())
                h["study"].extend(te_study_arr.tolist())

        fold_timing[held_pub] = {"n_train_pos": len(tr_feats), "n_held_pos": len(te_feats),
                                 "seconds": round(time.time()-t_fold, 1)}
        print(f"[fold] held={held_pub} done t={time.time()-t_fold:.1f}s", flush=True)

    # ---- evaluate: publication-macro position-AUPRC + delta + CI + permutation ----
    print("\n[p2v5] publication-macro position-AUPRC by model x seed:", flush=True)
    results = {}
    per_model = {}
    for m in models:
        per_model[m] = {}
        for seed in SEEDS:
            h = heldout[m][seed]
            metric, deg, nondeg = publication_macro_position_auprc_non_degenerate(
                h["pub"], h["label"], h["score"])
            # trivial baseline for delta
            th = heldout["trivial"][seed]
            triv_metric, triv_deg, triv_nondeg = publication_macro_position_auprc_non_degenerate(
                th["pub"], th["label"], th["score"])
            perm = publication_block_permutation(h["pub"], h["label"], h["score"], seed=seed)
            # per-publication AP for paired bootstrap delta
            per_pub_model = {}
            per_pub_triv = {}
            groups = defaultdict(list)
            for p, l, s in zip(h["pub"], h["label"], h["score"]):
                groups[p].append((int(bool(l)), float(s)))
            tgroups = defaultdict(list)
            for p, l, s in zip(th["pub"], th["label"], th["score"]):
                tgroups[p].append((int(bool(l)), float(s)))
            for p in sorted(set(h["pub"])):
                if p in nondeg or p in deg:
                    labs = [t for t, _ in groups[p]]; scos = [s for _, s in groups[p]]
                    if len(set(labs)) > 1:
                        per_pub_model[p] = average_precision_numeric(labs, scos)
                    tlab = [t for t, _ in tgroups[p]]; tsc = [s for _, s in tgroups[p]]
                    if len(set(tlab)) > 1:
                        per_pub_triv[p] = average_precision_numeric(tlab, tsc)
            common = [p for p in per_pub_model if p in per_pub_triv]
            mvals = [per_pub_model[p] for p in common]
            tvals = [per_pub_triv[p] for p in common]
            ci = paired_publication_block_delta_ci(mvals, tvals, seed=seed)
            delta = (metric - triv_metric) if (not is_unidentifiable(metric) and not is_unidentifiable(triv_metric)) else None
            per_model[m][seed] = {
                "metric": metric, "trivial_metric": triv_metric, "delta": delta,
                "ci": ci, "permutation_p": perm["p_value"],
                "n_non_degenerate": perm["n_non_degenerate"],
                "degenerate_publications": perm["degenerate"],
                "metric_identifiable": not is_unidentifiable(metric),
            }
            results[f"{m}:{seed}"] = per_model[m][seed]
            print(f"  {m} s{seed}: metric={metric} triv={triv_metric} delta={delta} "
                  f"ci={ci.get('lower')}..{ci.get('upper')} perm_p={perm['p_value']}", flush=True)

    # ---- adjudicate ----
    go_by_model = {}
    for m in models:
        if m == "trivial":
            continue
        hits = []
        for seed in SEEDS:
            r = per_model[m][seed]
            cond = {
                "delta_gt_0": (r["delta"] is not None and r["delta"] > 0),
                "ci_low_gt_0": (r["ci"]["lower"] is not None and r["ci"]["lower"] > 0),
                "perm_p_lt_0.05": (r["permutation_p"] is not None and r["permutation_p"] < 0.05),
                "n_ndeg_ge_5": (r["n_non_degenerate"] is not None and r["n_non_degenerate"] >= MIN_NON_DEGEN_PUBS),
            }
            hits.append(all(cond.values()))
        go_by_model[m] = {
            "all_seeds_go": all(hits),
            "n_seeds_go": sum(hits),
            "criteria_by_seed": {
                s: {k: per_model[m][s].get(k if k in ("delta_gt_0",) else k, None)
                    for k in []} for s in SEEDS},
            "seed_go": {s: bool(h) for s, h in zip(SEEDS, hits)},
        }

    verdict = "GO" if any(go_by_model[m]["all_seeds_go"] for m in go_by_model) else "STOP"
    print(f"\n[p2v5] ADJUDICATION: {verdict}", flush=True)

    # ---- write results ----
    summary = {
        "run_id": out.name,
        "endpoint": "endpoint_v5",
        "authority_epoch": 17,
        "route": "Route C position-granularity",
        "Z_CUT": Z_CUT,
        "n_pool_pairs": len(pair_recs),
        "n_distinct_publications": len(pubs),
        "publications": pubs,
        "fold_stats": fold_stats,
        "fold_timing": fold_timing,
        "models": models,
        "seeds": SEEDS,
        "verdict": verdict,
        "go_by_model": go_by_model,
        "pre_registered_thresholds": {
            "Z_CUT": Z_CUT, "n_seeds": len(SEEDS), "n_boot": N_BOOT,
            "n_perm": N_PERM, "alpha": ALPHA, "min_non_degenerate_pubs": MIN_NON_DEGEN_PUBS,
            "go_requires": ["delta>0", "ci_low>0", "perm_p<0.05", "n_ndeg>=5"],
        },
        "per_model": per_model,
        "table": results,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    for m in models:
        for seed in SEEDS:
            h = heldout[m][seed]
            np.savez_compressed(
                out / f"heldout_{m}_seed{seed}.npz",
                pub=np.array(h["pub"]), label=np.array(h["label"]),
                score=np.array(h["score"]), study=np.array(h["study"]))

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proj = Path(__file__).resolve().parent.parent.parent
    source_hashes = {}
    for rel in ["configs/reactflow_delta/endpoint_v5.yaml",
                "configs/reactflow_delta/split_v2.yaml",
                "configs/reactflow_delta/authority_epoch_17.sentinel.yaml",
                "docs/contracts/amendments/reactflow_delta_v4_epoch17_endpoint_v5_position_granularity_20260808.yaml",
                "scripts/reactflow_delta/caller_v3.py",
                "scripts/reactflow_delta/evaluate_v2.py",
                "scripts/reactflow_delta/run_p2_v3.py",
                "scripts/reactflow_delta/run_p2_position_v5.py"]:
        fp = proj / rel
        if fp.exists():
            source_hashes[rel] = sha(fp)
    (out / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[p2v5] wrote results to {out}", flush=True)


if __name__ == "__main__":
    main()
