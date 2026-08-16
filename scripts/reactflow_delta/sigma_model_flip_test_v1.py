#!/usr/bin/env python3
"""Test whether a train-only monotone (isotonic) per-position sigma model
reduces the STRICT vs TRANSDUCTIVE caller label flip, vs the current
single train-global-median constant.
Read-only diagnostic; no frozen endpoint changed.
"""
import pickle, numpy as np, sys, math, json
from collections import defaultdict
sys.path.insert(0, '/home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/scripts/reactflow_delta')
from run_p2_v3 import build_rep_groups
from caller_v3 import _empirical_scatter, _med_positive
from caller_v4 import CallerV4
from sklearn.isotonic import IsotonicRegression
from run_baselines_v6 import (
    load_publication_map, build_pair_recs, build_pair_features_aligned_robust,
)
from caller_v2 import _p_value

CACHE = '/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/p2_cache/p2_cache.pkl'
REG = '/home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/data_registry/reactflow_delta/pair_publication_registry_v1.tsv'

d = pickle.load(open(CACHE, 'rb'))
pm = load_publication_map(REG)
pair_recs, _ = build_pair_recs(d, pm)
all_groups = build_rep_groups(d['rec_index'])
resolved = sorted({v['pub'] for v in pair_recs.values() if (v['pub'] or '').startswith('pmid_')})
pub_study = defaultdict(set)
for v in pair_recs.values():
    pub_study[v['pub']].add(v['study'])

held_sigma = {g.group_key: _empirical_scatter(g) for g in all_groups if g.n_replicates >= 2}


def build_sigma_model(train_groups):
    X = []; S = []
    for g in train_groups:
        if g.n_replicates < 2:
            continue
        arr = [np.asarray(p, float) for p in g.wt_profiles]
        L0 = min([len(a) for a in arr]) if arr else 0
        if L0 < 1:
            continue
        wt = np.stack([a[:L0] for a in arr])
        sig = _empirical_scatter(g)
        L = min(L0, len(sig), len(g.eligibility_mask))
        mu = wt[:, :L].mean(axis=0)
        for i in range(L):
            if not g.eligibility_mask[i]:
                continue
            if not np.isfinite(sig[i]) or sig[i] <= 0:
                continue
            if not np.isfinite(mu[i]):
                continue
            X.append(mu[i]); S.append(sig[i])
    X = np.array(X); S = np.array(S)
    if len(X) < 10:
        return None
    iso = IsotonicRegression(out_of_bounds='clip').fit(X, np.log(S))
    return iso


def z_from_model(pf, iso, med_sigma):
    n = len(pf); z = [None] * n
    react = np.asarray(pf.wt_reactivity, float)
    for i in range(n):
        if not pf.eligibility_mask[i]:
            continue
        wt, mut = pf.wt_reactivity[i], pf.mutant_reactivity[i]
        if not (math.isfinite(wt) and math.isfinite(mut)):
            continue
        r = react[i]
        s = None
        if np.isfinite(r):
            s = float(np.exp(iso.predict([r])[0]))
        if s is None or not (s > 0):
            s = med_sigma
        if s is None or not (s > 0):
            continue
        z[i] = (float(mut) - float(wt)) / (s * math.sqrt(2.0))
    return z


def z_from_sigma(pf, sig, med_sigma):
    n = len(pf); z = [None] * n
    for i in range(n):
        if not pf.eligibility_mask[i]:
            continue
        wt, mut = pf.wt_reactivity[i], pf.mutant_reactivity[i]
        if not (math.isfinite(wt) and math.isfinite(mut)):
            continue
        s = None
        if sig is not None and i < len(sig) and np.isfinite(sig[i]) and sig[i] > 0:
            s = float(sig[i])
        if s is None:
            s = med_sigma
        if s is None or not (s > 0):
            continue
        z[i] = (float(mut) - float(wt)) / (s * math.sqrt(2.0))
    return z


def cluster_one(z, eligible, window):
    best = 0.0; n = len(z); i = 0
    while i < n:
        if not eligible[i] or z[i] is None:
            i += 1; continue
        run = []; j = i
        while j < n and eligible[j]:
            if z[j] is not None:
                run.append(float(z[j]))
            j += 1
        rl = len(run)
        for st in range(rl):
            s = 0.0
            for e in range(st, min(st + window, rl)):
                s += run[e] ** 2
                k = e - st + 1
                if k >= 1:
                    v = math.sqrt(s / k)
                    if v > best:
                        best = v
        i = j
    return best


def p_from_stat(null, stat):
    return 1.0 - _p_value(null, stat)


# baseline: current single-train-median constant (recompute reference flip)
results = {"sigma_model": {}, "constant": {}}
for held in resolved:
    train_groups = [g for g in all_groups if g.study in pub_study[held]]
    # note: STRICT needs train = all EXCEPT held
    train_studies = set()
    for p_ in resolved:
        if p_ != held:
            train_studies |= pub_study[p_]
    train_groups = [g for g in all_groups if g.study in train_studies]
    caller = CallerV4(mode="STRICT_INDUCTIVE_WT_ALLOWED", seed=20260809).fit(train_groups, [])
    iso = build_sigma_model(train_groups)
    train_med = caller._train_median_sigma
    for variant in ("sigma_model", "constant"):
        if variant == "sigma_model" and iso is None:
            continue
        flips = 0; tot = 0
        for v in pair_recs.values():
            if v['pub'] != held:
                continue
            pf = build_pair_features_aligned_robust(v['pair'], v['wt'], v['mut'])
            rel = caller._unit_reliability(pf.group_key)
            if (not caller._structure_ok) or rel is None or rel < caller.icc_threshold:
                continue
            if variant == "sigma_model":
                zm = z_from_model(pf, iso, train_med)
            else:
                zm = z_from_sigma(pf, None, train_med)  # constant train_med
            if not any(y is not None for y in zm):
                continue
            sm = cluster_one(zm, list(pf.eligibility_mask), caller.cluster_window)
            pm_ = p_from_stat(caller._null, sm); lm = 1 if pm_ >= 0.5 else 0
            hs = held_sigma.get(pf.group_key)
            zh = z_from_sigma(pf, hs, train_med)
            sh = cluster_one(zh, list(pf.eligibility_mask), caller.cluster_window)
            ph = p_from_stat(caller._null, sh); lt = 1 if ph >= 0.5 else 0
            tot += 1
            if lm != lt:
                flips += 1
        results[variant][held] = {"flip": flips, "n": tot,
                                  "rate": round(flips / tot, 4) if tot else None}

# aggregate
for variant in ("sigma_model", "constant"):
    tot = sum(r['n'] for r in results[variant].values())
    fl = sum(r['flip'] for r in results[variant].values())
    print(f"[{variant}] total_pairs={tot} overall_flip={round(fl/tot,4) if tot else None}")
    print(f"  per-pub flip rates: " + ", ".join(
        f"{k}={r['rate']}" for k, r in results[variant].items() if r['rate'] is not None))

out = "/tmp/sigma_model_flip_test.json"
json.dump(results, open(out, "w"), indent=2, sort_keys=True)
print("WROTE", out)
