#!/usr/bin/env python3
"""prototype_caller_v5_stability_v1 — prototype validation of the CallerV5
continuous + measurement-error-abstention label against the CALLER_ENDPOINT_UNSTABLE
blocker (epoch 20 learnability gate).

Design (pre-registered rationale, no tuning):
  * continuous score  p_changer = 1 - null_exceedance  in [0,1] (higher = likely changer).
  * measurement-error abstention: sigma is unknown for held pairs in STRICT mode;
    model it as train_global_median * m where m is drawn (with replacement) from the
    distribution of per-group-median-sigma / train_global_median across TRAIN groups
    (STRICT-legal, train-only). Recompute p_changer under each draw. If the 95% CI of
    p_changer straddles the decision boundary 0.5, abstain (label not robust to sigma
    uncertainty). This is exactly the sigma-induced flip mechanism that caused the 53.4%
    binary flip between STRICT and TRANSDUCTIVE.

This prototype reports, on the development pool, the stability gate on the CALLED set:
  * publication-block Spearman rank correlation of p_changer (STRICT vs TRANSDUCTIVE)
  * binary-decision (p_changer>=0.5) flip rate on called set
  * called coverage (overall + per-pub)
Read-only; no frozen endpoint changed; no training; no confirmatory outcome.
"""
from __future__ import annotations

import argparse, json, math, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caller_v2 import _p_value, CallerV2
from caller_v3 import _empirical_scatter, _med_positive, CallerV3
from caller_v4 import CallerV4
from run_baselines_v6 import (
    load_cache, load_publication_map, build_pair_recs, build_rep_groups,
    rep_groups_for_train, build_pair_features_aligned_robust,
)

N_DRAWS = 15
CI_LEVEL = 0.95


def _cluster_one(z, eligible, window):
    """size-normalised RMS cluster statistic (same as CallerV2._cluster_with)."""
    best = 0.0
    n = len(z)
    i = 0
    while i < n:
        if not eligible[i] or z[i] is None:
            i += 1
            continue
        run_z = []
        j = i
        while j < n and eligible[j]:
            if z[j] is not None:
                run_z.append(float(z[j]))
            j += 1
        run_len = len(run_z)
        for start in range(run_len):
            s = 0.0
            for end in range(start, min(start + window, run_len)):
                s += run_z[end] ** 2
                k = end - start + 1
                if k < 1:
                    continue
                v = math.sqrt(s / k)
                if v > best:
                    best = v
        i = j
    return best


def _z_vector(pair, sigma, train_med):
    """per-position z with optional sigma override (array) or None."""
    n = len(pair)
    z = [None] * n
    for i in range(n):
        if not pair.eligibility_mask[i]:
            continue
        wt, mut = pair.wt_reactivity[i], pair.mutant_reactivity[i]
        if not (math.isfinite(wt) and math.isfinite(mut)):
            continue
        s = None
        if sigma is not None and i < len(sigma) and math.isfinite(sigma[i]) and sigma[i] > 0:
            s = float(sigma[i])
        if s is None and train_med is not None:
            s = train_med
        if s is None or not (s > 0):
            continue
        z[i] = (float(mut) - float(wt)) / (s * math.sqrt(2.0))
    return z


def _p_changer_from_stat(null, stat):
    e = _p_value(null, stat)
    return 1.0 - e


def prototype(cache_path, registry_path, out_path, n_draws=N_DRAWS, ci_level=CI_LEVEL):
    cache = load_cache(cache_path)
    pub_map = load_publication_map(registry_path)
    pair_recs, _ = build_pair_recs(cache, pub_map)
    all_rep_groups = build_rep_groups(cache["rec_index"])
    pf_all = {pid: build_pair_features_aligned_robust(v["pair"], v["wt"], v["mut"])
              for pid, v in pair_recs.items()}

    resolved = sorted({v["pub"] for v in pair_recs.values() if (v["pub"] or "").startswith("pmid_")})
    pub_study = defaultdict(set)
    for v in pair_recs.values():
        pub_study[v["pub"]].add(v["study"])

    # precompute held-group per-position sigma (TRANSDUCTIVE) per pub
    held_sigma = {}   # group_key -> sigma array
    for g in all_rep_groups:
        if g.n_replicates >= 2:
            held_sigma[g.group_key] = _empirical_scatter(g)

    lo_q = (1 - ci_level) / 2
    hi_q = 1 - lo_q

    per_pub = {}
    all_called_scores = {"strict": [], "trans": []}  # aligned by pair on called set
    pool = {"n_pairs_total": 0, "n_called": 0, "n_abstained": 0,
            "n_abstained_sigma": 0, "n_abstained_reliability": 0,
            "flip_rate_called": None, "spearman_called": None}

    for held_pub in resolved:
        train_studies = set()
        for p_ in resolved:
            if p_ != held_pub:
                train_studies |= pub_study[p_]
        train_groups = rep_groups_for_train(all_rep_groups, train_studies)
        caller = CallerV4(mode="STRICT_INDUCTIVE_WT_ALLOWED", seed=20260809).fit(train_groups, [])

        # train sigma distribution (group median sigma / train_global_median)
        train_med = caller._train_median_sigma
        mults = []
        for sc in caller._train_sigma_by_group.values():
            m = _med_positive(sc)
            if m is not None and train_med:
                mults.append(m / train_med)
        mults = np.asarray(mults, dtype=float) if mults else np.ones(1)
        rng = np.random.default_rng(20260811)

        pairs_of = [v for v in pair_recs.values() if v["pub"] == held_pub]
        pub_called = []
        pub_flip = 0
        for v in pairs_of:
            pf = pf_all[v["pair"].get("source_accession") + ":" + str(v["pair"]["mutant_profile_index"])]
            pool["n_pairs_total"] += 1
            rel = caller._unit_reliability(pf.group_key)
            if (not caller._structure_ok) or rel is None or rel < caller.icc_threshold:
                pool["n_abstained"] += 1
                pool["n_abstained_reliability"] += 1
                continue
            # STRICT base p_changer (train-med sigma)
            zs = _z_vector(pf, None, train_med)
            if not any(y is not None for y in zs):
                pool["n_abstained"] += 1
                pool["n_abstained_reliability"] += 1
                continue
            stat_base = _cluster_one(zs, list(pf.eligibility_mask), caller.cluster_window)
            pc_base = _p_changer_from_stat(caller._null, stat_base)
            # sigma-perturbation draws
            draws = []
            for _ in range(n_draws):
                m = float(rng.choice(mults))
                zd = _z_vector(pf, None, train_med * m)
                sd = _cluster_one(zd, list(pf.eligibility_mask), caller.cluster_window)
                draws.append(_p_changer_from_stat(caller._null, sd))
            draws.sort()
            lo, hi = draws[int(lo_q * len(draws))], draws[int(hi_q * len(draws))]
            # abstain if CI straddles 0.5
            if lo <= 0.5 <= hi:
                pool["n_abstained"] += 1
                pool["n_abstained_sigma"] += 1
                continue
            # called: mean p_changer
            pc = float(np.mean(draws))
            # TRANSDUCTIVE p_changer (held sigma)
            hs = held_sigma.get(pf.group_key)
            z_h = _z_vector(pf, hs, train_med)
            st_h = _cluster_one(z_h, list(pf.eligibility_mask), caller.cluster_window)
            pc_trans = _p_changer_from_stat(caller._null, st_h)
            pool["n_called"] += 1
            pub_called.append(1)
            all_called_scores["strict"].append(pc)
            all_called_scores["trans"].append(pc_trans)
            if (pc >= 0.5) != (pc_trans >= 0.5):
                pub_flip += 1
        per_pub[held_pub] = {
            "n_pairs": len(pairs_of),
            "n_called": len(pub_called),
            "call_coverage": len(pub_called) / len(pairs_of) if pairs_of else None,
            "n_called_flip": pub_flip,
        }

    # overall stability
    s = np.asarray(all_called_scores["strict"])
    t = np.asarray(all_called_scores["trans"])
    if len(s):
        from scipy.stats import spearmanr
        rho, _ = spearmanr(s, t)
        pool["spearman_called"] = float(rho)
        pool["flip_rate_called"] = float(np.mean((s >= 0.5) != (t >= 0.5)))
    pool["call_coverage"] = pool["n_called"] / pool["n_pairs_total"] if pool["n_pairs_total"] else None

    report = {
        "schema": "prototype.caller_v5.stability.v1",
        "authority_epoch": 20,
        "n_draws": n_draws, "ci_level": ci_level,
        "pool": pool,
        "per_publication": per_pub,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE -> {out}")
    print(f"spearman(called)={pool['spearman_called']}  flip_rate(called)={pool['flip_rate_called']}  "
          f"coverage={pool['call_coverage']}  n_called={pool['n_called']}/{pool['n_pairs_total']}")
    for pub, r in per_pub.items():
        print(f"{pub:12s} cov={r['call_coverage']:.3f} n_call={r['n_called']} flip={r['n_called_flip']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--ci-level", type=float, default=CI_LEVEL)
    args = ap.parse_args()
    prototype(args.cache, args.registry, args.out, args.n_draws, args.ci_level)


if __name__ == "__main__":
    sys.exit(main())