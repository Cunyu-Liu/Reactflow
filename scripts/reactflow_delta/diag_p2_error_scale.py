#!/usr/bin/env python3
"""CPU-only diagnostic: characterize P2 label degeneracy (error-scale audit).

Loads the frozen P2 cache and examines:
  1. Distribution of reported per-position errors (wt_err / mut_err)
  2. Distribution of WT-WT replicate z-scores (replicate disagreement / error)
  3. The identity + observed deltas of the caller "changer" pairs
  4. Whether per-position reactivity/error magnitude scales are plausible
This is data analysis only (no GPU training) and makes NO scientific claim.
"""
import math
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/cunyuliu/reactflow_delta_goal_20260729/scripts/reactflow_delta")
from caller_v2 import CallerV2  # noqa: E402

CACHE = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/p2_cache/p2_cache.pkl"


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def main():
    with open(CACHE, "rb") as fh:
        cache = pickle.load(fh)
    rec_index = cache["rec_index"]
    pairs = cache["pairs"]
    pool = set(cache["pool"])
    print(f"[diag] pairs={len(pairs)} pool_studies={len(pool)} rec_index={len(rec_index)}")

    # ---- 1. reported error scale ----
    all_err = []
    all_react = []
    n_nan_err = 0
    n_zero_err = 0
    for rec in rec_index.values():
        tf = rec.get("reactivity_layers", {}).get("train_frozen", {}) or rec.get("reactivity_layers", {}).get("raw", {})
        err = tf.get("error") or []
        react = tf.get("reactivity") or []
        for e, r in zip(err, react):
            if not finite(e):
                n_nan_err += 1
                continue
            if e <= 0:
                n_zero_err += 1
            all_err.append(float(e))
            if finite(r):
                all_react.append(float(r))
    arr_e = np.asarray(all_err)
    arr_r = np.asarray(all_react)
    print("\n[diag1] reported per-position ERROR distribution:")
    print(f"  n_nonzero_error={int(np.sum(arr_e>0))} n_zero_or_nan_error={n_zero_err + n_nan_err}")
    if arr_e.size:
        print(f"  error: min={arr_e.min():.4g} p25={np.percentile(arr_e,25):.4g} "
              f"median={np.median(arr_e):.4g} mean={arr_e.mean():.4g} "
              f"p75={np.percentile(arr_e,75):.4g} p99={np.percentile(arr_e,99):.4g} max={arr_e.max():.4g}")
    if arr_r.size:
        print(f"  reactivity: min={arr_r.min():.4g} median={np.median(arr_r):.4g} "
              f"mean={arr_r.mean():.4g} max={arr_r.max():.4g}")

    # ---- 2. WT-WT replicate z distribution ----
    from run_p2_v1 import build_rep_groups, sanitize_records, build_pair_features_aligned
    sanitize_records(rec_index)
    rep_groups = build_rep_groups(rec_index, study_whitelist=pool)
    zz = []
    n_groups = 0
    n_rep_pairs = 0
    for g in rep_groups:
        k = g.n_replicates
        if k < 2:
            continue
        n_groups += 1
        mask = g.eligibility_mask
        length = min(len(p) for p in g.wt_profiles)
        elig = [i for i in range(min(length, len(mask))) if mask[i]]
        for a in range(k):
            for b in range(a + 1, k):
                n_rep_pairs += 1
                for i in elig:
                    wa, wb = g.wt_profiles[a][i], g.wt_profiles[b][i]
                    ea, eb = g.wt_errors[a][i], g.wt_errors[b][i]
                    if not (finite(wa) and finite(wb) and finite(ea) and finite(eb)):
                        continue
                    noise = math.sqrt(float(ea) ** 2 + float(eb) ** 2)
                    if not math.isfinite(noise) or noise <= 0:
                        continue
                    zz.append((float(wa) - float(wb)) / noise)
    arr_z = np.asarray(zz)
    print(f"\n[diag2] WT-WT replicate z (disagreement/error):")
    print(f"  groups={n_groups} replicate_pairs={n_rep_pairs} n_z={arr_z.size}")
    if arr_z.size:
        print(f"  |z|: min={np.abs(arr_z).min():.3f} median={np.median(np.abs(arr_z)):.3f} "
              f"mean={np.mean(np.abs(arr_z)):.3f} p99={np.percentile(np.abs(arr_z),99):.3f} "
              f"max={np.abs(arr_z).max():.3f}")

    # ---- 3. caller changers identity ----
    caller = CallerV2(seed=20260807).fit(rep_groups, [])
    print(f"\n[diag3] caller (fit on full pool): null_median={caller.null_median}")
    changers = []
    all_calls = Counter()
    for p in pairs:
        if p["source_accession"].split("_")[0] not in pool:
            continue
        wt = rec_index.get((p["source_accession"], p["wt_profile_index"], p["asset_name"]))
        mu = rec_index.get((p["source_accession"], p["mutant_profile_index"], p["asset_name"]))
        if wt is None or mu is None:
            continue
        pf = build_pair_features_aligned(p, wt, mu)
        res = caller.call(pf)
        all_calls[res.label] += 1
        if res.label == "1":
            changers.append((p, res, pf))
    print(f"  call distribution: {dict(all_calls)}")
    print(f"  n_changers={len(changers)}")
    for p, res, pf in changers:
        st = p["source_accession"].split("_")[0]
        print(f"  changer: study={st} src={p['source_accession']} mut_idx={p['mutant_profile_index']} "
              f"ref={p.get('ref_allele')} alt={p.get('alt_allele')} pos={p.get('position')} "
              f"stat={res.statistic:.2f} p={res.p_value:.4f}")

    # ---- 4. observed |delta_r| scale for all usable pairs ----
    print("\n[diag4] observed |delta_r| scale (eligible positions):")
    big_delta = 0
    n_pos = 0
    max_delta = 0.0
    for p in pairs:
        if p["source_accession"].split("_")[0] not in pool:
            continue
        wt = rec_index.get((p["source_accession"], p["wt_profile_index"], p["asset_name"]))
        mu = rec_index.get((p["source_accession"], p["mutant_profile_index"], p["asset_name"]))
        if wt is None or mu is None:
            continue
        pf = build_pair_features_aligned(p, wt, mu)
        for i in range(len(pf)):
            if not pf.eligibility_mask[i]:
                continue
            wtv, mutv = pf.wt_reactivity[i], pf.mutant_reactivity[i]
            if not (finite(wtv) and finite(mutv)):
                continue
            d = abs(mutv - wtv)
            max_delta = max(max_delta, d)
            n_pos += 1
            if d > 0.3:
                big_delta += 1
    print(f"  eligible positions={n_pos} max_abs_delta={max_delta:.4g} "
          f"frac_delta>0.3={big_delta/n_pos if n_pos else 0:.4f}")


if __name__ == "__main__":
    main()
