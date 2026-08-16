#!/usr/bin/env python3
"""DIAGNOSTIC (read-only): does per-study reactivity normalization + empirical-
scatter error recalibration recover a NON-DEGENERATE, putatively learnable P2
binary changer label?

This is a DECISION diagnostic for a potential endpoint_v3 / caller_v3 fix. It
does NOT modify any frozen/authority/contract/endpoint/split/caller/evaluate
file, does NOT train a model, and makes NO confirmatory claim. It only quantifies
whether the R5 label degeneracy (3 changers / 6359) is a caller/null calibration
artifact that a corrected caller would resolve.

Input: the frozen P2 cache built by build_cache_p2.py (rec_index + pairs + pool),
mirroring the R5 run input. Outputs a JSON summary + prints key numbers.
"""
from __future__ import annotations

import json
import math
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/cunyuliu/reactflow_delta_goal_20260729/scripts/reactflow_delta")

from caller_v2 import (  # noqa: E402  (pure-Python stdlib, no numpy/torch needed)
    ReplicateGroup, compute_eligible_mask, max_cluster_stat, spatial_block_null,
    _quantile, _p_value, icc_one_way, CLUSTER_WINDOW, N_NULL, NULL_BLOCK_LEN,
    RNG_SEED, ALPHA, ICC_THRESHOLD, MIN_REPLICATES, MIN_REPLICATE_GROUPS,
    PLUS_ONE_NULL,
)

CACHE = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/p2_cache/p2_cache.pkl"
SPLIT_YAML = "/home/cunyuliu/reactflow_delta_goal_20260729/configs/reactflow_delta/split_v2.yaml"
OUT = "/home/cunyuliu/reactflow_delta_goal_20260729/results/p2_v1_learnability_20260808/diag_p2_calibration_fix_summary.json"

# Nominal two-sided per-position significance threshold (documented; z-scaled)
POS_Z_ALPHA = 0.05
POS_Z_THRESH = 1.959964  # |N(0,1)| two-sided 5%


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def study_of(sa: str) -> str:
    return (sa or "").split("_")[0]


def get_tf(rec) -> dict:
    rl = rec.get("reactivity_layers", {}) or {}
    return rl.get("train_frozen", {}) or rl.get("raw", {}) or {}


def build_rep_groups(rec_index, whitelist):
    wt_by_key = defaultdict(list)
    for key, r in rec_index.items():
        if not r.get("is_wt"):
            continue
        st = study_of(r.get("source_accession") or "")
        if whitelist is not None and st not in whitelist:
            continue
        gk = (st, r.get("canonical_sequence"), tuple(r.get("probe") or []),
              tuple(r.get("temperature") or []))
        wt_by_key[gk].append(r)
    groups = []
    for gk, recs in wt_by_key.items():
        if len(recs) < 2:
            continue
        rl0 = recs[0].get("reactivity_layers", {})
        mask = compute_eligible_mask(rl0.get("eligibility_reason_codes") or [])
        profs, errs = [], []
        for r in recs:
            tf = get_tf(r)
            react = list(tf.get("reactivity") or [])
            err = list(tf.get("error") or [])
            L = min(len(react), len(err))
            profs.append(react[:L])
            errs.append(err[:L])
        groups.append(ReplicateGroup(group_key=gk, wt_profiles=profs,
                                     wt_errors=errs, eligibility_mask=mask,
                                     study=gk[0]))
    return groups


def build_pair_features(pair, wt_rec, mut_rec):
    def get(r, key):
        return list(get_tf(r).get(key) or [])
    wt_react, wt_err = get(wt_rec, "reactivity"), get(wt_rec, "error")
    mut_react, mut_err = get(mut_rec, "reactivity"), get(mut_rec, "error")
    mask = compute_eligible_mask(pair.get("eligibility_reason_codes") or [])
    L = min(len(mask), len(wt_react), len(wt_err), len(mut_react), len(mut_err))
    grp = (study_of(pair.get("source_accession") or ""),
           wt_rec.get("canonical_sequence") or "",
           tuple(wt_rec.get("probe") or []),
           tuple(wt_rec.get("temperature") or []))
    return dict(pair_id=f"{pair.get('source_accession')}:{pair.get('mutant_profile_index')}",
                wt_reactivity=wt_react[:L], mutant_reactivity=mut_react[:L],
                wt_error=wt_err[:L], mutant_error=mut_err[:L],
                eligibility_mask=mask[:L], group_key=grp,
                source_accession=pair.get("source_accession"),
                mutant_profile_index=pair.get("mutant_profile_index"))


def corrected_z(pf, sigma, fallback_fn):
    """z_i = (mut_i - wt_i) / (sqrt(2) * empirical_scatter_i).
    (Per-study normalization is a common linear scale and cancels in this ratio;
    empirical scatter is the noise term.)"""
    n = len(pf["eligibility_mask"])
    z = [None] * n
    for i in range(n):
        if not pf["eligibility_mask"][i]:
            continue
        wt, mut = pf["wt_reactivity"][i], pf["mutant_reactivity"][i]
        if not (finite(wt) and finite(mut)):
            continue
        s = None
        if sigma is not None and i < len(sigma) and finite(sigma[i]) and sigma[i] > 0:
            s = float(sigma[i])
        if s is None:
            s = fallback_fn(i)
        if not (finite(s) and s > 0):
            continue
        z[i] = (float(mut) - float(wt)) / (s * math.sqrt(2.0))
    return z, list(pf["eligibility_mask"])


def main() -> int:
    summary = {}
    with open(CACHE, "rb") as fh:
        cache = pickle.load(fh)
    rec_index, pairs, pool = cache["rec_index"], cache["pairs"], set(cache["pool"])
    summary["n_rec_index"] = len(rec_index)
    summary["n_pairs"] = len(pairs)
    summary["pool_studies"] = sorted(pool)

    # split -> publication map
    import yaml
    split = yaml.safe_load(Path(SPLIT_YAML).read_text(encoding="utf-8"))
    pub_map = split.get("publication_map", {})

    # ------------------------------------------------------------------
    # Part A. per-study reactivity scale (robust: median + MAD)
    # ------------------------------------------------------------------
    react_by_study = defaultdict(list)
    for r in rec_index.values():
        st = study_of(r.get("source_accession") or "")
        if st not in pool:
            continue
        tf = get_tf(r)
        for v in tf.get("reactivity") or []:
            if finite(v):
                react_by_study[st].append(float(v))
    study_med, study_mad = {}, {}
    for st in pool:
        a = np.asarray(react_by_study[st], dtype=float)
        if a.size == 0:
            study_med[st], study_mad[st] = np.nan, np.nan
            continue
        m = np.median(a)
        mad = np.median(np.abs(a - m))
        study_med[st], study_mad[st] = float(m), float(mad)

    meds = [study_med[s] for s in pool if finite(study_med[s])]
    mads = [study_mad[s] for s in pool if finite(study_mad[s])]
    scale_spread = {
        "median_range": [float(min(meds)), float(max(meds))],
        "median_spread_ratio": float(max(meds) / min(meds)) if min(meds) != 0 else None,
        "mad_range": [float(min(mads)), float(max(mads))],
        "mad_spread_ratio": float(max(mads) / min(mads)) if min(mads) != 0 else None,
        "n_studies": len(pool),
    }
    summary["A_per_study_scale"] = {
        "per_study_median": {s: (round(study_med[s], 5) if finite(study_med[s]) else None) for s in pool},
        "per_study_mad": {s: (round(study_mad[s], 5) if finite(study_mad[s]) else None) for s in pool},
        "spread_before_norm": scale_spread,
    }
    # after per-study (r-med)/MAD normalization -> all medians 0, MADs 1
    norm_meds = [0.0] * len(pool)
    norm_mads = [1.0] * len(pool)
    summary["A_per_study_scale"]["spread_after_norm"] = {
        "median_range": [min(norm_meds), max(norm_meds)],
        "mad_range": [min(norm_mads), max(norm_mads)],
        "note": "per-study (r-median)/MAD maps every study to median 0, MAD 1 by construction; "
                "spread ratio -> 1.0",
    }

    # ------------------------------------------------------------------
    # replicate groups + empirical per-position scatter
    # ------------------------------------------------------------------
    rep_groups = build_rep_groups(rec_index, whitelist=pool)
    summary["n_replicate_groups_ge2"] = len(rep_groups)

    sigma_by_group = {}
    rep_err_by_group = {}
    for g in rep_groups:
        L = min(len(p) for p in g.wt_profiles)
        arr = np.full((g.n_replicates, L), np.nan, dtype=float)
        for r in range(g.n_replicates):
            for i in range(L):
                v = g.wt_profiles[r][i]
                if finite(v):
                    arr[r, i] = v
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.nanstd(arr, axis=0, ddof=1)
        # reported error per position: RMS across replicates
        errarr = np.full((g.n_replicates, L), np.nan, dtype=float)
        for r in range(g.n_replicates):
            for i in range(L):
                v = g.wt_errors[r][i]
                if finite(v):
                    errarr[r, i] = v
        with np.errstate(invalid="ignore", divide="ignore"):
            rms_err = np.sqrt(np.nanmean(np.square(errarr), axis=0))
        sigma_by_group[g.group_key] = sd
        rep_err_by_group[g.group_key] = rms_err

    # ------------------------------------------------------------------
    # Part B. reported-error vs empirical scatter
    # ------------------------------------------------------------------
    ratios = []
    ratio_positions = 0
    for g in rep_groups:
        sd = sigma_by_group[g.group_key]
        re = rep_err_by_group[g.group_key]
        mask = g.eligibility_mask
        L = min(len(sd), len(re), len(mask))
        for i in range(L):
            if not mask[i]:
                continue
            s, e = sd[i], re[i]
            if finite(s) and s > 0 and finite(e) and e > 0:
                ratios.append(s / e)
                ratio_positions += 1
    rarr = np.asarray(ratios)
    summary["B_error_recalibration"] = {
        "n_positions_ratio": ratio_positions,
        "empirical_sd_over_reported_err": {
            "median": float(np.median(rarr)) if rarr.size else None,
            "mean": float(np.mean(rarr)) if rarr.size else None,
            "p90": float(np.percentile(rarr, 90)) if rarr.size else None,
            "frac_gt_1": float(np.mean(rarr > 1.0)) if rarr.size else None,
            "frac_gt_5": float(np.mean(rarr > 5.0)) if rarr.size else None,
            "max": float(np.max(rarr)) if rarr.size else None,
        },
    }

    # WT-WT null z calibration: reported-error vs empirical-scatter
    def wt_wt_absz(use_empirical):
        zz = []
        for g in rep_groups:
            if g.n_replicates < 2:
                continue
            sd = sigma_by_group[g.group_key]
            re = rep_err_by_group[g.group_key]
            mask = g.eligibility_mask
            L = min(len(g.wt_profiles[0]), len(sd), len(re), len(mask))
            for a in range(g.n_replicates):
                for b in range(a + 1, g.n_replicates):
                    for i in range(L):
                        if not mask[i]:
                            continue
                        wa, wb = g.wt_profiles[a][i], g.wt_profiles[b][i]
                        if not (finite(wa) and finite(wb)):
                            continue
                        if use_empirical:
                            s = sd[i]
                            if not (finite(s) and s > 0):
                                continue
                            z = (wa - wb) / (s * math.sqrt(2.0))
                        else:
                            ea, eb = g.wt_errors[a][i], g.wt_errors[b][i]
                            if not (finite(ea) and finite(eb)) or (ea <= 0 or eb <= 0):
                                continue
                            z = (wa - wb) / math.sqrt(ea * ea + eb * eb)
                        zz.append(abs(z))
        return np.asarray(zz)

    zz_rep = wt_wt_absz(False)
    zz_emp = wt_wt_absz(True)
    summary["B_error_recalibration"]["wt_wt_abs_z_reported_err"] = {
        "median": float(np.median(zz_rep)) if zz_rep.size else None,
        "mean": float(np.mean(zz_rep)) if zz_rep.size else None,
        "p99": float(np.percentile(zz_rep, 99)) if zz_rep.size else None,
        "max": float(np.max(zz_rep)) if zz_rep.size else None,
    }
    summary["B_error_recalibration"]["wt_wt_abs_z_empirical_scatter"] = {
        "median": float(np.median(zz_emp)) if zz_emp.size else None,
        "mean": float(np.mean(zz_emp)) if zz_emp.size else None,
        "p99": float(np.percentile(zz_emp, 99)) if zz_emp.size else None,
        "max": float(np.max(zz_emp)) if zz_emp.size else None,
        "note": "if errors were well-calibrated, WT-WT |z| ~ half-normal(0,1): median~0.67, p99~2.58",
    }

    # ------------------------------------------------------------------
    # Part C. corrected caller labels (normalization + empirical-scatter noise)
    # ------------------------------------------------------------------
    # corrected null profiles from WT-WT replicate diffs / (sqrt2 * empirical scatter)
    cnull = []
    for g in rep_groups:
        if g.n_replicates < 2:
            continue
        sd = sigma_by_group[g.group_key]
        mask = g.eligibility_mask
        L = min(len(g.wt_profiles[0]), len(sd), len(mask))
        for a in range(g.n_replicates):
            for b in range(a + 1, g.n_replicates):
                prof = []
                for i in range(L):
                    if not mask[i]:
                        continue
                    wa, wb = g.wt_profiles[a][i], g.wt_profiles[b][i]
                    s = sd[i]
                    if not (finite(wa) and finite(wb)) or not (finite(s) and s > 0):
                        continue
                    prof.append((wa - wb) / (s * math.sqrt(2.0)))
                if len(prof) >= 2:
                    cnull.append(prof)
    cnull.sort(key=len)
    mask_any = [1] * (max((len(p) for p in cnull), default=0))
    corrected_null = spatial_block_null(cnull, mask_any, n_null=N_NULL,
                                        block_len=NULL_BLOCK_LEN, seed=RNG_SEED)
    corrected_null.sort()
    null_q95 = _quantile(corrected_null, 1 - ALPHA)
    null_med = _quantile(corrected_null, 0.5)
    summary["C_corrected_null"] = {
        "n_null_profiles": len(cnull),
        "null_median": null_med,
        "null_q95": null_q95,
        "note": "frozen caller_v2 pooled null median was ~44; a well-calibrated null has q95 ~ few units",
    }

    # per-group ICC reliability (same gate as frozen caller)
    rel_map, rel_vals = {}, []
    for g in rep_groups:
        if g.n_replicates < MIN_REPLICATES:
            rel_map[g.group_key] = None
            continue
        icc = icc_one_way(g.wt_profiles, g.eligibility_mask)
        rel_map[g.group_key] = icc
        if icc is not None:
            rel_vals.append(icc)
    global_rel = float(np.mean(rel_vals)) if rel_vals else None
    structure_ok = len([g for g in rep_groups if g.n_replicates >= MIN_REPLICATES]) >= MIN_REPLICATE_GROUPS

    # build pair features
    pf_list = []
    for p in pairs:
        if study_of(p.get("source_accession") or "") not in pool:
            continue
        wt = rec_index.get((p.get("source_accession"), p.get("wt_profile_index"), p.get("asset_name")))
        mu = rec_index.get((p.get("source_accession"), p.get("mutant_profile_index"), p.get("asset_name")))
        if wt is None or mu is None:
            continue
        pf_list.append((p, build_pair_features(p, wt, mu)))
    summary["n_pairs_usable"] = len(pf_list)

    def label_pairs(use_icc_gate):
        labels = []
        stats = []
        for p, pf in pf_list:
            gk = pf["group_key"]
            sd = sigma_by_group.get(gk)
            # fallback noise for positions lacking empirical scatter
            if sd is not None:
                med_sigma = float(np.nanmedian(np.where(np.isfinite(sd) & (sd > 0), sd, np.nan))) if np.any(np.isfinite(sd) & (sd > 0)) else None
            else:
                med_sigma = None

            def fallback(i):
                if med_sigma is not None:
                    return med_sigma
                # fall back to reported error RMS of this pair's WT
                we, me = pf["wt_error"][i], pf["mutant_error"][i]
                if finite(we) and finite(me) and (we > 0 or me > 0):
                    return math.sqrt(we * we + me * me)
                return None

            z, emask = corrected_z(pf, sd, fallback)
            if not any(emask):
                labels.append("NO_CALL")
                stats.append(None)
                continue
            if use_icc_gate:
                rel = rel_map.get(gk, global_rel)
                if not structure_ok or rel is None or rel < ICC_THRESHOLD:
                    labels.append("NO_CALL")
                    stats.append(None)
                    continue
            stat = max_cluster_stat(z, emask)
            stats.append(stat)
            pv = _p_value(corrected_null, stat, plus_one=PLUS_ONE_NULL)
            labels.append("1" if pv <= ALPHA else "0")
        return labels, stats

    for gate_name, gate in (("with_icc_gate", True), ("without_icc_gate", False)):
        labels, stats = label_pairs(gate)
        cnt = Counter(labels)
        n = len(labels)
        # publication fraction
        pub_changer = set()
        pub_all = set()
        for (p, pf), lab in zip(pf_list, labels):
            st = study_of(p.get("source_accession") or "")
            pub = pub_map.get(st, f"UNKNOWN_PUBLICATION:{st}")
            pub_all.add(pub)
            if lab == "1":
                pub_changer.add(pub)
        summary[f"C_corrected_labels_{gate_name}"] = {
            "n_units": n,
            "changer": cnt.get("1", 0),
            "nonchanger": cnt.get("0", 0),
            "no_call": cnt.get("NO_CALL", 0),
            "changer_fraction_of_called": (cnt.get("1", 0) / (n - cnt.get("NO_CALL", 0)))
                                          if (n - cnt.get("NO_CALL", 0)) else None,
            "n_publications_total": len(pub_all),
            "n_publications_with_changer": len(pub_changer),
            "frac_publications_with_changer": len(pub_changer) / len(pub_all) if pub_all else None,
        }

    # ------------------------------------------------------------------
    # Part D. learnability signal after correction
    # ------------------------------------------------------------------
    # fraction of pairs with >=1 eligible position |z|>POS_Z_THRESH
    pos_pairs_signif = 0
    pos_pairs_total = 0
    # corrected statistic distribution across changers vs nonchangers (with ICC gate)
    labels_g, stats_g = label_pairs(True)
    ch_stats = [s for s, l in zip(stats_g, labels_g) if s is not None and l == "1"]
    nc_stats = [s for s, l in zip(stats_g, labels_g) if s is not None and l == "0"]

    pair_z_signif = []
    for p, pf in pf_list:
        gk = pf["group_key"]
        sd = sigma_by_group.get(gk)
        z, emask = corrected_z(pf, sd, lambda i: None)
        if not any(emask):
            continue
        pos_pairs_total += 1
        if any(zi is not None and abs(zi) > POS_Z_THRESH for zi in z):
            pos_pairs_signif += 1

    summary["D_learnability_signal"] = {
        "pairs_with_any_eligible_pos": pos_pairs_total,
        "pairs_with_ge1_eligible_signif_pos": pos_pairs_signif,
        "frac_pairs_with_ge1_signif_pos": (pos_pairs_signif / pos_pairs_total) if pos_pairs_total else None,
        "corrected_statistic_changers": {
            "n": len(ch_stats),
            "median": float(np.median(ch_stats)) if ch_stats else None,
            "mean": float(np.mean(ch_stats)) if ch_stats else None,
            "q25": float(np.percentile(ch_stats, 25)) if ch_stats else None,
            "q75": float(np.percentile(ch_stats, 75)) if ch_stats else None,
        },
        "corrected_statistic_nonchangers": {
            "n": len(nc_stats),
            "median": float(np.median(nc_stats)) if nc_stats else None,
            "mean": float(np.mean(nc_stats)) if nc_stats else None,
            "q25": float(np.percentile(nc_stats, 25)) if nc_stats else None,
            "q75": float(np.percentile(nc_stats, 75)) if nc_stats else None,
        },
        "null_q95_reference": null_q95,
        "note": "separability = changers should sit clearly above the corrected-null q95 and above nonchanger statistic",
    }

    summary["global_reliability"] = global_rel

    print(json.dumps(summary, indent=2, default=str))
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n[written] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
