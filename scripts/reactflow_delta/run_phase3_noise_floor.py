#!/usr/bin/env python3
"""Phase 3 closure -> benchmark/resource route (deliverable 2).

Contract §12 Phase 3 failure handling (all three architecture schemes FAIL-CLOSED)
directs "使用最简单generic并转benchmark/resource" and §9.2/9.3 "分析 caller reliability
和 domain shift". Deliverable 1 (run_phase3_benchmark_resource.py) characterized caller
coverage / NO_CALL, publication label shift, and feature shift.

This deliverable answers the most decisive question for a magnitude-regression negative
result: is the conditional-magnitude TARGET (|mut reactivity - WT reactivity| over
eligible positions, the same magnitude as `pair_magnitude`) above the REPLICATE
MEASUREMENT NOISE FLOOR?

If a large fraction of (pair, position) mutation effects are below the within-WT
replicate std, then the effect signal sits at/below the measurement noise floor, and
no architecture can predict magnitude better than trivial/generic -- which is exactly
what Phase 3 observed (all three schemes CI lower bound <= 0).

CPU-only statistical diagnostic; no model training, no CUDA required.
"""
from __future__ import annotations

import argparse, json, pickle, sys, math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_p2_v3 import build_rep_groups, build_pair_features_aligned  # noqa: E402


def _study_of(sa: str) -> str:
    return (sa or "").split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split-yaml", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import yaml
    split = yaml.safe_load(Path(args.split_yaml).read_text(encoding="utf-8"))
    pub_map = split["publication_map"]
    study_roles = split["study_roles"]
    test_studies = {s for s, r in study_roles.items() if r == "test"}

    with open(args.cache, "rb") as fh:
        cache = pickle.load(fh)
    rec_index = cache["rec_index"]
    pairs = cache["pairs"]
    pool_studies = set(cache["pool"]) - test_studies

    pair_recs = {}
    for p in pairs:
        st = _study_of(p["source_accession"])
        if st in test_studies:
            continue
        wt = rec_index.get((p["source_accession"], p["wt_profile_index"], p["asset_name"]))
        mu = rec_index.get((p["source_accession"], p["mutant_profile_index"], p["asset_name"]))
        if wt is None or mu is None:
            continue
        pair_recs[p["source_accession"] + ":" + str(p["mutant_profile_index"])] = {
            "pair": p, "wt": wt, "mut": mu, "study": st,
            "pub": pub_map.get(st, "UNKNOWN:" + st),
        }

    # Per-position replicate noise floor from within-WT replicate groups (pool, train).
    groups = build_rep_groups(rec_index, study_whitelist=pool_studies)
    noise_by_key = {}   # group_key -> per-position std across replicates
    n_reps_by_key = {}
    for g in groups:
        profs = [np.asarray(p, dtype=np.float64) for p in g.wt_profiles if p]
        if len(profs) < 2:
            continue
        L = min(len(p) for p in profs)
        A = np.stack([p[:L] for p in profs])          # (n_rep, L)
        finite = np.isfinite(A)
        with np.errstate(all="ignore"):
            sd = np.where(finite.sum(0) >= 2, A.std(0), np.nan)
        noise_by_key[g.group_key] = sd
        n_reps_by_key[g.group_key] = len(profs)

    # For each pair, compare per-eligible-position |delta| to the WT group noise.
    ratio_all = []        # (pair,pos) ratio = |delta| / noise
    n_below_1 = 0
    n_below_196 = 0
    n_with_noise = 0
    pair_level_ratio = []  # per-pair mean(|delta|)/mean(noise) over eligible
    pair_stats = {}
    n_pairs_no_noise = 0

    for pid, pr in pair_recs.items():
        pf = build_pair_features_aligned(pr["pair"], pr["wt"], pr["mut"])
        st = pr["study"]
        gk = (st, pr["wt"].get("canonical_sequence"),
              tuple(pr["wt"].get("probe") or []), tuple(pr["wt"].get("temperature") or []))
        noise = noise_by_key.get(gk)
        mask = pf.eligibility_mask
        L = min(len(pf.wt_reactivity), len(pf.mutant_reactivity), len(mask))
        deltas = []
        noises = []
        for i in range(L):
            if not mask[i]:
                continue
            a, b = pf.wt_reactivity[i], pf.mutant_reactivity[i]
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            d = abs(b - a)
            deltas.append(d)
            if noise is None or i >= len(noise) or not (math.isfinite(noise[i])) or noise[i] <= 0:
                continue
            n_i = float(noise[i])
            noises.append(n_i)
            ratio = d / n_i
            ratio_all.append(ratio)
            n_with_noise += 1
            if ratio < 1.0:
                n_below_1 += 1
            if ratio < 1.96:
                n_below_196 += 1
        if noise is None:
            n_pairs_no_noise += 1
        if deltas and noises:
            pair_level_ratio.append(sum(deltas) / len(deltas) / (sum(noises) / len(noises)))

    # rebuild per-pub tally cleanly (independent of the aggregate pass above)
    pair_stats = {}
    for pid, pr in pair_recs.items():
        pf = build_pair_features_aligned(pr["pair"], pr["wt"], pr["mut"])
        st = pr["study"]
        gk = (st, pr["wt"].get("canonical_sequence"),
              tuple(pr["wt"].get("probe") or []), tuple(pr["wt"].get("temperature") or []))
        noise = noise_by_key.get(gk)
        mask = pf.eligibility_mask
        L = min(len(pf.wt_reactivity), len(pf.mutant_reactivity), len(mask))
        s = pair_stats.setdefault(pr["pub"], {"n": 0, "with_noise": 0, "pos_total": 0,
                                              "pos_below1": 0, "pair_below1": 0, "pair_with_noise": 0})
        s["n"] += 1
        if noise is not None:
            s["with_noise"] += 1
            below = 0
            cnt = 0
            for i in range(L):
                if not mask[i]:
                    continue
                a, b = pf.wt_reactivity[i], pf.mutant_reactivity[i]
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                d = abs(b - a)
                if i < len(noise) and math.isfinite(noise[i]) and noise[i] > 0:
                    cnt += 1
                    s["pos_total"] += 1
                    if d / noise[i] < 1.0:
                        s["pos_below1"] += 1
                        below += 1
            if cnt:
                s["pair_with_noise"] += 1
                if below == cnt:
                    s["pair_below1"] += 1

    n_pairs = len(pair_recs)
    ratio_all = np.asarray(ratio_all, dtype=np.float64)
    pair_level_ratio = np.asarray(pair_level_ratio, dtype=np.float64)

    def _q(a):
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return None
        # Heavy right tail from (near-)zero-variance replicate positions -> robust
        # percentiles are the defensible summary; raw/clipped means are meaningless.
        return {"n": int(len(a)),
                "median": float(np.median(a)),
                "p25": float(np.percentile(a, 25)),
                "p75": float(np.percentile(a, 75)),
                "p90": float(np.percentile(a, 90)),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99))}

    report = {
        "schema": "reactflow_delta.phase3.benchmark_resource.noise_floor.v1",
        "run_id": Path(args.out_dir).name,
        "authority_epoch": 18,
        "phase": "PHASE3-BENCHMARK-RESOURCE",
        "question": ("Is the conditional-magnitude target |mut-WT| over eligible positions "
                     "above the within-WT replicate noise floor?"),
        "noise_floor": {
            "definition": "per-position std across >=2 within-WT replicates (train_frozen reactivity)",
            "n_replicate_groups_with_noise": len(noise_by_key),
            "n_pairs": n_pairs,
            "n_pairs_without_noise_est": n_pairs_no_noise,
            "fraction_pairs_without_noise_est": n_pairs_no_noise / max(n_pairs, 1),
            "pair_position_ratio_distribution": _q(ratio_all) if len(ratio_all) else None,
            "fraction_pair_position_below_1x_noise": (n_below_1 / max(n_with_noise, 1)) if n_with_noise else None,
            "fraction_pair_position_below_1.96x_noise": (n_below_196 / max(n_with_noise, 1)) if n_with_noise else None,
            "n_pair_position_with_noise": n_with_noise,
            "pair_mean_ratio_distribution": _q(pair_level_ratio) if len(pair_level_ratio) else None,
        },
        "per_publication": {
            k: {
                "n": v["n"], "with_noise": v["with_noise"],
                "pos_total": v["pos_total"], "pos_below_1x": v["pos_below1"],
                "fraction_pos_below_1x": (v["pos_below1"] / v["pos_total"]) if v["pos_total"] else None,
                "pair_with_noise": v["pair_with_noise"],
                "pair_entirely_below_1x": v["pair_below1"],
            }
            for k, v in sorted(pair_stats.items())
        },
        "notes": (
            "Target magnitude is identical to `pair_magnitude` (mean |mut-wt| over eligible "
            "positions). If a large share of effects are below the within-WT replicate std, "
            "the magnitude signal is at/below measurement noise and no model can produce a "
            "stable within-publication increment over trivial/generic, consistent with all "
            "three Phase 3 schemes failing the CI-lower-bound>0 gate."),
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "noise_floor_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[noise_floor] wrote -> {out/'noise_floor_report.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
