#!/usr/bin/env python3
"""caller_stability_diagnosis_v1 — read-only quantification of the STRICT vs
TRANSDUCTIVE label instability driver (Phase 2 learnability gate blocker).

Hypothesis (from reading caller_v4): in STRICT mode every held pair (whose
publication is absent from the train-only sigma map) falls back to a single
train-global median sigma constant, whereas TRANSDUCTIVE uses each held group's
own per-position empirical scatter. Because z_i = (mut-wt)/(sqrt(2)*sigma_i),
the sigma-collapse changes z and flips labels. This diagnostic quantifies how
much the train-global median deviates from held per-position sigmas per
publication, WITHOUT changing any frozen endpoint/label. Read-only; no training;
no confirmatory outcome.
"""
from __future__ import annotations

import argparse, json, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caller_v3 import _empirical_scatter, _med_positive
from caller_v4 import CallerV4, MODE_STRICT
from run_baselines_v6 import (
    load_cache, load_publication_map, build_pair_recs, build_rep_groups,
    rep_groups_for_train, build_pair_features_aligned_robust,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    cache = load_cache(args.cache)
    pub_map = load_publication_map(args.registry)
    pair_recs, _ = build_pair_recs(cache, pub_map)
    all_rep_groups = build_rep_groups(cache["rec_index"])
    pf_all = {pid: build_pair_features_aligned_robust(v["pair"], v["wt"], v["mut"])
              for pid, v in pair_recs.items()}

    resolved = sorted({v["pub"] for v in pair_recs.values() if (v["pub"] or "").startswith("pmid_")})
    pub_study = defaultdict(set)
    for v in pair_recs.values():
        pub_study[v["pub"]].add(v["study"])

    report = {"schema": "reactflow_delta.caller_stability_diagnosis.v1",
              "authority_epoch": 20, "endpoint": "endpoint_v6",
              "n_resolved_publications": len(resolved), "per_publication": {}}

    all_ratios = []
    all_frac_outside = []
    for held_pub in resolved:
        train_studies = set()
        for p_ in resolved:
            if p_ != held_pub:
                train_studies |= pub_study[p_]
        train_groups = rep_groups_for_train(all_rep_groups, train_studies)
        caller = CallerV4(mode=MODE_STRICT, seed=args.seed).fit(train_groups, [])
        train_med = caller._train_median_sigma

        held_groups = [g for g in all_rep_groups if g.study in pub_study[held_pub]]
        # all eligible held pairs of this pub that use the train-med fallback
        n_affected = 0
        ratios = []
        frac_outside = []
        for g in held_groups:
            if g.n_replicates < 2:
                continue
            sig = _empirical_scatter(g)          # held per-position sigma (TRANSDUCTIVE)
            sig_pos = sig[np.isfinite(sig) & (sig > 0)]
            gmed = _med_positive(sig)
            if gmed is None or train_med is None:
                continue
            n_affected += len(sig_pos)
            ratios.append(train_med / gmed)
            # fraction of held positions where the train-med constant is outside
            # a 2x band of the held per-position sigma
            if len(sig_pos):
                frac_outside.append(float(np.mean((sig_pos < train_med / 2.0) |
                                                  (sig_pos > train_med * 2.0))))
        n_held_pairs = sum(1 for pid, v in pair_recs.items() if v["pub"] == held_pub)
        row = {
            "n_held_pairs": n_held_pairs,
            "n_held_groups_sigma": n_affected,
            "train_median_sigma": train_med,
            "median_ratio_trainmed_to_held": float(np.median(ratios)) if ratios else None,
            "ratio_p10": float(np.percentile(ratios, 10)) if ratios else None,
            "ratio_p90": float(np.percentile(ratios, 90)) if ratios else None,
            "frac_held_positions_outside_2x_band": (
                float(np.mean(frac_outside)) if frac_outside else None),
        }
        report["per_publication"][held_pub] = row
        all_ratios += ratios
        all_frac_outside += frac_outside

    report["summary"] = {
        "median_ratio_trainmed_to_held": float(np.median(all_ratios)) if all_ratios else None,
        "ratio_iqr": ((float(np.percentile(all_ratios, 25)), float(np.percentile(all_ratios, 75)))
                      if all_ratios else None),
        "frac_held_positions_outside_2x_band_median": (
            float(np.median(all_frac_outside)) if all_frac_outside else None),
        "interpretation": "ratio>>1 => STRICT train-median sigma is much larger than held "
                          "per-position sigma (underestimates z, may flip labels); ratio<<1 "
                          "=> much smaller (overestimates z). Band-outside fraction ~ 1 "
                          "means the train-med constant is unrepresentative of held groups.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE -> {out}")
    # console summary
    print("held_pub  train_med  held_groups  med_ratio  p10  p90  frac_outside_2x")
    for pub, r in report["per_publication"].items():
        print(f"{pub:12s} {r['train_median_sigma']:.4f} {r['n_held_groups_sigma']:6d} "
              f"{_f(r['median_ratio_trainmed_to_held'])} {_f(r['ratio_p10'])} {_f(r['ratio_p90'])} "
              f"{_f(r['frac_held_positions_outside_2x_band'])}")


def _f(x):
    return f"{x:.3f}" if x is not None else "   -"


if __name__ == "__main__":
    sys.exit(main())