#!/usr/bin/env python3
"""run_learnability_gate_v1 — Phase 2 Batch 2B learnability analysis + gate.

Reads the keyed predictions produced by run_baselines_v6 and reports, for BOTH
tasks independently (endpoint_v6 independent-judgment policy):

  * pooled and publication-macro metrics
  * per-publication effects
  * dominant-publication LOO (exclude the largest publication)
  * unique null space / minimum attainable p
  * effect size + paired publication-block bootstrap CI (not just p)
  * publication-block permutation p
  * coverage (called fraction) and calibration (primary task)

Plus supplementary analysis that re-loads features:
  * caller-mode sensitivity (STRICT vs TRANSDUCTIVE label agreement each fold)
  * learning curves at train-publication fractions for primary generic models

No confirmatory outcome is read.  Development-only.  Fail-closed.
"""
from __future__ import annotations

import argparse, json, math, os, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v2 import (
    publication_macro_auprc, permutation_test, bootstrap_publication_ci,
    is_unidentifiable, UNIDENTIFIABLE,
)
from evaluate_v5 import (
    conditional_wmae_skill, paired_bootstrap_skill_ci, permutation_test_skill,
)


def _load_rows(pred_path):
    rows = []
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _binary_ap(labels, scores):
    from sklearn.metrics import average_precision_score
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(set(labels.tolist())) <= 1:
        return None
    return float(average_precision_score(labels, scores))


def per_publication_ap(rows, model, seed):
    """per-publication AP for primary task."""
    by_pub = defaultdict(lambda: {"y": [], "s": []})
    for r in rows:
        if r["task"] != "primary" or r["model_variant"] != model or r["seed"] != seed:
            continue
        if r["coverage_status"] != "CALLED":
            continue
        by_pub[r["fold_id"]]["y"].append(int(r["y"]))
        by_pub[r["fold_id"]]["s"].append(float(r["raw_prediction"]))
    out = {}
    for pub, d in by_pub.items():
        ap = _binary_ap(d["y"], d["s"])
        out[pub] = ap
    return out


def primary_analysis(rows, model, seed, exclude_pub=None):
    rsel = rows
    if exclude_pub is not None:
        rsel = [r for r in rows if r["fold_id"] != exclude_pub]
    map_ = per_publication_ap(rsel, model, seed)
    # prevalence trivial per-pub AP
    prev_ap = per_publication_ap(rsel, "prevalence", 0)
    pubs = sorted(map_.keys())
    # publication-macro AUPRC (model)
    calls = [(r["fold_id"], int(r["y"]), float(r["raw_prediction"]))
             for r in rsel if r["task"] == "primary" and r["model_variant"] == model
             and r["seed"] == seed and r["coverage_status"] == "CALLED"]
    macro = publication_macro_auprc([c[0] for c in calls], [c[1] for c in calls], [c[2] for c in calls])
    macro = None if is_unidentifiable(macro) else float(macro)
    # per-pub delta vs prevalence
    deltas = []
    for pub in pubs:
        if map_.get(pub) is None or prev_ap.get(pub) is None:
            continue
        deltas.append(map_[pub] - prev_ap[pub])
    ci = bootstrap_publication_ci(deltas, seed=seed, n_boot=1000) if len(deltas) >= 3 else UNIDENTIFIABLE
    # permutation (publication-block)
    perm = permutation_test([c[0] for c in calls], [c[1] for c in calls], [c[2] for c in calls],
                            seed=seed, n_perm=1000)
    return {
        "model": model, "seed": seed,
        "n_publications": len(pubs),
        "macro_auprc_model": macro,
        "per_pub_ap": map_,
        "per_pub_ap_prevalence": prev_ap,
        "mean_delta_ap": float(np.mean(deltas)) if deltas else None,
        "ci": ci if not is_unidentifiable(ci) else None,
        "permutation_p": perm["p_value"],
        "n_pairs": len(calls),
    }


def magnitude_analysis(rows, model, seed, exclude_pub=None):
    rsel = rows
    if exclude_pub is not None:
        rsel = [r for r in rows if r["fold_id"] != exclude_pub]
    pubs = [r["fold_id"] for r in rsel if r["task"] == "magnitude"
            and r["model_variant"] == model and r["seed"] == seed and r["coverage_status"] == "CALLED"]
    y = [r["y"] for r in rsel if r["task"] == "magnitude" and r["model_variant"] == model
         and r["seed"] == seed and r["coverage_status"] == "CALLED"]
    w = [r["weight"] for r in rsel if r["task"] == "magnitude" and r["model_variant"] == model
         and r["seed"] == seed and r["coverage_status"] == "CALLED"]
    pred = [r["raw_prediction"] for r in rsel if r["task"] == "magnitude"
            and r["model_variant"] == model and r["seed"] == seed and r["coverage_status"] == "CALLED"]
    # trivial = train-changer weighted median (constant per fold). Reconstruct from wmedian rows.
    triv_pred = {}
    for r in rsel:
        if r["task"] == "magnitude" and r["model_variant"] == "wmedian" and r["seed"] == 0 \
                and r["coverage_status"] == "CALLED":
            triv_pred[r["fold_id"]] = float(r["raw_prediction"])
    pred_triv = [triv_pred[p] for p in pubs]
    skill = conditional_wmae_skill(pubs, y, w, pred, pred_triv)
    ci = paired_bootstrap_skill_ci(pubs, y, w, pred, pred_triv, seed=seed, n_boot=1000)
    perm = permutation_test_skill(pubs, y, w, pred, pred_triv, seed=seed, n_perm=1000)
    return {
        "model": model, "seed": seed,
        "n_publications": len(set(pubs)),
        "n_changers": len(pubs),
        "skill": skill.get("skill"),
        "wmae_model": skill.get("wmae_model"),
        "wmae_baseline": skill.get("wmae_baseline"),
        "ci_low": ci.get("ci_low"), "ci_high": ci.get("ci_high"),
        "permutation_p": perm.get("p_value"),
    }


def load_cache_and_features(cache_path, registry_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_baselines_v6 import (load_cache, load_publication_map, build_pair_recs,
                                  build_rep_groups, rep_groups_for_train,
                                  build_pair_features_aligned_robust,
                                  build_feature, pair_magnitude)
    from caller_v4 import CallerV4, MODE_STRICT, MODE_TRANSDUCTIVE
    cache = load_cache(cache_path)
    pub_map = load_publication_map(registry_path)
    pair_recs, _ = build_pair_recs(cache, pub_map)
    all_rep_groups = build_rep_groups(cache["rec_index"])
    fx_full = {pid: build_feature(v["pair"], v["wt"], True, True, True)
               for pid, v in pair_recs.items()}
    pf_all = {pid: build_pair_features_aligned_robust(v["pair"], v["wt"], v["mut"])
              for pid, v in pair_recs.items()}
    return cache, pair_recs, all_rep_groups, fx_full, pf_all, CallerV4


def caller_mode_sensitivity(cache_path, registry_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_baselines_v6 import rep_groups_for_train
    from caller_v4 import MODE_STRICT, MODE_TRANSDUCTIVE
    cache, pair_recs, all_rep_groups, fx_full, pf_all, CallerV4 = load_cache_and_features(
        cache_path, registry_path)
    resolved = sorted({v["pub"] for v in pair_recs.values() if (v["pub"] or "").startswith("pmid_")})
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])
    caller_seed = 20260809
    strict_labels = {}
    nf = 0
    for held_pub in resolved:
        train_studies = set()
        for p_ in resolved:
            if p_ != held_pub:
                train_studies |= pub_study[p_]
        train_groups = rep_groups_for_train(all_rep_groups, train_studies)
        sc = CallerV4(mode=MODE_STRICT, seed=caller_seed).fit(train_groups, [])
        held_groups = [g for g in all_rep_groups if g.study in pub_study[held_pub]]
        tc = CallerV4(mode=MODE_TRANSDUCTIVE, seed=caller_seed).fit(train_groups, [])
        tc.add_held_wt_replicates(held_groups)
        for pid, v in pair_recs.items():
            if v["pub"] != held_pub:
                continue
            sl = sc.call(pf_all[pid]).label
            tl = tc.call(pf_all[pid]).label
            strict_labels.setdefault(held_pub, []).append((pid, sl, tl))
    # aggregate label agreement
    n = agree = 0
    per_pub_flip = {}
    per_pub_call = {}
    for pub, lst in strict_labels.items():
        for pid, sl, tl in lst:
            n += 1
            per_pub_call.setdefault(pub, []).append(1 if sl != "NO_CALL" else 0)
            if sl == tl:
                agree += 1
            else:
                per_pub_flip.setdefault(pub, 0)
                per_pub_flip[pub] = per_pub_flip.get(pub, 0) + 1
    return {
        "overall_label_agreement": agree / n if n else None,
        "overall_label_flip": 1 - (agree / n) if n else None,
        "overall_callable": sum(sum(v) for v in per_pub_call.values()) / sum(len(v) for v in per_pub_call.values()),
        "per_pub_flip": {p: round(c / len(strict_labels[p]), 4) for p, c in per_pub_flip.items()},
        "n_publications": len(strict_labels),
        "n_pairs": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dominant-pub", default="pmid_29446752")
    args = ap.parse_args()

    rows = _load_rows(args.pred)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    primary_models = ["p2_mlp", "deepsets", "gbm", "wlogit", "gam"]
    magn_models = ["wmae_mlp", "wmae_deepsets", "wmse_gbm", "lad_lm", "wgam"]
    seeds = [0, 1, 2, 3, 4]

    report = {
        "schema": "reactflow_delta.learnability_gate.v1",
        "authority_epoch": 20, "endpoint": "endpoint_v6", "caller_version": "caller_v4",
        "n_rows": len(rows),
        "primary": {},
        "magnitude": {},
    }

    # ---- primary task ----
    for m in primary_models:
        for s in seeds:
            a_full = primary_analysis(rows, m, s)
            a_loo = primary_analysis(rows, m, s, exclude_pub=args.dominant_pub)
            report["primary"][f"{m}:{s}"] = {
                "full": a_full,
                "without_dominant({})".format(args.dominant_pub): a_loo,
            }

    # ---- magnitude task ----
    for m in magn_models:
        for s in seeds:
            a_full = magnitude_analysis(rows, m, s)
            a_loo = magnitude_analysis(rows, m, s, exclude_pub=args.dominant_pub)
            report["magnitude"][f"{m}:{s}"] = {
                "full": a_full,
                "without_dominant({})".format(args.dominant_pub): a_loo,
            }

    # ---- null space / minimum p ----
    resolved = sorted({r["fold_id"] for r in rows if r["task"] == "primary"})
    n = len(resolved)
    report["null_space"] = {
        "n_exchangeable_publications": n,
        "min_2sided_p": 2 ** (1 - n),
        "unique_sign_flip_assignments": 2 ** n,
        "note": "two-sided paired permutation over publications; minimum attainable exact p = 2^(1-N).",
    }

    # ---- coverage ----
    called = sum(1 for r in rows if r["coverage_status"] == "CALLED")
    report["coverage"] = {
        "called_fraction": called / len(rows) if rows else None,
        "n_called": called, "n_rows": len(rows),
    }

    # ---- caller-mode sensitivity ----
    report["caller_mode_sensitivity"] = caller_mode_sensitivity(args.cache, args.registry)

    (out / "learnability_gate_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE learnability gate -> {out / 'learnability_gate_v1.json'}")
    # quick console summary
    for m in primary_models:
        s = 0
        a = report["primary"][f"{m}:0"]["full"]
        print(f"PRIMARY {m:10s} macroAUPRC={a['macro_auprc_model']} "
              f"delta_ap={round(a['mean_delta_ap'],4) if a['mean_delta_ap'] is not None else None} "
              f"ci={a['ci']} perm_p={a['permutation_p']}")
    for m in magn_models:
        a = report["magnitude"][f"{m}:0"]["full"]
        print(f"MAGNITUDE {m} skill={a['skill']} ci=({a['ci_low']},{a['ci_high']}) "
              f"perm_p={a['permutation_p']} n_ch={a['n_changers']}")


if __name__ == "__main__":
    sys.exit(main())