#!/usr/bin/env python3
"""Phase 3 closure -> benchmark/resource route (deliverable 1).

Contract §12 Phase 3 failure handling (all three architecture schemes FAIL-CLOSED)
directs: "使用最简单generic并转benchmark/resource" and §9.2/9.3 direct us to
"分析 caller reliability 和 domain shift" rather than search more architectures.

This script produces an auditable characterization that answers WHY no architecture
scheme beat the same-capacity generic (all with paired CI lower bound <= 0):

  A. Caller reliability  : global ICC(1,1), per-group ICC distribution, and the
     fraction of pairs the frozen caller cannot call (NO_CALL) because structure
     or unit reliability fails. Low reliability / high NO_CALL -> labels near noise
     -> no architecture can produce a stable within-publication increment.
  B. Publication label shift: per-publication true-changers rate (frozen global
     caller). Severe cross-publication heterogeneity => domain shift in the
     target/label space, so a single model cannot generalize across publications.
  C. Feature domain shift : per-publication mean of the generic [WT,Mut,cond]
     features vs the pool training mean (standardized mean difference, SMD).

Statistical / CPU-only (no model training); run under pc_cng_gpu env for the
same numpy/torch stack. CUDA not required; this is a data/measurement diagnostic,
not a training or GPU-validation step.
"""
from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_p2_v3 import build_rep_groups, build_pair_features_aligned  # noqa: E402
from caller_v3 import CallerV3  # noqa: E402
from models.pair_v2 import build_scheme2_features  # noqa: E402


def _study_of(sa: str) -> str:
    return (sa or "").split("_")[0]


def _pct(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split-yaml", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--caller-seed", type=int, default=20260807)
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

    pf_all = {pid: build_pair_features_aligned(pr["pair"], pr["wt"], pr["mut"])
              for pid, pr in pair_recs.items()}

    all_rep_groups = build_rep_groups(rec_index, study_whitelist=pool_studies)
    caller = CallerV3(seed=args.caller_seed)
    caller.fit(all_rep_groups, [], noise_replicate_groups=all_rep_groups)

    # ---- A. caller reliability + per-pair call status ----
    global_icc = getattr(caller, "_global_reliability", None)
    rel_by_group = getattr(caller, "_reliability_by_group", {})
    per_group_icc = list(rel_by_group.values())
    structure_ok = getattr(caller, "_structure_ok", None)

    labels = {}
    reliabilities = {}
    for pid, pf in pf_all.items():
        cr = caller.call(pf)
        labels[pid] = cr.label
        reliabilities[pid] = cr.reliability

    n_total = len(pair_recs)
    n_nocall = sum(1 for l in labels.values() if l == "NO_CALL")
    n_called = n_total - n_nocall
    n_changers = sum(1 for l in labels.values() if l == "1")
    called_labels = [l for l in labels.values() if l != "NO_CALL"]

    # ---- B. publication label shift ----
    pub_stats = {}
    for pid, pr in pair_recs.items():
        s = pub_stats.setdefault(pr["pub"], {"n": 0, "called": 0, "changers": 0, "nocall": 0})
        s["n"] += 1
        lab = labels[pid]
        if lab == "NO_CALL":
            s["nocall"] += 1
        else:
            s["called"] += 1
            if lab == "1":
                s["changers"] += 1
    for s in pub_stats.values():
        s["changers_rate"] = (s["changers"] / s["called"]) if s["called"] else None
    rates = [s["changers_rate"] for s in pub_stats.values() if s["changers_rate"] is not None]

    # ---- C. feature domain shift (generic features, per publication) ----
    fx = {pid: build_scheme2_features(pr["pair"], pr["wt"], False, True)
          for pid, pr in pair_recs.items()}
    F = np.stack([fx[pid] for pid in pair_recs])
    mu_train = F.mean(axis=0)
    sd_train = F.std(axis=0) + 1e-9
    pub_feat = {}
    for pid, pr in pair_recs.items():
        s = pub_feat.setdefault(pr["pub"], [])
        s.append(fx[pid])
    feat_shift = {}
    for pub, vecs in pub_feat.items():
        m = np.mean(np.stack(vecs), axis=0)
        smd = (m - mu_train) / sd_train
        feat_shift[pub] = {
            "n": len(vecs),
            "mean_abs_smd": float(np.abs(smd).mean()),
            "max_abs_smd": float(np.abs(smd).max()),
        }

    report = {
        "schema": "reactflow_delta.phase3.benchmark_resource.v1",
        "run_id": Path(args.out_dir).name,
        "authority_epoch": 18,
        "phase": "PHASE3-BENCHMARK-RESOURCE",
        "route": "benchmark/resource/negative-result (Phase 3 fail-closed disposition)",
        "purpose": ("Diagnose why no Phase 3 architecture scheme beat the same-capacity "
                    "generic: caller reliability, publication label shift, feature domain shift."),
        "caller_reliability": {
            "global_icc": global_icc,
            "structure_ok": structure_ok,
            "n_replicate_groups": len(per_group_icc),
            "group_icc_distribution": _pct(per_group_icc),
            "n_groups_icc_above_threshold": sum(
                1 for v in per_group_icc if v is not None and v >= caller.icc_threshold),
            "n_pairs_total": n_total,
            "n_pairs_called": n_called,
            "n_pairs_nocall": n_nocall,
            "fraction_nocall": n_nocall / max(n_total, 1),
            "overall_changers_rate": (n_changers / max(n_called, 1)),
            "icc_threshold": getattr(caller, "icc_threshold", None),
        },
        "publication_label_shift": {
            "n_publications": len(pub_stats),
            "n_publications_with_called_pairs": sum(1 for s in pub_stats.values() if s["called"] > 0),
            "changers_rate_distribution": _pct(rates) if rates else None,
            "changers_rate_max_min_ratio": (max(rates) / min(rates)) if rates and min(rates) > 0 else None,
            "per_publication": {k: v for k, v in sorted(pub_stats.items())},
        },
        "feature_domain_shift": {
            "method": "standardized mean difference of generic [WT,Mut,cond] features "
                      "per publication vs pool training mean",
            "per_publication_mean_abs_smd": {
                k: v for k, v in sorted(feat_shift.items(), key=lambda kv: -kv[1]["mean_abs_smd"])},
            "overall_mean_abs_smd": float(np.mean([v["mean_abs_smd"] for v in feat_shift.values()])),
            "n_publications": len(feat_shift),
        },
        "notes": (
            "Frozen GLOBAL caller fitted on all pool training studies (diagnostic only, "
            "not the per-fold caller used in the scheme runs). global ICC / NO_CALL fraction "
            "characterize caller label reliability; publication changers-rate and feature SMD "
            "characterize domain shift. If reliability is low or shift is large, no within-"
            "publication architecture increment can be detected, consistent with all three "
            "Phase 3 schemes failing the CI-lower-bound>0 gate."),
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_resource_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[benchmark_resource] wrote -> {out/'benchmark_resource_report.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
