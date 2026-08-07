#!/usr/bin/env python3
"""M0-X unified horizontal comparison: within_pair_z_max continuous-burden proxy.

Replaces the v5 mean-burden endpoint (which was NEGATIVE for dev12 because of
cross-pair absolute-magnitude miscalibration) with the scale-invariant proxy
`within_pair_z_max` -- the max within-pair z-score of per-position magnitudes.

WHY (diagnosis 20260807):
  - model ranks positions WITHIN a pair well (+0.49) but cross-pair absolute
    magnitude is miscalibrated (pred std 0.26 vs true 0.08) -> raw mean-burden
    gives negative Spearman.
  - within_pair_z_max removes the cross-pair scale drift and restores a
    POSITIVE correlation for dev12 (+0.408), matching the zero-shot folding
    baselines' sign.

This script applies the SAME proxy to EVERY model (efold, ufold, rnaformer,
moefold2d, vienna, eternafold, epro_dev12) so the ranking is apples-to-apples.
Each model's per-position score is fed through the identical generic
`pair_magnitude_proxies` -> `within_pair_z_max`.

Read-only; reuses eval_recovery score-loading conventions (no feature rebuild).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path.cwd() / "src"))
if (_HERE.parents[2] / "src").exists():
    sys.path.insert(0, str(_HERE.parents[2] / "src"))

import m0x_dev12_magnitude_calibration as calib  # noqa: E402
import m0x_eval_recovery as er  # noqa: E402  (score loading, metrics, sources)

SCHEMA = "reactflow_delta.m0x_unified_zmax_compare.v1"
RUN_ID = "m0x_unified_zmax_compare_20260807"
ITERATION_ID = "M0X_UNIFIED_ZMAX_COMPARE"


def _load_scores(base_dir, spec):
    """Load a per-pair position-score dict from a npz (model-agnostic)."""
    p = Path(spec["path"]) if isinstance(spec["path"], str) else spec["path"]
    if not p.is_absolute():
        p = base_dir / p
    return er._npz_to_dict(p, key=spec.get("key"))


def _model_position_magnitude(name, pos_scores, spec, val_pairs):
    """Return {pair_id: np.array of per-position MAGNITUDE over eligible pos}.

    - Regression heads (dev12, score_type=delta_magnitude_signed) -> |signed|.
    - Folding/zero-shot baselines -> per-position score used directly as the
      magnitude proxy (structure-derived delta magnitude).
    - ChangerClassifier (dev10/07/09) are sigmoid P(changer) in [0,1] and are
      NOT eligible for the magnitude/burden endpoint -> skipped.
    """
    mags = {}
    for p in val_pairs:
        s = pos_scores.get(p.pair_id)
        if s is None:
            continue
        arr = np.asarray(s, dtype=np.float32)
        if spec.get("score_type") == "delta_magnitude_signed":
            arr = np.abs(arr)
        mask = np.asarray(p.mask, dtype=bool)
        elig = mask & (np.arange(len(arr)) < len(mask))
        if elig.sum() == 0:
            continue
        mags[p.pair_id] = arr[elig]
    return mags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--project-root", type=Path,
                    default=Path("/home/cunyuliu/reactflow_delta_goal_20260729"))
    ap.add_argument("--prior-v3-dir", type=Path,
                    default=Path("/home/cunyuliu/reactflow_delta_goal_20260729/results/sota_pairlevel_v3_20260806"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.project_root)
    prior = Path(args.prior_v3_dir)

    from b0x_data import load_pairs, split_groups  # noqa: E402
    from b0x_baselines import _pair_scale  # noqa: E402

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"validation"})
    groups = split_groups(pairs)
    val = groups.get("validation", [])
    pair_ids = [p.pair_id for p in val]
    print(f"[data] validation pairs={len(val)} (test SEALED)", flush=True)

    # ---- true burden per pair: mean(|delta|/scale) over eligible ----
    true_burden = {}
    for p in val:
        delta = np.asarray(p.delta, dtype=np.float64)
        mask = np.asarray(p.mask, dtype=bool)
        scale = _pair_scale(p)
        elig = mask & np.isfinite(delta)
        if elig.sum() == 0:
            continue
        true_burden[p.pair_id] = float(np.mean(np.abs(delta[elig]) / scale))

    # ---- gather per-model position magnitudes ----
    model_mags = {}
    model_meta = {}

    # supervised sources
    for name, spec in er.SUPERVISED_SOURCES.items():
        if spec.get("path") is None:
            model_meta[name] = {"category": "supervised", "status": "NOT_RUN",
                                "n_pairs": 0}
            continue
        # epro_dev10_best lives in prior-v3 dir
        base = prior if name == "epro_dev10_best" else root
        p = Path(spec["path"])
        full = p if p.is_absolute() else base / p
        if not full.exists():
            model_meta[name] = {"category": "supervised", "status":
                                "SOURCE_MISSING", "n_pairs": 0, "path": str(full)}
            continue
        if er.burden_applicability(name)["status"] == "NOT_APPLICABLE_FOR_BURDEN":
            model_meta[name] = {"category": "supervised",
                                "status": "NOT_APPLICABLE_FOR_BURDEN",
                                "n_pairs": 0,
                                "reason": er.burden_applicability(name)["reason"]}
            continue
        pos = _load_scores(base, {"path": spec["path"], "key": spec.get("key")})
        mags = _model_position_magnitude(name, pos, spec, val)
        model_mags[name] = mags
        model_meta[name] = {"category": "supervised", "status": "OK",
                            "n_pairs": len(mags), "path": str(full)}

    # zero-shot sources (all in prior-v3 exactalt_scores)
    for name, rel in er.ZERO_SHOT_SOURCES.items():
        p = prior / rel
        if not p.exists():
            model_meta[name] = {"category": "zero_shot", "status":
                                "SOURCE_MISSING", "n_pairs": 0, "path": str(p)}
            continue
        pos = er._npz_to_dict(p)
        mags = _model_position_magnitude(name, pos, {"score_type": "structure"},
                                         val)
        model_mags[name] = mags
        model_meta[name] = {"category": "zero_shot", "status": "OK",
                            "n_pairs": len(mags), "path": str(p)}

    # ---- apply UNIFIED within_pair_z_max proxy to every model ----
    results = {}
    for name, mags in model_mags.items():
        if not mags:
            results[name] = {"n": 0, "status": "NO_PAIRS"}
            continue
        proxies = calib.pair_magnitude_proxies(mags)
        zm = proxies["within_pair_z_max"]
        zm_raw = proxies["raw"]
        # align with true burden (only pairs present in both)
        pids = [pid for pid in pair_ids if pid in zm and pid in true_burden]
        if len(pids) < 3:
            results[name] = {"n": len(pids), "status": "TOO_FEW"}
            continue
        true_arr = np.array([true_burden[pid] for pid in pids])
        z_arr = np.array([zm[pid] for pid in pids])
        raw_arr = np.array([zm_raw[pid] for pid in pids])
        results[name] = {
            "n": len(pids),
            "within_pair_z_max": {
                "spearman": er.spearman(true_arr, z_arr),
                "kendall": er.kendall(true_arr, z_arr),
                "ndcg_at_10": er.ndcg_at_k_scores(true_arr, z_arr, er.K),
            },
            "raw_mean_burden": {
                "spearman": er.spearman(true_arr, raw_arr),
                "kendall": er.kendall(true_arr, raw_arr),
                "ndcg_at_10": er.ndcg_at_k_scores(true_arr, raw_arr, er.K),
            },
            "category": model_meta[name]["category"],
            "status": "OK",
        }

    report = {
        "report": SCHEMA,
        "run_id": RUN_ID,
        "iteration": ITERATION_ID,
        "n_val_pairs": len(val),
        "proxy": "within_pair_z_max",
        "burden_truth": "mean(|delta|/scale) over eligible positions",
        "models": results,
        "note": ("dev10/07/09 are ChangerClassifier (sigmoid P(changer)) and are "
                 "NOT_APPLICABLE_FOR_BURDEN; dev12 regression head uses |delta_r_hat|. "
                 "All eligible models share the identical within_pair_z_max proxy."),
    }
    (out_dir / "unified_zmax_compare.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== UNIFIED within_pair_z_max BURDEN RANKING (Spearman vs true) ===")
    print(f"{'model':22s} {'cat':10s} {'n':>4s} {'zmax_spear':>10s} "
          f"{'zmax_ndcg':>10s} {'raw_spear':>10s}")
    order = sorted(results.keys(),
                   key=lambda nm: (results[nm]["within_pair_z_max"]["spearman"]
                                   if results[nm].get("status") == "OK" else -9),
                   reverse=True)
    for nm in order:
        r = results[nm]
        if r.get("status") != "OK":
            print(f"{nm:22s} {model_meta[nm].get('category','?'):10s} "
                  f"{'':>4s} {r['status']:>10s}")
            continue
        zm = r["within_pair_z_max"]
        raw = r["raw_mean_burden"]
        print(f"{nm:22s} {r['category']:10s} {r['n']:4d} "
              f"{zm['spearman']:10.4f} {zm['ndcg_at_10']:10.4f} "
              f"{raw['spearman']:10.4f}")
    print(f"\n[done] -> {out_dir/'unified_zmax_compare.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())