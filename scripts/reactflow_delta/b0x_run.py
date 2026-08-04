#!/usr/bin/env python3
"""B0-X Strong Baseline Qualification runner (contract §20.8).

Runs the full capacity ladder on the frozen train/validation split, computes
the frozen evaluator metrics, cluster CI, group-aware permutation, and learning
curve, and writes the baseline registry + strongest-baseline freeze.  GPU is
required for the P2 paired baseline (CUDA fallback=0).  The test split is
never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b0x_data import load_pairs, split_groups  # noqa: E402
from b0x_baselines import run_baseline, REGISTRY  # noqa: E402
from b0x_evaluate import (  # noqa: E402
    cluster_ci,
    group_aware_permutation,
    learning_curve,
    per_pair_loss,
    pooled_skill,
)

RUN_ID = "b0x_strong_baseline_20260804_v1"
SCHEMA = "reactflow_delta.b0x_registry.v1"


def _is_finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and _finite_float(v)


def _finite_float(v) -> bool:
    import math
    return math.isfinite(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--p2-hidden", type=int, default=64)
    ap.add_argument("--p2-epochs", type=int, default=20)
    ap.add_argument("--p2-lr", type=float, default=1e-3)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--include-tree", action="store_true",
                    help="also run the CPU-bound tree baseline (optional ladder step)")
    args = ap.parse_args()

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"train", "validation"})
    groups = split_groups(pairs)
    train = groups.get("train", [])
    val = groups.get("validation", [])
    print(f"[b0x] train={len(train)} validation={len(val)}", flush=True)
    if not train or not val:
        raise RuntimeError("need both train and validation")

    # ---- trivial baselines (ladder step 1) ----
    trifl_names = ["zero", "train_mean", "mutation_type_mean", "edit_only", "wt_only"]
    trivial_results = {}
    for name in trifl_names:
        res = run_baseline(name, train, val, device=args.device)
        trivial_results[name] = res
        print(f"[b0x] trivial {name}: status={res.status} params={res.param_count}", flush=True)

    # strongest trivial baseline = best WMAE Skill vs zero reference
    zero_preds = trivial_results["zero"].predictions
    def skill_abs(name):
        res = trivial_results[name]
        if res.status != "ok":
            return float("-inf")
        sk = pooled_skill(val, res.predictions, zero_preds)
        return sk["skill_mae"]
    strongest_trivial = max(trifl_names, key=skill_abs)
    print(f"[b0x] strongest trivial baseline: {strongest_trivial}", flush=True)

    # ---- ridge/classical (ladder step 2) ----
    # Ridge (wt_only) is already in the trivial set; the contract §14 ladder
    # uses "linear/ridge/tree" (selected OR).  The tree baseline is deliberately
    # not run here: its per-position feature construction is CPU-bound on the
    # 386k eligible-position rows and is not part of the PASS criteria.  Ridge
    # covers the classical linear step; P2 covers the learned paired model.
    nontriv = {}
    for name in ["wt_only", "tree"]:
        if name == "wt_only" and name in trivial_results:
            continue
        if name == "tree" and not args.include_tree:
            continue
        res = run_baseline(name, train, val, device=args.device)
        nontriv[name] = res
        print(f"[b0x] {name}: status={res.status}", flush=True)

    # ---- P2 paired (ladder step 3, GPU) ----
    p2_res = run_baseline(
        "p2_paired", train, val, device=args.device,
        hidden=args.p2_hidden, epochs=args.p2_epochs, lr=args.p2_lr,
    )
    print(f"[b0x] p2_paired: status={p2_res.status} params={p2_res.param_count} "
          f"runtime={p2_res.runtime_seconds:.1f}s", flush=True)
    nontriv["p2_paired"] = p2_res

    # ---- evaluator ----
    ref_preds = trivial_results[strongest_trivial].predictions
    # use zero as skill reference (contract: Skill vs reference = strongest baseline)
    skill_ref = trivial_results[strongest_trivial].predictions

    registry = {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "pairs": {"train": len(train), "validation": len(val)},
        "device": args.device,
        "strongest_trivial_baseline": strongest_trivial,
        "baselines": {},
    }

    def add_baseline(name, res, pairs_val, with_ci_perm):
        entry = {
            "status": res.status,
            "param_count": res.param_count,
            "is_learned": res.is_learned,
            "runtime_seconds": res.runtime_seconds,
            "error": res.error,
        }
        if res.status == "ok":
            sk = pooled_skill(pairs_val, res.predictions, ref_preds)
            entry["metrics"] = {
                "skill_wmae": sk["skill_wmae"],
                "skill_mae": sk["skill_mae"],
                "den_w": sk["den_w"],
            }
            losses = [per_pair_loss(p, res.predictions[p.pair_id]) for p in pairs_val
                      if p.pair_id in res.predictions]
            entry["metrics"]["wmae_mean"] = float(np.mean([l["wmae"] for l in losses])) if losses else None
            entry["metrics"]["mae_mean"] = float(np.mean([l["mae"] for l in losses])) if losses else None
            if with_ci_perm:
                # cluster CI vs strongest trivial
                entry["cluster_ci_vs_strongest_trivial"] = cluster_ci(
                    pairs_val, res.predictions, ref_preds, n_boot=args.n_boot)
                # group-aware permutation
                entry["permutation"] = group_aware_permutation(
                    pairs_val, name, res.predictions, ref_preds, n_perm=args.n_perm)
        registry["baselines"][name] = entry

    for name, res in trivial_results.items():
        add_baseline(name, res, val, with_ci_perm=(name == strongest_trivial))
    for name, res in nontriv.items():
        add_baseline(name, res, val, with_ci_perm=(name == "p2_paired"))

    # ---- PASS criteria ----
    p2_ok = p2_res.status == "ok"
    p2_skill = registry["baselines"].get("p2_paired", {}).get("metrics", {}).get("skill_wmae")
    p2_perm = registry["baselines"].get("p2_paired", {}).get("permutation", {})
    p2_ci = registry["baselines"].get("p2_paired", {}).get("cluster_ci_vs_strongest_trivial", {})
    strongest_skill = registry["baselines"].get(strongest_trivial, {}).get("metrics", {}).get("skill_wmae")
    # PASS: P2 beats group-aware permutation AND strongest trivial baseline,
    # with a positive cluster CI lower bound (vs strongest trivial).
    p2_beats_perm = p2_perm.get("pass_real_gt_null", False)
    p2_beats_trivial = p2_ok and p2_skill is not None and strongest_skill is not None and p2_skill > strongest_skill
    ci_lb = p2_ci.get("ci_low")
    p2_ci_positive = p2_ok and ci_lb is not None and ci_lb > 0
    pass_all = p2_ok and p2_beats_perm and p2_beats_trivial and p2_ci_positive

    registry["pass_criteria"] = {
        "p2_ok": p2_ok,
        "p2_beats_group_aware_permutation": p2_beats_perm,
        "p2_beats_strongest_trivial": p2_beats_trivial,
        "p2_cluster_ci_low_positive": p2_ci_positive,
        "p2_skill": p2_skill,
        "strongest_trivial_skill": strongest_skill,
        "p2_perm_p_value": p2_perm.get("p_value"),
        "p2_cluster_ci_low": ci_lb,
        "all_pass": pass_all,
    }
    registry["gate_result"] = "PASS" if pass_all else "FAIL"

    # ---- per-study skill (verifies no single-group dominance, §20.8) ----
    from collections import defaultdict
    by_study = defaultdict(list)
    for p in val:
        by_study[p.study].append(p)
    p2_preds = nontriv["p2_paired"].predictions if nontriv["p2_paired"].status == "ok" else {}
    per_study = {}
    for study in sorted(by_study):
        sks = pooled_skill(by_study[study], p2_preds, ref_preds)
        per_study[study] = {
            "n_pairs": len(by_study[study]),
            "skill_wmae": sks["skill_wmae"],
            "skill_mae": sks["skill_mae"],
        }
    positives = [round(v["skill_wmae"], 6) for v in per_study.values() if _is_finite(v["skill_wmae"])]
    no_single_dominance = bool(positives) and all(v > 0 for v in positives)
    registry["per_study"] = {
        "studies": per_study,
        "no_single_group_dominance": no_single_dominance,
    }
    if not no_single_dominance:
        registry["gate_result"] = "FAIL"

    # learning curve
    registry["learning_curve"] = learning_curve(
        val, train, p2_res.predictions if p2_ok else {}, ref_preds)

    args.out_json.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({
        "gate_result": registry["gate_result"],
        "strongest_trivial": strongest_trivial,
        "pass_criteria": registry["pass_criteria"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    import numpy as np  # noqa: E402
    raise SystemExit(main())