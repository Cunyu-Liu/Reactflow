#!/usr/bin/env python3
"""M0-X GATE HARNESS: unique final-candidate selection + §13.4 confirmatory check.

This is the CORRECT gate-evaluation harness required by the contract. The prior
v3 snapshot compared the best supervised candidate against ZERO-SHOT folding
baselines (rnaformer/efold/...), but contract §13.4/§13.5 require comparison
against the STRONGEST SIMPLE BASELINE from the §14 capacity ladder
(zero, train_mean, mutation_type_mean, edit_only, wt_only, tree, p2_paired).

Pipeline (pair-level, label = `majority` sensitivity changer label):
  1. Load frozen train+validation pairs. Assert test SEALED.
  2. Fit each §14 simple baseline on TRAIN pairs; convert its per-position delta
     prediction to a pair score = max |delta_pred| over eligible positions
     (pre-registered max rule, applied uniformly to every model).
  3. Compute pair-level study-macro AUPRC for every baseline under `majority`.
  4. strongest simple baseline = argmax study-macro AUPRC.
  5. Unique valid supervised candidate = dev10_best (only candidate trained on
     the correct publication split with full 548-pair validation coverage;
     dev07/dev09 wrong_190 excluded; dev11 no saved model; dev12 is a magnitude
     regression on a different estimand, not the changer-primary).
  6. §13.4 checks vs strongest simple baseline:
       a. group-aware (study,parent)-block permutation null on study-macro AUPRC
       b. study->parent cluster bootstrap 95% CI of the study-macro AUPRC gain
       c. (matched-generic / magnitude / sensitivity reported when available)
  7. Writes a JSON gate report. NEVER fabricates a PASS: overall is PASS only
     if every applicable §13.4 conjunctive condition has located evidence.

DEVELOPMENT_ONLY. This does not unlock P0-X or claim SOTA.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path.cwd() / "src"))

from b0x_data import load_pairs, split_groups  # noqa: E402
from b0x_baselines import _pair_scale, REGISTRY  # noqa: E402

SEED = 20260804
CHANGER_TOL = 0.05
PAIR_FRACTION_THRESHOLD = 0.5
K = 10
SCHEMA = "reactflow_delta.m0x_gate_harness.v1"
RUN_ID = "m0x_gate_harness_20260807"

# §14 capacity ladder: non-GPU simple baselines first; p2_paired handled on GPU.
SIMPLE_BASELINES = ["zero", "train_mean", "mutation_type_mean", "edit_only",
                    "wt_only", "tree", "p2_paired"]
# Unique valid supervised candidate (correct split, full-val coverage).
CANDIDATE = "epro_dev10_best"
CANDIDATE_PREDS = "results/sota_pairlevel_v3_20260806/epro_dev10_predictions.npz"


def _eligible_idx(pair):
    return [i for i in range(len(pair.mask)) if pair.mask[i]]


def _changer_positions(pair):
    scale = _pair_scale(pair)
    return [i for i in _eligible_idx(pair)
            if abs(float(pair.delta[i])) > CHANGER_TOL * scale]


def _pair_label(pair) -> int:
    """majority sensitivity changer label."""
    elig = _eligible_idx(pair)
    if not elig:
        return 0
    ch = len(_changer_positions(pair))
    return 1 if ch / len(elig) >= PAIR_FRACTION_THRESHOLD else 0


def _pair_score(pair, pos) -> float:
    """Pre-registered max rule over eligible positions."""
    idx = _eligible_idx(pair)
    s = np.asarray(pos, dtype=np.float64)
    if len(idx) == 0:
        return 0.0
    return float(np.max(s[idx]))


def _average_precision(y_true, score):
    y_true = np.asarray(y_true, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    prec = tp / np.maximum(tp + fp, 1.0)
    npos = y.sum()
    if npos == 0:
        return 0.0
    rec = tp / npos
    return float(np.sum((rec - np.concatenate([[0.0], rec[:-1]])) * prec))


def _study_macro_auprc(pairs_meta, labels, scores, studies):
    aps = []
    for st in studies:
        idx = [i for i, m in enumerate(pairs_meta) if m["study"] == st]
        if not idx:
            continue
        y = np.array([labels[i] for i in idx])
        s = np.array([scores[i] for i in idx])
        aps.append(_average_precision(y, s))
    return float(np.mean(aps)) if aps else float("nan")


def _pooled_auprc(labels, scores):
    return _average_precision(labels, scores)


def _clusters(pairs_meta):
    cl = defaultdict(list)
    for i, m in enumerate(pairs_meta):
        cl[(m["study"], m["parent"])].append(i)
    return list(cl.items())


def _study_macro_gain_bootstrap(pairs_meta, labels, scores_a, scores_b,
                                studies, n_boot=1000, seed=SEED):
    """study->parent cluster bootstrap 95% CI of study-macro AUPRC gain (a-b)."""
    cl = _clusters(pairs_meta)
    rng = random.Random(seed)
    n = len(pairs_meta)
    real = _study_macro_auprc(pairs_meta, labels, scores_a, studies) - \
        _study_macro_auprc(pairs_meta, labels, scores_b, studies)
    diffs = []
    for _ in range(n_boot):
        sel = [rng.choice(cl) for _ in range(len(cl))]
        idxs = [i for _, group in sel for i in group]
        idxs = sorted(idxs)
        meta_b = [pairs_meta[i] for i in idxs]
        lab_b = [labels[i] for i in idxs]
        sa = [scores_a[i] for i in idxs]
        sb = [scores_b[i] for i in idxs]
        d = (_study_macro_auprc(meta_b, lab_b, sa, studies) -
             _study_macro_auprc(meta_b, lab_b, sb, studies))
        diffs.append(d)
    diffs = np.array(diffs)
    return {"point": float(real),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "n_boot": n_boot, "metric": "study-macro AUPRC gain (candidate - baseline)"}


def _group_aware_permutation_null(pairs_meta, labels, scores, studies,
                                  n_perm=100, seed=SEED):
    """Permute labels within (study,parent) blocks; compare study-macro AUPRC."""
    rng = np.random.default_rng(seed)
    real = _study_macro_auprc(pairs_meta, labels, scores, studies)
    by_block = defaultdict(list)
    for i, m in enumerate(pairs_meta):
        by_block[(m["study"], m["parent"])].append(i)
    blocks = list(by_block.values())
    count = 0
    for _ in range(n_perm):
        perm_labels = np.array(labels, dtype=np.float64)
        for group in blocks:
            yg = np.array([labels[i] for i in group])
            perm_labels[group] = yg[rng.permutation(len(group))]
        if _study_macro_auprc(pairs_meta, perm_labels.tolist(), scores, studies) >= real:
            count += 1
    return {"p_value": float(count / n_perm), "n_perm": n_perm,
            "real_study_macro_auprc": float(real), "seed": seed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/m0x_gate_20260807"))
    ap.add_argument("--project-root", type=Path,
                    default=Path("/home/cunyuliu/reactflow_delta_goal_20260729"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=100)
    args = ap.parse_args()

    root = Path(args.project_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"train", "validation"})
    groups = split_groups(pairs)
    train = groups.get("train", [])
    val = groups.get("validation", [])
    for p in val:
        assert p.split == "validation", "PROTECT: only validation pairs consumed"
    n_val = len(val)
    print(f"[data] train={len(train)} validation={n_val} (test SEALED)", flush=True)

    studies = sorted({p.study for p in val})
    pairs_meta = [{"study": p.study, "parent": p.parent} for p in val]
    labels = [_pair_label(p) for p in val]
    nch = sum(labels)
    print(f"[labels] majority changer prevalence={nch/n_val:.4f} ({nch}/{n_val})", flush=True)

    # ---- fit simple baselines on TRAIN, score on VAL ----
    def _npz_to_dict(path):
        d = np.load(path, allow_pickle=True)
        return {k: np.asarray(d[k], dtype=np.float32) for k in d.keys()}

    cand_pos = _npz_to_dict(root / CANDIDATE_PREDS)
    cand_scores = {p.pair_id: _pair_score(p, cand_pos.get(p.pair_id))
                   for p in val if p.pair_id in cand_pos}
    cand_covered = len(cand_scores)

    baseline_scores = {}
    baseline_meta = {}
    for name in SIMPLE_BASELINES:
        t0 = time.perf_counter()
        try:
            cls = REGISTRY[name]
            if name == "p2_paired":
                base = cls(device=args.device)
            elif name == "wt_only":
                base = cls(alpha=1.0)
            else:
                base = cls()
            base.fit(train)
            sc = {}
            for p in val:
                pred = base.predict(p)
                sc[p.pair_id] = _pair_score(p, pred)
            baseline_scores[name] = sc
            baseline_meta[name] = {"param_count": 0, "is_learned": base.is_learned,
                                   "runtime_s": round(time.perf_counter() - t0, 2)}
            print(f"[baseline] {name}: fit+scored {len(sc)} pairs", flush=True)
        except Exception as exc:  # noqa: BLE001
            baseline_meta[name] = {"error": f"{type(exc).__name__}: {exc}",
                                   "runtime_s": round(time.perf_counter() - t0, 2)}
            print(f"[baseline] {name}: FAILED {exc}", flush=True)

    # ---- study-macro AUPRC per model ----
    def smap(scores_map):
        idx = [i for i, p in enumerate(val) if p.pair_id in scores_map]
        return _study_macro_auprc([pairs_meta[i] for i in idx],
                                  [labels[i] for i in idx],
                                  [scores_map[val[i].pair_id] for i in idx],
                                  studies)

    cand_sm = smap(cand_scores)
    base_sm = {nm: (smap(sc) if nm in baseline_scores else None)
               for nm, sc in baseline_scores.items()}

    print("\n=== study-macro AUPRC (label=majority) ===")
    print(f"  candidate {CANDIDATE:<18s} (n={cand_covered:4d}) {cand_sm:.4f}")
    valid_base = {nm: v for nm, v in base_sm.items() if v is not None}
    for nm, v in valid_base.items():
        print(f"  baseline  {nm:<18s} (n={len(baseline_scores[nm]):4d}) {v:.4f}")

    # ---- strongest simple baseline ----
    if not valid_base:
        print("ERROR: no simple baseline scored; cannot run gate.")
        return 1
    strongest = max(valid_base, key=lambda nm: valid_base[nm])
    strongest_sm = valid_base[strongest]
    print(f"\n[gate] strongest simple baseline = {strongest} "
          f"(study-macro AUPRC {strongest_sm:.4f})")
    print(f"[gate] candidate study-macro AUPRC = {cand_sm:.4f} "
          f"(n_covered={cand_covered})")

    # ---- overlap pairs for the comparison ----
    over_ids = [p.pair_id for p in val
                if p.pair_id in cand_scores and p.pair_id in baseline_scores[strongest]]
    over_meta = [pairs_meta[i] for i, p in enumerate(val) if p.pair_id in over_ids]
    over_lab = [labels[i] for i, p in enumerate(val) if p.pair_id in over_ids]
    over_a = [cand_scores[pid] for pid in over_ids]
    over_b = [baseline_scores[strongest][pid] for pid in over_ids]
    n_over = len(over_ids)

    perm = _group_aware_permutation_null(over_meta, over_lab, over_a, studies,
                                         n_perm=args.n_perm, seed=SEED)
    boot = _study_macro_gain_bootstrap(over_meta, over_lab, over_a, over_b,
                                       studies, n_boot=args.n_boot, seed=SEED)

    # ---- §13.4 verdict ----
    ci_low_pos = bool(boot["ci_low"] > 0)
    perm_sig = bool(perm["p_value"] < 0.05)
    cand_beats_base = bool(cand_sm > strongest_sm)
    gate_pass = bool(ci_low_pos and perm_sig and cand_beats_base)

    verdict = {
        "schema": SCHEMA, "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "evidence_class": "DEVELOPMENT_ONLY",
        "candidate": {
            "model": CANDIDATE, "n_val_pairs_covered": cand_covered,
            "study_macro_auprc": cand_sm,
            "trained_on_split": "correct publication (train 3516 / val 548)",
            "note": "only valid full-coverage supervised candidate; "
                    "dev07/dev09 wrong_190 excluded, dev11 no saved model",
        },
        "label": {"mode": "majority",
                  "definition": "C_pair=1 iff >=50% eligible changer positions",
                  "heldout_prevalence": nch / n_val, "n_changers": nch},
        "aggregation_rule": {"primary": "max over eligible positions"},
        "strongest_simple_baseline": {
            "model": strongest, "study_macro_auprc": strongest_sm,
            "param_count": baseline_meta.get(strongest, {}).get("param_count"),
            "meta": baseline_meta.get(strongest),
        },
        "all_simple_baselines": {
            nm: {"study_macro_auprc": base_sm.get(nm),
                 "n_pairs": len(baseline_scores.get(nm, {})),
                 "meta": baseline_meta.get(nm)}
            for nm in SIMPLE_BASELINES
        },
        "permutation_null": perm,
        "bootstrap_ci": {"n_overlap_pairs": n_over, **boot},
        "gate_13_4": {
            "real_labels_beat_permutation_null": {
                "status": "PASS" if perm_sig else "FAIL",
                "p_value": perm["p_value"],
            },
            "cluster_ci_lb_gt_0_vs_strongest_simple": {
                "status": "PASS" if ci_low_pos else "FAIL",
                "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                "point": boot["point"],
            },
            "candidate_beats_strongest_simple": {
                "status": "PASS" if cand_beats_base else "FAIL",
                "candidate": cand_sm, "baseline": strongest_sm,
            },
        },
    }
    statuses = [v["status"] for v in verdict["gate_13_4"].values()]
    verdict["overall_gate"] = "PASS" if (gate_pass and all(s == "PASS" for s in statuses)) else "FAIL"
    verdict["honesty_note"] = (
        "PASS only if all three conjunctive §13.4 conditions hold with located "
        "evidence on the frozen publication validation split. FAIL does not "
        "unlock P0-X and does NOT claim SOTA (see §17.10/§20.11).")

    print(f"\n=== GATE VERDICT ===")
    for k, v in verdict["gate_13_4"].items():
        print(f"  {k:<44s} {v['status']}")
    print(f"  permutation p={perm['p_value']:.4f}")
    print(f"  bootstrap CI={boot['point']:.4f} [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
    print(f"  OVERALL GATE = {verdict['overall_gate']}")

    (out_dir / "gate_report.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"[done] -> {out_dir}/gate_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
