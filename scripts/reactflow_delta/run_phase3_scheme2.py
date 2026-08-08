#!/usr/bin/env python3
"""Phase 3 scheme-2 (contract §9.3): exact-alt WT/mutant explicit interaction.

Nested leave-one-publication-out over TRUE CHANGERS (caller_v3), comparing:
  * candidate : explicit-interaction model, input=[WT, Mut, Mut-WT, WT*(Mut-WT), cond]
  * concat    : same-capacity generic CONCAT model, input=[WT, Mut, cond]
  * seq_only  : candidate with use_wt_anchor=False (WT anchor ablation, seq-only)
  * trivial   : train-fold weighted-mean constant (endpoint_v5 baseline)

Metric (endpoint_v5): conditional WMAE skill vs trivial. Scheme-2 acceptance =
the explicit-interaction candidate beats the same-capacity concat baseline:
paired publication-block bootstrap CI of (skill_candidate - skill_concat)
lower bound > 0 across 5 seeds.

All neural models run on CUDA; CUDA-unavailable => STOP (no silent CPU fallback).
"""
from __future__ import annotations

import argparse, json, os, pickle, sys, time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_p2_v3 import (  # noqa: E402
    build_rep_groups, rep_groups_for_train, build_pair_features_aligned,
    require_cuda,
)
from caller_v3 import CallerV3  # noqa: E402
from evaluate_v5 import conditional_wmae_skill, paired_bootstrap_skill_ci  # noqa: E402
from samplers import publication_folds, pair_magnitude  # noqa: E402
from models.pair_v1 import CapacityMatchedMLP, count_params  # noqa: E402
from models.pair_v2 import build_scheme2_features  # noqa: E402
from train_v2 import train_flat, predict_flat  # noqa: E402
from run_phase3 import paired_skill_diff_ci  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
# common pre-registered capacity budget (scheme-1 candidate params, kept for comparability)
TARGET_PARAMS = 11777


def _study_of(sa: str) -> str:
    return (sa or "").split("_")[0]


def fit_variant(variant, target_params, Xtr, ytr, wtr, Xte, seed, device):
    if variant == "trivial":
        c = float(np.sum(np.asarray(wtr) * np.asarray(ytr)) / max(np.sum(wtr), 1e-12))
        return np.full(len(Xte), c, dtype=np.float32)
    model = CapacityMatchedMLP(Xtr.shape[1], target_params, seed=seed)
    train_flat(model, Xtr, ytr, wtr, device, seed)
    return predict_flat(model, Xte, device).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split-yaml", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cuda-device", default="6")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after CUDA_VISIBLE_DEVICES=" + args.cuda_device +
                           ". Contract: STOP, no silent CPU fallback.")
    device = require_cuda()
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    free_mem, tot_mem = torch.cuda.mem_get_info(gpu_index)
    print(f"[scheme2] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
          f"name={gpu_name} free={free_mem/1e9:.1f}GB", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import yaml
    split = yaml.safe_load(Path(args.split_yaml).read_text(encoding="utf-8"))
    pub_map = split["publication_map"]
    study_roles = split["study_roles"]

    with open(args.cache, "rb") as fh:
        cache = pickle.load(fh)
    rec_index = cache["rec_index"]
    pairs = cache["pairs"]
    pool_studies = set(cache["pool"])
    test_studies = {s for s, r in study_roles.items() if r == "test"}
    pool_studies = pool_studies - test_studies

    pair_recs = {}
    for p in pairs:
        if _study_of(p["source_accession"]) in test_studies:
            continue
        wt = rec_index.get((p["source_accession"], p["wt_profile_index"], p["asset_name"]))
        mu = rec_index.get((p["source_accession"], p["mutant_profile_index"], p["asset_name"]))
        if wt is None or mu is None:
            continue
        pair_recs[p["source_accession"] + ":" + str(p["mutant_profile_index"])] = {
            "pair": p, "wt": wt, "mut": mu,
            "study": _study_of(p["source_accession"]),
            "pub": pub_map.get(_study_of(p["source_accession"]), "UNKNOWN"),
        }
    print(f"[scheme2] pool_studies={sorted(pool_studies)} n_pairs_usable={len(pair_recs)}", flush=True)

    all_rep_groups = build_rep_groups(rec_index, study_whitelist=pool_studies)
    pubs = sorted({v["pub"] for v in pair_recs.values()})
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])
    caller_seed = 20260807

    # fold-invariant scheme-2 features
    VARIANT_ORDER = ["candidate", "concat", "seq_only", "trivial"]
    fx = {}
    for v in VARIANT_ORDER:
        if v == "trivial":
            continue
        if v == "candidate":
            fx[v] = {pid: build_scheme2_features(pr["pair"], pr["wt"], True, True)
                     for pid, pr in pair_recs.items()}
        elif v == "concat":
            fx[v] = {pid: build_scheme2_features(pr["pair"], pr["wt"], False, True)
                     for pid, pr in pair_recs.items()}
        elif v == "seq_only":
            fx[v] = {pid: build_scheme2_features(pr["pair"], pr["wt"], True, False)
                     for pid, pr in pair_recs.items()}
        print(f"[scheme2] precomputed features variant={v} dim={next(iter(fx[v].values())).shape[0]}",
              flush=True)
    pf_all = {pid: build_pair_features_aligned(pr["pair"], pr["wt"], pr["mut"])
              for pid, pr in pair_recs.items()}

    heldout = {v: {s: {"pub": [], "y": [], "w": [], "pred": []} for s in SEEDS}
               for v in VARIANT_ORDER}
    fold_info = {}

    for fold in publication_folds(pubs, pair_recs, pub_study):
        held_pub = fold["held_pub"]
        t0 = time.time()
        train_groups = rep_groups_for_train(all_rep_groups, fold["train_studies"])
        caller = CallerV3(seed=caller_seed).fit(train_groups, [], noise_replicate_groups=all_rep_groups)
        labels = {}
        mags = {}
        for pid, pr in pair_recs.items():
            lab = caller.call(pf_all[pid]).label
            labels[pid] = lab
            if lab == "1":
                mval, wval = pair_magnitude(pf_all[pid])
                mags[pid] = (mval, wval) if mval is not None else (None, 0)
        tr_ch = [pid for pid in fold["train_pids"] if labels[pid] == "1" and mags[pid][1] > 0]
        he_ch = [pid for pid in fold["held_pids"] if labels[pid] == "1" and mags[pid][1] > 0]
        fold_info[held_pub] = {"n_train_changers": len(tr_ch), "n_held_changers": len(he_ch)}
        if not tr_ch or not he_ch:
            print(f"[fold] held={held_pub} SKIP (train_ch={len(tr_ch)} held_ch={len(he_ch)})", flush=True)
            continue

        ytr = np.array([mags[pid][0] for pid in tr_ch], dtype=np.float32)
        wtr = np.array([mags[pid][1] for pid in tr_ch], dtype=np.float32)
        yte = np.array([mags[pid][0] for pid in he_ch], dtype=np.float32)
        wte = np.array([mags[pid][1] for pid in he_ch], dtype=np.float32)

        for v in VARIANT_ORDER:
            if v == "trivial":
                Xtr_v = np.zeros((len(tr_ch), 1), dtype=np.float32)
                Xte_v = np.zeros((len(he_ch), 1), dtype=np.float32)
            else:
                Xtr_v = np.stack([fx[v][pid] for pid in tr_ch])
                Xte_v = np.stack([fx[v][pid] for pid in he_ch])
            for seed in SEEDS:
                pred = fit_variant(v, TARGET_PARAMS, Xtr_v, ytr, wtr, Xte_v, seed, device)
                h = heldout[v][seed]
                h["pub"].extend([held_pub] * len(pred))
                h["y"].extend(yte.tolist())
                h["w"].extend(wte.tolist())
                h["pred"].extend(np.clip(pred, 0.0, None).tolist())
        fold_info[held_pub]["seconds"] = round(time.time() - t0, 1)
        print(f"[fold] held={held_pub} train_ch={len(tr_ch)} held_ch={len(he_ch)} "
              f"t={time.time()-t0:.1f}s", flush=True)

    # ---- evaluate conditional WMAE skill per variant x seed ----
    table = {}
    for v in VARIANT_ORDER:
        for seed in SEEDS:
            h = heldout[v][seed]
            triv = heldout["trivial"][seed]
            skill_res = conditional_wmae_skill(h["pub"], h["y"], h["w"], h["pred"], triv["pred"])
            ci = paired_bootstrap_skill_ci(h["pub"], h["y"], h["w"], h["pred"], triv["pred"],
                                           seed=seed, n_boot=1000)
            table[(v, seed)] = {"skill": skill_res.get("skill"),
                                "wmae_model": skill_res.get("wmae_model"),
                                "wmae_baseline": skill_res.get("wmae_baseline"),
                                "n_changers": skill_res.get("n_changers"),
                                "n_publications": skill_res.get("n_publications"),
                                "ci_low": ci.get("ci_low"), "ci_high": ci.get("ci_high")}
    print("\n[scheme2] skill by variant x seed:", flush=True)
    for v in VARIANT_ORDER:
        print(f"  {v}: {[table[(v, s)]['skill'] for s in SEEDS]}", flush=True)

    # ---- scheme-2 acceptance: explicit-interaction beats same-capacity concat ----
    cand_skills = [table[("candidate", s)]["skill"] for s in SEEDS]
    concat_skills = [table[("concat", s)]["skill"] for s in SEEDS]
    diff_ci = {}
    for s in SEEDS:
        h = heldout["candidate"][s]; g = heldout["concat"][s]; t = heldout["trivial"][s]
        diff_ci[s] = paired_skill_diff_ci(h["pub"], h["y"], h["w"], h["pred"],
                                          g["pred"], t["pred"], seed=s, n_boot=1000)
    diff_ci_low_min = min(d["ci_low"] for d in diff_ci.values()
                          if isinstance(d["ci_low"], (int, float)))
    cand_beats_concat = bool(diff_ci_low_min > 0.0)
    all_identifiable = all(isinstance(x, (int, float)) for x in cand_skills + concat_skills)

    verdict = {
        "schema": "reactflow_delta.phase3.scheme2.v1",
        "run_id": out.name,
        "authority_epoch": 18,
        "endpoint": "endpoint_v5",
        "phase": "PHASE3-ARCH",
        "scheme": "exact_alt_v1_generic_interaction",
        "candidate_mean_skill": float(np.mean(cand_skills)) if all_identifiable else None,
        "concat_mean_skill": float(np.mean(concat_skills)) if all_identifiable else None,
        "candidate_vs_concat_skill_diff_ci_low_min": diff_ci_low_min,
        "candidate_beats_same_capacity_concat": cand_beats_concat,
        "criteria_checks": {
            "estimand_identifiable_all_seeds": all_identifiable,
            "candidate_ci_low_gt_0_vs_concat": cand_beats_concat,
            "n_seeds_5": True,
        },
        "per_variant_seed_skill": {v: [table[(v, s)]["skill"] for s in SEEDS]
                                   for v in VARIANT_ORDER},
        "diff_ci_by_seed": {str(s): diff_ci[s] for s in SEEDS},
        "n_distinct_publications": len(pubs),
        "fold_info": fold_info,
        "target_params": TARGET_PARAMS,
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("Phase 3 scheme-2 (exact-alt WT/mutant explicit interaction). "
                 "Candidate [WT,Mut,Mut-WT,WT*(Mut-WT),cond] vs same-capacity concat "
                 "[WT,Mut,cond]; paired publication-block bootstrap CI of "
                 "(skill_candidate - skill_concat); seq_only drops WT anchor."),
    }
    (out / "phase3_scheme2_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"schema": "reactflow_delta.phase3.scheme2.v1",
                                                  "table": {f"{v}|{s}": table[(v, s)]
                                                            for v in VARIANT_ORDER for s in SEEDS},
                                                  "verdict": verdict},
                                                 indent=2, default=str), encoding="utf-8")
    for v in VARIANT_ORDER:
        for s in SEEDS:
            h = heldout[v][s]
            np.savez_compressed(out / f"heldout_{v}_seed{s}.npz",
                                pub=np.array(h["pub"]), y=np.array(h["y"]),
                                w=np.array(h["w"]), pred=np.array(h["pred"]))
    print(f"[scheme2] wrote verdict -> {out/'phase3_scheme2_verdict.json'}", flush=True)
    print(f"[scheme2] DONE. candidate_beats_same_capacity_concat={cand_beats_concat} "
          f"diff_ci_low_min={diff_ci_low_min}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
