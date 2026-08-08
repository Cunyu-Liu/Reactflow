#!/usr/bin/env python3
"""Phase 3 — model architecture iteration (epoch 18, scheme-1 conditional-magnitude pair head).

Nested leave-one-publication-out over TRUE CHANGERS (caller_v3), comparing:
  * candidate      : PairHeadV1 (DeepSets conditional magnitude head), full features
  * generic        : CapacityMatchedMLP (flat, params matched to candidate)  [baseline]
  * ablation_exact : candidate with use_exact_alt=False (drops exact-alt global features)
  * ablation_nonloc: candidate with use_wt_anchor=False (drops WT reactivity local context)
  * trivial        : train-fold weighted-mean constant (endpoint_v5 baseline)

Metric (endpoint_v5): conditional WMAE skill vs trivial. Phase 3 acceptance = the
candidate beats the capacity-matched generic: paired publication-block bootstrap CI of
(skill_candidate - skill_generic) lower bound > 0, 5 seeds.

Every neural model runs on CUDA; CUDA-unavailable => STOP (no silent CPU fallback).
"""
from __future__ import annotations

import argparse, hashlib, json, os, pickle, sys, time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_p2_v3 import (  # noqa: E402
    build_rep_groups, rep_groups_for_train, build_pair_features_aligned,
    build_feature, require_cuda,
)
from caller_v3 import CallerV3  # noqa: E402
from evaluate_v5 import conditional_wmae_skill, paired_bootstrap_skill_ci  # noqa: E402
from samplers import publication_folds, pair_magnitude  # noqa: E402
from train_v2 import train_flat, train_pair, predict_flat, predict_pair  # noqa: E402
from models.pair_v1 import (  # noqa: E402
    PairHeadV1, CapacityMatchedMLP, count_params, split_pos_glob,
)

SEEDS = [0, 1, 2, 3, 4]
W = 21
VARIANT_ORDER = ["candidate", "generic", "ablation_exact", "ablation_nonloc", "trivial"]


def _study_of(sa: str) -> str:
    return (sa or "").split("_")[0]


def _feat_flags(variant):
    return {
        "candidate": (True, True, True),
        "generic": (True, True, True),
        "ablation_exact": (True, False, True),
        "ablation_nonloc": (False, True, True),
    }[variant]


def _pos_dim_for(variant):
    return 7 if _feat_flags(variant)[0] else 5


def fit_variant(variant, cand_target_params, Xtr, ytr, wtr, Xte, seed, device, pos_dim):
    if variant == "trivial":
        c = float(np.sum(np.asarray(wtr) * np.asarray(ytr)) / max(np.sum(wtr), 1e-12))
        return np.full(len(Xte), c, dtype=np.float32)
    if variant == "generic":
        in_dim = Xtr.shape[1]
        model = CapacityMatchedMLP(in_dim, cand_target_params, seed=seed)
        train_flat(model, Xtr, ytr, wtr, device, seed)
        return predict_flat(model, Xte, device).astype(np.float32)
    # pair-head variants (candidate / ablations)
    glob_dim = Xtr.shape[1] - W * pos_dim
    model = PairHeadV1(pos_dim, glob_dim, hidden=64, seed=seed)
    pos_tr, gl_tr = split_pos_glob(torch.from_numpy(np.asarray(Xtr, dtype=np.float32)), W, pos_dim)
    train_pair(model, pos_tr.numpy(), gl_tr.numpy(), ytr, wtr, device, seed)
    pos_te, gl_te = split_pos_glob(torch.from_numpy(np.asarray(Xte, dtype=np.float32)), W, pos_dim)
    return predict_pair(model, pos_te.numpy(), gl_te.numpy(), device).astype(np.float32)


def paired_skill_diff_ci(pubs, y, w, cand_pred, gen_pred, base_pred, seed, n_boot=1000, alpha=0.05):
    """Paired publication-block bootstrap CI of (skill_candidate - skill_generic)."""
    import random
    from collections import defaultdict
    groups = defaultdict(lambda: {"y": [], "w": [], "c": [], "g": [], "b": []})
    for p, yi, wi, ci, gi, bi in zip(pubs, y, w, cand_pred, gen_pred, base_pred):
        groups[p]["y"].append(float(yi)); groups[p]["w"].append(float(wi))
        groups[p]["c"].append(float(ci)); groups[p]["g"].append(float(gi))
        groups[p]["b"].append(float(bi))
    pub_ids = sorted(groups, key=str)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        rs = [rng.choice(pub_ids) for _ in pub_ids]
        y_, w_, c_, g_, b_ = [], [], [], [], []
        for p in rs:
            y_.extend(groups[p]["y"]); w_.extend(groups[p]["w"])
            c_.extend(groups[p]["c"]); g_.extend(groups[p]["g"]); b_.extend(groups[p]["b"])
        sw_, sb_ = sum(w_), sum(w_)
        wmae_c = sum(w_ * abs(np.array(y_) - np.array(c_))) / sw_
        wmae_g = sum(w_ * abs(np.array(y_) - np.array(g_))) / sw_
        wmae_b = sum(w_ * abs(np.array(y_) - np.array(b_))) / sw_
        if wmae_b <= 0:
            continue
        skill_c = 1 - wmae_c / wmae_b
        skill_g = 1 - wmae_g / wmae_b
        diffs.append(skill_c - skill_g)
    if not diffs:
        return {"ci_low": None, "ci_high": None, "n_publications": len(pub_ids)}
    diffs.sort()
    lo = diffs[int(np.floor((alpha / 2.0) * (len(diffs) - 1)))]
    hi = diffs[int(np.ceil((1.0 - alpha / 2.0) * (len(diffs) - 1)))]
    return {"ci_low": float(lo), "ci_high": float(hi),
            "n_publications": len(pub_ids), "n_boot": n_boot}


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
    print(f"[phase3] GPU OK: cuda_visible={args.cuda_device} idx={gpu_index} "
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
    print(f"[phase3] pool_studies={sorted(pool_studies)} n_pairs_usable={len(pair_recs)}", flush=True)

    all_rep_groups = build_rep_groups(rec_index, study_whitelist=pool_studies)
    pubs = sorted({v["pub"] for v in pair_recs.values()})
    pub_study = {}
    for v in pair_recs.values():
        pub_study.setdefault(v["pub"], set()).add(v["study"])
    caller_seed = 20260807

    # candidate param count for capacity-matching the generic
    full_X = build_feature(list(pair_recs.values())[0]["pair"],
                           list(pair_recs.values())[0]["wt"], True, True, True)
    full_dim = full_X.shape[0]
    glob_dim_full = full_dim - W * 7
    cand_ref = PairHeadV1(7, glob_dim_full, hidden=64, seed=0)
    cand_target_params = count_params(cand_ref)
    print(f"[phase3] full_dim={full_dim} glob_dim={glob_dim_full} "
          f"candidate_params={cand_target_params}", flush=True)

    # precompute features per variant
    fx = {}
    for v in VARIANT_ORDER:
        if v == "trivial":
            continue
        flags = _feat_flags(v)
        fx[v] = {pid: build_feature(pr["pair"], pr["wt"], *flags)
                 for pid, pr in pair_recs.items()}
        print(f"[phase3] precomputed features variant={v} pos_dim={_pos_dim_for(v)}", flush=True)
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
                Xtr_v = Xte_v = None
            else:
                Xtr_v = np.stack([fx[v][pid] for pid in tr_ch])
                Xte_v = np.stack([fx[v][pid] for pid in he_ch])
            for seed in SEEDS:
                pred = fit_variant(v, cand_target_params, Xtr_v, ytr, wtr,
                                   Xte_v, seed, device, _pos_dim_for(v))
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
    print("\n[phase3] skill by variant x seed:", flush=True)
    for v in VARIANT_ORDER:
        print(f"  {v}: {[table[(v,s)]['skill'] for s in SEEDS]}", flush=True)

    # ---- Phase 3 acceptance: candidate beats capacity-matched generic ----
    c_skills = [table[("candidate", s)]["skill"] for s in SEEDS]
    g_skills = [table[("generic", s)]["skill"] for s in SEEDS]
    diff_ci = {}
    for s in SEEDS:
        h = heldout["candidate"][s]; g = heldout["generic"][s]; t = heldout["trivial"][s]
        diff_ci[s] = paired_skill_diff_ci(h["pub"], h["y"], h["w"], h["pred"],
                                          g["pred"], t["pred"], seed=s, n_boot=1000)
    diff_ci_low_min = min(d["ci_low"] for d in diff_ci.values()
                          if isinstance(d["ci_low"], (int, float)))
    cand_beats_generic = bool(diff_ci_low_min > 0.0)
    all_identifiable = all(isinstance(x, (int, float)) for x in c_skills + g_skills)

    verdict = {
        "schema": "reactflow_delta.phase3.v1",
        "run_id": out.name,
        "authority_epoch": 18,
        "endpoint": "endpoint_v5",
        "phase": "PHASE3-ARCH",
        "primary_scheme": "pair_v1_conditional_magnitude_head",
        "candidate_mean_skill": float(np.mean(c_skills)) if all_identifiable else None,
        "generic_mean_skill": float(np.mean(g_skills)) if all_identifiable else None,
        "candidate_vs_generic_skill_diff_ci_low_min": diff_ci_low_min,
        "candidate_beats_capacity_matched_generic": cand_beats_generic,
        "criteria_checks": {
            "estimand_identifiable_all_seeds": all_identifiable,
            "candidate_ci_low_gt_0_vs_generic": cand_beats_generic,
            "n_seeds_5": True,
        },
        "per_variant_seed_skill": {v: [table[(v, s)]["skill"] for s in SEEDS]
                                   for v in VARIANT_ORDER},
        "per_variant_seed_ci_low": {v: [table[(v, s)]["ci_low"] for s in SEEDS]
                                    for v in VARIANT_ORDER},
        "diff_ci_by_seed": {str(s): diff_ci[s] for s in SEEDS},
        "n_distinct_publications": len(pubs),
        "fold_info": fold_info,
        "candidate_params": cand_target_params,
        "generic_params": count_params(CapacityMatchedMLP(full_dim, cand_target_params, seed=0)),
        "adjudicated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": ("Phase 3 scheme-1 architecture iteration. Candidate PairHeadV1 vs "
                 "capacity-matched generic MLP, paired publication-block bootstrap CI of "
                 "skill difference. endpoint_v5 conditional WMAE skill vs trivial baseline."),
    }
    (out / "phase3_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")

    results = {"run_id": out.name, "authority_epoch": 18, "endpoint": "endpoint_v5",
               "variants": VARIANT_ORDER, "seeds": SEEDS, "verdict": verdict,
               "table": {f"{v}:{s}": table[(v, s)] for v in VARIANT_ORDER for s in SEEDS}}
    (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    for v in VARIANT_ORDER:
        for seed in SEEDS:
            h = heldout[v][seed]
            np.savez_compressed(out / f"heldout_{v}_seed{seed}.npz",
                                pub=np.array(h["pub"]), y=np.array(h["y"]),
                                w=np.array(h["w"]), pred=np.array(h["pred"]))

    print("\nDONE phase3 ->", out)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if cand_beats_generic else 1


if __name__ == "__main__":
    sys.exit(main())
