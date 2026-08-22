#!/usr/bin/env python3
"""patch_p3_p20: recompute fold-19 (P20) B* with WT-profile fallback and patch result."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta.evaluator_crps_v1 import crps_gaussian
from scripts.reactflow_delta.run_p3_lrso_v2 import _fit_ridge_bstar, _feat
from scripts.reactflow_delta.p2_learnability import d_p_p2, puzzle_level_ci20, studentized_sign_flip, leave_one_puzzle_influence

M2 = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv"
RES = "/mnt/cunyuliu/prospective_v2_p3_20260813/p3_lrso_v2_result.json"
device = "cpu"
univ = M2Universe(Path(M2)); led = univ.build()
puzzles = sorted(set(r.puzzle for r in univ.get_records()))
split = build_split_v4(puzzles)
fold19 = [f for f in split["folds"] if f.held_puzzle == "P20"][0]
train_records = [r for r in univ.get_records() if r.puzzle in set(fold19.train_puzzles)]
held_records = [r for r in univ.get_records() if r.puzzle == "P20"]

coef, _ = _fit_ridge_bstar(univ, train_records, device)
# fallback to WT profile if coef not finite
if not np.all(np.isfinite(coef)):
    coef = None
total = 0.0; n = 0
for r in held_records:
    c = univ.get_construct(r.construct_id)
    tprof, _ = univ.mutant_full_profile(
        r.wt_id, r.design_pos, r.ref, r.alt
    )
    if tprof is None or not r.target_observed:
        continue
    we = (
        c.wt_reactivity[r.full_pos]
        if not np.isnan(c.wt_reactivity[r.full_pos])
        else 0.0
    )
    nz = ~np.isnan(c.wt_reactivity) & ~np.isnan(tprof)
    idx = np.where(nz)[0]
    prof = np.full(len(c.wt_reactivity), np.nan)
    for i in idx:
        if coef is not None:
            f = _feat(we, c.wt_reactivity[i], i - r.full_pos, r.ref, r.alt)
            prof[i] = float(np.dot(coef, np.concatenate([[1.0], f])))
        else:
            prof[i] = c.wt_reactivity[i]  # WT-profile fallback (zero response)
    q = ~np.isnan(tprof) & ~np.isnan(prof)
    total += float(np.nanmean([crps_gaussian(prof[i], 0.3, tprof[i]) for i in np.where(q)[0]]))
    n += 1
b20 = total / n if n else float("nan")
print("P20 B*_held_crps (patched) =", round(b20, 4), "coef_fallback=", coef is None)

d = json.loads(Path(RES).read_text(encoding="utf-8"))
d["b_star_held_crps"]["P20"] = b20
for k in ["2", "4", "8"]:
    lrso20 = d["rank_held_crps"][k]["P20"]
    d["rank_d_p3"][k]["P20"] = d_p_p2(b20, lrso20)
    effects = [d["rank_d_p3"][k][f.held_puzzle] for f in split["folds"]]
    d[f"ci_rank_{k}"] = puzzle_level_ci20(effects)
    d[f"sign_rank_{k}"] = studentized_sign_flip(effects)
    d[f"lop_rank_{k}"] = leave_one_puzzle_influence(effects, [f.held_puzzle for f in split["folds"]])
d["verdict"] = {str(k): ("NO_INCREMENTAL_LRSO_SKILL" if not d[f"ci_rank_{k}"].get("ci_low_gt_0")
                         else "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT") for k in ["2", "4", "8"]}
Path(RES).write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
for k in ["2", "4", "8"]:
    ci = d[f"ci_rank_{k}"]
    dp = d["rank_d_p3"][k]
    npos = sum(1 for v in dp.values() if v > 0)
    print(f"rank {k}: mean={ci['mean']:.4f} CI=[{ci['ci_low']:.4f},{ci['ci_high']:.4f}] ci_low_gt_0={ci['ci_low_gt_0']} n_pos={npos}")
print("verdict", d["verdict"])
