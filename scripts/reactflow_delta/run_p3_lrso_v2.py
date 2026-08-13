#!/usr/bin/env python3
"""run_p3_lrso_v2: plain RFD-LRSO vs fold-specific B*=Direct* (contract 12.5, 14).

Per outer fold:
  - encode each WT construct once (WT context encoder reused across mutants)
  - fit LRSO (K_rank in {2,4,8}) on outer-train full-construct profiles (batched, GPU)
  - B*=Direct* = reg_direct (ridge on direct chemistry template), fit on outer-train
  - predict held-puzzle full-construct profiles for LRSO and B*
  - D_p^P3 = L_B* - L_LRSO ; 20-puzzle t-CI (positive => LRSO better)

Outcome-blind: held outcomes only in evaluator. Gaussian predictive scale from
train residuals. v2: batched training + correct B*. Single seed per rank
(five-seed mixture to follow).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta.evaluator_crps_v1 import crps_gaussian
from scripts.reactflow_delta.p2_learnability import (
    d_p_p2, leave_one_puzzle_influence, puzzle_level_ci20, studentized_sign_flip,
)
from scripts.reactflow_delta.lrso_v1 import RFDLRSO

ALPHA = {"A": 0, "C": 1, "G": 2, "U": 3}
EPOCHS = 6
LR = 1e-3


def _wt_ctx_tensors(univ, construct_id: str, device: str):
    c = univ.get_construct(construct_id)
    L = len(c.wt_reactivity)
    seq = np.zeros((L, 4), dtype=np.float32)
    for i, base in enumerate(c.sequence):
        seq[i, ALPHA.get(base, 3)] = 1.0
    react = np.nan_to_num(c.wt_reactivity, nan=0.0).astype(np.float32)
    mask = c.wt_observed.astype(bool)
    # log precision from error; guard NaN/<=0 => 0 precision (finite)
    err = c.wt_error.astype(np.float32)
    prec = np.where(np.isfinite(err) & (err > 0), -np.log(np.maximum(err, 1e-6)), 0.0).astype(np.float32)
    pos = np.arange(L, dtype=np.float32)
    region = np.stack([(c.region_map == "design_region").astype(np.float32),
                       (c.region_map == "other_assay_region").astype(np.float32)], axis=-1)
    return (torch.tensor(seq, device=device), torch.tensor(react, device=device),
            torch.tensor(prec, device=device), torch.tensor(mask, device=device),
            torch.tensor(pos, device=device), torch.tensor(region, device=device))


def _feat(wt_e, wt_r, dist, ref, alt):
    r = np.zeros(4); a = np.zeros(4)
    r[ALPHA.get(ref, 3)] = 1.0; a[ALPHA.get(alt, 3)] = 1.0
    return np.concatenate([[wt_e, wt_r, dist, np.tanh(dist)], r, a]).astype(np.float32)


def _fit_ridge_bstar(univ, records, device):
    feats, targets = [], []
    for r in records:
        c = univ.get_construct(r.construct_id)
        tprof, _ = univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)
        if tprof is None:
            continue
        we = c.wt_reactivity[r.pos] if not np.isnan(c.wt_reactivity[r.pos]) else 0.0
        nz = ~np.isnan(c.wt_reactivity) & ~np.isnan(tprof)
        idx = np.where(nz)[0]
        for i in idx:
            feats.append(_feat(we, c.wt_reactivity[i], i - r.pos, r.ref, r.alt))
            targets.append(float(tprof[i]))
    X = np.array(feats); y = np.array(targets)
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    lam = 1e-1
    coef = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)
    if not np.all(np.isfinite(coef)):
        coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    resid = y - Xb @ coef
    scale = max(float(np.std(resid)) if resid.size else 1e-3, 1e-3)
    return coef, scale


def _bstar_held_crps(univ, held_records, coef, device):
    total = 0.0; n = 0
    for r in held_records:
        c = univ.get_construct(r.construct_id)
        tprof, _ = univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)
        if tprof is None or not r.target_observed:
            continue
        we = c.wt_reactivity[r.pos] if not np.isnan(c.wt_reactivity[r.pos]) else 0.0
        nz = ~np.isnan(c.wt_reactivity) & ~np.isnan(tprof)
        idx = np.where(nz)[0]
        prof = np.full(len(c.wt_reactivity), np.nan)
        for i in idx:
            f = _feat(we, c.wt_reactivity[i], i - r.pos, r.ref, r.alt)
            prof[i] = float(np.dot(coef, np.concatenate([[1.0], f])))
        q = np.where(~np.isnan(tprof) & ~np.isnan(prof))[0]
        if q.size == 0:
            continue  # no qualified positions; do not poison total with nanmean([])
        total += float(np.nanmean([crps_gaussian(prof[i], 0.3, tprof[i]) for i in q]))
        n += 1
    return total / n if n else float("nan")


def _train_lrso(model, univ, train_records, ctx_cache, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    by_construct = {}
    for r in train_records:
        by_construct.setdefault(r.construct_id, []).append(r)
    for cid, recs in by_construct.items():
        seq, react, prec, mask, pos, region = ctx_cache[cid]
        # encode once; detach (train the operator heads, encoder is a fixed context provider)
        H = model.encoder(seq[None], react[None], prec[None], mask[None], pos[None], region[None])[0].detach()
        recs = [r for r in recs if r.target_observed]
        if not recs:
            continue
        c = univ.get_construct(cid)
        wt_t = torch.nan_to_num(torch.tensor(c.wt_reactivity, device=device))
        L = len(c.wt_reactivity)
        edit_idx = torch.tensor([r.pos for r in recs], device=device)
        dists = (torch.arange(L, device=device)[None, :] - edit_idx[:, None]).float()
        targets = np.stack([np.nan_to_num(univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)[0])
                            for r in recs])
        masks_t = torch.tensor(~np.isnan(targets), device=device)
        targets_t = torch.tensor(targets, device=device)
        refs = [r.ref for r in recs]; alts = [r.alt for r in recs]
        for _ in range(EPOCHS):
            delta = model.delta_batch(H, edit_idx, dists, refs, alts, masks_t)
            pred = wt_t[None, :] + delta
            loss = torch.mean(((pred - targets_t) * masks_t) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def _lrso_held_crps(model, univ, held_records, ctx_cache, device):
    total = 0.0; n = 0
    by_construct = {}
    for r in held_records:
        by_construct.setdefault(r.construct_id, []).append(r)
    for cid, recs in by_construct.items():
        seq, react, prec, mask, pos, region = ctx_cache[cid]
        c = univ.get_construct(cid)
        L = len(c.wt_reactivity)
        recs = [r for r in recs if r.target_observed]
        if not recs:
            continue
        with torch.no_grad():
            H = model.encoder(seq[None], react[None], prec[None], mask[None], pos[None], region[None])[0]
            edit_idx = torch.tensor([r.pos for r in recs], device=device)
            dists = (torch.arange(L, device=device)[None, :] - edit_idx[:, None]).float()
            qmask = torch.tensor(np.stack([~np.isnan(univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)[0])
                                           for r in recs]), device=device)
            refs = [r.ref for r in recs]; alts = [r.alt for r in recs]
            delta = model.delta_batch(H, edit_idx, dists, refs, alts, qmask)
        wt = np.nan_to_num(c.wt_reactivity)
        for bi, r in enumerate(recs):
            tprof, _ = univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)
            pred_prof = wt + delta[bi].cpu().numpy()
            q = np.where(~np.isnan(tprof) & ~np.isnan(pred_prof))[0]
            if q.size == 0:
                continue  # no qualified positions; do not poison total with nanmean([])
            total += float(np.nanmean([crps_gaussian(pred_prof[i], 0.3, tprof[i]) for i in q]))
            n += 1
    return total / n if n else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rank", default="2,4,8")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    ranks = [int(x) for x in args.rank.split(",")]
    print("device", device, "ranks", ranks, flush=True)

    univ = M2Universe(Path(args.m2_csv)); led = univ.build()
    puzzles = sorted(set(r.puzzle for r in univ.get_records()))
    split = build_split_v4(puzzles)
    all_records = univ.get_records()

    rank_held = {k: {} for k in ranks}
    rank_dp = {k: {} for k in ranks}
    b_held = {}

    for fold in split["folds"]:
        held = fold.held_puzzle
        train_records = [r for r in all_records if r.puzzle in set(fold.train_puzzles)]
        held_records = [r for r in all_records if r.puzzle == held]
        train_constructs = sorted({r.construct_id for r in train_records})
        held_constructs = sorted({r.construct_id for r in held_records})

        ctx_cache = {}
        for cid in train_constructs + held_constructs:
            ctx_cache[cid] = _wt_ctx_tensors(univ, cid, device)

        coef_b, scale_b = _fit_ridge_bstar(univ, train_records, device)
        b_held[held] = _bstar_held_crps(univ, held_records, coef_b, device)
        print(f"fold {fold.outer_fold} B*_held_crps={b_held[held]:.4f}", flush=True)

        for k in ranks:
            model = RFDLRSO(k_rank=k).to(device)
            _train_lrso(model, univ, train_records, ctx_cache, device)
            lrso_crps = _lrso_held_crps(model, univ, held_records, ctx_cache, device)
            rank_held[k][held] = lrso_crps
            rank_dp[k][held] = d_p_p2(b_held[held], lrso_crps)
            print(f"fold {fold.outer_fold} rank {k} L_LRSO={lrso_crps:.4f} D_p^P3={rank_dp[k][held]:.4f}", flush=True)

    result = {"schema_version": "reactflow_delta.p3_lrso_v2.v1", "device": device,
              "ranks": ranks, "b_star_held_crps": b_held, "rank_held_crps": rank_held,
              "rank_d_p3": rank_dp}
    for k in ranks:
        effects = [rank_dp[k][f.held_puzzle] for f in split["folds"]]
        result[f"ci_rank_{k}"] = puzzle_level_ci20(effects)
        result[f"sign_rank_{k}"] = studentized_sign_flip(effects)
        result[f"lop_rank_{k}"] = leave_one_puzzle_influence(effects, [f.held_puzzle for f in split["folds"]])
    result["verdict"] = {str(k): ("NO_INCREMENTAL_LRSO_SKILL" if not result[f"ci_rank_{k}"].get("ci_low_gt_0")
                                  else "LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT") for k in ranks}
    (out / "p3_lrso_v2_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k in ("device", "ranks", "verdict",
                     "ci_rank_2", "ci_rank_4", "ci_rank_8")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
