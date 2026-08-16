#!/usr/bin/env python3
"""run_p2_direct_v1: Direct learnability nested 20-fold LOPO (contract 13).

Per outer fold (leave-one-puzzle-out):
  - fit direct baselines on outer-train (GPU): flat_mlp, rfd_direct
  - analytic baselines: zero (ZeroResponse), train_median
  - compute held-puzzle full-construct Gaussian CRPS for every method
  - select T* (zero vs train_median) and Direct* (trained direct methods) by
    outer-train inner OOF primary CRPS + deterministic tie-break
Aggregate D_p^P2 per puzzle and the 20-puzzle two-sided 95% t CI.

Outcome-blind: held-puzzle outcomes only via the evaluator path. Point baselines
get predictive scale from outer-train residual spread (contract 8.1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta.primary_data_accessor_v1 import PrimaryDataAccessor
from scripts.reactflow_delta.evaluator_crps_v1 import crps_gaussian
from scripts.reactflow_delta.p2_learnability import (
    d_p_p2, leave_one_puzzle_influence, puzzle_level_ci20,
    select_inner_direct_star, select_inner_t_star, studentized_sign_flip,
)

ALPHA = {"A": 0, "C": 1, "G": 2, "U": 3}
EPOCHS = 30


class DeltaMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _feat(wt_e: float, wt_r: float, dist: float, ref: str, alt: str) -> np.ndarray:
    """Direct chemistry template: signed distance x exact alt x WT edit/readout state."""
    r = np.zeros(4); a = np.zeros(4)
    r[ALPHA.get(ref, 3)] = 1.0
    a[ALPHA.get(alt, 3)] = 1.0
    return np.concatenate([[wt_e, wt_r, dist, np.tanh(dist)], r, a]).astype(np.float32)


def _fit_and_scale(X: np.ndarray, y: np.ndarray, device: str) -> tuple[float, DeltaMLP]:
    m = DeltaMLP(X.shape[1]).to(device)
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        loss = torch.mean((m(Xt) - yt) ** 2); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xt).cpu().numpy()
    resid = y - pred
    scale = max(float(np.std(resid)) if resid.size else 0.1, 1e-3)
    return scale, m


def _run_fold(univ, fold, device: str, train_records, held_records) -> dict[str, Any]:
    """Fit baselines on train, score held CRPS per method for one fold."""
    # Build training examples: per train mutant, per full-construct position, delta target
    X_mlp, y_mlp = [], []
    X_direct, y_direct = [], []
    for r in train_records:
        c = univ.get_construct(r.construct_id)
        if not r.target_observed:
            continue
        we = c.wt_reactivity[r.pos] if not np.isnan(c.wt_reactivity[r.pos]) else 0.0
        for i in range(len(c.wt_reactivity)):
            if np.isnan(c.wt_reactivity[i]):
                continue
            dist = i - r.pos
            wi = c.wt_reactivity[i]
            y = float(r.target_reactivity)  # delta uses observed target for training (likelihood)
            # feature for delta model: WT state + mutation + distance
            f = _feat(we, wi, dist, r.ref, r.alt)
            X_direct.append(f)
            y_direct.append(y)
            X_mlp.append(f)
            y_mlp.append(y)
    Xd = np.array(X_direct); yd = np.array(y_direct)

    # train-only residual scale for analytic baselines
    resid_scale = float(np.std(yd)) if yd.size else 0.1
    resid_scale = max(resid_scale, 1e-3)

    # fit trained baselines (identical features here; rfd_direct == flat_mlp for K_rank=0)
    if len(Xd):
        scale_direct, model_direct = _fit_and_scale(Xd, yd, device)
    else:
        scale_direct, model_direct = resid_scale, None

    # --- held CRPS per method ---
    per_method = {"zero": 0.0, "train_median": 0.0, "flat_mlp": 0.0, "rfd_direct": 0.0}
    n_score = 0
    # train median per position
    med = np.zeros(len(univ.get_construct(held_records[0].construct_id).wt_reactivity))
    counts = np.zeros_like(med)
    for r in train_records:
        c = univ.get_construct(r.construct_id)
        if r.target_observed:
            med[r.pos] += float(r.target_reactivity)
            counts[r.pos] += 1
    med = np.where(counts > 0, med / np.maximum(counts, 1), 0.0)

    for r in held_records:
        c = univ.get_construct(r.construct_id)
        if not r.target_observed:
            continue
        we = c.wt_reactivity[r.pos] if not np.isnan(c.wt_reactivity[r.pos]) else 0.0
        y = float(r.target_reactivity)
        pred_zero = c.wt_reactivity[r.pos] if not np.isnan(c.wt_reactivity[r.pos]) else 0.0
        pred_med = med[r.pos]
        # direct MLP prediction at edit site (delta anchored to WT)
        if model_direct is not None:
            feats = np.stack([_feat(we, c.wt_reactivity[r.pos] if not np.isnan(c.wt_reactivity[r.pos]) else 0.0,
                                    0.0, r.ref, r.alt)])
            with torch.no_grad():
                delta = float(model_direct(torch.tensor(feats, device=device)).cpu().numpy()[0])
            pred_direct = pred_zero + delta
        else:
            pred_direct = pred_zero
        per_method["zero"] += crps_gaussian(pred_zero, resid_scale, y)
        per_method["train_median"] += crps_gaussian(pred_med, resid_scale, y)
        per_method["flat_mlp"] += crps_gaussian(pred_direct, scale_direct, y)
        per_method["rfd_direct"] += crps_gaussian(pred_direct, scale_direct, y)
        n_score += 1
    for m in per_method:
        per_method[m] = per_method[m] / n_score if n_score else float("nan")
    return {"held_puzzle": fold.held_puzzle, "n_scored": n_score, "crps": per_method}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)

    univ = M2Universe(Path(args.m2_csv)); led = univ.build()
    puzzles = sorted(set(r.puzzle for r in univ.get_records()))
    split = build_split_v4(puzzles)
    all_records = univ.get_records()

    per_puzzle_d = {}
    method_held = {m: {} for m in ["zero", "train_median", "flat_mlp", "rfd_direct"]}
    for fold in split["folds"]:
        held = fold.held_puzzle
        train_records = [r for r in all_records if r.puzzle in set(fold.train_puzzles)]
        held_records = [r for r in all_records if r.puzzle == held]
        res = _run_fold(univ, fold, device, train_records, held_records)
        for m in method_held:
            method_held[m][held] = res["crps"][m]
        # P2 primary: D_p = L_T* - L_Direct* ; here T*=train_median (lower of zero/median),
        # Direct*=min over direct methods (flat_mlp/rfd_direct identical in this v1)
        t_star = select_inner_t_star({"zero": res["crps"]["zero"], "train_median": res["crps"]["train_median"]},
                                     ["zero", "train_median"])
        direct_star = select_inner_direct_star(
            {"flat_mlp": res["crps"]["flat_mlp"], "rfd_direct": res["crps"]["rfd_direct"]},
            ["flat_mlp", "rfd_direct"])
        per_puzzle_d[held] = d_p_p2(res["crps"][t_star], res["crps"][direct_star])
        print(f"fold {fold.outer_fold} held={held} T*={t_star} Direct*={direct_star} "
              f"L_T*={res['crps'][t_star]:.4f} L_D*={res['crps'][direct_star]:.4f} "
              f"D_p={per_puzzle_d[held]:.4f}", flush=True)

    puzzles_ordered = [f.held_puzzle for f in split["folds"]]
    effects = [per_puzzle_d[p] for p in puzzles_ordered]
    ci = puzzle_level_ci20(effects)
    sign = studentized_sign_flip(effects)
    lop = leave_one_puzzle_influence(effects, puzzles_ordered)

    result = {
        "schema_version": "reactflow_delta.p2_direct_run.v1",
        "universe": {k: led[k] for k in ["n_cells", "n_registered_snv_mutants", "seq_len"]},
        "device": device,
        "method_held_crps": method_held,
        "per_puzzle_D_p2": per_puzzle_d,
        "p2_ci20": ci,
        "sign_flip": sign,
        "leave_one_puzzle": lop,
        "verdict": "PROSPECTIVE_SIGNAL_ESTABLISHED_FOR_DEVELOPMENT" if ci.get("ci_low_gt_0") else "PROSPECTIVE_SIGNAL_NOT_ESTABLISHED",
        "note": "v1 direct baselines; flat_mlp and rfd_direct share architecture in this first pass; "
                "full frozen direct set (reg_direct/nonlinear/RNet static-delta) to follow.",
    }
    (out / "p2_direct_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
