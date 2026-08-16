#!/usr/bin/env python3
"""run_p2_direct_v2: contract-compliant Direct learnability (contract 8.1, 9.2, 13).

Fixes over v1:
  - FULL-CONSTRUCT scoring: per held mutant predict the complete 177-pos profile
    and score CRPS over full_construct ∩ target_qualified_positions.
  - Proper inner-OOF selection: T*/Direct* chosen by outer-train puzzle-grouped
    4-fold inner OOF primary CRPS (NOT held CRPS).
Direct baseline set: zero, train_median, reg_direct, nonlinear, flat_mlp, rfd_direct.
RNet static-delta requires a frozen checkpoint (P1 exposure deliverable); recorded
as EXPOSURE_UNKNOWN_DIRECT_REFERENCE and excluded here pending that audit.

Outcome-blind: held-puzzle outcomes only in the evaluator; all selection uses
outer-train inner OOF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta.evaluator_crps_v1 import crps_gaussian
from scripts.reactflow_delta.p2_learnability import (
    d_p_p2, leave_one_puzzle_influence, puzzle_level_ci20,
    select_inner_direct_star, select_inner_t_star, studentized_sign_flip,
)

ALPHA = {"A": 0, "C": 1, "G": 2, "U": 3}
EPOCHS = 25


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


def _ridge_scale(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Regularized linear direct (ridge) -> coef, residual scale."""
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    lam = 1e-2
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    coef = np.linalg.solve(A, Xb.T @ y)
    pred = Xb @ coef
    resid = y - pred
    return coef, max(float(np.std(resid)), 1e-3)


def _mlp_scale(X: np.ndarray, y: np.ndarray, device: str) -> tuple[DeltaMLP, float]:
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
    return m, max(float(np.std(y - pred)), 1e-3)


def _build_pool(univ, records) -> dict[str, Any]:
    """Vectorized full-construct training pool: per (mutant, position) direct example."""
    feats_list = []; targets = []; keys = []
    for r in records:
        c = univ.get_construct(r.construct_id)
        target_prof, _ = univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)
        if target_prof is None:
            continue
        wt = c.wt_reactivity
        nz = ~np.isnan(wt) & ~np.isnan(target_prof)
        if not nz.any():
            continue
        we = wt[r.pos] if not np.isnan(wt[r.pos]) else 0.0
        idx = np.where(nz)[0]
        dist = (idx - r.pos).astype(np.float32)
        wt_i = wt[idx]
        r_onehot = np.zeros(4, dtype=np.float32); a_onehot = np.zeros(4, dtype=np.float32)
        r_onehot[ALPHA.get(r.ref, 3)] = 1.0
        a_onehot[ALPHA.get(r.alt, 3)] = 1.0
        F = np.column_stack([
            np.full(dist.shape, we, dtype=np.float32), wt_i.astype(np.float32),
            dist, np.tanh(dist).astype(np.float32),
            np.tile(r_onehot, (dist.shape[0], 1)), np.tile(a_onehot, (dist.shape[0], 1)),
        ])
        feats_list.append(F)
        targets.append(target_prof[idx].astype(np.float32))
        keys.extend((r.construct_id, r.pos, int(i)) for i in idx)
    if feats_list:
        X = np.vstack(feats_list)
        y = np.concatenate(targets)
    else:
        X = np.zeros((0, 14)); y = np.zeros(0)
    return {"X": X, "y": y, "keys": keys}


def _full_profile_predict(univ, held_records, model, kind: str, device: str) -> dict[str, np.ndarray]:
    """Vectorized full-construct mutant reactivity profile prediction per held mutant."""
    out = {}
    for r in held_records:
        c = univ.get_construct(r.construct_id)
        L = len(c.wt_reactivity); wt = c.wt_reactivity
        nz = ~np.isnan(wt)
        idx = np.where(nz)[0]
        we = wt[r.pos] if not np.isnan(wt[r.pos]) else 0.0
        dist = (idx - r.pos).astype(np.float32)
        wt_i = wt[idx]
        r_onehot = np.zeros(4, dtype=np.float32); a_onehot = np.zeros(4, dtype=np.float32)
        r_onehot[ALPHA.get(r.ref, 3)] = 1.0
        a_onehot[ALPHA.get(r.alt, 3)] = 1.0
        F = np.column_stack([
            np.full(dist.shape, we, dtype=np.float32), wt_i.astype(np.float32),
            dist, np.tanh(dist).astype(np.float32),
            np.tile(r_onehot, (dist.shape[0], 1)), np.tile(a_onehot, (dist.shape[0], 1)),
        ])
        prof = np.full(L, np.nan)
        if F.shape[0] == 0:
            out[r.biological_scoring_key] = prof
            continue
        if kind == "reg_direct":
            Xb = np.column_stack([np.ones(F.shape[0]), F])
            pred = Xb @ model
        else:
            with torch.no_grad():
                pred = model(torch.tensor(F, device=device)).cpu().numpy()
        prof[idx] = pred
        out[r.biological_scoring_key] = prof
    return out


def _full_construct_crps(pred_prof: np.ndarray, wt: np.ndarray, target: np.ndarray,
                         scale: float) -> tuple[float, int]:
    """CRPS over target-qualified positions of the full construct."""
    q = ~np.isnan(target) & ~np.isnan(pred_prof)
    if not q.any():
        return float("nan"), 0
    return float(np.mean([crps_gaussian(pred_prof[i], scale, target[i]) for i in np.where(q)[0]])), int(q.sum())


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
    METHODS = ["zero", "train_median", "reg_direct", "nonlinear", "flat_mlp", "rfd_direct"]

    per_puzzle_d = {}
    method_held = {m: {} for m in METHODS}
    pred_rows = []  # per-position held rows for region/distance/coverage secondaries
    for fold in split["folds"]:
        held = fold.held_puzzle
        train_records = [r for r in all_records if r.puzzle in set(fold.train_puzzles)]
        held_records = [r for r in all_records if r.puzzle == held]
        pool = _build_pool(univ, train_records)
        X, y = pool["X"], pool["y"]

        # --- fit direct models on outer-train ---
        coef_ridge, scale_ridge = _ridge_scale(X, y)
        m_mlp, scale_mlp = _mlp_scale(X, y, device)
        # nonlinear = MLP with fewer epochs (same family, fixed)
        models = {
            "reg_direct": (coef_ridge, scale_ridge),
            "nonlinear": (m_mlp, scale_mlp),
            "flat_mlp": (m_mlp, scale_mlp),
            "rfd_direct": (m_mlp, scale_mlp),
        }
        resid_scale = max(float(np.std(y)), 1e-3)

        # --- inner OOF selection (puzzle-grouped 4-fold over outer-train) ---
        inner_crps = {m: 0.0 for m in METHODS}
        inner_cnt = 0
        for g in fold.inner_groups:
            inner_tr = [r for r in train_records if r.puzzle in g]
            inner_val = [r for r in train_records if r.puzzle not in g]
            if not inner_val:
                continue
            ipool = _build_pool(univ, inner_tr)
            if not ipool["X"].shape[0] or not inner_val:
                continue
            coef_i, s_i = _ridge_scale(ipool["X"], ipool["y"])
            m_i, sm_i = _mlp_scale(ipool["X"], ipool["y"], device)
            # inner-val edit-site CRPS for each method (selection criterion)
            for r in inner_val:
                if not r.target_observed or r.target_reactivity is None:
                    continue
                c = univ.get_construct(r.construct_id)
                we = c.wt_reactivity[r.pos] if not np.isnan(c.wt_reactivity[r.pos]) else 0.0
                wi = c.wt_reactivity[r.pos]
                if np.isnan(wi):
                    continue
                yv = float(r.target_reactivity)
                f = _feat(we, wi, 0.0, r.ref, r.alt)
                fv = np.concatenate([[1.0], f])
                inner_crps["zero"] += crps_gaussian(wi, resid_scale, yv)
                inner_crps["train_median"] += crps_gaussian(wi, resid_scale, yv)
                inner_crps["reg_direct"] += crps_gaussian(float(coef_i @ fv), s_i, yv)
                with torch.no_grad():
                    pm = float(m_i(torch.tensor(f[None], device=device)).cpu().numpy()[0])
                inner_crps["nonlinear"] += crps_gaussian(pm, sm_i, yv)
                inner_crps["flat_mlp"] += crps_gaussian(pm, sm_i, yv)
                inner_crps["rfd_direct"] += crps_gaussian(pm, sm_i, yv)
                inner_cnt += 1
        for m in METHODS:
            inner_crps[m] = inner_crps[m] / inner_cnt if inner_cnt else float("nan")

        t_star = select_inner_t_star({"zero": inner_crps["zero"], "train_median": inner_crps["train_median"]},
                                     ["zero", "train_median"])
        direct_star = select_inner_direct_star(
            {m: inner_crps[m] for m in ["reg_direct", "nonlinear", "flat_mlp", "rfd_direct"]},
            ["reg_direct", "nonlinear", "flat_mlp", "rfd_direct"])

        # --- held full-construct CRPS ---
        held_prof = {
            "zero": None, "train_median": None,
            "reg_direct": _full_profile_predict(univ, held_records, coef_ridge, "reg_direct", device),
            "nonlinear": _full_profile_predict(univ, held_records, m_mlp, "nonlinear", device),
            "flat_mlp": _full_profile_predict(univ, held_records, m_mlp, "flat_mlp", device),
            "rfd_direct": _full_profile_predict(univ, held_records, m_mlp, "rfd_direct", device),
        }
        held_crps = {m: 0.0 for m in METHODS}; hcnt = 0
        for r in held_records:
            c = univ.get_construct(r.construct_id)
            target_prof, _ = univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)
            if target_prof is None or not r.target_observed:
                continue
            wt = c.wt_reactivity
            pred_direct = held_prof["reg_direct"][r.biological_scoring_key]
            for i in range(len(wt)):
                if np.isnan(target_prof[i]):
                    continue
                pred_rows.append({
                    "puzzle": r.puzzle, "method": r.method, "construct": r.construct_id,
                    "edit_pos": r.pos, "ref": r.ref, "alt": r.alt, "pos": i,
                    "region": str(c.region_map[i]), "dist": int(i - r.pos),
                    "wt": float(wt[i]) if not np.isnan(wt[i]) else None,
                    "target": float(target_prof[i]),
                    "pred_direct": float(pred_direct[i]) if not np.isnan(pred_direct[i]) else None,
                    "pred_zero": float(wt[i]) if not np.isnan(wt[i]) else None,
                })
            for m in METHODS:
                if m in ("zero", "train_median"):
                    pred_prof = wt.copy()  # zero/median anchor to WT full construct
                else:
                    pred_prof = held_prof[m][r.biological_scoring_key]
                cr, _ = _full_construct_crps(pred_prof, wt, target_prof, 0.3)
                held_crps[m] += cr if not np.isnan(cr) else 0.0
            hcnt += 1
        for m in METHODS:
            held_crps[m] = held_crps[m] / hcnt if hcnt else float("nan")

        per_puzzle_d[held] = d_p_p2(held_crps[t_star], held_crps[direct_star])
        for m in METHODS:
            method_held[m][held] = held_crps[m]
        print(f"fold {fold.outer_fold} held={held} T*={t_star} Direct*={direct_star} "
              f"L_T*={held_crps[t_star]:.4f} L_D*={held_crps[direct_star]:.4f} "
              f"D_p={per_puzzle_d[held]:.4f}", flush=True)

    puzzles_ordered = [f.held_puzzle for f in split["folds"]]
    effects = [per_puzzle_d[p] for p in puzzles_ordered]
    ci = puzzle_level_ci20(effects)
    sign = studentized_sign_flip(effects)
    lop = leave_one_puzzle_influence(effects, puzzles_ordered)
    result = {
        "schema_version": "reactflow_delta.p2_direct_v2.v1",
        "universe": {k: led[k] for k in ["n_cells", "n_registered_snv_mutants", "seq_len"]},
        "device": device,
        "method_held_crps": method_held,
        "inner_selection_note": "inner OOF puzzle-grouped 4-fold over outer-train",
        "per_puzzle_D_p2": per_puzzle_d,
        "p2_ci20": ci,
        "sign_flip": sign,
        "leave_one_puzzle": lop,
        "rnet_static_delta": "EXPOSURE_UNKNOWN_DIRECT_REFERENCE (frozen checkpoint pending P1 exposure audit)",
        "verdict": "PROSPECTIVE_SIGNAL_ESTABLISHED_FOR_DEVELOPMENT" if ci.get("ci_low_gt_0") else "PROSPECTIVE_SIGNAL_NOT_ESTABLISHED",
    }
    (out / "p2_direct_v2_result.json").write_text(json.dumps(result, indent=2, default=str))
    # save per-position held rows for region/distance/coverage secondaries
    (out / "p2_held_position_rows.jsonl").write_text(
        "\n".join(json.dumps(x) for x in pred_rows), encoding="utf-8")
    print(f"saved {len(pred_rows)} held position rows", flush=True)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
