#!/usr/bin/env python3
"""run_p2_v3: unified target-invariant prediction ledger + frozen method-balanced
evaluator + genuinely distinct Direct* vs K_rank=0 comparison (audit 2026-08-17).

Implements the audit's Phase 1/2/3 (P0) prerequisites in ONE protocol:
  * frozen split_v4 20-fold LOPO (held puzzle fully excluded from fit/tuning/calib)
  * prediction/score separation: the model API takes only WT inputs + mutation
    identity (target-invariance); targets are joined by the evaluator-side ledger
  * frozen method-balanced estimand (position -> mutant -> cell -> method -> puzzle)
    via evaluator_v2 for BOTH single-distribution and 5-seed mixture models
  * genuinely distinct baselines: ZeroResponse, TrainMedian, RidgeDirect,
    NonlinearDirect (independent MLP), RFD-Direct (LRSOv3 k_rank=0)
  * nested rank selection: per outer fold, one rank is inner-selected from the
    pre-frozen {2,4,8} (inner 4-fold puzzle-grouped, outer-train only); the final
    comparison is rank0 -> selected-rank paired effect, NOT three separate
    significances vs ridge.

Protocol parity between rank0 and the selected positive rank (audit §9.2 / §6.5):
  * same LRSOv3 architecture (encoder + nonlinear direct head + learned scale)
  * same likelihood (frozen HP: Student-t), same cfg {lr=1e-3, wd=0}
  * same seeds {0..4}, same inner-selected epoch count, same WT-context inputs
  * only K_rank differs (0 vs the inner-selected positive rank)

Outputs (all keyed, outcome-blind):
  * per-fold keyed OOF prediction ledger (.npz, prediction only, NO target)
  * p2_v3_scores.json : per-model per-puzzle method-balanced L + paired effects + CI
  * p2_v3_selection_ledger.json : per-fold selected rank / cfg / epochs / inner CRPS

--smoke runs 2 outer folds with tiny epoch budgets (P0-7 engineering smoke ONLY;
smoke numbers must never enter any scientific conclusion).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta import evaluator_v2 as E
from scripts.reactflow_delta import baseline_v1 as B
from scripts.reactflow_delta.run_p3_lrso_v3 import (
    LRSOv3, _wt_ctx_tensors, _fit_epochs, _epoch_select_fixed_cfg,
    _maybe_compile, _mixture_crps_vec, _wt_filled, _target_matrix,
    SEEDS, SCALE_FLOOR,
)
from scripts.reactflow_delta.p2_learnability import (
    puzzle_level_ci20, studentized_sign_flip, leave_one_puzzle_influence,
)

SCHEMA = "reactflow_delta.run_p2_v3.v1"
FROZEN_CFG = {"lr": 1e-3, "wd": 0.0, "likelihood": "student_t"}
CANDIDATE_RANKS = [2, 4, 8]
FAST_MODELS = ["zero", "train_median", "reg_direct", "nonlinear"]
RANK0_ID = "rfd_direct_rank0"
RANKPOS_ID = "rfd_direct_rankpos"


# --------------------------------------------------------------------------- #
# ledger builders (prediction side is target-invariant)
# --------------------------------------------------------------------------- #
def _bio_key(univ, r, pos: int) -> str:
    return f"openknot_m2|{r.puzzle}|{r.method}|{r.construct_id}|{r.pos}|{r.ref}>{r.alt}|{pos}"


def _target_ledger(univ, held_records) -> list[E.TargetPoint]:
    """Evaluator-side held target ledger (outcome data lives HERE, never in the
    prediction file). Missing target -> target=None, qualified=WT-observed; the
    frozen evaluator excludes target=None positions (never scores them as 0)."""
    rows = []
    for r in held_records:
        c = univ.get_construct(r.construct_id)
        wt_obs = c.wt_observed.astype(bool)
        tprof, _ = univ.mutant_full_profile(r.wt_id, r.pos, r.ref, r.alt)
        for pos in range(len(wt_obs)):
            y = None
            if tprof is not None and not np.isnan(tprof[pos]):
                y = float(tprof[pos])
            rows.append(E.TargetPoint(
                biological_scoring_key=_bio_key(univ, r, pos),
                target=y, qualified=bool(wt_obs[pos])))
    return rows


def _ledger_from_profile(pred_profile: dict[str, tuple], model_id: str, outer_fold: int,
                         seed: int = 0) -> list[E.PredPoint]:
    """Wrap a {bio_key: (loc, scale, family, df)} profile into PredPoint rows.
    Non-finite predictions are marked failure (never silently zeroed)."""
    rows = []
    for k, (loc, scale, _fam, _dfv) in pred_profile.items():
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(loc):
            rows.append(E.PredPoint(biological_scoring_key=k, model_id=model_id,
                                    seed_or_component_id=seed, outer_fold=outer_fold,
                                    family="gaussian", location=0.0, scale=SCALE_FLOOR,
                                    df=None, status="failure",
                                    failure_reason="non-finite loc/scale"))
            continue
        rows.append(E.PredPoint(biological_scoring_key=k, model_id=model_id,
                                seed_or_component_id=seed, outer_fold=outer_fold,
                                family="gaussian", location=float(loc), scale=float(scale),
                                df=None))
    return rows


def _per_seed_ledger(univ, held_records, model, seed, device, outer_fold,
                     model_id: str, ctx_cache) -> list[E.PredPoint]:
    """Per-seed prediction ledger for one LRSO model (target-invariant)."""
    rows = []
    by = {}
    for r in held_records:
        by.setdefault(r.construct_id, []).append(r)
    with torch.no_grad():
        for cid, recs in by.items():
            if not recs:
                continue
            ctx = ctx_cache[cid]
            H = model.encode(ctx)
            L = H.shape[0]
            wt_obs = univ.get_construct(cid).wt_observed.astype(bool)
            wt = _wt_filled(univ, cid)
            for r in recs:
                edit_idx = torch.tensor([r.pos], device=device)
                dists = torch.tensor((np.arange(L) - r.pos).astype(np.float32),
                                     device=device)[None, :]
                delta, scale_t = model.forward_op(
                    H, edit_idx, dists, [r.ref], [r.alt],
                    torch.tensor(wt_obs[None], device=device))
                pred = wt[None, :] + delta.cpu().numpy()
                scl = scale_t.cpu().numpy()
                for pos in range(L):
                    rows.append(E.PredPoint(
                        biological_scoring_key=_bio_key(univ, r, pos),
                        model_id=model_id, seed_or_component_id=seed,
                        outer_fold=outer_fold, family="gaussian",
                        location=float(pred[0, pos]), scale=float(scl[pos]), df=None))
    return rows


def _mixture_position_losses(univ, held_records, models: list, ctx_cache, device,
                             model_id: str) -> dict[str, float]:
    """Per-bio-key five-seed equal-weight Gaussian-mixture CRPS (contract 9.1),
    target-invariant: forward_op receives the WT-observed mask ONLY. Returns
    {bio_key: mixture CRPS} over target-qualified & WT-observed positions."""
    models = [m.eval() for m in models]
    losses: dict[str, float] = {}
    by = {}
    for r in held_records:
        by.setdefault(r.construct_id, []).append(r)
    with torch.no_grad():
        for cid, recs in by.items():
            if not recs:
                continue
            tmat, wt_obs = _target_matrix(univ, recs)
            obs = wt_obs[0]
            edit_idx = torch.tensor([r.pos for r in recs], device=device)
            dists = (torch.arange(tmat.shape[1], device=device)[None, :] - edit_idx[:, None]).float()
            refs = [r.ref for r in recs]; alts = [r.alt for r in recs]
            ctx = ctx_cache[cid]
            masks = torch.tensor(wt_obs, device=device)  # WT-obs ONLY (target-invariant)
            preds = []; scales = []
            for m in models:
                H = m.encode(ctx)
                delta, scale = m.forward_op(H, edit_idx, dists, refs, alts, masks)
                pred = torch.tensor(_wt_filled(univ, cid), device=device)[None, :] + delta
                preds.append(pred.cpu().numpy()); scales.append(scale.cpu().numpy())
            for bi, r in enumerate(recs):
                tprof = tmat[bi]
                q = np.where(~np.isnan(tprof) & obs)[0]
                if q.size == 0:
                    continue
                locs = [p[bi][q] for p in preds]
                scs = [s[q] for s in scales]
                mix = _mixture_crps_vec(locs, scs, tprof[q])
                for j, pos in enumerate(q):
                    losses[_bio_key(univ, r, int(pos))] = float(mix[j])
    return losses


def _puzzle_l_map(pos_losses: dict[str, float]) -> dict[str, float]:
    """Per-puzzle method-balanced L from per-key position losses (frozen evaluator)."""
    res = E.score_position_losses(pos_losses, method_balanced=True)
    return {p: res["puzzles"][p]["L"] for p in sorted(res["puzzles"])}


# --------------------------------------------------------------------------- #
# nested rank selection (inner 4-fold, outer-train only)
# --------------------------------------------------------------------------- #
def _select_rank_inner(univ, train_records, inner_groups, ctx_cache, device,
                       candidate_ranks, cfg, max_epochs, patience, seed=0):
    """Inner 4-fold puzzle-grouped rank selection over candidate_ranks (frozen cfg).
    max_epochs/patience here are the RANKING-SCAN budget (reduced, e.g. 40/10, so
    selecting a rank costs a fraction of a full-budget fit — mirrors the P3
    hp_max_epochs=40 HP-ranking scan). Returns (best_rank, best_epochs,
    {rank: {"inner_crps","epochs"}}); best_epochs is the chosen rank's mean best
    inner epoch and is the shared training length for BOTH rank0 and the selected
    positive rank (parity)."""
    best_rank = None; best_score = float("inf"); best_ep = 0
    scores = {}
    for k in candidate_ranks:
        ep, val = _epoch_select_fixed_cfg(univ, train_records, inner_groups, ctx_cache,
                                          device, k, cfg, max_epochs, patience, seed)
        scores[k] = {"inner_crps": float(val) if np.isfinite(val) else None,
                     "epochs": int(ep)}
        if np.isfinite(val) and val < best_score:
            best_score = val; best_rank = k; best_ep = ep
    if best_rank is None:
        best_rank = candidate_ranks[0]; best_ep = 0
    return best_rank, best_ep, scores


def _fit_lrso_family(univ, train_records, ctx_cache, device, k_rank, cfg, epochs):
    """Fit five seeds of LRSOv3(k_rank) with the shared frozen cfg + epoch count."""
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = _maybe_compile(LRSOv3(k_rank=k_rank, likelihood=cfg["likelihood"]).to(device))
        _fit_epochs(m, univ, train_records, ctx_cache, device, cfg, max(epochs, 1))
        m.eval()
        models.append(m)
    return models


# --------------------------------------------------------------------------- #
# per-fold driver
# --------------------------------------------------------------------------- #
def run_fold(univ, fold, all_records, ctx_cache, device, *, fast_models,
             cfg, candidate_ranks, rank_max_epochs, rank_patience, seed=0):
    """Run one outer fold; returns per-model per-puzzle method-balanced L + the
    per-seed ledgers (prediction only) + the rank-selection fragment.
    rank_max_epochs/rank_patience bound the inner rank-selection SCAN only; the
    final rank0 + selected positive rank are both trained for the SAME
    inner-selected epoch count (protocol parity)."""
    held = fold.held_puzzle
    train_records = [r for r in all_records if r.puzzle in set(fold.train_puzzles)]
    held_records = [r for r in all_records if r.puzzle == held]
    out: dict = {"outer_fold": fold.outer_fold, "held_puzzle": held,
                 "models": {}, "seed_ledgers": {}, "selection": None}

    # ---- fast baselines: genuinely distinct, independently fitted on outer-train
    for model_id in fast_models:
        model = B.BASELINES[model_id]()
        model.fit(univ, train_records, device)
        prof = model.predict_full_profile(univ, held_records, device)
        pred_rows = _ledger_from_profile(prof, model_id, fold.outer_fold)
        tgt = _target_ledger(univ, held_records)
        score = E.score_ledger(pred_rows, tgt, method_balanced=True)
        out["models"][model_id] = {
            p: score["puzzles"][p]["L"] for p in sorted(score["puzzles"])}
        out["models"][model_id + "_n_positions"] = score["n_matched_positions"]

    # ---- LRSO family: rank0 + inner-selected positive rank, shared protocol
    rank_pos, rank_pos_epochs, inner_scores = _select_rank_inner(
        univ, train_records, fold.inner_groups, ctx_cache, device,
        candidate_ranks, cfg, rank_max_epochs, rank_patience, seed=seed)
    out["selection"] = {
        "selected_rank": rank_pos,
        "epochs": rank_pos_epochs,
        "cfg": cfg,
        "inner_rank_scores": inner_scores,
    }
    # rank0 uses the SAME epoch count as the selected positive rank (parity), so
    # the only difference is K_rank.
    models_r0 = _fit_lrso_family(univ, train_records, ctx_cache, device,
                                 0, cfg, rank_pos_epochs)
    models_rp = _fit_lrso_family(univ, train_records, ctx_cache, device,
                                 rank_pos, cfg, rank_pos_epochs)
    out["seed_ledgers"]["rank0"] = {
        seed: _per_seed_ledger(univ, held_records, m, seed, device, fold.outer_fold,
                               RANK0_ID, ctx_cache)
        for seed, m in enumerate(models_r0)}
    out["seed_ledgers"]["rankpos"] = {
        seed: _per_seed_ledger(univ, held_records, m, seed, device, fold.outer_fold,
                               RANKPOS_ID, ctx_cache)
        for seed, m in enumerate(models_rp)}
    pos_r0 = _mixture_position_losses(univ, held_records, models_r0, ctx_cache,
                                      device, RANK0_ID)
    pos_rp = _mixture_position_losses(univ, held_records, models_rp, ctx_cache,
                                      device, RANKPOS_ID)
    out["models"][RANK0_ID] = _puzzle_l_map(pos_r0)
    out["models"][RANKPOS_ID] = _puzzle_l_map(pos_rp)
    out["models"][RANK0_ID + "_n_positions"] = len(pos_r0)
    out["models"][RANKPOS_ID + "_n_positions"] = len(pos_rp)
    return out


# --------------------------------------------------------------------------- #
# artifact persistence
# --------------------------------------------------------------------------- #
def _dump_seed_ledgers(out: Path, fold: int, tag: str, seed_ledgers: dict) -> None:
    """Persist per-seed prediction ledgers as .npz (prediction only, no target)."""
    if not seed_ledgers:
        return
    all_rows = []
    for seed in sorted(seed_ledgers):
        all_rows.extend((seed, r) for r in seed_ledgers[seed])
    keys = np.asarray([r.biological_scoring_key for _, r in all_rows], dtype=object)
    loc = np.array([r.location for _, r in all_rows], dtype=np.float64)
    scale = np.array([r.scale for _, r in all_rows], dtype=np.float64)
    seeds = np.array([s for s, _ in all_rows], dtype=np.int64)
    np.savez_compressed(
        out / f"p2_v3_oof_predictions_{tag}_fold{fold}.npz",
        keys=keys, loc=loc, scale=scale, seed=seeds)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Unified Direct*/K_rank=0 protocol (audit P0)")
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--folds", default="",
                    help="comma-separated outer fold indices to run (default: all)")
    ap.add_argument("--smoke", action="store_true",
                    help="P0-7 engineering smoke: 2 folds, tiny epochs (never scientific)")
    ap.add_argument("--max-epochs", type=int, default=200,
                    help="cap for the final model training length (unused when the "
                         "inner-selected epoch count is lower; kept for protocol parity)")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--rank-max-epochs", type=int, default=40,
                    help="epoch budget for the inner rank-selection SCAN (P3 "
                         "hp_max_epochs=40 precedent); the chosen rank's mean best "
                         "epoch is the shared training length for rank0 + selected rank")
    ap.add_argument("--rank-patience", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fast-models", default=",".join(FAST_MODELS))
    args = ap.parse_args(argv)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    fast_models = [x for x in args.fast_models.split(",") if x]

    # wire the torch.compile engineering flag through to the shared P3 machinery
    # (the LRSOv3 fit path reads run_p3_lrso_v3._COMPILE_FLAG via _maybe_compile).
    import scripts.reactflow_delta.run_p3_lrso_v3 as _P3
    _P3._COMPILE_FLAG = args.compile

    max_epochs = args.max_epochs; patience = args.patience
    rank_max_epochs = args.rank_max_epochs; rank_patience = args.rank_patience
    if args.smoke:
        max_epochs = min(max_epochs, 3)
        patience = min(patience, 1)
        rank_max_epochs = min(rank_max_epochs, 3)
        rank_patience = min(rank_patience, 1)

    univ = M2Universe(Path(args.m2_csv)); univ.build()
    puzzles = sorted(set(r.puzzle for r in univ.get_records()))
    split = build_split_v4(puzzles)
    all_records = univ.get_records()

    fold_ids = [int(x) for x in args.folds.split(",") if x]
    if not fold_ids:
        fold_ids = [f.outer_fold for f in split["folds"]]
    if args.smoke:
        fold_ids = fold_ids[:2]

    print(f"[p2_v3] device={device} folds={fold_ids} frozen_cfg={FROZEN_CFG} "
          f"candidate_ranks={CANDIDATE_RANKS} rank_scan_budget=({rank_max_epochs},{rank_patience}) "
          f"final_cap={max_epochs} fast_models={fast_models}", flush=True)

    # precompute all WT contexts once (outcome-blind)
    ctx_cache = {}
    for cid in sorted({r.construct_id for r in all_records}):
        ctx_cache[cid] = _wt_ctx_tensors(univ, cid, device)

    per_fold = {}
    selection_ledger = {}
    import time as _time
    for fold in split["folds"]:
        if fold.outer_fold not in fold_ids:
            continue
        t0 = _time.time()
        fo = run_fold(univ, fold, all_records, ctx_cache, device,
                      fast_models=fast_models, cfg=FROZEN_CFG,
                      candidate_ranks=CANDIDATE_RANKS,
                      rank_max_epochs=rank_max_epochs,
                      rank_patience=rank_patience)
        sel = fo["selection"]
        selection_ledger[fold.outer_fold] = sel
        _dump_seed_ledgers(out, fold.outer_fold, "rank0", fo["seed_ledgers"]["rank0"])
        _dump_seed_ledgers(out, fold.outer_fold, "rankpos", fo["seed_ledgers"]["rankpos"])
        fo.pop("seed_ledgers", None)
        fo.pop("selection", None)
        per_fold[fold.outer_fold] = fo
        print(f"fold {fold.outer_fold} ({fold.held_puzzle}) selection={sel} "
              f"elapsed={_time.time()-t0:.0f}s", flush=True)

    # ---- aggregate per-puzzle method-balanced L for each model (puzzle -> L)
    model_l: dict[str, dict[str, float]] = {}
    for m in fast_models + [RANK0_ID, RANKPOS_ID]:
        model_l[m] = {}
        for fid in fold_ids:
            fo = per_fold.get(fid)
            if fo is not None and m in fo["models"]:
                model_l[m].update(fo["models"][m])

    # ---- paired effects (per-puzzle D_p = L_baseline - L_candidate; positive => cand better)
    contrasts = [
        ("reg_direct", "zero", "Direct(ridge) vs WT-anchor"),
        ("reg_direct", "train_median", "Direct(ridge) vs train-median"),
        ("nonlinear", "zero", "Direct(MLP) vs WT-anchor"),
        ("nonlinear", "train_median", "Direct(MLP) vs train-median"),
        (RANK0_ID, "reg_direct", "RFD-Direct(K_rank=0) vs ridge"),
        (RANKPOS_ID, RANK0_ID, "selected-rank vs K_rank=0 (main null)"),
    ]
    effects_out = {}
    for cand, base, label in contrasts:
        puzzle_effects = {}
        for fid in fold_ids:
            fo = per_fold.get(fid)
            if fo is None:
                continue
            held = fo["held_puzzle"]
            lc = model_l[cand].get(held); lb = model_l[base].get(held)
            if lc is not None and lb is not None:
                puzzle_effects[held] = lb - lc
        eff_list = list(puzzle_effects.values())
        ci = puzzle_level_ci20(eff_list) if len(eff_list) >= 2 else {}
        effects_out[f"{cand}__vs__{base}"] = {
            "label": label,
            "per_puzzle": puzzle_effects,
            "mean": float(np.mean(eff_list)) if eff_list else None,
            "n": len(eff_list),
            "ci95": ci,
            "sign_flip": studentized_sign_flip(eff_list) if eff_list else None,
            "lop": leave_one_puzzle_influence(eff_list,
                                              [per_fold[fid]["held_puzzle"] for fid in fold_ids
                                               if per_fold.get(fid) is not None])
                   if eff_list else None,
        }

    result = {
        "schema_version": SCHEMA,
        "device": device,
        "smoke": args.smoke,
        "folds_run": fold_ids,
        "frozen_cfg": FROZEN_CFG,
        "candidate_ranks": CANDIDATE_RANKS,
        "fast_models": fast_models,
        "model_puzzle_L": model_l,
        "effects": effects_out,
        "selection_ledger": selection_ledger,
    }
    (out / "p2_v3_scores.json").write_text(json.dumps(result, indent=2, default=str))
    (out / "p2_v3_selection_ledger.json").write_text(
        json.dumps(selection_ledger, indent=2, default=str))
    for k, v in effects_out.items():
        lo = v["ci95"].get("ci_low") if v["ci95"] else None
        hi = v["ci95"].get("ci_high") if v["ci95"] else None
        print(f"effect {k}: mean={v['mean']:.5f} n={v['n']} ci95=[{lo}..{hi}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
