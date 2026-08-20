#!/usr/bin/env python3
"""run_p4_external_lrso_v1: FINAL LRSO external cluster validation (audit
P0-4, decision tree §9.1, solution 3 §9.2.3: "外部候选必须是最终 LRSO").

This supersedes the legacy ridge external run (`run_p4_external_v1.py`) as the
external candidate evidence. Two audit-grade corrections vs the legacy run:

  * candidate = FINAL LRSO (rank=2, 5-seed equal-weight Gaussian mixture,
    frozen cfg lr=1e-3/wd=0/Student-t, epoch count selected on DEVELOPMENT
    data by puzzle-grouped inner 4-fold early stopping, then trained on ALL
    development OK7a_M2 records). NOT reg_direct / ridge.
  * seqpos-CORRECT alignment: the external RDAT reactivity arrays cover a
    window [seqpos[0]-1, seqpos[0]-1+n_seqpos) of the full construct sequence,
    NOT index 0. The legacy run indexed reactivity by raw sequence index
    (misalignment ~+26), corrupting candidate features and scoring positions
    (audit finding P4-M1). This run aligns via seqpos so the shared-region
    positions, the WT context encoding, the edit position and the distances all
    share one coordinate system, and positions outside the observed window are
    non-observed (frozen attrition rule 3).

Outcome-blind discipline:
  - rank (2), cfg, seeds and the epoch count are frozen BEFORE any external
    outcome access (all selected on development data only).
  - the frozen outcome-blind component graph (24 anchors, shared-region masks
    from sequence identity only) is loaded before opening the RDAT files.
  - external reactivity is opened once (locked_outcome_access_count = 1 for
    this LRSO evaluation) and only after the model is trained.

Statistics:
  - component = one WT anchor + its single-SNV mutant library; per component
    macro CRPS over the shared-region ∩ observed window; D = L_zero - L_lrso.
  - primary exploratory unit = study cluster (K_joint = 2: SL5, Ribonanza);
    cluster-macro D + two-sided 95% t-CI (df = K_cluster-1) + LOSO stability.
  - verdict: DEVELOPMENT_REPLICATION_EXPLORATORY (K_joint=2 < K_required=9);
    NO confirmatory claim is authorized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as _st

from scripts.reactflow_delta.run_p3_lrso_v3 import (
    LRSOv3, SEEDS, ALPHA, _crps_gaussian_vec, _mixture_crps_vec,
    _epoch_select_fixed_cfg, _fit_epochs, _maybe_compile, ctx_cache_once,
)
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.split_v4_lopo_puzzle import _grouped_folds

K_REQUIRED_PLANNED = 9
K_PREACCESS_EXPECTED = 24
MIN_SHARED_NONMISSING = 20
MIN_SCORED_MUTANTS = 20
FIXED_SCALE = 0.3            # baseline predictive scale (frozen, P2/P3 evaluator)
FINAL_RANK = 2               # majority per-fold inner-selected rank (P2-v3: 18/20)
FROZEN_CFG = {"lr": 1e-3, "wd": 0.0, "likelihood": "student_t"}
MAX_EPOCHS = 200
PATIENCE = 20
P4_DATASETS = ["M2SL5_2A3_0000", "M3SARS_2A3_0000", "15KLIB_2A3_0000"]

# dataset -> study cluster (from external_provenance_registry_v1, audit P0-4)
STUDY_BY_DATASET = {
    "M2SL5_2A3_0000": "study_sl5",
    "M3SARS_2A3_0000": "study_ribonanza",
    "15KLIB_2A3_0000": "study_ribonanza",
}


def _ref_alt(name: str) -> tuple[str, str]:
    """Extract ref->alt from a single-SNV mutant profile name (last SNV token).
    Identical logic to run_p4_external_v1 (inlined to keep this module
    locally importable without the reactflow package)."""
    for tok in reversed(name.split("_")):
        if len(tok) == 3 and tok[0].isdigit() and tok[1] in ALPHA and tok[2] == "-":
            return tok[1], tok[2]
    import re as _re
    m = _re.search(r"(\d+)([ACGU])-([ACGU])", name)
    if m:
        return m.group(2), m.group(3)
    return "A", "U"


def seqpos_offset(seqpos: list[str]) -> int:
    """Sequence index (0-based) of reactivity array element 0.

    RDAT 'seqpos' entries are like 'X27'; reactivity[k] corresponds to
    sequence index (int(seqpos[k]) - 1). Element 0 therefore maps to
    (int(seqpos[0]) - 1)."""
    if not seqpos:
        return 0
    raw = seqpos[0]
    for prefix in ("X", "x", "N"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    return int(float(raw)) - 1


def window_bounds(profile: dict, seqpos: list[str]) -> tuple[int, int]:
    """(start, end) of the observed reactivity window in sequence coordinates.

    start = seqpos_offset(seqpos); end = start + len(profile['reactivity']).
    All positions outside [start, end) are non-observed (attrition rule 3).
    """
    off = seqpos_offset(seqpos)
    return off, off + len(profile["reactivity"])


def build_external_ctx(profile: dict, seqpos: list[str], device: str):
    """Build the (seq, react, prec, obs_token, pos, region) WT-context tensors
    consumed by LRSOv3.encode, from an external RDAT WT profile.

    The context covers ONLY the observed reactivity window (analogous to a
    development construct whose reactivity is fully observed). The whole window
    is treated as 'design_region' (external RDAT carries no per-position region
    annotation; pads/barcodes are outside the window and hence non-observed).
    Missing WT reactivity is mean-filled (NOT 0) with a WT-observed token,
    exactly as in run_p3_lrso_v3._wt_ctx_tensors.
    """
    seq_full = profile["profile_sequence"]
    react_arr = np.asarray(profile["reactivity"], dtype=float)
    err_arr = np.asarray(profile["reactivity_error"], dtype=float)
    off, end = window_bounds(profile, seqpos)
    W = len(react_arr)
    seq_win = seq_full[off:end] if len(seq_full) >= end else seq_full[off:]
    if len(seq_win) < W:
        raise ValueError(
            f"window {off}:{end} exceeds profile sequence length {len(seq_full)}")

    obs = ~np.isnan(react_arr)
    fill = float(np.nanmean(react_arr[obs])) if obs.any() else 0.0
    react_f = np.where(obs, react_arr, fill).astype(np.float32)
    prec = np.where(np.isfinite(err_arr) & (err_arr > 0) & obs,
                    -np.log(np.maximum(err_arr, 1e-6)), 0.0).astype(np.float32)

    seq_o = np.zeros((W, 4), dtype=np.float32)
    for i, base in enumerate(seq_win):
        seq_o[i, ALPHA.get(base, 3)] = 1.0
    obs_token = obs.astype(np.float32)
    pos = np.arange(W, dtype=np.float32)
    region = np.stack([np.ones(W, np.float32), np.zeros(W, np.float32)], axis=-1)

    return (torch_tensor(seq_o, device), torch_tensor(react_f, device),
            torch_tensor(prec, device), torch_tensor(obs_token, device),
            torch_tensor(pos, device), torch_tensor(region, device))


def torch_tensor(arr: np.ndarray, device: str):
    return torch.tensor(arr, device=device)


# --------------------------------------------------------------------------- #
# final LRSO training (development only; outcome-blind to external)
# --------------------------------------------------------------------------- #
def train_final_lrso(univ: M2Universe, all_records, device: str,
                     rank: int = FINAL_RANK, max_epochs: int = MAX_EPOCHS,
                     patience: int = PATIENCE) -> tuple[list, dict]:
    """Train the FINAL LRSO deployment model.

    1. Epoch count: dev-level puzzle-grouped inner 4-fold early stopping with
       the frozen cfg (never touches external outcomes).
    2. Final model: for each seed in {0..4}, train on ALL development records
       for that epoch count. Returns (models, meta)."""
    puzzles = sorted(set(r.puzzle for r in all_records))
    inner = _grouped_folds(puzzles, 4, seed=0)
    ctx_cache = ctx_cache_once(univ, all_records, device)
    mean_ep, mean_val = _epoch_select_fixed_cfg(
        univ, all_records, inner, ctx_cache, device, rank,
        FROZEN_CFG, max_epochs, patience, seed=0)
    mean_ep = max(int(mean_ep), 1)
    print(f"[final-lrso] inner epoch selection rank={rank} cfg={FROZEN_CFG} "
          f"mean_best_epoch={mean_ep} inner_crps={mean_val:.5f}", flush=True)

    models = []
    for s in SEEDS:
        torch_manual_seed(s)
        m = _maybe_compile(LRSOv3(k_rank=rank, likelihood=FROZEN_CFG["likelihood"]).to(device))
        _fit_epochs(m, univ, all_records, ctx_cache, device, FROZEN_CFG, mean_ep)
        models.append(m)
        print(f"[final-lrso] trained seed {s} ({len(models)}/5)", flush=True)

    meta = {
        "rank": rank, "cfg": FROZEN_CFG, "epochs": mean_ep, "seeds": SEEDS,
        "inner_epoch_selection": {
            "n_inner_folds": len(inner), "mean_best_epoch": mean_ep,
            "mean_inner_crps": mean_val, "seed": 0,
            "note": "dev-only puzzle-grouped inner early stopping; no external outcome accessed"},
        "n_dev_puzzles": len(puzzles),
        "n_dev_records": len(all_records),
    }
    return models, meta


def torch_manual_seed(seed: int) -> None:
    torch.manual_seed(seed)


def save_models(models: list, save_dir: Path, rank: int, epochs: int) -> list[dict]:
    """Persist the 5 final-model state_dicts and return per-seed file records
    (path + sha256) for reproducibility. No-op if save_dir is None."""
    if save_dir is None:
        return []
    save_dir.mkdir(parents=True, exist_ok=True)
    import hashlib
    records = []
    for m, s in zip(models, SEEDS):
        name = f"final_lrso_rank{rank}_epoch{epochs}_seed{s}.pt"
        path = save_dir / name
        torch.save(m.state_dict(), path)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"seed": s, "file": str(path), "sha256": h})
    return records


# --------------------------------------------------------------------------- #
# external scoring (locked outcome access)
# --------------------------------------------------------------------------- #
def _load_profiles(rdat_dir: Path) -> tuple[dict[str, dict], dict[str, list[str]], dict[str, str]]:
    """Parse the 3 direct_external RDAT files; return (profiles_by_name,
    seqpos_by_dataset, name_to_dataset). Outcome access happens here."""
    from reactflow.delta.rdat import parse_rdat
    by_name: dict[str, dict] = {}
    seqpos_by: dict[str, list[str]] = {}
    name_to_dataset: dict[str, str] = {}
    for cid in P4_DATASETS:
        r = parse_rdat(rdat_dir / f"{cid}.rdat")
        seqpos_by[cid] = list(r["seqpos"])
        for x in r["profiles"]:
            by_name[x["profile_name"]] = x
            name_to_dataset[x["profile_name"]] = cid
    return by_name, seqpos_by, name_to_dataset


def _load_frozen_graph(components_path: Path) -> list[dict]:
    """Load the frozen, outcome-blind external component graph (24 anchors).
    Fails closed unless the count matches K_PREACCESS_EXPECTED (no access)."""
    doc = json.loads(components_path.read_text(encoding="utf-8"))
    comps = doc["direct_external"]["components"]
    if len(comps) != K_PREACCESS_EXPECTED:
        raise RuntimeError(
            f"frozen graph mismatch: expected {K_PREACCESS_EXPECTED} components, "
            f"got {len(comps)} -> STOP (no outcome access)")
    return comps


def _score_component(models, comp: dict, profiles_by_name: dict[str, dict],
                     seqpos: list[str], device: str) -> tuple[dict | None, dict | None]:
    """Score one frozen component under the locked attrition rules with the
    final LRSO mixture.

    Shared-region positions are in full-sequence coordinates; the observed
    reactivity window is [off, off+W). A shared position p is scored iff
    off <= p < off+W, WT reactivity observed at p, and mutant reactivity
    observed at p. The LRSO context is the observed window, so distances are
    sequence-identical (offset cancels).
    """
    wt = profiles_by_name.get(comp["wt_name"])
    if wt is None:
        return None, {"wt_name": comp["wt_name"], "rule": 1, "status": "DROP",
                      "reason": "wt profile missing"}
    react_arr = np.asarray(wt["reactivity"], dtype=float)
    W = len(react_arr)
    if W == 0:
        return None, {"wt_name": comp["wt_name"], "rule": 1, "status": "DROP",
                      "reason": "empty wt reactivity"}
    off, _end = window_bounds(wt, seqpos)
    obs = ~np.isnan(react_arr)
    wt_median = float(np.nanmedian(react_arr)) if obs.any() else 0.0
    ctx = build_external_ctx(wt, seqpos, device)
    wt_filled = np.where(obs, react_arr,
                         float(np.nanmean(react_arr[obs])) if obs.any() else 0.0)

    # collect valid mutants (attrition rule 3 per-mutant filter)
    valid = []
    for m in comp["mutants"]:
        mu = profiles_by_name.get(m["name"])
        if mu is None:
            continue
        mut_react = np.asarray(mu["reactivity"], dtype=float)
        if len(mut_react) != W:
            continue
        edit_k = int(m["edit_pos"]) - off
        if edit_k < 0 or edit_k >= W:
            continue
        qpos = [int(p) - off for p in m["shared_region"]
                if off <= int(p) < off + W and obs[int(p) - off]
                and not np.isnan(mut_react[int(p) - off])]
        if len(qpos) < MIN_SHARED_NONMISSING:
            continue
        valid.append((m, mu, edit_k, qpos))
    if len(valid) < MIN_SCORED_MUTANTS:
        return None, {"wt_name": comp["wt_name"], "rule": 2, "status": "DROP",
                      "n_matched": len(comp["mutants"]), "n_scored": len(valid)}

    B = len(valid)
    edit_idx = torch_tensor(np.array([v[2] for v in valid], dtype=np.int64), device)
    dists = (torch_tensor(np.arange(W, dtype=np.float32), device)[None, :]
             - edit_idx[:, None]).float()
    refs = [_ref_alt(v[0]["name"])[0] for v in valid]
    alts = [_ref_alt(v[0]["name"])[1] for v in valid]
    obs_t = torch_tensor(obs, device)[None, :].expand(B, -1)

    preds_list, scales_list = [], []
    for mod in models:
        mod.eval()
        with torch.no_grad():
            H = mod.encode(ctx)
            delta, scale = mod.forward_op(H, edit_idx, dists, refs, alts, obs_t)
            pred = torch_tensor(wt_filled, device)[None, :] + delta
            preds_list.append(pred.cpu().numpy())
            scales_list.append(scale.cpu().numpy())

    c_lrso, c_zero, c_median = [], [], []
    n_positions = 0
    for bi, (m, mu, _edit_k, qpos) in enumerate(valid):
        mut_react = np.asarray(mu["reactivity"], dtype=float)
        q = np.asarray(qpos, dtype=np.int64)
        # vectorized CRPS over the mutant's qualified window positions
        locs = [preds_list[s][bi][q] for s in range(len(models))]
        scs = [scales_list[s][q] for s in range(len(models))]
        y = mut_react[q]
        c_lrso.extend(_mixture_crps_vec(locs, scs, y).tolist())
        c_zero.extend(_crps_gaussian_vec(react_arr[q],
                                         np.full(len(q), FIXED_SCALE), y).tolist())
        c_median.extend(_crps_gaussian_vec(np.full(len(q), wt_median),
                                           np.full(len(q), FIXED_SCALE), y).tolist())
        n_positions += len(q)

    return {
        "wt_name": comp["wt_name"],
        "n_matched": len(comp["mutants"]),
        "n_scored": len(valid),
        "n_positions": n_positions,
        "crps_lrso": float(np.mean(c_lrso)),
        "crps_zero": float(np.mean(c_zero)),
        "crps_median": float(np.mean(c_median)),
        "D_vs_zero": float(np.mean(c_zero) - np.mean(c_lrso)),
        "D_vs_median": float(np.mean(c_median) - np.mean(c_lrso)),
    }, None


def _ci_two_sided(x: list[float]) -> dict:
    n = len(x)
    if n < 2:
        return {"n": n, "mean": None, "ci_low": None, "ci_high": None}
    arr = np.asarray(x, float)
    m = float(arr.mean()); s = float(arr.std(ddof=1))
    t = _st.t.ppf(1 - 0.025, n - 1)
    return {"n": n, "mean": m, "sd": s,
            "ci_low": m - t * s / np.sqrt(n), "ci_high": m + t * s / np.sqrt(n)}


def aggregate_clusters(comp_rows: list[dict]) -> dict:
    """Cluster-macro aggregation at the study level (K_joint = 2).

    cluster_macro_D = mean of component D_vs_zero within the cluster; primary
    exploratory CI is the two-sided 95% t-CI over the K cluster means. LOSO
    reports the leave-one-cluster-out means."""
    clusters: dict[str, list[dict]] = {"study_sl5": [], "study_ribonanza": []}
    unknown = []
    for row in comp_rows:
        # row carries its dataset (attached by caller)
        ds = row.get("dataset", "")
        study = STUDY_BY_DATASET.get(ds)
        if study is None:
            unknown.append(row["wt_name"])
            continue
        clusters[study].append(row)

    cluster_d = {}
    for study, rows in clusters.items():
        vals = [r["D_vs_zero"] for r in rows]
        cluster_d[study] = {
            "n_components": len(rows),
            "mean_D_vs_zero": float(np.mean(vals)) if vals else None,
            "component_D_vs_zero": vals,
        }

    means = [c["mean_D_vs_zero"] for c in cluster_d.values()
             if c["mean_D_vs_zero"] is not None]
    ci = _ci_two_sided(means) if len(means) >= 2 else {
        "n": len(means), "mean": float(np.mean(means)) if means else None,
        "ci_low": None, "ci_high": None}

    # LOSO (K=2 => each leave-one-out set is the single other cluster)
    loso = {}
    for study in clusters:
        others = [cluster_d[s]["mean_D_vs_zero"] for s in cluster_d
                  if s != study and cluster_d[s]["mean_D_vs_zero"] is not None]
        loso[study] = {"leave_out_mean_D_vs_zero": float(np.mean(others)) if others else None,
                       "n_left": len(others)}

    return {
        "K_joint": len(clusters),
        "N_study": len(clusters),
        "cluster_macro_D_vs_zero": cluster_d,
        "cluster_level_ci": ci,
        "loso": loso,
        "unknown_study_components": unknown,
        "exploratory_note": (
            "K_joint=2 < K_required_planned=9 => cluster-level inference is "
            "EXPLORATORY only; no confirmatory claim authorized."),
    }


def run_p4_lrso(rdat_dir: Path, dev_csv: Path, components_path: Path,
                out: Path, *, device: str = "cpu", rank: int = FINAL_RANK,
                max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
                save_models_dir: Path | None = None) -> dict:
    # 1) outcome-blind: train the final LRSO on ALL development data
    univ = M2Universe(dev_csv)
    univ.build()
    all_records = univ.get_records()
    models, meta = train_final_lrso(univ, all_records, device, rank=rank,
                                    max_epochs=max_epochs, patience=patience)
    model_records = save_models(models, save_models_dir, rank, meta["epochs"])

    # 2) load frozen outcome-blind component graph
    comps = _load_frozen_graph(components_path)

    # 3) locked outcome access: parse external RDAT
    profiles_by_name, seqpos_by, name_to_dataset = _load_profiles(rdat_dir)

    # dataset attribution by WT identity presence in each rdat (outcome-blind)
    for c in comps:
        c["dataset"] = name_to_dataset.get(c["wt_name"], "")
    counts = {cid: sum(1 for c in comps if c["dataset"] == cid) for cid in P4_DATASETS}
    expected = {"M2SL5_2A3_0000": 3, "M3SARS_2A3_0000": 3, "15KLIB_2A3_0000": 18}
    if counts != expected:
        raise RuntimeError(
            f"dataset attribution mismatch (frozen graph drift): {counts} != {expected}; "
            f"STOP before scoring")

    comp_rows, attrition = [], []
    for comp in comps:
        row, drop = _score_component(models, comp, profiles_by_name,
                                     seqpos_by[comp["dataset"]], device)
        if drop is not None:
            attrition.append(drop)
            continue
        assert row is not None
        row["dataset"] = comp["dataset"]
        comp_rows.append(row)

    K_eff = len(comp_rows)
    D_zero = np.array([c["D_vs_zero"] for c in comp_rows])
    D_med = np.array([c["D_vs_median"] for c in comp_rows])

    cluster = aggregate_clusters(comp_rows)

    report = {
        "schema_version": "reactflow_delta.p4_external_lrso.v1",
        "candidate": "final LRSO (rank=2, 5-seed equal-weight Gaussian mixture, "
                     "frozen cfg lr=1e-3/wd=0/Student-t, dev-inner-selected epoch)",
        "baseline": "ZeroResponse (WT-anchor prediction, fixed scale 0.3)",
        "supersedes_ridge": ("legacy run_p4_external_v1 (reg_direct/ridge) used "
                             "index-0 reactivity alignment; audit finding P4-M1 "
                             "confirms seqpos starts at X27 (offset ~26) so the "
                             "legacy features/positions were misaligned. This run "
                             "uses seqpos-correct alignment."),
        "frozen_settings": {
            "rank": rank, "cfg": FROZEN_CFG, "seeds": SEEDS,
            "K_required_planned": K_REQUIRED_PLANNED,
            "K_preaccess": K_PREACCESS_EXPECTED,
            "min_shared_nonmissing": MIN_SHARED_NONMISSING,
            "min_scored_mutants": MIN_SCORED_MUTANTS,
            "fixed_scale": FIXED_SCALE,
            "final_model": meta,
        },
        "saved_models": model_records,
        "K_preaccess": K_PREACCESS_EXPECTED,
        "K_eff_realized": K_eff,
        "attrition": attrition,
        "component_rows": comp_rows,
        "component_level_ci_zero": _ci_two_sided(D_zero.tolist()),
        "component_level_ci_median": _ci_two_sided(D_med.tolist()),
        "cluster_level": cluster,
        "locked_outcome_access_count": 1,
        "locked_outcome_note": (
            "single locked LRSO external evaluation; external RDAT reactivity "
            "opened once after model training and frozen-graph load"),
        "verdict": "DEVELOPMENT_REPLICATION_EXPLORATORY",
        "confirmatory_eligibility": "NOT_ESTABLISHED",
        "practical_importance": "NOT_ESTABLISHED",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    summary = {k: v for k, v in report.items()
               if k not in ("component_rows", "attrition")}
    print(json.dumps(summary, indent=2, default=str))
    print("\n--- component_rows ---")
    print(json.dumps(comp_rows, indent=2, default=str))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Final LRSO external cluster validation")
    ap.add_argument("--rdat-dir", required=True)
    ap.add_argument("--dev-csv", required=True)
    ap.add_argument("--components", required=True,
                    help="frozen outcome-blind component manifest (p4_external_components.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rank", type=int, default=FINAL_RANK)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--save-models-dir", default=None,
                    help="persist the 5 final-model state_dicts (path+sha256 in report)")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args(argv)
    if args.compile:
        import run_p3_lrso_v3 as _p3
        _p3._COMPILE_FLAG = True
    run_p4_lrso(Path(args.rdat_dir), Path(args.dev_csv), Path(args.components),
                Path(args.out), device=args.device, rank=args.rank,
                max_epochs=args.max_epochs, patience=args.patience,
                save_models_dir=Path(args.save_models_dir) if args.save_models_dir else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
