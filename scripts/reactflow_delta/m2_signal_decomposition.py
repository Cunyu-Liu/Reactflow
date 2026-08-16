#!/usr/bin/env python3
"""m2_signal_decomposition.py — decompose the v5 model's residual to decide the
next method direction.

Background (junction project lessons):
  - r68: measured-layer residual sd (0.548) >> err10 (0.248) is NOT automatically
    "learnable headroom" — it must be decomposed by context visibility.
  - r69: OOD-context residual sd (0.694) vs train-visible (0.525) => the gap is a
    context random effect, only partly recoverable by post-hoc EB.

For M2 response-spectrum we ask the analogous questions:

  1. Feature-signal check:  does the as-yet-UNUSED M2_structure / target_structure
     (per-position dot-bracket) carry signal for the residual that the current
     features (base + WT reactivity + error) do NOT?
     -> regress v5 residual on (paired, bracket_depth) derived from M2_structure
        and target_structure, per held fold, OOF.  If OOF R2 > 0 => structure
        feature is a real, unused lever.

  2. Context-visibility decomposition (junction r69 analog):
     - train-visible cells: (design,position) cells whose similar contexts appear
       in train folds -> residual sd_small
     - OOD cells: contexts never seen -> residual sd_large
     The gap (if large) is context random effect, NOT model headroom.

  3. Per-design residual spread: how much of residual variance is a per-design
     constant shift (removable by a leak-free per-design estimate via
     leave-one-design-out), vs irreducible per-position noise.

CPU-only, reads v5 keyed predictions + M2 CSV.
"""
from __future__ import annotations

import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = [0, 1, 2, 3, 4]
BASELINE = "wmed_spectrum"
W = 21


def _load_rows(pred_path):
    rows = []
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_m2(csv_path):
    """Return design-level info: design_id -> (sequence, sub_start, sub_end,
    m2_structure, target_structure)."""
    info = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("mutA"):
                key = f"OK7a_M2_{row['puzzle']}_{row['method']}"
                info[key] = {
                    "sequence": row["sequence"],
                    "sub_start": int(row["sub_start"]),
                    "sub_end": int(row["sub_end"]),
                    "m2_structure": row.get("M2_structure") or "",
                    "target_structure": row.get("target_structure") or "",
                }
    return info


def _dot_to_depth(structure):
    """Map a dot-bracket string to per-position (paired, depth) arrays.

    paired: 1 if the char is an opening/closing bracket of any type.
    depth:  number of currently-open bracket levels at this position.
    Uses the standard mapping: (,),[,],{,} are brackets; . is unpaired.
    """
    n = len(structure)
    paired = np.zeros(n, dtype=np.float64)
    depth = np.zeros(n, dtype=np.float64)
    stack = []
    openers = "([{"
    closers = ")]}"
    for i, ch in enumerate(structure):
        if ch in openers:
            stack.append(ch)
            paired[i] = 1.0
            depth[i] = len(stack)
        elif ch in closers:
            paired[i] = 1.0
            depth[i] = len(stack)
            if stack:
                stack.pop()
        else:
            depth[i] = len(stack)
    return paired, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--variant", default="wmae_resid_attn_spectrum")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cells", type=int, default=200)
    args = ap.parse_args()

    rows = _load_rows(args.pred)
    m2info = _read_m2(args.m2_csv)

    # --- collect per-position observations: y, pred (mu-ens), residual, prior,
    #     and design-level structure features ---
    ys, ps, bs, des, positions, paired_m2, depth_m2, paired_tgt, depth_tgt = (
        [], [], [], [], [], [], [], [], [])
    for r in rows:
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        if r["model_variant"] != BASELINE and r["model_variant"] != args.variant:
            continue
        pid = r["pair_id"]
        d = pid.split(":")[0]
        info = m2info.get(d)
        if info is None:
            continue
        # per-position structure at FULL-sequence index; edit region is
        # sub_start..sub_end (1-indexed).  M2_structure covers only the design
        # region (len = sub_end - sub_start + 1 in many rows).
        seq = info["sequence"]
        sub0 = info["sub_start"] - 1  # 0-indexed design start in full seq
        m2str = info["m2_structure"]
        tgtstr = info["target_structure"]

        yv = r.get("y") or []
        wv = r.get("weight") or []
        pv = r.get("raw_prediction")
        if not (isinstance(yv, list) and isinstance(wv, list) and isinstance(pv, list)):
            continue
        if r["model_variant"] == BASELINE:
            b = np.array([float(a) for a, ww in zip(yv, wv) if ww], dtype=np.float64)
            ys_b = np.array([float(a) for a, ww in zip(yv, wv) if ww], dtype=np.float64)
            # baseline pred = prior (per-position median) — store per pair
            continue
        # model variant: accumulate seeds
    # simpler two-pass: store baseline per pair + model per (pair,seed)
    base = {}
    model = defaultdict(dict)
    for r in rows:
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        yv = r.get("y") or []
        wv = r.get("weight") or []
        pv = r.get("raw_prediction")
        if not (isinstance(yv, list) and isinstance(wv, list) and isinstance(pv, list)):
            continue
        y = np.array([float(a) for a, ww in zip(yv, wv) if ww], dtype=np.float64)
        p = np.array([float(a) for a, ww in zip(pv, wv) if ww], dtype=np.float64)
        if r["model_variant"] == BASELINE and r["seed"] == 0:
            base[r["pair_id"]] = {"y": y, "pred": p}
        elif r["model_variant"] == args.variant:
            model[r["pair_id"]][r["seed"]] = p

    # --- assemble pooled per-position records with structure features ---
    recs = []
    for pid in base:
        if len(model.get(pid, {})) < len(SEEDS):
            continue
        d = pid.split(":")[0]
        info = m2info.get(d)
        if info is None:
            continue
        ens = np.mean([model[pid][s] for s in SEEDS], axis=0)
        y = base[pid]["y"]
        priorv = base[pid]["pred"]
        if len(y) != len(ens) or len(priorv) != len(ens):
            continue
        # find the edit site position for this pair (mutA index within design)
        # pid format: <design>:<mutA>
        try:
            mutA = int(pid.rsplit(":", 1)[1])
        except ValueError:
            mutA = 0
        # design-relative index of the edit (1-indexed in M2), 0-indexed here:
        # edit_seq_pos (full seq) = sub0 + (mutA - 1)
        # we place the W=21 window centered at edit_seq_pos in FULL seq coords
        edit_full = info["sub_start"] - 1 + (mutA - 1)
        # full-sequence alignment of structures
        m2pa, m2dp = _dot_to_depth(info["m2_structure"])
        tgpa, tgdp = _dot_to_depth(info["target_structure"])
        # M2_structure may cover only the design region; align to full seq
        # try direct: if len==len(seq), already full; else assume starts at sub_start
        Lm = len(info["m2_structure"])
        if Lm == len(info["sequence"]):
            m2_al = (m2pa, m2dp)
        else:
            # assume covers design region [sub_start..sub_end]
            pad_l = info["sub_start"] - 1
            m2_al = (np.concatenate([np.zeros(pad_l), m2pa]),
                     np.concatenate([np.zeros(pad_l), m2dp]))
        # target_structure is full-length (177) in all rows
        tg_al = (tgpa, tgdp)
        for j in range(min(W, len(y))):
            idx = edit_full - (W // 2) + j
            if idx < 0 or idx >= len(info["sequence"]):
                continue
            if idx < len(m2_al[0]):
                m2_p = m2_al[0][idx]; m2_d = m2_al[1][idx]
            else:
                m2_p = m2_d = 0.0
            if idx < len(tg_al[0]):
                t_p = tg_al[0][idx]; t_d = tg_al[1][idx]
            else:
                t_p = t_d = 0.0
            recs.append({
                "y": float(y[j]), "pred": float(ens[j]), "prior": float(priorv[j]),
                "design": d, "position": j, "mutA": mutA,
                "m2_paired": m2_p, "m2_depth": m2_d,
                "tgt_paired": t_p, "tgt_depth": t_d,
            })

    if not recs:
        print("no records assembled — check design key format")
        sys.exit(1)
    y = np.array([r["y"] for r in recs])
    pred = np.array([r["pred"] for r in recs])
    prior = np.array([r["prior"] for r in recs])
    resid = y - pred
    adt = np.abs(y - prior)
    des = np.array([r["design"] for r in recs])
    pos = np.array([r["position"] for r in recs])
    m2p = np.array([r["m2_paired"] for r in recs])
    m2d = np.array([r["m2_depth"] for r in recs])
    tgtp = np.array([r["tgt_paired"] for r in recs])
    tgtd = np.array([r["tgt_depth"] for r in recs])

    def wmae(a, b):
        return float(np.mean(np.abs(a - b)))

    bw = wmae(y, prior); mw = wmae(y, pred)
    skill = 1.0 - mw / bw

    # --- 1. feature-signal check: residual ~ structure features (OOF via LOO by design) ---
    # Fit per-design-independent ridge-like (closed form) on train designs, predict held design.
    # Design matrix: [1, m2_paired, m2_depth, tgt_paired, tgt_depth] (also absolute-value version)
    Xfeat = np.stack([np.ones_like(y), m2p, m2d, tgtp, tgtd], axis=1)
    designs = sorted(set(des.tolist()))
    r2_oof = []
    for held in designs:
        m = des != held
        if m.sum() < args.min_cells:
            continue
        Xtr, ytr = Xfeat[m], resid[m]
        Xte, yte = Xfeat[~m], resid[~m]
        # ridge closed form with small ridge
        lam = 1.0
        beta = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
        pred_te = Xte @ beta
        ss_res = float(((yte - pred_te) ** 2).sum())
        ss_tot = float(((yte - yte.mean()) ** 2).sum())
        if ss_tot > 0:
            r2_oof.append(1.0 - ss_res / ss_tot)
    r2_oof = np.array(r2_oof) if r2_oof else np.array([])

    # --- 2. per-design residual constant shift, leak-free (leave-one-design-out) ---
    # Estimate per-design mean residual from OTHER designs, apply to held design.
    resid_design_means = {d: float(resid[des == d].mean()) for d in designs}
    pred_shifted = pred.copy()
    for held in designs:
        train_mean = float(np.mean([resid_design_means[d] for d in designs if d != held]))
        pred_shifted[des == held] = pred[des == held] + train_mean
    skill_shift = 1.0 - wmae(y, pred_shifted) / bw
    # in-sample per-design shift (leaky upper bound)
    pred_shift_ins = pred + np.array([resid_design_means[d] for d in des])
    skill_shift_ins = 1.0 - wmae(y, pred_shift_ins) / bw

    # --- 3. residual sd by structure-context groups ---
    grp_paired = resid[m2p > 0]
    grp_unpaired = resid[m2p == 0]
    grp_tgt_paired = resid[tgtp > 0]
    grp_tgt_unpaired = resid[tgtp == 0]

    # --- 4. residual sd by |adt| (deviation magnitude) ---
    bins = np.quantile(adt, [0.0, 0.5, 0.9, 1.0])
    adt_sd = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        sel = (adt >= lo) & (adt <= hi)
        if sel.sum() > 0:
            adt_sd.append({"bin": f"[{lo:.3f},{hi:.3f}]", "n": int(sel.sum()),
                           "resid_sd": float(resid[sel].std())})

    # --- 5. residual sd by window position ---
    pos_sd = []
    for j in range(W):
        sel = pos == j
        if sel.sum() > 0:
            pos_sd.append({"position": int(j), "n": int(sel.sum()),
                           "resid_sd": float(resid[sel].std()),
                           "mean_resid": float(resid[sel].mean())})

    report = {
        "schema": "reactflow_delta.response_spectrum.m2_signal_decomposition.v1",
        "variant": args.variant, "dataset": "OpenKnot_M2",
        "n_pairs": int(len(base)), "n_positions_pooled": int(len(y)),
        "wmae_baseline": float(bw), "wmae_model": float(mw), "skill": float(skill),
        "resid_sd_pooled": float(resid.std()),
        "feature_signal_structure": {
            "n_oof_folds": int(len(r2_oof)),
            "mean_oof_r2": float(r2_oof.mean()) if len(r2_oof) else None,
            "median_oof_r2": float(np.median(r2_oof)) if len(r2_oof) else None,
            "pct_folds_r2_gt_0": float((r2_oof > 0).mean()) if len(r2_oof) else None,
            "min_oof_r2": float(r2_oof.min()) if len(r2_oof) else None,
            "max_oof_r2": float(r2_oof.max()) if len(r2_oof) else None,
        },
        "per_design_shift_calibration": {
            "skill_raw": float(skill),
            "skill_leakfree_design_shift": float(skill_shift),
            "skill_in_sample_design_shift_leaky_upper": float(skill_shift_ins),
        },
        "structure_context_residual_sd": {
            "m2_paired_sd": float(grp_paired.std()) if len(grp_paired) else None,
            "m2_unpaired_sd": float(grp_unpaired.std()) if len(grp_unpaired) else None,
            "tgt_paired_sd": float(grp_tgt_paired.std()) if len(grp_tgt_paired) else None,
            "tgt_unpaired_sd": float(grp_tgt_unpaired.std()) if len(grp_tgt_unpaired) else None,
            "m2_paired_n": int(len(grp_paired)), "m2_unpaired_n": int(len(grp_unpaired)),
        },
        "deviation_magnitude_resid_sd": adt_sd,
        "position_resid_sd": pos_sd,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2_signal_decomposition.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== M2 signal decomposition ===")
    print(f"skill={skill:+.4f} wmae_b={bw:.4f} wmae_m={mw:.4f} resid_sd={resid.std():.4f}")
    print("\n-- structure feature OOF signal --")
    fs = report["feature_signal_structure"]
    print(f"  mean_oof_r2={fs['mean_oof_r2']} median={fs['median_oof_r2']} "
          f"pct>0={fs['pct_folds_r2_gt_0']} min={fs['min_oof_r2']} max={fs['max_oof_r2']}")
    print("\n-- per-design shift calibration --")
    ps_ = report["per_design_shift_calibration"]
    print(f"  leakfree={ps_['skill_leakfree_design_shift']:+.4f} "
          f"in-sample-leaky={ps_['skill_in_sample_design_shift_leaky_upper']:+.4f}")
    print("\n-- structure context residual sd --")
    sc = report["structure_context_residual_sd"]
    print(f"  m2_paired={sc['m2_paired_sd']} (n={sc['m2_paired_n']}) "
          f"m2_unpaired={sc['m2_unpaired_sd']} (n={sc['m2_unpaired_n']})")
    print(f"  tgt_paired={sc['tgt_paired_sd']} tgt_unpaired={sc['tgt_unpaired_sd']}")
    print("\n-- position residual sd --")
    for x in pos_sd:
        print(f"  pos{x['position']:2d} n={x['n']:6d} resid_sd={x['resid_sd']:.4f} "
              f"mean_resid={x['mean_resid']:+.4f}")
    print(f"DONE -> {out / 'm2_signal_decomposition.json'}")


if __name__ == "__main__":
    sys.exit(main())
