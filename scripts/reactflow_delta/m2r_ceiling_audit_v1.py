#!/usr/bin/env python3
"""m2r_ceiling_audit_v1.py — stress-test the M2R honest-ceiling claim.

MOTIVATION (method-level, fail-closed repair):
The chapter claims "legal-feature representation saturates near R2 0.37-0.41"
because a CIRCULAR oracle (which legally adds the double-mutant profile as
features) only reaches R2 0.407 with a GBDT.  That 0.407 was computed with the
SAME weak default GBDT (100 trees, depth 3) used everywhere — and the number has
NO backing script (it only lives hardcoded in the submission table).

This audit answers two decisive questions with a proper design-level LOO:

  Q1 (capacity artifact?):  Does a STRONG model (tuned LightGBM / XGBoost /
      ExtraTrees, more trees + early stopping, deeper, subsampling) on the
      circular oracle jump far above 0.407?  If yes, the "representation
      saturates near 0.41" claim is a WEAK-MODEL artifact — the legal features
      have real headroom the default GBDT leaves on the table.

  Q2 (legal feature lever?):  Does adding the LEGAL design-region aggregate
      features (the exact denominator of the rescue formula,
      sqrt(RMSD(sA,wt)^2 + RMSD(sB,wt)^2)) improve the model beyond the
      current +26.59% / R2 0.370 headline?

Cells (design-level LOO, exchangeable unit = design):
  legal-236          default GBDT            (baseline, reproduces chapter)
  legal-236+dr       default GBDT            (Q2 with default model)
  legal-236          strong GBDT             (Q1 legal with strong model)
  legal-236+dr       strong GBDT             (Q2 with strong model)
  oracle (legal+dbl) default GBDT            (reproduces ~0.407)
  oracle (legal+dbl) strong GBDT             (Q1 decisive: capacity artifact?)
  oracle (legal+dbl) XGBoost (if avail)      (independent strong check)

Output: report JSON + per-cell OOF for later significance testing.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_features_v1 as m2rf
import m2r_design_region_features_v1 as drf

SEED = 20260817


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _skill(mae_model, mae_bl):
    return 1.0 - mae_model / mae_bl if mae_bl > 0 else 0.0


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _loo_factory(X, y, keys, des_list, model_name: str, **model_kw):
    """Return (preds,) by running a design-level LOO with the given model.

    model_name in {"gbdt_default", "gbdt_strong", "xgb_strong", "et_strong"}.
    Returns an (n,) prediction array.
    """
    preds = np.zeros(len(y))
    for held in des_list:
        m = keys != held
        if m.sum() <= 10:
            preds[~m] = np.median(y)
            continue
        pred = _fit_predict(model_name, X[m], y[m], X[~m], **model_kw)
        preds[~m] = pred
    return preds


def _fit_predict(model_name, Xtr, ytr, Xte, **kw):
    if model_name == "gbdt_default":
        import lightgbm as lgb
        g = lgb.LGBMRegressor(n_estimators=100, max_depth=3,
                              random_state=SEED, verbose=-1, n_jobs=2)
        g.fit(Xtr, ytr)
        return g.predict(Xte)
    if model_name == "gbdt_strong":
        import lightgbm as lgb
        g = lgb.LGBMRegressor(
            n_estimators=kw.get("n_estimators", 800),
            max_depth=kw.get("max_depth", -1),
            num_leaves=kw.get("num_leaves", 63),
            learning_rate=kw.get("learning_rate", 0.03),
            min_child_samples=kw.get("min_child_samples", 20),
            subsample=kw.get("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=kw.get("colsample_bytree", 0.8),
            reg_alpha=kw.get("reg_alpha", 0.1),
            reg_lambda=kw.get("reg_lambda", 1.0),
            random_state=SEED, verbose=-1, n_jobs=2,
        )
        g.fit(Xtr, ytr)
        return g.predict(Xte)
    if model_name == "xgb_strong":
        from xgboost import XGBRegressor
        g = XGBRegressor(
            n_estimators=kw.get("n_estimators", 800),
            max_depth=kw.get("max_depth", 8),
            learning_rate=kw.get("learning_rate", 0.03),
            min_child_weight=kw.get("min_child_weight", 3),
            subsample=kw.get("subsample", 0.8),
            colsample_bytree=kw.get("colsample_bytree", 0.8),
            reg_alpha=kw.get("reg_alpha", 0.1),
            reg_lambda=kw.get("reg_lambda", 1.0),
            random_state=SEED, n_jobs=2, verbosity=0,
        )
        g.fit(Xtr, ytr)
        return g.predict(Xte)
    if model_name == "et_strong":
        from sklearn.ensemble import ExtraTreesRegressor
        g = ExtraTreesRegressor(
            n_estimators=kw.get("n_estimators", 500),
            max_depth=kw.get("max_depth", None),
            min_samples_leaf=kw.get("min_samples_leaf", 3),
            random_state=SEED, n_jobs=2,
        )
        g.fit(Xtr, ytr)
        return g.predict(Xte)
    raise ValueError(f"unknown model {model_name}")


def run_cells(X_legal, X_dr, X_oracle, y, keys, args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    des_list = sorted(set(keys.tolist()))
    y_med = np.median(y)
    mae_bl = _mae(y, np.full_like(y, y_med))
    n = len(y)

    X_dr_full = np.concatenate([X_legal, X_dr], axis=1)          # legal + dr
    X_oracle_full = np.concatenate([X_legal, X_oracle], axis=1)  # legal + dbl
    X_oracle_dr = np.concatenate([X_legal, X_dr, X_oracle], axis=1)

    cells = {
        "legal_default": (X_legal, "gbdt_default", {}),
        "legal_dr_default": (X_dr_full, "gbdt_default", {}),
        "legal_strong": (X_legal, "gbdt_strong", {}),
        "legal_dr_strong": (X_dr_full, "gbdt_strong", {}),
        "oracle_default": (X_oracle_full, "gbdt_default", {}),
        "oracle_strong": (X_oracle_full, "gbdt_strong", {}),
        "oracle_dr_strong": (X_oracle_dr, "gbdt_strong", {}),
    }
    if args.xgb:
        cells["oracle_xgb"] = (X_oracle_full, "xgb_strong", {})

    results = {}
    oof = {}
    for name, (Xc, model, kw) in cells.items():
        t0 = time.time()
        preds = _loo_factory(Xc, y, keys, des_list, model, **kw)
        mae = _mae(y, preds)
        sk = _skill(mae, mae_bl)
        r2 = _r2(y, preds)
        results[name] = {"mae": mae, "skill": sk, "r2": r2,
                         "n_features": int(Xc.shape[1]), "model": model,
                         "wall": round(time.time() - t0, 1)}
        oof[name] = preds
        print(f"[audit] {name:18s} model={model:12s} nf={Xc.shape[1]:4d} "
              f"MAE={mae:.4f} skill={sk:+.4f} R2={r2:.4f} "
              f"wall={results[name]['wall']}s", flush=True)

    # ---- decisive comparisons ----
    d = results
    report = {
        "schema": "reactflow_delta.m2r_ceiling_audit.v1",
        "dataset": "OpenKnot_M2R",
        "exchangeable_unit": "design",
        "n_samples": n, "n_designs": len(des_list),
        "baseline_mae": mae_bl, "seed": SEED,
        "cells": results,
        "comparisons": {
            "q1_oracle_default_vs_strong": {
                "oracle_default_r2": d["oracle_default"]["r2"],
                "oracle_strong_r2": d["oracle_strong"]["r2"],
                "delta_r2": d["oracle_strong"]["r2"] - d["oracle_default"]["r2"],
                "conclusion": ("capacity_artifact"
                               if d["oracle_strong"]["r2"] > 0.45
                               else "representation_limited"),
            },
            "q2_legal_vs_legal_dr": {
                "legal_default_r2": d["legal_default"]["r2"],
                "legal_dr_default_r2": d["legal_dr_default"]["r2"],
                "delta_r2": d["legal_dr_default"]["r2"] - d["legal_default"]["r2"],
                "legal_strong_r2": d["legal_strong"]["r2"],
                "legal_dr_strong_r2": d["legal_dr_strong"]["r2"],
                "delta_r2_strong": d["legal_dr_strong"]["r2"] - d["legal_strong"]["r2"],
            },
            "q3_strong_legal_gap_to_oracle": {
                "legal_strong_r2": d["legal_strong"]["r2"],
                "oracle_strong_r2": d["oracle_strong"]["r2"],
                "gap_r2": d["oracle_strong"]["r2"] - d["legal_strong"]["r2"],
                "interpretation": (
                    "if strong-oracle >> strong-legal, the double-mutant effect "
                    "is genuinely unrecoverable from legal inputs even with a "
                    "strong model; if strong-legal approaches strong-oracle, "
                    "legal features carry most of the recoverable signal"),
            },
        },
        "chapter_claim_audit": {
            "claimed_ceiling": "R2 0.37-0.41 (legal-feature representation saturation)",
            "oracle_default_r2": d["oracle_default"]["r2"],
            "oracle_strong_r2": d["oracle_strong"]["r2"],
            "claim_supported_by_strong_model": bool(d["oracle_strong"]["r2"] <= 0.45),
            "note": ("previous 0.407 had NO backing script and used the default "
                     "GBDT; this audit runs a strong model with the same "
                     "partition and records the honest number"),
        },
    }
    (out / "m2r_ceiling_audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(out / "m2r_ceiling_audit_oof.npz", **oof, y=y, keys=keys)

    print("\n=== DECISIVE ===")
    c = report["comparisons"]
    print(f"Q1 oracle default R2={c['q1_oracle_default_vs_strong']['oracle_default_r2']:.4f} "
          f"-> strong R2={c['q1_oracle_default_vs_strong']['oracle_strong_r2']:.4f} "
          f"({c['q1_oracle_default_vs_strong']['conclusion']})")
    q2 = c["q2_legal_vs_legal_dr"]
    print(f"Q2 legal default R2={q2['legal_default_r2']:.4f} -> +dr {q2['legal_dr_default_r2']:.4f} "
          f"(+{q2['delta_r2']*100:.2f}pp); strong {q2['legal_strong_r2']:.4f} -> +dr "
          f"{q2['legal_dr_strong_r2']:.4f} (+{q2['delta_r2_strong']*100:.2f}pp)")
    q3 = c["q3_strong_legal_gap_to_oracle"]
    print(f"Q3 strong-legal R2={q3['legal_strong_r2']:.4f} vs strong-oracle "
          f"R2={q3['oracle_strong_r2']:.4f} (gap {q3['gap_r2']:.4f})")
    print(f"\nDONE -> {out / 'm2r_ceiling_audit_report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xgb", action="store_true",
                    help="also run XGBoost oracle cell (needs xgboost installed)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X, y, keys, _ = m2rf.build_all(samples)
    keys = np.array(keys)
    # design-region features need the sequence length per sample
    seq_lens = [len(s.sequence) for s in samples]
    X_dr = np.stack([drf.build_design_region_features(s, L)
                     for s, L in zip(samples, seq_lens)])
    X_oracle = np.stack([drf.build_design_region_oracle_features(s, L)
                         for s, L in zip(samples, seq_lens)])
    print(f"[audit] n_samples={len(y)} X_legal={X.shape} X_dr={X_dr.shape} "
          f"X_oracle={X_oracle.shape}", flush=True)

    run_cells(X, X_dr, X_oracle, y, keys, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
