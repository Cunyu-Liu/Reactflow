#!/usr/bin/env python3
"""M0-X dev12 feature-importance + distribution-shift audit (read-only).

Diagnoses WHY the delta/scale regression head yields NEGATIVE burden corr:
  1. Feature importance (Grad x Input over pooled eligible positions) on the
     trained best checkpoint.
  2. Per-feature correlation with the true delta/scale target (train vs val).
  3. Train vs val distribution of continuous burden (|delta|/scale) -- shift.
  4. Model-margin diagnosis: corr of |delta_r_hat| and signed delta_r_hat vs
     true scaled delta across positions and across pairs.

No training, no writes to frozen artifacts.  Outputs JSON to --out-dir.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path.cwd() / "src"))
if (_HERE.parents[2] / "src").exists():
    sys.path.insert(0, str(_HERE.parents[2] / "src"))

import m0x_epro_dev06 as dev06  # noqa: E402
import m0x_epro_dev12_regression as dev12  # noqa: E402
from b0x_baselines import run_baseline, _pair_scale  # noqa: E402
from b0x_data import load_pairs, split_groups  # noqa: E402

# Feature block layout (42 dims)
B0X_NAMES = (
    ["wt_reactivity", "dist_to_mut"]
    + [f"ctx{i}_{b}" for i in range(-2, 3) for b in "ACGU"]
    + [f"ref_{b}" for b in "ACGU"]
    + [f"alt_{b}" for b in "ACGU"]
    + ["is_mutation"])
DT_NAMES = [f"delta_thermo_{i}" for i in range(5)]
SA_NAMES = [f"struct_aware_{i}" for i in range(6)]
FEATURE_NAMES = B0X_NAMES + DT_NAMES + SA_NAMES
B0X_I = list(range(31))
DT_I = list(range(31, 36))
SA_I = list(range(36, 42))


def _spear(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _grad_importance(model, Xt, device):
    """Grad x Input magnitude per feature (mean |grad*in| over positions)."""
    model.eval()
    Xt = Xt.detach().clone().requires_grad_(True)
    out = model(Xt)
    # importance = mean over positions of |d(out)/dx_i * x_i|
    g = torch.autograd.grad(out.sum(), Xt, create_graph=False)[0]
    gi = (g * Xt).abs().mean(dim=0).detach().cpu().numpy()
    return gi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"train", "validation"})
    groups = split_groups(pairs)
    train = groups.get("train", [])
    val = groups.get("validation", [])
    print(f"[data] train={len(train)} val={len(val)}", flush=True)

    wt_cache, mut_cache = {}, {}
    train_recs = dev06._build_pair_records(train, wt_cache, mut_cache)
    val_recs = dev06._build_pair_records(val, wt_cache, mut_cache)

    # Attach standardized delta target (delta/scale) to records.
    for r, p in zip(train_recs, train):
        r["delta"] = dev12._standardized_delta_target(p, r["mask"], _pair_scale(p))
    for r, p in zip(val_recs, val):
        r["delta"] = dev12._standardized_delta_target(p, r["mask"], _pair_scale(p))

    Xtr, ytr = dev12._pool_delta_target(train_recs)
    Xva, yva = dev12._pool_delta_target(val_recs)
    print(f"[data] pooled train pos={Xtr.shape[0]} val pos={Xva.shape[0]}", flush=True)

    # ---- 1. Load best model ----
    model = dev12.DeltaMagnitudeRegressor(feat_dim=Xtr.shape[1])
    model.load_state_dict(torch.load(str(args.model_dir / "best_model.pt"),
                                     map_location=device))
    model.to(device)
    model.eval()
    print(f"[model] params={model.param_count():,}", flush=True)

    # ---- 2. Grad x Input importance (train pooled) ----
    # subsample to keep autograd tractable
    rng = np.random.default_rng(20260807)
    n_gi = min(Xtr.shape[0], 200000)
    idx = rng.choice(Xtr.shape[0], n_gi, replace=False)
    Xt = torch.tensor(Xtr[idx], device=device)
    gi = _grad_importance(model, Xt, device)
    gi_rank = np.argsort(-gi)
    imp = [{"feature": FEATURE_NAMES[i], "dim": i, "grad_x_input": float(gi[i])}
           for i in range(len(FEATURE_NAMES))]
    imp_sorted = sorted(imp, key=lambda d: -d["grad_x_input"])
    block_imp = {
        "b0x_31d": float(gi[B0X_I].sum()),
        "delta_thermo_5d": float(gi[DT_I].sum()),
        "struct_aware_6d": float(gi[SA_I].sum()),
    }

    # ---- 3. Per-feature corr with standardized target (train & val) ----
    feat_corr = {}
    for i in range(len(FEATURE_NAMES)):
        feat_corr[FEATURE_NAMES[i]] = {
            "dim": i,
            "corr_target_train": _spear(Xtr[:, i], ytr),
            "corr_target_val": _spear(Xva[:, i], yva),
        }

    # ---- 4. Train vs val burden distribution (|delta|/scale per pair) ----
    def pair_burden(pairs, recs):
        out = []
        for p, r in zip(pairs, recs):
            mask = np.asarray(p.mask, dtype=bool)
            sc = _pair_scale(p)
            d = np.asarray(p.delta, dtype=float)
            elig = mask & np.isfinite(d)
            if elig.sum() == 0:
                continue
            out.append({"pair_id": p.pair_id, "study": p.study,
                        "parent": p.parent,
                        "burden": float(np.mean(np.abs(d[elig]) / sc))})
        return out

    tr_burden = pair_burden(train, train_recs)
    va_burden = pair_burden(val, val_recs)
    tr_arr = np.array([b["burden"] for b in tr_burden])
    va_arr = np.array([b["burden"] for b in va_burden])
    dist = {
        "train": {"n": int(len(tr_arr)), "mean": float(tr_arr.mean()),
                  "std": float(tr_arr.std()), "median": float(np.median(tr_arr)),
                  "p10": float(np.percentile(tr_arr, 10)),
                  "p90": float(np.percentile(tr_arr, 90))},
        "val": {"n": int(len(va_arr)), "mean": float(va_arr.mean()),
                "std": float(va_arr.std()), "median": float(np.median(va_arr)),
                "p10": float(np.percentile(va_arr, 10)),
                "p90": float(np.percentile(va_arr, 90))},
        "kl_train_vs_val": None,
        "ttest_p": None,
    }
    # Kolmogorov-Smirnov-like via quantile overlap
    if len(tr_arr) and len(va_arr):
        # simple KS statistic
        from scipy import stats as _st
        try:
            ks, p = _st.ks_2samp(tr_arr, va_arr)
            dist["ks_stat"] = float(ks)
            dist["ks_p"] = float(p)
        except Exception as e:  # noqa
            dist["ks_err"] = str(e)

    # ---- 5. Model-margin diagnosis ----
    # Per-position: model |delta_r_hat| vs true |delta|/scale
    pr_tr, pt_tr = [], []
    pr_va, pt_va = [], []
    signed_tr, signed_va = [], []
    with torch.no_grad():
        for r in train_recs:
            x = torch.tensor(r["features"], device=device)
            mu = model(x).cpu().numpy()
            m = np.asarray(r["mask"], dtype=bool)
            d = np.asarray(r["delta"])
            for i in range(len(m)):
                if m[i]:
                    pr_tr.append(abs(float(mu[i])))
                    pt_tr.append(abs(float(d[i])))
                    signed_tr.append(float(mu[i]))
        for r in val_recs:
            x = torch.tensor(r["features"], device=device)
            mu = model(x).cpu().numpy()
            m = np.asarray(r["mask"], dtype=bool)
            d = np.asarray(r["delta"])
            for i in range(len(m)):
                if m[i]:
                    pr_va.append(abs(float(mu[i])))
                    pt_va.append(abs(float(d[i])))
                    signed_va.append(float(mu[i]))
    margin = {
        "train_pos": {"n": len(pr_tr),
                      "spearman_abs_vs_true": _spear(pr_tr, pt_tr),
                      "spearman_signed_vs_true": _spear(signed_tr, pt_tr)},
        "val_pos": {"n": len(pr_va),
                    "spearman_abs_vs_true": _spear(pr_va, pt_va),
                    "spearman_signed_vs_true": _spear(signed_va, pt_va)},
    }

    report = {
        "report": "m0x_dev12_feature_analysis.v1",
        "model_dir": str(args.model_dir),
        "feature_dim": len(FEATURE_NAMES),
        "feature_importance_top20": imp_sorted[:20],
        "feature_importance_block": block_imp,
        "feature_target_corr": feat_corr,
        "burden_distribution": dist,
        "model_margin": margin,
    }
    (out_dir / "feature_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] -> {out_dir/'feature_analysis.json'}", flush=True)
    print("=== GradxInput block ===", json.dumps(block_imp, indent=1))
    print("=== burden dist ===", json.dumps(dist, indent=1))
    print("=== model margin ===", json.dumps(margin, indent=1))
    print("=== top10 features ===")
    for d in imp_sorted[:10]:
        print(f"  {d['feature']:28s} {d['grad_x_input']:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())