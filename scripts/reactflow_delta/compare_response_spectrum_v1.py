#!/usr/bin/env python3
"""compare_response_spectrum_v1 — publication-level evaluation of the full-spectrum
response experiment from run_response_spectrum_v1.

Reads the keyed per-position-spectrum predictions and reports, for the magnitude
FULL-SPECTRUM task (independent of primary):

  * per-position conditional WMAE skill of wmae_mlp_spectrum vs wmed_spectrum
    (wmed_spectrum = per-window-position train-changer weighted median, sequence-free)
  * publication-block bootstrap CI and publication-block permutation p (vectorized)
  * per-publication effects
  * NOISE-CEILING diagnostic: the pooled mean |y| per (publication, position),
    i.e. the irreducible label-noise floor.

Statistical unit = publication (block).  Window positions are pooled within a
publication but the block (publication) is the exchangeable unit for CI/permutation,
mirroring evaluate_v5.  Fail-closed: <3 publications -> UNIDENTIFIABLE; constant
spectrum -> UNIDENTIFIABLE.

The permutation/bootstrap are implemented with vectorized numpy over pre-grouped
publication blocks to make the ~77k-position pooled statistic tractable.

No confirmatory outcome is read.  Development-only.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v2 import UNIDENTIFIABLE, is_unidentifiable


def _load_rows(pred_path):
    rows = []
    for line in Path(pred_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _wmae(y, w, pred):
    num = float(np.sum(w * np.abs(y - pred)))
    den = float(np.sum(w))
    return num / den if den > 0 else float("nan")


def _unroll_rows(rows, model_variant="wmae_mlp_spectrum"):
    """Return (model_pids, baseline_pids) dicts: pair_id -> {fold, y, w, pred} arrays.

    Unrolls CALLED magnitude_spectrum rows into per-position numpy arrays, keeping
    only positions with weight==1 (eligible + finite).  `model_variant` selects which
    neural family is treated as the model (default wmae_mlp_spectrum; use
    wmae_resid_spectrum for the residual-learning run).
    """
    model = {}   # (pair, fold, seed) -> arrays for the selected model variant
    base = {}    # (pair, fold)      -> arrays for wmed_spectrum (seed 0)
    for r in rows:
        if r["task"] != "magnitude_spectrum" or r["coverage_status"] != "CALLED":
            continue
        yv = r.get("y") or []
        wv = r.get("weight") or []
        pv = r.get("raw_prediction")
        if not (isinstance(yv, list) and isinstance(wv, list) and isinstance(pv, list)):
            continue
        if r["model_variant"] == "wmed_spectrum" and r["seed"] == 0:
            y = [float(a) for a, ww in zip(yv, wv) if ww]
            base[(r["pair_id"], r["fold_id"])] = {
                "fold": r["fold_id"],
                "y": np.array(y, dtype=np.float64),
                "w": np.ones(len(y), dtype=np.float64),
                "pred": np.array([float(p) for p, ww in zip(pv, wv) if ww], dtype=np.float64),
            }
        elif r["model_variant"] == model_variant:
            y = [float(a) for a, ww in zip(yv, wv) if ww]
            model[(r["pair_id"], r["fold_id"], r["seed"])] = {
                "fold": r["fold_id"],
                "y": np.array(y, dtype=np.float64),
                "w": np.ones(len(y), dtype=np.float64),
                "pred": np.array([float(p) for p, ww in zip(pv, wv) if ww], dtype=np.float64),
            }
    return model, base


def _skill_from_blocks(yb, wb, mb, bb):
    """Skill = 1 - WMAE(y, model_pred) / WMAE(y, baseline_pred).

    mb = model prediction blocks ; bb = baseline prediction blocks (both aligned
    with yb / wb).  Correct conditional-WMAE skill: measures how much the model's
    raw predictions beat the sequence-free per-position median baseline.
    """
    y = np.concatenate(yb); w = np.concatenate(wb)
    m = np.concatenate(mb); b = np.concatenate(bb)
    wmae_m = _wmae(y, w, m)
    wmae_b = _wmae(y, w, b)
    if not np.isfinite(wmae_b) or wmae_b <= 0.0:
        return None, wmae_m, wmae_b
    return 1.0 - wmae_m / wmae_b, wmae_m, wmae_b


def spectrum_analysis(model, base, seed, rng, n_perm=200, n_boot=200, exclude_pub=None):
    """Model residual skill vs sequence-free baseline, publication-block CI+perm p."""
    groups = defaultdict(lambda: {"y": [], "w": [], "m": [], "b": []})
    for (pid, fold, s), d in model.items():
        if s != seed:
            continue
        if exclude_pub is not None and d["fold"] == exclude_pub:
            continue
        b = base.get((pid, fold))
        if b is None or len(b["y"]) != len(d["y"]):
            continue
        groups[d["fold"]]["y"].append(d["y"])
        groups[d["fold"]]["w"].append(d["w"])
        groups[d["fold"]]["m"].append(d["pred"])
        groups[d["fold"]]["b"].append(b["pred"])
    pub_ids = list(groups.keys())
    if len(pub_ids) < 3:
        return {"n_publications": len(pub_ids), "n_positions": 0,
                "skill": None, "ci_low": None, "ci_high": None,
                "permutation_p": None, "note": "PUBLICATION_LT_3_NO_CONFIRMATORY_CI"}
    yb = [np.concatenate(groups[p]["y"]) for p in pub_ids]
    wb = [np.concatenate(groups[p]["w"]) for p in pub_ids]
    mb = [np.concatenate(groups[p]["m"]) for p in pub_ids]
    bb = [np.concatenate(groups[p]["b"]) for p in pub_ids]
    n_pos = sum(len(a) for a in yb)

    real, wmae_m, wmae_b = _skill_from_blocks(yb, wb, mb, bb)
    if real is None:
        return {"n_publications": len(pub_ids), "n_positions": n_pos,
                "skill": None, "wmae_model": None, "wmae_baseline": None,
                "permutation_p": None, "note": "SKILL_UNIDENTIFIABLE_OR_BASE_ZERO"}

    # ---- publication-block bootstrap CI ----
    rng_b = np.random.default_rng(seed)
    boot_skills = []
    idx = np.arange(len(pub_ids))
    for _ in range(n_boot):
        sel = rng_b.choice(idx, size=len(pub_ids), replace=True)
        sk, _, _ = _skill_from_blocks([yb[i] for i in sel], [wb[i] for i in sel],
                                      [mb[i] for i in sel], [bb[i] for i in sel])
        if sk is not None:
            boot_skills.append(sk)
    if boot_skills:
        lo = float(np.percentile(boot_skills, 2.5))
        hi = float(np.percentile(boot_skills, 97.5))
    else:
        lo = hi = None

    # ---- publication-block permutation p ----
    # permute model-prediction blocks across publications (labels y and baseline b
    # stay fixed); skill under null = (b+1)/(B+1).
    b_cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(idx)
        sk, _, _ = _skill_from_blocks(yb, wb, [mb[i] for i in perm], bb)
        if sk is not None and sk >= real:
            b_cnt += 1
    p = (b_cnt + 1) / (n_perm + 1)

    return {"n_publications": len(pub_ids), "n_positions": n_pos,
            "skill": float(real), "wmae_model": float(wmae_m),
            "wmae_baseline": float(wmae_b), "ci_low": lo, "ci_high": hi,
            "permutation_p": float(p), "n_perm": n_perm, "n_boot": n_boot}


def noise_ceiling(base):
    """Pooled per-(publication,position) mean |y| = irreducible label-noise floor."""
    by = defaultdict(list)
    for (pid, fold), d in base.items():
        for v in d["y"]:
            by[fold].append(abs(float(v)))
    all_abs = [x for lst in by.values() for x in lst]
    if not all_abs:
        return {"noise_floor_mean_abs": None, "n_positions": 0, "by_publication": {}}
    return {"noise_floor_mean_abs": float(np.mean(all_abs)),
            "n_positions": len(all_abs),
            "by_publication": {k: float(np.mean(v)) for k, v in by.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dominant-pub", default="pmid_29446752")
    ap.add_argument("--model-variant", default="wmae_mlp_spectrum",
                    choices=["wmae_mlp_spectrum", "wmae_resid_spectrum"])
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--perm-seed", type=int, default=20260812)
    args = ap.parse_args()

    rows = _load_rows(args.pred)
    model, base = _unroll_rows(rows, model_variant=args.model_variant)
    out = Path(args.out)

    report = {
        "schema": "reactflow_delta.response_spectrum_comparison.v1",
        "authority_epoch": 20, "endpoint": "endpoint_v6", "caller_version": "caller_v4",
        "model_variant": args.model_variant,
        "n_rows": len(rows), "n_model_pairs": len(set(k[:2] for k in model)),
        "n_base_pairs": len(base), "models": {},
    }
    rng = np.random.default_rng(args.perm_seed)
    for seed in [0, 1, 2, 3, 4]:
        a = spectrum_analysis(model, base, seed, rng, n_perm=args.n_perm,
                              n_boot=args.n_boot)
        a_loo = spectrum_analysis(model, base, seed, rng, n_perm=args.n_perm,
                                  n_boot=args.n_boot, exclude_pub=args.dominant_pub)
        report["models"][f"{args.model_variant}:{seed}"] = {
            "full": a, "without_dominant({})".format(args.dominant_pub): a_loo,
        }
    report["noise_ceiling"] = noise_ceiling(base)

    (out / "response_spectrum_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE -> {out / 'response_spectrum_comparison.json'}")
    for seed in [0, 1, 2, 3, 4]:
        a = report["models"][f"{args.model_variant}:{seed}"]["full"]
        print(f"SEED {seed} skill={a['skill']} ci=({a['ci_low']},{a['ci_high']}) "
              f"perm_p={a['permutation_p']} n_pos={a['n_positions']} n_pub={a['n_publications']}")


if __name__ == "__main__":
    sys.exit(main())