#!/usr/bin/env python3
"""m2r_shape_features_v1.py — SHAPE-guided (constrained) thermodynamic features.

MOTIVATION (extends the MFE modality, §3.8.6, method-level):
The MFE features fold WT / single-A / single-B / double-D with PLAIN
thermodynamics.  But the WT and single-mutant SHAPE reactivity profiles are
EXPERIMENTAL evidence that is already legal (they are inputs to the rescue
task; the target rescue_factor is computed from the DOUBLE-mutant profile,
which is the one thing we never fold with).

We therefore fold with ViennaRNA SOFT CONSTRAINTS (Deigan et al. 2009
pseudo-energies  dG_SHAPE(i) = m*ln(reactivity(i)+1) + b):

  * WT  -> fold(wt_seq,     WT SHAPE constraints)
  * A   -> fold(mutA_seq,   singleA SHAPE constraints)
  * B   -> fold(mutB_seq,   singleB SHAPE constraints)
  * D   -> fold(double_seq, NO constraints)   (double-mutant SHAPE is circular)

The resulting "experimentally-guided" structures and energies are new legal
signals that integrate the whole-molecule SHAPE profile — something the
per-site reactivity features cannot do — and the divergence between the
constrained and plain (pure-thermodynamic) folds measures how much experiment
overrides the thermodynamic prediction (a dynamics/uncertainty cue).

All features are legal: no experimental double-mutant reactivity is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_mfe_features_v1 as mfe

CACHE = {}

# Deigan SHAPE pseudo-energy parameters (kcal/mol)
SHAPE_M = 1.8
SHAPE_B = -0.6
# reactivity clipping range (data median 0.14, p95 1.42, outliers to 8 / -7)
SHAPE_LO, SHAPE_HI = 0.0, 8.0


def _clean_shape(shape) -> list[float]:
    """Normalize a SHAPE reactivity array for Deigan soft constraints."""
    out = []
    for x in shape:
        if x is None:
            out.append(0.0)
            continue
        try:
            x = float(x)
        except (TypeError, ValueError):
            out.append(0.0)
            continue
        if not np.isfinite(x):
            out.append(0.0)
        else:
            out.append(max(SHAPE_LO, min(SHAPE_HI, x)))
    return out


def _fold_constrained(seq: str, shape):
    """ViennaRNA MFE structure with Deigan SHAPE soft constraints (cached).

    The reactivity profile is deterministic given the sequence, so the cache
    is keyed by sequence only.
    """
    key = ("shapefold", seq)
    if key not in CACHE:
        import RNA
        fc = RNA.fold_compound(seq)
        vals = _clean_shape(shape)
        rc = fc.sc_add_SHAPE_deigan(vals, SHAPE_M, SHAPE_B, 0)
        if rc != 1:
            # constraints not applied (e.g. too-short sequence): plain fold
            CACHE[key] = mfe._fold(seq)
        else:
            ss, dG = fc.mfe()
            CACHE[key] = (ss, float(dG))
    return CACHE[key]


def _len_adjust(ss, n):
    return ss if len(ss) == n else ss[:n]


def build_shape_features(s) -> np.ndarray:
    """SHAPE-guided feature vector for one M2RPair (all legal)."""
    seq = s.sequence
    i, j = s.editA_seq_pos, s.editB_seq_pos
    seqA = s.mutA_seq if s.mutA_seq else seq
    seqB = s.mutB_seq if s.mutB_seq else seq
    seqD = list(seqA)
    if 0 <= j < len(seqD) and 0 <= j < len(seqB):
        seqD[j] = seqB[j]
    seqD = "".join(seqD)

    # constrained folds: WT/A/B guided by their own SHAPE; D plain
    ss_wt_s, dG_wt_s = _fold_constrained(seq, s.wt_reactivity)
    ss_a_s, dG_a_s = _fold_constrained(seqA, s.singleA_reactivity)
    ss_b_s, dG_b_s = _fold_constrained(seqB, s.singleB_reactivity)
    ss_d, dG_d = mfe._fold(seqD)

    n = len(seq)
    ss_wt_s = _len_adjust(ss_wt_s, n); ss_a_s = _len_adjust(ss_a_s, n)
    ss_b_s = _len_adjust(ss_b_s, n); ss_d = _len_adjust(ss_d, n)

    # ---- structure distances + SHAPE-guided rescue analog ----
    rA_s = mfe.pair_distance(ss_wt_s, ss_a_s)
    rB_s = mfe.pair_distance(ss_wt_s, ss_b_s)
    rD_s = mfe.pair_distance(ss_wt_s, ss_d)
    rnorm_s = np.sqrt(rA_s ** 2 + rB_s ** 2)
    rescue_s = (1.0 - rD_s / rnorm_s) if rnorm_s > 0 else 0.0

    # ---- constrained-vs-plain divergence (experiment vs thermodynamics) ----
    ss_wt_p, _ = mfe._fold(seq)
    ss_a_p, _ = mfe._fold(seqA)
    ss_b_p, _ = mfe._fold(seqB)
    ss_wt_p = _len_adjust(ss_wt_p, n); ss_a_p = _len_adjust(ss_a_p, n)
    ss_b_p = _len_adjust(ss_b_p, n)
    div_wt = mfe.pair_distance(ss_wt_p, ss_wt_s)
    div_a = mfe.pair_distance(ss_a_p, ss_a_s)
    div_b = mfe.pair_distance(ss_b_p, ss_b_s)

    # ---- constrained free energies ----
    dA_s = abs(dG_a_s - dG_wt_s); dB_s = abs(dG_b_s - dG_wt_s)
    dD_s = abs(dG_d - dG_wt_s)
    gnorm_s = dA_s + dB_s + 1e-9
    rescue_G_s = 1.0 - dD_s / gnorm_s

    # ---- (i,j) pair status in each constrained structure ----
    pa_wt = mfe.pairing_status(ss_wt_s, i, j)
    pa_a = mfe.pairing_status(ss_a_s, i, j)
    pa_b = mfe.pairing_status(ss_b_s, i, j)
    pa_d = mfe.pairing_status(ss_d, i, j)

    parts = [
        np.array([rA_s, rB_s, rD_s, rnorm_s, rescue_s]),
        np.array([div_wt, div_a, div_b]),
        np.array([dG_wt_s, dG_a_s, dG_b_s, dG_d,
                  dA_s, dB_s, dD_s, rescue_G_s]),
        np.array([pa_wt[0], pa_wt[1], pa_wt[2],
                  pa_a[0], pa_a[1], pa_a[2],
                  pa_b[0], pa_b[1], pa_b[2],
                  pa_d[0], pa_d[1], pa_d[2]]),
        np.array([pa_d[2] - max(pa_a[2], pa_b[2]),
                  pa_d[2] * (1 - pa_wt[2])]),
    ]
    return np.concatenate(parts).astype(np.float64)


def shape_feature_names() -> list[str]:
    return [
        "rA_s", "rB_s", "rD_s", "rnorm_s", "rescue_s",
        "div_wt", "div_a", "div_b",
        "dG_wt_s", "dG_a_s", "dG_b_s", "dG_d_s",
        "dA_s", "dB_s", "dD_s", "rescue_G_s",
        "ps_wt_i", "ps_wt_j", "ps_wt_pair",
        "ps_a_i", "ps_a_j", "ps_a_pair",
        "ps_b_i", "ps_b_j", "ps_b_pair",
        "ps_d_i", "ps_d_j", "ps_d_pair",
        "rescue_cue_s1", "rescue_cue_s2",
    ]


def build_all_shape(samples) -> tuple[np.ndarray, list[str]]:
    X = np.stack([build_shape_features(s) for s in samples])
    return X, shape_feature_names()


if __name__ == "__main__":
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import m2r_data_v1 as m2r
    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X, names = build_all_shape(samples)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "m2r_shape_features.npz", X=X)
    (out / "m2r_shape_feature_names.json").write_text(
        __import__("json").dumps(names, indent=2), encoding="utf-8")
    print(f"[shape] n={len(X)} dims={X.shape[1]} -> "
          f"{out / 'm2r_shape_features.npz'}")
    print(f"names: {names}")
