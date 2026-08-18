#!/usr/bin/env python3
"""m2r_mfe_features_v1.py — thermodynamic (ViennaRNA MFE) legal features.

MOTIVATION (new legal modality, method-level):
The current legal features are all EXPERIMENTAL (SHAPE reactivity) or
STRUCTURAL-DESIGN (target structure, M2 structure).  None of them provides a
PREDICTED view of the double-mutant structure.  But the double-mutant SEQUENCE
is legal (it is simply the two edits applied to the WT), so we can FOLD it with
ViennaRNA and get a thermodynamic prediction of whether the double mutant
restores the WT structure — the exact mechanistic quantity the rescue_factor
measures.

  rescue = 1 - rD/sqrt(rA^2 + rB^2)   (experimental, SHAPE design-region RMSD)

We add the analogous THERMODYNAMIC quantities computed from MFE structures:

  rA_mfe = base-pair distance(WT, A), rB_mfe = distance(WT, B)
  rD_mfe = base-pair distance(WT, D)
  mfe_rescue = 1 - rD_mfe/sqrt(rA_mfe^2 + rB_mfe^2)   (thermodynamic rescue)
  mfe_rescue_G = 1 - |dG_D - dG_WT| / (|dG_A-dG_WT| + |dG_B-dG_WT| + eps)
  pair status at (i,j) in each of {WT, A, B, D} MFE structures
  centroid-structure distance analog (robust to MFE degeneracy)

LEGALITY: everything is computed from the WT / single-mutant / double-mutant
SEQUENCES (all known before the double-mutant SHAPE experiment).  No
experimental double-mutant reactivity is used.  This is exactly what a
researcher would do: fold the double-mutant sequence and check if it restores
the WT fold.

Implementation: ViennaRNA Python binding (RNA.fold).  Folds are cached per
unique sequence.  Base-pair distance uses the pair-set difference
(McLachlan-style), robust to pseudoknots via per-bracket-type stacks.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

CACHE = {}


def _fold(seq: str):
    """Return (dot-bracket, mfe_kcal).  Cached per sequence."""
    key = ("fold", seq)
    if key not in CACHE:
        import RNA
        ss, mfe = RNA.fold(seq)
        CACHE[key] = (ss, float(mfe))
    return CACHE[key]


def _centroid(seq: str):
    key = ("centroid", seq)
    if key not in CACHE:
        import RNA
        fd = RNA.fold_compound(seq)
        fd.pf()
        CACHE[key] = fd.centroid()[0]
    return CACHE[key]


def pair_set(structure: str) -> set:
    """Extract {(a,b)} base pairs from a dot-bracket string (pseudoknot-aware)."""
    n = len(structure)
    pairs = set()
    from collections import defaultdict
    stacks = defaultdict(list)
    for idx, ch in enumerate(structure):
        if ch in "([{":
            stacks[ch].append(idx)
        elif ch in ")]}":
            open_ch = {"}": "{", "]": "[", ")": "("}[ch]
            st = stacks[open_ch]
            if st:
                mate = st.pop()
                pairs.add((mate, idx))
    return pairs


def pair_distance(s1: str, s2: str) -> float:
    """Base-pair (McLachlan) distance between two dot-bracket structures."""
    p1, p2 = pair_set(s1), pair_set(s2)
    return float(len(p1 ^ p2))


def pairing_status(structure: str, i: int, j: int) -> tuple[int, int, int]:
    """(paired_i, paired_j, paired_with_each_other) for positions i,j."""
    p = pair_set(structure)
    pi = 1 if any(a == i or b == i for a, b in p) else 0
    pj = 1 if any(a == j or b == j for a, b in p) else 0
    both = 1 if (i, j) in p or (j, i) in p else 0
    return pi, pj, both


def build_mfe_features(s) -> np.ndarray:
    """MFE feature vector for one M2RPair.  All legal (sequence-based)."""
    seq = s.sequence
    i, j = s.editA_seq_pos, s.editB_seq_pos
    seqA = s.mutA_seq if s.mutA_seq else seq
    seqB = s.mutB_seq if s.mutB_seq else seq
    # reconstruct the double-mutant sequence: mutA_seq with position j <- mutB_seq[j]
    seqD = list(seqA)
    if 0 <= j < len(seqD) and 0 <= j < len(seqB):
        seqD[j] = seqB[j]
    seqD = "".join(seqD)

    # ---- MFE structures + energies ----
    ss_wt, dG_wt = _fold(seq)
    ss_a, dG_a = _fold(seqA)
    ss_b, dG_b = _fold(seqB)
    ss_d, dG_d = _fold(seqD)
    ct_wt, ct_a, ct_b, ct_d = _centroid(seq), _centroid(seqA), \
        _centroid(seqB), _centroid(seqD)

    def _len_adjust(ss, n):
        return ss if len(ss) == n else ss[:n]

    n = len(seq)
    ss_wt = _len_adjust(ss_wt, n); ss_a = _len_adjust(ss_a, n)
    ss_b = _len_adjust(ss_b, n); ss_d = _len_adjust(ss_d, n)
    ct_wt = _len_adjust(ct_wt, n); ct_a = _len_adjust(ct_a, n)
    ct_b = _len_adjust(ct_b, n); ct_d = _len_adjust(ct_d, n)

    # ---- structure distances (pair-set) ----
    rA_mfe = pair_distance(ss_wt, ss_a)
    rB_mfe = pair_distance(ss_wt, ss_b)
    rD_mfe = pair_distance(ss_wt, ss_d)
    rnorm = np.sqrt(rA_mfe ** 2 + rB_mfe ** 2)
    mfe_rescue = (1.0 - rD_mfe / rnorm) if rnorm > 0 else 0.0

    # centroid analog (robust to MFE degeneracy)
    cA = pair_distance(ct_wt, ct_a); cB = pair_distance(ct_wt, ct_b)
    cD = pair_distance(ct_wt, ct_d)
    cnorm = np.sqrt(cA ** 2 + cB ** 2)
    ct_rescue = (1.0 - cD / cnorm) if cnorm > 0 else 0.0

    # ---- energy-based rescue (folding stabilities) ----
    dA_G = abs(dG_a - dG_wt); dB_G = abs(dG_b - dG_wt)
    dD_G = abs(dG_d - dG_wt)
    gnorm = dA_G + dB_G + 1e-9
    mfe_rescue_G = 1.0 - dD_G / gnorm
    # is the double mutant MORE stable than the sum of singles (restoration)?
    dG_restore = dG_wt - dG_d                     # >0 => double stabilizes
    dG_single_sum = (dG_wt - dG_a) + (dG_wt - dG_b)
    cooperativity = dG_restore - dG_single_sum    # positive => cooperative rescue

    # ---- (i,j) pair status in each MFE structure ----
    pa_wt = pairing_status(ss_wt, i, j)
    pa_a = pairing_status(ss_a, i, j)
    pa_b = pairing_status(ss_b, i, j)
    pa_d = pairing_status(ss_d, i, j)

    parts = [
        # structure distances + thermodynamic rescue analogs
        np.array([rA_mfe, rB_mfe, rD_mfe, rnorm, mfe_rescue,
                  cA, cB, cD, ct_rescue]),
        # energy features
        np.array([dG_wt, dG_a, dG_b, dG_d,
                  dA_G, dB_G, dD_G, mfe_rescue_G,
                  dG_restore, dG_single_sum, cooperativity]),
        # (i,j) pairing status in WT / A / B / D  (3 bits each = 12)
        np.array([pa_wt[0], pa_wt[1], pa_wt[2],
                  pa_a[0], pa_a[1], pa_a[2],
                  pa_b[0], pa_b[1], pa_b[2],
                  pa_d[0], pa_d[1], pa_d[2]]),
        # whether the (i,j) pair forms in D but not in the singles (rescue cue)
        np.array([pa_d[2] - max(pa_a[2], pa_b[2]),
                  pa_d[2] * (1 - pa_wt[2])]),
    ]
    return np.concatenate(parts).astype(np.float64)


def mfe_feature_names() -> list[str]:
    return [
        "rA_mfe", "rB_mfe", "rD_mfe", "rnorm_mfe", "mfe_rescue",
        "cA", "cB", "cD", "ct_rescue",
        "dG_wt", "dG_a", "dG_b", "dG_d",
        "dA_G", "dB_G", "dD_G", "mfe_rescue_G",
        "dG_restore", "dG_single_sum", "cooperativity",
        "pa_wt_i", "pa_wt_j", "pa_wt_pair",
        "pa_a_i", "pa_a_j", "pa_a_pair",
        "pa_b_i", "pa_b_j", "pa_b_pair",
        "pa_d_i", "pa_d_j", "pa_d_pair",
        "rescue_cue_1", "rescue_cue_2",
    ]


def build_all_mfe(samples) -> tuple[np.ndarray, list[str]]:
    X = np.stack([build_mfe_features(s) for s in samples])
    return X, mfe_feature_names()


def save_cache(path: str):
    Path(path).write_text(json.dumps(
        {f"{k[0]}:{k[1]}": v for k, v in CACHE.items()}), encoding="utf-8")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import m2r_data_v1 as m2r
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--m2r-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    designs, meta = m2r.parse_m2r_csv(args.m2r_csv)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X, names = build_all_mfe(samples)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "m2r_mfe_features.npz", X=X)
    (out / "m2r_mfe_feature_names.json").write_text(
        json.dumps(names, indent=2), encoding="utf-8")
    save_cache(out / "m2r_mfe_fold_cache.json")
    print(f"[mfe] n={len(X)} dims={X.shape[1]} -> "
          f"{out / 'm2r_mfe_features.npz'}")
    print(f"names: {names}")
