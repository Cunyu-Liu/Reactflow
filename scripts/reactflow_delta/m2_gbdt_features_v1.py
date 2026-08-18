#!/usr/bin/env python3
"""m2_gbdt_features_v1.py — per-position legal features for the M2
response-spectrum task, enabling a cross-architecture GBDT ensemble.

MOTIVATION (method-level, mirrors the M2R MFE lever):
The M2 deep models (MLP / posaware / attn) consume only base-one-hot +
reactivity + error per window position plus a few globals (`build_feature`,
POS_DIM=7).  They receive NO thermodynamic information: the model must infer
structure implicitly from sequence.  In M2R the single largest method-level
gain (+1.11pp, perm p=0.008) came from adding a legal thermodynamic modality
(ViennaRNA folding of the WT and mutant sequences).  We bring the same
modality to M2 as **per-position** features, and expose a feature-based GBDT
whose error structure is decorrelated from the deep attention model.

For every eligible window position k (mapping to sequence index
`idx = edit_seq_pos - WINDOW//2 + k`) the feature vector is:

  pos_rel               : k relative to window centre (normalised)
  edit_dist             : |idx - edit_seq_pos| / len(seq)
  wt_react window ±2    : 5 values (WT profile, legal pre-experiment)
  wt_err at idx         : 1 value
  base one-hot          : 5 values at idx
  MFE structure         : 6 values at idx (paired / depth / bpp for WT and mutant folds)
  pair_change           : whether mutant fold changes pairing status at idx
  ref/alt one-hot       : 10 values (the edited base pair)
  edit_pos_norm         : edit_seq_pos / len(seq)

LEAK-FREE (critical): the target is y = (mut_reactivity - wt_reactivity)/scale,
the single-mutant SHAPE RESPONSE.  The MUTANT reactivity profile is therefore
the target itself and must NEVER appear as a feature (feeding mut_reactivity
makes the response trivially predictable — a circular leak).  Only the WT
profile, the sequences (WT + mutant, both known before the experiment) and
the mutation identity (ref/alt allele) are legal inputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_mfe_features_v1 as mfe

WINDOW = 21
HALF = WINDOW // 2
BASE_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
CACHE = {}


def _base_oh(base: str) -> np.ndarray:
    idx = BASE_MAP.get(base.upper(), 4)
    oh = np.zeros(5, dtype=np.float64)
    oh[idx] = 1.0
    return oh


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _seq_at(seq: str, idx: int) -> str:
    return seq[idx] if 0 <= idx < len(seq) else "N"


def _structure_at(seq: str):
    """Cached per-sequence folding info: (paired, depth, bpp_paired) arrays.

    paired[i] = 1 if position i is in a base pair in the MFE structure.
    depth[i]   = bracket nesting depth at position i.
    bpp_paired[i] = 1 - bpp[i][i]  (probability position i is paired).
    """
    key = ("m2str", seq)
    if key not in CACHE:
        ss, _ = mfe._fold(seq)
        pairs = mfe.pair_set(ss)
        n = len(ss)
        paired = np.zeros(n, dtype=np.float64)
        depth = np.zeros(n, dtype=np.float64)
        # bracket depth
        d = 0
        for i, ch in enumerate(ss):
            if ch in "([{":
                d += 1
                depth[i] = d
            elif ch in ")]}":
                depth[i] = d
                d = max(d - 1, 0)
            else:
                depth[i] = d
        for a, b in pairs:
            paired[a] = 1.0
            paired[b] = 1.0
        # base-pair probability (from the partition function)
        try:
            bpp = np.asarray(mfe._bpp(seq), dtype=np.float64)
            bpp_paired = np.zeros(n, dtype=np.float64)
            m = min(n, bpp.shape[0])
            for i in range(m):
                bpp_paired[i] = max(0.0, 1.0 - bpp[i][i])
        except Exception:
            bpp_paired = paired.copy()
        CACHE[key] = (paired, depth, bpp_paired)
    return CACHE[key]


def mutant_sequence(s) -> str:
    """WT sequence with the edited base replaced by the mutant allele."""
    seq = list(s.sequence)
    alt = s.pair.get("alt_allele") or ""
    if alt and 0 <= s.edit_seq_pos < len(seq):
        seq[s.edit_seq_pos] = alt
    return "".join(seq)


def build_position_features(s, k: int) -> np.ndarray:
    """Feature vector for window position k (0..WINDOW-1) of an M2Sample.

    LEAK-FREE: the mutant reactivity profile is the target and is excluded.
    """
    n = len(s.sequence)
    idx = s.edit_seq_pos - HALF + k

    wt = [_f(s.wt_reactivity[i]) if 0 <= i < len(s.wt_reactivity) else 0.0
          for i in (idx - 2, idx - 1, idx, idx + 1, idx + 2)]
    wt_err = _f(s.wt_error[idx]) if 0 <= idx < len(s.wt_error) else 0.0

    base = _base_oh(_seq_at(s.sequence, idx))

    # MFE structure at idx for WT and mutant folds
    wt_s, wt_d, wt_b = _structure_at(s.sequence)
    mseq = mutant_sequence(s)
    mu_s, mu_d, mu_b = _structure_at(mseq)
    in_range = 0 <= idx < len(wt_s)
    wt_p = wt_s[idx] if in_range else 0.0
    wt_de = wt_d[idx] if in_range else 0.0
    wt_bp = wt_b[idx] if in_range else 0.0
    mu_p = mu_s[idx] if 0 <= idx < len(mu_s) else 0.0
    mu_de = mu_d[idx] if 0 <= idx < len(mu_d) else 0.0
    mu_bp = mu_b[idx] if 0 <= idx < len(mu_b) else 0.0
    pair_change = mu_p - wt_p

    ref = _base_oh(s.pair.get("ref_allele") or _seq_at(s.sequence, s.edit_seq_pos))
    alt = _base_oh(s.pair.get("alt_allele") or "")

    parts = [
        np.array([(k - HALF) / HALF, abs(idx - s.edit_seq_pos) / max(n, 1)]),
        np.array(wt),
        np.array([wt_err]),
        base,
        np.array([wt_p, wt_de, wt_bp, mu_p, mu_de, mu_bp, pair_change]),
        ref, alt,
        np.array([s.edit_seq_pos / max(n, 1)]),
    ]
    return np.concatenate(parts).astype(np.float64)


def feature_names() -> list[str]:
    return [
        "pos_rel", "edit_dist",
        "wt_m2", "wt_m1", "wt_0", "wt_p1", "wt_p2",
        "wt_err_0",
        "base_A", "base_C", "base_G", "base_U", "base_N",
        "wt_paired", "wt_depth", "wt_bpp",
        "mut_paired", "mut_depth", "mut_bpp", "pair_change",
        "ref_A", "ref_C", "ref_G", "ref_U", "ref_N",
        "alt_A", "alt_C", "alt_G", "alt_U", "alt_N",
        "edit_pos_norm",
    ]


def build_all(samples, spectra) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Build (X, y, w, keys, pids) per eligible window position over all M2 samples.

    ``spectra`` maps pair_id -> {"y": [...], "w": [...], "design_id": ...}.
    Only positions with weight 1 are kept (matching the pooled-WMAE metric).
    ``keys`` are design ids; ``pids`` are "design_id:mutA" (the exchangeable
    design is derived from either).
    """
    rows = []
    for s in samples:
        pid = f"{s.design_id}:{s.mutA}"
        if pid not in spectra:
            continue
        sp = spectra[pid]
        for k in range(WINDOW):
            if not (k < len(sp["w"]) and sp["w"][k] > 0):
                continue
            rows.append((s, k))
    X = np.stack([build_position_features(s, k) for s, k in rows])
    y = np.array([spectra[f"{s.design_id}:{s.mutA}"]["y"][k] for s, k in rows])
    w = np.ones(len(rows), dtype=np.float64)
    keys = np.array([s.design_id for s, _ in rows], dtype="U64")
    pids = np.array([f"{s.design_id}:{s.mutA}:{k}" for s, k in rows], dtype="U96")
    return X, y, w, keys, pids


def build_mutant_sequences(samples) -> dict:
    return {s.design_id: mutant_sequence(s) for s in samples}


if __name__ == "__main__":
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--m2-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import m2_data_v1 as m2d
    import response_spectrum_scinv_v1 as rss
    from run_baselines_v6 import WINDOW as W
    designs, dmeta = m2d.parse_m2_csv(args.m2_csv)
    samples = m2d.build_all_samples(designs)
    spectra = {}
    for s in samples:
        pid = f"{s.design_id}:{s.mutA}"
        yv, wv, sc = rss.pair_response_spectrum(
            s.wt_reactivity, s.mut_reactivity, s.eligibility_mask,
            s.edit_seq_pos, window=W)
        if sc is None or sum(wv) <= 0:
            continue
        spectra[pid] = {"y": yv, "w": wv, "design_id": s.design_id}
    X, y, w, keys = build_all(samples, spectra)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "m2_gbdt_features.npz", X=X, y=y, w=w, keys=keys)
    (out / "m2_gbdt_feature_names.json").write_text(
        __import__("json").dumps(feature_names(), indent=2), encoding="utf-8")
    print(f"[m2gbdt] n={len(y)} dims={X.shape[1]} designs={len(set(keys.tolist()))} "
          f"-> {out / 'm2_gbdt_features.npz'}")
    print(f"names: {feature_names()}")
