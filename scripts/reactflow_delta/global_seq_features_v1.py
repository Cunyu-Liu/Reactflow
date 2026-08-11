#!/usr/bin/env python3
"""global_seq_features_v1 — STRICT-legal global sequence representation for DeepSets.

The stock run_p2_v3.build_feature produces a PAIR feature that is a purely LOCAL
21-nt window around the edited site (base one-hot + WT reactivity + WT error per
position), plus exact ref/alt, relative edit position, and condition one-hots.
The DeepSets global branch therefore carries NO full-sequence context.  This
module adds a full-sequence, STRICT-legal global representation computed ONLY
from allowed inputs (WT sequence + edit position + condition):

  * k-mer composition of the FULL WT sequence (1/2/3-mer counts, length-normalised)
    -> captures the global sequence signature (GC content, dinucleotide motifs,
       trimer composition) that a local window cannot see.
  * ViennaRNA folding features of the WT sequence (MFE, ensemble free energy,
     edit-site base-pair probability from the partition function, edit-site
     structure context) -> captures the structural environment around the edited
     site (paired vs unpaired), a well-known driver of reactivity response.

Information-permission note (endpoint_v6 STRICT_INDUCTIVE_WT_ALLOWED):
  * WT sequence is an ALLOWED input.
  * k-mer composition and ViennaRNA folding are derived PURELY from the WT
    sequence -> STRICT-legal.
  * The target eligibility mask, mutant reactivity, and held-out scatter NEVER
    enter this representation.
  * Features are fold-invariant (identical for train and held pairs with the same
    WT sequence + edit position), so they cannot leak publication identity.

The feature vector is DETERMINISTIC and fixed-size:
  GLOBAL_SEQ_DIM = KMER_DIM + VIENNA_DIM
  KMER_DIM  = 4 (1-mer) + 16 (2-mer) + 64 (3-mer) = 84
  VIENNA_DIM = 1 (mfe_per_nt) + 1 (ensemble_free_per_nt)
             + 1 (edit_unpaired_prob) + 3 (edit_structure one-hot:
                unpaired / paired / other) + 1 (local_GC_in_window) = 7
  TOTAL = 91
"""
from __future__ import annotations

import numpy as np

# Canonical base mapping (mirrors run_p2_v3.BASE_MAP) and edit-position helper.
# Kept local so this module is self-contained and testable without the full
# run_p2_v3 dependency chain.
BASE_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


def edited_index(pair) -> int:
    """0-based index of the edited site within the WT sequence."""
    codes = pair.get("eligibility_reason_codes") or []
    for i, c in enumerate(codes):
        if c == "EDITED_SITE":
            return i
    coord = pair.get("coordinate") or {}
    off = coord.get("offset")
    if isinstance(off, int):
        return off
    return 0

KMER_DIM = 84
# edit_structure one-hot categories: index 0 = unpaired '.', index 1 = paired
# (any of "()[]{}<>"), index 2 = other/unknown
STRUCT_CATS = 3
VIENNA_DIM = 3 + STRUCT_CATS + 1  # mfe + ens + unpaired + struct_oh + local_gc
GLOBAL_SEQ_DIM = KMER_DIM + VIENNA_DIM


def _kmer_composition(seq: str) -> np.ndarray:
    """Length-normalised 1/2/3-mer composition of the full WT sequence."""
    L = max(len(seq), 1)
    out = np.zeros(KMER_DIM, dtype=np.float32)
    idx = 0
    for k in (1, 2, 3):
        n_kmers = 4 ** k
        counts = np.zeros(n_kmers, dtype=np.float32)
        for i in range(len(seq) - k + 1):
            code = 0
            matched = True
            for j in range(k):
                b = seq[i + j]
                v = BASE_MAP.get(b.upper())
                if v is None:
                    matched = False
                    break
                code = code * 4 + v  # BASE_MAP already maps T/U -> 3
            if matched:
                counts[code] += 1.0
        out[idx:idx + n_kmers] = counts / L
        idx += n_kmers
    return out


def _fold_wt(seq: str) -> dict:
    """ViennaRNA folding of the WT sequence. Returns dict of structural features."""
    try:
        import RNA
    except Exception:  # pragma: no cover - ViennaRNA optional
        return {"mfe": None, "ens_free": None, "struct": None, "bpp": None}
    fc = RNA.fold_compound(seq)
    res = fc.mfe()
    # ViennaRNA versions differ in return order: (energy, structure) or
    # (structure, energy). Normalise to energy + structure defensively.
    mfe = struct = None
    if isinstance(res, (tuple, list)) and len(res) >= 2:
        a, b = res[0], res[1]
        if isinstance(a, (int, float)):
            mfe, struct = float(a), str(b)
        else:
            mfe, struct = float(b), str(a)
    elif isinstance(res, (int, float)):
        mfe = float(res)
    ens_free = None
    try:
        ens = fc.pf()
        # pf() may return a tuple/list; take the first numeric element
        if isinstance(ens, (tuple, list)) and len(ens):
            ens_free = next((float(x) for x in ens if isinstance(x, (int, float))), None)
        elif isinstance(ens, (int, float)):
            ens_free = float(ens)
    except Exception:
        ens_free = None
    bpp = None
    try:
        (bpp, _) = fc.bpp()
    except Exception:
        bpp = None
    return {"mfe": mfe, "ens_free": ens_free, "struct": struct, "bpp": bpp}


def _edit_structure_onehot(struct: str, edit_idx: int) -> np.ndarray:
    """One-hot of the MFE structure char at the edit position."""
    v = np.zeros(STRUCT_CATS, dtype=np.float32)
    if struct is None or not (0 <= edit_idx < len(struct)):
        v[2] = 1.0  # other/unknown
        return v
    c = struct[edit_idx]
    if c == ".":
        v[0] = 1.0
    elif c in "()[]{}<>":
        v[1] = 1.0
    else:
        v[2] = 1.0
    return v


def _local_gc_in_window(seq: str, edit_idx: int, half: int = 10) -> float:
    lo = max(0, edit_idx - half)
    hi = min(len(seq), edit_idx + half + 1)
    win = seq[lo:hi]
    if not win:
        return 0.0
    return float(sum(1 for b in win if b.upper() in ("G", "C"))) / len(win)


def build_global_seq_feature(wt_rec: dict, pair: dict) -> np.ndarray:
    """Return the fixed-size STRICT-legal global sequence feature vector.

    Uses only the WT sequence (allowed) + edit position. Deterministic and
    fold-invariant. Returns float32 array of length GLOBAL_SEQ_DIM.
    """
    seq = (wt_rec.get("canonical_sequence") or "").upper()
    edit_idx = edited_index(pair)
    if not seq:
        edit_idx = 0
    edit_idx = int(np.clip(edit_idx, 0, max(len(seq) - 1, 0)))

    kmer = _kmer_composition(seq)

    fold = _fold_wt(seq)
    mfe = fold.get("mfe")
    ens = fold.get("ens_free")
    struct = fold.get("struct")
    bpp = fold.get("bpp")

    L = max(len(seq), 1)
    mfe_per_nt = float(mfe) / L if mfe is not None else 0.0
    ens_per_nt = float(ens) / L if ens is not None else mfe_per_nt

    # edit-site unpaired probability from base-pair probability matrix
    unpaired = 1.0
    if bpp is not None and list(bpp):
        # bpp is a dict {(i,j,_): prob}; unpaired prob at i = 1 - sum_j bpp[i,j]
        paired_sum = 0.0
        for (i, j, k), p in bpp.items():
            if k == 1:  # contribution to base i being paired
                if i == edit_idx:
                    paired_sum += p
                elif j == edit_idx:
                    paired_sum += p
        unpaired = float(np.clip(1.0 - paired_sum, 0.0, 1.0))

    struct_oh = _edit_structure_onehot(struct, edit_idx)
    local_gc = _local_gc_in_window(seq, edit_idx)

    vienna = np.array(
        [mfe_per_nt, ens_per_nt, unpaired], dtype=np.float32)
    vienna = np.concatenate([vienna, struct_oh, [local_gc]]).astype(np.float32)

    return np.concatenate([kmer, vienna]).astype(np.float32)


def build_global_seq_matrix(rec_index: dict, pairs) -> dict:
    """Precompute global seq features for all pairs keyed by pair_id (sa:mut_idx).

    Reuses the same pair_id convention as run_baselines_v6.build_pair_recs.
    Deterministic; a single fold-invariant pass.
    """
    seq_cache: dict = {}
    out: dict = {}
    for p in pairs:
        sa = p.get("source_accession")
        wt = rec_index.get((sa, p.get("wt_profile_index"), p.get("asset_name")))
        if wt is None:
            continue
        seq = wt.get("canonical_sequence") or ""
        if seq not in seq_cache:
            seq_cache[seq] = build_global_seq_feature(wt, p)
        out[f"{sa}:{p.get('mutant_profile_index')}"] = seq_cache[seq]
    return out


if __name__ == "__main__":
    import sys
    print(f"GLOBAL_SEQ_DIM = {GLOBAL_SEQ_DIM}")