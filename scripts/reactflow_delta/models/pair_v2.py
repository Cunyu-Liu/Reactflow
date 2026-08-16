"""Phase 3 scheme-2 (contract §9.3): exact-alt WT/mutant explicit interaction.

Builds explicit `[WT, Mut, Mut-WT, WT*(Mut-WT), condition]` representations from
ALLOWED inputs only (WT sequence, exact single-nucleotide mutation, WT
experimental reactivity state, condition). The mutant experimental reactivity
profile is NEVER used (prospective input forbidden).

  * WT  : per-position one-hot of WT base + WT reactivity anchor [r, e]
  * Mut : per-position one-hot with the actual ALT base at the edited site
          (+ same WT reactivity anchor at every position)
  * Mut-WT      : explicit difference (nonzero only at the edited site)
  * WT*(Mut-WT) : explicit interaction (which WT context the mutation couples to)
  * condition   : probe / modifier / experimentType / temperature

Scientific claim under test: EXPLICITLY engineering the exact-alt
difference/interaction terms provides a learnable increment over a
same-capacity generic CONCAT model that receives `[WT, Mut, condition]` without
the explicit difference/interaction features. Verified by paired
publication-block bootstrap CI of the conditional-WMAE-skill difference.
"""
from __future__ import annotations

import numpy as np

from run_p2_v3 import (  # noqa: E402
    WINDOW, HALF, _base_oh, _norm_react, _norm_err, edited_index,
    _oh, _oh_index, PROBES, MODIFIERS, EXPTYPES,
)
from models.pair_v1 import CapacityMatchedMLP, count_params  # noqa: E402
from train_v2 import train_flat, predict_flat  # noqa: E402

POS_DIM = 7  # base(5) + reactivity(1) + error(1)


def _pos_rep(base, r, e):
    return np.concatenate([_base_oh(base),
                           np.array([_norm_react(r), _norm_err(e)], dtype=np.float32)])


def _condition_feat(pair, wt_rec):
    cond = pair.get("condition") or {}
    probe = wt_rec.get("probe") or []
    parts = [_oh(probe[0] if probe else "", _oh_index(PROBES), len(PROBES))]
    mod = cond.get("modifier") or []
    parts.append(_oh(mod[0] if mod else "", _oh_index(MODIFIERS), len(MODIFIERS)))
    et = cond.get("experimentType") or []
    parts.append(_oh(et[0] if et else "", _oh_index(EXPTYPES), len(EXPTYPES)))
    temps = [t for t in (cond.get("temperature") or [])
             if str(t).replace(".", "").replace("C", "").isdigit()]
    tval = float(str(temps[0]).replace("C", "")) if temps else 37.0
    parts.append(np.array([tval / 100.0], dtype=np.float32))
    return np.concatenate(parts)


def build_scheme2_features(pair, wt_rec, explicit_interaction: bool = True,
                           use_wt_anchor: bool = True) -> np.ndarray:
    """Fold-invariant scheme-2 feature vector (ALLOWED inputs only).

    explicit_interaction=True  -> [WT, Mut, Mut-WT, WT*(Mut-WT), ei, cond]
    explicit_interaction=False -> [WT, Mut, ei, cond]   (same-capacity CONCAT baseline)
    use_wt_anchor=False        -> drops the WT reactivity/error anchor (seq-only)
    """
    seq = wt_rec.get("canonical_sequence") or ""
    rl = wt_rec.get("reactivity_layers", {})
    tf = rl.get("train_frozen", {}) or rl.get("raw", {})
    react = np.nan_to_num(np.asarray(tf.get("reactivity") or [], dtype=np.float32), nan=0.0)
    err = np.nan_to_num(np.asarray(tf.get("error") or [], dtype=np.float32), nan=0.0)
    n = len(seq)
    ei = edited_index(pair)
    alt = pair.get("alt_allele")
    wt_parts, mut_parts = [], []
    for k in range(WINDOW):
        idx = ei - HALF + k
        if 0 <= idx < n:
            base = seq[idx]
            r = react[idx] if idx < len(react) else 0.0
            e = err[idx] if idx < len(err) else 0.0
        else:
            base = None
            r, e = 0.0, 0.0
        mut_base = alt if idx == ei else base
        if use_wt_anchor:
            wt_parts.append(_pos_rep(base, r, e))
            mut_parts.append(_pos_rep(mut_base, r, e))
        else:
            wt_parts.append(_base_oh(base).astype(np.float32))
            mut_parts.append(_base_oh(mut_base).astype(np.float32))
    WT = np.concatenate(wt_parts).astype(np.float32)
    Mut = np.concatenate(mut_parts).astype(np.float32)
    ei_feat = np.array([float(ei) / max(n, 1)], dtype=np.float32)
    cond = _condition_feat(pair, wt_rec)
    if explicit_interaction:
        diff = (Mut - WT).astype(np.float32)
        inter = (WT * diff).astype(np.float32)
        return np.concatenate([WT, Mut, diff, inter, ei_feat, cond])
    return np.concatenate([WT, Mut, ei_feat, cond])
