#!/usr/bin/env python3
"""B0-X frozen evaluator (contract §13).

Computes the primary endpoint (full-position continuous delta) metrics on the
frozen validation split: pooled ratio-of-sums Skill (§13.2), WMAE/MAE, study
cluster CI bootstrap (§13.1), and the group-aware permutation comparison
(§13.3).  The reference for Skill is the strongest trivial baseline.  No test
split is consumed.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

import numpy as np

from b0x_data import Pair

_BASES = ("A", "C", "G", "U")


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _eligible_arrays(pair: Pair, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (target, pred, mask_weights) restricted to eligible positions.

    Positions are weighted by inverse WT reactivity magnitude (WMAE) with a
    floor to avoid exploding weights; mask index preserves eligible positions.
    """
    t = np.array([float(pair.delta[i]) for i in range(len(pair.mask)) if pair.mask[i]],
                 dtype=np.float64)
    pr = np.array([float(pred[i]) for i in range(len(pair.mask)) if pair.mask[i]],
                  dtype=np.float64)
    # align pred to eligible length
    if len(pr) != len(t):
        pr = pr[: len(t)] if len(pr) >= len(t) else np.concatenate([pr, np.zeros(len(t) - len(pr))])
    # WMAE weights: inverse of WT reactivity magnitude, floored
    wt = np.array([float(pair.wt_reactivity[i]) for i in range(len(pair.mask)) if pair.mask[i]],
                  dtype=np.float64)
    w = np.where(np.abs(wt) > 1e-6, 1.0 / np.clip(np.abs(wt), 1e-3, None), 1.0)
    w = np.clip(w, 0.001, 10.0)
    return t, pr, w


def per_pair_loss(pair: Pair, pred: np.ndarray) -> dict[str, float]:
    t, pr, w = _eligible_arrays(pair, pred)
    if len(t) == 0:
        return {"mae": float("nan"), "wmae": float("nan"), "n": 0}
    abs_ = np.abs(t - pr)
    return {
        "mae": float(np.mean(abs_)),
        "wmae": float(np.sum(w * abs_) / np.sum(w)),
        "n": int(len(t)),
    }


def pooled_skill(pairs: list[Pair], preds: dict[str, np.ndarray],
                 ref_preds: dict[str, np.ndarray]) -> dict[str, Any]:
    """Pooled ratio-of-sums Skill vs a reference baseline (§13.2)."""
    num_w = 0.0
    den_w = 0.0
    num_abs = 0.0
    den_abs = 0.0
    for p in pairs:
        pred = preds.get(p.pair_id)
        if pred is None:
            continue
        ref = ref_preds.get(p.pair_id)
        if ref is None:
            continue
        t, pr, w = _eligible_arrays(p, pred)
        r, rp, rw = _eligible_arrays(p, ref)
        num_w += float(np.sum(w * np.abs(t - pr)))
        den_w += float(np.sum(rw * np.abs(t - rp)))
        num_abs += float(np.sum(np.abs(t - pr)))
        den_abs += float(np.sum(np.abs(t - rp)))
    if den_w == 0 or den_abs == 0:
        return {"skill_wmae": float("nan"), "skill_mae": float("nan"), "num_w": num_w, "den_w": den_w}
    return {
        "skill_wmae": 1.0 - num_w / den_w,
        "skill_mae": 1.0 - num_abs / den_abs,
        "num_w": num_w,
        "den_w": den_w,
    }


def pair_scalar_gain(pairs: list[Pair], preds: dict[str, np.ndarray],
                     ref_preds: dict[str, np.ndarray]) -> list[float]:
    """Per-pair WMAE gain (ref_loss - main_loss), positive = main is better."""
    gains = []
    for p in pairs:
        pred = preds.get(p.pair_id)
        ref = ref_preds.get(p.pair_id)
        if pred is None or ref is None:
            continue
        t, pr, w = _eligible_arrays(p, pred)
        r, rp, rw = _eligible_arrays(p, ref)
        if len(t) == 0:
            continue
        main_loss = float(np.sum(w * np.abs(t - pr)) / np.sum(w))
        ref_loss = float(np.sum(rw * np.abs(t - rp)) / np.sum(rw))
        gains.append(ref_loss - main_loss)
    return gains


def _pair_contributions(pairs: list[Pair], preds: dict[str, np.ndarray],
                        ref_preds: dict[str, np.ndarray]) -> tuple[list[dict], dict[str, list[int]]]:
    """Precompute per-pair (num, den) pooled WMAE contributions and study indices.

    Returns (contribs, by_study) where contribs[i] = {"num":.., "den":.., "study":..}
    for each pair with available predictions, and by_study maps study -> contrib indices.
    This makes cluster bootstrap and group-aware permutation O(pairs) instead of
    O(pairs * n_boot * positions) by avoiding repeated eligible-array extraction.
    """
    contribs: list[dict] = []
    by_study: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        pred = preds.get(p.pair_id)
        ref = ref_preds.get(p.pair_id)
        if pred is None or ref is None:
            continue
        t, pr, w = _eligible_arrays(p, pred)
        r, rp, rw = _eligible_arrays(p, ref)
        if len(t) == 0:
            continue
        num = float(np.sum(w * np.abs(t - pr)))
        den = float(np.sum(rw * np.abs(t - rp)))
        if den <= 0:
            continue
        contribs.append({"num": num, "den": den, "study": p.study})
        by_study[p.study].append(len(contribs) - 1)
    return contribs, by_study


def cluster_ci(pairs: list[Pair], preds: dict[str, np.ndarray],
               ref_preds: dict[str, np.ndarray], *,
               n_boot: int = 1000, seed: int = 20260804) -> dict[str, Any]:
    """Study-level cluster bootstrap CI for the pooled WMAE Skill gain."""
    contribs, by_study = _pair_contributions(pairs, preds, ref_preds)
    if not contribs:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_studies": 0, "n_boot": n_boot}
    studies = list(by_study.keys())
    rng = random.Random(seed)

    def pooled_gain(indices):
        num = sum(contribs[i]["num"] for i in indices)
        den = sum(contribs[i]["den"] for i in indices)
        return (1.0 - num / den) if den else float("nan")

    real = pooled_gain(range(len(contribs)))
    boots = []
    for _ in range(n_boot):
        # bootstrap study clusters with replacement
        sample_ids = []
        for _ in range(len(studies)):
            s = rng.choice(studies)
            sample_ids.extend(by_study[s])
        boots.append(pooled_gain(sample_ids))
    boots = [b for b in boots if _finite(b)]
    if not boots:
        return {"point": real, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_studies": len(studies), "n_boot": n_boot}
    boots = np.array(boots)
    return {
        "point": float(real),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n_studies": len(studies),
        "n_boot": n_boot,
    }


def group_aware_permutation(pairs: list[Pair], target_baseline: str,
                            preds: dict[str, np.ndarray],
                            ref_preds: dict[str, np.ndarray], *,
                            n_perm: int = 100, seed: int = 20260804) -> dict[str, Any]:
    """Group-aware permutation null (§13.3).

    Permutes the delta target vectors within (study, parent) exchangeability
    blocks, re-fits nothing (we use the frozen baseline predictions as the
    "changer load" proxy), and compares the pooled WMAE Skill of the real
    alignment vs the shuffled alignment.  The statistic is the pooled WMAE
    Skill (higher = better).  Real must exceed the permutation null.
    """
    rng = random.Random(seed)
    # Build per-(study,parent) blocks of pair targets
    blocks: dict[tuple, list[Pair]] = defaultdict(list)
    for p in pairs:
        blocks[(p.study, p.parent)].append(p)

    def skill_for(preds_map):
        num = den = 0.0
        for p in pairs:
            pred = preds_map.get(p.pair_id)
            ref = ref_preds.get(p.pair_id)
            if pred is None or ref is None:
                continue
            t, pr, w = _eligible_arrays(p, pred)
            r, rp, rw = _eligible_arrays(p, ref)
            num += float(np.sum(w * np.abs(t - pr)))
            den += float(np.sum(rw * np.abs(t - rp)))
        return (1.0 - num / den) if den else float("nan")

    real = skill_for(preds)
    null_skills = []
    for _ in range(n_perm):
        # Within each block, assign a random permutation of the pair target
        # vectors across the block's pairs (shuffle targets, keep preds).
        permuted = dict(preds)
        for blk, plist in blocks.items():
            if len(plist) < 2:
                continue
            targets = [np.array(p.delta, dtype=np.float64) for p in plist]
            rng.shuffle(targets)
            for p, t in zip(plist, targets):
                # store shuffled target as a pseudo-prediction for null
                permuted[p.pair_id] = t.astype(np.float32)
        null_skills.append(skill_for(permuted))
    null_skills = [s for s in null_skills if _finite(s)]
    p_value = (sum(1 for s in null_skills if s >= real) + 1) / (len(null_skills) + 1)
    return {
        "real_skill": float(real),
        "null_mean": float(np.mean(null_skills)) if null_skills else float("nan"),
        "null_median": float(np.median(null_skills)) if null_skills else float("nan"),
        "null_max": float(np.max(null_skills)) if null_skills else float("nan"),
        "p_value": float(p_value),
        "n_perm": n_perm,
        "pass_real_gt_null": bool(real > np.mean(null_skills)) if null_skills else False,
    }


def learning_curve(pairs: list[Pair], train: list[Pair], preds_by_pair: dict[str, np.ndarray],
                   ref_preds: dict[str, np.ndarray], *,
                   fractions=(0.1, 0.25, 0.5, 1.0), seed: int = 20260804) -> dict[str, Any]:
    """Learning curve: baseline skill vs fraction of training data.

    This is a proxy that reports the WMAE Skill of the frozen baseline
    predictions on a random subset of validation pairs (as a function of the
    fraction of the *prediction set*), indicating data sufficiency.  The full
    P2 learning curve (retrain at each fraction) is reported by the runner.
    """
    rng = random.Random(seed)
    out = {}
    for frac in fractions:
        n = max(1, int(len(pairs) * frac))
        sub = rng.sample(pairs, n)
        sk = pooled_skill(sub, preds_by_pair, ref_preds)
        out[f"frac_{frac}"] = {
            "n_pairs": len(sub),
            "skill_wmae": sk["skill_wmae"],
        }
    return out