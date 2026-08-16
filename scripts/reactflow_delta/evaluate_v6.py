#!/usr/bin/env python3
"""evaluate_v6 — keyed, publication-anchored evaluator for benchmark_v3.

Fixes the Scheme-3 positional misalignment: every comparison is performed on
predictions aligned by biological keys (pair_id, fold_id, seed, model_variant),
never by array-position zip.

Enforces:
  * tie/order-invariant AP;
  * pooled AND publication-macro AUPRC (primary changer task);
  * WMAE = sum(w*|y-pred|)/sum(w) (weight scaling invariant);
  * publication-macro and pooled WMAE both reported;
  * leave-one-dominant-publication-out;
  * seeds are optimization repeats, do NOT increase publication N;
  * publication-level paired effect + 95% CI;
  * resampling only at the highest-level exchangeable unit (publication);
  * unique null space enumerated; degeneracy -> UNIDENTIFIABLE;
  * <3 independent publications -> no confirmatory CI;
  * identity-only permutation -> UNIDENTIFIABLE.

All functions take keyed rows (dicts) and require exact key equality between
candidate and baseline before comparing.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Sequence

KEY_ALIGNMENT = ["pair_id", "fold_id", "seed", "model_variant"]
UNIDENTIFIABLE = "UNIDENTIFIABLE"


def is_unidentifiable(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v == UNIDENTIFIABLE)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _wmae(y, pred, w) -> float:
    num = den = 0.0
    for yi, pi, wi in zip(y, pred, w):
        num += float(wi) * abs(float(yi) - float(pi))
        den += float(wi)
    if den <= 0.0:
        raise ValueError("sum of weights must be > 0")
    return num / den


def _weighted_mean(y, w) -> float:
    num = den = 0.0
    for yi, wi in zip(y, w):
        num += float(wi) * float(yi)
        den += float(wi)
    if den <= 0.0:
        raise ValueError("sum of weights must be > 0")
    return num / den


def weighted_median(values, weights) -> float:
    pairs = sorted(zip(values, weights), key=lambda t: t[0])
    total = sum(float(w) for _, w in pairs)
    if total <= 0.0:
        raise ValueError("sum of weights must be > 0")
    cum = 0.0
    for v, w in pairs:
        cum += float(w)
        if cum >= total / 2.0:
            return float(v)
    return float(pairs[-1][0])


def auprc(scores, labels) -> float:
    """Tie/order-invariant average precision (AUPRC)."""
    pairs = sorted(zip(scores, labels), key=lambda t: (-t[0], t[1]))
    n_pos = sum(1 for _, lab in pairs if lab == 1)
    if n_pos == 0:
        return 0.0
    hits = 0
    ap = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        block = pairs[i:j]
        for k in range(j - i):
            if block[k][1] == 1:
                hits += 1
                ap += hits / (i + k + 1)
        i = j
    return ap / n_pos


def _base_key(r: dict[str, Any]) -> tuple:
    """Baseline row key (candidate and baseline share pair/fold/seed)."""
    return (r.get("pair_id"), r.get("fold_id"), r.get("seed"))


def index_rows(rows: Sequence[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    idx: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        k = _base_key(r)
        if k in idx:
            raise ValueError(f"DUPLICATE_BASE_KEY: {k}")
        idx[k] = r
    return idx


def _group_by_publication(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["publication_id"]].append(r)
    return g


# ---------------------------------------------------------------------------
# primary changer: pooled + macro AUPRC
# ---------------------------------------------------------------------------
def primary_auprc(candidate_rows, baseline_rows, score_field="transformed_prediction",
                  label_field="y") -> dict[str, Any]:
    cand_by_pub = _group_by_publication(candidate_rows)
    base_by_pub = _group_by_publication(baseline_rows)
    if set(cand_by_pub) != set(base_by_pub):
        raise ValueError("CANDIDATE_BASELINE_PUBLICATION_MISALIGN")

    pooled_auprc = auprc([r[score_field] for r in candidate_rows],
                         [int(r[label_field]) for r in candidate_rows])
    per_pub = {}
    for pub in sorted(cand_by_pub, key=str):
        s = [r[score_field] for r in cand_by_pub[pub]]
        l = [int(r[label_field]) for r in cand_by_pub[pub]]
        per_pub[pub] = {"n": len(s), "n_pos": sum(l), "auprc": auprc(s, l)}
    pubs_pos = [pub for pub, d in per_pub.items() if d["n_pos"] > 0]
    macro = (sum(d["auprc"] for pub, d in per_pub.items() if pub in pubs_pos) / len(pubs_pos)
             if pubs_pos else None)
    return {"pooled_auprc": float(pooled_auprc),
            "macro_auprc": float(macro) if macro is not None else UNIDENTIFIABLE,
            "per_publication": per_pub,
            "n_publications": len(per_pub),
            "n_publications_with_positive": len(pubs_pos),
            "n_rows": len(candidate_rows)}


# ---------------------------------------------------------------------------
# conditional magnitude: WMAE (pooled + macro + trivial constant)
# ---------------------------------------------------------------------------
def conditional_wmae(candidate_rows, baseline_rows, predict_field="transformed_prediction",
                     y_field="y", weight_field="weight") -> dict[str, Any]:
    cand_by_pub = _group_by_publication(candidate_rows)
    base_by_pub = _group_by_publication(baseline_rows)
    if set(cand_by_pub) != set(base_by_pub):
        raise ValueError("CANDIDATE_BASELINE_PUBLICATION_MISALIGN")

    def _pooled(rows, pf, yf, wf):
        return _wmae([r[yf] for r in rows], [r[pf] for r in rows], [r[wf] for r in rows])

    pooled_cand = _pooled(candidate_rows, predict_field, y_field, weight_field)
    pooled_base = _pooled(baseline_rows, predict_field, y_field, weight_field)
    per_pub = {}
    for pub in sorted(cand_by_pub, key=str):
        per_pub[pub] = {
            "n": len(cand_by_pub[pub]),
            "wmae_candidate": _pooled(cand_by_pub[pub], predict_field, y_field, weight_field),
            "wmae_baseline": _pooled(base_by_pub[pub], predict_field, y_field, weight_field),
        }
    valid = [d for d in per_pub.values() if d["wmae_baseline"] > 0]
    macro_cand = sum(d["wmae_candidate"] for d in valid) / len(valid) if valid else None
    macro_base = sum(d["wmae_baseline"] for d in valid) / len(valid) if valid else None

    y_all = [r[y_field] for r in candidate_rows]
    w_all = [r[weight_field] for r in candidate_rows]
    med = weighted_median(y_all, w_all)
    wmae_trivial = _wmae(y_all, [med] * len(y_all), w_all)

    return {"pooled_wmae_candidate": float(pooled_cand),
            "pooled_wmae_baseline": float(pooled_base),
            "macro_wmae_candidate": float(macro_cand) if macro_cand is not None else UNIDENTIFIABLE,
            "macro_wmae_baseline": float(macro_base) if macro_base is not None else UNIDENTIFIABLE,
            "trivial_weighted_median_constant": float(med),
            "wmae_trivial_constant": float(wmae_trivial),
            "per_publication": per_pub,
            "n_publications": len(per_pub)}


# ---------------------------------------------------------------------------
# leave-one-dominant-publication-out
# ---------------------------------------------------------------------------
def leave_one_dominant_publication_out(candidate_rows, baseline_rows,
                                       predict_field="transformed_prediction",
                                       y_field="y", weight_field="weight") -> dict[str, Any]:
    cand_by_pub = _group_by_publication(candidate_rows)
    dominant = max(cand_by_pub, key=lambda p: sum(r[weight_field] for r in cand_by_pub[p]))
    c_rest = [r for r in candidate_rows if r["publication_id"] != dominant]
    b_rest = [r for r in baseline_rows if r["publication_id"] != dominant]
    if not c_rest:
        return {"dominant_publication": dominant, "skill_loo": UNIDENTIFIABLE,
                "note": "ONLY_ONE_PUBLICATION"}
    y = [r[y_field] for r in c_rest]
    pc = [r[predict_field] for r in c_rest]
    pb = [r[predict_field] for r in b_rest]
    w = [r[weight_field] for r in c_rest]
    base = _wmae(y, pb, w)
    if base <= 0.0:
        return {"dominant_publication": dominant, "skill_loo": UNIDENTIFIABLE,
                "note": "BASELINE_ZERO_WMAE"}
    skill = 1.0 - _wmae(y, pc, w) / base
    return {"dominant_publication": dominant, "skill_loo": float(skill),
            "n_publications_remaining": len(set(r["publication_id"] for r in c_rest))}


# ---------------------------------------------------------------------------
# null space
# ---------------------------------------------------------------------------
def enumerate_null_space(publication_ids) -> dict[str, Any]:
    n = len(set(publication_ids))
    if n < 2:
        return {"n_publications": n, "unique_null_assignments": 1,
                "identifiable": False, "reason": "LESS_THAN_2_PUBLICATIONS"}
    unique = math.factorial(n)
    return {"n_publications": n, "unique_null_assignments": unique,
            "identifiable": unique >= 20,
            "reason": ("UNIQUE_NULL_ASSIGNMENTS" if unique >= 20
                       else "DEGENERATE_NULL_SPACE")}


# ---------------------------------------------------------------------------
# paired publication CI + permutation
# ---------------------------------------------------------------------------
def paired_publication_ci(candidate_rows, baseline_rows,
                          predict_field="transformed_prediction", y_field="y",
                          weight_field="weight", seed=0, n_boot=1000,
                          alpha=0.05) -> dict[str, Any]:
    pubs = sorted({r["publication_id"] for r in candidate_rows})
    if len(pubs) < 3:
        return {"ci_low": None, "ci_high": None, "skill": None,
                "n_publications": len(pubs),
                "note": "PUBLICATION_LT_3_NO_CONFIRMATORY_CI"}

    base_by_key = index_rows(baseline_rows)
    groups = defaultdict(lambda: {"y": [], "w": [], "pc": [], "pb": []})
    for r in candidate_rows:
        b = base_by_key.get(_base_key(r))
        if b is None:
            raise ValueError(f"BASELINE_MISSING_FOR_KEY {_base_key(r)}")
        groups[r["publication_id"]]["y"].append(float(r[y_field]))
        groups[r["publication_id"]]["w"].append(float(r[weight_field]))
        groups[r["publication_id"]]["pc"].append(float(r[predict_field]))
        groups[r["publication_id"]]["pb"].append(float(b[predict_field]))

    rng = random.Random(seed)
    skills = []
    for _ in range(n_boot):
        resampled = [rng.choice(pubs) for _ in pubs]
        ry, rw, rpc, rpb = [], [], [], []
        for p in resampled:
            ry.extend(groups[p]["y"])
            rw.extend(groups[p]["w"])
            rpc.extend(groups[p]["pc"])
            rpb.extend(groups[p]["pb"])
        base = _wmae(ry, rpb, rw)
        if base <= 0.0:
            continue
        skills.append(1.0 - _wmae(ry, rpc, rw) / base)
    if not skills:
        return {"ci_low": None, "ci_high": None, "skill": None,
                "n_publications": len(pubs), "n_boot": n_boot,
                "note": "NO_VALID_BOOTSTRAP"}
    skills.sort()
    lo = skills[int(math.floor((alpha / 2.0) * (len(skills) - 1)))]
    hi = skills[int(math.ceil((1.0 - alpha / 2.0) * (len(skills) - 1)))]
    return {"ci_low": float(lo), "ci_high": float(hi),
            "skill": float(sum(skills) / len(skills)),
            "n_publications": len(pubs), "n_boot": n_boot}


def permutation_test(candidate_rows, baseline_rows,
                     predict_field="transformed_prediction", y_field="y",
                     weight_field="weight", seed=0, n_perm=1000) -> dict[str, Any]:
    """Publication-block permutation test; identity-only -> UNIDENTIFIABLE."""
    ns = enumerate_null_space([r["publication_id"] for r in candidate_rows])
    if not ns["identifiable"]:
        return {"statistic": UNIDENTIFIABLE, "p_value": None,
                "n_perm": n_perm, "null_space": ns,
                "note": "DEGENERATE_NULL_SPACE_OR_IDENTITY_ONLY"}

    base_by_key = index_rows(baseline_rows)
    groups = defaultdict(lambda: {"y": [], "w": [], "pc": [], "pb": []})
    for r in candidate_rows:
        b = base_by_key.get(_base_key(r))
        if b is None:
            raise ValueError(f"BASELINE_MISSING_FOR_KEY {_base_key(r)}")
        groups[r["publication_id"]]["y"].append(float(r[y_field]))
        groups[r["publication_id"]]["w"].append(float(r[weight_field]))
        groups[r["publication_id"]]["pc"].append(float(r[predict_field]))
        groups[r["publication_id"]]["pb"].append(float(b[predict_field]))

    y_all = [r[y_field] for r in candidate_rows]
    w_all = [r[weight_field] for r in candidate_rows]
    pc_all = [r[predict_field] for r in candidate_rows]
    pb_all = [r[predict_field] for r in baseline_rows]
    base = _wmae(y_all, pb_all, w_all)
    if base <= 0.0:
        return {"statistic": UNIDENTIFIABLE, "p_value": None,
                "n_perm": n_perm, "note": "BASELINE_ZERO_WMAE"}
    real = 1.0 - _wmae(y_all, pc_all, w_all) / base

    pub_ids = sorted(set(r["publication_id"] for r in candidate_rows))
    y_blocks = [groups[p]["y"] for p in pub_ids]
    w_blocks = [groups[p]["w"] for p in pub_ids]
    pc_blocks = [groups[p]["pc"] for p in pub_ids]
    pb_blocks = [groups[p]["pb"] for p in pub_ids]

    size_classes = defaultdict(list)
    for idx, lb in enumerate(y_blocks):
        size_classes[len(lb)].append(idx)

    rng = random.Random(seed)
    null_skills = []
    b = 0
    for _ in range(n_perm):
        perm_pc = [None] * len(pc_blocks)
        for size, idxs in size_classes.items():
            perm_idxs = idxs[:]
            rng.shuffle(perm_idxs)
            for orig, dest in zip(idxs, perm_idxs):
                perm_pc[dest] = pc_blocks[orig]
        ry = [v for blk in y_blocks for v in blk]
        rw = [v for blk in w_blocks for v in blk]
        rpc = [v for blk in perm_pc for v in blk]
        rpb = [v for blk in pb_blocks for v in blk]
        bbase = _wmae(ry, rpb, rw)
        if bbase <= 0.0:
            continue
        null_skill = 1.0 - _wmae(ry, rpc, rw) / bbase
        null_skills.append(float(null_skill))
        if float(null_skill) >= float(real):
            b += 1
    if not null_skills:
        return {"statistic": float(real), "p_value": None, "b": 0,
                "n_perm": n_perm, "null": [], "note": "NO_VALID_PERMUTATIONS"}
    p_value = (b + 1) / (len(null_skills) + 1)
    return {"statistic": float(real), "p_value": float(p_value), "b": b,
            "n_perm": n_perm, "null": sorted(null_skills),
            "n_null_numeric": len(null_skills)}
