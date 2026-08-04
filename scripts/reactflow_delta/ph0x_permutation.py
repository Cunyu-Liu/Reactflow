#!/usr/bin/env python3
"""PH0-X: group-aware permutation test (contract section 13.3, 20.7).

Tests whether the real caller signal exceeds a legal *group-aware* permutation
null.  Permutation is performed ONLY within frozen exchangeability blocks
(study) and preserves the study/parent/shared-WT/mask structure.

The statistic is the number of pairs whose max-cluster statistic exceeds the
frozen replicate-based null 95th percentile AND whose mutation position lies
inside that winning cluster.  This tests the MUTATION-POSITION-SPECIFIC signal
(the co-occurrence of the mutation with the strongest delta region), which is
the object that the exact-mutation delta task (Estimand A) depends on.

Permutation: within each study, the pair-level control-standardized delta
vectors are shuffled across pairs, so the study's marginal delta-amplitude
distribution is preserved while the specific mutation->position linkage is
destroyed.  If the real statistic is not in the upper tail of the permutation
null, the method claim (section 13.4) stops and PH0-X FAILS/REPORT-X.
"""

from __future__ import annotations

import argparse
import json
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Any

from ph0x_caller import (
    CLUSTER_WINDOW,
    NoiseModel,
    _finite,
    _max_cluster,
    _max_cluster_span,
    _null_distribution,
    _study_of,
    compute_delta,
)

PERM_SCHEMA = "reactflow_delta.ph0x_permutation.v1"
N_PERM = 100
NULL_QUANTILE = 0.95
SEED = 20260804


def _mutation_pos(record: dict) -> int | None:
    coord = record.get("mutation_coordinate_system") or {}
    idx = coord.get("sequence_index_0_based")
    if isinstance(idx, str):
        try:
            return int(idx)
        except ValueError:
            return None
    return idx if isinstance(idx, int) else None


def _pair_cells(records: list[dict], split: dict, noise_model: NoiseModel) -> list[dict]:
    assignment = split.get("assignment", {})
    cells = []
    for r in records:
        if r.get("data_role") != "PRIMARY_EXACT_DELTA":
            continue
        study = _study_of(r.get("source_accession") or "")
        probe = tuple(r.get("probe") or [])
        split_name = assignment.get(study, "UNASSIGNED")
        nstd = noise_model.noise_std(study, probe)
        delta, mask = compute_delta(r)
        mpos = _mutation_pos(r)
        if nstd is None or nstd <= 0:
            cells.append({"study": study, "split": split_name, "weights": None,
                          "mask": mask, "mpos": mpos})
            continue
        weights = [d / nstd if _finite(d) else None for d in delta]
        cells.append({"study": study, "split": split_name, "weights": weights,
                      "mask": mask, "mpos": mpos})
    return cells


def _statistic(cells: list[dict], null95: float) -> int:
    count = 0
    for c in cells:
        if not c["weights"] or c["mpos"] is None:
            continue
        T, span = _max_cluster_span(c["weights"], c["mask"])
        if T > null95 and c["mpos"] in span:
            count += 1
    return count


def _permutation_null(cells: list[dict], null95: float, seed: int) -> list[int]:
    """Run N_PERM group-aware permutations; return sorted null statistic list."""
    by_study: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        # exchangeability block = (study, length) so shuffling preserves the
        # per-position mask structure and keeps vectors length-aligned.
        by_study[(c["study"], len(c["mask"]) if c["mask"] else 0)].append(c)
    rng = secrets.SystemRandom()
    rng.seed(seed)
    null_stats: list[int] = []
    for _ in range(N_PERM):
        permuted = []
        for (study, length), plist in by_study.items():
            wvecs = [c["weights"] for c in plist]
            rng.shuffle(wvecs)
            for c, w in zip(plist, wvecs):
                permuted.append({"study": study, "weights": w, "mask": c["mask"],
                                 "mpos": c["mpos"]})
        null_stats.append(_statistic(permuted, null95))
    null_stats.sort()
    return null_stats


def _p_value(null_stats: list[int], real: int) -> float:
    ge = sum(1 for v in null_stats if v >= real)
    return (ge + 1) / (N_PERM + 1)


def run_permutation(records: list[dict], split: dict, noise_model: NoiseModel) -> dict[str, Any]:
    null_dist = _null_distribution(records, noise_model)
    nulls = null_dist["null_max_clusters"]
    if not nulls:
        return {"status": "FAIL", "reason": "no replicate null available"}
    null95 = sorted(nulls)[min(len(nulls) - 1, int(NULL_QUANTILE * len(nulls)))]
    cells = _pair_cells(records, split, noise_model)
    real = _statistic(cells, null95)
    null_stats = _permutation_null(cells, null95, SEED)
    p_value = _p_value(null_stats, real)
    pass_null = p_value <= 0.05

    # leave-one-study-out sensitivity (contract 20.7: no single-study driven)
    studies = sorted({c["study"] for c in cells if c["weights"]})
    loso: dict[str, Any] = {}
    for s in studies:
        sub = [c for c in cells if c["study"] != s]
        sub_real = _statistic(sub, null95)
        if sub_real == 0:
            loso[s] = {"real_statistic": 0, "p_value": None, "pass": False}
            continue
        sub_null = _permutation_null(sub, null95, SEED + hash(s) % 100000)
        sub_p = _p_value(sub_null, sub_real)
        loso[s] = {
            "real_statistic": sub_real,
            "null_median": sub_null[len(sub_null) // 2],
            "null_max": sub_null[-1],
            "p_value": round(sub_p, 6),
            "pass": sub_p <= 0.05,
        }
    all_loso_pass = all(v["pass"] for v in loso.values())

    return {
        "schema_version": PERM_SCHEMA,
        "run_id": "ph0x_identifiability_20260804_v1",
        "statistic": "n_pairs with max-cluster > replicate-null 95th pct AND mutation pos in winning cluster",
        "exchangeability_block": "study (preserves parent/shared-WT/mask structure)",
        "permutation": "shuffle within-study control-standardized delta vectors across pairs",
        "n_permutations": N_PERM,
        "null_quantile": NULL_QUANTILE,
        "null95": null95,
        "real_statistic": real,
        "null_statistic": {
            "min": null_stats[0],
            "median": null_stats[len(null_stats) // 2],
            "max": null_stats[-1],
        },
        "p_value": round(p_value, 6),
        "pass_real_gt_group_aware_null": pass_null,
        "leave_one_study_out": loso,
        "no_single_study_driven": all_loso_pass,
        "scientific_boundary": (
            "Group-aware permutation only. True-signal-under-null evidence for "
            "PH0-X; full method claim requires section 13.4 conditions."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    records = [json.loads(l) for l in open(args.canonical_jsonl, encoding="utf-8") if l.strip()]
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    noise_model = NoiseModel(records)
    result = run_permutation(records, split, noise_model)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "real_statistic": result["real_statistic"],
        "null_statistic": result["null_statistic"],
        "p_value": result["p_value"],
        "pass_real_gt_group_aware_null": result["pass_real_gt_group_aware_null"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())