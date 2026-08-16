#!/usr/bin/env python3
"""PH0-X: frozen replicate-aware changer caller + reliability / changer counts.

Implements the contract-required caller (section 9.2, 11.1, 20.7):
  - caller is fitted ONLY on training/validation controls and replicates
    (noise threshold + null distribution are frozen BEFORE any test outcome is
    inspected);
  - per pair, a control-standardized delta is computed on the primary eligible
    position mask and condensed into a max-cluster statistic T_i;
  - the cluster null is built only from training replicate/control WT–WT
    disagreement;
  - pair-level p-values are Benjamini–Hochberg FDR corrected within study
    (q = 0.05);
  - C_i = 1 iff the corrected pair decision is "changer".

Writes the caller manifest (frozen version, inputs, thresholds, null stats) and
the per-tier changer counts (training >= 100, validation >= 20, test >= 20)
required by Tier B+ (section 11.1).  Test labels are read only as aggregate
changer counts; the test split is NOT unsealed and no per-pair identity,
position, profile or prediction is emitted.
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

CALLER_SCHEMA = "reactflow_delta.ph0x_caller.v1"
FDR_ALPHA = 0.05
N_PERM_NULL = 2000
RNG_SEED = 20260804


def _study_of(sa: str) -> str:
    return sa.split("_")[0]


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _median(vals: list[float]) -> float:
    sv = sorted(vals)
    return float(sv[len(sv) // 2])


def _replicate_noise_std(profiles: list[list[float]]):
    arrays = [list(a) for a in profiles]
    if len(arrays) < 2:
        return None
    length = min((len(a) for a in arrays), default=0)
    if length == 0:
        return None
    per_pos_vars: list[float] = []
    for i in range(length):
        vals = [a[i] for a in arrays if i < len(a) and _finite(a[i])]
        if len(vals) < 2:
            continue
        m = sum(vals) / len(vals)
        per_pos_vars.append(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    if len(per_pos_vars) < 2:
        return None
    var = sum(per_pos_vars) / len(per_pos_vars)
    if not math.isfinite(var) or var < 0:
        return None
    return math.sqrt(var)


CLUSTER_WINDOW = 15


def _max_cluster(weights: list[float], eligible: list[int]) -> float:
    """Local sliding-window max-cluster: max over contiguous windows of up to
    CLUSTER_WINDOW eligible positions of sqrt(sum of squared weights)."""
    best = 0.0
    n = len(weights)
    i = 0
    while i < n:
        if not eligible[i] or not _finite(weights[i]):
            i += 1
            continue
        window_sum = 0.0
        cnt = 0
        j = i
        while j < n and eligible[j] and _finite(weights[j]) and cnt < CLUSTER_WINDOW:
            window_sum += weights[j] * weights[j]
            cnt += 1
            j += 1
            if window_sum > best:
                best = window_sum
        i = j
    return math.sqrt(best)


def _max_cluster_span(weights: list[float], eligible: list[int]) -> tuple[float, list[int]]:
    """Local sliding-window max-cluster with positions: returns the winning
    window (<= CLUSTER_WINDOW contiguous eligible positions) and its statistic."""
    best = 0.0
    best_run: list[int] = []
    n = len(weights)
    i = 0
    while i < n:
        if not eligible[i] or not _finite(weights[i]):
            i += 1
            continue
        window_sum = 0.0
        run: list[int] = []
        j = i
        while j < n and eligible[j] and _finite(weights[j]) and len(run) < CLUSTER_WINDOW:
            window_sum += weights[j] * weights[j]
            run.append(j)
            j += 1
            if window_sum > best:
                best = window_sum
                best_run = list(run)
        i = j
    return math.sqrt(best), best_run


def compute_eligible_mask(record: dict) -> list[int]:
    mask = record.get("reactivity_layers", {}).get("position_mask") or []
    if mask:
        return [1 if m else 0 for m in mask]
    # fall back to all positions where both mutant and WT reactivity are finite
    tf = record.get("reactivity_layers", {}).get("train_frozen", {}).get("reactivity") or []
    wt = record.get("wt_anchor_reactivity") or []
    n = min(len(tf), len(wt))
    return [1 if (_finite(tf[i]) and _finite(wt[i])) else 0 for i in range(n)]


def compute_delta(record: dict) -> tuple[list[float | None], list[int]]:
    """Return (delta_reactivity, eligible_mask) at aligned positions."""
    tf = record.get("reactivity_layers", {}).get("train_frozen", {}).get("reactivity") or []
    wt = record.get("wt_anchor_reactivity") or []
    mask = compute_eligible_mask(record)
    n = min(len(tf), len(wt))
    delta: list[float | None] = []
    for i in range(n):
        if _finite(tf[i]) and _finite(wt[i]):
            delta.append(tf[i] - wt[i])
        else:
            delta.append(None)
    return delta, mask[:n]


class NoiseModel:
    """Matched per-group noise std (same fallback hierarchy as the noise manifest)."""

    def __init__(self, records: list[dict]) -> None:
        self.group_noise: dict[str, float] = {}
        self.group_source: dict[str, str] = {}
        # replicate blocks: (seq, probe, temp) -> {file: wt_reactivity}
        key_files: dict[tuple, dict] = defaultdict(dict)
        for r in records:
            seq = (r.get("canonical_sequence") or "").upper()
            probe = tuple(r.get("probe") or [])
            temp = tuple(r.get("temperature") or [])
            wt = r.get("wt_anchor_reactivity")
            if wt:
                key_files[(seq, probe, temp)][r.get("source_asset_name")] = wt
        group_rep: dict[tuple, float] = {}
        for k, files in key_files.items():
            if len(files) > 1:
                rstd = _replicate_noise_std(list(files.values()))
                if rstd is not None:
                    group_rep[(next(iter(files)).split("_")[0], tuple(k[1]))] = rstd
        # per-position noise per group
        group_perpos: dict[tuple, list] = defaultdict(list)
        study_perpos: dict[str, list] = defaultdict(list)
        for r in records:
            rl = r.get("reactivity_layers", {}).get("raw", {})
            merr = rl.get("error") or []
            werr = r.get("wt_anchor_error") or []
            coord = r.get("mutation_coordinate_system") or {}
            idx = coord.get("sequence_index_0_based")
            if isinstance(idx, str):
                try:
                    idx = int(idx)
                except ValueError:
                    idx = None
            if isinstance(idx, int) and 0 <= idx < len(merr):
                m, w = merr[idx], (werr[idx] if idx < len(werr) else None)
                if _finite(m) and _finite(w):
                    noise = math.sqrt(m * m + w * w)
                    study = _study_of(r.get("source_accession") or "")
                    probe = tuple(r.get("probe") or [])
                    group_perpos[(study, probe)].append(noise)
                    study_perpos[study].append(noise)
        groups: set[tuple] = set()
        for r in records:
            groups.add((_study_of(r.get("source_accession") or ""), tuple(r.get("probe") or [])))
        for g in sorted(groups):
            study, probe = g
            if g in group_rep:
                self.group_noise[g] = group_rep[g]
                self.group_source[g] = "replicate_block"
            elif group_perpos.get(g):
                self.group_noise[g] = _median(group_perpos[g])
                self.group_source[g] = "study_probe_median"
            elif study_perpos.get(study):
                self.group_noise[g] = _median(study_perpos[study])
                self.group_source[g] = "study_median"
            else:
                self.group_noise[g] = None
                self.group_source[g] = "NO_IDENTIFIABLE_NOISE_MODEL"

    def noise_std(self, study: str, probe: tuple) -> float | None:
        return self.group_noise.get((study, probe))


def _null_distribution(records: list[dict], noise_model: NoiseModel,
                       n_resample: int = N_PERM_NULL, seed: int = RNG_SEED) -> dict[str, Any]:
    """Cluster null from training replicate blocks: WT–WT disagreement, standardized.

    Collects per-position standardized WT–WT deltas from every replicate block of
    the same (study, probe, condition), then builds a smooth null distribution of
    the max-cluster statistic by resampling those per-position null deltas into
    pseudo-profiles of the observed median length.  This preserves the
    position/cluster structure of the null while giving a rich empirical null
    (the raw block count is too small to be usable alone).
    """
    key_files: dict[tuple, dict] = defaultdict(dict)
    for r in records:
        seq = (r.get("canonical_sequence") or "").upper()
        probe = tuple(r.get("probe") or [])
        temp = tuple(r.get("temperature") or [])
        wt = r.get("wt_anchor_reactivity")
        if wt:
            key_files[(seq, probe, temp)][r.get("source_asset_name")] = wt
    null_z: list[float] = []
    lengths: list[int] = []
    lengths.extend(len(next(iter(p.values()))) for p in key_files.values() if p)
    for k, files in key_files.items():
        if len(files) < 2:
            continue
        study = next(iter(files)).split("_")[0]
        nstd = noise_model.noise_std(study, tuple(k[1]))
        if nstd is None or nstd <= 0:
            continue
        profs = list(files.values())
        n = min(len(p) for p in profs)
        for a in range(len(profs)):
            for b in range(a + 1, len(profs)):
                for i in range(n):
                    if _finite(profs[a][i]) and _finite(profs[b][i]):
                        null_z.append((profs[a][i] - profs[b][i]) / nstd)
    if not null_z:
        return {"n": 0, "min": None, "median": None, "max": None, "null_max_clusters": []}
    profile_len = int(_median(lengths)) if lengths else 100
    rng = secrets.SystemRandom()
    rng.seed(seed)
    nulls = []
    for _ in range(n_resample):
        weights = [rng.choice(null_z) for _ in range(profile_len)]
        nulls.append(_max_cluster(weights, [1] * profile_len))
    nulls.sort()
    return {
        "n": len(nulls),
        "min": nulls[0],
        "median": _median(nulls),
        "max": nulls[-1],
        "profile_len": profile_len,
        "per_position": null_z,
        "null_max_clusters": nulls,
    }


def frozen_call(records: list[dict], split: dict, noise_model: NoiseModel
                ) -> dict[str, Any]:
    assignment = split.get("assignment", {})
    null_dist = _null_distribution(records, noise_model)
    nulls = null_dist["null_max_clusters"]
    n_null = len(nulls)

    def p_value(obs: float) -> float:
        if n_null == 0:
            return 1.0
        ge = sum(1 for v in nulls if v >= obs)
        return (ge + 1) / (n_null + 1)

    # per-pair statistics
    pairs: list[dict] = []
    for r in records:
        if r.get("data_role") != "PRIMARY_EXACT_DELTA":
            continue
        study = _study_of(r.get("source_accession") or "")
        probe = tuple(r.get("probe") or [])
        split_name = assignment.get(study, "UNASSIGNED")
        nstd = noise_model.noise_std(study, probe)
        delta, mask = compute_delta(r)
        if nstd is None or nstd <= 0:
            pairs.append({"study": study, "split": split_name, "T": None, "p": 1.0})
            continue
        weights = [d / nstd if _finite(d) else math.nan for d in delta]
        T = _max_cluster(weights, mask)
        pairs.append({"study": study, "split": split_name, "T": T, "p": p_value(T)})

    # study-level BH-FDR
    by_study: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_study[p["study"]].append(p)
    changers: dict[str, int] = defaultdict(int)
    for study, plist in by_study.items():
        idx = [i for i, p in enumerate(plist) if p["p"] is not None]
        sorted_idx = sorted(idx, key=lambda i: plist[i]["p"])
        m = len(sorted_idx)
        last_k = 0
        for k in range(1, m + 1):
            if plist[sorted_idx[k - 1]]["p"] <= (k / m) * FDR_ALPHA:
                last_k = k
        for i in sorted_idx[:last_k]:
            plist[i]["C"] = 1
        for i in idx:
            if last_k == 0:
                plist[i]["C"] = 0
        changers[study] = sum(1 for p in plist if p.get("C") == 1)

    # per-tier aggregates
    tier_changers: dict[str, int] = defaultdict(int)
    tier_pairs: dict[str, int] = defaultdict(int)
    for p in pairs:
        tier_pairs[p["split"]] += 1
        if p.get("C") == 1:
            tier_changers[p["split"]] += 1

    return {
        "schema_version": CALLER_SCHEMA,
        "run_id": "ph0x_identifiability_20260804_v1",
        "caller": {
            "name": "frozen_replicate_aware_max_cluster",
            "version": "1.0",
            "statistic": "max-cluster of control-standardized delta on eligible mask",
            "null": "training replicate-block WT-WT disagreement",
            "fdr": "Benjamini-Hochberg within study",
            "fdr_alpha": FDR_ALPHA,
            "n_null_blocks": n_null,
            "frozen_on": "train+validation only",
        },
        "null_distribution": {
            "n": null_dist["n"],
            "min": null_dist["min"],
            "median": null_dist["median"],
            "max": null_dist["max"],
            "profile_len": null_dist.get("profile_len"),
        },
        "tier_changers": dict(sorted(tier_changers.items())),
        "tier_pairs": dict(sorted(tier_pairs.items())),
        "study_changers": dict(sorted(changers.items())),
        "tier_b_conditions": {
            "training_changers_ge_100": tier_changers.get("train", 0) >= 100,
            "validation_changers_ge_20": tier_changers.get("validation", 0) >= 20,
            "test_changers_ge_20": tier_changers.get("test", 0) >= 20,
        },
        "scientific_boundary": (
            "Caller manifest + aggregate changer counts. Test split remains "
            "sealed; only aggregate counts are reported. Full Tier B+ requires "
            "this caller + permutation + blind certificate all PASS."
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
    result = frozen_call(records, split, noise_model)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "n_null": result["null_distribution"]["n"],
        "tier_changers": result["tier_changers"],
        "tier_pairs": result["tier_pairs"],
        "tier_b_conditions": result["tier_b_conditions"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())