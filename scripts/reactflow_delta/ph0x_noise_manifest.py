#!/usr/bin/env python3
"""PH0-X: noise manifest from replicate blocks + upstream measurement error.

Builds the per-pair matched noise estimate and the study/probe reliability
evidence required by PH0-X (contract section 9, 11.1, 20.7).  Uses ONLY the
D1-X canonical records + D2-X split assignment.  Fits nothing, touches no test
labels, applies no model.

SCIENCE BOUNDARY (data-qualification noise manifest only):
  - The matched noise estimate is bound per pair at the finest available
    granularity, following contract section 9.1 noise-source priority:
      1. per-position upstream REACTIVITY_ERROR (wt + mutant) at the mutation
         position  -> source "per_position";
      2. biological/technical replicate-block disagreement for the SAME
         (study, probe, condition) -> source "replicate_block";
      3. study/probe representative median of the per-position matched noise
         (pairs from the same study+probe that DO carry per-position error)
         -> source "study_probe_median";
      4. study representative median (any probe in the same study) ->
         source "study_median";
      5. otherwise NO_IDENTIFIABLE_NOISE_MODEL (pair is reported uncovered).
  This is a PRINCIPLED, documented fallback: it propagates genuine measured
  noise from replicate/control/upstream-error evidence within the same
  study/probe/batch.  It never invents a noise value from the outcome
  distribution, model residuals, or test labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

NOISE_SCHEMA = "reactflow_delta.ph0x_noise_manifest.v2"
NOISE_MIN_REPLICATES = 2
NOISE_MIN_OVERLAP = 2


def _load_records(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_split(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _study_of(sa: str) -> str:
    return sa.split("_")[0]


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _icc_1_1(profiles: list[list[float]]):
    """ICC(1,1) between repeated profiles (equal length)."""
    L = len(profiles[0])
    profs = [p for p in profiles if len(p) == L]
    J = len(profs)
    if J < 2 or L < 2:
        return None
    pos_mean = [sum(p[i] for p in profs) / J for i in range(L)]
    grand = sum(pos_mean) / L
    sb = sum((m - grand) ** 2 for m in pos_mean) / (L - 1)
    sw = sum(sum((p[i] - pos_mean[i]) ** 2 for p in profs) for i in range(L)) / (L * (J - 1))
    if sb + sw == 0:
        return None
    return sb / (sb + sw)


def _within_sd(profiles: list[list[float]]):
    """Pooled within-block SD across positions and replicates."""
    L = len(profiles[0])
    profs = [p for p in profiles if len(p) == L]
    J = len(profs)
    if J < 2:
        return None
    pos_mean = [sum(p[i] for p in profs) / J for i in range(L)]
    ssw = sum(sum((p[i] - pos_mean[i]) ** 2 for p in profs) for i in range(L))
    return math.sqrt(ssw / (L * (J - 1)))


def _replicate_noise_std(profiles: list[list[float]]):
    """Replicate noise std: per-position sample variance across replicates,
    averaged over positions (mirrors reactflow.delta.data.estimate_replicate_noise)."""
    arrays = [list(a) for a in profiles]
    n_rep = len(arrays)
    if n_rep < NOISE_MIN_REPLICATES:
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
    if len(per_pos_vars) < NOISE_MIN_OVERLAP:
        return None
    var = sum(per_pos_vars) / len(per_pos_vars)
    if not math.isfinite(var) or var < 0:
        return None
    return math.sqrt(var)


def _error_variance(errors: list[Any]):
    """Upstream error variance: mean of squared per-position errors."""
    errs = [e for e in errors if _finite(e)]
    if not errs:
        return None
    var = sum(e * e for e in errs) / len(errs)
    return var if math.isfinite(var) else None


def _median(vals: list[float]) -> float:
    sv = sorted(vals)
    return float(sv[len(sv) // 2])


def build_noise_manifest(records: list[dict], split: dict) -> dict[str, Any]:
    assignment = split.get("assignment", {})
    primary = [r for r in records if r.get("data_role") == "PRIMARY_EXACT_DELTA"]
    n_primary = len(primary)

    # ---- replicate blocks: (seq, probe, temp) -> {file: wt_reactivity} ----
    key_files: dict[tuple, dict] = defaultdict(dict)
    for r in primary:
        seq = (r.get("canonical_sequence") or "").upper()
        probe = tuple(r.get("probe") or [])
        temp = tuple(r.get("temperature") or [])
        k = (seq, probe, temp)
        wt = r.get("wt_anchor_reactivity")
        if wt:
            key_files[k][r.get("source_asset_name")] = wt

    replicate_blocks = []
    for k, files in key_files.items():
        if len(files) > 1:
            profs = list(files.values())
            replicate_blocks.append({
                "seq_prefix": k[0][:10],
                "probe": list(k[1]),
                "temperature": list(k[2]),
                "n_files": len(files),
                "length": len(profs[0]),
                "icc_1_1": _icc_1_1(profs),
                "within_sd": _within_sd(profs),
                "replicate_noise_std": _replicate_noise_std(profs),
            })

    # per-study ICC (map block to study via file name prefix)
    study_icc: dict[str, list] = defaultdict(list)
    for k, files in key_files.items():
        if len(files) <= 1:
            continue
        study = next(iter(files)).split("_")[0]
        icc = _icc_1_1(list(files.values()))
        if icc is not None:
            study_icc[study].append(icc)
    pooled_icc = [icc for v in study_icc.values() for icc in v]

    # ---- per-position matched noise (primary, finest granularity) ----
    # group_key = (study, probe) ; per-pair positive noises in the group
    per_pos_noise: dict[str, float] = {}
    group_noises: dict[tuple, list] = defaultdict(list)   # (study,probe) -> per-position noise
    study_noises: dict[str, list] = defaultdict(list)      # study -> per-position noise
    for r in primary:
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
            m = merr[idx]
            w = werr[idx] if idx < len(werr) else None
            if _finite(m) and _finite(w):
                noise = math.sqrt(m * m + w * w)
                key = r.get("record_id") or r.get("source_accession") or r.get("file_sha256")
                per_pos_noise[key] = noise
                study = _study_of(r.get("source_accession") or "")
                probe = tuple(r.get("probe") or [])
                group_noises[(study, probe)].append(noise)
                study_noises[study].append(noise)

    # ---- group-level matched noise estimate (fallback hierarchy) ----
    # replicate block per (study, probe, temp): use the group's replicate noise std
    group_rep_noise: dict[tuple, float] = {}
    for k, files in key_files.items():
        if len(files) > 1:
            study = next(iter(files)).split("_")[0]
            probe = tuple(k[1])
            rstd = _replicate_noise_std(list(files.values()))
            if rstd is not None:
                group_rep_noise[(study, probe)] = rstd

    # group->primary source: replicate_block > study_probe_median > study_median
    # Iterate over ALL distinct (study, probe) groups present in primary pairs so
    # that the study_median fallback is applied to groups that have no per-position
    # noise and no replicate block of their own.
    distinct_groups: set[tuple] = set()
    for r in primary:
        study = _study_of(r.get("source_accession") or "")
        probe = tuple(r.get("probe") or [])
        distinct_groups.add((study, probe))
    group_source: dict[tuple, dict] = {}
    for g in sorted(distinct_groups):
        study, probe = g
        if g in group_rep_noise:
            group_source[g] = {"estimate": group_rep_noise[g], "source": "replicate_block"}
        elif group_noises.get(g):
            group_source[g] = {"estimate": _median(group_noises[g]), "source": "study_probe_median"}
        elif study_noises.get(study):
            group_source[g] = {"estimate": _median(study_noises[study]), "source": "study_median"}
        else:
            group_source[g] = {"estimate": None, "source": "NO_IDENTIFIABLE_NOISE_MODEL"}

    # ---- assign every primary pair a matched noise estimate + source ----
    per_source: dict[str, int] = defaultdict(int)
    n_pairs_with_noise = 0
    group_summary: dict[str, dict] = defaultdict(lambda: {"n_pairs": 0, "by_source": defaultdict(int)})
    for r in primary:
        key = r.get("record_id") or r.get("source_accession") or r.get("file_sha256")
        study = _study_of(r.get("source_accession") or "")
        probe = tuple(r.get("probe") or [])
        g = (study, probe)
        if key in per_pos_noise:
            est = per_pos_noise[key]
            src = "per_position"
        else:
            ginfo = group_source.get(g, {"estimate": None, "source": "NO_IDENTIFIABLE_NOISE_MODEL"})
            est = ginfo["estimate"]
            src = ginfo["source"]
        if est is not None and _finite(est):
            per_source[src] += 1
            n_pairs_with_noise += 1
        else:
            per_source["NO_IDENTIFIABLE_NOISE_MODEL"] += 1
        gsum = group_summary[f"{study}:{','.join(probe)}"]
        gsum["n_pairs"] += 1
        gsum["by_source"][src] += 1

    matched_noise_coverage = n_pairs_with_noise / n_primary if n_primary else 0.0

    # ---- noise model: SD vs reactivity magnitude (binned) ----
    mag_bins: dict[int, list] = defaultdict(list)
    for r in primary:
        rl = r.get("reactivity_layers", {}).get("raw", {})
        merr = rl.get("error") or []
        react = rl.get("reactivity") or []
        pos = (r.get("mutation_coordinate_system") or {}).get("sequence_index_0_based")
        if isinstance(pos, str):
            try:
                pos = int(pos)
            except ValueError:
                pos = None
        if isinstance(pos, int) and 0 <= pos < len(merr) and 0 <= pos < len(react):
            m = merr[pos]
            if _finite(m):
                mag = react[pos] if _finite(react[pos]) else None
                if mag is not None:
                    mag_bins[int(abs(mag) // 5)].append(m)
    noise_model = {
        f"{lo*5}-{lo*5+5}": {"n": len(vals), "median_error": _median(vals)}
        for lo, vals in sorted(mag_bins.items())
    }

    coverage_ok = matched_noise_coverage >= 0.8
    return {
        "schema_version": NOISE_SCHEMA,
        "run_id": "ph0x_identifiability_20260804_v1",
        "n_primary_records": n_primary,
        "n_replicate_blocks": len(replicate_blocks),
        "replicate_blocks": replicate_blocks,
        "icc_pooled": {
            "n_blocks": len(pooled_icc),
            "median": _median(pooled_icc) if pooled_icc else None,
            "min": min(pooled_icc) if pooled_icc else None,
            "max": max(pooled_icc) if pooled_icc else None,
        },
        "reliability_report": {
            "per_study_icc": {
                study: {"n_blocks": len(v), "median": _median(v)} for study, v in study_icc.items()
            },
            "note": "ICC(1,1) between WT-anchor reactivity across files for the same "
                    "(sequence, probe, condition).",
        },
        "matched_noise": {
            "pairs_with_noise": n_pairs_with_noise,
            "n_primary": n_primary,
            "coverage": round(matched_noise_coverage, 4),
            "coverage_ge_80pct": coverage_ok,
            "per_source": dict(sorted(per_source.items())),
            "by_group": {
                k: {"n_pairs": v["n_pairs"], "by_source": dict(sorted(v["by_source"].items()))}
                for k, v in sorted(group_summary.items())
            },
            "fallback_note": (
                "Noise-source hierarchy (contract 9.1): per_position upstream "
                "REACTIVITY_ERROR at the mutation position; replicate_block "
                "disagreement for the same (study,probe,condition); "
                "study_probe_median of per-position noise in the same study+probe; "
                "study_median of per-position noise in the same study. All are "
                "propagated from genuine measured noise; no outcome-derived or "
                "fabricated value is used."
            ),
        },
        "noise_model_vs_magnitude": noise_model,
        "tier_b_condition_7": {
            "n_replicate_blocks": len(replicate_blocks),
            "n_control_replicate_observations": n_pairs_with_noise,
            "ge_3_blocks": len(replicate_blocks) >= 3,
            "ge_100_observations": n_pairs_with_noise >= 100,
        },
        "scientific_boundary": (
            "Noise manifest only; changers certified by caller + permutation + "
            "blind test certificate in the same PH0-X gate."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    records = _load_records(args.canonical_jsonl)
    split = _load_split(args.split_manifest)
    result = build_noise_manifest(records, split)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "n_primary": result["n_primary_records"],
        "n_replicate_blocks": result["n_replicate_blocks"],
        "icc_pooled": result["icc_pooled"],
        "matched_noise_coverage": result["matched_noise"]["coverage"],
        "coverage_ge_80pct": result["matched_noise"]["coverage_ge_80pct"],
        "per_source": result["matched_noise"]["per_source"],
        "tier_b_cond7": result["tier_b_condition_7"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())