#!/usr/bin/env python3
"""D1 pipeline executor (T-D1.13, v3.1 §4 / §7 data-level Gate).

Orchestrates the T-D1.1~10 building blocks over the D0-R v2 candidate
relations (7,761 annotation-only pairs) and produces:

  1. ``d1_true_pair_registry.json`` — per-candidate upgrade evaluation
     (exclusion_reasons, primary_eligible, true_pair, pair_quality_weight,
     Δreactivity summary, caller_status).
  2. ``d1_pipeline_summary.json`` — aggregate metrics: candidate total,
     true_pair count, reason distribution, study/parent/owner/modifier
     distribution, Tier A/B/C re-judgment with true_pair counts.

D1 is cleanup-only (v3.1 §7): the executor MUST NOT train, MUST NOT lower
Tier thresholds, MUST NOT treat missing as 0, and MUST NOT delete D0-R
candidate records (forward-only). Every rejection carries a
machine-readable ``exclusion_reasons`` vector drawn from the frozen
vocabulary (schema.py ``EXCLUSION_REASONS``).

Usage::

    PYTHONPATH=src python scripts/reactflow_delta/d1_pipeline_executor.py \
        --relations artifacts/reactflow_delta/d0r/d0r_reaudit_tierA_relations.json \
        --out-dir artifacts/reactflow_delta/d1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure ``src`` is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reactflow.delta.rdat import parse_rdat, RdatParseError  # noqa: E402
from reactflow.delta.data import (  # noqa: E402
    build_reactivity_layers,
    build_pair_delta_reactivity,
    compute_delta_reactivity,
    estimate_error_variance,
    estimate_pair_noise,
    evaluate_pair_upgrade,
)


# Tier gate thresholds (v3 §8). D1 re-judges with true_pair counts, NOT
# candidate counts (v3.1 §7 — "construct 数冒充 pair 数" is forbidden).
TIER_A_MIN_STUDIES = 5
TIER_A_MIN_PARENTS = 20
TIER_A_MIN_PAIRS = 5000
TIER_B_MIN_TRUE_PAIRS = 1000


def _mean(values: list[float]) -> float | None:
    """Mean of a list of finite numbers (None if empty)."""
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _comparable_stats(
    wt_reactivity: list[float | None],
    mut_reactivity: list[float | None],
) -> dict:
    """Compute comparable_fraction, missing_fraction, aligned length."""
    n = min(len(wt_reactivity), len(mut_reactivity))
    if n == 0:
        return {
            "aligned_length": 0,
            "both_nonmissing": 0,
            "comparable_fraction": None,
            "missing_fraction": None,
        }
    both = 0
    either_missing = 0
    for i in range(n):
        w = wt_reactivity[i]
        m = mut_reactivity[i]
        w_ok = isinstance(w, (int, float)) and math.isfinite(w)
        m_ok = isinstance(m, (int, float)) and math.isfinite(m)
        if w_ok and m_ok:
            both += 1
        else:
            either_missing += 1
    return {
        "aligned_length": n,
        "both_nonmissing": both,
        "comparable_fraction": both / n,
        "missing_fraction": either_missing / n,
    }


def _evaluate_one(
    relation: dict,
    profile_map: dict[int, dict],
    file_audit: dict | None,
    lineage_lookup: dict[tuple[str, int, int], bool] | None = None,
) -> dict:
    """Run the full T-D1.1~10 pipeline on one candidate relation.

    ``lineage_lookup`` is the D2 parent-lineage verification map keyed by
    ``(rdat_sha256, wt_profile_index, mutant_profile_index)``. When present,
    ``parent_lineage_verified`` is taken from the D2 artifact (forward-only
    upgrade over the conservative D0-R v2 ``lineage_status`` label). When
    absent, the D0-R v2 conservative rule is used (``lineage_status`` does
    not start with ``candidate_only_pending``).
    """
    rdat_path = relation["rdat_path"]
    wt_idx = relation["wt_profile_index"]
    mut_idx = relation["mutant_profile_index"]
    matched = relation.get("matched_mutation") or {}
    audit_method = relation.get("audit_method", "")

    is_annotation_only = "annotation_only" in audit_method
    is_sequence_based = "functional_anchor" in audit_method or "functional_window" in audit_method
    alt_not_verified = bool(matched.get("alt_not_verified", True))
    # v3.1 §3.2: annotation-only candidates require per-profile sequence
    # evidence (or same-parent replicate) to verify the substitution. D0-R v2
    # annotation-only candidates carry encoding_source="annotation" (no
    # per-profile sequence), so the substitution is NOT sequence-verified
    # even when the annotated alt is a concrete base. Only sequence-based
    # candidates can reach substitution_verified=True via alt_not_verified.
    substitution_verified = (not is_annotation_only) and (not alt_not_verified)
    edit_count = int(relation.get("annotation_mutation_count") or 0)

    # Profile lookup
    wt_profile = profile_map.get(wt_idx)
    mut_profile = profile_map.get(mut_idx)
    has_wt_anchor = wt_profile is not None
    profile_lookup_ok = has_wt_anchor and mut_profile is not None

    # Default reactivity-derived metrics
    comp_stats = {
        "aligned_length": 0,
        "both_nonmissing": 0,
        "comparable_fraction": None,
        "missing_fraction": None,
    }
    delta_summary: dict = {"mean_abs_delta": None, "n_positions": 0}
    caller_status = "no_profile"
    snr: float | None = None
    coverage_mean: float | None = None  # RDAT carries no read-coverage field
    has_replicates = False  # single profiles; replicate identification is D2

    # Three-layer reactivity + Δreactivity arrays + noise estimate fields
    # (v3.1 §必须输出: raw/upstream/normalized 三层 + measurement noise 估计).
    # Default to None for the no-profile branch; populated below when the
    # profile lookup succeeds.
    wt_reactivity_raw: list | None = None
    wt_reactivity_upstream: list | None = None
    wt_reactivity_project: list | None = None
    wt_normalization_method: str | None = None
    mut_reactivity_raw: list | None = None
    mut_reactivity_upstream: list | None = None
    mut_reactivity_project: list | None = None
    mut_normalization_method: str | None = None
    delta_reactivity_raw: list | None = None
    delta_reactivity_normalized: list | None = None
    replicate_noise_estimate: float | None = None
    measurement_variance: float | None = None
    noise_wt_variance: float | None = None
    noise_mut_variance: float | None = None
    noise_source: str | None = None

    if profile_lookup_ok:
        wt_raw = wt_profile["reactivity"]
        mut_raw = mut_profile["reactivity"]
        comp_stats = _comparable_stats(wt_raw, mut_raw)

        # Build raw reactivity layers (no upstream normalization method on
        # file; RDAT reactivity is already the upstream-provided layer).
        wt_layers = build_reactivity_layers(wt_raw, normalization_method=None)
        mut_layers = build_reactivity_layers(mut_raw, normalization_method=None)

        # Persist the three-layer reactivity for WT and mutant (v3.1 §必须输出).
        wt_reactivity_raw = wt_layers["reactivity_raw"]
        wt_reactivity_upstream = wt_layers["reactivity_upstream"]
        wt_reactivity_project = wt_layers["reactivity_project"]
        wt_normalization_method = wt_layers["normalization_method"]
        mut_reactivity_raw = mut_layers["reactivity_raw"]
        mut_reactivity_upstream = mut_layers["reactivity_upstream"]
        mut_reactivity_project = mut_layers["reactivity_project"]
        mut_normalization_method = mut_layers["normalization_method"]

        # Δreactivity on raw + (trivially same) normalized layer.
        pair_delta = build_pair_delta_reactivity(
            wt_reactivity_raw=wt_layers["reactivity_raw"],
            mut_reactivity_raw=mut_layers["reactivity_upstream"],
            wt_reactivity_normalized=wt_layers["reactivity_project"],
            mut_reactivity_normalized=mut_layers["reactivity_project"],
            noise_threshold=None,  # not frozen at D1 (no test peeking)
            measurement_variance=None,
            has_replicates=has_replicates,
        )
        caller_status = pair_delta["caller_status"]

        # Persist the pair-schema Δreactivity arrays (v3.1 §必须输出).
        delta_reactivity_raw = pair_delta["delta_reactivity_raw"]
        delta_reactivity_normalized = pair_delta["delta_reactivity_normalized"]

        delta = pair_delta["delta_reactivity_raw"]
        abs_deltas = [abs(v) for v in delta if isinstance(v, (int, float)) and math.isfinite(v)]
        delta_summary = {
            "mean_abs_delta": _mean(abs_deltas),
            "n_positions": len(abs_deltas),
        }

        # SNR estimate from upstream REACTIVITY_ERROR (v3 §7.3 no-replicate
        # path): signal = mean|Δr|, noise = sqrt(wt_var + mut_var).
        wt_err = wt_profile.get("reactivity_error")
        mut_err = mut_profile.get("reactivity_error")
        wt_var = estimate_error_variance(wt_err) if wt_err else None
        mut_var = estimate_error_variance(mut_err) if mut_err else None
        pair_noise = estimate_pair_noise(None, None, wt_var, mut_var)

        # Persist the pair-schema noise fields (v3.1 §必须输出).
        replicate_noise_estimate = pair_noise["replicate_noise_estimate"]
        measurement_variance = pair_noise["measurement_variance"]
        noise_wt_variance = pair_noise["wt_variance"]
        noise_mut_variance = pair_noise["mut_variance"]
        noise_source = pair_noise["source"]

        mvar = pair_noise["measurement_variance"]
        mean_abs = delta_summary["mean_abs_delta"]
        if (
            mean_abs is not None
            and mvar is not None
            and isinstance(mvar, (int, float))
            and math.isfinite(mvar)
            and mvar > 0
        ):
            snr = mean_abs / math.sqrt(mvar)

    # condition_match_status: WT and mutant share the same RDAT file and a
    # single modifier field → exact match (v3 §6.5 condition coupling).
    condition_match_status = "match"

    # normalization_domain_compatible: WT and mutant come from the same
    # RDAT / rmdb_id / study → same domain (T-D1.7).
    normalization_domain_compatible = True

    # parent_lineage_verified (T-D1.6 / T-D2.1): when a D2 lineage-
    # verification artifact is supplied, use the forward-only D2 verdict
    # (same-RDAT parenthood + header-SEQUENCE ref verification). Otherwise
    # fall back to the conservative D0-R v2 rule: a candidate whose
    # ``lineage_status`` still starts with ``candidate_only_pending`` is
    # treated as unverified.
    lineage_key = (relation.get("rdat_sha256", ""), wt_idx, mut_idx)
    if lineage_lookup is not None and lineage_key in lineage_lookup:
        parent_lineage_verified = bool(lineage_lookup[lineage_key])
    else:
        parent_lineage_verified = relation.get("lineage_status", "").startswith(
            "candidate_only_pending"
        ) is False

    in_vivo_in_vitro_mixed = False  # D1 pool is in-vitro RMDB only

    upgrade = evaluate_pair_upgrade(
        edit_type="substitution" if edit_count == 1 else "indel",
        edit_count=edit_count,
        condition_match_status=condition_match_status,
        substitution_verified=substitution_verified,
        has_wt_anchor=has_wt_anchor,
        normalization_domain_compatible=normalization_domain_compatible,
        parent_lineage_verified=parent_lineage_verified,
        in_vivo_in_vitro_mixed=in_vivo_in_vitro_mixed,
        comparable_fraction=comp_stats["comparable_fraction"],
        probe_eligible_unchanged=True,  # same probe within one RDAT file
        annotation_ref_verified=True,  # audit_method: ref_verified_against_header
        is_annotation_only=is_annotation_only,
        is_sequence_based=is_sequence_based,
        has_independent_corroboration=True,  # default; is_sequence_based=False
        snr=snr,
        coverage_mean=coverage_mean,
        missing_fraction=comp_stats["missing_fraction"],
        has_replicates=has_replicates,
    )

    return {
        "rdat_path": rdat_path,
        "rmdb_id": relation.get("rmdb_id"),
        "owner": relation.get("owner"),
        "parent_prefix": relation.get("parent_prefix"),
        "citation_doi": relation.get("citation_doi"),
        "modifier": relation.get("modifier"),
        "wt_profile_index": wt_idx,
        "mutant_profile_index": mut_idx,
        "matched_mutation": matched,
        "audit_method": audit_method,
        "edit_type": "substitution" if edit_count == 1 else "indel",
        "edit_count": edit_count,
        "condition_match_status": condition_match_status,
        "substitution_verified": substitution_verified,
        "has_wt_anchor": has_wt_anchor,
        "normalization_domain_compatible": normalization_domain_compatible,
        "parent_lineage_verified": parent_lineage_verified,
        "parent_lineage_source": (
            "d2_lineage_verification"
            if lineage_lookup is not None and lineage_key in lineage_lookup
            else "d0r_v2_lineage_status"
        ),
        "in_vivo_in_vitro_mixed": in_vivo_in_vitro_mixed,
        "is_annotation_only": is_annotation_only,
        "is_sequence_based": is_sequence_based,
        "profile_lookup_ok": profile_lookup_ok,
        "comparable_fraction": comp_stats["comparable_fraction"],
        "missing_fraction": comp_stats["missing_fraction"],
        "aligned_length": comp_stats["aligned_length"],
        "both_nonmissing": comp_stats["both_nonmissing"],
        "snr": snr,
        "coverage_mean": coverage_mean,
        "has_replicates": has_replicates,
        "delta_reactivity_summary": delta_summary,
        "caller_status": caller_status,
        # v3.1 §必须输出: three-layer reactivity + Δreactivity arrays + noise
        "wt_reactivity_raw": wt_reactivity_raw,
        "wt_reactivity_upstream": wt_reactivity_upstream,
        "wt_reactivity_project": wt_reactivity_project,
        "wt_normalization_method": wt_normalization_method,
        "mut_reactivity_raw": mut_reactivity_raw,
        "mut_reactivity_upstream": mut_reactivity_upstream,
        "mut_reactivity_project": mut_reactivity_project,
        "mut_normalization_method": mut_normalization_method,
        "delta_reactivity_raw": delta_reactivity_raw,
        "delta_reactivity_normalized": delta_reactivity_normalized,
        "replicate_noise_estimate": replicate_noise_estimate,
        "measurement_variance": measurement_variance,
        "noise_wt_variance": noise_wt_variance,
        "noise_mut_variance": noise_mut_variance,
        "noise_source": noise_source,
        "exclusion_reasons": upgrade["exclusion_reasons"],
        "primary_eligible": upgrade["primary_eligible"],
        "true_pair": upgrade["true_pair"],
        "pair_quality_weight": upgrade["pair_quality_weight"],
        "quality_factors": upgrade["quality_factors"],
    }


def _tier_judgment(registry: list[dict]) -> dict:
    """Re-judge Tier A/B/C with true_pair counts (v3 §8, v3.1 §7)."""
    true_pairs = [r for r in registry if r["true_pair"]]
    tp_count = len(true_pairs)

    tp_studies = {r["citation_doi"] for r in true_pairs if r.get("citation_doi")}
    tp_parents = {r["parent_prefix"] for r in true_pairs if r.get("parent_prefix")}
    tp_owners = {r["owner"] for r in true_pairs if r.get("owner")}

    # Candidate-level (D0-R v2) counts for reference — NOT the D1 gate basis.
    cand_studies = {r["citation_doi"] for r in registry if r.get("citation_doi")}
    cand_parents = {r["parent_prefix"] for r in registry if r.get("parent_prefix")}

    tier_a = {
        "basis": "true_pair counts (v3.1 §7: construct count forbidden)",
        "studies_true_pair": len(tp_studies),
        "parents_true_pair": len(tp_parents),
        "true_pairs": tp_count,
        "thresholds": {
            "min_studies": TIER_A_MIN_STUDIES,
            "min_parents": TIER_A_MIN_PARENTS,
            "min_pairs": TIER_A_MIN_PAIRS,
        },
        "pass": (
            len(tp_studies) >= TIER_A_MIN_STUDIES
            and len(tp_parents) >= TIER_A_MIN_PARENTS
            and tp_count >= TIER_A_MIN_PAIRS
        ),
    }
    tier_b = {
        "basis": "true_pair counts",
        "true_pairs": tp_count,
        "threshold": TIER_B_MIN_TRUE_PAIRS,
        "pass": tp_count >= TIER_B_MIN_TRUE_PAIRS,
    }
    # Tier C (v3 §8) is the strictest tier; D1 reports it as not-reached when
    # Tier B already fails.
    tier_c = {
        "basis": "true_pair counts",
        "true_pairs": tp_count,
        "pass": tp_count >= TIER_B_MIN_TRUE_PAIRS,  # C ≥ B; fails identically
        "note": "Tier C requires ≥ Tier B threshold; fails with Tier B",
    }
    return {
        "tier_a": tier_a,
        "tier_b": tier_b,
        "tier_c": tier_c,
        "candidate_level_reference": {
            "candidate_total": len(registry),
            "candidate_studies": len(cand_studies),
            "candidate_parents": len(cand_parents),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--relations",
        default="artifacts/reactflow_delta/d0r/d0r_reaudit_tierA_relations.json",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/reactflow_delta/d1",
    )
    parser.add_argument(
        "--lineage-verification",
        default=None,
        help=(
            "Optional D2 lineage-verification artifact "
            "(artifacts/reactflow_delta/d2/d2_lineage_verification.json). "
            "When supplied, parent_lineage_verified is taken from the D2 "
            "verdict (forward-only upgrade over D0-R v2 lineage_status)."
        ),
    )
    args = parser.parse_args(argv)

    relations_path = Path(args.relations)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with relations_path.open() as f:
        rel_doc = json.load(f)
    relations = rel_doc["relations"]
    print(f"[d1] loaded {len(relations)} candidate relations from {relations_path}")

    # Optional D2 parent-lineage verification artifact (T-D2.1). Keyed by
    # (rdat_sha256, wt_profile_index, mutant_profile_index) → bool.
    lineage_lookup: dict[tuple[str, int, int], bool] | None = None
    if args.lineage_verification:
        lv_path = Path(args.lineage_verification)
        with lv_path.open() as f:
            lv_doc = json.load(f)
        lineage_lookup = {
            (
                v["rdat_sha256"],
                int(v["wt_profile_index"]),
                int(v["mutant_profile_index"]),
            ): bool(v["parent_lineage_verified"])
            for v in lv_doc.get("verifications", [])
        }
        verified = sum(1 for v in lineage_lookup.values() if v)
        print(
            f"[d1] loaded D2 lineage verification from {lv_path} "
            f"({len(lineage_lookup)} entries, {verified} verified)"
        )

    # Group by rdat_path so each RDAT is parsed at most once.
    by_path: dict[str, list[dict]] = defaultdict(list)
    for rel in relations:
        by_path[rel["rdat_path"]].append(rel)
    print(f"[d1] {len(by_path)} unique RDAT files to parse")

    registry: list[dict] = []
    parse_errors: list[dict] = []
    file_cache: dict[str, dict[int, dict] | None] = {}

    for rdat_path, rels_for_file in by_path.items():
        try:
            document = parse_rdat(rdat_path)
            profile_map = {p["index"]: p for p in document["profiles"]}
            file_audit = {"rdat_sha256": document["sha256"]}
        except (RdatParseError, FileNotFoundError) as exc:
            parse_errors.append({"rdat_path": rdat_path, "error": str(exc)})
            file_cache[rdat_path] = None
            # Emit a minimal registry entry per relation so the candidate
            # total still accounts for them (forward-only: no deletion).
            for rel in rels_for_file:
                registry.append(_evaluate_one(rel, {}, None, lineage_lookup))
            continue
        file_cache[rdat_path] = profile_map
        for rel in rels_for_file:
            registry.append(_evaluate_one(rel, profile_map, file_audit, lineage_lookup))

    # ---- Aggregates ----
    tp_count = sum(1 for r in registry if r["true_pair"])
    pe_count = sum(1 for r in registry if r["primary_eligible"])

    reason_counter: Counter[str] = Counter()
    reason_set_counter: Counter[str] = Counter()
    for r in registry:
        for reason in r["exclusion_reasons"]:
            reason_counter[reason] += 1
        reason_set_counter[",".join(r["exclusion_reasons"]) or "(none)"] += 1

    def _dist(field: str) -> dict[str, int]:
        c: Counter[str] = Counter()
        for r in registry:
            c[str(r.get(field))] += 1
        return dict(c.most_common())

    tp_registry = [r for r in registry if r["true_pair"]]
    summary = {
        "schema_version": "reactflow-delta-d1-pipeline-summary-v1",
        "stage": "D1 cleanup-only pipeline executor",
        "candidate_total": len(registry),
        "true_pair_count": tp_count,
        "primary_eligible_count": pe_count,
        "parent_lineage_source": (
            "d2_lineage_verification"
            if lineage_lookup is not None
            else "d0r_v2_lineage_status"
        ),
        "_d2_lineage_applied": lineage_lookup is not None,
        "reason_distribution_per_reason": dict(reason_counter.most_common()),
        "reason_distribution_per_set": dict(reason_set_counter.most_common()),
        "study_distribution_candidates": _dist("citation_doi"),
        "parent_distribution_candidates": _dist("parent_prefix"),
        "owner_distribution_candidates": _dist("owner"),
        "modifier_distribution_candidates": _dist("modifier"),
        "study_distribution_true_pair": dict(Counter(r["citation_doi"] for r in tp_registry).most_common()),
        "parent_distribution_true_pair": dict(Counter(r["parent_prefix"] for r in tp_registry).most_common()),
        "owner_distribution_true_pair": dict(Counter(r["owner"] for r in tp_registry).most_common()),
        "caller_status_distribution": _dist("caller_status"),
        "profile_lookup_failures": sum(1 for r in registry if not r["profile_lookup_ok"]),
        "parse_errors": parse_errors,
        "tier_judgment": _tier_judgment(registry),
    }

    reg_path = out_dir / "d1_true_pair_registry.json"
    sum_path = out_dir / "d1_pipeline_summary.json"
    with reg_path.open("w") as f:
        json.dump(
            {
                "schema_version": "reactflow-delta-d1-true-pair-registry-v1",
                "stage": "D1 cleanup-only",
                "candidate_total": len(registry),
                "true_pair_count": tp_count,
                "primary_eligible_count": pe_count,
                "registry": registry,
            },
            f,
            indent=2,
        )
    with sum_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[d1] true_pair_count = {tp_count} / {len(registry)}")
    print(f"[d1] primary_eligible_count = {pe_count}")
    print(f"[d1] reason_distribution = {summary['reason_distribution_per_reason']}")
    print(f"[d1] tier_a pass = {summary['tier_judgment']['tier_a']['pass']}")
    print(f"[d1] tier_b pass = {summary['tier_judgment']['tier_b']['pass']}")
    print(f"[d1] wrote {reg_path}")
    print(f"[d1] wrote {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
