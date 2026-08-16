#!/usr/bin/env python3
"""D2 RSIB-v1 build script (T-D2.1 / T-D2.2 / T-D2.10, v3 §9 + v3.1 §7).

Builds the parent / study / design-lineage graph from the D0-R v2 candidate
relations, verifies parent lineage per candidate (clearing
``parent_lineage_unverified`` when the same-RDAT + header-SEQUENCE ref
evidence holds), audits within-pool concentration, and emits the D2 Tier
A/B/C judgment.

D2 is non-learning (v3.1 §2.2): no training, no model forward, no test
peeking, no Tier-threshold lowering. This script does NOT re-run the D1
executor — it produces the lineage-verification artifact that the executor
consumes on its next run, plus the D2-level audit / Tier-judgment artifacts.

Outputs (under ``--out-dir``, default ``artifacts/reactflow_delta/d2``)::

    d2_lineage_graph.json         — T-D2.1 graph (constructs + groupings)
    d2_lineage_verification.json  — per-candidate parent_lineage_verified
    d2_overlap_audit.json         — T-D2.2 within-pool concentration report
    d2_tier_judgment.json         — T-D2.10 Tier A/B/C judgment

Usage::

    PYTHONPATH=src python scripts/reactflow_delta/build_rsib.py \
        --relations artifacts/reactflow_delta/d0r/d0r_reaudit_tierA_relations.json \
        --d1-summary artifacts/reactflow_delta/d1/d1_pipeline_summary.json \
        --out-dir artifacts/reactflow_delta/d2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure ``src`` is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reactflow.delta.contamination import (  # noqa: E402
    audit_split_overlap,
    build_lineage_graph,
    compute_overlap_report,
    compute_tier_judgment,
    verify_parent_lineage,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--relations",
        default="artifacts/reactflow_delta/d0r/d0r_reaudit_tierA_relations.json",
    )
    parser.add_argument(
        "--d1-summary",
        default="artifacts/reactflow_delta/d1/d1_pipeline_summary.json",
        help="D1 pipeline summary (for binding-blocker + reason distribution).",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/reactflow_delta/d2",
    )
    args = parser.parse_args(argv)

    relations_path = Path(args.relations)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with relations_path.open() as f:
        rel_doc = json.load(f)
    relations = rel_doc["relations"]
    print(f"[d2] loaded {len(relations)} candidate relations from {relations_path}")

    # --- T-D2.1: lineage graph ---
    graph = build_lineage_graph(relations)
    graph_path = out_dir / "d2_lineage_graph.json"
    with graph_path.open("w") as f:
        json.dump(
            {
                "schema_version": "reactflow-delta-d2-lineage-graph-v1",
                "stage": "D2 RSIB-v1 lineage graph",
                "source_relations": str(relations_path),
                **graph,
            },
            f,
            indent=2,
        )
    print(f"[d2] wrote {graph_path}")
    print(
        f"[d2] unique: rdat={graph['unique_counts']['rdat_sha256']} "
        f"rmdb_id={graph['unique_counts']['rmdb_id']} "
        f"parent={graph['unique_counts']['parent_prefix']} "
        f"study={graph['unique_counts']['citation_doi']} "
        f"owner={graph['unique_counts']['owner']}"
    )

    # --- T-D2.1: per-candidate parent lineage verification ---
    verifications = [verify_parent_lineage(rel) for rel in relations]
    verified_count = sum(1 for v in verifications if v["parent_lineage_verified"])
    verification_doc = {
        "schema_version": "reactflow-delta-d2-lineage-verification-v1",
        "stage": "D2 parent lineage verification",
        "source_relations": str(relations_path),
        "verification_rule": (
            "parent_lineage_verified = (wt_profile_index and "
            "mutant_profile_index share the same rdat_sha256) AND "
            "(matched_mutation.ref_verified_against == 'header_SEQUENCE')"
        ),
        "total_candidates": len(verifications),
        "parent_lineage_verified_count": verified_count,
        "parent_lineage_unverified_count": len(verifications) - verified_count,
        "note": (
            "Clearing parent_lineage_unverified does NOT upgrade "
            "annotation-only candidates to true_pair — the "
            "annotation_only_alt_not_verifiable reason (v3.1 §3.2) still "
            "applies when no per-profile sequence evidence exists."
        ),
        "verifications": verifications,
    }
    ver_path = out_dir / "d2_lineage_verification.json"
    with ver_path.open("w") as f:
        json.dump(verification_doc, f, indent=2)
    print(
        f"[d2] wrote {ver_path} "
        f"(verified {verified_count}/{len(verifications)})"
    )

    # --- T-D2.2: overlap audit ---
    # No splits frozen yet (true_pair = 0 expected after re-run). Audit the
    # candidate-pool concentration instead; split freezing (T-D2.3-5) is
    # deferred until true_pairs exist.
    overlap_report = compute_overlap_report(relations)
    # Also run the formal split-overlap audit on an empty split assignment
    # to demonstrate the gate machinery (overlap = 0 trivially when no
    # splits are frozen).
    empty_overlap = audit_split_overlap({})
    overlap_doc = {
        "schema_version": "reactflow-delta-d2-overlap-audit-v1",
        "stage": "D2 split overlap audit",
        "source_relations": str(relations_path),
        "split_freezing_status": {
            "status": "deferred",
            "reason": (
                "T-D2.3-5 split freezing deferred: no true_pairs to "
                "partition. The D2 Gate (split group overlap = 0) is "
                "trivially satisfied for the empty split assignment; it "
                "will be enforced when true_pairs materialise."
            ),
            "empty_split_overlap_gate": empty_overlap,
        },
        **overlap_report,
    }
    overlap_path = out_dir / "d2_overlap_audit.json"
    with overlap_path.open("w") as f:
        json.dump(overlap_doc, f, indent=2)
    print(f"[d2] wrote {overlap_path}")

    # --- T-D2.10: Tier A/B/C judgment ---
    # Load the D1 summary to get the re-run true_pair counts and the
    # binding-blocker reason distribution. If the D1 summary is absent
    # (e.g. first D2 run before executor re-run), fall back to zero
    # true_pairs with the candidate-level reference from the relations.
    d1_summary: dict = {}
    d1_summary_path = Path(args.d1_summary)
    if d1_summary_path.exists():
        with d1_summary_path.open() as f:
            d1_summary = json.load(f)

    tier_judgment_from_d1 = d1_summary.get("tier_judgment", {})
    cand_ref = tier_judgment_from_d1.get("candidate_level_reference", {})
    true_pair_count = d1_summary.get("true_pair_count", 0)
    tp_studies = len(tier_judgment_from_d1.get("study_distribution_true_pair", {}) or {}) if "study_distribution_true_pair" in tier_judgment_from_d1 else tier_judgment_from_d1.get("tier_a", {}).get("studies_true_pair", 0)
    tp_parents = tier_judgment_from_d1.get("tier_a", {}).get("parents_true_pair", 0)

    reason_dist = d1_summary.get("reason_distribution_per_reason", {})
    binding_blocker = None
    if reason_dist:
        # The binding blocker is the most frequent reason that is NOT
        # parent_lineage_unverified (which D2 just cleared). If
        # parent_lineage_unverified is still present, the D1 summary is the
        # pre-D2 run; report that explicitly.
        if "parent_lineage_unverified" in reason_dist and not d1_summary.get(
            "_d2_lineage_applied", False
        ):
            binding_blocker = (
                "pre-D2 summary loaded: parent_lineage_unverified still "
                "present — re-run d1_pipeline_executor with "
                "--lineage-verification to clear it."
            )
        else:
            # Pick the most frequent non-lineage reason as the binding blocker.
            non_lineage = {
                k: v
                for k, v in reason_dist.items()
                if k != "parent_lineage_unverified"
            }
            if non_lineage:
                top_reason = max(non_lineage, key=non_lineage.get)
                binding_blocker = (
                    f"{top_reason} ({non_lineage[top_reason]}/{d1_summary.get('candidate_total', 0)} "
                    "candidates) — v3.1 §3.2: annotation-only candidates "
                    "require per-profile sequence evidence or same-parent "
                    "replicate corroboration to upgrade."
                )

    tier = compute_tier_judgment(
        true_pair_count=true_pair_count,
        true_pair_studies=tp_studies,
        true_pair_parents=tp_parents,
        candidate_total=cand_ref.get("candidate_total", len(relations)),
        candidate_studies=cand_ref.get("candidate_studies", graph["unique_counts"]["citation_doi"]),
        candidate_parents=cand_ref.get("candidate_parents", graph["unique_counts"]["parent_prefix"]),
        binding_blocker=binding_blocker,
    )
    tier_doc = {
        "schema_version": "reactflow-delta-d2-tier-judgment-v1",
        "stage": "D2 Tier A/B/C judgment",
        "source_d1_summary": str(d1_summary_path) if d1_summary else None,
        **tier,
    }
    tier_path = out_dir / "d2_tier_judgment.json"
    with tier_path.open("w") as f:
        json.dump(tier_doc, f, indent=2)
    print(f"[d2] wrote {tier_path}")
    print(
        f"[d2] tier_a pass = {tier['tier_a']['pass']}, "
        f"tier_b pass = {tier['tier_b']['pass']}, "
        f"true_pairs = {true_pair_count}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
