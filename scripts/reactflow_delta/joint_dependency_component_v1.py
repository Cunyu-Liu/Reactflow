#!/usr/bin/env python3
"""joint_dependency_component_v1 (contract 7.5, 11.5) outcome-blind.

Builds a joint-dependency graph over candidate components and computes the
outcome-blind K_preaccess. Preaccess qualification uses ONLY
PREACCESS_METADATA_ONLY (identity/publication/study/batch/lineage/homology/
probe/platform/normalization/construct/registered-target structure/required-file
existence/schema-version). Any outcome-derived metadata (reactivity/error/mask/
effect/performance/summary) disqualifies a component from K_preaccess.

K_eff_realized / post-access fields are NOT filled before Phase4 access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OUTCOME_DERIVED_HINTS = [
    "reactivity", "error", "mask", "effect", "performance",
    "score", "outcome", "result", "profile",
]


@dataclass
class ComponentCandidate:
    component_id: str
    publication: str
    study_batch: str
    wt_accession: str | None = None
    lineage: str | None = None
    probe: str | None = None
    platform: str | None = None
    task_compatible: bool = True
    provenance_resolved: bool = True
    development_disconnected: bool = False
    metadata_keys: set[str] = field(default_factory=set)


def preaccess_metadata_allowed(metadata_keys: set[str]) -> bool:
    """Reject any outcome-derived metadata from preaccess qualification."""
    disallowed = [k for k in metadata_keys
                  if any(h in k.lower() for h in OUTCOME_DERIVED_HINTS)]
    return len(disallowed) == 0, disallowed


def compute_k_preaccess(candidates: list[ComponentCandidate],
                        development_component_ids: set[str]) -> dict[str, Any]:
    """Outcome-blind K_preaccess over qualified, development-disconnected candidates."""
    qualified: list[str] = []
    rejected: dict[str, list[str]] = {}
    for c in candidates:
        reasons = []
        ok_meta, bad = preaccess_metadata_allowed(c.metadata_keys)
        if not ok_meta:
            reasons.append(f"outcome-derived metadata: {bad}")
        if not c.provenance_resolved:
            reasons.append("provenance unresolved")
        if not c.task_compatible:
            reasons.append("task/assay incompatible")
        if c.component_id in development_component_ids:
            reasons.append("development-connected")
        if c.development_disconnected is False and c.component_id not in development_component_ids:
            # explicit flag required for confirmation eligibility
            reasons.append("not marked development-disconnected")
        if reasons:
            rejected[c.component_id] = reasons
        else:
            qualified.append(c.component_id)
    return {
        "schema_version": "reactflow_delta.joint_dependency_component_v1.v1",
        "K_preaccess": len(qualified),
        "qualified_components": qualified,
        "rejected_components": rejected,
        "K_eff_realized": None,  # only filled after Phase4 access (outcome-blind now)
        "K_required_planned": None,  # P4 power planning
        "note": "K_eff_realized/K_required_planned must NOT be filled before Phase4 locked access",
    }
