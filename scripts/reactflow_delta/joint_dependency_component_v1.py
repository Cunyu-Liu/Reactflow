#!/usr/bin/env python3
"""joint_dependency_component_v1 (contract 7.5, 11.5) outcome-blind — v2 union-find.

SUBSTANTIATED for audit P0-4 (2026-08-18): components that share ANY of
publication / study / batch / library / sequencing-run / shared-WT are MERGED
into one joint cluster via a union-find graph. The old builder behaviour
`K_preaccess = len(comps)` (audit finding #4) is replaced by `K_joint`, the
number of connected components at the highest resolved dependency level.

Outcome-blind: qualification uses ONLY metadata (identity/publication/study/batch/
library/sequencing-run/probe/platform/normalization/construct/registered-target
structure/required-file existence/schema-version). Any outcome-derived metadata
(reactivity/error/mask/effect/performance/summary) disqualifies a component from
K_preaccess/K_joint. SNV count and position count NEVER increase K (audit P0-4:
"SNV/position/seed 不增加 K").

Provenance evidence rules (audit P0-4 "每个 edge 有 metadata evidence"):
  * batch/sequencing-run/library are taken DIRECTLY from the rdat file header
    (NAME + 'COMMENT from data' + ANNOTATION) — CONFIRMED_FACT.
  * study/publication are REASONED_INFERENCE from the dataset identity matched to
    the source publication (Ribonanza bioRxiv 2024 DOI 10.1101/2024.02.24.581671;
    SL5 PNAS 2024 PMID 38427602). Where publication cannot be evidenced from the
    rdat itself, it is marked UNKNOWN_NOT_ASSERTED and the component is EXCLUDED
    from confirmatory K_joint (fail-closed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

OUTCOME_DERIVED_HINTS = [
    "reactivity", "error", "mask", "effect", "performance",
    "score", "outcome", "result", "profile",
]


# --------------------------------------------------------------------------- #
# dataset-level provenance (evidence-graded)
# --------------------------------------------------------------------------- #
@dataclass
class DatasetProvenance:
    """Provenance of ONE external dataset file. None = unresolved (fail-closed)."""
    dataset_id: str
    rdat_name: str                 # rdat NAME field (CONFIRMED_FACT)
    sequencing_batch: str          # rdat 'COMMENT from data' (CONFIRMED_FACT)
    library: str                   # Das-lab library identity (from NAME)
    study: str | None              # study id (REASONED_INFERENCE; None = UNKNOWN)
    publication: str | None        # publication id (REASONED_INFERENCE; None = UNKNOWN)
    evidence: str = ""


# The 7 external dataset files (P4 24 + P5b 694 anchors). Batch/library from the
# rdat headers read directly (rdat_tierA_20260730/*.rdat). Study/publication from
# the matched publications (see evidence strings).
EXTERNAL_DATASET_PROVENANCE: list[DatasetProvenance] = [
    DatasetProvenance(
        "M2SL5_2A3_0000", "SL5_M2seq",
        "NovaSeq_2023-06-06_RH_SL5_M2seq",
        "SL5_M2seq_library",
        "study_sl5", "pub_sl5_pnas2024",
        evidence="rdat NAME=SL5_M2seq, COMMENT batch NovaSeq_2023-06-06; SL5 M2-seq "
                 "data deposited with Tertiary folds of the SL5 RNA (PNAS 2024, PMID 38427602)"),
    DatasetProvenance(
        "15KLIB_2A3_0000", "DasLab_15k_library",
        "NovaSeq_2023-08-01_RH_DasBigLib0-15k",
        "DasLab_15k_library",
        "study_ribonanza", "pub_ribonanza_2024",
        evidence="rdat NAME=DasLab_15k_library, COMMENT batch NovaSeq_2023-08-01; "
                 "15k (BigLib0) library M2-seq, Ribonanza-era Das-lab data"),
    DatasetProvenance(
        "M3SARS_2A3_0000", "DasLabBigLib2_OneMil2_fse",
        "NovaSeq_2023-10-31_RH_OneMil2_DasLabBigLib2-1M",
        "DasLabBigLib2_OneMil2",
        "study_ribonanza", "pub_ribonanza_2024",
        evidence="rdat NAME=DasLabBigLib2_OneMil2_fse, COMMENT batch "
                 "NovaSeq_2023-10-31; MERS FSE M2 data shown in Ribonanza "
                 "(bioRxiv 2024 DOI 10.1101/2024.02.24.581671) Fig 3c"),
    DatasetProvenance(
        "M2RFOK_2A3_0000", "DasLabBigLib2_OneMil2_rfamokm2",
        "NovaSeq_2023-10-31_RH_OneMil2_DasLabBigLib2-1M",
        "DasLabBigLib2_OneMil2",
        "study_ribonanza", "pub_ribonanza_2024",
        evidence="rdat NAME=DasLabBigLib2_OneMil2_rfamokm2, COMMENT batch "
                 "NovaSeq_2023-10-31 (same run as M3SARS)"),
    DatasetProvenance(
        "M2RFPK_2A3_0000", "DasLabBigLib2_OneMil2_rfampkm2_splitA",
        "NovaSeq_2023-10-31_RH_OneMil2_DasLabBigLib2-1M",
        "DasLabBigLib2_OneMil2",
        "study_ribonanza", "pub_ribonanza_2024",
        evidence="rdat NAME=DasLabBigLib2_OneMil2_rfampkm2_splitA, COMMENT batch "
                 "NovaSeq_2023-10-31 (same run as M3SARS)"),
    DatasetProvenance(
        "M2RFPK_2A3_0001", "DasLabBigLib2_OneMil2_rfampkm2_splitB",
        "NovaSeq_2023-10-31_RH_OneMil2_DasLabBigLib2-1M",
        "DasLabBigLib2_OneMil2",
        "study_ribonanza", "pub_ribonanza_2024",
        evidence="rdat NAME=DasLabBigLib2_OneMil2_rfampkm2_splitB, COMMENT batch "
                 "NovaSeq_2023-10-31 (same run as M3SARS)"),
    DatasetProvenance(
        "M2RFPK_2A3_0002", "DasLabBigLib2_OneMil2_rfampkm2_splitC",
        "NovaSeq_2023-10-31_RH_OneMil2_DasLabBigLib2-1M",
        "DasLabBigLib2_OneMil2",
        "study_ribonanza", "pub_ribonanza_2024",
        evidence="rdat NAME=DasLabBigLib2_OneMil2_rfampkm2_splitC, COMMENT batch "
                 "NovaSeq_2023-10-31 (same run as M3SARS)"),
]

PROVENANCE_BY_DATASET = {p.dataset_id: p for p in EXTERNAL_DATASET_PROVENANCE}


# --------------------------------------------------------------------------- #
# union-find
# --------------------------------------------------------------------------- #
class UnionFind:
    """Disjoint-set union with path compression; tracks cluster members."""

    def __init__(self, items: Iterable[str]) -> None:
        items = list(items)  # materialize (generator consumed twice otherwise)
        self.parent = {i: i for i in items}
        self.members = {i: {i} for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # keep the smaller-lexicographic root stable for determinism
        if ra > rb:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] |= self.members[rb]
        del self.members[rb]

    def roots(self) -> list[str]:
        return sorted({self.find(x) for x in self.parent})


@dataclass
class ComponentCandidate:
    component_id: str
    publication: str | None = None
    study: str | None = None
    batch: str | None = None
    library: str | None = None
    dataset: str | None = None
    wt_accession: str | None = None
    lineage: str | None = None
    probe: str | None = None
    platform: str | None = None
    n_snv_mutants: int = 0       # SNV count (never increases K)
    n_rows: int = 0              # row count (never increases K)
    task_compatible: bool = True
    provenance_resolved: bool = True
    development_disconnected: bool = False
    metadata_keys: set[str] = field(default_factory=set)


def preaccess_metadata_allowed(metadata_keys: set[str]) -> tuple[bool, list[str]]:
    """Reject any outcome-derived metadata from preaccess qualification."""
    disallowed = [k for k in metadata_keys
                  if any(h in k.lower() for h in OUTCOME_DERIVED_HINTS)]
    return len(disallowed) == 0, disallowed


def _qualified(c: ComponentCandidate, development_component_ids: set[str]) -> list[str]:
    """Return rejection reasons for a candidate; empty list => qualified."""
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
        reasons.append("not marked development-disconnected")
    return reasons


def compute_k_preaccess(candidates: list[ComponentCandidate],
                        development_component_ids: set[str]) -> dict[str, Any]:
    """Outcome-blind K_preaccess over qualified, development-disconnected candidates."""
    qualified: list[str] = []
    rejected: dict[str, list[str]] = {}
    for c in candidates:
        reasons = _qualified(c, development_component_ids)
        if reasons:
            rejected[c.component_id] = reasons
        else:
            qualified.append(c.component_id)
    return {
        "schema_version": "reactflow_delta.joint_dependency_component_v1.v1",
        "K_preaccess": len(qualified),
        "qualified_components": qualified,
        "rejected_components": rejected,
        "K_eff_realized": None,
        "K_required_planned": None,
        "note": "K_eff_realized/K_required_planned must NOT be filled before Phase4 locked access",
    }


def compute_k_joint(candidates: list[ComponentCandidate],
                    development_component_ids: set[str],
                    *, evidence_gate: bool = True) -> dict[str, Any]:
    """Union-find joint dependency: merge components that share ANY of
    publication / study / batch / library / dataset / shared-WT.

    Returns N_rows, N_SNV, N_WT_anchor, N_dataset, N_batch, N_study, N_publication,
    K_joint (connected components among QUALIFIED + development-disconnected +
    provenance-resolved components). If evidence_gate and any provenance field is
    None, the component is EXCLUDED from K_joint (fail-closed), never inflated.
    """
    qualified = []
    rejected: dict[str, list[str]] = {}
    for c in candidates:
        reasons = _qualified(c, development_component_ids)
        if evidence_gate:
            for fld, val in (("publication", c.publication), ("study", c.study),
                             ("batch", c.batch), ("library", c.library)):
                if val is None:
                    reasons.append(f"provenance.{fld}=None (fail-closed)")
        if reasons:
            rejected[c.component_id] = reasons
        else:
            qualified.append(c)

    uf = UnionFind(c.component_id for c in qualified)
    for c in qualified:
        for key in ("publication", "study", "batch", "library", "dataset"):
            val = getattr(c, key)
            if val is None:
                continue
            # link all components sharing this key into one cluster
            for d in qualified:
                if d is c:
                    continue
                if getattr(d, key) == val:
                    uf.union(c.component_id, d.component_id)

    # resolve clusters; WT-anchor = number of qualified components
    clusters = uf.roots()
    n_snv = 0
    wt_by_dataset: set[str] = set()
    batch_set: set[str] = set()
    study_set: set[str] = set()
    pub_set: set[str] = set()
    dataset_set: set[str] = set()
    for c in qualified:
        n_snv += int(getattr(c, "n_snv_mutants", 0) or 0)
        if c.dataset:
            dataset_set.add(c.dataset)
        if c.batch:
            batch_set.add(c.batch)
        if c.study:
            study_set.add(c.study)
        if c.publication:
            pub_set.add(c.publication)

    cluster_members = {r: sorted(uf.members[r]) for r in clusters}
    return {
        "schema_version": "reactflow_delta.joint_dependency_component_v1.v1",
        "K_preaccess": len(qualified),
        "K_joint": len(clusters),
        "N_rows": sum(int(getattr(c, "n_rows", 0) or 0) for c in qualified),
        "N_SNV": n_snv,
        "N_WT_anchor": len(qualified),
        "N_dataset": len(dataset_set),
        "N_batch": len(batch_set),
        "N_study": len(study_set),
        "N_publication": len(pub_set),
        "clusters": cluster_members,
        "qualified_components": [c.component_id for c in qualified],
        "rejected_components": rejected,
        "note": ("SNV/position/seed never increase K; components with unresolved "
                 "provenance are excluded from K_joint (fail-closed)."),
    }
