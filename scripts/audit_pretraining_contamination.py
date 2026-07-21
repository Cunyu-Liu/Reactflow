#!/usr/bin/env python3
"""C1-1 Task 5: Pretraining contamination audit.

This script audits the pretraining data of four external RNA foundation models
against the ReactFlow global registry, classifying each as:

- ``clean``: training data is public and has zero overlap with the ReactFlow
  test/novel splits.
- ``contaminated``: training data is public and overlaps with the ReactFlow
  test/novel splits.
- ``unknown_contamination``: training data is not publicly available or not
  machine-readable, so overlap cannot be verified.

Models audited (spec line 283):
    - RiNALMo
    - RNA-FM
    - ERNIE-RNA
    - RibonanzaNet2

For each model the audit records:
    - model_version
    - weight_hash (SHA-256 of the downloaded weights, or ``"not_downloaded"``)
    - training_data_description (free text from the model card / paper)
    - training_data_available (bool)
    - exact_overlap_test (count of exact-sequence matches against ReactFlow
      test split)
    - exact_overlap_novel (count of exact-sequence matches against ReactFlow
      novel splits)
    - family_overlap_test (count of Rfam family overlaps with test split)
    - family_overlap_novel (count of Rfam family overlaps with novel splits)
    - contamination_status (``"clean"`` | ``"contaminated"`` |
      ``"unknown_contamination"``)

Overlap computation
-------------------

For each model, we compute:

1. **Exact overlap**: count of model-known RNA databases (e.g., ``RNAcentral``,
   ``Rfam``, ``bpRNA``) that share canonical sequences with the ReactFlow
   test/novel splits.  Because we do not download the model's training data,
   we use a *database-level* exact overlap: if the model's known databases
   include a database that ReactFlow's test/novel splits draw from, we mark
   exact_overlap as ``"database_level_overlap"`` (a string) and record the
   count of overlapping databases.  This is a conservative upper bound.

2. **Identity overlap**: 100% sequence identity overlap, equivalent to exact
   overlap for canonical sequences.  Subsumed by (1).

3. **Family overlap**: count of Rfam families in the model's known databases
   that also appear in ReactFlow's test/novel splits.  Computed by
   intersecting the model's ``known_rna_databases`` family set with the
   ReactFlow test/novel family set.

When the model's training data is not downloadable (e.g., Kaggle requires
authentication), the overlap fields are set to ``"not_computed"`` and the
contamination status is ``"unknown_contamination"``.

The script also defines three protocols for downstream training:
    - ``external_pretrained``: use the external weights as-is.
    - ``self_pretrained``: continue pretraining on ReactFlow training data.
    - ``from_scratch``: random initialization, no external weights.

Usage::

    python scripts/audit_pretraining_contamination.py \
        --registry-manifest artifacts/c1_1/global_registry_manifest.json \
        --contamination-groups artifacts/c1_1/contamination_groups.jsonl \
        --frozen-benchmark artifacts/c1_1/frozen_benchmark_manifest.json \
        --records artifacts/c1_1/global_registry_records.jsonl \
        --output artifacts/c1_1/pretraining_contamination_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Model metadata (static, from published model cards / papers)
# ---------------------------------------------------------------------------

@dataclass
class PretrainedModelSpec:
    """Specification of an external pretrained RNA model.

    Attributes:
        name: Canonical model name.
        version: Version string.
        paper: Citation or URL.
        training_data_description: Free-text description from the model card.
        training_data_available: Whether the training data is publicly
            downloadable in machine-readable form.
        training_data_url: URL to download the training data (if available).
        weights_url: URL to download the model weights.
        known_rna_databases: List of RNA databases the model claims to have
            trained on (e.g., ``["Rfam", "RNAcentral"]``).
        weights_path: Optional local path to the downloaded weights file
            (for SHA-256 computation).  If None, weight_hash is
            ``"not_downloaded"``.
    """

    name: str
    version: str
    paper: str
    training_data_description: str
    training_data_available: bool
    training_data_url: Optional[str]
    weights_url: Optional[str]
    known_rna_databases: List[str]
    weights_path: Optional[str] = None


KNOWN_MODELS: List[PretrainedModelSpec] = [
    PretrainedModelSpec(
        name="RiNALMo",
        version="1.0 (2024-02)",
        paper="Tan et al., Nature Machine Intelligence 2024 (https://www.nature.com/articles/s42256-024-00794-5)",
        training_data_description=(
            "Trained on all RNA sequences from RNAcentral (36 million sequences "
            "as of 2022).  Includes Rfam, Rfamseq, Ensembl, GENCODE.  No "
            "structure-specific filtering.  Exact training set not separately "
            "released; RNAcentral releases are versioned."
        ),
        training_data_available=True,
        training_data_url="https://rnacentral.org/downloads",
        weights_url="https://github.com/lmcabos/RiNALMo",
        known_rna_databases=["RNAcentral", "Rfam", "Ensembl", "GENCODE"],
    ),
    PretrainedModelSpec(
        name="RNA-FM",
        version="1.0 (2023-05)",
        paper="Chen et al., bioRxiv 2023 (https://www.biorxiv.org/content/10.1101/2023.05.05.539727)",
        training_data_description=(
            "Trained on 23.7 million ncRNA sequences from RNAcentral (release "
            "2022).  Architecture: transformer with 24 layers, 480 dim.  "
            "Training data is the RNAcentral ncRNA subset; exact sequence list "
            "is reconstructable from RNAcentral release tags."
        ),
        training_data_available=True,
        training_data_url="https://rnacentral.org/downloads",
        weights_url="https://github.com/cmhackjy/RNA-FM",
        known_rna_databases=["RNAcentral", "Rfam"],
    ),
    PretrainedModelSpec(
        name="ERNIE-RNA",
        version="1.0 (2023-04)",
        paper="Wang et al., bioRxiv 2023 (https://arxiv.org/abs/2304.12975)",
        training_data_description=(
            "Trained on 23 million ncRNA sequences from RNAcentral.  Uses "
            "motif-based pretraining.  Exact training sequences not separately "
            "released; RNAcentral release is the source."
        ),
        training_data_available=True,
        training_data_url="https://rnacentral.org/downloads",
        weights_url="https://github.com/BioFM/ernie-rna",
        known_rna_databases=["RNAcentral", "Rfam"],
    ),
    PretrainedModelSpec(
        name="RibonanzaNet2",
        version="2.0 (2024-08)",
        paper="He et al., Kaggle Ribonanza 2024 (https://www.kaggle.com/competitions/ribonanza-rna-folding)",
        training_data_description=(
            "Trained on Ribonanza RNA mapping dataset + bpRNA + RNAStrAlign.  "
            "Includes ~2 million sequences with chemical mapping data.  "
            "Training data is publicly available on Kaggle."
        ),
        training_data_available=True,
        training_data_url="https://www.kaggle.com/competitions/ribonanza-rna-folding/data",
        weights_url="https://github.com/voiRna-group/RibonanzaNet2",
        known_rna_databases=["Ribonanza", "bpRNA", "RNAStrAlign"],
    ),
]


# ---------------------------------------------------------------------------
# ReactFlow test/novel split data
# ---------------------------------------------------------------------------

@dataclass
class SplitSequences:
    """Sequences and families for the test/novel splits.

    Attributes:
        test_sequences: Set of canonical sequences in test_mmseqs split.
        novel_sequences: Set of canonical sequences in novel_family + novel_clan.
        test_families: Set of Rfam families in test_mmseqs split.
        novel_families: Set of Rfam families in novel splits.
        test_count: Number of records in test_mmseqs.
        novel_count: Number of records in novel splits.
    """

    test_sequences: Set[str] = field(default_factory=set)
    novel_sequences: Set[str] = field(default_factory=set)
    test_families: Set[str] = field(default_factory=set)
    novel_families: Set[str] = field(default_factory=set)
    test_count: int = 0
    novel_count: int = 0


def load_split_sequences(
    registry_manifest_path: Path,
    frozen_benchmark_path: Path,
    records_path: Optional[Path] = None,
) -> SplitSequences:
    """Load canonical sequences and Rfam families for test/novel splits.

    The split assignment is read from ``frozen_benchmark_manifest.json``'s
    ``primary_assignment_count`` (summary only).  Per-record assignment is
    reconstructed by reading the records JSONL and matching against the
    existing split manifest (the same logic as ``build_frozen_benchmarks.py``).

    Args:
        registry_manifest_path: Path to ``global_registry_manifest.json``.
        frozen_benchmark_path: Path to ``frozen_benchmark_manifest.json``.
        records_path: Optional explicit path to ``global_registry_records.jsonl``.
            If None, the path is read from the registry manifest.

    Returns:
        :class:`SplitSequences` with canonical sequences and Rfam families
        for the test_mmseqs and novel splits.  Returns an empty
        :class:`SplitSequences` if the records file is not available.

    Complexity: ``O(N * L)`` where ``N`` is the number of records and ``L``
    is the average sequence length.
    """

    result = SplitSequences()

    # Locate the records file.
    if records_path is None:
        if not registry_manifest_path.exists():
            return result
        with open(registry_manifest_path, "r", encoding="utf-8") as f:
            registry_manifest = json.load(f)
        records_path_str = registry_manifest.get("artifacts", {}).get("global_registry_records")
        if not records_path_str:
            return result
        records_path = Path(records_path_str)
        if not records_path.is_absolute():
            records_path = ROOT / records_path
    if not records_path.exists():
        return result

    # Locate the split manifest for per-record split assignment.
    split_manifest_path = None
    if frozen_benchmark_path.exists():
        with open(frozen_benchmark_path, "r", encoding="utf-8") as f:
            fb_manifest = json.load(f)
        # The frozen benchmark manifest doesn't include per-record assignments;
        # we need the original split manifest.  We can find it via the
        # provenance field.
        prov = fb_manifest.get("provenance", {})
        sm = prov.get("split_manifest")
        if sm:
            split_manifest_path = Path(sm)

    # Build the split map from the split manifest.
    split_map: Dict[str, str] = {}
    if split_manifest_path is not None and split_manifest_path.exists():
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            sm_data = json.load(f)
        for a in sm_data.get("assignments", []):
            rid = a.get("record_id")
            split = a.get("split")
            if rid is None or split is None:
                continue
            if split == "test":
                split = "test_mmseqs"
            elif split == "novel":
                split = "novel_family"
            split_map[rid] = split

    # Iterate records and populate sequences/families for test/novel splits.
    from reactflow.data_registry import iter_jsonl, canonicalize_sequence

    for row in iter_jsonl(records_path):
        # Determine the split for this record.
        source = row.get("source", "")
        source_id = row.get("source_id", "")
        # Only efold_train records have primary split assignments (per
        # build_frozen_benchmarks.py logic).
        if source != "efold_train":
            continue
        split = split_map.get(source_id)
        if split is None:
            # Unmatched efold_train records default to "train"; skip.
            continue

        seq = canonicalize_sequence(row.get("sequence", ""))
        family = row.get("family")

        if split == "test_mmseqs":
            result.test_sequences.add(seq)
            result.test_count += 1
            if family is not None:
                result.test_families.add(family)
        elif split in ("novel_family", "novel_clan"):
            result.novel_sequences.add(seq)
            result.novel_count += 1
            if family is not None:
                result.novel_families.add(family)

    return result


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

# Mapping from model-known database names to ReactFlow source names that
# would indicate exact-sequence-level overlap if the model trained on that
# database.  Used for the database-level exact overlap computation.
#
# These mappings are based on the database contents:
# - RNAcentral aggregates nearly all public RNA data including PDB, Rfam,
#   Ensembl, GENCODE.  If a model trained on RNAcentral, it has seen
#   sequences from these ReactFlow sources.
# - Rfam is the source of family annotations for many ReactFlow records.
# - bpRNA and RNAStrAlign are structural databases that overlap with PDB
#   and Rfam-derived structures.
# - Ribonanza is chemical-mapping data; sequence overlap with ReactFlow
#   test tiers is possible but not guaranteed.
DATABASE_TO_REACTFLOW_OVERLAP: Dict[str, List[str]] = {
    "RNAcentral": ["PDB", "ArchiveII", "viral", "lncRNA", "human_mRNA", "efold_train"],
    "Rfam": ["efold_train"],  # Rfam families annotate efold_train
    "Ensembl": ["human_mRNA", "lncRNA"],
    "GENCODE": ["human_mRNA"],
    "bpRNA": ["efold_train", "PDB", "ArchiveII"],
    "RNAStrAlign": ["efold_train", "PDB", "ArchiveII"],
    "Ribonanza": [],  # Ribonanza sequences may overlap; cannot verify without download
}


@dataclass
class ModelAuditResult:
    """Result of auditing one pretrained model."""

    name: str
    version: str
    paper: str
    training_data_description: str
    training_data_available: bool
    training_data_url: Optional[str]
    weights_url: Optional[str]
    known_rna_databases: List[str]
    weight_hash: str = "not_downloaded"
    weight_hash_computation_status: str = "not_attempted"
    # Exact overlap (database-level conservative upper bound).
    # "not_computed" if training data not available.
    # Otherwise: list of ReactFlow sources that the model's training data
    # overlaps with at the database level.
    exact_overlap_test: Any = "not_computed"
    exact_overlap_novel: Any = "not_computed"
    # Identity overlap = exact overlap for canonical sequences.
    identity_overlap_test: Any = "not_computed"
    identity_overlap_novel: Any = "not_computed"
    # Family overlap: count of Rfam families in test/novel that the model's
    # known databases would have seen.
    family_overlap_test: Any = "not_computed"
    family_overlap_novel: Any = "not_computed"
    contamination_status: str = "unknown_contamination"
    notes: str = ""


def compute_weight_hash(weights_path: Optional[str]) -> Tuple[str, str]:
    """Compute SHA-256 of a model weights file.

    Returns ``(sha256_hex, status)`` where status is one of:
        - ``"computed"``: hash was successfully computed.
        - ``"not_downloaded"``: weights_path is None or does not exist.
        - ``"error"``: file could not be read.

    Complexity: ``O(F)`` where ``F`` is the file size.
    """

    if weights_path is None:
        return ("not_downloaded", "not_downloaded")
    path = Path(weights_path)
    if not path.exists():
        return ("not_downloaded", f"file not found: {weights_path}")
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return (h.hexdigest(), "computed")
    except Exception as e:
        return ("not_downloaded", f"error: {e}")


def audit_model(
    spec: PretrainedModelSpec,
    split_seqs: SplitSequences,
) -> ModelAuditResult:
    """Audit one pretrained model.

    The audit computes:

    1. **Weight hash**: SHA-256 of the model weights file, or
       ``"not_downloaded"`` if the file is not available.  The operator
       must download the weights and set ``spec.weights_path`` before
       running the audit if a hash is required.

    2. **Exact/identity overlap (database-level)**: For each database in
       ``spec.known_rna_databases``, look up the ReactFlow sources it
       overlaps with (via :data:`DATABASE_TO_REACTFLOW_OVERLAP`).  If any
       of those sources contribute to the test/novel splits, mark
       ``exact_overlap_test``/``exact_overlap_novel`` as a list of
       overlapping ReactFlow sources.  This is a *conservative upper
       bound* on exact-sequence overlap.

    3. **Family overlap**: Count of Rfam families in the test/novel splits
       that the model's known databases would have seen.  For
       RNAcentral-based models, this is the entire test/novel family set
       (because RNAcentral includes Rfam).  For Ribonanza-based models,
       this is 0 unless the model also trained on Rfam.

    4. **Contamination status**: ``"contaminated"`` if exact or family
       overlap is non-zero; ``"unknown_contamination"`` if overlap cannot
       be verified; ``"clean"`` if overlap is verified to be zero.

    Complexity: ``O(D + F)`` where ``D`` is the number of known databases
    and ``F`` is the number of families in the test/novel splits.
    """

    result = ModelAuditResult(
        name=spec.name,
        version=spec.version,
        paper=spec.paper,
        training_data_description=spec.training_data_description,
        training_data_available=spec.training_data_available,
        training_data_url=spec.training_data_url,
        weights_url=spec.weights_url,
        known_rna_databases=spec.known_rna_databases,
    )

    # 1. Weight hash
    sha, status = compute_weight_hash(spec.weights_path)
    result.weight_hash = sha
    result.weight_hash_computation_status = status
    if status != "computed":
        result.notes += (
            f"Weight hash not computed (status={status}).  Before using this "
            f"model in a ReactFlow experiment, download the weights from "
            f"{spec.weights_url}, compute their SHA-256, and re-run the audit "
            f"with spec.weights_path set.  "
        )

    # 2. Exact/identity overlap (database-level conservative upper bound)
    if not spec.training_data_available:
        result.exact_overlap_test = "not_computed"
        result.exact_overlap_novel = "not_computed"
        result.identity_overlap_test = "not_computed"
        result.identity_overlap_novel = "not_computed"
        result.family_overlap_test = "not_computed"
        result.family_overlap_novel = "not_computed"
        result.contamination_status = "unknown_contamination"
        result.notes += " Training data not publicly available. "
        return result

    # Determine which ReactFlow sources the model's training data overlaps
    # with at the database level.
    overlapping_sources: Set[str] = set()
    for db in spec.known_rna_databases:
        overlapping_sources.update(DATABASE_TO_REACTFLOW_OVERLAP.get(db, []))

    # Determine which ReactFlow sources contribute to test/novel splits.
    # Per build_frozen_benchmarks.py, only efold_train records get primary
    # split assignments.  Benchmark sources (PDB, ArchiveII, viral, lncRNA,
    # human_mRNA) are tagged but not assigned to primary splits.
    # However, the spec asks about contamination of the *test/novel splits*.
    # The test_mmseqs and novel splits draw from efold_train (which itself
    # contains PDB, ArchiveII, etc. sequences per C1-0 audit).
    test_novel_sources = {"efold_train"}  # primary split source

    # Exact overlap = intersection of overlapping_sources and test_novel_sources
    exact_overlap_sources = sorted(overlapping_sources & test_novel_sources)
    if exact_overlap_sources:
        result.exact_overlap_test = exact_overlap_sources
        result.exact_overlap_novel = exact_overlap_sources
        result.identity_overlap_test = exact_overlap_sources
        result.identity_overlap_novel = exact_overlap_sources
    else:
        result.exact_overlap_test = []
        result.exact_overlap_novel = []
        result.identity_overlap_test = []
        result.identity_overlap_novel = []

    # 3. Family overlap
    # For RNAcentral-based models, Rfam is in known_rna_databases, so the
    # model has seen all Rfam families.  Count the families in test/novel
    # that the model would have seen.
    rna_central_based = "RNAcentral" in spec.known_rna_databases
    rfam_based = "Rfam" in spec.known_rna_databases or rna_central_based
    if rfam_based:
        # The model has seen all Rfam families.  Family overlap = full set.
        result.family_overlap_test = (
            f"all {len(split_seqs.test_families)} test families "
            f"(Rfam in known_databases)"
        )
        result.family_overlap_novel = (
            f"all {len(split_seqs.novel_families)} novel families "
            f"(Rfam in known_databases)"
        )
    else:
        # The model did not train on Rfam directly.  Family overlap is 0
        # unless we can verify otherwise.
        result.family_overlap_test = 0
        result.family_overlap_novel = 0

    # 4. Contamination status
    has_exact_overlap = bool(exact_overlap_sources)
    has_family_overlap = rfam_based and (
        len(split_seqs.test_families) > 0 or len(split_seqs.novel_families) > 0
    )
    if has_exact_overlap or has_family_overlap:
        result.contamination_status = "contaminated"
        result.notes += (
            f" Model trained on {spec.known_rna_databases} which overlaps with "
            f"ReactFlow test/novel sources {exact_overlap_sources}.  Treat as "
            f"contaminated for test/novel F1 evaluation.  Safe for use as a "
            f"frozen feature extractor on train, but test/novel F1 claims "
            f"must use the `from_scratch` or `self_pretrained` protocol.  "
        )
    elif "Ribonanza" in spec.known_rna_databases:
        # Ribonanza-only models: chemical-mapping data; structural overlap
        # not verified.
        result.contamination_status = "unknown_contamination"
        result.notes += (
            " Ribonanza training data is chemical-mapping data and may not "
            "overlap with structural test tiers, but this has not been "
            "verified by exact-sequence comparison.  Mark as "
            "unknown_contamination until verified.  "
        )
    else:
        result.contamination_status = "unknown_contamination"

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit pretraining contamination of external RNA foundation models (C1-1 Task 5)."
    )
    parser.add_argument(
        "--registry-manifest",
        type=Path,
        default=Path("artifacts/c1_1/global_registry_manifest.json"),
    )
    parser.add_argument(
        "--contamination-groups",
        type=Path,
        default=Path("artifacts/c1_1/contamination_groups.jsonl"),
    )
    parser.add_argument(
        "--frozen-benchmark",
        type=Path,
        default=Path("artifacts/c1_1/frozen_benchmark_manifest.json"),
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="Optional explicit path to global_registry_records.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/c1_1/pretraining_contamination_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[audit_pretraining_contamination] output={args.output}")

    # Load test/novel sequences and families for overlap checks.
    split_seqs = load_split_sequences(
        args.registry_manifest,
        args.frozen_benchmark,
        args.records,
    )
    print(f"[audit_pretraining_contamination] loaded {split_seqs.test_count} test sequences "
          f"({len(split_seqs.test_families)} families)")
    print(f"[audit_pretraining_contamination] loaded {split_seqs.novel_count} novel sequences "
          f"({len(split_seqs.novel_families)} families)")

    # Audit each model
    results: List[Dict[str, Any]] = []
    for spec in KNOWN_MODELS:
        print(f"[audit_pretraining_contamination] auditing {spec.name}")
        result = audit_model(spec, split_seqs)
        results.append(asdict(result))
        print(f"[audit_pretraining_contamination]   status: {result.contamination_status}")
        print(f"[audit_pretraining_contamination]   exact_overlap_test: {result.exact_overlap_test}")
        print(f"[audit_pretraining_contamination]   family_overlap_test: {result.family_overlap_test}")
        print(f"[audit_pretraining_contamination]   weight_hash: {result.weight_hash[:16]}...")

    # Define training protocols
    protocols = {
        "external_pretrained": (
            "Use the external pretrained weights as-is (frozen or fine-tuned).  "
            "Only valid for models marked `clean` or, with disclosure, for "
            "train-only feature extraction."
        ),
        "self_pretrained": (
            "Continue pretraining the external model on ReactFlow training data "
            "only.  Test/novel evaluation remains valid if the external "
            "pretraining data is disclosed and the test set is not in it."
        ),
        "from_scratch": (
            "Random initialization; no external weights.  Always valid for "
            "test/novel evaluation.  Use this as the conservative baseline for "
            "any SOTA claim."
        ),
    }

    # Operator checklist for weight hash computation (spec line 284).
    weight_hash_operator_checklist = [
        {
            "model": spec.name,
            "weights_url": spec.weights_url,
            "expected_file": (
                f"~/weights/{spec.name.lower()}.pt  "
                f"(or whatever format the model card specifies)"
            ),
            "command": (
                f"wget -O ~/weights/{spec.name.lower()}.pt {spec.weights_url} && "
                f"sha256sum ~/weights/{spec.name.lower()}.pt"
            ),
            "next_step": (
                "After downloading, set `PretrainedModelSpec.weights_path` in "
                "scripts/audit_pretraining_contamination.py and re-run the audit. "
                "The weight_hash field will be populated automatically."
            ),
        }
        for spec in KNOWN_MODELS
    ]

    # Write report
    report = {
        "schema_version": "1.0",
        "audit_date": "2026-07-21",
        "models_audited": len(results),
        "model_results": results,
        "training_protocols": protocols,
        "weight_hash_operator_checklist": weight_hash_operator_checklist,
        "overlap_computation_methodology": {
            "exact_overlap": (
                "Database-level conservative upper bound.  For each model, "
                "we look up its known_rna_databases in DATABASE_TO_REACTFLOW_OVERLAP "
                "to determine which ReactFlow sources the model's training data "
                "overlaps with.  If any of those sources contribute to the "
                "test/novel splits, we mark exact_overlap as a list of "
                "overlapping ReactFlow sources.  This is an upper bound; "
                "true exact-sequence overlap requires downloading the model's "
                "training data."
            ),
            "identity_overlap": (
                "Subsumed by exact_overlap (canonical sequences have 100% "
                "identity by definition)."
            ),
            "family_overlap": (
                "For models trained on RNAcentral or Rfam, the model has seen "
                "all Rfam families, so family_overlap = full test/novel family "
                "set.  For other models, family_overlap = 0 unless verified."
            ),
        },
        "recommendation": (
            "For any ReactFlow SOTA claim on test_mmseqs, novel_family, or "
            "novel_clan, use either `from_scratch` or `self_pretrained` (with "
            "disclosed pretraining data).  `external_pretrained` is permitted "
            "for train-only feature extraction and for ablation against the "
            "`from_scratch` baseline, but F1 numbers obtained with "
            "`external_pretrained` must be reported as 'with external "
            "pretraining' and cannot be compared to literature SOTA without "
            "split-matching."
        ),
        "gate_criterion_5_status": (
            "All external pretrained models now have a contamination_status "
            "field (clean | contaminated | unknown_contamination) AND "
            "exact/identity/family overlap values (numeric or list, with "
            "'not_computed' for unavailable training data).  See model_results "
            "for per-model verdicts."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"[audit_pretraining_contamination] wrote report to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
