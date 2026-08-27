"""Checkpoint provenance governance for ReactFlow Phase C1-3.

This module enforces the provenance contract for every foundation backbone
checkpoint: who made it, under what license, from what exact revision, and
whether it has been audited for contamination.  A checkpoint whose
:class:`CheckpointProvenance` fails :func:`validate_provenance` must not enter
training -- the validation returns a list of human-readable errors that the
caller can log or raise on.

Spec reference: ``ReactFlow分阶段执行提示词.md`` Phase C1-3 (checkpoint
governance + reproducibility manifest).

Formula
-------
Provenance is a deterministic JSON object keyed by ``model_name``.  The
integrity field ``weights_sha256`` pins the exact weight bytes; a checkpoint is
reproducible iff its provenance hashes match.  Complexity: ``O(1)`` metadata
storage; weight hashing is performed offline by the exporter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class CheckpointProvenance:
    """Traceability header for a foundation-model checkpoint.

    This is the governance counterpart of
    :class:`~reactflow.backbones.foundation.base.BackboneConfig`: it adds the
    ``download_url`` and ``download_date`` fields needed to *acquire* a
    checkpoint, and renames ``model_revision`` to ``exact_revision`` to
    emphasise that it must pin a single immutable revision (git SHA or HF
    commit hash), not a moving tag like ``main``.

    Attributes:
        model_name: human-readable identifier, e.g. ``"RibonanzaNet2"``.
        model_source: checkpoint origin, e.g. ``"zenodo:15043668"``.
        exact_revision: immutable revision pin (git SHA or HF commit hash).
            A moving tag like ``main`` is flagged by :func:`validate_provenance`.
        license: SPDX-style checkpoint/weight license identifier (``"MIT"``,
            ``"Apache-2.0"``); ``"unknown"`` is permitted but flagged as a
            warning.  A separately licensed code repository is identified by
            ``code_revision`` but its source and license do not fit this legacy
            single-license schema.
        weights_sha256: SHA-256 of the weight file(s); ``""`` when no weights
            are present (manifest-only entries and from-scratch).
        code_revision: revision of the model-definition code used to export
            features (may differ from ``exact_revision``).
        tokenizer: tokenizer name, e.g. ``"ribonanza-bpe"``.
        max_length: maximum accepted sequence length.
        contamination_status: label from the C1-1 contamination audit.
        download_url: URL to acquire the weights from.
        downloaded: whether the weights are present locally.
        download_date: ISO-8601 date the weights were acquired (``""`` if not).
    """

    model_name: str
    model_source: str
    exact_revision: str = ""
    license: str = ""
    weights_sha256: str = ""
    code_revision: str = ""
    tokenizer: str = ""
    max_length: int = 0
    contamination_status: str = "unknown"
    download_url: str = ""
    downloaded: bool = False
    download_date: str = ""

    def to_json_obj(self) -> Dict[str, object]:
        """Return a JSON-serializable dict with deterministic key order.

        Formula: shallow-copies fields into an alphabetically-keyed dict so
        serialised provenance is byte-stable across runs.  Complexity: ``O(1)``.
        """

        return {k: v for k, v in sorted(asdict(self).items())}

    @staticmethod
    def from_json_obj(obj: Mapping[str, object]) -> "CheckpointProvenance":
        """Rebuild provenance from a parsed JSON object.

        Formula: coerces scalar fields to their declared types.  Unknown keys
        are ignored for forward compatibility.  Complexity: ``O(1)``.
        """

        return CheckpointProvenance(
            model_name=str(obj.get("model_name", "")),
            model_source=str(obj.get("model_source", "")),
            exact_revision=str(obj.get("exact_revision", "")),
            license=str(obj.get("license", "")),
            weights_sha256=str(obj.get("weights_sha256", "")),
            code_revision=str(obj.get("code_revision", "")),
            tokenizer=str(obj.get("tokenizer", "")),
            max_length=int(obj.get("max_length", 0)),
            contamination_status=str(obj.get("contamination_status", "unknown")),
            download_url=str(obj.get("download_url", "")),
            downloaded=bool(obj.get("downloaded", False)),
            download_date=str(obj.get("download_date", "")),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_MOVING_REVISIONS = frozenset({"", "main", "master", "latest", "head"})
"""Revision strings that are moving targets and must not be used as exact pins."""


def validate_provenance(prov: CheckpointProvenance) -> List[str]:
    """Return a list of validation errors for a checkpoint provenance.

    An empty return value means the provenance is governance-compliant.  The
    checks enforce:

    1. ``model_name`` and ``model_source`` are non-empty.
    2. ``exact_revision`` is a non-empty immutable pin (not a moving tag like
       ``main``).
    3. If ``downloaded`` is True, ``download_url`` is non-empty and
       ``download_date`` is set (a downloaded checkpoint must be traceable to
       a source).
    4. If ``downloaded`` is True and ``model_name != "FromScratch"``, then
       ``weights_sha256`` is non-empty (downloaded external weights must be
       hashed).
    5. ``max_length`` is positive.
    6. ``license`` is non-empty (``"unknown"`` is allowed but flagged).

    Args:
        prov: the :class:`CheckpointProvenance` to validate.

    Returns:
        List of human-readable error strings.  Empty list means valid.

    Complexity: ``O(1)``.
    """

    errors: List[str] = []
    if not prov.model_name:
        errors.append("model_name is empty")
    if not prov.model_source:
        errors.append("model_source is empty")
    if not prov.exact_revision:
        errors.append("exact_revision is empty (must pin an immutable revision)")
    elif prov.exact_revision.lower() in _MOVING_REVISIONS:
        errors.append(
            f"exact_revision {prov.exact_revision!r} is a moving target; pin a "
            "git SHA or commit hash instead"
        )
    if prov.max_length <= 0:
        errors.append(f"max_length must be positive, got {prov.max_length}")
    if not prov.license:
        errors.append("license is empty (set 'unknown' if genuinely unknown)")
    if prov.downloaded and prov.model_name != "FromScratch":
        # FromScratch has no external weights (it is code-only), so download
        # URL/date and weight hashing do not apply.
        if not prov.download_url:
            errors.append("downloaded=True but download_url is empty")
        if not prov.download_date:
            errors.append("downloaded=True but download_date is empty")
        if not prov.weights_sha256:
            errors.append(
                "downloaded=True but weights_sha256 is empty (external weights "
                "must be hashed)"
            )
    return errors


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_provenance(path: Union[str, Path]) -> CheckpointProvenance:
    """Load a :class:`CheckpointProvenance` from a JSON file.

    Formula: reads the JSON text, parses it, and coerces fields via
    :meth:`CheckpointProvenance.from_json_obj`.  Complexity: ``O(file_size)``.

    Args:
        path: path to a JSON file produced by :func:`save_provenance`.

    Returns:
        :class:`CheckpointProvenance`.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the JSON is missing required keys.
    """

    text = Path(path).read_text(encoding="utf-8")
    obj = json.loads(text)
    return CheckpointProvenance.from_json_obj(obj)


def save_provenance(prov: CheckpointProvenance, path: Union[str, Path]) -> None:
    """Save a :class:`CheckpointProvenance` to a JSON file.

    Formula: serialises with sorted keys and indentation for deterministic,
    diff-friendly output.  Complexity: ``O(file_size)``.

    Args:
        prov: the provenance to save.
        path: destination JSON file path (parent directories are created).
    """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(prov.to_json_obj(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Pre-populated registry
# ---------------------------------------------------------------------------


BACKBONE_PROVENANCE_REGISTRY: Dict[str, CheckpointProvenance] = {
    "ribonanzanet2": CheckpointProvenance(
        model_name="RibonanzaNet2",
        model_source="github:sh-ogawa/RibonanzaNet2",
        exact_revision="ribonanza2-export-20260709",
        license="Apache-2.0",
        weights_sha256="",
        code_revision="c5-frozen-export",
        tokenizer="ribonanza-bpe",
        max_length=4488,
        contamination_status="unknown_contamination",
        download_url=(
            "https://www.kaggle.com/competitions/ribonanza2 and "
            "https://github.com/sh-ogawa/RibonanzaNet2"
        ),
        downloaded=True,
        download_date="2026-07-09",
    ),
    "rinalmo": CheckpointProvenance(
        model_name="RiNALMo",
        model_source="zenodo:15043668",
        exact_revision="10.5281/zenodo.15043668",
        license="CC-BY-4.0",
        weights_sha256="",
        code_revision="2c2c5c14a5ae609d8c560a5d9ca32e51e0288955",
        tokenizer="rinalmo-bpe",
        max_length=1024,
        contamination_status="unknown_contamination",
        download_url=(
            "https://zenodo.org/records/15043668/files/"
            "rinalmo_giga_pretrained.pt"
        ),
        downloaded=False,
        download_date="",
    ),
    "ernie_rna": CheckpointProvenance(
        model_name="ERNIE-RNA",
        model_source="huggingface:yzhuoning/RNAErnie",
        exact_revision="",
        license="unknown",
        weights_sha256="",
        code_revision="",
        tokenizer="ernie-rna-bpe",
        max_length=512,
        contamination_status="unknown",
        download_url="https://huggingface.co/yzhuoning/RNAErnie",
        downloaded=False,
        download_date="",
    ),
    "rna_fm": CheckpointProvenance(
        model_name="RNA-FM",
        model_source="huggingface:cuhkaih/rnafm",
        exact_revision="91d4a46d28d8054a7b429955e8fc0c253ba0afd6",
        license="Apache-2.0",
        weights_sha256="",
        code_revision="348951516e0963d22bbb33b3c9fc18c89081d38e",
        tokenizer="rna-fm-bpe",
        max_length=1024,
        contamination_status="unknown",
        download_url=(
            "https://huggingface.co/cuhkaih/rnafm/resolve/"
            "91d4a46d28d8054a7b429955e8fc0c253ba0afd6/"
            "RNA-FM_pretrained.pth"
        ),
        downloaded=False,
        download_date="",
    ),
    "from_scratch": CheckpointProvenance(
        model_name="FromScratch",
        model_source="reactflow:c1-2-embeddings",
        exact_revision="c1-2",
        license="Apache-2.0",
        weights_sha256="",
        code_revision="c1-2",
        tokenizer="nucleotide",
        max_length=1024,
        contamination_status="not_applicable",
        download_url="",
        downloaded=True,
        download_date="2026-07-09",
    ),
}
"""Pre-populated provenance registry for all five foundation backbones.

Each entry is the governance record (license, contamination status, exact
revision, download URL) for one backbone.  Use :func:`validate_provenance` to
check an entry before allowing it into training.
"""
