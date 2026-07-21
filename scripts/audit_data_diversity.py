#!/usr/bin/env python3
"""Audit ReactFlow data diversity and structural complexity.

The eFold/RNAndria cross-family lesson is data-centric: a structure model
should not rely only on more parameters or more rows from the same narrow
domain.  This script streams ReactFlow JSONL caches/splits and writes an
artifact-level audit of source mix, family/clan coverage, length buckets,
window provenance, long-range pairs, and simple stem complexity.

Complexity: O(N * P), where N is the number of records and P is the mean number
of base pairs per record; memory is O(U) for unique source/family counters.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence


DEFAULT_FULL_RUN = Path("artifacts/full_runs/full_ablation_20260709_003012")
DEFAULT_CACHE_FILES = {
    "efold_train": Path("cache/efold_train.jsonl"),
    "PDB": Path("cache/PDB.jsonl"),
    "archiveII": Path("cache/archiveII.jsonl"),
    "viral": Path("cache/viral.jsonl"),
    "lncRNA": Path("cache/lncRNA.jsonl"),
    "human_mRNA": Path("cache/human_mRNA.jsonl"),
}
DEFAULT_SPLIT_FILES = {
    "exact_train": Path("splits/rfam_current_exact_seed0/train.jsonl"),
    "exact_val": Path("splits/rfam_current_exact_seed0/val.jsonl"),
    "exact_test": Path("splits/rfam_current_exact_seed0/test.jsonl"),
    "exact_novel": Path("splits/rfam_current_exact_seed0/novel.jsonl"),
    "mmseqs_train": Path("splits/rfam_current_mmseqs_seed0/train.jsonl"),
    "mmseqs_test": Path("splits/rfam_current_mmseqs_seed0/test.jsonl"),
    "mmseqs_novel": Path("splits/rfam_current_mmseqs_seed0/novel.jsonl"),
}
LENGTH_BUCKETS = (64, 128, 256, 512, 1024)


def _bucket_length(length: int) -> str:
    """Return a stable length bucket label.

    Formula: choose the first threshold ``b`` such that ``length <= b``; records
    above the largest threshold use ``len_gt_<max>``.  Complexity: O(B), with B
    fixed by ``LENGTH_BUCKETS``.
    """

    for bound in LENGTH_BUCKETS:
        if length <= bound:
            return f"len_le_{bound}"
    return f"len_gt_{LENGTH_BUCKETS[-1]}"


def _as_counter_key(value: object) -> str:
    """Normalize optional JSON metadata into a counter key.

    Formula: missing, ``None``, and empty strings map to ``unknown``; all other
    values are stringified.  Complexity: O(len(str(value))).
    """

    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def _source_group(record: Mapping[str, object], dataset: str) -> str:
    """Infer a coarse source group for diversity diagnostics.

    Formula: prefer explicit clan/family metadata when present; otherwise use a
    stable prefix from ``source_id`` before window/CSV delimiters.  Complexity:
    O(len(source_id)).
    """

    for key in ("source", "source_group", "clan", "family"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    source_id = _as_counter_key(record.get("source_id"))
    if source_id == "unknown":
        return dataset
    for delimiter in (":", ".fa.csv", "_IRAlu", "_"):
        if delimiter in source_id:
            prefix = source_id.split(delimiter, 1)[0]
            if prefix:
                return prefix
    return source_id


def _is_pseudo_clan(key: str) -> bool:
    """Return whether a clan key is fallback metadata rather than Rfam clan.

    Formula: ReactFlow metadata builders use prefixes such as ``component:`` and
    ``unannotated:`` when true Rfam family/clan membership is unavailable.
    Complexity: O(len(key)).
    """

    return key.startswith("component:") or key.startswith("unannotated:")


def _pairs(record: Mapping[str, object], *, length: int) -> list[tuple[int, int]]:
    """Extract valid zero-based base pairs from one record.

    Formula: retain pair-like entries ``(i, j)`` with ``0 <= i < j < length``;
    reversed pairs are normalized and diagonal/out-of-bounds artifacts are
    skipped because they are not legal RNA contacts.  Complexity: O(P).
    """

    raw_pairs = record.get("pairs", [])
    if not isinstance(raw_pairs, list):
        return []
    out: list[tuple[int, int]] = []
    for item in raw_pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        left, right = item
        if not isinstance(left, int) or not isinstance(right, int):
            continue
        if left == right:
            continue
        if right < left:
            left, right = right, left
        if 0 <= left < right < length:
            out.append((left, right))
    return sorted(set(out))


def _stem_stats(pairs: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Return approximate stem count and maximum stacked-stem length.

    Formula: a pair ``(i, j)`` starts a stem when ``(i-1, j+1)`` is absent; the
    stack length is the number of consecutive ``(i+k, j-k)`` pairs.  Complexity:
    O(P) expected with set membership.
    """

    pair_set = set(pairs)
    stem_count = 0
    max_stem = 0
    for left, right in pairs:
        if (left - 1, right + 1) in pair_set:
            continue
        stem_count += 1
        run = 0
        probe = (left, right)
        while probe in pair_set:
            run += 1
            probe = (probe[0] + 1, probe[1] - 1)
        max_stem = max(max_stem, run)
    return stem_count, max_stem


def _mean(total: float, count: int) -> Optional[float]:
    """Return ``total / count`` or ``None`` for empty inputs.

    Complexity: O(1).
    """

    return None if count <= 0 else total / count


def _top(counter: Counter[str], limit: int = 10) -> list[dict[str, object]]:
    """Return top counter entries as JSON-serializable rows.

    Complexity: O(U log U) for U unique keys.
    """

    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


@dataclass
class DatasetAudit:
    """Streaming accumulator for one dataset.

    Complexity: O(U) storage for unique metadata keys and O(1) numeric state.
    """

    label: str
    path: Path
    long_range_min_distance: int
    records: int = 0
    sequence_total: int = 0
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pair_total: int = 0
    long_pair_total: int = 0
    stem_total: int = 0
    max_stem: int = 0
    windowed_records: int = 0
    parent_ids: set[str] = field(default_factory=set)
    parent_min_length: Optional[int] = None
    parent_max_length: Optional[int] = None
    with_reactivity: int = 0
    length_buckets: Counter[str] = field(default_factory=Counter)
    provided_length_buckets: Counter[str] = field(default_factory=Counter)
    clans: Counter[str] = field(default_factory=Counter)
    families: Counter[str] = field(default_factory=Counter)
    clusters: Counter[str] = field(default_factory=Counter)
    source_groups: Counter[str] = field(default_factory=Counter)
    reactivity_sources: Counter[str] = field(default_factory=Counter)
    accessions: Counter[str] = field(default_factory=Counter)
    sequence_hashes: set[str] = field(default_factory=set)

    def add(self, record: Mapping[str, object]) -> None:
        """Add one JSONL record to the audit.

        Formula: update counters from sequence length, metadata fields, pair
        distances, and optional window provenance.  Complexity: O(P).
        """

        sequence = record.get("sequence", "")
        if not isinstance(sequence, str):
            sequence = ""
        length = len(sequence)
        self.sequence_hashes.add(hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest())
        pairs = _pairs(record, length=length)
        long_pairs = sum(1 for left, right in pairs if right - left >= self.long_range_min_distance)
        stem_count, max_stem = _stem_stats(pairs)

        self.records += 1
        self.sequence_total += length
        self.min_length = length if self.min_length is None else min(self.min_length, length)
        self.max_length = length if self.max_length is None else max(self.max_length, length)
        self.pair_total += len(pairs)
        self.long_pair_total += long_pairs
        self.stem_total += stem_count
        self.max_stem = max(self.max_stem, max_stem)
        self.length_buckets[_bucket_length(length)] += 1
        self.provided_length_buckets[_as_counter_key(record.get("length_bucket"))] += 1
        self.clans[_as_counter_key(record.get("clan"))] += 1
        self.families[_as_counter_key(record.get("family"))] += 1
        self.clusters[_as_counter_key(record.get("cluster"))] += 1
        self.source_groups[_source_group(record, self.label)] += 1
        self.reactivity_sources[_as_counter_key(record.get("reactivity_source"))] += 1
        source_id = _as_counter_key(record.get("source_id"))
        accession_match = re.search(r"\b(RF\d{5})\b", source_id)
        self.accessions[accession_match.group(1) if accession_match else "unknown"] += 1

        reactivity = record.get("reactivity")
        if isinstance(reactivity, list) and any(value is not None for value in reactivity):
            self.with_reactivity += 1

        window = record.get("window")
        if isinstance(window, Mapping):
            self.windowed_records += 1
            parent_id = str(record.get("source_id", "")).split(":", 1)[0]
            if parent_id:
                self.parent_ids.add(parent_id)
            parent_length = window.get("parent_length")
            if isinstance(parent_length, int) and not isinstance(parent_length, bool):
                self.parent_min_length = (
                    parent_length if self.parent_min_length is None else min(self.parent_min_length, parent_length)
                )
                self.parent_max_length = (
                    parent_length if self.parent_max_length is None else max(self.parent_max_length, parent_length)
                )

    def summary(self) -> dict[str, object]:
        """Return the dataset summary row.

        Formula: derive ratios from accumulated totals and include top metadata
        counters for source/family/length diagnosis.  Complexity: O(U log U).
        """

        top_source_count = self.source_groups.most_common(1)[0][1] if self.source_groups else 0
        unknown_clan_count = self.clans.get("unknown", 0)
        unknown_family_count = self.families.get("unknown", 0)
        pseudo_clan_count = sum(count for key, count in self.clans.items() if _is_pseudo_clan(key))
        true_clan_count = sum(
            count for key, count in self.clans.items() if key != "unknown" and not _is_pseudo_clan(key)
        )
        accession_count = self.records - self.accessions.get("unknown", 0)
        component_counter = self.clusters if any(key != "unknown" for key in self.clusters) else self.clans
        component_sizes = sorted(
            (count for key, count in component_counter.items() if key != "unknown"), reverse=True
        )
        return {
            "label": self.label,
            "path": str(self.path),
            "record_count": self.records,
            "length": {
                "min": self.min_length,
                "max": self.max_length,
                "mean": _mean(self.sequence_total, self.records),
                "buckets": dict(sorted(self.length_buckets.items())),
                "provided_buckets": dict(sorted(self.provided_length_buckets.items())),
            },
            "source_mix": {
                "unique_groups": len(self.source_groups),
                "top_groups": _top(self.source_groups),
                "top_group_fraction": _mean(top_source_count, self.records),
            },
            "family_clan": {
                "unique_clans": len([key for key in self.clans if key != "unknown"]),
                "unique_families": len([key for key in self.families if key != "unknown"]),
                "unique_clusters": len([key for key in self.clusters if key != "unknown"]),
                "unknown_clan_fraction": _mean(unknown_clan_count, self.records),
                "unknown_family_fraction": _mean(unknown_family_count, self.records),
                "pseudo_clan_fraction": _mean(pseudo_clan_count, self.records),
                "true_clan_fraction": _mean(true_clan_count, self.records),
                "rfam_accession_fraction": _mean(accession_count, self.records),
                "unique_rfam_accessions": len([key for key in self.accessions if key != "unknown"]),
                "component_size_distribution": {
                    "component_count": len(component_sizes),
                    "largest": component_sizes[0] if component_sizes else None,
                    "median": component_sizes[len(component_sizes) // 2] if component_sizes else None,
                    "singleton_count": sum(1 for size in component_sizes if size == 1),
                },
                "top_clans": _top(self.clans),
                "top_families": _top(self.families),
            },
            "structure_complexity": {
                "mean_pair_count": _mean(self.pair_total, self.records),
                "mean_stem_count": _mean(self.stem_total, self.records),
                "max_stem_length": self.max_stem,
                "long_range_min_distance": self.long_range_min_distance,
                "long_range_pair_count": self.long_pair_total,
                "long_range_pair_fraction": _mean(self.long_pair_total, self.pair_total),
                "mean_long_range_pairs_per_record": _mean(self.long_pair_total, self.records),
            },
            "reactivity": {
                "with_reactivity_count": self.with_reactivity,
                "with_reactivity_fraction": _mean(self.with_reactivity, self.records),
                "sources": dict(sorted(self.reactivity_sources.items())),
            },
            "windowing": {
                "windowed_record_count": self.windowed_records,
                "windowed_fraction": _mean(self.windowed_records, self.records),
                "unique_parent_count": len(self.parent_ids),
                "parent_length_min": self.parent_min_length,
                "parent_length_max": self.parent_max_length,
            },
            "warnings": _dataset_warnings(self),
        }

    def top_source_fraction(self) -> Optional[float]:
        """Return the dominant source-group fraction for warning logic.

        Complexity: O(U) for U source groups.
        """

        top_count = self.source_groups.most_common(1)[0][1] if self.source_groups else 0
        return _mean(top_count, self.records)


def _dataset_warnings(audit: DatasetAudit) -> list[str]:
    """Return conservative diversity warnings for one dataset.

    Formula: flag single-bucket length coverage, dominant source groups,
    missing family/clan metadata, absent long-range pairs, and windowed datasets
    without parent provenance.  Complexity: O(U).
    """

    warnings: list[str] = []
    if audit.records == 0:
        return ["empty_dataset"]
    if len([key for key, count in audit.length_buckets.items() if count > 0]) <= 1:
        warnings.append("single_length_bucket")
    top_fraction = audit.top_source_fraction()
    if top_fraction is not None and top_fraction >= 0.80 and audit.records >= 100:
        warnings.append("dominant_source_group_ge_80pct")
    unknown_clan_fraction = _mean(audit.clans.get("unknown", 0), audit.records)
    unknown_family_fraction = _mean(audit.families.get("unknown", 0), audit.records)
    pseudo_clan_count = sum(count for key, count in audit.clans.items() if _is_pseudo_clan(key))
    pseudo_clan_fraction = _mean(pseudo_clan_count, audit.records)
    if unknown_clan_fraction is not None and unknown_clan_fraction >= 0.80:
        warnings.append("mostly_missing_clan_metadata")
    if unknown_family_fraction is not None and unknown_family_fraction >= 0.80:
        warnings.append("mostly_missing_family_metadata")
    if pseudo_clan_fraction is not None and pseudo_clan_fraction >= 0.20:
        warnings.append("fallback_pseudo_clan_fraction_ge_20pct")
    long_fraction = _mean(audit.long_pair_total, audit.pair_total)
    if long_fraction is not None and long_fraction < 0.05 and audit.pair_total > 0:
        warnings.append("low_long_range_pair_fraction")
    if audit.windowed_records > 0 and len(audit.parent_ids) == 0:
        warnings.append("window_records_without_parent_ids")
    return warnings


def parse_label_path(raw: str) -> tuple[str, Path]:
    """Parse a ``label=path`` CLI argument.

    Formula: split on the first equals sign and require both sides to be
    non-empty.  Complexity: O(len(raw)).
    """

    if "=" not in raw:
        raise argparse.ArgumentTypeError("input must use label=path")
    label, path = raw.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("input must use non-empty label=path")
    return label, Path(path)


def iter_jsonl(path: Path, *, max_records: Optional[int] = None) -> Iterable[Mapping[str, object]]:
    """Yield mapping records from a JSONL file.

    Formula: parse non-empty lines and yield only JSON objects; stop after
    ``max_records`` when provided.  Complexity: O(file bytes).
    """

    emitted = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record: {exc}") from exc
            if isinstance(payload, Mapping):
                yield payload
                emitted += 1
                if max_records is not None and emitted >= max_records:
                    return


def collect_inputs(
    *,
    full_run_root: Path,
    input_jsonl: Sequence[tuple[str, Path]],
    include_defaults: bool,
) -> list[tuple[str, Path]]:
    """Collect existing audit inputs.

    Formula: include explicit ``label=path`` arguments plus default cache/split
    paths that exist under ``full_run_root``.  Complexity: O(K).
    """

    seen: set[Path] = set()
    items: list[tuple[str, Path]] = []

    def add(label: str, path: Path) -> None:
        """Add an existing path once.

        Complexity: O(1) expected for the resolved-path set lookup.
        """

        resolved = path if path.is_absolute() else full_run_root / path
        if not resolved.exists():
            return
        key = resolved.resolve()
        if key in seen:
            return
        seen.add(key)
        items.append((label, resolved))

    if include_defaults:
        for label, path in DEFAULT_CACHE_FILES.items():
            add(label, path)
        for label, path in DEFAULT_SPLIT_FILES.items():
            add(label, path)
    for label, path in input_jsonl:
        add(label, path)
    return items


def run_audit(
    inputs: Sequence[tuple[str, Path]],
    *,
    long_range_min_distance: int,
    max_records_per_input: Optional[int] = None,
) -> dict[str, object]:
    """Run the data diversity audit.

    Formula: stream each input JSONL into a ``DatasetAudit`` accumulator and
    combine top-level totals/warnings.  Complexity: O(total JSONL bytes).
    """

    datasets: list[dict[str, object]] = []
    accumulators: list[DatasetAudit] = []
    total_records = 0
    total_pairs = 0
    total_long_pairs = 0
    for label, path in inputs:
        audit = DatasetAudit(label=label, path=path, long_range_min_distance=long_range_min_distance)
        for record in iter_jsonl(path, max_records=max_records_per_input):
            audit.add(record)
        summary = audit.summary()
        accumulators.append(audit)
        datasets.append(summary)
        total_records += int(summary["record_count"])
        complexity = summary["structure_complexity"]
        if isinstance(complexity, Mapping):
            mean_pairs = complexity.get("mean_pair_count")
            long_pairs = complexity.get("long_range_pair_count")
            if isinstance(mean_pairs, (int, float)) and math.isfinite(float(mean_pairs)):
                total_pairs += int(round(float(mean_pairs) * int(summary["record_count"])))
            if isinstance(long_pairs, int):
                total_long_pairs += long_pairs

    all_warnings = {
        str(dataset["label"]): dataset.get("warnings", [])
        for dataset in datasets
        if isinstance(dataset.get("warnings"), list) and dataset.get("warnings")
    }
    leakage = []
    for left_index, left in enumerate(accumulators):
        for right in accumulators[left_index + 1 :]:
            sequence_overlap = left.sequence_hashes & right.sequence_hashes
            parent_overlap = left.parent_ids & right.parent_ids
            leakage.append(
                {
                    "left": left.label,
                    "right": right.label,
                    "sequence_overlap_count": len(sequence_overlap),
                    "parent_window_overlap_count": len(parent_overlap),
                    "parent_window_overlap_examples": sorted(parent_overlap)[:20],
                }
            )
    return {
        "summary": {
            "dataset_count": len(datasets),
            "record_count": total_records,
            "long_range_min_distance": long_range_min_distance,
            "overall_long_range_pair_fraction": _mean(total_long_pairs, total_pairs),
            "dataset_labels": [dataset["label"] for dataset in datasets],
            "warning_dataset_count": len(all_warnings),
        },
        "datasets": datasets,
        "warnings_by_dataset": all_warnings,
        "cross_dataset_leakage": leakage,
    }


def build_source_family_length_manifest(audit: Mapping[str, object]) -> dict[str, object]:
    """Build a compact manifest for source/family/length curriculum planning.

    Formula: project each dataset audit into source, family/clan, length, and
    windowing fields needed by balanced sampling/curriculum decisions.
    Complexity: O(D), where D is the number of datasets.
    """

    rows: list[dict[str, object]] = []
    datasets = audit.get("datasets", [])
    if isinstance(datasets, list):
        for dataset in datasets:
            if not isinstance(dataset, Mapping):
                continue
            rows.append(
                {
                    "label": dataset.get("label"),
                    "record_count": dataset.get("record_count"),
                    "length": dataset.get("length"),
                    "source_mix": dataset.get("source_mix"),
                    "family_clan": dataset.get("family_clan"),
                    "windowing": dataset.get("windowing"),
                    "structure_complexity": dataset.get("structure_complexity"),
                    "warnings": dataset.get("warnings", []),
                }
            )
    return {"rows": rows, "summary": audit.get("summary", {})}


def write_markdown(audit: Mapping[str, object], path: Path) -> None:
    """Write a Markdown diversity audit report.

    Complexity: O(D + W), where D is dataset count and W is warning count.
    """

    summary = audit.get("summary", {})
    lines = [
        "# ReactFlow Data Diversity Audit",
        "",
        f"- dataset_count: `{summary.get('dataset_count')}`",
        f"- record_count: `{summary.get('record_count')}`",
        f"- long_range_min_distance: `{summary.get('long_range_min_distance')}`",
        f"- overall_long_range_pair_fraction: `{summary.get('overall_long_range_pair_fraction')}`",
        f"- warning_dataset_count: `{summary.get('warning_dataset_count')}`",
        "",
        "| Dataset | Records | Length mean/max | Top source fraction | Unique clans/families | Pseudo-clan fraction | Long-range pair fraction | Windowed fraction | Warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    datasets = audit.get("datasets", [])
    if isinstance(datasets, list):
        for dataset in datasets:
            if not isinstance(dataset, Mapping):
                continue
            length = dataset.get("length", {})
            source_mix = dataset.get("source_mix", {})
            family_clan = dataset.get("family_clan", {})
            complexity = dataset.get("structure_complexity", {})
            windowing = dataset.get("windowing", {})
            warnings = dataset.get("warnings", [])
            length_text = ""
            if isinstance(length, Mapping):
                length_text = f"{_fmt(length.get('mean'))}/{_fmt(length.get('max'))}"
            clan_text = ""
            if isinstance(family_clan, Mapping):
                clan_text = f"{family_clan.get('unique_clans')}/{family_clan.get('unique_families')}"
            lines.append(
                "| {label} | {records} | {length_text} | {top_source} | {clan_text} | {pseudo_clan} | {long_fraction} | {window_fraction} | {warnings} |".format(
                    label=dataset.get("label"),
                    records=dataset.get("record_count"),
                    length_text=length_text,
                    top_source=_fmt(source_mix.get("top_group_fraction") if isinstance(source_mix, Mapping) else None),
                    clan_text=clan_text,
                    pseudo_clan=_fmt(
                        family_clan.get("pseudo_clan_fraction") if isinstance(family_clan, Mapping) else None
                    ),
                    long_fraction=_fmt(
                        complexity.get("long_range_pair_fraction") if isinstance(complexity, Mapping) else None
                    ),
                    window_fraction=_fmt(windowing.get("windowed_fraction") if isinstance(windowing, Mapping) else None),
                    warnings=", ".join(str(item) for item in warnings) if isinstance(warnings, list) else "",
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `mostly_missing_*_metadata` means the JSONL can still be scored, but cannot support source/family-balanced sampling without additional metadata joins.",
            "- `single_length_bucket` is acceptable for a fixed public tier, but is a warning for train-time curriculum diversity.",
            "- Long-range pairs use `|i-j| >= long_range_min_distance`; RF-CF2 should improve this slice without reducing `mmseqs_component_holdout` F1/MCC.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    """Format scalar values for compact Markdown tables.

    Complexity: O(1).
    """

    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(total JSONL bytes).
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="ReactFlow project root")
    parser.add_argument(
        "--full-run-root",
        default=str(DEFAULT_FULL_RUN),
        help="full-run artifact root, relative to --project-root unless absolute",
    )
    parser.add_argument(
        "--input-jsonl",
        action="append",
        default=[],
        type=parse_label_path,
        help="additional input as label=path; relative paths resolve under --full-run-root",
    )
    parser.add_argument("--no-defaults", action="store_true", help="only audit explicitly provided --input-jsonl files")
    parser.add_argument("--long-range-min-distance", type=int, default=24)
    parser.add_argument("--max-records-per-input", type=int, default=0, help="debug/test limit; 0 means no limit")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--manifest-json", default="")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root)
    full_run_root = Path(args.full_run_root)
    if not full_run_root.is_absolute():
        full_run_root = project_root / full_run_root
    output_json = Path(args.output_json) if args.output_json else full_run_root / "data_diversity_audit.json"
    output_md = Path(args.output_md) if args.output_md else full_run_root / "data_diversity_audit.md"
    manifest_json = Path(args.manifest_json) if args.manifest_json else full_run_root / "source_family_length_manifest.json"
    if not output_json.is_absolute():
        output_json = project_root / output_json
    if not output_md.is_absolute():
        output_md = project_root / output_md
    if not manifest_json.is_absolute():
        manifest_json = project_root / manifest_json

    inputs = collect_inputs(
        full_run_root=full_run_root,
        input_jsonl=args.input_jsonl,
        include_defaults=not args.no_defaults,
    )
    if not inputs:
        raise SystemExit("no JSONL inputs found")

    audit = run_audit(
        inputs,
        long_range_min_distance=args.long_range_min_distance,
        max_records_per_input=args.max_records_per_input or None,
    )
    manifest = build_source_family_length_manifest(audit)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(audit, output_md)
    print(json.dumps(audit["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
