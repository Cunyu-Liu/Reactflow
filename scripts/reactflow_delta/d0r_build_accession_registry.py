#!/usr/bin/env python3
"""D0-R: build paper-accession registry from RMDB ``_entries`` YAML front matter.

Replaces the D0 filename-keyword recall with a principled paper-accession -> asset
mapping. Each of the 1024 RMDB entry markdown files has YAML front matter with
rmdb_id, name, category, citation (authors/title/journal/year/doi/pubmed),
annotation (modifier/temperature/chemical), rdat URL, and construct_count.

Output: ``data_registry/d0r_accession_registry.jsonl`` (one JSON object per line).

No construct/pair/tier claim is made by this script. Raw metadata is preserved
exactly; missing fields are recorded as null, never imputed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "reactflow-delta-d0r-accession-registry-v1"


def parse_yaml_front_matter(text: str) -> dict[str, Any]:
    """Parse the minimal YAML front matter used by RMDB ``_entries`` files.

    The front matter is delimited by ``---`` lines and uses simple key: value
    pairs, with nested dicts for ``annotation`` and ``citation``, and ``|`` block
    scalars for ``comments``. We parse only the fields needed for accession
    mapping; unknown fields are ignored.
    """

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, Any] = {}
    i = 1
    n = len(lines)
    while i < n:
        raw = lines[i]
        if raw.strip() == "---":
            break
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        # top-level key: value
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", raw)
        if not match:
            i += 1
            continue
        key, value = match.group(1), match.group(2).rstrip()
        if value == "":
            # could be a nested dict or a block scalar; peek ahead
            if i + 1 < n and re.match(r"^\s{2,}([A-Za-z_][A-Za-z0-9_]*)\s*:", lines[i + 1]):
                nested, i = _parse_nested_dict(lines, i + 1, n)
                result[key] = nested
            elif i + 1 < n and lines[i + 1].strip().startswith("|"):
                block, i = _parse_block_scalar(lines, i + 2, n)
                result[key] = block
            else:
                result[key] = ""
            continue
        if value.startswith("|"):
            block, i = _parse_block_scalar(lines, i + 2, n)
            result[key] = block
            continue
        result[key] = _strip_yaml_quotes(value)
        i += 1
    return result


def _parse_nested_dict(lines: list[str], start: int, n: int) -> tuple[dict[str, list[str]], int]:
    """Parse a nested YAML dict (annotation/citation) with list values."""

    result: dict[str, list[str]] = {}
    i = start
    while i < n:
        raw = lines[i]
        if not raw.startswith("  "):
            break
        match = re.match(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", raw)
        if not match:
            i += 1
            continue
        key, value = match.group(1), match.group(2).rstrip()
        if value.startswith("["):
            items = _parse_yaml_list(value)
            result[key] = items
        else:
            result[key] = [_strip_yaml_quotes(value)] if value else []
        i += 1
    return result, i


def _parse_block_scalar(lines: list[str], start: int, n: int) -> tuple[str, int]:
    """Parse a YAML ``|`` block scalar (indented continuation lines)."""

    parts: list[str] = []
    i = start
    while i < n:
        raw = lines[i]
        if raw.strip() == "" or raw.startswith("  ") or raw.startswith("\t"):
            parts.append(raw.strip())
            i += 1
        else:
            break
    return "\n".join(parts), i


def _parse_yaml_list(value: str) -> list[str]:
    """Parse a YAML inline list like ``["1M7", "DMS"]``."""

    inner = value.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    items: list[str] = []
    for token in inner.split(","):
        token = token.strip()
        if token:
            items.append(_strip_yaml_quotes(token))
    return items


def _strip_yaml_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return value[1:-1]
    return value


def normalize_entry(raw: dict[str, Any], filename: str) -> dict[str, Any]:
    """Normalize a parsed entry to the registry schema, preserving nulls."""

    citation = raw.get("citation") or {}
    annotation = raw.get("annotation") or {}
    sequence = raw.get("sequence") or ""
    return {
        "schema_version": SCHEMA_VERSION,
        "rmdb_id": raw.get("rmdb_id") or "",
        "source_file": filename,
        "name": raw.get("name") or "",
        "category": raw.get("category") or "",
        "permalink": raw.get("permalink") or "",
        "date": raw.get("date") or "",
        "construct_count": raw.get("construct_count"),
        "data_points": raw.get("data_points"),
        "owner": raw.get("owner") or "",
        "description": raw.get("description") or "",
        "comments": raw.get("comments") or "",
        "sequence_masked": bool(sequence) and set(sequence).issubset({"X", "."}),
        "sequence_length": len(sequence) if sequence else 0,
        "offset": raw.get("offset"),
        "citation": {
            "authors": citation.get("authors", [""])[0] if isinstance(citation.get("authors"), list) else (citation.get("authors") or ""),
            "title": citation.get("title", [""])[0] if isinstance(citation.get("title"), list) else (citation.get("title") or ""),
            "journal": citation.get("journal", [""])[0] if isinstance(citation.get("journal"), list) else (citation.get("journal") or ""),
            "year": citation.get("year", [""])[0] if isinstance(citation.get("year"), list) else (citation.get("year") or ""),
            "doi": citation.get("doi", [""])[0] if isinstance(citation.get("doi"), list) else (citation.get("doi") or ""),
            "pubmed": citation.get("pubmed", [""])[0] if isinstance(citation.get("pubmed"), list) else (citation.get("pubmed") or ""),
        },
        "annotation": {
            "modifier": annotation.get("modifier", []) if isinstance(annotation, dict) else [],
            "temperature": annotation.get("temperature", []) if isinstance(annotation, dict) else [],
            "chemical": annotation.get("chemical", []) if isinstance(annotation, dict) else [],
            "processing": annotation.get("processing", []) if isinstance(annotation, dict) else [],
        },
        "rdat_url": raw.get("rdat") or "",
        "thumbnail": raw.get("thumbnail") or "",
    }


def build_registry(entries_dir: Path, output_path: Path) -> dict[str, Any]:
    """Parse all ``_entries/*.md`` files and write the accession registry."""

    if not entries_dir.is_dir():
        raise FileNotFoundError(f"entries directory not found: {entries_dir}")

    md_files = sorted(entries_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"no .md files in {entries_dir}")

    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8")
            raw = parse_yaml_front_matter(text)
            if not raw.get("rmdb_id"):
                parse_errors.append({"file": md_file.name, "error": "missing rmdb_id in front matter"})
                continue
            records.append(normalize_entry(raw, md_file.name))
        except Exception as exc:  # noqa: BLE001 - record all parse failures
            parse_errors.append({"file": md_file.name, "error": str(exc)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    # Summary statistics for audit
    categories: dict[str, int] = {}
    pubmed_count = 0
    doi_count = 0
    rdat_url_count = 0
    masked_count = 0
    for record in records:
        cat = record["category"] or "UNKNOWN"
        categories[cat] = categories.get(cat, 0) + 1
        if record["citation"]["pubmed"]:
            pubmed_count += 1
        if record["citation"]["doi"]:
            doi_count += 1
        if record["rdat_url"]:
            rdat_url_count += 1
        if record["sequence_masked"]:
            masked_count += 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "D0-R",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "entries_dir": str(entries_dir.resolve()),
        "output_path": str(output_path.resolve()),
        "total_entries": len(records),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:20],
        "categories": dict(sorted(categories.items())),
        "entries_with_pubmed": pubmed_count,
        "entries_with_doi": doi_count,
        "entries_with_rdat_url": rdat_url_count,
        "entries_with_masked_sequence": masked_count,
        "scientific_boundary": (
            "Paper-accession registry only. Each record maps an RMDB entry to its "
            "citation (doi/pubmed) and RDAT asset URL. No construct, pair, tier, "
            "or model claim is made. Masked header sequences require per-profile "
            "RDAT parsing to recover actual construct sequences."
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries-dir", required=True, help="Path to RMDB _entries/ directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args(argv)

    summary = build_registry(Path(args.entries_dir), Path(args.output))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
