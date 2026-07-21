#!/usr/bin/env python3
"""Build the protocol-safe ReactFlow SOTA alignment table.

The table is intentionally conservative: every row carries one of exactly three
protocol labels, and cited-only numbers are never merged with local reruns.  The
input side accepts the row-list JSON emitted by ``summarize_ablation_results.py``
as well as ``{"rows": [...]}`` wrappers and direct ``eval_summary`` payloads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence


FIELDS = (
    "model",
    "protocol",
    "split",
    "seed_count",
    "mean_f1",
    "mean_mcc",
    "long_f1",
    "long_recall",
    "reactivity_corr",
    "calibration_ece",
    "runtime_s_per_sample",
    "artifact",
)
ALLOWED_PROTOCOLS = ("same_split_local", "local_closest_protocol", "cited_only")
MMSEQS_TIER_ALIASES = {
    "in_clan": "mmseqs_component_test",
    "novel_clan": "mmseqs_component_holdout",
}
MMSEQS_TIERS = {
    "in_clan",
    "cross_clan",
    "novel_clan",
    "mmseqs_component_test",
    "mmseqs_component_holdout",
}
PUBLIC_TIERS = {"archiveII", "PDB", "viral", "lncRNA", "human_mRNA"}

DEFAULT_CITED_ROWS = (
    {
        "model": "RNADiffFold cited",
        "protocol": "cited_only",
        "split": "ArchiveII cited protocol",
        "seed_count": "cited_only",
        "mean_f1": 0.880,
        "mean_mcc": None,
        "long_f1": None,
        "long_recall": None,
        "reactivity_corr": None,
        "calibration_ece": None,
        "runtime_s_per_sample": None,
        "artifact": "Briefings in Bioinformatics 2025, DOI 10.1093/bib/bbae618",
    },
    {
        "model": "eFold/RNAndria cited",
        "protocol": "cited_only",
        "split": "viral cited protocol",
        "seed_count": "cited_only",
        "mean_f1": 0.730,
        "mean_mcc": None,
        "long_f1": None,
        "long_recall": None,
        "reactivity_corr": None,
        "calibration_ece": None,
        "runtime_s_per_sample": None,
        "artifact": "Science Advances 2026, DOI 10.1126/sciadv.adz4967",
    },
    {
        "model": "eFold/RNAndria cited",
        "protocol": "cited_only",
        "split": "lncRNA cited protocol",
        "seed_count": "cited_only",
        "mean_f1": 0.440,
        "mean_mcc": None,
        "long_f1": None,
        "long_recall": None,
        "reactivity_corr": None,
        "calibration_ece": None,
        "runtime_s_per_sample": None,
        "artifact": "Science Advances 2026, DOI 10.1126/sciadv.adz4967",
    },
)


def _first_json_value(text: str) -> object:
    """Parse the first JSON object or array embedded in ``text``.

    Historical wrapper logs sometimes prepend status text before the JSON.  We
    therefore look for the first ``{`` or ``[`` and parse strictly from there.
    Complexity: O(B), where B is the text length.
    """

    starts = sorted(idx for idx in (text.find("{"), text.find("[")) if idx >= 0)
    if not starts:
        raise ValueError("no JSON object or array found")
    last_error: Optional[json.JSONDecodeError] = None
    for start in starts:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError("no valid JSON object or array found") from last_error


def load_json_rows(path: Path) -> List[dict]:
    """Load alignment candidate rows from one JSON artifact."""

    payload = _first_json_value(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    if isinstance(payload.get("rows"), list):
        return [dict(row) for row in payload["rows"] if isinstance(row, Mapping)]
    tiers = payload.get("tiers")
    if isinstance(tiers, Mapping):
        run_id = str(payload.get("run_id") or payload.get("model") or path.parent.name)
        rows: List[dict] = []
        for tier, metrics in tiers.items():
            if not isinstance(metrics, Mapping):
                continue
            row = dict(metrics)
            row.setdefault("run_id", run_id)
            row.setdefault("tier", tier)
            row.setdefault("artifact", str(path))
            rows.append(row)
        return rows
    if any(field in payload for field in ("mean_f1", "f1", "model", "protocol")):
        row = dict(payload)
        row.setdefault("artifact", str(path))
        return [row]
    return []


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text in {"-", "--", "pending", "None", "null"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _pick(row: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _normalize_seed_count(value: object, protocol: str) -> str:
    if value is None or value == "":
        return "cited_only" if protocol == "cited_only" else "single_seed"
    if isinstance(value, int):
        return "single_seed" if value == 1 else f"{value}_seeds"
    text = str(value).strip()
    lower = text.lower().replace(" ", "_").replace("-", "_")
    if lower in {"1", "seed1", "one_seed", "single", "single_seed"}:
        return "single_seed"
    if lower in {"cited", "cited_only", "na", "n/a"}:
        return "cited_only"
    if lower.endswith("_seeds") or lower.endswith("_seed"):
        return lower
    if lower.isdigit():
        number = int(lower)
        return "single_seed" if number == 1 else f"{number}_seeds"
    return lower


def _validate_protocol(value: object) -> str:
    protocol = str(value or "").strip()
    if protocol not in ALLOWED_PROTOCOLS:
        raise ValueError(
            f"invalid protocol {protocol!r}; expected one of {', '.join(ALLOWED_PROTOCOLS)}"
        )
    return protocol


def infer_protocol(row: Mapping[str, object]) -> str:
    explicit = _pick(row, "protocol", "comparison_protocol")
    if explicit is not None:
        return _validate_protocol(explicit)
    tier = str(row.get("tier") or "")
    run_id = str(row.get("run_id") or row.get("model") or "")
    if tier in MMSEQS_TIERS and "mmseqs" in run_id.lower():
        return "same_split_local"
    return "local_closest_protocol"


def infer_split(row: Mapping[str, object], protocol: str) -> str:
    explicit = _pick(row, "split", "split_name", "benchmark")
    if explicit is not None:
        split = str(explicit)
        for legacy, honest in MMSEQS_TIER_ALIASES.items():
            if split == legacy:
                return honest
            if split.endswith(f":{legacy}"):
                return split[: -len(legacy)] + honest
        return split
    tier = str(row.get("tier") or "").strip()
    display_tier = MMSEQS_TIER_ALIASES.get(tier, tier)
    run_id = str(row.get("run_id") or row.get("model") or "")
    if protocol == "cited_only":
        return tier or "cited_protocol"
    if tier in MMSEQS_TIERS and "mmseqs" in run_id.lower():
        return f"MMseqs:{display_tier}"
    if tier in MMSEQS_TIERS and "rfam_current_exact" in run_id:
        return f"Rfam-current-exact:{tier}"
    if tier in PUBLIC_TIERS:
        return f"eFold-RNAndria:{tier}"
    return tier or "unspecified"


def _long_bin_mapping(row: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("distance_bins", "distance_bin_metrics", "bins"):
        value = row.get(key)
        if not isinstance(value, Mapping):
            continue
        direct = value.get("long")
        if isinstance(direct, Mapping):
            return direct
        tier = row.get("tier")
        tier_value = value.get(tier) if tier is not None else None
        if isinstance(tier_value, Mapping) and isinstance(tier_value.get("long"), Mapping):
            return tier_value["long"]
    return {}


def _runtime_s_per_sample(row: Mapping[str, object]) -> Optional[float]:
    explicit = _to_float(_pick(row, "runtime_s_per_sample", "seconds_per_sample"))
    if explicit is not None:
        return explicit
    samples_per_second = _to_float(row.get("samples_per_second"))
    if samples_per_second and samples_per_second > 0:
        return 1.0 / samples_per_second
    return None


def normalize_candidate_row(row: Mapping[str, object], *, source_path: Optional[Path]) -> Optional[dict]:
    """Normalize one input row to the fixed SOTA table schema."""

    status = str(row.get("status") or "ok")
    if status != "ok" and _to_float(_pick(row, "mean_f1", "f1")) is None:
        return None
    protocol = infer_protocol(row)
    long_bin = _long_bin_mapping(row)
    model = str(_pick(row, "model", "run_id", "name") or "unknown_model")
    artifact = str(_pick(row, "artifact", "path") or source_path or "")
    long_f1 = _to_float(_pick(row, "long_f1", "long_mean_f1", "mean_long_f1"))
    if long_f1 is None:
        long_f1 = _to_float(_pick(long_bin, "mean_f1", "f1", "micro_f1"))
    long_recall = _to_float(_pick(row, "long_recall", "long_mean_recall", "mean_long_recall"))
    if long_recall is None:
        long_recall = _to_float(_pick(long_bin, "recall", "mean_recall", "micro_recall"))
    normalized = {
        "model": model,
        "protocol": protocol,
        "split": infer_split(row, protocol),
        "seed_count": _normalize_seed_count(_pick(row, "seed_count", "seeds", "n_seeds"), protocol),
        "mean_f1": _to_float(_pick(row, "mean_f1", "f1", "novel_clan_mean_f1")),
        "mean_mcc": _to_float(_pick(row, "mean_mcc", "mcc", "novel_clan_mean_mcc")),
        "long_f1": long_f1,
        "long_recall": long_recall,
        "reactivity_corr": _to_float(
            _pick(row, "reactivity_corr", "reactivity_pearson", "pearson", "spearman")
        ),
        "calibration_ece": _to_float(_pick(row, "calibration_ece", "ece")),
        "runtime_s_per_sample": _runtime_s_per_sample(row),
        "artifact": artifact,
    }
    if normalized["mean_f1"] is None and protocol != "cited_only":
        return None
    return normalized


def build_alignment_rows(
    input_paths: Sequence[Path],
    *,
    include_default_cited: bool = True,
) -> List[dict]:
    """Build sorted, de-duplicated rows for the SOTA alignment table."""

    rows: List[dict] = [dict(row) for row in DEFAULT_CITED_ROWS] if include_default_cited else []
    for path in input_paths:
        for candidate in load_json_rows(path):
            normalized = normalize_candidate_row(candidate, source_path=path)
            if normalized is not None:
                rows.append(normalized)

    seen: set[tuple[str, str, str, str]] = set()
    deduped: List[dict] = []
    for row in rows:
        protocol = _validate_protocol(row["protocol"])
        fixed = {field: row.get(field) for field in FIELDS}
        fixed["protocol"] = protocol
        fixed["seed_count"] = _normalize_seed_count(fixed.get("seed_count"), protocol)
        key = (
            str(fixed["model"]),
            str(fixed["protocol"]),
            str(fixed["split"]),
            str(fixed["artifact"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fixed)

    order = {protocol: index for index, protocol in enumerate(ALLOWED_PROTOCOLS)}
    return sorted(
        deduped,
        key=lambda row: (
            order.get(str(row["protocol"]), 99),
            str(row["model"]),
            str(row["split"]),
            str(row["artifact"]),
        ),
    )


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    lines = [
        "# ReactFlow SOTA Alignment Table",
        "",
        "Generated by `scripts/build_sota_alignment_table.py`. Protocol values are limited to "
        "`same_split_local`, `local_closest_protocol`, and `cited_only`; cited-only rows are "
        "bookkeeping references and must not be used as same-split claims.",
        "",
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join("---" for _ in FIELDS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(field)) for field in FIELDS) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    payload = {
        "schema_version": 1,
        "fields": list(FIELDS),
        "allowed_protocols": list(ALLOWED_PROTOCOLS),
        "rows": list(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def discover_default_inputs(project_root: Path) -> List[Path]:
    """Discover likely result JSONs under the latest full-run artifact directory."""

    full_runs = project_root / "artifacts" / "full_runs"
    run_roots = [path for path in full_runs.glob("*") if path.is_dir()]
    if not run_roots:
        return []
    latest = max(run_roots, key=lambda path: path.stat().st_mtime)
    names = (
        "current_queue_status.json",
        "mmseqs_final_results.json",
        "cross_family_balanced_results.json",
        "cross_family_contact_sweep_results.json",
        "cross_family_long_range_results.json",
        "cross_family_capacity_results.json",
        "warm_rfam_current_exact_results.json",
        "contact_rfam_current_exact_results.json",
    )
    paths = [latest / name for name in names if (latest / name).exists()]
    paths.extend(sorted(latest.glob("baseline_*results*.json")))
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="ReactFlow project root")
    parser.add_argument("--input", action="append", default=[], help="result JSON path; may repeat")
    parser.add_argument("--output-md", default="docs/sota_alignment_table.md")
    parser.add_argument("--output-json", default="docs/sota_alignment_table.json")
    parser.add_argument("--no-default-cited", action="store_true")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    inputs = [Path(path) if Path(path).is_absolute() else project_root / path for path in args.input]
    if not inputs:
        inputs = discover_default_inputs(project_root)
    rows = build_alignment_rows(
        inputs,
        include_default_cited=not args.no_default_cited,
    )
    write_markdown(rows, project_root / args.output_md)
    write_json(rows, project_root / args.output_json)
    print(f"wrote {len(rows)} rows to {args.output_md} and {args.output_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
