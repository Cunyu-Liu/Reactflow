#!/usr/bin/env python3
"""Audit ReactFlow cross-family generation metrics.

The main RNA secondary-structure generalization claim should be judged on
family-disjoint tiers, not only on ordinary in-distribution evaluation.  This
script reads a metric-row JSON table such as ``current_queue_status.json`` or a
final result JSON file, pairs ``in_clan`` and ``novel_clan`` rows by ``run_id``,
then reports the novel-family F1/MCC and the generalization gap.

Formula: for run ``r``, let ``F_in(r)`` be ``mean_f1`` on ``in_clan`` and
``F_novel(r)`` be ``mean_f1`` on ``novel_clan``.  The cross-family gap is
``gap(r) = F_in(r) - F_novel(r)`` and the retention is
``retention(r) = F_novel(r) / max(F_in(r), eps)``.  A row is claim-ready when
``F_novel(r) >= min_novel_mean_f1`` and ``gap(r) <= max_generalization_gap``.
Complexity: O(R), where R is the number of metric rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence


def row(status: str, item: str, path: Optional[Path] = None, detail: str = "") -> dict:
    """Return one normalized audit row.

    Formula: accept ``status`` iff it belongs to the finite set
    ``{pass, warn, fail}``; the row then carries one evidence item and a
    human-readable detail string.  Complexity: O(1).
    """

    if status not in {"pass", "warn", "fail"}:
        raise ValueError("status must be pass/warn/fail")
    return {"detail": detail, "item": item, "path": "" if path is None else str(path), "status": status}


def _finite_float(value: object) -> Optional[float]:
    """Return ``value`` as a finite float when possible.

    Formula: accept iff ``value`` is numeric, not bool, and
    ``isfinite(float(value))``.  Complexity: O(1).
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def load_metric_rows(path: Path) -> List[Mapping[str, object]]:
    """Load a JSON metric-row table.

    Formula: parse JSON list ``R`` and retain only mapping rows.  A non-list
    payload is invalid because queue/final summaries use one row per
    ``(run_id, tier)`` evidence item.  Complexity: O(file bytes + R).
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON row list")
    return [item for item in payload if isinstance(item, Mapping)]


def group_rows_by_run(rows: Iterable[Mapping[str, object]]) -> MutableMapping[str, dict]:
    """Group in-clan and novel-clan metric rows by run id.

    Formula: for each row ``x`` with ``run_id=r`` and tier
    ``t in {in_clan, novel_clan}``, store ``G[r][t]=x``.  Running rows without a
    tier are counted separately so active runs remain visible as pending
    evidence.  Complexity: O(R).
    """

    grouped: MutableMapping[str, dict] = {}
    for item in rows:
        run_id = str(item.get("run_id", ""))
        if not run_id:
            continue
        bucket = grouped.setdefault(run_id, {"pending": 0})
        tier = str(item.get("tier", ""))
        status = str(item.get("status", ""))
        if tier in {"in_clan", "novel_clan"} and status == "ok":
            bucket[tier] = item
        elif status != "ok":
            bucket["pending"] = int(bucket.get("pending", 0)) + 1
    return grouped


def summarize_run(
    run_id: str,
    bucket: Mapping[str, object],
    *,
    min_novel_mean_f1: float,
    max_generalization_gap: float,
    source_path: Path,
) -> dict:
    """Summarize one run's cross-family evidence.

    Formula: compute ``gap = F_in - F_novel`` and
    ``retention = F_novel / max(F_in, eps)`` when both tiers expose finite
    ``mean_f1``.  Missing completed evidence is a warning, invalid metric values
    are failures, and low-but-parseable metrics are warnings rather than hard
    failures because they are scientific improvement targets.  Complexity: O(1).
    """

    in_row = bucket.get("in_clan")
    novel_row = bucket.get("novel_clan")
    if not isinstance(in_row, Mapping) or not isinstance(novel_row, Mapping):
        pending = int(bucket.get("pending", 0)) if isinstance(bucket.get("pending", 0), int) else 0
        return row(
            "warn",
            f"run:{run_id}:cross_family_pending",
            source_path,
            f"missing in_clan or novel_clan metric row; pending_rows={pending}",
        )

    in_mean_f1 = _finite_float(in_row.get("mean_f1"))
    novel_mean_f1 = _finite_float(novel_row.get("mean_f1"))
    in_micro_f1 = _finite_float(in_row.get("micro_f1"))
    novel_micro_f1 = _finite_float(novel_row.get("micro_f1"))
    novel_mean_mcc = _finite_float(novel_row.get("mean_mcc"))
    novel_count = novel_row.get("count")
    if (
        in_mean_f1 is None
        or novel_mean_f1 is None
        or in_micro_f1 is None
        or novel_micro_f1 is None
        or novel_mean_mcc is None
        or not isinstance(novel_count, int)
        or isinstance(novel_count, bool)
        or novel_count <= 0
    ):
        return row("fail", f"run:{run_id}:cross_family_metrics", source_path, "invalid finite metric/count fields")

    gap = in_mean_f1 - novel_mean_f1
    retention = None if in_mean_f1 <= 0.0 else novel_mean_f1 / in_mean_f1
    claim_ready = novel_mean_f1 >= min_novel_mean_f1 and gap <= max_generalization_gap
    status = "pass" if claim_ready else "warn"
    retention_text = "undefined" if retention is None else f"{retention:.4f}"
    return row(
        status,
        f"run:{run_id}:cross_family_metrics",
        source_path,
        (
            f"in_mean_f1={in_mean_f1:.4f}; novel_mean_f1={novel_mean_f1:.4f}; "
            f"gap={gap:.4f}; retention={retention_text}; "
            f"in_micro_f1={in_micro_f1:.4f}; novel_micro_f1={novel_micro_f1:.4f}; "
            f"novel_mean_mcc={novel_mean_mcc:.4f}; novel_count={novel_count}; "
            f"min_novel_mean_f1={min_novel_mean_f1:.4f}; max_generalization_gap={max_generalization_gap:.4f}"
        ),
    )


def _detail_float(detail: str, key: str) -> Optional[float]:
    """Extract ``key=value`` numeric tokens from an audit detail string.

    Formula: split semicolon-delimited tokens and parse the first token whose
    left-hand side equals ``key``.  Complexity: O(len(detail)).
    """

    prefix = f"{key}="
    for token in detail.split(";"):
        stripped = token.strip()
        if stripped.startswith(prefix):
            try:
                return float(stripped[len(prefix) :])
            except ValueError:
                return None
    return None


def summarize(rows: Sequence[Mapping[str, str]]) -> dict:
    """Summarize cross-family audit rows.

    Formula: count statuses, then choose the maximum ``novel_mean_f1`` among
    rows with parseable metric details.  ``cross_family_claim_ready`` is true
    iff at least one row passes the configured claim gate.  Complexity: O(N).
    """

    counts = {"pass": 0, "warn": 0, "fail": 0}
    best_run_id: Optional[str] = None
    best_novel_mean_f1: Optional[float] = None
    best_gap: Optional[float] = None
    for item in rows:
        counts[str(item["status"])] += 1
        novel = _detail_float(str(item.get("detail", "")), "novel_mean_f1")
        if novel is None:
            continue
        if best_novel_mean_f1 is None or novel > best_novel_mean_f1:
            best_novel_mean_f1 = novel
            best_gap = _detail_float(str(item.get("detail", "")), "gap")
            best_run_id = str(item.get("item", "")).split(":")[1] if ":" in str(item.get("item", "")) else None
    return {
        "best_generalization_gap": best_gap,
        "best_novel_mean_f1": best_novel_mean_f1,
        "best_run_id": best_run_id,
        "counts": counts,
        "cross_family_claim_ready": counts["pass"] > 0,
        "cross_family_healthy": counts["fail"] == 0,
    }


def run_audit(
    results_json: Path,
    *,
    min_novel_mean_f1: float,
    max_generalization_gap: float,
) -> dict:
    """Run the cross-family metric audit.

    Formula: load metric rows, group by run, audit every group, and summarize
    claim-readiness.  Complexity: O(R), where R is the row count.
    """

    metric_rows = load_metric_rows(results_json)
    grouped = group_rows_by_run(metric_rows)
    rows = [
        summarize_run(
            run_id,
            bucket,
            min_novel_mean_f1=min_novel_mean_f1,
            max_generalization_gap=max_generalization_gap,
            source_path=results_json,
        )
        for run_id, bucket in sorted(grouped.items())
    ]
    if not rows:
        rows.append(row("fail", "cross_family:no_runs", results_json, "no run rows found"))
    return {
        "results_json": str(results_json),
        "rows": rows,
        "summary": summarize(rows),
        "thresholds": {
            "max_generalization_gap": max_generalization_gap,
            "min_novel_mean_f1": min_novel_mean_f1,
        },
    }


def write_markdown(result: Mapping[str, object], path: Path) -> None:
    """Write a Markdown cross-family audit report.

    Complexity: O(N) for N audit rows.
    """

    summary = result["summary"]
    thresholds = result["thresholds"]
    lines = [
        "# ReactFlow Cross-Family Metric Audit",
        "",
        f"- cross_family_healthy: `{summary['cross_family_healthy']}`",
        f"- cross_family_claim_ready: `{summary['cross_family_claim_ready']}`",
        f"- best_run_id: `{summary.get('best_run_id')}`",
        f"- best_novel_mean_f1: `{summary.get('best_novel_mean_f1')}`",
        f"- best_generalization_gap: `{summary.get('best_generalization_gap')}`",
        f"- min_novel_mean_f1: `{thresholds['min_novel_mean_f1']}`",
        f"- max_generalization_gap: `{thresholds['max_generalization_gap']}`",
        "",
        "| Status | Item | Path | Detail |",
        "|---|---|---|---|",
    ]
    for item in result["rows"]:
        detail = str(item["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['status']} | {item['item']} | {item['path']} | {detail} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(R), where R is the result row count.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--min-novel-mean-f1", type=float, default=0.15)
    parser.add_argument("--max-generalization-gap", type=float, default=0.10)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    result = run_audit(
        Path(args.results_json),
        min_novel_mean_f1=args.min_novel_mean_f1,
        max_generalization_gap=args.max_generalization_gap,
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, Path(args.output_md))
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if result["summary"]["cross_family_healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
