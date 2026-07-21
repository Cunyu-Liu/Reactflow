#!/usr/bin/env python3
"""Audit ReactFlow profile bottlenecks for active or completed runs.

The run monitor reports phase totals, but publication-scale experiments also
need a stable bottleneck ledger: which phase dominates, what fraction of
profiled time it consumes, and whether the expected optimization phase appears
after an engineering change.  This script reads each run's
``monitor_snapshot.json`` when available, falls back to ``profile.jsonl`` via
``reactflow.run_monitor.summarize_profile``, and writes JSON/Markdown audit
artifacts.

Formula: for phase ``p`` with total time ``T_p`` and total profiled time
``T = sum_q T_q``, the bottleneck share is ``rho_p = T_p / T``.  A phase is over
budget when ``rho_p`` exceeds its configured fraction threshold.  Complexity:
O(R * P) for R run directories and P phases per monitor summary; profile
fallback is O(E) in the number of profile events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from reactflow.run_monitor import summarize_profile


def row(status: str, item: str, path: Optional[Path] = None, detail: str = "") -> dict:
    """Return one normalized bottleneck audit row.

    Complexity: O(1).
    """

    if status not in {"pass", "warn", "fail"}:
        raise ValueError("status must be pass/warn/fail")
    return {"detail": detail, "item": item, "path": "" if path is None else str(path), "status": status}


def _read_json_mapping(path: Path) -> Optional[Mapping[str, object]]:
    """Read a JSON mapping, returning ``None`` for missing or invalid files.

    Complexity: O(file bytes).
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, Mapping) else None


def run_completed(run_dir: Path) -> bool:
    """Return whether ``run_dir`` has durable completed-run evidence.

    Formula: ``complete = 1[stdout is Map and checkpoint_size > 0 and tiers is
    Map]``.  Completed runs are retained in the bottleneck ledger for historical
    interpretation, but their known slow phases no longer indicate active
    runtime risk.  Complexity: O(stdout bytes + 1).
    """

    stdout = _read_json_mapping(run_dir / "stdout.json")
    checkpoint = run_dir / "training_checkpoint.json"
    return bool(stdout and isinstance(stdout.get("tiers"), Mapping) and checkpoint.exists() and checkpoint.stat().st_size > 0)


def monitor_summary_for_run(run_dir: Path, *, total_samples: Optional[int] = None) -> Optional[Mapping[str, object]]:
    """Return a monitor summary for ``run_dir``.

    ``monitor_snapshot.json`` is preferred because the unified refresh script has
    already parsed the streaming profile.  When absent, the function summarizes
    ``profile.jsonl`` directly so the audit remains useful for standalone tests
    and completed runs.

    Complexity: O(1) for an existing monitor snapshot, otherwise O(E) profile
    events.
    """

    snapshot = _read_json_mapping(run_dir / "monitor_snapshot.json")
    if snapshot is not None:
        return snapshot
    profile = run_dir / "profile.jsonl"
    if not profile.exists() or profile.stat().st_size == 0:
        return None
    return summarize_profile(profile, total_samples=total_samples, stderr_path=run_dir / "stderr.log")


def _phase_rows(summary: Mapping[str, object]) -> List[Mapping[str, object]]:
    """Return monitor phase rows sorted by total time.

    Complexity: O(P).
    """

    rows = summary.get("phases_by_total_seconds")
    if not isinstance(rows, list):
        return []
    return [item for item in rows if isinstance(item, Mapping)]


def _phase_total(summary: Mapping[str, object], phase: str) -> Optional[float]:
    """Return total seconds for one phase if present.

    Complexity: O(P).
    """

    phases = summary.get("phases")
    if isinstance(phases, Mapping):
        entry = phases.get(phase)
        if isinstance(entry, Mapping) and isinstance(entry.get("total_seconds"), (int, float)):
            return float(entry["total_seconds"])
    for item in _phase_rows(summary):
        if str(item.get("phase", "")) == phase and isinstance(item.get("total_seconds"), (int, float)):
            return float(item["total_seconds"])
    return None


def phase_fraction(summary: Mapping[str, object], phase: str) -> Optional[float]:
    """Return ``T_phase / T_total`` for a monitor summary.

    Complexity: O(P).
    """

    total = summary.get("total_profiled_seconds")
    phase_total = _phase_total(summary, phase)
    if not isinstance(total, (int, float)) or float(total) <= 0.0 or phase_total is None:
        return None
    return phase_total / float(total)


def audit_run(
    run_dir: Path,
    *,
    total_samples: Optional[int],
    max_slowest_fraction: float,
    target_phase: str,
    max_target_fraction: float,
    expected_optimization_phase: str,
    min_events: int,
) -> List[dict]:
    """Audit bottleneck rows for one run directory.

    Formula: ``rho_slowest = T_slowest / T`` and
    ``rho_target = T_target / T``.  Missing/empty profiles fail; over-budget
    fractions warn for active long runs.  Completed runs keep the same measured
    fractions as pass rows tagged ``completed_history`` so historical
    bottlenecks remain visible without inflating active warning counts.

    Complexity: O(P) after monitor summary loading.
    """

    run_id = run_dir.name
    summary = monitor_summary_for_run(run_dir, total_samples=total_samples)
    completed = run_completed(run_dir)
    rows: List[dict] = []
    if summary is None:
        return [row("fail", f"run:{run_id}:monitor_summary", run_dir, "missing monitor_snapshot.json and profile.jsonl")]

    events = summary.get("events")
    if isinstance(events, int) and events >= min_events:
        rows.append(row("pass", f"run:{run_id}:profile_events", run_dir, f"events={events}; min={min_events}"))
    else:
        rows.append(row("fail", f"run:{run_id}:profile_events", run_dir, f"events={events!r}; min={min_events}"))

    total = summary.get("total_profiled_seconds")
    if isinstance(total, (int, float)) and float(total) > 0.0:
        rows.append(row("pass", f"run:{run_id}:profile_total_seconds", run_dir, f"total={float(total):.3f}"))
    else:
        rows.append(row("fail", f"run:{run_id}:profile_total_seconds", run_dir, f"invalid total={total!r}"))
        return rows

    slowest = summary.get("slowest_phase")
    if isinstance(slowest, Mapping):
        slowest_phase = str(slowest.get("phase", ""))
        slowest_fraction = phase_fraction(summary, slowest_phase)
        status = "pass" if slowest_fraction is not None and (completed or slowest_fraction <= max_slowest_fraction) else "warn"
        prefix = "completed_history; " if completed else ""
        rows.append(
            row(
                status,
                f"run:{run_id}:slowest_phase_fraction",
                run_dir,
                f"{prefix}phase={slowest_phase}; fraction={slowest_fraction:.4f}; max={max_slowest_fraction:.4f}" if slowest_fraction is not None else f"{prefix}fraction missing",
            )
        )
    else:
        rows.append(row("fail", f"run:{run_id}:slowest_phase_fraction", run_dir, "slowest_phase missing"))

    target_fraction = phase_fraction(summary, target_phase)
    if target_fraction is None:
        rows.append(row("warn", f"run:{run_id}:target_phase_fraction", run_dir, f"phase={target_phase} missing"))
    else:
        status = "pass" if completed or target_fraction <= max_target_fraction else "warn"
        prefix = "completed_history; " if completed else ""
        rows.append(
            row(
                status,
                f"run:{run_id}:target_phase_fraction",
                run_dir,
                f"{prefix}phase={target_phase}; fraction={target_fraction:.4f}; max={max_target_fraction:.4f}",
            )
        )

    optimization_fraction = phase_fraction(summary, expected_optimization_phase)
    if optimization_fraction is None:
        status = "pass" if completed else "warn"
        prefix = "completed_history; " if completed else ""
        rows.append(
            row(
                status,
                f"run:{run_id}:optimization_phase_present",
                run_dir,
                f"{prefix}phase={expected_optimization_phase} missing; active run may predate optimization",
            )
        )
    else:
        rows.append(
            row(
                "pass",
                f"run:{run_id}:optimization_phase_present",
                run_dir,
                f"phase={expected_optimization_phase}; fraction={optimization_fraction:.4f}",
            )
        )
    return rows


def summarize(rows: Iterable[Mapping[str, str]], *, audited_run_count: int) -> dict:
    """Return audit counts and overall health.

    Warnings are retained for known bottlenecks, while only failures make the
    audit unhealthy.  Complexity: O(N).
    """

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in rows:
        counts[str(item["status"])] += 1
    return {"audited_run_count": audited_run_count, "bottleneck_healthy": counts["fail"] == 0, "counts": counts}


def run_audit(
    run_dirs: Sequence[Path],
    *,
    total_samples: Optional[int],
    max_slowest_fraction: float,
    target_phase: str,
    max_target_fraction: float,
    expected_optimization_phase: str,
    min_events: int,
) -> dict:
    """Run bottleneck audit for all supplied run directories.

    Complexity: O(R * P) when monitor snapshots exist.
    """

    rows: List[dict] = []
    for run_dir in run_dirs:
        rows.extend(
            audit_run(
                run_dir,
                total_samples=total_samples,
                max_slowest_fraction=max_slowest_fraction,
                target_phase=target_phase,
                max_target_fraction=max_target_fraction,
                expected_optimization_phase=expected_optimization_phase,
                min_events=min_events,
            )
        )
    summary = summarize(rows, audited_run_count=len(run_dirs))
    return {"audited_run_dirs": [str(path) for path in run_dirs], "rows": rows, "summary": summary}


def write_markdown(rows: Sequence[Mapping[str, str]], summary: Mapping[str, object], path: Path) -> None:
    """Write bottleneck audit rows as Markdown.

    Complexity: O(N).
    """

    lines = [
        "# ReactFlow Profile Bottleneck Audit",
        "",
        f"- bottleneck_healthy: `{summary['bottleneck_healthy']}`",
        f"- counts: `{summary['counts']}`",
        f"- audited_run_count: `{summary['audited_run_count']}`",
        "",
        "| Status | Item | Path | Detail |",
        "|---|---|---|---|",
    ]
    for item in rows:
        lines.append(f"| {item['status']} | {item['item']} | {item['path']} | {item['detail']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(R * P) with monitor snapshots.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--total-samples", type=int)
    parser.add_argument("--target-phase", default="path_sample_features")
    parser.add_argument("--expected-optimization-phase", default="frozen_batch_prefetch")
    parser.add_argument("--max-slowest-fraction", type=float, default=0.75)
    parser.add_argument("--max-target-fraction", type=float, default=0.50)
    parser.add_argument("--min-events", type=int, default=1)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args(argv)

    result = run_audit(
        [Path(path) for path in args.run_dir],
        total_samples=args.total_samples,
        max_slowest_fraction=args.max_slowest_fraction,
        target_phase=args.target_phase,
        max_target_fraction=args.max_target_fraction,
        expected_optimization_phase=args.expected_optimization_phase,
        min_events=args.min_events,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    print(text, end="")
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if args.output_md:
        write_markdown(result["rows"], result["summary"], Path(args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
