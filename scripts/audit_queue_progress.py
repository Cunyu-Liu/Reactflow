#!/usr/bin/env python3
"""Audit active ReactFlow queue progress over time.

Runtime health proves that files are fresh and watcher PIDs are alive.  This
script adds the missing time-series check: is an active run's reported progress
actually increasing across refreshes?

The audit reads ``current_queue_status.json``, optionally appends a compact
snapshot to a JSONL history file, and compares the newest snapshot with an older
snapshot inside a bounded time window.  Running jobs with no progress increase
and very low throughput become explicit audit failures; early histories with too
few snapshots become warnings rather than false blockers.

Formula: for a running run ``r`` with latest progress ``p_t`` and previous
progress ``p_s`` from the history window, progress is healthy when
``p_t - p_s >= min_progress_delta``.  If the run artifact already contains
``profile.summary.json`` evidence that ``epoch_total`` closed, a stalled
training progress fraction is downgraded to warning because the process may be
in unprofiled tier evaluation or final JSON materialization.  Throughput is
healthy when ``samples_per_second >= min_samples_per_second``.  Complexity:
O(H * R + S), where H is the number of bounded history snapshots, R is the queue
row count and S is the optional profile summary size.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Iterable, List, Mapping, Optional, Sequence


RUNNING_STATUSES = {"running_or_pending_json"}


def row(status: str, item: str, path: Optional[Path] = None, detail: str = "") -> dict:
    """Return one normalized progress audit row.

    Complexity: O(1).
    """

    if status not in {"pass", "warn", "fail"}:
        raise ValueError("status must be pass/warn/fail")
    return {"detail": detail, "item": item, "path": "" if path is None else str(path), "status": status}


def _read_json(path: Path) -> object:
    """Read any JSON payload from ``path``.

    Complexity: O(file bytes).
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_mapping(path: Path) -> Optional[Mapping[str, object]]:
    """Read ``path`` only when it contains a JSON object.

    Formula: parse byte string ``B`` into JSON value ``x`` and accept iff
    ``x in Map``.  Malformed, missing, empty, or non-object files return
    ``None`` so queue-progress auditing can remain conservative when optional
    run-finalization evidence is absent.  Complexity: O(file bytes).
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def profile_epoch_closed(run_dir: Path) -> bool:
    """Return whether a run's profiled training epoch has closed.

    Formula: parse ``profile.summary.json`` and accept iff either
    ``summary.phases.epoch_total.total_seconds > 0`` or any
    ``summary.phases_by_total_seconds`` row has ``phase='epoch_total'``.  In
    that state, the training progress fraction can legitimately stop changing
    while the same process evaluates tiers or materializes final JSON files.
    Complexity: O(summary bytes + P), where P is the number of summarized
    phases.
    """

    summary = _read_json_mapping(run_dir / "profile.summary.json")
    if summary is None:
        return False
    phases = summary.get("phases")
    if isinstance(phases, Mapping):
        epoch = phases.get("epoch_total")
        if (
            isinstance(epoch, Mapping)
            and isinstance(epoch.get("total_seconds"), (int, float))
            and float(epoch["total_seconds"]) > 0.0
        ):
            return True
    phase_rows = summary.get("phases_by_total_seconds")
    if isinstance(phase_rows, list):
        for item in phase_rows:
            if isinstance(item, Mapping) and str(item.get("phase", "")) == "epoch_total":
                return True
    return False


def _last_profile_event(path: Path) -> Optional[Mapping[str, object]]:
    """Return the last parseable profile JSON event.

    Complexity: O(tail bytes), bounded to the latest 20 KiB.
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - 20_000))
        text = handle.read().decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, Mapping):
            return obj
    return None


def profile_load_in_progress(run_dir: Path) -> bool:
    """Return whether the active profile is still in a pre-training load phase."""

    event = _last_profile_event(run_dir / "profile.jsonl")
    return isinstance(event, Mapping) and str(event.get("phase", "")).startswith("load_")


def row_run_dir(latest_row: Mapping[str, object]) -> Optional[Path]:
    """Return the artifact directory referenced by a queue row when available.

    Formula: accept ``artifact`` as a filesystem path ``a`` iff it is a non-empty
    string; otherwise no run directory evidence is available.  Complexity: O(1).
    """

    artifact = latest_row.get("artifact")
    if not isinstance(artifact, str) or not artifact:
        return None
    return Path(artifact)


def load_queue_rows(path: Path) -> List[Mapping[str, object]]:
    """Load queue rows from ``current_queue_status.json``.

    Complexity: O(queue JSON bytes).
    """

    obj = _read_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: List[Mapping[str, object]] = []
    for item in obj:
        if isinstance(item, Mapping):
            rows.append(item)
    return rows


def _float_or_none(value: object) -> Optional[float]:
    """Convert numeric values to float and reject all other values.

    Complexity: O(1).
    """

    return float(value) if isinstance(value, (int, float)) else None


def make_snapshot(rows: Sequence[Mapping[str, object]], *, observed_at: float) -> dict:
    """Create a compact queue-progress snapshot.

    The snapshot stores only fields needed for progress trend auditing rather
    than copying full queue rows.

    Complexity: O(R).
    """

    compact_rows = []
    for item in rows:
        compact_rows.append(
            {
                "progress_fraction": _float_or_none(item.get("progress_fraction")),
                "run_id": str(item.get("run_id", "")),
                "samples_per_second": _float_or_none(item.get("samples_per_second")),
                "status": str(item.get("status", "")),
            }
        )
    return {"observed_at": observed_at, "rows": compact_rows}


def append_snapshot(path: Path, snapshot: Mapping[str, object]) -> None:
    """Append one snapshot as JSONL.

    Complexity: O(snapshot bytes).
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n")


def load_history(path: Path, *, max_lines: int) -> List[Mapping[str, object]]:
    """Load up to ``max_lines`` latest parseable history snapshots.

    Complexity: O(history bytes), bounded by the history file retained on disk.
    """

    if not path.exists() or path.stat().st_size == 0:
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    snapshots: List[Mapping[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, Mapping):
            snapshots.append(obj)
    return snapshots


def _snapshot_rows_by_run(snapshot: Mapping[str, object]) -> dict:
    """Return ``run_id -> compact row`` for one snapshot.

    Complexity: O(R).
    """

    rows = snapshot.get("rows")
    result = {}
    if not isinstance(rows, list):
        return result
    for item in rows:
        if isinstance(item, Mapping):
            run_id = str(item.get("run_id", ""))
            if run_id:
                result[run_id] = item
    return result


def _older_snapshot_for_run(
    history: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    latest_time: float,
    window_seconds: float,
) -> Optional[Mapping[str, object]]:
    """Return the oldest usable snapshot for ``run_id`` inside the window.

    Complexity: O(H * R).
    """

    lower_bound = latest_time - window_seconds
    candidates: List[Mapping[str, object]] = []
    for snapshot in history:
        observed_at = _float_or_none(snapshot.get("observed_at"))
        if observed_at is None or observed_at >= latest_time or observed_at < lower_bound:
            continue
        row_map = _snapshot_rows_by_run(snapshot)
        if run_id in row_map:
            candidates.append(snapshot)
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item.get("observed_at", latest_time)))


def audit_run_progress(
    latest_row: Mapping[str, object],
    *,
    latest_time: float,
    history: Sequence[Mapping[str, object]],
    history_path: Path,
    window_seconds: float,
    min_progress_delta: float,
    min_samples_per_second: float,
) -> List[dict]:
    """Audit progress and throughput for one queue row.

    Complexity: O(H * R).
    """

    run_id = str(latest_row.get("run_id", ""))
    status = str(latest_row.get("status", ""))
    progress = _float_or_none(latest_row.get("progress_fraction"))
    samples_per_second = _float_or_none(latest_row.get("samples_per_second"))
    run_dir = row_run_dir(latest_row)
    post_training_eval = bool(run_dir is not None and profile_epoch_closed(run_dir))
    loading = bool(run_dir is not None and (not post_training_eval) and profile_load_in_progress(run_dir))
    rows: List[dict] = []

    if status not in RUNNING_STATUSES:
        rows.append(row("pass", f"run:{run_id}:not_running", history_path, f"status={status!r}"))
        return rows

    if progress is None:
        progress_status = "warn" if (post_training_eval or loading) else "fail"
        if post_training_eval:
            prefix = "post_training_eval; "
        elif loading:
            prefix = "loading; "
        else:
            prefix = ""
        rows.append(row(progress_status, f"run:{run_id}:progress_value", history_path, f"{prefix}invalid progress={progress!r}"))
    elif not 0.0 <= progress <= 1.0:
        rows.append(row("fail", f"run:{run_id}:progress_value", history_path, f"invalid progress={progress!r}"))
    else:
        rows.append(row("pass", f"run:{run_id}:progress_value", history_path, f"progress={100.0 * progress:.2f}%"))

    if samples_per_second is None:
        rows.append(row("warn", f"run:{run_id}:throughput", history_path, "samples_per_second missing"))
    else:
        throughput_status = "pass" if samples_per_second >= min_samples_per_second else "fail"
        rows.append(
            row(
                throughput_status,
                f"run:{run_id}:throughput",
                history_path,
                f"samples_per_second={samples_per_second:.4f}; min={min_samples_per_second:.4f}",
            )
        )

    older = _older_snapshot_for_run(history, run_id=run_id, latest_time=latest_time, window_seconds=window_seconds)
    if older is None:
        rows.append(
            row(
                "warn",
                f"run:{run_id}:progress_window",
                history_path,
                f"insufficient history inside {window_seconds:.0f}s window",
            )
        )
        return rows
    older_row = _snapshot_rows_by_run(older).get(run_id)
    older_progress = _float_or_none(older_row.get("progress_fraction") if isinstance(older_row, Mapping) else None)
    older_time = _float_or_none(older.get("observed_at"))
    if progress is None or older_progress is None or older_time is None:
        rows.append(row("warn", f"run:{run_id}:progress_window", history_path, "history progress missing"))
        return rows
    delta = progress - older_progress
    elapsed = max(0.0, latest_time - older_time)
    if delta >= min_progress_delta:
        progress_status = "pass"
        prefix = ""
    elif post_training_eval:
        progress_status = "warn"
        prefix = "post_training_eval; "
    else:
        progress_status = "fail"
        prefix = ""
    rows.append(
        row(
            progress_status,
            f"run:{run_id}:progress_window",
            history_path,
            f"{prefix}delta={delta:.6f}; elapsed_seconds={elapsed:.1f}; min_delta={min_progress_delta:.6f}",
        )
    )
    return rows


def trend_for_run(
    latest_row: Mapping[str, object],
    *,
    latest_time: float,
    history: Sequence[Mapping[str, object]],
    window_seconds: float,
) -> Optional[dict]:
    """Estimate progress rate and remaining wall-clock time for one run.

    Formula: ``rate = (p_t - p_s) / (t - s)`` and
    ``eta = (1 - p_t) / rate`` when ``rate > 0``.  Complexity: O(H * R).
    """

    run_id = str(latest_row.get("run_id", ""))
    status = str(latest_row.get("status", ""))
    if status not in RUNNING_STATUSES:
        return None
    progress = _float_or_none(latest_row.get("progress_fraction"))
    if progress is None or not 0.0 <= progress <= 1.0:
        return None
    older = _older_snapshot_for_run(history, run_id=run_id, latest_time=latest_time, window_seconds=window_seconds)
    if older is None:
        return None
    older_row = _snapshot_rows_by_run(older).get(run_id)
    older_progress = _float_or_none(older_row.get("progress_fraction") if isinstance(older_row, Mapping) else None)
    older_time = _float_or_none(older.get("observed_at"))
    if older_progress is None or older_time is None:
        return None
    elapsed = latest_time - older_time
    if elapsed <= 0.0:
        return None
    delta = progress - older_progress
    rate = delta / elapsed
    eta: Optional[float] = 0.0 if progress >= 1.0 else None
    if rate > 0.0 and progress < 1.0:
        eta = (1.0 - progress) / rate
    samples_per_second = _float_or_none(latest_row.get("samples_per_second"))
    return {
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": eta if eta is None or math.isfinite(eta) else None,
        "progress_delta": delta,
        "progress_fraction": progress,
        "progress_rate_per_second": rate,
        "run_id": run_id,
        "samples_per_second": samples_per_second,
        "status": status,
    }


def summarize(rows: Iterable[Mapping[str, str]], trends: Sequence[Mapping[str, object]]) -> dict:
    """Return progress audit counts and health.

    Complexity: O(N).
    """

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in rows:
        counts[str(item["status"])] += 1
    eta_values = [
        float(item["estimated_remaining_seconds"])
        for item in trends
        if isinstance(item.get("estimated_remaining_seconds"), (int, float))
    ]
    return {
        "counts": counts,
        "min_estimated_remaining_seconds": min(eta_values) if eta_values else None,
        "progress_healthy": counts["fail"] == 0,
        "trend_count": len(trends),
    }


def run_audit(
    queue_json: Path,
    history_jsonl: Path,
    *,
    append_current: bool,
    observed_at: Optional[float],
    max_history_lines: int,
    window_seconds: float,
    min_progress_delta: float,
    min_samples_per_second: float,
) -> dict:
    """Run queue progress auditing and optional history append.

    Complexity: O(H * R).
    """

    now = time.time() if observed_at is None else observed_at
    rows: List[dict] = []
    queue_rows = load_queue_rows(queue_json)
    snapshot = make_snapshot(queue_rows, observed_at=now)
    if append_current:
        append_snapshot(history_jsonl, snapshot)
        rows.append(row("pass", "history_append", history_jsonl, f"rows={len(queue_rows)}"))
    history = load_history(history_jsonl, max_lines=max_history_lines)
    if not history:
        rows.append(row("warn", "history_exists", history_jsonl, "missing or empty"))
    else:
        rows.append(row("pass", "history_exists", history_jsonl, f"snapshots={len(history)}"))
    trends: List[dict] = []
    for item in queue_rows:
        rows.extend(
            audit_run_progress(
                item,
                latest_time=now,
                history=history,
                history_path=history_jsonl,
                window_seconds=window_seconds,
                min_progress_delta=min_progress_delta,
                min_samples_per_second=min_samples_per_second,
            )
        )
        trend = trend_for_run(item, latest_time=now, history=history, window_seconds=window_seconds)
        if trend is not None:
            trends.append(trend)
    summary = summarize(rows, trends)
    return {
        "history_jsonl": str(history_jsonl),
        "rows": rows,
        "snapshot": snapshot,
        "summary": summary,
        "trends": trends,
    }


def write_markdown(result: Mapping[str, object], path: Path) -> None:
    """Write queue progress audit rows as Markdown.

    Complexity: O(N).
    """

    summary = result["summary"]
    rows = result["rows"]
    lines = [
        "# ReactFlow Queue Progress Audit",
        "",
        f"- progress_healthy: `{summary['progress_healthy']}`",
        f"- counts: `{summary['counts']}`",
        f"- trend_count: `{summary.get('trend_count', 0)}`",
        f"- min_estimated_remaining_seconds: `{summary.get('min_estimated_remaining_seconds')}`",
        f"- history_jsonl: `{result['history_jsonl']}`",
        "",
    ]
    trends = result.get("trends")
    if isinstance(trends, list) and trends:
        lines.extend(
            [
                "## Trend ETA",
                "",
                "| Run ID | Progress | Delta | Window(s) | Rate/s | ETA(s) | Samples/s |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for trend in trends:
            if not isinstance(trend, Mapping):
                continue
            eta = trend.get("estimated_remaining_seconds")
            samples_per_second = trend.get("samples_per_second")
            lines.append(
                "| {run_id} | {progress:.2f}% | {delta:.6f} | {elapsed:.1f} | {rate:.8f} | {eta} | {sps} |".format(
                    run_id=trend.get("run_id", ""),
                    progress=100.0 * float(trend.get("progress_fraction", 0.0)),
                    delta=float(trend.get("progress_delta", 0.0)),
                    elapsed=float(trend.get("elapsed_seconds", 0.0)),
                    rate=float(trend.get("progress_rate_per_second", 0.0)),
                    eta="" if eta is None else f"{float(eta):.1f}",
                    sps="" if samples_per_second is None else f"{float(samples_per_second):.4f}",
                )
            )
        lines.append("")
    lines.extend(
        [
            "| Status | Item | Path | Detail |",
            "|---|---|---|---|",
        ]
    )
    for item in rows:
        detail = str(item["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['status']} | {item['item']} | {item['path']} | {detail} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(H * R).
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", required=True)
    parser.add_argument("--history-jsonl", required=True)
    parser.add_argument("--append-current", action="store_true")
    parser.add_argument("--max-history-lines", type=int, default=200)
    parser.add_argument("--window-seconds", type=float, default=1800.0)
    parser.add_argument("--min-progress-delta", type=float, default=0.0001)
    parser.add_argument("--min-samples-per-second", type=float, default=0.1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    result = run_audit(
        Path(args.queue_json),
        Path(args.history_jsonl),
        append_current=args.append_current,
        observed_at=None,
        max_history_lines=args.max_history_lines,
        window_seconds=args.window_seconds,
        min_progress_delta=args.min_progress_delta,
        min_samples_per_second=args.min_samples_per_second,
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, Path(args.output_md))
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
