#!/usr/bin/env python3
"""Audit runtime health for long ReactFlow jobs.

This complements paper-artifact auditing.  Paper artifacts answer "is the
evidence complete enough for a table?"; runtime health answers "is an active
background job still alive and producing evidence?".

Checks are intentionally conservative and read-only:

* profile JSONL exists, is non-empty and has a recent modification time;
* profile tail contains parseable JSON events;
* stderr is empty (or absent);
* optional pidfiles point to live processes;
* optional monitor snapshot has progress in ``[0, 1]``.

Complexity: O(P + B), where P is the number of pidfiles and B is the bounded
profile/stderr tail size.  The script never scans full profile files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Iterable, List, Mapping, Optional, Sequence


def row(status: str, item: str, path: Optional[Path] = None, detail: str = "") -> dict:
    """Return one normalized health row.

    Status is ``pass``, ``warn`` or ``fail``.  Complexity: O(1).
    """

    if status not in {"pass", "warn", "fail"}:
        raise ValueError("status must be pass/warn/fail")
    return {"detail": detail, "item": item, "path": "" if path is None else str(path), "status": status}


def _tail_text(path: Path, limit: int) -> str:
    """Read at most ``limit`` trailing bytes from ``path`` as UTF-8 text.

    Complexity: O(limit).
    """

    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")


def _last_json_event(path: Path, limit: int = 20000) -> Optional[Mapping[str, object]]:
    """Return the last parseable JSON object in a profile tail.

    A profile may be read while another process is writing a final partial line;
    scanning backward through the bounded tail avoids false failures.

    Complexity: O(limit).
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    for line in reversed(_tail_text(path, limit).splitlines()):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, Mapping):
            return obj
    return None


def process_alive(pid: int) -> bool:
    """Return whether ``pid`` appears alive for the current user.

    ``os.kill(pid, 0)`` does not send a signal; it only asks the kernel whether
    the process exists and is signalable.  Complexity: O(1).
    """

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_json_mapping(path: Path) -> Optional[Mapping[str, object]]:
    """Return a JSON object from ``path`` when it is present and mapping-shaped.

    Formula: parse bytes ``B`` into object ``x = parse(B)`` and accept only when
    ``x in Map``.  Returning ``None`` rather than raising lets health checks
    compose several independent evidence sources without masking the original
    missing or malformed file.  Complexity: O(file bytes).
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, Mapping):
        return obj
    return None


def run_completed(run_dir: Path) -> bool:
    """Return whether ``run_dir`` has durable completed-run evidence.

    Formula: ``complete = 1[stdout is Map and checkpoint_size > 0 and tiers is
    Map]``.  A completed run no longer writes profile events, so profile mtime is
    not a liveness signal after this predicate becomes true; freshness remains
    mandatory for active runs.  Complexity: O(stdout bytes + 1).
    """

    stdout = _read_json_mapping(run_dir / "stdout.json")
    checkpoint = run_dir / "training_checkpoint.json"
    return bool(stdout and isinstance(stdout.get("tiers"), Mapping) and checkpoint.exists() and checkpoint.stat().st_size > 0)


def profile_epoch_closed(run_dir: Path) -> bool:
    """Return whether training profile evidence has reached ``epoch_total``.

    Formula: parse ``profile.summary.json`` and accept iff
    ``summary.phases.epoch_total.total_seconds > 0`` or a
    ``phases_by_total_seconds`` row has ``phase='epoch_total'``.  This state
    means the profiled training epoch has closed, while the process may still be
    spending unprofiled wall time in evaluation or final JSON/checkpoint
    materialization.  Complexity: O(summary bytes + P).
    """

    summary = _read_json_mapping(run_dir / "profile.summary.json")
    if summary is None:
        return False
    phases = summary.get("phases")
    if isinstance(phases, Mapping):
        epoch = phases.get("epoch_total")
        if isinstance(epoch, Mapping) and isinstance(epoch.get("total_seconds"), (int, float)) and float(epoch["total_seconds"]) > 0.0:
            return True
    phase_rows = summary.get("phases_by_total_seconds")
    if isinstance(phase_rows, list):
        for item in phase_rows:
            if isinstance(item, Mapping) and str(item.get("phase", "")) == "epoch_total":
                return True
    return False


def profile_load_in_progress(run_dir: Path) -> bool:
    """Return whether the latest profile event is a pre-training load heartbeat.

    Formula: ``loading = phase(last(profile.jsonl)) startswith 'load_'``.  This
    state can legitimately have ``monitor_snapshot.progress_fraction=None``
    because no training step has executed yet.  Complexity: O(profile tail
    bytes).
    """

    event = _last_json_event(run_dir / "profile.jsonl")
    return isinstance(event, Mapping) and str(event.get("phase", "")).startswith("load_")


def audit_profile(
    profile: Path,
    *,
    max_age_seconds: float,
    now: Optional[float] = None,
    completed: bool = False,
    loading: bool = False,
    post_training_eval: bool = False,
) -> List[dict]:
    """Audit profile freshness and parseability.

    Formula: active runs require ``age(profile) <= max_age_seconds``; completed
    runs require only a non-empty parseable profile because no new events should
    be appended after ``stdout.json`` and the checkpoint are closed.  When
    ``post_training_eval`` is true, a stale profile is downgraded to warning
    because profiling has reached ``epoch_total`` and the process may be in
    unprofiled evaluation/finalization.  ``loading`` is also a warning state:
    the profile has emitted a ``load_*`` heartbeat, but large JSONL cache loads
    may not emit per-row progress.  Complexity: O(profile tail bytes).
    """

    now = time.time() if now is None else now
    rows: List[dict] = []
    if not profile.exists():
        return [row("fail", "profile_exists", profile, "missing")]
    if profile.stat().st_size == 0:
        return [row("fail", "profile_nonempty", profile, "empty")]
    rows.append(row("pass", "profile_nonempty", profile, f"bytes={profile.stat().st_size}"))
    age = max(0.0, now - profile.stat().st_mtime)
    if completed:
        rows.append(row("pass", "profile_fresh", profile, f"completed run; age_seconds={age:.1f}"))
    elif loading and age > max_age_seconds:
        rows.append(row("warn", "profile_fresh", profile, f"loading; age_seconds={age:.1f}; max={max_age_seconds:.1f}"))
    elif post_training_eval and age > max_age_seconds:
        rows.append(row("warn", "profile_fresh", profile, f"post-training eval/finalize; age_seconds={age:.1f}; max={max_age_seconds:.1f}"))
    else:
        status = "pass" if age <= max_age_seconds else "fail"
        rows.append(row(status, "profile_fresh", profile, f"age_seconds={age:.1f}; max={max_age_seconds:.1f}"))
    event = _last_json_event(profile)
    if event is None:
        rows.append(row("fail", "profile_tail_json", profile, "no parseable JSON event in tail"))
    else:
        rows.append(row("pass", "profile_tail_json", profile, json.dumps(dict(event), sort_keys=True)[:500]))
    return rows


def audit_stderr(stderr: Path) -> List[dict]:
    """Audit stderr log.

    Empty or absent stderr is considered healthy for active runs.  Complexity:
    O(1) unless stderr has content, in which case a bounded tail is read.
    """

    if not stderr.exists():
        return [row("pass", "stderr_empty", stderr, "absent")]
    size = stderr.stat().st_size
    if size == 0:
        return [row("pass", "stderr_empty", stderr, "empty")]
    return [row("fail", "stderr_empty", stderr, f"bytes={size}; tail={_tail_text(stderr, 1000)!r}")]


def audit_pidfile(path: Path) -> List[dict]:
    """Audit one pidfile.

    Complexity: O(1).
    """

    if not path.exists():
        return [row("warn", f"pidfile:{path.name}", path, "missing")]
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        pid = int(text)
    except ValueError:
        return [row("fail", f"pidfile:{path.name}", path, f"invalid pid {text!r}")]
    status = "pass" if process_alive(pid) else "fail"
    return [row(status, f"pidfile:{path.name}", path, f"pid={pid}")]


def audit_monitor(
    path: Path,
    *,
    completed: bool = False,
    loading: bool = False,
    post_training_eval: bool = False,
) -> List[dict]:
    """Audit optional monitor snapshot progress.

    Complexity: O(B), where B is snapshot size.
    """

    if not path.exists() or path.stat().st_size == 0:
        return [row("warn", "monitor_snapshot", path, "missing or empty")]
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [row("fail", "monitor_snapshot", path, f"invalid JSON: {exc}")]
    progress = obj.get("progress_fraction") if isinstance(obj, Mapping) else None
    if isinstance(progress, (int, float)) and 0.0 <= float(progress) <= 1.0:
        return [row("pass", "monitor_progress", path, f"progress={100.0 * float(progress):.2f}%")]
    if progress is None and completed:
        return [row("pass", "monitor_progress", path, "completed; progress=None")]
    if progress is None and post_training_eval:
        return [row("warn", "monitor_progress", path, "post_training_eval; progress=None")]
    if progress is None and loading:
        return [row("warn", "monitor_progress", path, "loading; progress=None")]
    return [row("fail", "monitor_progress", path, f"invalid progress={progress!r}")]


def summarize(rows: Iterable[Mapping[str, str]]) -> dict:
    """Return health counts and overall health.

    Complexity: O(N).
    """

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in rows:
        counts[str(item["status"])] += 1
    return {"counts": counts, "healthy": counts["fail"] == 0}


def write_markdown(rows: Sequence[Mapping[str, str]], summary: Mapping[str, object], path: Path) -> None:
    """Write health rows as Markdown.

    Complexity: O(N).
    """

    lines = [
        "# ReactFlow Runtime Health Audit",
        "",
        f"- healthy: `{summary['healthy']}`",
        f"- counts: `{summary['counts']}`",
        f"- audited_run_count: `{summary.get('audited_run_count', 1)}`",
        "",
        "| Status | Item | Path | Detail |",
        "|---|---|---|---|",
    ]
    for item in rows:
        lines.append(f"| {item['status']} | {item['item']} | {item['path']} | {item['detail']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    run_dir: Path,
    *,
    pidfiles: Sequence[Path],
    max_profile_age_seconds: float,
) -> dict:
    """Run all runtime health checks for one active run.

    Formula: run-local checks use ``completed = run_completed(run_dir)`` and
    ``post_training_eval = profile_epoch_closed(run_dir) and not completed`` so
    heartbeat freshness is strict during training, downgraded during unprofiled
    evaluation/finalization, and unnecessary after final artifacts close.
    Complexity: O(P + stdout/summary bytes + bounded tails).
    """

    rows: List[dict] = []
    completed = run_completed(run_dir)
    post_training_eval = (not completed) and profile_epoch_closed(run_dir)
    loading = (not completed) and (not post_training_eval) and profile_load_in_progress(run_dir)
    rows.extend(
        audit_profile(
            run_dir / "profile.jsonl",
            max_age_seconds=max_profile_age_seconds,
            completed=completed,
            loading=loading,
            post_training_eval=post_training_eval,
        )
    )
    rows.extend(audit_stderr(run_dir / "stderr.log"))
    rows.extend(
        audit_monitor(
            run_dir / "monitor_snapshot.json",
            completed=completed,
            loading=loading,
            post_training_eval=post_training_eval,
        )
    )
    for pidfile in pidfiles:
        rows.extend(audit_pidfile(pidfile))
    summary = summarize(rows)
    return {"rows": rows, "summary": summary}


def run_multi_audit(
    run_dirs: Sequence[Path],
    *,
    pidfiles: Sequence[Path],
    max_profile_age_seconds: float,
) -> dict:
    """Run runtime health checks for one or more active run directories.

    Pidfiles are process-level queue guardians and are checked once, after all
    run-local profile/stderr/monitor checks.  Formula: each run ``r`` uses its
    own completed predicate ``c_r`` and profile-closed predicate ``e_r`` to
    distinguish active training from post-training evaluation/finalization.
    Complexity: O(R * (stdout/summary bytes + bounded tails) + P).
    """

    rows: List[dict] = []
    audited = []
    for run_dir in run_dirs:
        audited.append(str(run_dir))
        completed = run_completed(run_dir)
        post_training_eval = (not completed) and profile_epoch_closed(run_dir)
        loading = (not completed) and (not post_training_eval) and profile_load_in_progress(run_dir)
        for item in audit_profile(
            run_dir / "profile.jsonl",
            max_age_seconds=max_profile_age_seconds,
            completed=completed,
            loading=loading,
            post_training_eval=post_training_eval,
        ):
            rows.append({**item, "item": f"run:{run_dir.name}:{item['item']}"})
        for item in audit_stderr(run_dir / "stderr.log"):
            rows.append({**item, "item": f"run:{run_dir.name}:{item['item']}"})
        for item in audit_monitor(
            run_dir / "monitor_snapshot.json",
            completed=completed,
            loading=loading,
            post_training_eval=post_training_eval,
        ):
            rows.append({**item, "item": f"run:{run_dir.name}:{item['item']}"})
    for pidfile in pidfiles:
        rows.extend(audit_pidfile(pidfile))
    summary = summarize(rows)
    summary["audited_run_count"] = len(run_dirs)
    return {"audited_run_dirs": audited, "rows": rows, "summary": summary}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--pidfile", action="append", default=[])
    parser.add_argument("--max-profile-age-seconds", type=float, default=900.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    result = run_multi_audit(
        [Path(path) for path in args.run_dir],
        pidfiles=[Path(path) for path in args.pidfile],
        max_profile_age_seconds=args.max_profile_age_seconds,
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result["rows"], result["summary"], Path(args.output_md))
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
