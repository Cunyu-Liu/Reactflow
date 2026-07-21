#!/usr/bin/env python3
"""Audit the ReactFlow final-result watcher chain.

The goal-readiness audit intentionally fails until final result files exist.
This audit answers the complementary operational question: while those files are
missing, is there still a live watcher responsible for producing them?

Formula: for each stage ``s`` with expected result file ``r_s`` and watcher
pidfile ``p_s``, the stage is healthy when ``r_s`` contains completed metric
rows or, if ``r_s`` is still absent, ``alive(pid(p_s))`` is true.  A non-empty
but malformed result is a failure because it would otherwise let the queue
appear ready while goal-readiness later rejects the metrics.  Complexity:
O(S * R), where S is the number of final queue stages and R is rows per result
file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from reactflow.final_results import EXPECTED_RESULT_TIERS, result_file_ready, validate_final_result_file

FINAL_STAGES: Tuple[Tuple[str, str, str], ...] = (
    (
        "warm_rfam_current_exact",
        "warm_rfam_current_exact_results.json",
        "logs/warm_after_export_rfam_current_exact.pid",
    ),
    (
        "contact_rfam_current_exact",
        "contact_rfam_current_exact_results.json",
        "logs/contact_after_warm_rfam_current_exact.pid",
    ),
    (
        "mmseqs_final",
        "mmseqs_final_results.json",
        "logs/mmseqs_final_after_exact_queue.pid",
    ),
)
FINAL_READINESS_WATCHER = "logs/goal_readiness_after_final_results.pid"
WARM_RECOVERY_WATCHER = "logs/warm_tail_recovery_after_watcher_exit.pid"


def row(status: str, item: str, path: Optional[Path] = None, detail: str = "") -> dict:
    """Return one normalized queue-audit row.

    Complexity: O(1).
    """

    if status not in {"pass", "warn", "fail"}:
        raise ValueError("status must be pass/warn/fail")
    return {"detail": detail, "item": item, "path": "" if path is None else str(path), "status": status}


def process_alive(pid: int) -> bool:
    """Return whether ``pid`` appears alive for the current user.

    ``os.kill(pid, 0)`` performs a kernel liveness check without sending a
    signal.  Complexity: O(1).
    """

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(path: Path) -> Optional[int]:
    """Read an integer pid from ``path``.

    Complexity: O(file bytes), practically O(1).
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return int(path.read_text(encoding="utf-8", errors="replace").strip())
    except ValueError:
        return None


def result_ready(path: Path) -> bool:
    """Return whether a result file exists and has valid completed metrics.

    Formula: a file is ready iff the shared final-result contract returns
    ``state='ready'``.  Complexity: O(file bytes + result rows).
    """

    return result_file_ready(path)


def audit_stage(full_run_root: Path, *, stage: str, result_name: str, pidfile_name: str) -> List[dict]:
    """Audit one final-result-producing stage.

    Complexity: O(R) rows in the result file when present.
    """

    result_path = full_run_root / result_name
    pidfile_path = full_run_root / pidfile_name
    rows: List[dict] = []
    result_validation = validate_final_result_file(result_path)
    if result_validation.state == "ready":
        rows.append(row("pass", f"stage:{stage}:result", result_path, result_validation.detail))
        return rows
    if result_validation.state == "invalid":
        rows.append(row("fail", f"stage:{stage}:result", result_path, result_validation.detail))
        return rows
    rows.append(row("warn", f"stage:{stage}:result", result_path, result_validation.detail))
    pid = read_pid(pidfile_path)
    if pid is None:
        rows.append(row("fail", f"stage:{stage}:watcher", pidfile_path, "missing or invalid pidfile"))
        return rows
    alive = process_alive(pid)
    rows.append(
        row(
            "pass" if alive else "fail",
            f"stage:{stage}:watcher",
            pidfile_path,
            f"pid={pid}; alive={alive}",
        )
    )
    if stage == "warm_rfam_current_exact" and not alive:
        recovery_path = full_run_root / WARM_RECOVERY_WATCHER
        recovery_pid = read_pid(recovery_path)
        if recovery_pid is not None and process_alive(recovery_pid):
            rows[-1] = row("warn", f"stage:{stage}:watcher", pidfile_path, f"pid={pid}; alive=False; recovery watcher active")
            rows.append(
                row(
                    "pass",
                    f"stage:{stage}:recovery_watcher",
                    recovery_path,
                    f"pid={recovery_pid}; alive=True",
                )
            )
    return rows


def audit_final_readiness_watcher(full_run_root: Path, *, final_results_ready: bool) -> List[dict]:
    """Audit the final readiness watcher while final results are pending.

    Once all final results exist, this watcher is allowed to exit after running
    the strict goal-readiness check.  Complexity: O(1).
    """

    pidfile_path = full_run_root / FINAL_READINESS_WATCHER
    if final_results_ready:
        return [row("pass", "final_readiness_watcher", pidfile_path, "final results are present")]
    pid = read_pid(pidfile_path)
    if pid is None:
        return [row("fail", "final_readiness_watcher", pidfile_path, "missing or invalid pidfile")]
    alive = process_alive(pid)
    return [
        row(
            "pass" if alive else "fail",
            "final_readiness_watcher",
            pidfile_path,
            f"pid={pid}; alive={alive}; waiting for final results",
        )
    ]


def summarize(rows: Iterable[Mapping[str, str]]) -> dict:
    """Return queue-health counts.

    Complexity: O(N).
    """

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in rows:
        counts[str(item["status"])] += 1
    return {"counts": counts, "final_queue_healthy": counts["fail"] == 0}


def run_audit(full_run_root: Path) -> dict:
    """Run final queue audit for all configured stages.

    Complexity: O(S).
    """

    rows: List[dict] = []
    final_ready = True
    for stage, result_name, pidfile_name in FINAL_STAGES:
        result_path = full_run_root / result_name
        final_ready = final_ready and result_ready(result_path)
        rows.extend(
            audit_stage(
                full_run_root,
                stage=stage,
                result_name=result_name,
                pidfile_name=pidfile_name,
            )
        )
    rows.extend(audit_final_readiness_watcher(full_run_root, final_results_ready=final_ready))
    summary = summarize(rows)
    summary["final_results_ready"] = final_ready
    return {"rows": rows, "summary": summary}


def write_markdown(result: Mapping[str, object], path: Path) -> None:
    """Write final queue audit rows as Markdown.

    Complexity: O(N).
    """

    summary = result["summary"]
    rows = result["rows"]
    lines = [
        "# ReactFlow Final Queue Audit",
        "",
        f"- final_queue_healthy: `{summary['final_queue_healthy']}`",
        f"- final_results_ready: `{summary['final_results_ready']}`",
        f"- counts: `{summary['counts']}`",
        "",
        "| Status | Item | Path | Detail |",
        "|---|---|---|---|",
    ]
    for item in rows:
        detail = str(item["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['status']} | {item['item']} | {item['path']} | {detail} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(S).
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    result = run_audit(Path(args.full_run_root))
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, Path(args.output_md))
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
