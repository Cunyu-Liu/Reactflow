#!/usr/bin/env python3
"""Snapshot system resources for active ReactFlow scale-up runs.

Long full-scale experiments need resource evidence in addition to model metrics:
GPU utilization, memory pressure, and whether watcher/training PIDs are alive.
This script records those signals as JSON/Markdown without changing any running
process.  It is intentionally observational: low GPU utilization is a warning
instead of a failure because RF-A1 is known to be I/O-bound in
``path_sample_features``.

Formula: GPU memory fraction is ``m_used / m_total`` for each device, and
process RSS is reported in MiB as ``rss_kib / 1024``.  Complexity: O(G + P),
where G is GPU count and P is pidfile count.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable, List, Mapping, Optional, Sequence


def row(status: str, item: str, path: Optional[Path] = None, detail: str = "") -> dict:
    """Return one normalized resource audit row.

    Complexity: O(1).
    """

    if status not in {"pass", "warn", "fail"}:
        raise ValueError("status must be pass/warn/fail")
    return {"detail": detail, "item": item, "path": "" if path is None else str(path), "status": status}


def _run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a short resource-inspection command.

    Complexity: O(command output bytes).
    """

    return subprocess.run(args, check=False, capture_output=True, text=True)


def _float_or_none(value: str) -> Optional[float]:
    """Parse a float, returning ``None`` for non-numeric values.

    Complexity: O(len(value)).
    """

    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_gpu_lines(text: str) -> List[dict]:
    """Parse ``nvidia-smi`` CSV rows into GPU dictionaries.

    Expected columns are ``index,name,memory.used,memory.total,utilization.gpu``.
    Formula: ``memory_fraction = memory_used_mb / memory_total_mb`` when total is
    positive.  Complexity: O(G).
    """

    rows: List[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        memory_used = _float_or_none(parts[2])
        memory_total = _float_or_none(parts[3])
        utilization = _float_or_none(parts[4])
        memory_fraction = None
        if memory_used is not None and memory_total and memory_total > 0:
            memory_fraction = memory_used / memory_total
        rows.append(
            {
                "index": parts[0],
                "memory_fraction": memory_fraction,
                "memory_total_mb": memory_total,
                "memory_used_mb": memory_used,
                "name": parts[1],
                "utilization_gpu_percent": utilization,
            }
        )
    return rows


def collect_gpus() -> tuple[List[dict], List[dict]]:
    """Collect GPU utilization rows with ``nvidia-smi``.

    Missing ``nvidia-smi`` returns a warning row rather than failing, so the
    script remains portable on development laptops.  Complexity: O(G).
    """

    try:
        completed = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
    except FileNotFoundError:
        return [], [row("warn", "gpu:nvidia_smi", None, "nvidia-smi not found")]
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return [], [row("warn", "gpu:nvidia_smi", None, detail[:500])]
    gpu_rows = parse_gpu_lines(completed.stdout)
    if not gpu_rows:
        return [], [row("warn", "gpu:list", None, "no GPU rows parsed")]
    audit_rows = [row("pass", "gpu:list", None, f"count={len(gpu_rows)}")]
    active = [gpu for gpu in gpu_rows if (gpu.get("utilization_gpu_percent") or 0.0) > 0.0]
    audit_rows.append(row("pass" if active else "warn", "gpu:active_utilization", None, f"active_gpu_count={len(active)}"))
    return gpu_rows, audit_rows


def read_pid(path: Path) -> Optional[int]:
    """Read one integer PID from ``path``.

    Complexity: O(file bytes).
    """

    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8", errors="replace").strip())
    except ValueError:
        return None


def process_alive(pid: int) -> bool:
    """Return whether ``pid`` appears alive for the current user.

    Complexity: O(1).
    """

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def child_pids(pid: int) -> List[int]:
    """Return direct child PIDs for ``pid`` using ``pgrep -P``.

    Complexity: O(children).
    """

    completed = _run_command(["pgrep", "-P", str(pid)])
    if completed.returncode != 0:
        return []
    result: List[int] = []
    for line in completed.stdout.splitlines():
        try:
            child = int(line.strip())
        except ValueError:
            continue
        if child > 0:
            result.append(child)
    return result


def descendant_pids(pid: int) -> List[int]:
    """Return all descendant PIDs below ``pid`` in breadth-first order.

    Formula: recursively apply ``children(p) = pgrep -P p`` and de-duplicate
    visited PIDs.  Complexity: O(D), where D is descendant count.
    """

    queue = list(child_pids(pid))
    seen = set()
    result: List[int] = []
    while queue:
        child = queue.pop(0)
        if child in seen:
            continue
        seen.add(child)
        result.append(child)
        queue.extend(child_pids(child))
    return result


def process_stats(pid: int) -> Optional[dict]:
    """Return one process resource snapshot from ``ps``.

    Formula: ``rss_mib = rss_kib / 1024``.  Complexity: O(1) command output.
    """

    completed = _run_command(["ps", "-p", str(pid), "-o", "pid=,pcpu=,pmem=,rss=,etime=,comm="])
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    line = completed.stdout.strip().splitlines()[-1]
    parts = line.split(None, 5)
    if len(parts) < 6:
        return None
    rss_kib = _float_or_none(parts[3])
    return {
        "command": parts[5],
        "elapsed": parts[4],
        "pcpu": _float_or_none(parts[1]),
        "pid": int(parts[0]),
        "pmem": _float_or_none(parts[2]),
        "rss_kib": rss_kib,
        "rss_mib": None if rss_kib is None else rss_kib / 1024.0,
    }


def collect_processes(pidfiles: Sequence[Path]) -> tuple[List[dict], List[dict]]:
    """Collect resource rows for pidfiles.

    Complexity: O(P).
    """

    processes: List[dict] = []
    rows: List[dict] = []
    recorded_pids = set()
    for pidfile in pidfiles:
        pid = read_pid(pidfile)
        if pid is None:
            rows.append(row("warn", f"pidfile:{pidfile.name}", pidfile, "missing or invalid pid"))
            continue
        alive = process_alive(pid)
        stats = process_stats(pid) if alive else None
        status = "pass" if alive else "fail"
        detail = f"pid={pid}; alive={alive}"
        if stats is not None:
            detail += f"; pcpu={stats.get('pcpu')}; rss_mib={stats.get('rss_mib')}"
            if pid not in recorded_pids:
                processes.append({"pidfile": str(pidfile), "root_pid": pid, "role": "pidfile", **stats})
                recorded_pids.add(pid)
        rows.append(row(status, f"pidfile:{pidfile.name}", pidfile, detail))
        if not alive:
            continue
        descendants = descendant_pids(pid)
        rows.append(
            row(
                "pass" if descendants else "warn",
                f"pidfile:{pidfile.name}:descendants",
                pidfile,
                f"count={len(descendants)}",
            )
        )
        for child in descendants:
            stats = process_stats(child)
            if stats is None or child in recorded_pids:
                continue
            processes.append({"pidfile": str(pidfile), "root_pid": pid, "role": "descendant", **stats})
            recorded_pids.add(child)
    return processes, rows


def summarize(rows: Iterable[Mapping[str, str]], *, gpu_count: int, process_count: int) -> dict:
    """Return resource audit summary counts.

    Complexity: O(N).
    """

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in rows:
        counts[str(item["status"])] += 1
    return {
        "counts": counts,
        "gpu_count": gpu_count,
        "process_count": process_count,
        "resource_healthy": counts["fail"] == 0,
    }


def run_audit(pidfiles: Sequence[Path]) -> dict:
    """Collect GPU and pidfile resource evidence.

    Complexity: O(G + P).
    """

    gpus, gpu_rows = collect_gpus()
    processes, process_rows = collect_processes(pidfiles)
    rows = gpu_rows + process_rows
    summary = summarize(rows, gpu_count=len(gpus), process_count=len(processes))
    return {"gpus": gpus, "processes": processes, "rows": rows, "summary": summary}


def write_markdown(result: Mapping[str, object], path: Path) -> None:
    """Write a Markdown resource report.

    Complexity: O(G + P + N).
    """

    summary = result["summary"]
    lines = [
        "# ReactFlow System Resource Audit",
        "",
        f"- resource_healthy: `{summary['resource_healthy']}`",
        f"- counts: `{summary['counts']}`",
        f"- gpu_count: `{summary['gpu_count']}`",
        f"- process_count: `{summary['process_count']}`",
        "",
        "## GPUs",
        "",
        "| Index | Name | GPU % | Memory MiB | Memory Fraction |",
        "|---|---|---:|---:|---:|",
    ]
    for gpu in result.get("gpus", []):
        if not isinstance(gpu, Mapping):
            continue
        used = gpu.get("memory_used_mb")
        total = gpu.get("memory_total_mb")
        fraction = gpu.get("memory_fraction")
        lines.append(
            "| {index} | {name} | {util} | {used}/{total} | {fraction} |".format(
                index=gpu.get("index"),
                name=gpu.get("name"),
                util=gpu.get("utilization_gpu_percent"),
                used="n/a" if used is None else f"{float(used):.0f}",
                total="n/a" if total is None else f"{float(total):.0f}",
                fraction="n/a" if fraction is None else f"{float(fraction):.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Processes",
            "",
        "| PID | Role | Root PID | CPU % | MEM % | RSS MiB | Elapsed | Command |",
        "|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for proc in result.get("processes", []):
        if not isinstance(proc, Mapping):
            continue
        rss_mib = proc.get("rss_mib")
        lines.append(
            "| {pid} | {role} | {root_pid} | {pcpu} | {pmem} | {rss} | {elapsed} | {command} |".format(
                pid=proc.get("pid"),
                role=proc.get("role"),
                root_pid=proc.get("root_pid"),
                pcpu=proc.get("pcpu"),
                pmem=proc.get("pmem"),
                rss="n/a" if rss_mib is None else f"{float(rss_mib):.2f}",
                elapsed=proc.get("elapsed"),
                command=proc.get("command"),
            )
        )
    lines.extend(["", "## Audit Rows", "", "| Status | Item | Path | Detail |", "|---|---|---|---|"])
    for item in result.get("rows", []):
        if not isinstance(item, Mapping):
            continue
        lines.append(f"| {item['status']} | {item['item']} | {item['path']} | {item['detail']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(G + P).
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pidfile", action="append", default=[])
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args(argv)
    result = run_audit([Path(path) for path in args.pidfile])
    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    print(text, end="")
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if args.output_md:
        write_markdown(result, Path(args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
