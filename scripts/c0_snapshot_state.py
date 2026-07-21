#!/usr/bin/env python3
"""Capture the read-only state required before a ReactFlow C0 run.

The snapshot deliberately hashes only source-sized files and selected evidence
artifacts.  It never opens frozen-feature shards or mutates the active project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


PROJECT_PATTERNS = (
    "pyproject.toml",
    "README.md",
    "src/reactflow/*.py",
    "scripts/*.py",
    "scripts/*.sh",
    "tests/*.py",
    "docs/*.md",
)
EVIDENCE_NAMES = (
    "cross_family_long_range_results.json",
    "cross_family_capacity_results.json",
    "baseline_efold_results.json",
    "baseline_efold_in_clan_progress_results.json",
    "baseline_efold_novel_progress_results.json",
    "data_diversity_audit.json",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_path(path),
    }


def project_files(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in PROJECT_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def command_output(command: list[str]) -> dict:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {"available": False, "returncode": None, "stdout": "", "stderr": "not found"}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_snapshot(project_root: Path, full_run_root: Path) -> dict:
    evidence = []
    for name in EVIDENCE_NAMES:
        path = full_run_root / name
        if path.is_file():
            evidence.append(file_record(path, full_run_root))
        else:
            evidence.append({"path": name, "present": False})
    processes = command_output(
        [
            "ps",
            "-eo",
            "pid,ppid,etimes,pcpu,pmem,lstart,args",
        ]
    )
    process_lines = [
        line
        for line in processes["stdout"].splitlines()
        if "reactflow" in line.lower() or "run_capacity_after_long_range" in line
    ]
    return {
        "schema_version": 1,
        "project_root": str(project_root.resolve()),
        "full_run_root": str(full_run_root.resolve()),
        "project_files": [file_record(path, project_root) for path in sorted(project_files(project_root))],
        "evidence_files": evidence,
        "active_processes": process_lines,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--full-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_snapshot(args.project_root, args.full_run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "process_count": len(payload["active_processes"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
