#!/usr/bin/env python3
"""Deploy a verified C0 stage only after process and hash safety gates pass."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from c0_snapshot_state import PROJECT_PATTERNS, project_files


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_reactflow_processes(active_root: Path) -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "pid,ppid,etimes,args"],
        check=True,
        capture_output=True,
        text=True,
    )
    root = str(active_root.resolve())
    return [
        line
        for line in completed.stdout.splitlines()
        if (
            root in line
            or "run_capacity_after_long_range.sh" in line
            or "reactflow.cli evaluate-efold" in line
        )
        and "deploy_c0_stage.py" not in line
    ]


def build_gate(stage_root: Path, active_root: Path, initial_manifest: Path) -> dict:
    initial = json.loads(initial_manifest.read_text(encoding="utf-8"))
    initial_hashes = {
        row["path"]: row["sha256"]
        for row in initial.get("project_files", [])
        if row.get("sha256")
    }
    conflicts = []
    changed = []
    stage_files = {path.relative_to(stage_root).as_posix(): path for path in project_files(stage_root)}
    for relative, stage_path in sorted(stage_files.items()):
        active_path = active_root / relative
        stage_hash = sha256_path(stage_path)
        initial_hash = initial_hashes.get(relative)
        current_hash = sha256_path(active_path) if active_path.is_file() else None
        if initial_hash is not None and current_hash != initial_hash:
            conflicts.append(
                {
                    "path": relative,
                    "initial_sha256": initial_hash,
                    "current_active_sha256": current_hash,
                }
            )
        if current_hash != stage_hash:
            changed.append(
                {
                    "path": relative,
                    "stage_sha256": stage_hash,
                    "current_active_sha256": current_hash,
                    "new_file": not active_path.exists(),
                }
            )
    processes = active_reactflow_processes(active_root)
    return {
        "schema_version": 1,
        "stage_root": str(stage_root.resolve()),
        "active_root": str(active_root.resolve()),
        "active_processes": processes,
        "hash_conflicts": conflicts,
        "files_to_deploy": changed,
        "safe_to_deploy": not processes and not conflicts,
    }


def apply_deployment(gate: dict, stage_root: Path, active_root: Path) -> Path:
    if not gate["safe_to_deploy"]:
        raise RuntimeError("deployment gate is not satisfied")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = active_root.parent / f"reactflow_c0_backup_{timestamp}"
    backup_root.mkdir(parents=True, exist_ok=False)
    for row in gate["files_to_deploy"]:
        relative = Path(row["path"])
        source = stage_root / relative
        target = active_root / relative
        if target.exists():
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return backup_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--initial-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    gate = build_gate(args.stage_root, args.active_root, args.initial_manifest)
    if args.apply:
        gate["backup_root"] = str(apply_deployment(gate, args.stage_root, args.active_root))
        gate["deployed"] = True
    else:
        gate["deployed"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"safe_to_deploy": gate["safe_to_deploy"], "deployed": gate["deployed"], "output": str(args.output)}, sort_keys=True))
    return 0 if gate["safe_to_deploy"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
