#!/usr/bin/env python3
"""Build a reproducibility manifest for ReactFlow paper artifacts.

The manifest is a deterministic JSON ledger containing:

* Python/platform environment metadata;
* SHA-256 hashes for source, docs, tests and key experiment artifacts;
* summaries copied from algorithm/runtime/paper/goal audits when available.

Large binary frozen-feature shards are intentionally not globbed by default.
Their parent ``sharded_manifest.json`` and per-shard provenance hashes are the
auditable source of truth; hashing every shard during active training would add
unnecessary I/O pressure.

Complexity: O(F + B), where F is the number of included files and B is the total
hashed bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Iterable, List, Mapping, Optional, Sequence


DEFAULT_PROJECT_GLOBS = (
    "README.md",
    "docs/*.md",
    "src/reactflow/*.py",
    "scripts/*.py",
    "scripts/*.sh",
    "tests/*.py",
)
DEFAULT_FULL_RUN_GLOBS = (
    "*.sh",
    "active_eval_progress_audit.json",
    "active_eval_progress_audit.md",
    "algorithm_doc_audit.json",
    "algorithm_doc_audit.md",
    "coverage.json",
    "coverage_audit.json",
    "coverage_audit.md",
    "cross_family_metric_audit.json",
    "cross_family_metric_audit.md",
    "final_queue_audit.json",
    "final_queue_audit.md",
    "goal_readiness_audit.json",
    "goal_readiness_audit.md",
    "paper_artifact_audit.json",
    "paper_artifact_audit.md",
    "queue_preflight_audit.json",
    "queue_preflight_audit.md",
    "profile_bottleneck_audit.json",
    "profile_bottleneck_audit.md",
    "queue_progress_audit.json",
    "queue_progress_audit.md",
    "runtime_health_audit.json",
    "runtime_health_audit.md",
    "system_resource_audit.json",
    "system_resource_audit.md",
    "current_queue_status.json",
    "current_queue_status.md",
    "*.svg",
    "metadata/*.manifest.json",
    "splits/*/split_manifest.json",
    "frozen/*/sharded_manifest.json",
    "logs/*.log",
    "logs/*.json",
    "logs/*.jsonl",
    "logs/*.pid",
    "runs/*/stderr.log",
    "runs/*/stdout*.json",
    "runs/*/monitor_snapshot.json",
    "runs/*/monitor_snapshot.md",
    "runs/*/profile.summary.json",
    "runs/*/training_checkpoint.json",
    "runs/*/eval_summary*.json",
    "*results.json",
    "*results.md",
)
AUDIT_FILES = (
    "active_eval_progress_audit.json",
    "algorithm_doc_audit.json",
    "cross_family_metric_audit.json",
    "final_queue_audit.json",
    "goal_readiness_audit.json",
    "paper_artifact_audit.json",
    "queue_preflight_audit.json",
    "profile_bottleneck_audit.json",
    "queue_progress_audit.json",
    "runtime_health_audit.json",
    "system_resource_audit.json",
)
DEFAULT_PACKAGE_NAMES = (
    "pytest",
    "pytest-cov",
    "sympy",
    "numpy",
    "torch",
    "h5py",
    "pandas",
    "scipy",
    "matplotlib",
)
DEFAULT_TOOL_NAMES = ("mmseqs", "kaggle")
DEFAULT_TOOL_CANDIDATES = {
    "mmseqs": (
        "/home/liucunyu/tools/mmseqs2-avx2/bin/mmseqs",
        "/home/cunyuliu/tools/mmseqs2-avx2/bin/mmseqs",
    )
}


def sha256_path(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 for ``path``.

    Complexity: O(file bytes).
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(path: Path) -> str:
    """Return a POSIX-style relative path string.

    Complexity: O(path length).
    """

    return path.as_posix()


def iter_globbed_files(base: Path, patterns: Sequence[str]) -> List[Path]:
    """Return unique files matching ``patterns`` under ``base``.

    Complexity: O(P + F log F), where P is pattern count and F matched files.
    """

    files = set()
    for pattern in patterns:
        for path in base.glob(pattern):
            if path.is_file():
                files.add(path)
    return sorted(files)


def file_record(path: Path, *, root: Path, max_hash_bytes: int) -> dict:
    """Return one manifest entry for ``path``.

    Files larger than ``max_hash_bytes`` retain size/mtime but skip hashing; this
    prevents accidental multi-GB shard reads during active experiments.

    Complexity: O(min(file bytes, max_hash_bytes)).
    """

    stat = path.stat()
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    can_hash = stat.st_size <= max_hash_bytes
    return {
        "path": _normalize(rel),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_path(path) if can_hash else None,
        "sha256_skipped_reason": "" if can_hash else f"size>{max_hash_bytes}",
    }


def _read_json(path: Path) -> Optional[Mapping[str, object]]:
    """Read a JSON mapping if available.

    Complexity: O(file bytes).
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, Mapping) else None


def collect_audit_summaries(full_run_root: Path) -> dict:
    """Collect summaries from existing audit JSON files.

    Complexity: O(total audit JSON bytes).
    """

    summaries = {}
    for name in AUDIT_FILES:
        path = full_run_root / name
        obj = _read_json(path)
        if obj is None:
            summaries[name] = None
            continue
        summaries[name] = obj.get("summary")
    return summaries


def environment_record() -> dict:
    """Return runtime environment metadata.

    Complexity: O(1).
    """

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "packages": package_versions(DEFAULT_PACKAGE_NAMES),
        "tool_paths": tool_paths(DEFAULT_TOOL_NAMES),
    }


def package_versions(names: Sequence[str]) -> dict:
    """Return installed package versions for ``names``.

    Missing packages are recorded explicitly rather than failing the manifest, so
    optional dependencies such as torch/h5py remain auditable across environments.

    Complexity: O(P) metadata lookups for P package names.
    """

    versions = {}
    for name in names:
        try:
            version = importlib.metadata.version(name)
            versions[name] = {"available": True, "version": version}
        except importlib.metadata.PackageNotFoundError:
            versions[name] = {"available": False, "version": None}
    return versions


def tool_paths(names: Sequence[str]) -> dict:
    """Return executable paths discovered on ``PATH``.

    ``<TOOL>_BIN`` environment variables and known user-level install paths are
    checked after ``PATH`` so server-side tools such as MMseqs2 remain auditable
    even when launched via an explicit absolute path.

    Complexity: O(T * (PATH entries + candidate paths)) for T tools.
    """

    result = {}
    for name in names:
        path = shutil.which(name)
        env_path = os.environ.get(f"{name.upper()}_BIN")
        if path is None and env_path and Path(env_path).exists():
            path = env_path
        if path is None:
            for candidate in DEFAULT_TOOL_CANDIDATES.get(name, ()):
                expanded = Path(candidate).expanduser()
                if expanded.exists():
                    path = str(expanded)
                    break
        result[name] = path
    return result


def build_manifest(
    project_root: Path,
    full_run_root: Path,
    *,
    max_hash_bytes: int = 256 * 1024 * 1024,
    project_globs: Sequence[str] = DEFAULT_PROJECT_GLOBS,
    full_run_globs: Sequence[str] = DEFAULT_FULL_RUN_GLOBS,
) -> dict:
    """Build the reproducibility manifest.

    Complexity: O(F + B), where B is total hashed bytes.
    """

    project_root = project_root.resolve()
    full_run_root = full_run_root.resolve()
    project_files = iter_globbed_files(project_root, project_globs)
    run_files = iter_globbed_files(full_run_root, full_run_globs) if full_run_root.exists() else []
    records = [
        file_record(path, root=project_root, max_hash_bytes=max_hash_bytes)
        for path in sorted(set(project_files + run_files))
    ]
    return {
        "schema_version": 1,
        "environment": environment_record(),
        "project_root": str(project_root),
        "full_run_root": str(full_run_root),
        "max_hash_bytes": max_hash_bytes,
        "file_count": len(records),
        "files": records,
        "audit_summaries": collect_audit_summaries(full_run_root),
    }


def write_markdown(manifest: Mapping[str, object], path: Path) -> None:
    """Write a compact Markdown view of the manifest.

    Complexity: O(F).
    """

    env = manifest["environment"]
    audit_summaries = manifest["audit_summaries"]
    lines = [
        "# ReactFlow Reproducibility Manifest",
        "",
        f"- file_count: `{manifest['file_count']}`",
        f"- max_hash_bytes: `{manifest['max_hash_bytes']}`",
        f"- project_root: `{manifest['project_root']}`",
        f"- full_run_root: `{manifest['full_run_root']}`",
        f"- python: `{env['python_version'].split()[0]}`",
        f"- platform: `{env['platform']}`",
        "",
        "## Environment Packages",
        "",
        "| Package | Available | Version |",
        "|---|---|---|",
    ]
    for name, info in env["packages"].items():
        lines.append(f"| {name} | `{info['available']}` | `{info['version']}` |")
    lines.extend(
        [
            "",
            "## Tool Paths",
            "",
            "| Tool | Path |",
            "|---|---|",
        ]
    )
    for name, tool_path in env["tool_paths"].items():
        lines.append(f"| {name} | `{tool_path}` |")
    lines.extend(
        [
            "",
            "## Audit Summaries",
            "",
            "| Audit | Summary |",
            "|---|---|",
        ]
    )
    for name, summary in audit_summaries.items():
        lines.append(f"| {name} | `{summary}` |")
    lines.extend(["", "## Files", "", "| Path | Size | SHA256 |", "|---|---:|---|"])
    for item in manifest["files"]:
        sha = item["sha256"] or item["sha256_skipped_reason"]
        lines.append(f"| {item['path']} | {item['size_bytes']} | `{sha}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(F + B), where B is total hashed bytes.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--full-run-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-hash-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args(argv)

    manifest = build_manifest(
        Path(args.project_root),
        Path(args.full_run_root),
        max_hash_bytes=args.max_hash_bytes,
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(manifest, Path(args.output_md))
    print(json.dumps({"file_count": manifest["file_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
