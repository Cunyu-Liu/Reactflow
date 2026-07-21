#!/usr/bin/env python3
"""Audit ReactFlow paper artifacts for publication readiness.

The goal is not to prove model quality.  It verifies that the evidence needed
for a paper table exists and is internally consistent: public-data caches,
Rfam/MMseqs metadata, leakage-safe split manifests, run logs, profiles,
checkpoints and final metrics.  The output is deliberately machine-readable
JSON plus a compact Markdown table so experiment status can be reviewed without
manual shell spelunking.

Complexity: O(A + M), where A is the number of checked artifact files and M is
the number of split assignments when validating a split manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reactflow.splits import manifest_from_json  # noqa: E402


DEFAULT_CACHE_FILES = (
    "efold_train.jsonl",
    "archiveII.jsonl",
    "PDB.jsonl",
    "viral.jsonl",
    "lncRNA.jsonl",
    "human_mRNA.jsonl",
)


def check(status: str, item: str, path: Optional[Path] = None, detail: str = "") -> dict:
    """Return one normalized audit row.

    Status is one of ``pass``, ``warn`` or ``fail``.  Complexity: O(1).
    """

    if status not in {"pass", "warn", "fail"}:
        raise ValueError("status must be pass/warn/fail")
    return {
        "detail": detail,
        "item": item,
        "path": "" if path is None else str(path),
        "status": status,
    }


def _line_count(path: Path) -> int:
    """Return line count for a text artifact.

    Complexity: O(B), where B is file size.
    """

    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def audit_cache_files(cache_dir: Path, names: Sequence[str] = DEFAULT_CACHE_FILES) -> List[dict]:
    """Check public-data cache JSONL files.

    A cache passes when it exists, is non-empty and has at least one JSONL line.
    Complexity: O(total cache bytes) for line counts.
    """

    rows: List[dict] = []
    for name in names:
        path = cache_dir / name
        if not path.exists():
            rows.append(check("fail", f"cache:{name}", path, "missing"))
            continue
        if path.stat().st_size == 0:
            rows.append(check("fail", f"cache:{name}", path, "empty"))
            continue
        rows.append(check("pass", f"cache:{name}", path, f"lines={_line_count(path)}"))
    return rows


def _load_json(path: Path) -> Optional[Mapping[str, object]]:
    """Load a JSON mapping or return ``None``.

    Complexity: O(B), where B is file size.
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, Mapping) else None


def audit_metadata_manifest(path: Path, *, require_mmseqs: bool = True) -> List[dict]:
    """Check Rfam/MMseqs metadata provenance.

    For final paper tables, ``cluster_method`` must be ``mmseqs`` and
    ``mmseqs_error`` must be null.  Complexity: O(B).
    """

    obj = _load_json(path)
    if obj is None:
        return [check("fail", "metadata_manifest", path, "missing or invalid JSON")]
    rows = [check("pass", "metadata_manifest", path, "valid JSON")]
    method = obj.get("cluster_method")
    if require_mmseqs and method != "mmseqs":
        rows.append(check("fail", "metadata_cluster_method", path, f"expected mmseqs, got {method!r}"))
    else:
        rows.append(check("pass", "metadata_cluster_method", path, f"cluster_method={method!r}"))
    if obj.get("mmseqs_error"):
        rows.append(check("fail", "metadata_mmseqs_error", path, str(obj.get("mmseqs_error"))[:500]))
    else:
        rows.append(check("pass", "metadata_mmseqs_error", path, "mmseqs_error is null"))
    for key in ("input_records", "metadata_records", "cluster_count", "split_group_count"):
        value = obj.get(key)
        status = "pass" if isinstance(value, int) and value > 0 else "fail"
        rows.append(check(status, f"metadata_{key}", path, f"{key}={value!r}"))
    return rows


def audit_split_manifest(path: Path) -> List[dict]:
    """Load and validate a split manifest.

    :func:`manifest_from_json` reruns clan/cluster leakage validation, so a pass
    means no clan or cluster appears in more than one split.  Complexity: O(M)
    for M assignments.
    """

    if not path.exists():
        return [check("fail", "split_manifest", path, "missing")]
    try:
        manifest = manifest_from_json(path)
    except Exception as exc:  # noqa: BLE001 - audit should report, not crash.
        return [check("fail", "split_manifest", path, f"invalid/leaky: {exc}")]
    counts = manifest.counts_by_split()
    rows = [check("pass", "split_manifest", path, f"assignments={len(manifest.assignments)} counts={counts}")]
    for split, count in sorted(counts.items()):
        rows.append(check("pass" if count > 0 else "fail", f"split_count:{split}", path, f"{count}"))
    return rows


def _has_final_metrics(run_dir: Path) -> bool:
    """Return whether a run directory contains a non-empty final metrics JSON."""

    for name in ("eval_summary.json", "eval_summary.recovered.json", "stdout.json", "stdout.recovered.json"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            return True
    return False


def audit_run_dir(run_dir: Path, *, require_final_metrics: bool = False) -> List[dict]:
    """Audit one run directory.

    Active runs may pass with a profile or monitor snapshot even before final
    metrics exist.  Finished paper-table runs should call this with
    ``require_final_metrics=True``.  Complexity: O(1) file stats.
    """

    if not run_dir.exists():
        return [check("fail", f"run:{run_dir.name}", run_dir, "missing")]
    rows = [check("pass", f"run:{run_dir.name}", run_dir, "directory exists")]
    stderr = run_dir / "stderr.log"
    if stderr.exists() and stderr.stat().st_size > 0:
        rows.append(check("fail", f"run_stderr:{run_dir.name}", stderr, f"bytes={stderr.stat().st_size}"))
    else:
        rows.append(check("pass", f"run_stderr:{run_dir.name}", stderr, "empty or absent"))
    has_profile = (run_dir / "profile.summary.json").exists() or (run_dir / "monitor_snapshot.json").exists() or (run_dir / "profile.jsonl").exists()
    rows.append(check("pass" if has_profile else "warn", f"run_profile:{run_dir.name}", run_dir, f"profile_or_monitor={has_profile}"))
    has_checkpoint = (run_dir / "training_checkpoint.json").exists()
    rows.append(check("pass" if has_checkpoint else "warn", f"run_checkpoint:{run_dir.name}", run_dir / "training_checkpoint.json", f"checkpoint={has_checkpoint}"))
    has_metrics = _has_final_metrics(run_dir)
    if require_final_metrics and not has_metrics:
        rows.append(check("fail", f"run_metrics:{run_dir.name}", run_dir, "missing final metrics"))
    else:
        rows.append(check("pass" if has_metrics else "warn", f"run_metrics:{run_dir.name}", run_dir, f"final_metrics={has_metrics}"))
    return rows


def summarize(rows: Sequence[Mapping[str, str]]) -> dict:
    """Return aggregate audit counts.

    Complexity: O(N) for N rows.
    """

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for row in rows:
        counts[str(row["status"])] += 1
    return {"counts": counts, "ok_for_paper_table": counts["fail"] == 0}


def write_markdown(rows: Sequence[Mapping[str, str]], summary: Mapping[str, object], path: Path) -> None:
    """Write audit rows to Markdown.

    Complexity: O(N) for N rows.
    """

    lines = [
        "# ReactFlow Paper Artifact Audit",
        "",
        f"- ok_for_paper_table: `{summary['ok_for_paper_table']}`",
        f"- counts: `{summary['counts']}`",
        "",
        "| Status | Item | Path | Detail |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['status']} | {row['item']} | {row['path']} | {row['detail']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    full_run_root: Path,
    *,
    require_final_metrics: bool,
    run_globs: Sequence[str],
    require_mmseqs: bool = True,
) -> dict:
    """Run the full paper artifact audit.

    Complexity: O(cache bytes + split assignments + checked runs).
    """

    rows: List[dict] = []
    rows.extend(audit_cache_files(full_run_root / "cache"))
    rows.extend(audit_metadata_manifest(full_run_root / "metadata/rfam_current_mmseqs_metadata.manifest.json", require_mmseqs=require_mmseqs))
    rows.extend(audit_split_manifest(full_run_root / "splits/rfam_current_mmseqs_seed0/split_manifest.json"))
    run_root = full_run_root / "runs"
    for pattern in run_globs:
        matches = sorted(path for path in run_root.glob(pattern) if path.is_dir())
        if not matches:
            rows.append(check("warn", f"run_glob:{pattern}", run_root, "no matches"))
        for run_dir in matches:
            rows.extend(audit_run_dir(run_dir, require_final_metrics=require_final_metrics))
    summary = summarize(rows)
    return {"rows": rows, "summary": summary}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for paper artifact auditing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-root", required=True)
    parser.add_argument("--run-glob", action="append", default=[])
    parser.add_argument("--require-final-metrics", action="store_true")
    parser.add_argument("--allow-non-mmseqs", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    result = run_audit(
        Path(args.full_run_root),
        require_final_metrics=args.require_final_metrics,
        run_globs=args.run_glob,
        require_mmseqs=not args.allow_non_mmseqs,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result["rows"], result["summary"], Path(args.output_md))
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
