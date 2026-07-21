#!/usr/bin/env python3
"""Preflight-check queued ReactFlow full-run stages.

Long-running RF-A1/RF-A2/RF-A3/MMseqs jobs are chained by shell watchers.  This
script verifies the queue contract before a watcher wakes up: scripts are
present and syntactically valid, required input data files exist, and each stage
still references the result file names consumed by downstream readiness checks.

Formula: stage readiness is the conjunction of file-existence predicates
``exists(f_i)`` and text-marker predicates ``marker_j in script_text``.  The
audit is intentionally static and read-only.  Complexity: O(S + F + B), where S
is script count, F is checked file count and B is total script bytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple


EVAL_TIERS: Tuple[str, ...] = ("archiveII", "PDB", "viral", "lncRNA", "human_mRNA")
EXACT_SPLIT_FILES: Tuple[str, ...] = ("train.jsonl", "test.jsonl", "novel.jsonl", "split_manifest.json")
MMSEQS_SPLIT_FILES: Tuple[str, ...] = ("train.jsonl", "test.jsonl", "novel.jsonl", "split_manifest.json")


def row(status: str, item: str, path: Optional[Path] = None, detail: str = "") -> dict:
    """Return one normalized preflight row.

    Complexity: O(1).
    """

    if status not in {"pass", "warn", "fail"}:
        raise ValueError("status must be pass/warn/fail")
    return {"detail": detail, "item": item, "path": "" if path is None else str(path), "status": status}


def audit_exists(path: Path, item: str, *, allow_empty: bool = False) -> List[dict]:
    """Audit that ``path`` exists and, by default, is non-empty.

    Complexity: O(1).
    """

    if not path.exists():
        return [row("fail", item, path, "missing")]
    if path.is_file() and not allow_empty and path.stat().st_size == 0:
        return [row("fail", item, path, "empty")]
    return [row("pass", item, path, f"bytes={path.stat().st_size}" if path.is_file() else "directory")]


def audit_bash_syntax(script: Path) -> List[dict]:
    """Run ``bash -n`` on one script.

    Complexity: O(script bytes).
    """

    if not script.exists():
        return [row("fail", f"script:{script.name}:syntax", script, "missing")]
    completed = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        return [row("pass", f"script:{script.name}:syntax", script, "bash -n ok")]
    detail = (completed.stderr or completed.stdout).strip()
    return [row("fail", f"script:{script.name}:syntax", script, detail[:500])]


def audit_script_markers(script: Path, *, stage: str, markers: Sequence[str]) -> List[dict]:
    """Audit that each required marker appears in ``script`` text.

    Markers are simple string predicates because the queue scripts intentionally
    use shell variables.  Complexity: O(script bytes * marker count).
    """

    if not script.exists():
        return [row("fail", f"stage:{stage}:script", script, "missing")]
    text = script.read_text(encoding="utf-8", errors="replace")
    rows = [row("pass", f"stage:{stage}:script_exists", script, f"bytes={len(text)}")]
    for marker in markers:
        status = "pass" if marker in text else "fail"
        rows.append(row(status, f"stage:{stage}:marker:{marker}", script, "present" if status == "pass" else "missing"))
    return rows


def audit_split(split_dir: Path, *, stage: str, files: Sequence[str]) -> List[dict]:
    """Audit required split files for one stage.

    Complexity: O(file count).
    """

    rows: List[dict] = []
    for name in files:
        rows.extend(audit_exists(split_dir / name, f"stage:{stage}:split:{name}"))
    return rows


def audit_eval_cache(full_run_root: Path, *, stage: str) -> List[dict]:
    """Audit named eFold/RNAndria evaluation cache files.

    Complexity: O(number of tiers).
    """

    rows: List[dict] = []
    for tier in EVAL_TIERS:
        rows.extend(audit_exists(full_run_root / "cache" / f"{tier}.jsonl", f"stage:{stage}:cache:{tier}"))
    return rows


def audit_queue(project_root: Path, full_run_root: Path) -> dict:
    """Run all static queue preflight checks.

    Complexity: O(total script bytes + checked file count).
    """

    scripts = {
        "warm": full_run_root / "run_warm_after_export_rfam_current_exact.sh",
        "warm_recovery": project_root / "scripts" / "run_warm_tail_recovery_after_watcher_exit.sh",
        "contact": project_root / "scripts" / "run_contact_after_warm_rfam_current_exact.sh",
        "mmseqs": project_root / "scripts" / "run_mmseqs_final_after_exact_queue.sh",
        "cross_family": project_root / "scripts" / "run_cross_family_after_mmseqs_final.sh",
        "contact_sweep": project_root / "scripts" / "run_contact_sweep_after_cross_family_balanced.sh",
        "long_range": project_root / "scripts" / "run_long_range_after_contact_sweep.sh",
        "capacity": project_root / "scripts" / "run_capacity_after_long_range.sh",
        "goal_readiness": project_root / "scripts" / "run_goal_readiness_after_final_results.sh",
    }
    rows: List[dict] = []
    for stage, script in scripts.items():
        rows.extend(audit_bash_syntax(script))

    rows.extend(
        audit_script_markers(
            scripts["warm"],
            stage="warm",
            markers=(
                "RF-A1-warm",
                "RF-A2-adapter4",
                "RF-A2-adapter16",
                "warm_rfam_current_exact_results.json",
                "--backend torch",
                "--adapter-dim",
                "summarize_ablation_results.py",
                "for bs in 16 8 4 2 1",
                "out of memory|cuda out|oom|killed|MemoryError",
                "retrying smaller",
                "exhausted batch retries",
            ),
        )
    )
    rows.extend(
        audit_script_markers(
            scripts["warm_recovery"],
            stage="warm_recovery",
            markers=(
                "warm_tail_recovery_after_watcher_exit.pid",
                "warm_after_export_rfam_current_exact.pid",
                "warm_rfam_current_exact_results.json",
                "RF-A1-warm",
                "RF-A2-adapter4",
                "RF-A2-adapter16",
                "label_done",
                "--frozen-cache-shards 4",
                "for bs in 16 8 4 2 1",
                "instability_pattern",
                "FloatingPointError",
                "non-finite",
                "retrying smaller",
                "exhausted batch retries",
                "summarize_ablation_results.py",
            ),
        )
    )
    rows.extend(
        audit_script_markers(
            scripts["contact"],
            stage="contact",
            markers=(
                "warm_rfam_current_exact_results.json",
                "RF-A3-contact",
                "--lambda-contact",
                "contact_rfam_current_exact_results.json",
                "summarize_ablation_results.py",
                "for bs in 16 8 4 2 1",
                "instability_pattern",
                "FloatingPointError",
                "non-finite",
                "retrying smaller",
                "exhausted batch retries",
            ),
        )
    )
    rows.extend(
        audit_script_markers(
            scripts["mmseqs"],
            stage="mmseqs",
            markers=(
                "contact_rfam_current_exact_results.json",
                "RF-M0-base",
                "RF-M1-warm",
                "--frozen-cache-shards 4",
                "mmseqs_final_results.json",
                "rfam_current_mmseqs_seed0",
                "for bs in 16 8 4 2 1",
                "instability_pattern",
                "FloatingPointError",
                "non-finite",
                "retrying smaller",
                "exhausted batch retries",
            ),
        )
    )
    rows.extend(
        audit_script_markers(
            scripts["cross_family"],
            stage="cross_family",
            markers=(
                "mmseqs_final_results.json",
                "RF-CF3-family-balanced",
                "--family-balanced-batches",
                "cross_family_balanced_results.json",
                "audit_cross_family_metrics.py",
                "cross_family_claim_ready",
                "for bs in 16 8 4 2 1",
                "instability_pattern",
                "FloatingPointError",
                "non-finite",
                "retrying smaller",
                "exhausted batch retries",
            ),
        )
    )
    rows.extend(
        audit_script_markers(
            scripts["contact_sweep"],
            stage="contact_sweep",
            markers=(
                "cross_family_balanced_results.json",
                "RF-CF1-contact-strong",
                "CONTACT_SWEEP_LAMBDAS",
                "--lambda-contact",
                "--frozen-cache-shards 4",
                "cross_family_contact_sweep_results.json",
                "cross_family_contact_sweep_metric_audit.json",
                "audit_cross_family_metrics.py",
                "cross_family_claim_ready",
                "for bs in 16 8 4 2 1",
                "instability_pattern",
                "FloatingPointError",
                "non-finite",
                "retrying smaller",
                "exhausted batch retries",
            ),
        )
    )
    rows.extend(
        audit_script_markers(
            scripts["long_range"],
            stage="long_range",
            markers=(
                "cross_family_contact_sweep_results.json",
                "RF-CF2-long-range",
                "LONG_RANGE_WEIGHTS",
                "--contact-long-range-min-distance",
                "--contact-long-range-weight",
                "cross_family_long_range_results.json",
                "cross_family_long_range_metric_audit.json",
                "audit_cross_family_metrics.py",
                "cross_family_claim_ready",
                "for bs in 16 8 4 2 1",
                "instability_pattern",
                "FloatingPointError",
                "non-finite",
                "retrying smaller",
                "exhausted batch retries",
            ),
        )
    )
    rows.extend(
        audit_script_markers(
            scripts["capacity"],
            stage="capacity",
            markers=(
                "cross_family_long_range_results.json",
                "RF-CF5-capacity",
                "CAPACITY_GRID",
                "--hidden-size",
                "--adapter-dim",
                "--family-balanced-batches",
                "--contact-long-range-weight",
                "cross_family_capacity_results.json",
                "cross_family_capacity_metric_audit.json",
                "audit_cross_family_metrics.py",
                "cross_family_claim_ready",
                "for bs in 16 8 4 2 1",
                "instability_pattern",
                "FloatingPointError",
                "non-finite",
                "retrying smaller",
                "exhausted batch retries",
            ),
        )
    )
    rows.extend(
        audit_script_markers(
            scripts["goal_readiness"],
            stage="goal_readiness",
            markers=(
                "warm_rfam_current_exact_results.json",
                "contact_rfam_current_exact_results.json",
                "mmseqs_final_results.json",
                "--fail-if-not-ready",
            ),
        )
    )

    rows.extend(audit_split(full_run_root / "splits" / "rfam_current_exact_seed0", stage="exact", files=EXACT_SPLIT_FILES))
    rows.extend(audit_split(full_run_root / "splits" / "rfam_current_mmseqs_seed0", stage="mmseqs", files=MMSEQS_SPLIT_FILES))
    rows.extend(audit_eval_cache(full_run_root, stage="eval"))
    rows.extend(audit_exists(full_run_root / "frozen" / "ribonanzanet2_sharded_full" / "sharded_manifest.json", "stage:warm:frozen_manifest"))

    expected_outputs = (
        "warm_rfam_current_exact_results.json",
        "contact_rfam_current_exact_results.json",
        "mmseqs_final_results.json",
        "cross_family_balanced_results.json",
        "cross_family_contact_sweep_results.json",
        "cross_family_long_range_results.json",
        "cross_family_capacity_results.json",
    )
    for output in expected_outputs:
        rows.append(row("pass", f"expected_output:{output}", full_run_root / output, "tracked by final_queue/goal_readiness"))

    summary = summarize(rows)
    return {"rows": rows, "summary": summary}


def summarize(rows: Iterable[Mapping[str, str]]) -> dict:
    """Summarize preflight rows.

    Complexity: O(N).
    """

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in rows:
        counts[str(item["status"])] += 1
    return {"counts": counts, "preflight_healthy": counts["fail"] == 0}


def write_markdown(result: Mapping[str, object], path: Path) -> None:
    """Write queue preflight rows as Markdown.

    Complexity: O(N).
    """

    summary = result["summary"]
    rows = result["rows"]
    lines = [
        "# ReactFlow Queue Preflight Audit",
        "",
        f"- preflight_healthy: `{summary['preflight_healthy']}`",
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

    Complexity: O(total script bytes + checked file count).
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--full-run-root", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args(argv)
    result = audit_queue(Path(args.project_root), Path(args.full_run_root))
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
