#!/usr/bin/env python3
"""Aggregate ReactFlow goal-readiness evidence.

This audit is deliberately stricter than the individual artifact audits.  It
answers one question: "Can we truthfully claim the full user goal is complete?"

The answer is only true when algorithm documentation is strict-ready, public data
and split evidence exist, README/data-governance sections are present, active
runtime health is clean, resource auditing is healthy, performance bottleneck
auditing is healthy, and the final result tables for the queued full-scale
experiments exist.

Complexity: O(A + R), where A is the number of checked artifact files and R is
the bounded README/documentation text size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from reactflow.final_results import EXPECTED_RESULT_TIERS, FINAL_RESULT_FILES, validate_final_result_file


REQUIRED_README_MARKERS = (
    "## 科学问题与定制化模型路线",
    "```mermaid",
    "## 公共数据源",
    "## 数据预处理契约",
    "## SOTA / 竞品表",
    "## 环境",
    "可复现",
)
REQUIRED_DATA_MARKERS = (
    "https://doi.org/10.5061/dryad.79cnp5j95",
    "https://www.kaggle.com/models/shujun717/ribonanzanet2/PyTorch/alpha/1",
    "Preprocessing Contract",
    "MMseqs",
)
def row(status: str, item: str, path: Optional[Path] = None, detail: str = "") -> dict:
    """Return one normalized readiness row.

    Complexity: O(1).
    """

    if status not in {"pass", "warn", "fail"}:
        raise ValueError("status must be pass/warn/fail")
    return {"detail": detail, "item": item, "path": "" if path is None else str(path), "status": status}


def _read_json(path: Path) -> Optional[Mapping[str, object]]:
    """Read a JSON object, returning ``None`` for missing/invalid files.

    Complexity: O(file bytes).
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, Mapping) else None


def audit_algorithm_doc(full_run_root: Path) -> List[dict]:
    """Audit algorithm documentation strict-readiness.

    Complexity: O(audit JSON size).
    """

    path = full_run_root / "algorithm_doc_audit.json"
    obj = _read_json(path)
    if obj is None:
        return [row("fail", "algorithm_doc_audit", path, "missing or invalid JSON")]
    summary = obj.get("summary")
    if not isinstance(summary, Mapping):
        return [row("fail", "algorithm_doc_summary", path, "missing summary")]
    strict_ready = bool(summary.get("strict_ready"))
    status = "pass" if strict_ready else "fail"
    return [row(status, "algorithm_doc_strict_ready", path, json.dumps(dict(summary), sort_keys=True))]


def audit_runtime_health(full_run_root: Path) -> List[dict]:
    """Audit active runtime health.

    Complexity: O(audit JSON size).
    """

    path = full_run_root / "runtime_health_audit.json"
    obj = _read_json(path)
    if obj is None:
        return [row("warn", "runtime_health_audit", path, "missing or invalid JSON")]
    summary = obj.get("summary")
    if not isinstance(summary, Mapping):
        return [row("warn", "runtime_health_summary", path, "missing summary")]
    healthy = bool(summary.get("healthy"))
    status = "pass" if healthy else "fail"
    return [row(status, "runtime_health", path, json.dumps(dict(summary), sort_keys=True))]


def audit_system_resources(full_run_root: Path) -> List[dict]:
    """Audit persisted system-resource health evidence.

    Resource auditing records GPU utilization, memory pressure and active
    training/watchdog process RSS/CPU.  It must be parseable and healthy before
    the overall goal can be declared complete, so scale-up claims are backed by
    an auditable resource snapshot.

    Complexity: O(audit JSON size).
    """

    path = full_run_root / "system_resource_audit.json"
    obj = _read_json(path)
    if obj is None:
        return [row("warn", "system_resource_audit", path, "missing or invalid JSON")]
    summary = obj.get("summary")
    if not isinstance(summary, Mapping):
        return [row("warn", "system_resource_summary", path, "missing summary")]
    healthy = bool(summary.get("resource_healthy"))
    status = "pass" if healthy else "fail"
    return [row(status, "system_resources", path, json.dumps(dict(summary), sort_keys=True))]


def audit_queue_progress(full_run_root: Path) -> List[dict]:
    """Audit active queue progress trend health.

    Complexity: O(audit JSON size).
    """

    path = full_run_root / "queue_progress_audit.json"
    obj = _read_json(path)
    if obj is None:
        return [row("warn", "queue_progress_audit", path, "missing or invalid JSON")]
    summary = obj.get("summary")
    if not isinstance(summary, Mapping):
        return [row("warn", "queue_progress_summary", path, "missing summary")]
    healthy = bool(summary.get("progress_healthy"))
    status = "pass" if healthy else "fail"
    return [row(status, "queue_progress", path, json.dumps(dict(summary), sort_keys=True))]


def audit_cross_family_metrics(full_run_root: Path) -> List[dict]:
    """Audit cross-family metric tracking evidence.

    The cross-family audit records ``novel_clan`` F1/MCC and the
    ``in_clan -> novel_clan`` generalization gap.  The full goal cannot be
    marked complete merely because this artifact is parseable: it must also show
    ``cross_family_claim_ready=true`` so the top-tier objective does not pass
    while novel-family generation is still below the declared claim gate.

    Complexity: O(audit JSON size).
    """

    path = full_run_root / "cross_family_metric_audit.json"
    obj = _read_json(path)
    if obj is None:
        return [row("warn", "cross_family_metric_audit", path, "missing or invalid JSON")]
    summary = obj.get("summary")
    if not isinstance(summary, Mapping):
        return [row("warn", "cross_family_metric_summary", path, "missing summary")]
    healthy = bool(summary.get("cross_family_healthy"))
    claim_ready = bool(summary.get("cross_family_claim_ready"))
    status = "pass" if healthy and claim_ready else "fail"
    return [row(status, "cross_family_metrics", path, json.dumps(dict(summary), sort_keys=True))]


def audit_profile_bottlenecks(full_run_root: Path) -> List[dict]:
    """Audit active profile-bottleneck evidence.

    The bottleneck audit may contain warnings for known optimization targets
    (for example RF-A1 predating ``frozen_batch_prefetch``), but it must itself
    be healthy and parseable before the overall goal can be marked complete.

    Complexity: O(audit JSON size).
    """

    path = full_run_root / "profile_bottleneck_audit.json"
    obj = _read_json(path)
    if obj is None:
        return [row("warn", "profile_bottleneck_audit", path, "missing or invalid JSON")]
    summary = obj.get("summary")
    if not isinstance(summary, Mapping):
        return [row("warn", "profile_bottleneck_summary", path, "missing summary")]
    healthy = bool(summary.get("bottleneck_healthy"))
    status = "pass" if healthy else "fail"
    return [row(status, "profile_bottlenecks", path, json.dumps(dict(summary), sort_keys=True))]


def audit_final_queue(full_run_root: Path) -> List[dict]:
    """Audit final-result watcher-chain health.

    Complexity: O(audit JSON size).
    """

    path = full_run_root / "final_queue_audit.json"
    obj = _read_json(path)
    if obj is None:
        return [row("warn", "final_queue_audit", path, "missing or invalid JSON")]
    summary = obj.get("summary")
    if not isinstance(summary, Mapping):
        return [row("warn", "final_queue_summary", path, "missing summary")]
    healthy = bool(summary.get("final_queue_healthy"))
    status = "pass" if healthy else "fail"
    return [row(status, "final_queue", path, json.dumps(dict(summary), sort_keys=True))]


def audit_paper_artifacts(full_run_root: Path) -> List[dict]:
    """Audit public-data, split and paper artifact readiness.

    Complexity: O(audit JSON size).
    """

    path = full_run_root / "paper_artifact_audit.json"
    obj = _read_json(path)
    if obj is None:
        return [row("fail", "paper_artifact_audit", path, "missing or invalid JSON")]
    summary = obj.get("summary")
    if not isinstance(summary, Mapping):
        return [row("fail", "paper_artifact_summary", path, "missing summary")]
    counts = summary.get("counts")
    fails = counts.get("fail", 1) if isinstance(counts, Mapping) else 1
    status = "pass" if int(fails) == 0 else "fail"
    return [row(status, "paper_artifact_failures", path, json.dumps(dict(summary), sort_keys=True))]


def audit_reproducibility_manifest(full_run_root: Path) -> List[dict]:
    """Audit reproducibility manifest presence and basic validity.

    Complexity: O(manifest JSON size).
    """

    path = full_run_root / "reproducibility_manifest.json"
    obj = _read_json(path)
    if obj is None:
        return [row("fail", "reproducibility_manifest", path, "missing or invalid JSON")]
    file_count = obj.get("file_count")
    files = obj.get("files")
    audit_summaries = obj.get("audit_summaries")
    ok = isinstance(file_count, int) and file_count > 0 and isinstance(files, list) and isinstance(audit_summaries, Mapping)
    status = "pass" if ok else "fail"
    return [row(status, "reproducibility_manifest", path, f"file_count={file_count!r}")]


def audit_coverage_gate(full_run_root: Path) -> List[dict]:
    """Audit persisted coverage gate evidence.

    Complexity: O(coverage audit JSON size).
    """

    path = full_run_root / "coverage_audit.json"
    obj = _read_json(path)
    if obj is None:
        return [row("fail", "coverage_gate", path, "missing or invalid JSON")]
    passed = bool(obj.get("passed"))
    percent = obj.get("percent_covered")
    threshold = obj.get("threshold")
    status = "pass" if passed else "fail"
    return [row(status, "coverage_gate", path, f"percent={percent!r}; threshold={threshold!r}")]


def audit_text_markers(path: Path, markers: Sequence[str], *, item_prefix: str) -> List[dict]:
    """Audit required documentation markers in one text file.

    Complexity: O(file bytes * marker count).
    """

    if not path.exists():
        return [row("fail", f"{item_prefix}:exists", path, "missing")]
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: List[dict] = []
    for marker in markers:
        status = "pass" if marker in text else "fail"
        rows.append(row(status, f"{item_prefix}:{marker}", path, "present" if status == "pass" else "missing"))
    return rows


def audit_final_results(full_run_root: Path) -> List[dict]:
    """Audit final result files needed before declaring the full goal complete.

    Final result files must be non-empty lists of completed metric rows produced
    by ``scripts/summarize_ablation_results.py``.  A mere file presence check is
    too weak: a premature summary can contain ``running_or_pending_json`` rows
    without F1/MCC values.  The required evidence is at least one ``status=ok``
    row per expected evaluation tier, with numeric F1/MCC metrics and positive
    sample counts, and no non-ok rows.

    Complexity: O(K * R), where K is expected result files and R rows/file.
    """

    rows: List[dict] = []
    for name in FINAL_RESULT_FILES:
        path = full_run_root / name
        validation = validate_final_result_file(path, missing_detail="missing; full-scale queue not complete")
        status = "pass" if validation.ready else "fail"
        rows.append(row(status, f"final_result:{name}", path, validation.detail))
    return rows


def summarize(rows: Iterable[Mapping[str, str]]) -> dict:
    """Summarize readiness rows.

    Complexity: O(N).
    """

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in rows:
        counts[str(item["status"])] += 1
    return {"counts": counts, "ready_for_goal_completion": counts["fail"] == 0}


def run_audit(project_root: Path, full_run_root: Path) -> dict:
    """Run all goal-readiness checks.

    Complexity: O(documentation bytes + audit JSON bytes).
    """

    rows: List[dict] = []
    rows.extend(audit_algorithm_doc(full_run_root))
    rows.extend(audit_runtime_health(full_run_root))
    rows.extend(audit_system_resources(full_run_root))
    rows.extend(audit_queue_progress(full_run_root))
    rows.extend(audit_cross_family_metrics(full_run_root))
    rows.extend(audit_profile_bottlenecks(full_run_root))
    rows.extend(audit_final_queue(full_run_root))
    rows.extend(audit_paper_artifacts(full_run_root))
    rows.extend(audit_reproducibility_manifest(full_run_root))
    rows.extend(audit_coverage_gate(full_run_root))
    rows.extend(audit_text_markers(project_root / "README.md", REQUIRED_README_MARKERS, item_prefix="README"))
    rows.extend(audit_text_markers(project_root / "docs/data_governance.md", REQUIRED_DATA_MARKERS, item_prefix="data_governance"))
    rows.extend(audit_final_results(full_run_root))
    summary = summarize(rows)
    return {"rows": rows, "summary": summary}


def write_markdown(result: Mapping[str, object], path: Path) -> None:
    """Write readiness audit rows as Markdown.

    Complexity: O(N).
    """

    summary = result["summary"]
    rows = result["rows"]
    lines = [
        "# ReactFlow Goal Readiness Audit",
        "",
        f"- ready_for_goal_completion: `{summary['ready_for_goal_completion']}`",
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

    Complexity: O(documentation bytes + audit JSON bytes).
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--full-run-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args(argv)

    result = run_audit(Path(args.project_root), Path(args.full_run_root))
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, Path(args.output_md))
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 1 if args.fail_if_not_ready and not result["summary"]["ready_for_goal_completion"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
