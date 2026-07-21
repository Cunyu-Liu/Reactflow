#!/usr/bin/env python3
"""Summarize ReactFlow ablation artifacts into Markdown, JSON and SVG.

Each run directory may contain several independent artifacts:

* ``stdout.json`` / ``eval_summary*.json``: final evaluation metrics.
* ``profile.summary.json``: closed-run timing aggregates.
* ``monitor_snapshot.json``: active-run progress, throughput and ETA.
* ``training_checkpoint.json``: model/config/history provenance.
* ``stderr.log``: failure evidence.

The summary table intentionally separates metric availability from run state.
A long-running run can be reported as ``running`` with profile throughput even
before its final F1/MCC rows exist; a completed run with recovered stdout can be
reported as ``ok`` without manual copy-paste.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple


SUMMARY_CANDIDATES: Tuple[str, ...] = (
    "eval_summary.json",
    "eval_summary.recovered.json",
    "stdout.json",
    "stdout.recovered.json",
)


def parse_first_json(text: str) -> Mapping[str, object]:
    """Parse the first JSON object embedded in ``text``.

    Some historical full-run artifacts were captured through shell wrappers that
    printed log text before the JSON payload.  The recovery rule is therefore:
    discard everything before the first ``"{"`` and parse the remainder as one
    JSON object.  This is deliberately strict after the first brace, so malformed
    recovered files are skipped rather than silently partially parsed.

    Complexity: O(B) for B input characters.
    """

    start = text.find("{")
    if start < 0:
        raise ValueError("text does not contain a JSON object")
    return json.loads(text[start:])


def load_run_summary(run_dir: Path) -> Optional[Mapping[str, object]]:
    """Load the first available final-metrics JSON from ``run_dir``.

    Candidate files are ordered from most explicit to most generic:
    ``eval_summary.json``, ``eval_summary.recovered.json``, ``stdout.json`` and
    ``stdout.recovered.json``.  Empty or malformed files are ignored so a run
    directory can still be summarized as running/failed using monitor/profile
    artifacts.

    Complexity: O(C * B), with C fixed at four candidates and B the size of the
    first non-empty candidate that parses.
    """

    for name in SUMMARY_CANDIDATES:
        path = run_dir / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            return parse_first_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def load_json_file(path: Path) -> Optional[Mapping[str, object]]:
    """Load a JSON object from ``path`` when it exists and parses.

    Non-object JSON values are treated as absent because downstream table rows
    expect key/value metadata.  Complexity: O(B), where B is file size.
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return obj if isinstance(obj, Mapping) else None


def run_artifacts(run_dir: Path) -> dict:
    """Return compact artifact metadata for ``run_dir``.

    Complexity: O(1) file stats plus bounded stderr tail.
    """

    stderr = run_dir / "stderr.log"
    profile = load_json_file(run_dir / "profile.summary.json")
    monitor = load_json_file(run_dir / "monitor_snapshot.json")
    checkpoint = load_json_file(run_dir / "training_checkpoint.json")
    stderr_size = stderr.stat().st_size if stderr.exists() else 0
    stderr_tail = stderr.read_text(encoding="utf-8", errors="replace")[-1000:] if stderr_size else ""
    return {
        "checkpoint_present": checkpoint is not None,
        "monitor": monitor,
        "profile": profile,
        "profile_jsonl_present": (run_dir / "profile.jsonl").exists(),
        "stderr_size": stderr_size,
        "stderr_tail": stderr_tail,
    }


def _phase_seconds(profile: Optional[Mapping[str, object]], phase: str) -> Optional[float]:
    """Return total profiled seconds for one phase.

    ``TrainingProfiler.close`` writes ``{"phases": {phase: ...}}``.  This helper
    centralizes the nested type checks so a partially written or hand-edited
    profile summary degrades to ``None`` instead of raising.

    Complexity: O(1).
    """

    if not profile:
        return None
    phases = profile.get("phases")
    if not isinstance(phases, Mapping):
        return None
    metrics = phases.get(phase)
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get("total_seconds")
    return None if value is None else float(value)


def _profile_total_seconds(profile: Optional[Mapping[str, object]], monitor: Optional[Mapping[str, object]]) -> Optional[float]:
    """Prefer wall-clock epoch total, then monitor/profile aggregate seconds.

    Closed runs report ``epoch_total`` and that value is closest to wall-clock
    training time.  Active runs have no closed summary, so their monitor snapshot
    contributes ``total_profiled_seconds`` instead.

    Complexity: O(1).
    """

    epoch_total = _phase_seconds(profile, "epoch_total")
    if epoch_total is not None:
        return epoch_total
    for source in (monitor, profile):
        if source and source.get("total_profiled_seconds") is not None:
            return float(source["total_profiled_seconds"])
    return None


def _sample_count_from_profile(profile: Optional[Mapping[str, object]]) -> Optional[int]:
    """Infer the number of profiled samples from phase event counts.

    The per-sample phases ``model_forward``, ``path_sample_features`` and
    ``projection_f1`` are logged once per sample in the main training loop.  We
    check them in that order to recover throughput even when the final epoch
    summary is unavailable.

    Complexity: O(1), because the candidate phase list is fixed.
    """

    if not profile:
        return None
    phases = profile.get("phases")
    if not isinstance(phases, Mapping):
        return None
    for phase in ("model_forward", "path_sample_features", "projection_f1"):
        metrics = phases.get(phase)
        if isinstance(metrics, Mapping) and metrics.get("count") is not None:
            return int(metrics["count"])
    return None


def _throughput(profile: Optional[Mapping[str, object]], monitor: Optional[Mapping[str, object]]) -> Optional[float]:
    """Return samples per profiled second when available.

    Monitor snapshots already compute this online.  For closed runs without a
    monitor, we compute ``samples / seconds`` from profile aggregates.

    Complexity: O(1).
    """

    if monitor and monitor.get("samples_per_second") is not None:
        return float(monitor["samples_per_second"])
    seconds = _profile_total_seconds(profile, monitor)
    samples = _sample_count_from_profile(profile)
    if seconds and samples:
        return samples / seconds
    return None


def _slowest_phase(profile: Optional[Mapping[str, object]], monitor: Optional[Mapping[str, object]]) -> str:
    """Return the slowest phase label from monitor/profile metadata.

    ``slowest_step_phase`` excludes aggregate phases such as ``epoch_total`` and
    is preferred.  Falling back to ``slowest_phase`` keeps active monitor
    snapshots usable.

    Complexity: O(1).
    """

    for source in (monitor, profile):
        if not source:
            continue
        slowest = source.get("slowest_step_phase") or source.get("slowest_phase")
        if isinstance(slowest, Mapping):
            return str(slowest.get("phase") or "")
    return ""


def _progress(monitor: Optional[Mapping[str, object]]) -> Optional[float]:
    """Return active-run progress fraction from a monitor snapshot.

    Complexity: O(1).
    """

    if monitor and monitor.get("progress_fraction") is not None:
        return float(monitor["progress_fraction"])
    return None


def _status(summary: Optional[Mapping[str, object]], artifacts: Mapping[str, object]) -> str:
    """Classify a run directory using final metrics and failure evidence.

    Status precedence is:

    1. ``ok`` when final metrics exist.
    2. ``failed_or_stderr`` when stderr has content and no metrics exist.
    3. ``running_or_pending_json`` when profile/monitor artifacts exist.
    4. ``missing_json`` otherwise.

    Complexity: O(1).
    """

    if summary is not None:
        return "ok"
    if artifacts.get("stderr_size"):
        return "failed_or_stderr"
    if artifacts.get("monitor") is not None or artifacts.get("profile_jsonl_present"):
        return "running_or_pending_json"
    return "missing_json"


def _base_row(run_dir: Path, status: str, artifacts: Mapping[str, object]) -> dict:
    """Build fields shared by metric and non-metric rows.

    These fields describe run cost/provenance rather than per-tier accuracy:
    throughput, progress, slowest phase, checkpoint presence and stderr state.

    Complexity: O(1).
    """

    profile = artifacts.get("profile") if isinstance(artifacts.get("profile"), Mapping) else None
    monitor = artifacts.get("monitor") if isinstance(artifacts.get("monitor"), Mapping) else None
    return {
        "artifact": str(run_dir),
        "checkpoint_present": artifacts.get("checkpoint_present", False),
        "profile_seconds": _profile_total_seconds(profile, monitor),
        "progress_fraction": _progress(monitor),
        "run_id": run_dir.name,
        "samples_per_second": _throughput(profile, monitor),
        "slowest_phase": _slowest_phase(profile, monitor),
        "status": status,
        "stderr_size": artifacts.get("stderr_size", 0),
        "stderr_tail": artifacts.get("stderr_tail", ""),
    }


def collect_rows(run_dirs: Iterable[Path]) -> List[dict]:
    """Collect one output row per run/tier.

    A completed evaluation contributes one row per tier in ``summary["tiers"]``.
    A running or failed run contributes a single row with empty tier and null
    metric fields, preserving operational state in the same table.

    Complexity: O(R * (C + T)), where R is run count, C is the fixed artifact
    candidate count and T is the number of evaluation tiers.
    """

    rows: List[dict] = []
    for run_dir in sorted(run_dirs):
        summary = load_run_summary(run_dir)
        artifacts = run_artifacts(run_dir)
        status = _status(summary, artifacts)
        base = _base_row(run_dir, status, artifacts)
        if summary is None:
            rows.append({**base, "tier": "", "count": None, "mean_f1": None, "micro_f1": None, "mean_mcc": None, "micro_mcc": None})
            continue
        for tier, metrics in sorted(dict(summary.get("tiers") or {}).items()):
            if not isinstance(metrics, Mapping):
                continue
            rows.append(
                {
                    **base,
                    "tier": tier,
                    "count": metrics.get("count"),
                    "mean_f1": metrics.get("mean_f1"),
                    "micro_f1": metrics.get("micro_f1"),
                    "mean_mcc": metrics.get("mean_mcc"),
                    "micro_mcc": metrics.get("micro_mcc"),
                }
            )
    return rows


def write_markdown(rows: List[dict], path: Path, *, title: str) -> None:
    """Write the ablation table as Markdown.

    The table includes both scientific metrics (F1/MCC) and engineering cost
    metrics (seconds, samples/s, progress, slowest phase).  This matches the
    paper checklist requirement that effect size and runtime evidence be
    reported together.

    Complexity: O(R) for R rows.
    """

    lines = [
        f"# {title}",
        "",
        "| Run ID | Status | Tier | Count | Mean F1 | Micro F1 | Mean MCC | Micro MCC | Seconds | Samples/s | Progress | Slowest phase | Checkpoint | Artifact |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        progress = row.get("progress_fraction")
        lines.append(
            "| {run_id} | {status} | {tier} | {count} | {mean_f1} | {micro_f1} | {mean_mcc} | {micro_mcc} | {seconds} | {sps} | {progress} | {slowest} | {checkpoint} | {artifact} |".format(
                run_id=row.get("run_id", ""),
                status=row.get("status", ""),
                tier=row.get("tier", ""),
                count="" if row.get("count") is None else row.get("count"),
                mean_f1="" if row.get("mean_f1") is None else row.get("mean_f1"),
                micro_f1="" if row.get("micro_f1") is None else row.get("micro_f1"),
                mean_mcc="" if row.get("mean_mcc") is None else row.get("mean_mcc"),
                micro_mcc="" if row.get("micro_mcc") is None else row.get("micro_mcc"),
                seconds="" if row.get("profile_seconds") is None else f"{float(row['profile_seconds']):.2f}",
                sps="" if row.get("samples_per_second") is None else f"{float(row['samples_per_second']):.4f}",
                progress="" if progress is None else f"{100.0 * float(progress):.2f}%",
                slowest=row.get("slowest_phase", ""),
                checkpoint="yes" if row.get("checkpoint_present") else "no",
                artifact=row.get("artifact", ""),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg(rows: List[dict], path: Path, *, title: str) -> None:
    """Render a deterministic SVG bar chart of mean F1 rows.

    Only rows with ``status == "ok"`` and a numeric ``mean_f1`` are plotted;
    running/failed rows remain in JSON/Markdown.  If no completed metric rows
    are available yet, a deterministic placeholder SVG is still written so
    automation can distinguish "no completed metrics yet" from "chart writer
    failed".  Bar lengths are linearly normalized by the maximum plotted mean
    F1.

    Complexity: O(R) for R rows.
    """

    metric_rows = [row for row in rows if row.get("status") == "ok" and row.get("mean_f1") is not None]
    width = 1200
    path.parent.mkdir(parents=True, exist_ok=True)
    if not metric_rows:
        status_counts: dict[str, int] = {}
        for row in rows:
            status = str(row.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        details = ", ".join(f"{name}={count}" for name, count in sorted(status_counts.items())) or "no rows"
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="220" viewBox="0 0 {width} 220">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{escape(title)}</text>',
            '<rect x="24" y="64" width="1152" height="108" rx="12" fill="#f5f7fb" stroke="#d8deea"/>',
            '<text x="48" y="104" font-family="Arial" font-size="17" fill="#2f3b52">No completed metric rows yet.</text>',
            '<text x="48" y="134" font-family="Arial" font-size="14" fill="#5d6b82">JSON and Markdown contain active-run progress; this chart will populate after eval_summary/stdout metrics exist.</text>',
            f'<text x="48" y="158" font-family="Arial" font-size="13" fill="#5d6b82">Rows: {len(rows)}; statuses: {escape(details)}</text>',
            "</svg>",
        ]
        path.write_text("\n".join(parts), encoding="utf-8")
        return
    height = max(260, 38 * len(metric_rows) + 80)
    max_value = max(float(row["mean_f1"]) for row in metric_rows) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="30" font-family="Arial" font-size="20" font-weight="700">{escape(title)}</text>',
    ]
    y = 64
    for row in metric_rows:
        label = f"{row['run_id']} / {row['tier']}"
        value = float(row["mean_f1"])
        bar = 720 * value / max_value
        parts.append(f'<text x="24" y="{y + 16}" font-family="Arial" font-size="13">{escape(label)}</text>')
        parts.append(f'<rect x="400" y="{y}" width="{bar:.1f}" height="22" fill="#4f7cff"/>')
        parts.append(f'<text x="{410 + bar:.1f}" y="{y + 16}" font-family="Arial" font-size="13">{value:.4f}</text>')
        y += 38
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    """CLI entry point for ablation summarization.

    Complexity is dominated by :func:`collect_rows`; writing JSON, Markdown and
    SVG is linear in the collected row count.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="directory containing run subdirectories")
    parser.add_argument(
        "--glob",
        action="append",
        default=None,
        help="run directory glob under --run-root; may be repeated",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-svg", required=True)
    parser.add_argument("--title", default="ReactFlow ablation results")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    patterns = args.glob or ["*"]
    run_dirs = sorted({path for pattern in patterns for path in run_root.glob(pattern) if path.is_dir()})
    rows = collect_rows(run_dirs)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(rows, Path(args.output_md), title=args.title)
    write_svg(rows, Path(args.output_svg), title=args.title)
    print(json.dumps({"rows": len(rows), "runs": len(run_dirs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
