#!/usr/bin/env python3
"""Audit active post-training evaluation progress from streaming profiles.

Training progress can legitimately become ``None`` after ``epoch_total`` closes:
the same process is then evaluating named tiers and materializing final JSON.
This audit reads the tail of each active run's ``profile.jsonl``, finds the
latest ``eval_sample_total`` event, and estimates tier-level progress.

Formula: for latest event ``e=(tier, sample_index, seconds)``, processed samples
are ``n = sample_index + 1``.  The tier denominator ``N_t`` is the JSONL line
count of the corresponding split/cache file, and progress is ``n / N_t``.  ETA
uses the recent mean ``eval_sample_total`` seconds per sample in the same tier:
``ETA = max(0, N_t - n) * mean_seconds``.  Complexity:
O(A * (tail_bytes + tier_file_lines)), where A is the number of active rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence


RUNNING_STATUSES = {"running_or_pending_json"}
EVAL_TIER_ORDER = ("in_clan", "novel_clan", "archiveII", "PDB", "viral", "lncRNA", "human_mRNA")


def row(status: str, item: str, path: Optional[Path] = None, detail: str = "") -> dict:
    """Return one normalized audit row.

    Complexity: O(1).
    """

    if status not in {"pass", "warn", "fail"}:
        raise ValueError("status must be pass/warn/fail")
    return {"detail": detail, "item": item, "path": "" if path is None else str(path), "status": status}


def read_queue_rows(path: Path) -> List[Mapping[str, object]]:
    """Read ``current_queue_status.json`` rows.

    Complexity: O(queue JSON bytes).
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [item for item in payload if isinstance(item, Mapping)]


def active_rows(rows: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    """Return active queue rows.

    Complexity: O(R).
    """

    return [item for item in rows if str(item.get("status", "")) in RUNNING_STATUSES]


def tail_profile_events(path: Path, *, tail_bytes: int) -> List[Mapping[str, object]]:
    """Return parseable JSON events from the bounded profile tail.

    Reading only the tail makes the audit cheap even for hundreds of megabytes
    of profile events.  The first line may be partial, so malformed JSON is
    skipped.  Complexity: O(tail_bytes).
    """

    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - tail_bytes))
        text = handle.read().decode("utf-8", errors="replace")
    events: List[Mapping[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            events.append(payload)
    return events


def latest_eval_event(events: Sequence[Mapping[str, object]]) -> Optional[Mapping[str, object]]:
    """Return the latest ``eval_sample_total`` event with a tier.

    Complexity: O(E).
    """

    for event in reversed(events):
        if str(event.get("phase", "")) == "eval_sample_total" and event.get("tier") is not None:
            return event
    return None


def recent_eval_mean_seconds(events: Sequence[Mapping[str, object]], *, tier: str, max_events: int = 256) -> Optional[float]:
    """Return recent mean seconds per evaluated sample for ``tier``.

    Formula: average the latest at most ``max_events`` events satisfying
    ``phase='eval_sample_total'`` and matching ``tier``.  Complexity: O(E).
    """

    values: List[float] = []
    for event in reversed(events):
        if str(event.get("phase", "")) != "eval_sample_total" or str(event.get("tier", "")) != tier:
            continue
        seconds = event.get("seconds")
        if isinstance(seconds, (int, float)):
            values.append(float(seconds))
        if len(values) >= max_events:
            break
    if not values:
        return None
    return sum(values) / len(values)


def count_jsonl_rows(path: Path) -> Optional[int]:
    """Count non-empty rows in a JSONL file.

    Complexity: O(file lines).
    """

    if not path.exists() or path.stat().st_size == 0:
        return None
    count = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def tier_path(full_run_root: Path, tier: str, *, run_id: str = "") -> Path:
    """Return the expected JSONL source for an evaluation tier.

    Formula: MMseqs active runs map ``in_clan`` to ``test.jsonl`` and
    ``novel_clan`` to ``novel.jsonl`` in ``rfam_current_mmseqs_seed0``; named
    public tiers map to ``cache/<tier>.jsonl``.  Complexity: O(1).
    """

    split = "rfam_current_mmseqs_seed0" if "_mmseqs_" in run_id else "rfam_current_exact_seed0"
    if tier == "in_clan":
        return full_run_root / "splits" / split / "test.jsonl"
    if tier == "novel_clan":
        return full_run_root / "splits" / split / "novel.jsonl"
    return full_run_root / "cache" / f"{tier}.jsonl"


def tier_offset(full_run_root: Path, tier: str, *, run_id: str = "") -> Optional[int]:
    """Return global eval-sample offset before ``tier``.

    ReactFlow evaluation profiles use one global ``sample_index`` across all
    eval tiers in CLI order.  For tier ``t_k``, the tier-local index is therefore
    ``sample_index - sum_{q<k} N_q``.  If a preceding denominator is unavailable,
    return ``None`` and the caller can conservatively fall back to local indexing.
    Complexity: O(K * tier file lines), where K is the tier position.
    """

    offset = 0
    for name in EVAL_TIER_ORDER:
        if name == tier:
            return offset
        count = count_jsonl_rows(tier_path(full_run_root, name, run_id=run_id))
        if count is None:
            return None
        offset += count
    return 0


def run_dir_for_row(full_run_root: Path, queue_row: Mapping[str, object]) -> Path:
    """Return a run directory for one queue row.

    Complexity: O(1).
    """

    artifact = queue_row.get("artifact")
    if isinstance(artifact, str) and artifact:
        path = Path(artifact)
        if path.exists():
            return path
    return full_run_root / "runs" / str(queue_row.get("run_id", ""))


def audit_active_row(full_run_root: Path, queue_row: Mapping[str, object], *, tail_bytes: int) -> List[dict]:
    """Audit active evaluation progress for one run.

    Complexity: O(tail_bytes + tier file lines).
    """

    run_id = str(queue_row.get("run_id", "unknown_run"))
    run_dir = run_dir_for_row(full_run_root, queue_row)
    profile = run_dir / "profile.jsonl"
    events = tail_profile_events(profile, tail_bytes=tail_bytes)
    if not events:
        return [row("warn", f"run:{run_id}:eval_progress", profile, "profile tail missing or empty")]
    latest = latest_eval_event(events)
    if latest is None:
        return [row("warn", f"run:{run_id}:eval_progress", profile, "no eval_sample_total event in profile tail")]
    tier = str(latest.get("tier", ""))
    try:
        sample_index = int(latest.get("sample_index"))
    except (TypeError, ValueError):
        return [row("warn", f"run:{run_id}:eval_progress", profile, f"invalid sample_index={latest.get('sample_index')!r}")]
    total_path = tier_path(full_run_root, tier, run_id=run_id)
    total = count_jsonl_rows(total_path)
    if total is None or total <= 0:
        return [row("warn", f"run:{run_id}:eval_progress", total_path, f"missing tier denominator for tier={tier}")]
    offset = tier_offset(full_run_root, tier, run_id=run_id)
    if offset is not None and sample_index >= offset:
        tier_sample_index = sample_index - offset
    else:
        tier_sample_index = sample_index
        offset = None
    processed = min(total, tier_sample_index + 1)
    fraction = processed / float(total)
    mean_seconds = recent_eval_mean_seconds(events, tier=tier)
    eta = None if mean_seconds is None else max(0.0, total - processed) * mean_seconds
    detail = {
        "eta_seconds": eta,
        "mean_seconds_per_sample": mean_seconds,
        "processed": processed,
        "progress_fraction": fraction,
        "sample_index_global": sample_index,
        "tier": tier,
        "tier_offset": offset,
        "tier_sample_index": tier_sample_index,
        "total": total,
    }
    return [row("pass", f"run:{run_id}:eval_progress", profile, json.dumps(detail, sort_keys=True))]


def summarize(rows: Iterable[Mapping[str, str]]) -> dict:
    """Return summary counts and health.

    Complexity: O(N).
    """

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in rows:
        counts[str(item["status"])] += 1
    return {"counts": counts, "eval_progress_healthy": counts["fail"] == 0}


def run_audit(full_run_root: Path, *, queue_json: Optional[Path] = None, tail_bytes: int = 2_000_000) -> dict:
    """Run active eval-progress audit.

    Complexity: O(A * (tail_bytes + tier file lines)).
    """

    queue_path = queue_json or full_run_root / "current_queue_status.json"
    rows: List[dict] = []
    active = active_rows(read_queue_rows(queue_path))
    if not active:
        rows.append(row("pass", "active_eval_progress:no_active_runs", queue_path, "no running_or_pending_json rows"))
    for item in active:
        rows.extend(audit_active_row(full_run_root, item, tail_bytes=tail_bytes))
    return {"rows": rows, "summary": summarize(rows)}


def write_markdown(result: Mapping[str, object], path: Path) -> None:
    """Write active eval progress rows as Markdown.

    Complexity: O(N).
    """

    summary = result["summary"]
    lines = [
        "# ReactFlow Active Eval Progress Audit",
        "",
        f"- eval_progress_healthy: `{summary['eval_progress_healthy']}`",
        f"- counts: `{summary['counts']}`",
        "",
        "| Status | Item | Path | Detail |",
        "|---|---|---|---|",
    ]
    for item in result.get("rows", []):
        if not isinstance(item, Mapping):
            continue
        detail = str(item["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['status']} | {item['item']} | {item['path']} | {detail} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(A * (tail_bytes + tier file lines)).
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-root", required=True)
    parser.add_argument("--queue-json")
    parser.add_argument("--tail-bytes", type=int, default=2_000_000)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    result = run_audit(
        Path(args.full_run_root),
        queue_json=Path(args.queue_json) if args.queue_json else None,
        tail_bytes=args.tail_bytes,
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, Path(args.output_md))
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
