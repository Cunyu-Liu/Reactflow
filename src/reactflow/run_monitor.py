"""Progress monitoring utilities for long ReactFlow training runs.

The full eFold/Rfam experiments write streaming ``profile.jsonl`` events while
training.  This module turns that append-only trace into an auditable progress
snapshot: how many samples have been processed, which phase dominates runtime,
and a coarse ETA based on profiled wall-clock seconds.

The estimator is intentionally simple.  For event ``e`` with duration
``seconds_e`` and optional ``sample_index_e``, we aggregate

    T_phase = sum_{e: phase_e=phase} seconds_e,
    n_phase = count(e: phase_e=phase),
    processed = 1 + max_e sample_index_e.

Given a known dataset size ``N``, progress is ``processed / N`` and throughput
is ``processed / sum_e seconds_e``.  The ETA is therefore

    ETA = (N - processed) / throughput.

Because the profiler is written from a sequential training loop, the sum of
phase durations is a useful lower-noise proxy for elapsed training time.  It is
not used for final benchmark timing; final timing still comes from closed
``*.summary.json`` artifacts.

Complexity: O(E) time and O(P) memory for E profile events and P phases.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union


@dataclass(frozen=True)
class PhaseStats:
    """Aggregate timing statistics for one profiler phase.

    Complexity: O(1) summary storage.
    """

    count: int
    total_seconds: float
    max_seconds: float

    @property
    def mean_seconds(self) -> float:
        """Return ``total_seconds / count`` with the empty case guarded.

        Complexity: O(1).
        """

        return 0.0 if self.count == 0 else self.total_seconds / self.count

    def to_json_obj(self) -> Dict[str, float]:
        """Return a JSON-serializable representation.

        Complexity: O(1).
        """

        return {
            "count": self.count,
            "max_seconds": self.max_seconds,
            "mean_seconds": self.mean_seconds,
            "total_seconds": self.total_seconds,
        }


def _iter_profile_events(path: Path) -> Iterable[Mapping[str, object]]:
    """Yield valid JSON profile events from ``path``.

    Malformed trailing lines can happen if a monitor reads while another process
    is appending.  They are skipped instead of failing the whole snapshot.

    Complexity: O(E).
    """

    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, Mapping):
                yield event


def summarize_profile(
    profile_path: Union[str, Path],
    *,
    total_samples: Optional[int] = None,
    stderr_path: Optional[Union[str, Path]] = None,
    stderr_tail_bytes: int = 2000,
) -> Dict[str, object]:
    """Summarize a streaming ``profile.jsonl`` file.

    Args:
        profile_path: profiler JSONL path.
        total_samples: optional train-set size.  When present, progress and ETA
            are included.
        stderr_path: optional stderr log to attach as a small tail.
        stderr_tail_bytes: maximum stderr bytes to include.

    Complexity: O(E + S), where S is the bounded stderr tail length.
    """

    path = Path(profile_path)
    phase_totals: Dict[str, Tuple[int, float, float]] = {}
    latest_event: Optional[Mapping[str, object]] = None
    max_sample_index: Optional[int] = None
    total_profiled_seconds = 0.0
    events = 0

    for event in _iter_profile_events(path):
        latest_event = dict(event)
        events += 1
        phase = str(event.get("phase", "unknown"))
        seconds = float(event.get("seconds") or 0.0)
        count, total, max_seconds = phase_totals.get(phase, (0, 0.0, 0.0))
        phase_totals[phase] = (count + 1, total + seconds, max(max_seconds, seconds))
        total_profiled_seconds += seconds
        sample_index = event.get("sample_index")
        if sample_index is not None:
            try:
                as_int = int(sample_index)
            except (TypeError, ValueError):
                continue
            max_sample_index = as_int if max_sample_index is None else max(max_sample_index, as_int)

    phases = {
        phase: PhaseStats(count=count, total_seconds=total, max_seconds=max_seconds).to_json_obj()
        for phase, (count, total, max_seconds) in sorted(phase_totals.items())
    }
    ranked = [
        {"phase": phase, **metrics}
        for phase, metrics in sorted(
            phases.items(),
            key=lambda item: float(item[1]["total_seconds"]),
            reverse=True,
        )
    ]
    processed = None if max_sample_index is None else max_sample_index + 1
    samples_per_second = None
    progress_fraction = None
    eta_seconds = None
    if processed is not None and total_profiled_seconds > 0:
        samples_per_second = processed / total_profiled_seconds
    if processed is not None and total_samples:
        progress_fraction = min(1.0, processed / float(total_samples))
        if samples_per_second and samples_per_second > 0:
            eta_seconds = max(0.0, (total_samples - processed) / samples_per_second)

    stderr_tail = ""
    stderr_size = None
    if stderr_path is not None:
        err = Path(stderr_path)
        if err.exists():
            stderr_size = err.stat().st_size
            if stderr_tail_bytes > 0 and stderr_size:
                with err.open("rb") as handle:
                    handle.seek(max(0, stderr_size - stderr_tail_bytes))
                    stderr_tail = handle.read().decode("utf-8", errors="replace")

    return {
        "eta_seconds": eta_seconds,
        "events": events,
        "latest_event": latest_event,
        "phases": phases,
        "phases_by_total_seconds": ranked,
        "processed_samples": processed,
        "profile_path": str(path),
        "profile_size_bytes": path.stat().st_size if path.exists() else 0,
        "progress_fraction": progress_fraction,
        "samples_per_second": samples_per_second,
        "slowest_phase": ranked[0] if ranked else None,
        "stderr_size_bytes": stderr_size,
        "stderr_tail": stderr_tail,
        "total_profiled_seconds": total_profiled_seconds,
        "total_samples": total_samples,
    }


def write_monitor_markdown(summary: Mapping[str, object], path: Union[str, Path]) -> None:
    """Write a compact Markdown monitor report.

    Complexity: O(P), where P is the number of phases.
    """

    progress = summary.get("progress_fraction")
    progress_text = "n/a" if progress is None else f"{100.0 * float(progress):.2f}%"
    eta = summary.get("eta_seconds")
    eta_text = "n/a" if eta is None else _format_seconds(float(eta))
    lines = [
        "# ReactFlow Run Monitor",
        "",
        f"- profile: `{summary.get('profile_path')}`",
        f"- processed: `{summary.get('processed_samples')}` / `{summary.get('total_samples')}`",
        f"- progress: `{progress_text}`",
        f"- ETA: `{eta_text}`",
        f"- stderr bytes: `{summary.get('stderr_size_bytes')}`",
        "",
        "| Phase | Count | Total seconds | Mean seconds | Max seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("phases_by_total_seconds") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {phase} | {count} | {total:.6f} | {mean:.6f} | {maxv:.6f} |".format(
                phase=row.get("phase", ""),
                count=int(row.get("count", 0)),
                total=float(row.get("total_seconds", 0.0)),
                mean=float(row.get("mean_seconds", 0.0)),
                maxv=float(row.get("max_seconds", 0.0)),
            )
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_seconds(seconds: float) -> str:
    """Format seconds as a compact ``HH:MM:SS`` string."""

    rounded = max(0, int(round(seconds)))
    hours, rem = divmod(rounded, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"
