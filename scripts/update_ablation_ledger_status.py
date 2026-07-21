#!/usr/bin/env python3
"""Update the ablation ledger from full-run monitoring artifacts.

The full-run queue emits machine-readable JSON audits, while the experiment
ledger in ``docs/ablation_experiment_filled.md`` is the human-facing record.
This script keeps those two views synchronized by rewriting only the monitoring
rows whose values are derived directly from current artifacts.

Formula: let ``J`` be the set of audit JSON files under a full-run root and
``M`` be the Markdown ledger.  The updater computes a deterministic row map
``U = f(J)`` for the monitor, queue, resource, bottleneck and progress rows,
then replaces matching table rows in ``M`` by key.  Complexity: O(|J| + |M|)
because each JSON file and ledger line is read once.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence


def read_json(path: Path) -> object:
    """Read one JSON artifact.

    Formula: parse byte string ``B`` into JSON object ``x = parse(B)``.  The
    caller validates the expected shape because different audit artifacts use
    either mappings or row lists.  Complexity: O(file bytes).
    """

    return json.loads(path.read_text(encoding="utf-8"))


def require_mapping(obj: object, label: str) -> Mapping[str, object]:
    """Return ``obj`` as a mapping or raise a shape error.

    Formula: accept iff ``obj`` belongs to the mapping type family; otherwise
    the artifact cannot prove a ledger value and the update must fail loudly.
    Complexity: O(1).
    """

    if not isinstance(obj, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return obj


def require_rows(obj: object, label: str) -> list[Mapping[str, object]]:
    """Return ``obj`` as a list of mapping rows.

    Formula: accept iff every row ``r_i`` in list ``R`` is a mapping.  This
    preserves the one-row-per-run semantics of ``current_queue_status.json``.
    Complexity: O(N) rows.
    """

    if not isinstance(obj, list) or any(not isinstance(row, Mapping) for row in obj):
        raise TypeError(f"{label} must be a JSON row list")
    return list(obj)


def summary_counts(summary: Mapping[str, object]) -> tuple[int, int, int]:
    """Return ``(pass, warn, fail)`` counts from an audit summary.

    Formula: for counts mapping ``C``, extract
    ``(C_pass, C_warn, C_fail)`` with missing values interpreted as zero.
    Complexity: O(1).
    """

    counts = summary.get("counts")
    if not isinstance(counts, Mapping):
        return (0, 0, 0)
    return (int(counts.get("pass", 0)), int(counts.get("warn", 0)), int(counts.get("fail", 0)))


def first_queue_row(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Choose the queue row that should drive the active ledger snapshot.

    Formula: prefer the first row whose status is ``running_or_pending_json``;
    if no run is active, use the first row so completed metrics can still be
    reflected.  Complexity: O(R).
    """

    if not rows:
        raise ValueError("current_queue_status.json has no rows")
    for row in rows:
        if str(row.get("status", "")) == "running_or_pending_json":
            return row
    return rows[0]


def percent(value: object) -> str:
    """Format a fraction as a two-decimal percentage string.

    Formula: ``pct = 100 * value`` when ``value`` is numeric; missing values
    become ``"missing"`` so newly started active rows can enter the ledger before
    a monitor snapshot has emitted progress.  Complexity: O(1).
    """

    if value is None:
        return "missing"
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "missing"


def fixed(value: object, digits: int) -> str:
    """Format a numeric value with a fixed number of decimal places.

    Formula: ``round(value, digits)`` under Python's decimal string formatter.
    Missing values become ``"missing"``.  Complexity: O(1).
    """

    if value is None:
        return "missing"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "missing"


def find_audit_row(rows: object, item_suffix: str) -> Optional[Mapping[str, object]]:
    """Return the first audit row whose item ends with ``item_suffix``.

    Formula: scan rows ``r_i`` and return the first row satisfying
    ``str(r_i.item).endswith(item_suffix)``.  Complexity: O(N).
    """

    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("item", "")).endswith(item_suffix):
            return row
    return None


def find_run_audit_row(rows: object, run_id: str, item_suffix: str) -> Optional[Mapping[str, object]]:
    """Return the audit row for ``run_id`` and ``item_suffix`` when available.

    Formula: choose the first row satisfying
    ``item == 'run:' + run_id + ':' + item_suffix``; if none exists, fall back to
    suffix-only matching so older single-run audit files remain readable.
    Complexity: O(N).
    """

    target = f"run:{run_id}:{item_suffix}"
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping) and str(row.get("item", "")) == target:
                return row
    return find_audit_row(rows, item_suffix)


def detail_float(detail: str, key: str) -> Optional[float]:
    """Extract ``key=<float>`` from an audit detail string.

    Formula: use regex tokenization over ``detail`` and parse the first numeric
    capture as a float.  Complexity: O(len(detail)).
    """

    match = re.search(rf"{re.escape(key)}=([-+]?(?:\d+(?:\.\d*)?|\.\d+))", detail)
    return None if match is None else float(match.group(1))


def detail_token(detail: str, key: str) -> Optional[str]:
    """Extract a semicolon-delimited token value from an audit detail string.

    Formula: parse the first token matching ``key=value`` where value extends
    until ``;`` or string end, then strip whitespace.  Complexity:
    O(len(detail)).
    """

    match = re.search(rf"{re.escape(key)}=([^;]+)", detail)
    return None if match is None else match.group(1).strip()


def active_gpu_count(system_audit: Mapping[str, object]) -> Optional[int]:
    """Return the non-zero utilization GPU count from the resource audit.

    Formula: find row ``gpu:active_utilization`` and parse ``active_gpu_count``.
    Complexity: O(N) audit rows.
    """

    row = find_audit_row(system_audit.get("rows"), "gpu:active_utilization")
    if row is None:
        return None
    value = detail_float(str(row.get("detail", "")), "active_gpu_count")
    return None if value is None else int(value)


def training_process(system_audit: Mapping[str, object]) -> Optional[Mapping[str, object]]:
    """Return the Python child process for the active warm watcher.

    Formula: among process rows, select the descendant with command ``python``
    and a pidfile ending in ``warm_after_export_rfam_current_exact.pid``.  If
    multiple match, choose the largest RSS process because it is the training
    worker.  Complexity: O(P).
    """

    processes = system_audit.get("processes")
    if not isinstance(processes, list):
        return None
    matches = [
        proc
        for proc in processes
        if isinstance(proc, Mapping)
        and str(proc.get("command", "")) == "python"
        and str(proc.get("pidfile", "")).endswith("warm_after_export_rfam_current_exact.pid")
    ]
    if not matches:
        return None
    return max(matches, key=lambda proc: float(proc.get("rss_mib", 0.0)))


def pid_for_pidfile(system_audit: Mapping[str, object], suffix: str) -> Optional[int]:
    """Return the pid recorded for a watcher pidfile suffix.

    Formula: scan process records ``p_i`` and return ``pid`` for the first
    non-descendant process whose ``pidfile`` ends with ``suffix``.  Complexity:
    O(P).
    """

    processes = system_audit.get("processes")
    if not isinstance(processes, list):
        return None
    for proc in processes:
        if (
            isinstance(proc, Mapping)
            and str(proc.get("pidfile", "")).endswith(suffix)
            and str(proc.get("role", "")) == "pidfile"
            and isinstance(proc.get("pid"), int)
        ):
            return int(proc["pid"])
    return None


def parse_profile_fraction(profile_audit: Mapping[str, object], run_id: str) -> tuple[Optional[str], Optional[float]]:
    """Return ``(phase, fraction)`` for ``run_id`` from the bottleneck audit.

    Formula: locate ``run:<run_id>:slowest_phase_fraction`` and parse
    ``phase=p`` plus ``fraction=rho`` where ``rho = T_p / sum_q T_q``.
    Complexity: O(N + len(detail)).
    """

    row = find_run_audit_row(profile_audit.get("rows"), run_id, "slowest_phase_fraction")
    if row is None:
        return (None, None)
    detail = str(row.get("detail", ""))
    return (detail_token(detail, "phase"), detail_float(detail, "fraction"))


def parse_progress_window(queue_progress: Mapping[str, object], run_id: str) -> tuple[Optional[float], Optional[float], str]:
    """Return ``(delta, elapsed_seconds, note)`` for ``run_id``.

    Formula: parse ``delta`` and ``elapsed_seconds`` from
    ``run:<run_id>:progress_window`` detail, representing ``p_now - p_old`` and
    ``t_now - t_old``.  If the row is a warning whose detail starts with
    ``post_training_eval``, emit a human-readable note because zero training
    progress is expected after ``epoch_total`` closes and evaluation/final JSON
    writing begins.  Complexity: O(N + len(detail)).
    """

    row = find_run_audit_row(queue_progress.get("rows"), run_id, "progress_window")
    if row is None:
        return (None, None, "")
    detail = str(row.get("detail", ""))
    note = ""
    if str(row.get("status", "")) == "warn" and detail.startswith("post_training_eval"):
        note = "；progress_window warning 来自训练 profile 已闭合后的 eval/finalize 等待落盘阶段，不按训练 delta 判定卡死"
    return (detail_float(detail, "delta"), detail_float(detail, "elapsed_seconds"), note)


def summarize_queue_runs(rows: Sequence[Mapping[str, object]]) -> tuple[str, str]:
    """Return ``(completed_text, active_text)`` for the warm queue ledger row.

    Formula: group queue rows by ``run_id``.  For each group, count rows with
    ``status='ok'`` as completed evaluation tiers and count other rows as active
    or pending evidence.  The active text is driven by
    :func:`first_queue_row`, so the ledger reports the same run as the monitor.
    Complexity: O(R).
    """

    grouped: MutableMapping[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("run_id", "unknown_run")), []).append(row)

    completed_parts: list[str] = []
    for run_id in sorted(grouped):
        ok_rows = [row for row in grouped[run_id] if str(row.get("status", "")) == "ok"]
        if not ok_rows:
            continue
        pdb = next((row for row in ok_rows if str(row.get("tier", "")) == "PDB"), ok_rows[0])
        mean_f1 = pdb.get("mean_f1")
        metric_text = "mean_f1=missing" if mean_f1 is None else f"PDB mean_f1={float(mean_f1):.4f}"
        completed_parts.append(f"`{run_id}` {len(ok_rows)} tier ok ({metric_text})")

    active = first_queue_row(rows)
    if str(active.get("status", "")) == "ok":
        active_text = "当前没有 running_or_pending_json 行"
    else:
        active_text = (
            f"当前 active `{active.get('run_id', 'unknown_run')}` progress `{percent(active.get('progress_fraction'))}`, "
            f"`samples/s={fixed(active.get('samples_per_second', 0.0), 4)}`, slowest phase `{active.get('slowest_phase', '')}`"
        )
    completed_text = "暂无已完成 tier" if not completed_parts else "；".join(completed_parts)
    return (completed_text, active_text)


def build_row_updates(full_run_root: Path) -> Mapping[str, str]:
    """Build replacement Markdown table rows from current full-run artifacts.

    Formula: compute the ledger projection ``U = f(J)`` from queue, progress,
    resource and bottleneck audits.  Each output row is a pure function of the
    authoritative JSON artifacts, so repeated executions are idempotent until
    the underlying artifacts change.  Complexity: O(total JSON bytes + rows).
    """

    queue_rows = require_rows(read_json(full_run_root / "current_queue_status.json"), "current_queue_status")
    queue_row = first_queue_row(queue_rows)
    active_run_id = str(queue_row.get("run_id", "unknown_run"))
    queue_progress = require_mapping(read_json(full_run_root / "queue_progress_audit.json"), "queue_progress")
    profile_audit = require_mapping(read_json(full_run_root / "profile_bottleneck_audit.json"), "profile_bottleneck")
    system_audit = require_mapping(read_json(full_run_root / "system_resource_audit.json"), "system_resource")
    runtime_audit = require_mapping(read_json(full_run_root / "runtime_health_audit.json"), "runtime_health")
    paper_audit = require_mapping(read_json(full_run_root / "paper_artifact_audit.json"), "paper_artifact")

    progress_text = percent(queue_row.get("progress_fraction"))
    samples_text = fixed(queue_row.get("samples_per_second", 0.0), 4)
    stderr_size = int(queue_row.get("stderr_size", 0))
    slowest_phase = str(queue_row.get("slowest_phase", ""))
    run_status = str(queue_row.get("status", ""))

    queue_pass, queue_warn, queue_fail = summary_counts(require_mapping(queue_progress.get("summary", {}), "queue_progress.summary"))
    eta = queue_progress.get("summary", {}).get("min_estimated_remaining_seconds") if isinstance(queue_progress.get("summary"), Mapping) else None
    eta_text = "null" if eta is None else f"{float(eta):.1f}s"
    delta, elapsed, progress_window_note = parse_progress_window(queue_progress, active_run_id)
    delta_text = "missing" if delta is None else f"{delta:.6f}"
    elapsed_text = "missing" if elapsed is None else f"{elapsed:.1f}"

    profile_pass, profile_warn, profile_fail = summary_counts(require_mapping(profile_audit.get("summary", {}), "profile_bottleneck.summary"))
    profile_phase, fraction = parse_profile_fraction(profile_audit, active_run_id)
    phase_text = slowest_phase if profile_phase is None else profile_phase
    fraction_text = "missing" if fraction is None else f"{fraction:.4f}"

    system_summary = require_mapping(system_audit.get("summary", {}), "system_resource.summary")
    system_pass, system_warn, system_fail = summary_counts(system_summary)
    process_count = int(system_summary.get("process_count", 0))
    gpu_count = int(system_summary.get("gpu_count", 0))
    resource_healthy = str(bool(system_summary.get("resource_healthy", False))).lower()
    active_gpus = active_gpu_count(system_audit)
    active_gpu_text = "unknown" if active_gpus is None else str(active_gpus)
    proc = training_process(system_audit)
    proc_text = f"{active_run_id} 训练 Python 子进程暂未匹配"
    if proc is not None:
        proc_text = (
            f"{active_run_id} 训练 Python 子进程 `pid={int(proc.get('pid', 0))}` 继续运行，"
            f"`pcpu={float(proc.get('pcpu', 0.0)):.1f}`, `rss_mib={float(proc.get('rss_mib', 0.0)):.2f}`"
        )
    recovery_pid = pid_for_pidfile(system_audit, "warm_tail_recovery_after_watcher_exit.pid")
    recovery_pid_text = "unknown" if recovery_pid is None else str(recovery_pid)
    bottleneck_healthy = str(
        bool(require_mapping(profile_audit.get("summary", {}), "profile_bottleneck.summary").get("bottleneck_healthy", False))
    ).lower()
    progress_healthy = str(
        bool(require_mapping(queue_progress.get("summary", {}), "queue_progress.summary").get("progress_healthy", False))
    ).lower()
    runtime_summary = require_mapping(runtime_audit.get("summary", {}), "runtime_health.summary")
    runtime_pass, runtime_warn, runtime_fail = summary_counts(runtime_summary)
    runtime_healthy = str(bool(runtime_summary.get("healthy", False))).lower()
    audited_run_count = int(runtime_summary.get("audited_run_count", 1))
    paper_summary = require_mapping(paper_audit.get("summary", {}), "paper_artifact.summary")
    paper_pass, paper_warn, paper_fail = summary_counts(paper_summary)
    ok_for_paper_table = str(bool(paper_summary.get("ok_for_paper_table", False))).lower()
    completed_text, active_text = summarize_queue_runs(queue_rows)
    warm_result = full_run_root / "warm_rfam_current_exact_results.json"
    warm_state = "completed" if warm_result.exists() and warm_result.stat().st_size > 0 else "active"

    return {
        "RF-A1-warm full-data": (
            f"| RF-A1-warm full-data | {warm_state} | "
            f"full-run 队列状态：{completed_text}；{active_text} "
            "| 已生成的 final result 由 `audit_final_queue.py` 内容契约证明；当前未完成 active 行继续进入 queue/progress/cross-family audits |"
        ),
        "run monitor snapshot": (
            "| run monitor snapshot | active | "
            "`scripts/monitor_reactflow_run.py` 已同步到服务器并生成当前 active run 的 `monitor_snapshot.json` 与 `.md`；"
            f"当前 `{active_run_id}` snapshot: progress `{progress_text}`, stderr `{stderr_size} bytes`, slowest phase `{slowest_phase}` "
            "| 后续巡检统一使用该脚本，避免临时解析日志 |"
        ),
        "current queue status": (
            "| current queue status | active | "
            "`current_queue_status.json/md/svg` 由统一刷新脚本自动生成；刷新脚本支持多 `--glob`，会同时覆盖 exact、MMseqs 和 RF-CF 队列；"
            f"当前 active 行 `{active_run_id}` 显示 `{run_status}`, `samples/s={samples_text}`, progress `{progress_text}`；"
            "SVG 会在存在 `status=ok` 且有 `mean_f1` 的行时自动生成 mean-F1 bar chart，否则生成“暂无完成指标”占位图 "
            "| 后续 RF-CF、capacity 和 multi-seed run 目录出现后会自动进入同一张状态表 |"
        ),
        "paper artifact audit snapshot": (
            "| paper artifact audit snapshot | active | "
            "`paper_artifact_audit.json/md` 由统一刷新脚本生成，检查 public cache、MMseqs metadata、leakage-safe split、run stderr/profile/checkpoint/metrics "
            f"| 当前 `ok_for_paper_table={ok_for_paper_table}`, `pass={paper_pass}`, `warn={paper_warn}`, `fail={paper_fail}`；"
            "paper artifact 完整性通过后，最终论文 claim 仍需 cross-family claim gate 和多 seed 统计共同支撑 |"
        ),
        "runtime health audit": (
            "| runtime health audit | active | "
            "`runtime_health_audit.json/md` 已刷新；completed run 使用 `stdout.json + training_checkpoint.json + tiers` 作为完成证据，active run 仍要求 profile heartbeat "
            f"| 当前 `healthy={runtime_healthy}`, `audited_run_count={audited_run_count}`, `pass={runtime_pass}`, `warn={runtime_warn}`, `fail={runtime_fail}`；"
            "只有未完成阶段的 watcher pidfile 纳入 liveness gate；已完成阶段改由 final-result 内容契约证明 |"
        ),
        "system resource audit": (
            "| system resource audit | active | "
            "新增 `scripts/audit_system_resources.py`，固化 `nvidia-smi` GPU 利用率/显存和 watcher pidfile 的递归子进程 CPU/RSS 快照，输出 `system_resource_audit.json/md` "
            f"| 当前 `resource_healthy={resource_healthy}`, `pass={system_pass}`, `warn={system_warn}`, `fail={system_fail}`, `process_count={process_count}`；"
            f"服务器有 {gpu_count} 张 RTX 4090，{active_gpu_text} 张 GPU 非零 utilization；{proc_text}，"
            f"tracked recovery pid `{recovery_pid_text}`；该快照用于后续 OOM、CPU/I/O 瓶颈与 batch-size 调整证据 |"
        ),
        "profile bottleneck audit": (
            "| profile bottleneck audit | active | "
            "新增 `scripts/audit_profile_bottlenecks.py`，读取 active run 的 `monitor_snapshot.json` 并计算 phase 占比 `rho_p=T_p/sum_qT_q`，同时检查 `frozen_batch_prefetch` 是否出现在 profile 中 "
            f"| 当前 `profile_bottleneck_audit.json/md` 显示 `bottleneck_healthy={bottleneck_healthy}`, `pass={profile_pass}`, `warn={profile_warn}`, `fail={profile_fail}`；"
            f"`{active_run_id}` 的 `{phase_text}` 占比 `{fraction_text}`；"
            "该 audit 同时记录 `frozen_batch_prefetch` 是否出现，用于量化预取收益 |"
        ),
        "queue progress audit": (
            "| queue progress audit | active | "
            "新增 `scripts/audit_queue_progress.py`，把 `current_queue_status.json` 追加到 `logs/current_queue_status_history.jsonl`，并审计运行中任务的 progress delta、throughput floor 和趋势 ETA "
            f"| 远程 `queue_progress_audit.json/md` 已生成；当前 `progress_healthy={progress_healthy}`, `pass={queue_pass}`, `warn={queue_warn}`, `fail={queue_fail}`；"
            f"`{active_run_id}` progress `{progress_text}`, `samples/s={samples_text}`；本轮窗口 `delta={delta_text}`, `elapsed_seconds={elapsed_text}`，"
            f"趋势估计剩余约 `{eta_text}`{progress_window_note} |"
        ),
    }


def replace_ledger_rows(text: str, updates: Mapping[str, str]) -> tuple[str, Mapping[str, bool]]:
    """Replace keyed Markdown table rows and report which keys were found.

    Formula: for each line ``l`` in the ledger, if it starts with
    ``"| key |"`` for some update key, replace ``l`` with ``updates[key]``.
    Complexity: O(L * K) for L lines and K update keys; K is five here.
    """

    found: MutableMapping[str, bool] = {key: False for key in updates}
    out_lines: list[str] = []
    for line in text.splitlines():
        replacement = None
        for key, value in updates.items():
            if line.startswith(f"| {key} |"):
                replacement = value
                found[key] = True
                break
        out_lines.append(replacement if replacement is not None else line)
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(out_lines) + suffix, found


def update_ledger(ledger_path: Path, full_run_root: Path) -> Mapping[str, object]:
    """Update ``ledger_path`` in place from ``full_run_root`` artifacts.

    Formula: ``M' = replace(M, f(J))`` where ``M`` is the original Markdown
    ledger and ``J`` is the current audit JSON set.  Complexity: O(|J| + |M|).
    """

    updates = build_row_updates(full_run_root)
    original = ledger_path.read_text(encoding="utf-8")
    updated, found = replace_ledger_rows(original, updates)
    missing = [key for key, ok in found.items() if not ok]
    if missing:
        raise ValueError(f"ledger is missing rows: {', '.join(missing)}")
    ledger_path.write_text(updated, encoding="utf-8")
    return {"updated_rows": sorted(updates), "changed": updated != original}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Complexity: O(|J| + |M|), delegated to :func:`update_ledger`.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="docs/ablation_experiment_filled.md")
    parser.add_argument("--full-run-root", required=True)
    args = parser.parse_args(argv)

    result = update_ledger(Path(args.ledger), Path(args.full_run_root))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
