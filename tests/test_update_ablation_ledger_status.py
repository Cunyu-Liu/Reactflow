import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "update_ablation_ledger_status.py"
    spec = importlib.util.spec_from_file_location("update_ablation_ledger_status", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_artifacts(full: Path) -> None:
    _write_json(
        full / "current_queue_status.json",
        [
            {
                "progress_fraction": 0.7621328889814621,
                "run_id": "RF-A1-warm",
                "samples_per_second": 4.286553567457686,
                "slowest_phase": "path_sample_features",
                "status": "running_or_pending_json",
                "stderr_size": 0,
            }
        ],
    )
    _write_json(
        full / "runtime_health_audit.json",
        {
            "rows": [],
            "summary": {
                "audited_run_count": 1,
                "counts": {"fail": 0, "pass": 10, "warn": 0},
                "healthy": True,
            },
        },
    )
    _write_json(
        full / "paper_artifact_audit.json",
        {
            "rows": [],
            "summary": {
                "counts": {"fail": 0, "pass": 21, "warn": 2},
                "ok_for_paper_table": True,
            },
        },
    )
    _write_json(
        full / "queue_progress_audit.json",
        {
            "rows": [
                {"detail": "rows=1", "item": "history_append", "status": "pass"},
                {"detail": "snapshots=46", "item": "history_exists", "status": "pass"},
                {"detail": "progress=76.21%", "item": "run:RF-A1:progress_value", "status": "pass"},
                {"detail": "samples_per_second=4.2866; min=0.1000", "item": "run:RF-A1:throughput", "status": "pass"},
                {
                    "detail": "delta=0.006888; elapsed_seconds=1005.0; min_delta=0.000100",
                    "item": "run:RF-A1:progress_window",
                    "status": "pass",
                },
            ],
            "summary": {
                "counts": {"fail": 0, "pass": 5, "warn": 0},
                "min_estimated_remaining_seconds": 34706.78263370852,
                "progress_healthy": True,
                "trend_count": 1,
            },
        },
    )
    _write_json(
        full / "profile_bottleneck_audit.json",
        {
            "rows": [
                {"detail": "events=100", "item": "run:RF-A1:profile_events", "status": "pass"},
                {"detail": "total=10.0", "item": "run:RF-A1:profile_total_seconds", "status": "pass"},
                {
                    "detail": "phase=path_sample_features; fraction=0.8048; max=0.7500",
                    "item": "run:RF-A1:slowest_phase_fraction",
                    "status": "warn",
                },
                {
                    "detail": "phase=path_sample_features; fraction=0.8048; max=0.5000",
                    "item": "run:RF-A1:target_phase_fraction",
                    "status": "warn",
                },
                {
                    "detail": "phase=frozen_batch_prefetch missing; active run may predate optimization",
                    "item": "run:RF-A1:optimization_phase_present",
                    "status": "warn",
                },
            ],
            "summary": {
                "audited_run_count": 1,
                "bottleneck_healthy": True,
                "counts": {"fail": 0, "pass": 2, "warn": 3},
            },
        },
    )
    _write_json(
        full / "system_resource_audit.json",
        {
            "processes": [
                {
                    "command": "bash",
                    "pcpu": 0.0,
                    "pid": 1346773,
                    "pidfile": "artifacts/full_runs/full_ablation_20260709_003012/logs/warm_tail_recovery_after_watcher_exit.pid",
                    "role": "pidfile",
                    "rss_mib": 3.5,
                },
                {
                    "command": "python",
                    "pcpu": 99.9,
                    "pid": 3396482,
                    "pidfile": "artifacts/full_runs/full_ablation_20260709_003012/logs/warm_after_export_rfam_current_exact.pid",
                    "role": "descendant",
                    "rss_mib": 72726.55,
                },
            ],
            "rows": [
                {"detail": "count=8", "item": "gpu:list", "status": "pass"},
                {"detail": "active_gpu_count=6", "item": "gpu:active_utilization", "status": "pass"},
            ],
            "summary": {
                "counts": {"fail": 0, "pass": 12, "warn": 0},
                "gpu_count": 8,
                "process_count": 10,
                "resource_healthy": True,
            },
        },
    )


def _ledger_text() -> str:
    return "\n".join(
        [
            "| status | state | evidence | next |",
            "|---|---|---|---|",
            "| RF-A1-warm full-data | active | old | old |",
            "| run monitor snapshot | active | old | old |",
            "| current queue status | active | old | old |",
            "| paper artifact audit snapshot | active | old | old |",
            "| runtime health audit | active | old | old |",
            "| system resource audit | active | old | old |",
            "| profile bottleneck audit | active | old | old |",
            "| queue progress audit | active | old | old |",
            "",
        ]
    )


def test_update_ablation_ledger_status_rewrites_monitor_rows(tmp_path):
    mod = _load_module()
    full = tmp_path / "full"
    ledger = tmp_path / "docs/ablation_experiment_filled.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(_ledger_text(), encoding="utf-8")
    _write_artifacts(full)

    result = mod.update_ledger(ledger, full)

    text = ledger.read_text(encoding="utf-8")
    assert result["changed"] is True
    assert "full-run 队列状态" in text
    assert "progress `76.21%`" in text
    assert "`ok_for_paper_table=true`, `pass=21`, `warn=2`, `fail=0`" in text
    assert "`healthy=true`, `audited_run_count=1`, `pass=10`, `warn=0`, `fail=0`" in text
    assert "`samples/s=4.2866`" in text
    assert "趋势估计剩余约 `34706.8s`" in text
    assert "`rss_mib=72726.55`" in text
    assert "tracked recovery pid `1346773`" in text
    assert "占比 `0.8048`" in text


def test_update_ablation_ledger_status_uses_active_run_labels(tmp_path):
    mod = _load_module()
    full = tmp_path / "full"
    ledger = tmp_path / "docs/ablation_experiment_filled.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(_ledger_text(), encoding="utf-8")
    _write_artifacts(full)

    rows = json.loads((full / "current_queue_status.json").read_text(encoding="utf-8"))
    rows[0] = {
        "progress_fraction": 1.0,
        "run_id": "RF-A1-warm",
        "samples_per_second": 4.0,
        "slowest_phase": "path_sample_features",
        "status": "ok",
        "stderr_size": 0,
    }
    rows.append(
        {
            "progress_fraction": 0.9127,
            "run_id": "RF-A2-adapter4",
            "samples_per_second": 16.73,
            "slowest_phase": "projection_f1",
            "status": "running_or_pending_json",
            "stderr_size": 0,
        }
    )
    _write_json(full / "current_queue_status.json", rows)
    _write_json(
        full / "profile_bottleneck_audit.json",
        {
            "rows": [
                {
                    "detail": "phase=path_sample_features; fraction=0.8048; max=0.7500",
                    "item": "run:RF-A1-warm:slowest_phase_fraction",
                    "status": "warn",
                },
                {
                    "detail": "phase=projection_f1; fraction=0.2773; max=0.7500",
                    "item": "run:RF-A2-adapter4:slowest_phase_fraction",
                    "status": "pass",
                },
            ],
            "summary": {
                "audited_run_count": 2,
                "bottleneck_healthy": True,
                "counts": {"fail": 0, "pass": 2, "warn": 1},
            },
        },
    )
    _write_json(
        full / "queue_progress_audit.json",
        {
            "rows": [
                {
                    "detail": "delta=0.010000; elapsed_seconds=60.0; min_delta=0.000100",
                    "item": "run:RF-A2-adapter4:progress_window",
                    "status": "pass",
                }
            ],
            "summary": {
                "counts": {"fail": 0, "pass": 5, "warn": 0},
                "min_estimated_remaining_seconds": 2201.0,
                "progress_healthy": True,
                "trend_count": 1,
            },
        },
    )

    mod.update_ledger(ledger, full)

    text = ledger.read_text(encoding="utf-8")
    assert "当前 `RF-A2-adapter4` snapshot: progress `91.27%`" in text
    assert "`RF-A1-warm` 1 tier ok" in text
    assert "当前 active `RF-A2-adapter4` progress `91.27%`" in text
    assert "当前 active 行 `RF-A2-adapter4` 显示 `running_or_pending_json`" in text
    assert "`RF-A2-adapter4` 的 `projection_f1` 占比 `0.2773`" in text
    assert "`RF-A2-adapter4` progress `91.27%`, `samples/s=16.7300`" in text


def test_update_ablation_ledger_status_explains_post_training_eval_progress_warning(tmp_path):
    mod = _load_module()
    full = tmp_path / "full"
    ledger = tmp_path / "docs/ablation_experiment_filled.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(_ledger_text(), encoding="utf-8")
    _write_artifacts(full)
    _write_json(
        full / "queue_progress_audit.json",
        {
            "rows": [
                {
                    "detail": "post_training_eval; delta=0.000000; elapsed_seconds=388.2; min_delta=0.000100",
                    "item": "run:RF-A1-warm:progress_window",
                    "status": "warn",
                }
            ],
            "summary": {
                "counts": {"fail": 0, "pass": 4, "warn": 1},
                "min_estimated_remaining_seconds": None,
                "progress_healthy": True,
                "trend_count": 1,
            },
        },
    )

    mod.update_ledger(ledger, full)

    text = ledger.read_text(encoding="utf-8")
    assert "`pass=4`, `warn=1`, `fail=0`" in text
    assert "progress_window warning 来自训练 profile 已闭合后的 eval/finalize 等待落盘阶段" in text


def test_update_ablation_ledger_status_handles_pending_active_row_missing_progress(tmp_path):
    mod = _load_module()
    full = tmp_path / "full"
    ledger = tmp_path / "docs/ablation_experiment_filled.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(_ledger_text(), encoding="utf-8")
    _write_artifacts(full)
    _write_json(
        full / "current_queue_status.json",
        [
            {
                "progress_fraction": 1.0,
                "run_id": "RF-M1-warm_mmseqs",
                "samples_per_second": 3.0,
                "status": "ok",
                "stderr_size": 0,
                "tier": "PDB",
                "mean_f1": 0.045,
            },
            {
                "progress_fraction": None,
                "run_id": "RF-CF3-family-balanced",
                "samples_per_second": None,
                "slowest_phase": "",
                "status": "running_or_pending_json",
                "stderr_size": 0,
            },
        ],
    )
    _write_json(
        full / "queue_progress_audit.json",
        {
            "rows": [],
            "summary": {
                "counts": {"fail": 0, "pass": 3, "warn": 1},
                "min_estimated_remaining_seconds": None,
                "progress_healthy": True,
                "trend_count": 0,
            },
        },
    )
    _write_json(
        full / "profile_bottleneck_audit.json",
        {
            "rows": [],
            "summary": {
                "audited_run_count": 2,
                "bottleneck_healthy": True,
                "counts": {"fail": 0, "pass": 2, "warn": 0},
            },
        },
    )

    mod.update_ledger(ledger, full)

    text = ledger.read_text(encoding="utf-8")
    assert "当前 active `RF-CF3-family-balanced` progress `missing`" in text
    assert "当前 active 行 `RF-CF3-family-balanced` 显示 `running_or_pending_json`, `samples/s=missing`, progress `missing`" in text
    assert "`RF-CF3-family-balanced` progress `missing`, `samples/s=missing`" in text
    assert "本轮窗口 `delta=missing`, `elapsed_seconds=missing`" in text
    assert "`RF-CF3-family-balanced` 的 `` 占比 `missing`" in text


def test_update_ablation_ledger_status_fails_missing_row(tmp_path):
    mod = _load_module()
    full = tmp_path / "full"
    ledger = tmp_path / "docs/ablation_experiment_filled.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("| run monitor snapshot | active | old | old |\n", encoding="utf-8")
    _write_artifacts(full)

    with pytest.raises(ValueError, match="current queue status"):
        mod.update_ledger(ledger, full)


def test_update_ablation_ledger_status_cli(tmp_path, capsys):
    mod = _load_module()
    full = tmp_path / "full"
    ledger = tmp_path / "docs/ablation_experiment_filled.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(_ledger_text(), encoding="utf-8")
    _write_artifacts(full)

    assert mod.main(["--ledger", str(ledger), "--full-run-root", str(full)]) == 0

    out = capsys.readouterr().out
    assert '"changed": true' in out
    assert "76.21%" in ledger.read_text(encoding="utf-8")
