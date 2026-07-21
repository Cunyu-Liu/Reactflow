import importlib.util
import json
import math
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_queue_progress.py"
    spec = importlib.util.spec_from_file_location("audit_queue_progress", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_queue(
    path: Path,
    *,
    progress: object,
    samples_per_second: float = 2.0,
    artifact: str = "",
    run_id: str = "RF-running",
) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "artifact": artifact,
                    "progress_fraction": progress,
                    "run_id": run_id,
                    "samples_per_second": samples_per_second,
                    "status": "running_or_pending_json",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_queue_progress_passes_when_progress_increases(tmp_path):
    mod = _load_module()
    queue = tmp_path / "current_queue_status.json"
    history = tmp_path / "logs/current_queue_status_history.jsonl"
    _write_queue(queue, progress=0.50)
    old = mod.make_snapshot(mod.load_queue_rows(queue), observed_at=100.0)
    old["rows"][0]["progress_fraction"] = 0.40
    mod.append_snapshot(history, old)

    result = mod.run_audit(
        queue,
        history,
        append_current=True,
        observed_at=200.0,
        max_history_lines=10,
        window_seconds=180.0,
        min_progress_delta=0.01,
        min_samples_per_second=0.1,
    )

    assert result["summary"]["progress_healthy"] is True
    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert failed == []
    assert result["summary"]["trend_count"] == 1
    assert math.isclose(result["summary"]["min_estimated_remaining_seconds"], 500.0)
    assert math.isclose(result["trends"][0]["progress_rate_per_second"], 0.001)
    assert math.isclose(result["trends"][0]["estimated_remaining_seconds"], 500.0)


def test_queue_progress_fails_when_running_progress_stalls(tmp_path):
    mod = _load_module()
    queue = tmp_path / "current_queue_status.json"
    history = tmp_path / "logs/current_queue_status_history.jsonl"
    _write_queue(queue, progress=0.50)
    old = mod.make_snapshot(mod.load_queue_rows(queue), observed_at=100.0)
    mod.append_snapshot(history, old)

    result = mod.run_audit(
        queue,
        history,
        append_current=True,
        observed_at=200.0,
        max_history_lines=10,
        window_seconds=180.0,
        min_progress_delta=0.01,
        min_samples_per_second=0.1,
    )

    assert result["summary"]["progress_healthy"] is False
    failed_items = {row["item"] for row in result["rows"] if row["status"] == "fail"}
    assert "run:RF-running:progress_window" in failed_items


def test_queue_progress_warns_when_epoch_closed_and_progress_stalls(tmp_path):
    mod = _load_module()
    queue = tmp_path / "current_queue_status.json"
    history = tmp_path / "logs/current_queue_status_history.jsonl"
    run_dir = tmp_path / "runs/RF-running"
    run_dir.mkdir(parents=True)
    (run_dir / "profile.summary.json").write_text(
        json.dumps({"phases": {"epoch_total": {"total_seconds": 12.0}}}),
        encoding="utf-8",
    )
    _write_queue(queue, progress=0.50, artifact=str(run_dir))
    old = mod.make_snapshot(mod.load_queue_rows(queue), observed_at=100.0)
    mod.append_snapshot(history, old)

    result = mod.run_audit(
        queue,
        history,
        append_current=True,
        observed_at=200.0,
        max_history_lines=10,
        window_seconds=180.0,
        min_progress_delta=0.01,
        min_samples_per_second=0.1,
    )

    assert result["summary"]["progress_healthy"] is True
    progress_rows = [row for row in result["rows"] if row["item"] == "run:RF-running:progress_window"]
    assert len(progress_rows) == 1
    assert progress_rows[0]["status"] == "warn"
    assert progress_rows[0]["detail"].startswith("post_training_eval; delta=0.000000")


def test_queue_progress_warns_when_epoch_closed_and_progress_missing(tmp_path):
    mod = _load_module()
    queue = tmp_path / "current_queue_status.json"
    history = tmp_path / "logs/current_queue_status_history.jsonl"
    run_dir = tmp_path / "runs/RF-running"
    run_dir.mkdir(parents=True)
    (run_dir / "profile.summary.json").write_text(
        json.dumps({"phases": {"epoch_total": {"total_seconds": 12.0}}}),
        encoding="utf-8",
    )
    _write_queue(queue, progress=None, samples_per_second=2.0, artifact=str(run_dir))
    old = mod.make_snapshot(mod.load_queue_rows(queue), observed_at=100.0)
    mod.append_snapshot(history, old)

    result = mod.run_audit(
        queue,
        history,
        append_current=True,
        observed_at=200.0,
        max_history_lines=10,
        window_seconds=180.0,
        min_progress_delta=0.01,
        min_samples_per_second=0.1,
    )

    assert result["summary"]["progress_healthy"] is True
    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert failed == []
    progress_rows = [row for row in result["rows"] if row["item"] == "run:RF-running:progress_value"]
    assert len(progress_rows) == 1
    assert progress_rows[0]["status"] == "warn"
    assert progress_rows[0]["detail"].startswith("post_training_eval; invalid progress=None")


def test_queue_progress_warns_when_loading_and_progress_missing(tmp_path):
    mod = _load_module()
    queue = tmp_path / "current_queue_status.json"
    history = tmp_path / "logs/current_queue_status_history.jsonl"
    run_dir = tmp_path / "runs/RF-loading"
    run_dir.mkdir(parents=True)
    (run_dir / "profile.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"phase": "load_frozen_start", "seconds": 0.0}),
                json.dumps({"phase": "load_train_start", "seconds": 0.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_queue(queue, progress=None, samples_per_second=None, artifact=str(run_dir), run_id="RF-loading")

    result = mod.run_audit(
        queue,
        history,
        append_current=True,
        observed_at=200.0,
        max_history_lines=10,
        window_seconds=180.0,
        min_progress_delta=0.01,
        min_samples_per_second=0.1,
    )

    assert result["summary"]["progress_healthy"] is True
    progress_rows = [row for row in result["rows"] if row["item"] == "run:RF-loading:progress_value"]
    assert len(progress_rows) == 1
    assert progress_rows[0]["status"] == "warn"
    assert progress_rows[0]["detail"].startswith("loading; invalid progress=None")


def test_queue_progress_warns_for_initial_history_and_writes_markdown(tmp_path):
    mod = _load_module()
    queue = tmp_path / "current_queue_status.json"
    history = tmp_path / "logs/current_queue_status_history.jsonl"
    out = tmp_path / "queue_progress.md"
    _write_queue(queue, progress=0.25)

    result = mod.run_audit(
        queue,
        history,
        append_current=True,
        observed_at=200.0,
        max_history_lines=10,
        window_seconds=180.0,
        min_progress_delta=0.01,
        min_samples_per_second=0.1,
    )
    mod.write_markdown(result, out)

    warned_items = {row["item"] for row in result["rows"] if row["status"] == "warn"}
    assert "run:RF-running:progress_window" in warned_items
    assert "ReactFlow Queue Progress Audit" in out.read_text(encoding="utf-8")


def test_queue_progress_markdown_includes_trend_eta(tmp_path):
    mod = _load_module()
    queue = tmp_path / "current_queue_status.json"
    history = tmp_path / "logs/current_queue_status_history.jsonl"
    out = tmp_path / "queue_progress.md"
    _write_queue(queue, progress=0.50, samples_per_second=3.5)
    old = mod.make_snapshot(mod.load_queue_rows(queue), observed_at=100.0)
    old["rows"][0]["progress_fraction"] = 0.40
    mod.append_snapshot(history, old)

    result = mod.run_audit(
        queue,
        history,
        append_current=True,
        observed_at=200.0,
        max_history_lines=10,
        window_seconds=180.0,
        min_progress_delta=0.01,
        min_samples_per_second=0.1,
    )
    mod.write_markdown(result, out)

    text = out.read_text(encoding="utf-8")
    assert "## Trend ETA" in text
    assert "| RF-running | 50.00% | 0.100000 | 100.0 | 0.00100000 | 500.0 | 3.5000 |" in text
