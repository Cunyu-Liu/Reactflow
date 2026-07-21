import importlib.util
import json
from pathlib import Path
import os
import sys


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_final_queue.py"
    spec = importlib.util.spec_from_file_location("audit_final_queue", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pidfiles(root: Path, mod) -> None:
    for _stage, _result, pidfile in mod.FINAL_STAGES:
        path = root / pidfile
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(os.getpid()), encoding="utf-8")
    readiness = root / mod.FINAL_READINESS_WATCHER
    readiness.parent.mkdir(parents=True, exist_ok=True)
    readiness.write_text(str(os.getpid()), encoding="utf-8")


def _write_valid_result(path: Path, mod) -> None:
    rows = []
    for tier in mod.EXPECTED_RESULT_TIERS:
        rows.append(
            {
                "artifact": "runs/RF-test",
                "checkpoint_present": True,
                "count": 3,
                "mean_f1": 0.4,
                "mean_mcc": 0.3,
                "micro_f1": 0.35,
                "micro_mcc": 0.25,
                "profile_seconds": 10.0,
                "progress_fraction": None,
                "run_id": "RF-test",
                "samples_per_second": 1.0,
                "slowest_phase": "model_forward",
                "status": "ok",
                "stderr_size": 0,
                "stderr_tail": "",
                "tier": tier,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_final_queue_passes_when_results_missing_but_watchers_alive(tmp_path):
    mod = _load_module()
    _write_pidfiles(tmp_path, mod)

    result = mod.run_audit(tmp_path)

    assert result["summary"]["final_queue_healthy"] is True
    assert result["summary"]["final_results_ready"] is False
    assert result["summary"]["counts"]["fail"] == 0
    assert result["summary"]["counts"]["warn"] == 3


def test_final_queue_fails_when_missing_result_has_no_watcher(tmp_path):
    mod = _load_module()

    result = mod.run_audit(tmp_path)

    assert result["summary"]["final_queue_healthy"] is False
    failed_items = {row["item"] for row in result["rows"] if row["status"] == "fail"}
    assert "stage:warm_rfam_current_exact:watcher" in failed_items
    assert "final_readiness_watcher" in failed_items


def test_final_queue_accepts_warm_recovery_watcher(tmp_path):
    mod = _load_module()
    _write_pidfiles(tmp_path, mod)
    warm_pidfile = tmp_path / mod.FINAL_STAGES[0][2]
    warm_pidfile.write_text("999999999", encoding="utf-8")
    recovery = tmp_path / mod.WARM_RECOVERY_WATCHER
    recovery.parent.mkdir(parents=True, exist_ok=True)
    recovery.write_text(str(os.getpid()), encoding="utf-8")

    result = mod.run_audit(tmp_path)

    assert result["summary"]["final_queue_healthy"] is True
    failed_items = {row["item"] for row in result["rows"] if row["status"] == "fail"}
    assert "stage:warm_rfam_current_exact:watcher" not in failed_items
    assert any(row["item"] == "stage:warm_rfam_current_exact:recovery_watcher" for row in result["rows"])


def test_final_queue_allows_finished_watchers_after_results_exist(tmp_path):
    mod = _load_module()
    for _stage, result_name, _pidfile in mod.FINAL_STAGES:
        _write_valid_result(tmp_path / result_name, mod)
    out = tmp_path / "final_queue.md"

    result = mod.run_audit(tmp_path)
    mod.write_markdown(result, out)

    assert result["summary"]["final_queue_healthy"] is True
    assert result["summary"]["final_results_ready"] is True
    assert "ReactFlow Final Queue Audit" in out.read_text(encoding="utf-8")


def test_final_queue_fails_non_metric_result_file(tmp_path):
    mod = _load_module()
    _write_pidfiles(tmp_path, mod)
    bad = tmp_path / mod.FINAL_STAGES[0][1]
    bad.write_text(
        json.dumps(
            [
                {
                    "count": None,
                    "mean_f1": None,
                    "micro_f1": None,
                    "mean_mcc": None,
                    "micro_mcc": None,
                    "run_id": "RF-running",
                    "status": "running_or_pending_json",
                    "tier": "",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = mod.run_audit(tmp_path)

    assert result["summary"]["final_queue_healthy"] is False
    failed = [row for row in result["rows"] if row["status"] == "fail"]
    assert any(row["item"] == "stage:warm_rfam_current_exact:result" for row in failed)
    assert "non_ok_rows=1" in next(row["detail"] for row in failed if row["item"] == "stage:warm_rfam_current_exact:result")


def test_final_queue_cli_writes_outputs(tmp_path):
    mod = _load_module()
    _write_pidfiles(tmp_path, mod)
    out_json = tmp_path / "final_queue.json"
    out_md = tmp_path / "final_queue.md"

    assert mod.main(["--full-run-root", str(tmp_path), "--output-json", str(out_json), "--output-md", str(out_md)]) == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["summary"]["final_queue_healthy"] is True
    assert "ReactFlow Final Queue Audit" in out_md.read_text(encoding="utf-8")
