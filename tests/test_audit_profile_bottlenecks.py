import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_profile_bottlenecks.py"
    spec = importlib.util.spec_from_file_location("audit_profile_bottlenecks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_monitor(run_dir, *, include_prefetch=False):
    phases = {
        "path_sample_features": {"count": 10, "max_seconds": 2.0, "mean_seconds": 1.0, "total_seconds": 10.0},
        "model_forward": {"count": 10, "max_seconds": 0.5, "mean_seconds": 0.2, "total_seconds": 2.0},
    }
    if include_prefetch:
        phases["frozen_batch_prefetch"] = {
            "count": 5,
            "max_seconds": 0.3,
            "mean_seconds": 0.1,
            "total_seconds": 0.5,
        }
    rows = [{"phase": phase, **metrics} for phase, metrics in phases.items()]
    rows.sort(key=lambda item: item["total_seconds"], reverse=True)
    payload = {
        "events": 20,
        "phases": phases,
        "phases_by_total_seconds": rows,
        "slowest_phase": rows[0],
        "total_profiled_seconds": sum(item["total_seconds"] for item in phases.values()),
    }
    run_dir.mkdir(parents=True)
    (run_dir / "monitor_snapshot.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_completed_evidence(run_dir):
    (run_dir / "training_checkpoint.json").write_text(json.dumps({"weights": [0.1]}), encoding="utf-8")
    (run_dir / "stdout.json").write_text(
        json.dumps(
            {
                "tiers": {
                    "PDB": {
                        "count": 1,
                        "mean_f1": 0.1,
                        "mean_mcc": 0.1,
                        "micro_f1": 0.1,
                        "micro_mcc": 0.1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_profile_bottleneck_audit_warns_for_known_slow_phase(tmp_path):
    mod = _load_module()
    run = tmp_path / "RF-A1"
    _write_monitor(run, include_prefetch=False)

    result = mod.run_audit(
        [run],
        total_samples=None,
        max_slowest_fraction=0.5,
        target_phase="path_sample_features",
        max_target_fraction=0.5,
        expected_optimization_phase="frozen_batch_prefetch",
        min_events=1,
    )

    assert result["summary"]["bottleneck_healthy"] is True
    assert result["summary"]["counts"]["warn"] == 3
    warning_items = {row["item"] for row in result["rows"] if row["status"] == "warn"}
    assert "run:RF-A1:slowest_phase_fraction" in warning_items
    assert "run:RF-A1:target_phase_fraction" in warning_items
    assert "run:RF-A1:optimization_phase_present" in warning_items


def test_profile_bottleneck_audit_keeps_completed_slow_phase_as_history(tmp_path):
    mod = _load_module()
    run = tmp_path / "RF-A1"
    _write_monitor(run, include_prefetch=False)
    _write_completed_evidence(run)

    result = mod.run_audit(
        [run],
        total_samples=None,
        max_slowest_fraction=0.5,
        target_phase="path_sample_features",
        max_target_fraction=0.5,
        expected_optimization_phase="frozen_batch_prefetch",
        min_events=1,
    )

    assert result["summary"]["bottleneck_healthy"] is True
    assert result["summary"]["counts"]["warn"] == 0
    history_rows = [row for row in result["rows"] if "completed_history" in row["detail"]]
    assert {row["item"] for row in history_rows} == {
        "run:RF-A1:slowest_phase_fraction",
        "run:RF-A1:target_phase_fraction",
        "run:RF-A1:optimization_phase_present",
    }
    assert {row["status"] for row in history_rows} == {"pass"}


def test_profile_bottleneck_audit_passes_when_prefetch_phase_exists(tmp_path):
    mod = _load_module()
    run = tmp_path / "RF-A2"
    _write_monitor(run, include_prefetch=True)

    result = mod.run_audit(
        [run],
        total_samples=None,
        max_slowest_fraction=0.9,
        target_phase="path_sample_features",
        max_target_fraction=0.9,
        expected_optimization_phase="frozen_batch_prefetch",
        min_events=1,
    )

    passed_items = {row["item"] for row in result["rows"] if row["status"] == "pass"}
    assert "run:RF-A2:optimization_phase_present" in passed_items
    assert result["summary"]["counts"]["fail"] == 0


def test_profile_bottleneck_audit_fails_missing_profile(tmp_path):
    mod = _load_module()
    run = tmp_path / "missing"
    run.mkdir()

    result = mod.run_audit(
        [run],
        total_samples=None,
        max_slowest_fraction=0.9,
        target_phase="path_sample_features",
        max_target_fraction=0.9,
        expected_optimization_phase="frozen_batch_prefetch",
        min_events=1,
    )

    assert result["summary"]["bottleneck_healthy"] is False
    assert result["rows"][0]["status"] == "fail"


def test_profile_bottleneck_markdown(tmp_path):
    mod = _load_module()
    rows = [mod.row("pass", "run:RF:test", tmp_path / "run", "ok")]
    summary = mod.summarize(rows, audited_run_count=1)
    out = tmp_path / "audit.md"

    mod.write_markdown(rows, summary, out)

    text = out.read_text(encoding="utf-8")
    assert "ReactFlow Profile Bottleneck Audit" in text
    assert "| pass | run:RF:test |" in text
