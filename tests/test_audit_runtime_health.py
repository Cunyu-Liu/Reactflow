import importlib.util
import json
import os
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_runtime_health.py"
    spec = importlib.util.spec_from_file_location("audit_runtime_health", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_audit_passes_fresh_active_run(tmp_path):
    mod = _load_module()
    run = tmp_path / "run"
    run.mkdir()
    (run / "profile.jsonl").write_text(
        json.dumps({"phase": "model_forward", "seconds": 0.1, "sample_index": 1}) + "\n",
        encoding="utf-8",
    )
    (run / "stderr.log").write_text("", encoding="utf-8")
    (run / "monitor_snapshot.json").write_text(json.dumps({"progress_fraction": 0.25}), encoding="utf-8")
    pidfile = tmp_path / "alive.pid"
    pidfile.write_text(str(os.getpid()), encoding="utf-8")

    result = mod.run_audit(run, pidfiles=[pidfile], max_profile_age_seconds=3600)

    assert result["summary"]["healthy"] is True
    assert result["summary"]["counts"]["fail"] == 0


def test_runtime_audit_detects_stale_profile_and_stderr(tmp_path):
    mod = _load_module()
    run = tmp_path / "run"
    run.mkdir()
    profile = run / "profile.jsonl"
    profile.write_text("{bad json\n" + json.dumps({"phase": "x", "seconds": 1.0}) + "\n", encoding="utf-8")
    (run / "stderr.log").write_text("boom\n", encoding="utf-8")
    (run / "monitor_snapshot.json").write_text(json.dumps({"progress_fraction": 1.5}), encoding="utf-8")

    result = mod.run_audit(run, pidfiles=[tmp_path / "missing.pid"], max_profile_age_seconds=-1)
    failed_items = {row["item"] for row in result["rows"] if row["status"] == "fail"}
    warned_items = {row["item"] for row in result["rows"] if row["status"] == "warn"}

    assert result["summary"]["healthy"] is False
    assert "profile_fresh" in failed_items
    assert "stderr_empty" in failed_items
    assert "monitor_progress" in failed_items
    assert "pidfile:missing.pid" in warned_items


def test_runtime_audit_warns_for_loading_monitor_without_progress(tmp_path):
    mod = _load_module()
    run = tmp_path / "run"
    run.mkdir()
    (run / "profile.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"phase": "load_frozen_start", "seconds": 0.0}),
                json.dumps({"phase": "load_train_start", "seconds": 0.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "stderr.log").write_text("", encoding="utf-8")
    (run / "monitor_snapshot.json").write_text(json.dumps({"progress_fraction": None}), encoding="utf-8")

    result = mod.run_audit(run, pidfiles=[], max_profile_age_seconds=3600)

    assert result["summary"]["healthy"] is True
    monitor_rows = [row for row in result["rows"] if row["item"] == "monitor_progress"]
    assert len(monitor_rows) == 1
    assert monitor_rows[0]["status"] == "warn"
    assert monitor_rows[0]["detail"] == "loading; progress=None"


def test_runtime_audit_warns_for_stale_loading_profile(tmp_path):
    mod = _load_module()
    run = tmp_path / "run"
    run.mkdir()
    profile = run / "profile.jsonl"
    profile.write_text(json.dumps({"phase": "load_train_start", "seconds": 0.0}) + "\n", encoding="utf-8")
    (run / "stderr.log").write_text("", encoding="utf-8")
    (run / "monitor_snapshot.json").write_text(json.dumps({"progress_fraction": None}), encoding="utf-8")

    result = mod.run_audit(run, pidfiles=[], max_profile_age_seconds=-1)

    assert result["summary"]["healthy"] is True
    freshness = [row for row in result["rows"] if row["item"] == "profile_fresh"]
    assert len(freshness) == 1
    assert freshness[0]["status"] == "warn"
    assert freshness[0]["detail"].startswith("loading; age_seconds=")


def test_runtime_audit_allows_completed_monitor_without_progress(tmp_path):
    mod = _load_module()
    run = tmp_path / "run"
    run.mkdir()
    (run / "profile.jsonl").write_text(json.dumps({"phase": "epoch_total", "seconds": 12.0}) + "\n", encoding="utf-8")
    (run / "stderr.log").write_text("", encoding="utf-8")
    (run / "monitor_snapshot.json").write_text(json.dumps({"progress_fraction": None}), encoding="utf-8")
    (run / "training_checkpoint.json").write_text(json.dumps({"weights": [0.1]}), encoding="utf-8")
    (run / "stdout.json").write_text(
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

    result = mod.run_audit(run, pidfiles=[], max_profile_age_seconds=3600)

    assert result["summary"]["healthy"] is True
    monitor_rows = [row for row in result["rows"] if row["item"] == "monitor_progress"]
    assert len(monitor_rows) == 1
    assert monitor_rows[0]["status"] == "pass"
    assert monitor_rows[0]["detail"] == "completed; progress=None"


def test_runtime_audit_warns_for_post_training_eval_monitor_without_progress(tmp_path):
    mod = _load_module()
    run = tmp_path / "run"
    run.mkdir()
    (run / "profile.jsonl").write_text(json.dumps({"phase": "epoch_total", "seconds": 12.0}) + "\n", encoding="utf-8")
    (run / "profile.summary.json").write_text(
        json.dumps({"phases": {"epoch_total": {"total_seconds": 12.0}}}),
        encoding="utf-8",
    )
    (run / "stderr.log").write_text("", encoding="utf-8")
    (run / "monitor_snapshot.json").write_text(json.dumps({"progress_fraction": None}), encoding="utf-8")
    (run / "stdout.json").write_text("", encoding="utf-8")

    result = mod.run_audit(run, pidfiles=[], max_profile_age_seconds=3600)

    assert result["summary"]["healthy"] is True
    monitor_rows = [row for row in result["rows"] if row["item"] == "monitor_progress"]
    assert len(monitor_rows) == 1
    assert monitor_rows[0]["status"] == "warn"
    assert monitor_rows[0]["detail"] == "post_training_eval; progress=None"


def test_runtime_audit_allows_stale_profile_for_completed_run(tmp_path):
    mod = _load_module()
    run = tmp_path / "run"
    run.mkdir()
    profile = run / "profile.jsonl"
    profile.write_text(json.dumps({"phase": "epoch_total", "seconds": 12.0}) + "\n", encoding="utf-8")
    (run / "stderr.log").write_text("", encoding="utf-8")
    (run / "monitor_snapshot.json").write_text(json.dumps({"progress_fraction": 0.75}), encoding="utf-8")
    (run / "training_checkpoint.json").write_text(json.dumps({"weights": [0.1]}), encoding="utf-8")
    (run / "stdout.json").write_text(
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

    result = mod.run_audit(run, pidfiles=[], max_profile_age_seconds=-1)

    assert result["summary"]["healthy"] is True
    freshness = [row for row in result["rows"] if row["item"] == "profile_fresh"]
    assert len(freshness) == 1
    assert freshness[0]["status"] == "pass"
    assert freshness[0]["path"] == str(profile)
    assert freshness[0]["detail"].startswith("completed run; age_seconds=")


def test_runtime_audit_warns_for_stale_post_training_eval_profile(tmp_path):
    mod = _load_module()
    run = tmp_path / "run"
    run.mkdir()
    profile = run / "profile.jsonl"
    profile.write_text(json.dumps({"phase": "epoch_total", "seconds": 12.0}) + "\n", encoding="utf-8")
    (run / "profile.summary.json").write_text(
        json.dumps({"phases": {"epoch_total": {"total_seconds": 12.0}}}),
        encoding="utf-8",
    )
    (run / "stderr.log").write_text("", encoding="utf-8")
    (run / "monitor_snapshot.json").write_text(json.dumps({"progress_fraction": 0.75}), encoding="utf-8")
    (run / "stdout.json").write_text("", encoding="utf-8")

    result = mod.run_audit(run, pidfiles=[], max_profile_age_seconds=-1)

    assert result["summary"]["healthy"] is True
    assert result["summary"]["counts"]["warn"] == 1
    freshness = [row for row in result["rows"] if row["item"] == "profile_fresh"]
    assert len(freshness) == 1
    assert freshness[0]["status"] == "warn"
    assert freshness[0]["detail"].startswith("post-training eval/finalize; age_seconds=")


def test_write_runtime_health_markdown(tmp_path):
    mod = _load_module()
    rows = [mod.row("pass", "profile_nonempty", tmp_path / "profile.jsonl", "bytes=10")]
    summary = mod.summarize(rows)
    out = tmp_path / "health.md"

    mod.write_markdown(rows, summary, out)

    text = out.read_text(encoding="utf-8")
    assert "ReactFlow Runtime Health Audit" in text
    assert "| pass | profile_nonempty |" in text


def test_runtime_multi_audit_prefixes_run_items(tmp_path):
    mod = _load_module()
    run_a = tmp_path / "RF-A"
    run_b = tmp_path / "RF-B"
    for run, progress in [(run_a, 0.25), (run_b, 0.5)]:
        run.mkdir()
        (run / "profile.jsonl").write_text(
            json.dumps({"phase": "model_forward", "seconds": 0.1, "sample_index": 1}) + "\n",
            encoding="utf-8",
        )
        (run / "stderr.log").write_text("", encoding="utf-8")
        (run / "monitor_snapshot.json").write_text(json.dumps({"progress_fraction": progress}), encoding="utf-8")

    result = mod.run_multi_audit([run_a, run_b], pidfiles=[], max_profile_age_seconds=3600)

    assert result["summary"]["healthy"] is True
    assert result["summary"]["audited_run_count"] == 2
    assert result["audited_run_dirs"] == [str(run_a), str(run_b)]
    items = {row["item"] for row in result["rows"]}
    assert "run:RF-A:profile_nonempty" in items
    assert "run:RF-B:monitor_progress" in items
