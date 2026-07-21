import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_active_eval_progress.py"
    spec = importlib.util.spec_from_file_location("audit_active_eval_progress", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_jsonl(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(count)), encoding="utf-8")


def test_active_eval_progress_reports_tier_fraction_and_eta(tmp_path):
    mod = _load_module()
    root = tmp_path / "run"
    run_dir = root / "runs/RF-CF3-family-balanced_mmseqs_torch_full_data_e1_bs16"
    run_dir.mkdir(parents=True)
    profile = run_dir / "profile.jsonl"
    profile.write_text(
        "\n".join(
            [
                json.dumps({"phase": "eval_sample_total", "tier": "novel_clan", "sample_index": 2, "seconds": 0.5}),
                json.dumps({"phase": "eval_model_forward", "tier": "novel_clan", "sample_index": 3, "seconds": 0.2}),
                json.dumps({"phase": "eval_sample_total", "tier": "novel_clan", "sample_index": 3, "seconds": 0.7}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(root / "splits/rfam_current_mmseqs_seed0/novel.jsonl", 10)
    _write_json(
        root / "current_queue_status.json",
        [
            {
                "artifact": str(run_dir),
                "run_id": "RF-CF3-family-balanced_mmseqs_torch_full_data_e1_bs16",
                "status": "running_or_pending_json",
            }
        ],
    )

    result = mod.run_audit(root, tail_bytes=10_000)

    assert result["summary"]["eval_progress_healthy"] is True
    rows = result["rows"]
    assert len(rows) == 1
    assert rows[0]["status"] == "pass"
    detail = json.loads(rows[0]["detail"])
    assert detail["tier"] == "novel_clan"
    assert detail["processed"] == 4
    assert detail["total"] == 10
    assert detail["progress_fraction"] == 0.4
    assert detail["mean_seconds_per_sample"] == 0.6
    assert detail["eta_seconds"] == 3.5999999999999996


def test_active_eval_progress_converts_global_eval_index_to_tier_index(tmp_path):
    mod = _load_module()
    root = tmp_path / "run"
    run_dir = root / "runs/RF-CF3-family-balanced_mmseqs_torch_full_data_e1_bs16"
    run_dir.mkdir(parents=True)
    (run_dir / "profile.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"phase": "eval_sample_total", "tier": "novel_clan", "sample_index": 5, "seconds": 1.0}),
                json.dumps({"phase": "eval_sample_total", "tier": "novel_clan", "sample_index": 6, "seconds": 1.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(root / "splits/rfam_current_mmseqs_seed0/test.jsonl", 4)
    _write_jsonl(root / "splits/rfam_current_mmseqs_seed0/novel.jsonl", 10)
    _write_json(
        root / "current_queue_status.json",
        [
            {
                "artifact": str(run_dir),
                "run_id": "RF-CF3-family-balanced_mmseqs_torch_full_data_e1_bs16",
                "status": "running_or_pending_json",
            }
        ],
    )

    result = mod.run_audit(root, tail_bytes=10_000)

    detail = json.loads(result["rows"][0]["detail"])
    assert detail["sample_index_global"] == 6
    assert detail["tier_offset"] == 4
    assert detail["tier_sample_index"] == 2
    assert detail["processed"] == 3
    assert detail["progress_fraction"] == 0.3


def test_active_eval_progress_passes_without_active_rows(tmp_path):
    mod = _load_module()
    root = tmp_path / "run"
    _write_json(root / "current_queue_status.json", [{"run_id": "done", "status": "ok"}])

    result = mod.run_audit(root)

    assert result["summary"]["eval_progress_healthy"] is True
    assert result["rows"][0]["item"] == "active_eval_progress:no_active_runs"


def test_active_eval_progress_warns_without_eval_tail(tmp_path):
    mod = _load_module()
    root = tmp_path / "run"
    run_dir = root / "runs/RF-running"
    run_dir.mkdir(parents=True)
    (run_dir / "profile.jsonl").write_text(json.dumps({"phase": "epoch_total", "seconds": 1.0}) + "\n", encoding="utf-8")
    _write_json(
        root / "current_queue_status.json",
        [{"artifact": str(run_dir), "run_id": "RF-running", "status": "running_or_pending_json"}],
    )

    result = mod.run_audit(root)

    assert result["summary"]["eval_progress_healthy"] is True
    assert result["rows"][0]["status"] == "warn"
    assert "no eval_sample_total" in result["rows"][0]["detail"]


def test_active_eval_progress_markdown(tmp_path):
    mod = _load_module()
    out = tmp_path / "eval_progress.md"
    result = {
        "rows": [mod.row("pass", "run:x:eval_progress", tmp_path / "profile.jsonl", "ok")],
        "summary": {"counts": {"pass": 1, "warn": 0, "fail": 0}, "eval_progress_healthy": True},
    }

    mod.write_markdown(result, out)

    text = out.read_text(encoding="utf-8")
    assert "ReactFlow Active Eval Progress Audit" in text
    assert "| pass | run:x:eval_progress |" in text
