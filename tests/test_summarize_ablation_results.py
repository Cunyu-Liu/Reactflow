import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_ablation_results.py"
    spec = importlib.util.spec_from_file_location("summarize_ablation_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_rows_reads_recovered_metrics_and_profile(tmp_path):
    mod = _load_module()
    run = tmp_path / "RF-test"
    run.mkdir()
    (run / "eval_summary.recovered.json").write_text(
        json.dumps(
            {
                "tiers": {
                    "novel_clan": {
                        "count": 3,
                        "mean_f1": 0.25,
                        "micro_f1": 0.2,
                        "mean_mcc": 0.1,
                        "micro_mcc": 0.08,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "profile.summary.json").write_text(
        json.dumps(
            {
                "phases": {
                    "epoch_total": {"count": 1, "total_seconds": 10.0},
                    "model_forward": {"count": 5, "total_seconds": 2.0},
                },
                "slowest_step_phase": {"phase": "model_forward"},
                "total_profiled_seconds": 12.0,
            }
        ),
        encoding="utf-8",
    )
    (run / "training_checkpoint.json").write_text("{}", encoding="utf-8")

    rows = mod.collect_rows([run])

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["tier"] == "novel_clan"
    assert row["count"] == 3
    assert row["mean_f1"] == 0.25
    assert row["profile_seconds"] == 10.0
    assert row["samples_per_second"] == 0.5
    assert row["slowest_phase"] == "model_forward"
    assert row["checkpoint_present"] is True


def test_collect_rows_reports_running_monitor_without_metrics(tmp_path):
    mod = _load_module()
    run = tmp_path / "RF-running"
    run.mkdir()
    (run / "profile.jsonl").write_text('{"phase":"model_forward","seconds":1.0,"sample_index":0}\\n', encoding="utf-8")
    (run / "monitor_snapshot.json").write_text(
        json.dumps(
            {
                "progress_fraction": 0.5,
                "samples_per_second": 4.0,
                "slowest_phase": {"phase": "path_sample_features"},
                "total_profiled_seconds": 20.0,
            }
        ),
        encoding="utf-8",
    )

    rows = mod.collect_rows([run])

    assert rows == [
        {
            "artifact": str(run),
            "checkpoint_present": False,
            "count": None,
            "mean_f1": None,
            "mean_mcc": None,
            "micro_f1": None,
            "micro_mcc": None,
            "profile_seconds": 20.0,
            "progress_fraction": 0.5,
            "run_id": "RF-running",
            "samples_per_second": 4.0,
            "slowest_phase": "path_sample_features",
            "status": "running_or_pending_json",
            "stderr_size": 0,
            "stderr_tail": "",
            "tier": "",
        }
    ]


def test_write_markdown_includes_speed_columns(tmp_path):
    mod = _load_module()
    rows = [
        {
            "artifact": "runs/RF",
            "checkpoint_present": True,
            "count": 7,
            "mean_f1": 0.3,
            "mean_mcc": 0.2,
            "micro_f1": 0.25,
            "micro_mcc": 0.15,
            "profile_seconds": 12.345,
            "progress_fraction": 0.75,
            "run_id": "RF",
            "samples_per_second": 5.4321,
            "slowest_phase": "model_forward",
            "status": "ok",
            "tier": "PDB",
        }
    ]
    out = tmp_path / "summary.md"

    mod.write_markdown(rows, out, title="Ablation")

    text = out.read_text(encoding="utf-8")
    assert "Samples/s" in text
    assert "75.00%" in text
    assert "| RF | ok | PDB | 7 | 0.3 | 0.25 | 0.2 | 0.15 | 12.35 | 5.4321 | 75.00% | model_forward | yes | runs/RF |" in text


def test_write_svg_creates_placeholder_without_completed_metrics(tmp_path):
    mod = _load_module()
    rows = [
        {
            "artifact": "runs/RF-running",
            "checkpoint_present": False,
            "count": None,
            "mean_f1": None,
            "mean_mcc": None,
            "micro_f1": None,
            "micro_mcc": None,
            "profile_seconds": 10.0,
            "progress_fraction": 0.5,
            "run_id": "RF-running",
            "samples_per_second": 2.0,
            "slowest_phase": "path_sample_features",
            "status": "running_or_pending_json",
            "tier": "",
        }
    ]
    out = tmp_path / "summary.svg"

    mod.write_svg(rows, out, title="Queue & Status")

    text = out.read_text(encoding="utf-8")
    assert "No completed metric rows yet." in text
    assert "Queue &amp; Status" in text
    assert "running_or_pending_json=1" in text


def test_main_accepts_repeated_globs_without_duplicate_runs(tmp_path, monkeypatch, capsys):
    mod = _load_module()
    run_root = tmp_path / "runs"
    run = run_root / "RF-A1-warm_rfam_current_exact_torch_full_data_e1_bs16"
    run.mkdir(parents=True)
    (run / "monitor_snapshot.json").write_text(
        json.dumps({"progress_fraction": 0.25, "samples_per_second": 2.0}),
        encoding="utf-8",
    )
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"
    out_svg = tmp_path / "summary.svg"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_ablation_results.py",
            "--run-root",
            str(run_root),
            "--glob",
            "RF-A1*",
            "--glob",
            "*rfam_current_exact*",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-svg",
            str(out_svg),
        ],
    )

    assert mod.main() == 0

    rows = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert "No completed metric rows yet." in out_svg.read_text(encoding="utf-8")
    assert json.loads(capsys.readouterr().out)["runs"] == 1
