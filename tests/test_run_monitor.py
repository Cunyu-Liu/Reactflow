import json

from reactflow.run_monitor import summarize_profile, write_monitor_markdown


def test_summarize_profile_reports_progress_eta_and_phase_rank(tmp_path):
    profile = tmp_path / "profile.jsonl"
    profile.write_text(
        "\n".join(
            [
                json.dumps({"epoch": 0, "sample_index": 0, "phase": "path_sample_features", "seconds": 0.25, "length": 10}),
                json.dumps({"epoch": 0, "sample_index": 0, "phase": "model_forward", "seconds": 0.10, "length": 10}),
                json.dumps({"epoch": 0, "sample_index": 1, "phase": "path_sample_features", "seconds": 0.15, "length": 12}),
                "{truncated",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stderr = tmp_path / "stderr.log"
    stderr.write_text("first line\nlast line\n", encoding="utf-8")

    summary = summarize_profile(profile, total_samples=4, stderr_path=stderr, stderr_tail_bytes=10)

    assert summary["events"] == 3
    assert summary["processed_samples"] == 2
    assert summary["progress_fraction"] == 0.5
    assert summary["samples_per_second"] == 4.0
    assert summary["eta_seconds"] == 0.5
    assert summary["slowest_phase"]["phase"] == "path_sample_features"
    assert summary["phases"]["path_sample_features"]["count"] == 2
    assert summary["stderr_tail"] == "last line\n"


def test_write_monitor_markdown(tmp_path):
    summary = {
        "profile_path": "run/profile.jsonl",
        "processed_samples": 5,
        "total_samples": 10,
        "progress_fraction": 0.5,
        "eta_seconds": 3661.2,
        "stderr_size_bytes": 0,
        "phases_by_total_seconds": [
            {
                "phase": "model_forward",
                "count": 2,
                "total_seconds": 1.5,
                "mean_seconds": 0.75,
                "max_seconds": 1.0,
            }
        ],
    }
    out = tmp_path / "monitor.md"

    write_monitor_markdown(summary, out)

    text = out.read_text(encoding="utf-8")
    assert "ReactFlow Run Monitor" in text
    assert "`50.00%`" in text
    assert "`01:01:01`" in text
    assert "| model_forward | 2 | 1.500000 | 0.750000 | 1.000000 |" in text


def test_summarize_missing_profile_is_empty(tmp_path):
    summary = summarize_profile(tmp_path / "missing.jsonl", total_samples=10)

    assert summary["events"] == 0
    assert summary["processed_samples"] is None
    assert summary["progress_fraction"] is None
    assert summary["slowest_phase"] is None
