"""CLI tests for real eFold/RNAndria JSON training and evaluation paths."""

import json

from reactflow.cli import main


def _write_efold_fixture(path):
    """Write a tiny legal eFold-style mapping fixture."""

    payload = {
        "hairpin_a": {
            "sequence": "GGGAAACCC",
            "structure": [[0, 8], [1, 7], [2, 6]],
            "shape": [0.1, 0.2, 0.3, None, 0.5, 0.6, 0.7, 0.8, 0.9],
        },
        "hairpin_b": {
            "sequence": "GGCAAAAGCC",
            "structure": [[0, 9], [1, 8], [2, 7]],
        },
        "hairpin_c": {
            "sequence": "GCAAAAGC",
            "structure": [[0, 7], [1, 6]],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_train_efold_runs_on_structure_json(tmp_path, capsys):
    data = _write_efold_fixture(tmp_path / "efold.json")
    out_dir = tmp_path / "train_efold"

    exit_code = main(
        [
            "train-efold",
            str(data),
            "--epochs",
            "2",
            "--limit",
            "2",
            "--max-length",
            "20",
            "--family-balanced-batches",
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["dataset"] == "efold"
    assert payload["mode"] == "base"
    assert payload["samples"] == 2
    assert payload["lambda_react"] == 0.0
    assert (out_dir / "training_curves.svg").exists()
    assert (out_dir / "pairing_marginals.svg").exists()
    assert (out_dir / "training_checkpoint.json").exists()


def test_cli_train_efold_accepts_contact_auxiliary(tmp_path, capsys):
    data = _write_efold_fixture(tmp_path / "efold.json")
    out_dir = tmp_path / "train_efold_contact"

    exit_code = main(
        [
            "train-efold",
            str(data),
            "--epochs",
            "2",
            "--limit",
            "2",
            "--max-length",
            "20",
            "--lambda-contact",
            "0.2",
            "--contact-negative-weight",
            "0.3",
            "--contact-long-range-min-distance",
            "4",
            "--contact-long-range-weight",
            "2.5",
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["last"]["contact"] > 0.0
    assert payload["last"]["total"] != payload["last"]["dfm"]
    assert payload["contact_long_range_min_distance"] == 4
    assert payload["contact_long_range_weight"] == 2.5


def test_cli_evaluate_efold_scores_named_tier(tmp_path, capsys):
    data = _write_efold_fixture(tmp_path / "efold.json")
    out_dir = tmp_path / "eval_efold"

    exit_code = main(
        [
            "evaluate-efold",
            "--inference-mode",
            "legacy_direct",
            "--train-json",
            str(data),
            "--eval-json",
            f"archiveII={data}",
            "--epochs",
            "2",
            "--train-limit",
            "2",
            "--eval-limit",
            "2",
            "--max-length",
            "20",
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["dataset"] == "efold"
    assert payload["eval_samples"] == {"archiveII": 2}
    assert payload["tiers"]["archiveII"]["count"] == 2
    assert {"short", "medium", "long"} <= set(payload["distance_bins"]["archiveII"])
    assert "local_archiveII" in payload["comparison_markdown"]
    assert (out_dir / "comparison_table.md").exists()


def test_cli_evaluate_efold_appends_eval_profile_heartbeat(tmp_path, capsys):
    data = _write_efold_fixture(tmp_path / "efold.json")
    out_dir = tmp_path / "eval_profiled"
    profile = tmp_path / "profile.jsonl"

    exit_code = main(
        [
            "evaluate-efold",
            "--inference-mode",
            "legacy_direct",
            "--train-json",
            str(data),
            "--eval-json",
            f"archiveII={data}",
            "--epochs",
            "1",
            "--train-limit",
            "2",
            "--eval-limit",
            "2",
            "--max-length",
            "20",
            "--profile-path",
            str(profile),
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    events = [json.loads(line) for line in profile.read_text(encoding="utf-8").splitlines()]
    phases = {event["phase"] for event in events}

    assert exit_code == 0
    assert payload["profile"]["events_path"] == str(profile)
    assert "epoch_total" in phases
    assert "eval_sample_total" in phases
    assert "eval_tier_total" in phases
    assert "eval_artifact_write_total" in phases
    assert any(event.get("tier") == "archiveII" for event in events if event["phase"].startswith("eval_"))


def test_cli_evaluate_efold_profiles_pretrain_load_failure(tmp_path, capsys):
    data = _write_efold_fixture(tmp_path / "efold.json")
    profile = tmp_path / "pretrain_profile.jsonl"

    exit_code = main(
        [
            "evaluate-efold",
            "--inference-mode",
            "legacy_direct",
            "--train-json",
            str(data),
            "--eval-json",
            f"archiveII={data}",
            "--max-length",
            "1",
            "--profile-path",
            str(profile),
            "--output-dir",
            str(tmp_path / "eval_empty"),
        ]
    )
    err = json.loads(capsys.readouterr().err)
    events = [json.loads(line) for line in profile.read_text(encoding="utf-8").splitlines()]
    phases = [event["phase"] for event in events]

    assert exit_code == 2
    assert err["error"] == "no train samples passed filters"
    assert phases == ["load_frozen_start", "load_frozen_total", "load_train_start", "load_train_total"]
    assert events[-1]["length"] == 0


def test_cli_train_efold_adapter_requires_frozen_dir(tmp_path, capsys):
    data = _write_efold_fixture(tmp_path / "efold.json")

    exit_code = main(["train-efold", str(data), "--adapter-dim", "2", "--epochs", "1"])
    err = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert "--frozen-dir" in err["error"]


def test_cli_prepare_efold_cache_writes_jsonl_and_train_reads_it(tmp_path, capsys):
    data = _write_efold_fixture(tmp_path / "efold.json")
    cache = tmp_path / "cache.jsonl"
    out_dir = tmp_path / "train_cache"

    assert main(["prepare-efold-cache", str(data), "--output", str(cache), "--max-length", "20"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["accepted"] == 3
    assert cache.exists()
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 3

    assert main(
        [
            "train-efold",
            str(cache),
            "--epochs",
            "2",
            "--limit",
            "2",
            "--output-dir",
            str(out_dir),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["samples"] == 2
    assert (out_dir / "training_curves.svg").exists()
    assert (out_dir / "training_checkpoint.json").exists()


def test_cli_train_efold_profile_and_buckets(tmp_path, capsys):
    data = _write_efold_fixture(tmp_path / "efold.json")
    out_dir = tmp_path / "train_profiled"
    profile = tmp_path / "profile.jsonl"

    exit_code = main(
        [
            "train-efold",
            str(data),
            "--epochs",
            "1",
            "--limit",
            "2",
            "--max-length",
            "20",
            "--bucket-boundaries",
            "8,12",
            "--profile-path",
            str(profile),
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["backend"] == "stdlib"
    assert payload["profile"]["events_path"] == str(profile)
    assert payload["profile"]["slowest_phase"]["phase"] in payload["profile"]["phases"]
    assert payload["profile"]["slowest_step_phase"]["phase"] in payload["profile"]["phases"]
    assert payload["profile"]["slowest_step_phase"]["phase"] != "epoch_total"
    assert payload["length_buckets"] == {"len_9_12": 2}
    assert profile.exists()
    assert profile.with_suffix(".summary.json").exists()
    assert (out_dir / "training_checkpoint.json").exists()


def test_cli_prepare_efold_cache_windows_long_records(tmp_path, capsys):
    data = tmp_path / "efold_long.json"
    data.write_text(
        json.dumps(
            {
                "long": {
                    "sequence": "GGGAAACCCGGGAAACCC",
                    "structure": [[0, 8], [1, 7], [2, 6], [9, 17], [10, 16], [11, 15]],
                }
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "windowed.jsonl"

    exit_code = main(
        [
            "prepare-efold-cache",
            str(data),
            "--output",
            str(cache),
            "--max-length",
            "9",
            "--window-size",
            "9",
            "--window-stride",
            "9",
            "--bucket-boundaries",
            "9",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["accepted"] == 2
    assert summary["windowed_records"] == 1
    assert summary["length_buckets"] == {"len_le_9": 2}
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2


def test_cli_split_efold_cache_writes_leakage_safe_artifacts(tmp_path, capsys):
    cache = tmp_path / "cache.jsonl"
    rows = []
    for index, family in enumerate(("CL0", "CL0", "CL1", "CL2", "CL3")):
        rows.append(
            {
                "source_id": f"r{index}",
                "family": family,
                "sequence": "GGGAAACCC",
                "pairs": [[0, 8], [1, 7], [2, 6]],
                "probe": "2A3",
                "reactivity": [0.0 for _ in range(9)],
                "reactivity_source": "structure_forward_proxy",
            }
        )
    cache.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    out = tmp_path / "split"

    exit_code = main(
        [
            "split-efold-cache",
            str(cache),
            "--output-dir",
            str(out),
            "--bucket-boundaries",
            "9",
            "--novel-clan-fraction",
            "0.2",
            "--seed",
            "2",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["input_records"] == len(rows)
    assert summary["counts_by_split"]["novel"] > 0
    assert (out / "split_manifest.json").exists()
    total = 0
    for split in ("train", "val", "test", "novel"):
        path = out / f"{split}.jsonl"
        assert path.exists()
        total += len([line for line in path.read_text(encoding="utf-8").splitlines() if line])
    assert total == len(rows)
