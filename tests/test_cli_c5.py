"""CLI coverage for the cycle-C5 warm-start ``train`` flags and ``evaluate``.

These tests drive the *user-facing* entry points end to end (argument parsing ->
training -> scoring -> JSON/Markdown output) so the project is demonstrably
trainable and usable from the command line, not just from the Python API.
"""

import array
import json
import random

import pytest

from reactflow.cli import main
from reactflow.frozen import (
    FrozenFeatureProvenance,
    FrozenFeatureRecord,
    default_schema,
    write_frozen_shard,
)
from reactflow.npio import NdArray
from reactflow.synthetic import make_dataset


def _write_pilot_frozen_shard(directory, *, d_single=6, seed=1):
    """Write a labelled dry-run frozen shard keyed to the pilot sequences.

    The per-nucleotide vectors are deterministic random fixtures (``weights_sha256
    == ""``) so the warm-start CLI path can be exercised offline without any real
    encoder weights, matching the C5 "frozen features are data" contract.
    """

    samples = make_dataset(count=6, stem=4, loop=4, probe="2A3", seed=seed)
    rng = random.Random(0)
    records = []
    for index, sample in enumerate(samples):
        length = len(sample.sequence)
        flat = array.array("f", [rng.random() for _ in range(length * d_single)])
        nd = NdArray(descr="<f4", shape=(length, d_single), data=flat)
        records.append(
            FrozenFeatureRecord(
                record_id=f"r{index}",
                sequence=sample.sequence,
                arrays={"single": nd},
                family="CL_pilot",
            )
        )
    provenance = FrozenFeatureProvenance(
        model_name="DryRun",
        model_version="fixture",
        weights_sha256="",
        produced_by="test_cli_c5",
        date="2026-07-08",
        schema=default_schema(d_single=d_single),
        notes="dry-run: random fixture features, not real weights",
    )
    write_frozen_shard(directory, records, provenance)
    return directory


# --------------------------------------------------------------------------- #
# evaluate subcommand (base path)
# --------------------------------------------------------------------------- #
def test_cli_evaluate_reports_tiers_gap_and_honest_table(tmp_path, capsys):
    out_dir = tmp_path / "eval_base"

    exit_code = main(["evaluate", "--epochs", "5", "--samples", "3", "--output-dir", str(out_dir)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["mode"] == "base"
    # all three generalization tiers are scored
    assert set(payload["tiers"]) == {"in_clan", "cross_clan", "novel_clan"}
    for tier in payload["tiers"].values():
        assert tier["count"] == 3
    assert set(payload["distance_bins"]) == {"in_clan", "cross_clan", "novel_clan"}
    for tier_bins in payload["distance_bins"].values():
        assert {"short", "medium", "long"} <= set(tier_bins)

    # the headline generalization gap is F1(in_clan) - F1(novel_clan)
    gap = payload["generalization_gap"]
    assert gap["gap"] == pytest.approx(gap["in_clan_f1"] - gap["novel_clan_f1"], abs=1e-9)

    # reactivity metrics exist per tier with finite correlations
    for tier in payload["reactivity"].values():
        assert -1.0 <= tier["pearson"] <= 1.0
        assert -1.0 <= tier["spearman"] <= 1.0
        assert tier["calibrated_mae"] >= 0.0

    # honest cited-vs-local table: eFold public sets stay pending, never merged
    markdown = payload["comparison_markdown"]
    assert "Cited F1 (source)" in markdown
    assert "10.1126/sciadv.adz4967" in markdown
    assert "local-pending" in markdown  # viral_mRNA / lncRNA not back-filled
    assert (out_dir / "comparison_table.md").exists()
    assert (out_dir / "comparison_table.md").read_text(encoding="utf-8") == markdown


def test_cli_evaluate_is_bit_for_bit_deterministic(tmp_path, capsys):
    out_a = tmp_path / "eval_a"
    out_b = tmp_path / "eval_b"

    assert main(["evaluate", "--epochs", "4", "--samples", "3", "--output-dir", str(out_a)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(["evaluate", "--epochs", "4", "--samples", "3", "--output-dir", str(out_b)]) == 0
    second = json.loads(capsys.readouterr().out)

    assert first["tiers"] == second["tiers"]
    assert first["generalization_gap"] == second["generalization_gap"]
    assert first["reactivity"] == second["reactivity"]


# --------------------------------------------------------------------------- #
# warm-start path (train + evaluate)
# --------------------------------------------------------------------------- #
def test_cli_train_warm_start_consumes_frozen_shard(tmp_path, capsys):
    shard = _write_pilot_frozen_shard(tmp_path / "shard")
    out_dir = tmp_path / "train_ws"

    exit_code = main(
        [
            "train",
            "--epochs",
            "6",
            "--samples",
            "6",
            "--adapter-dim",
            "4",
            "--adapter-lr",
            "0.1",
            "--frozen-dir",
            str(shard),
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["mode"] == "warm_start"
    assert payload["feature_size"] == 12  # FEATURE_SIZE (8) + adapter_dim (4)
    warm = payload["warm_start"]
    assert warm["adapter_dim"] == 4
    assert warm["d_single"] == 6
    assert warm["frozen_records"] == 6
    assert warm["matched_pilot_sequences"] == 6  # every pilot sequence has a frozen row
    assert payload["last"]["total"] <= payload["first"]["total"]
    assert (out_dir / "training_curves.svg").exists()


def test_cli_evaluate_warm_start_runs_and_scores_tiers(tmp_path, capsys):
    shard = _write_pilot_frozen_shard(tmp_path / "shard")
    out_dir = tmp_path / "eval_ws"

    exit_code = main(
        [
            "evaluate",
            "--epochs",
            "6",
            "--samples",
            "6",
            "--adapter-dim",
            "4",
            "--adapter-lr",
            "0.1",
            "--frozen-dir",
            str(shard),
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["mode"] == "warm_start"
    assert set(payload["tiers"]) == {"in_clan", "cross_clan", "novel_clan"}
    gap = payload["generalization_gap"]
    assert gap["gap"] == pytest.approx(gap["in_clan_f1"] - gap["novel_clan_f1"], abs=1e-9)


# --------------------------------------------------------------------------- #
# guard rails
# --------------------------------------------------------------------------- #
def test_cli_train_adapter_dim_requires_frozen_dir(tmp_path, capsys):
    exit_code = main(["train", "--adapter-dim", "4", "--epochs", "2", "--output-dir", str(tmp_path / "x")])
    err = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert "--frozen-dir" in err["error"]


def test_cli_evaluate_adapter_dim_requires_frozen_dir(tmp_path, capsys):
    exit_code = main(["evaluate", "--adapter-dim", "4", "--epochs", "2", "--output-dir", str(tmp_path / "y")])
    err = json.loads(capsys.readouterr().err)

    assert exit_code == 2
    assert "--frozen-dir" in err["error"]
