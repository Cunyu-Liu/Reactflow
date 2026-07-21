"""Tests for portable JSON training checkpoints."""

import json

from reactflow.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    read_training_checkpoint,
    write_training_checkpoint,
)
from reactflow.synthetic import make_dataset
from reactflow.train import TrainConfig, train_pilot


def test_training_checkpoint_round_trips_config_parameters_history(tmp_path):
    samples = make_dataset(count=2, stem=4, loop=4, probe="2A3", seed=2)
    config = TrainConfig(epochs=2, seed=7, length_bucket_boundaries=(8, 16))
    result = train_pilot(samples=samples, config=config)
    path = tmp_path / "checkpoint.json"

    write_training_checkpoint(
        path,
        config=config,
        result=result,
        metadata={"dataset": "fixture", "backend": "stdlib"},
    )
    restored = read_training_checkpoint(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert restored.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert restored.metadata == {"dataset": "fixture", "backend": "stdlib"}
    assert restored.config == config
    assert restored.result.parameters.input_weight == result.parameters.input_weight
    assert restored.result.parameters.pair_matrix == result.parameters.pair_matrix
    assert restored.result.history == result.history
    assert restored.result.adapter_parameters is None


def test_training_checkpoint_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 999}\n', encoding="utf-8")

    try:
        read_training_checkpoint(path)
    except ValueError as exc:
        assert "unsupported checkpoint" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected schema rejection")
