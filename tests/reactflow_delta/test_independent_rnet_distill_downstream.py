from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.reactflow_delta.independent_rnet_distill import (
    IndependentRNetDistillStudent,
)
from scripts.reactflow_delta.run_independent_rnet_distill_downstream import (
    EXPECTED_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
    PRETRAIN_CHECKPOINT_SCHEMA,
    _artifact_paths,
    _assert_tensor_cuda,
    _downstream_epoch_order,
    _refuse_fold_overwrite,
    _rename_v11_prediction,
    _reset_downstream_rng,
    validate_pretrained_pair,
)


def _checkpoint_payload(
    *, condition: str, encoder_value: float, residual_value: float = 0.0
) -> dict:
    return {
        "schema_version": PRETRAIN_CHECKPOINT_SCHEMA,
        "experiment_id": "RND1_TEST",
        "condition": condition,
        "seed": 20260828,
        "data_order_seed": 20260828,
        "epochs": 1,
        "training_device": "cuda:0",
        "precision": "float32",
        "source": {"model_name": "RibonanzaNet2", "record_count": 208905},
        "model": {"width": 256, "context_blocks": 6},
        "point_model_state_dict": {
            "input_projection.weight": torch.tensor([encoder_value]),
            "residual_head.7.weight": torch.tensor([residual_value]),
        },
        "distill_head_excluded_from_downstream": True,
    }


def test_pretrained_pair_requires_equal_residual_but_different_encoder() -> None:
    candidate = _checkpoint_payload(condition="aligned_candidate", encoder_value=1.0)
    null = _checkpoint_payload(condition="cyclic_shift_17_null", encoder_value=2.0)
    audit = validate_pretrained_pair(candidate, null)
    assert audit["residual_heads_identical"] is True
    assert audit["pretrained_encoders_different"] is True
    assert audit["changed_encoder_tensor_count"] == 1

    null["point_model_state_dict"]["residual_head.7.weight"] = torch.tensor([3.0])
    with pytest.raises(RuntimeError, match="residual initialization differs"):
        validate_pretrained_pair(candidate, null)


def test_pretrained_pair_rejects_identical_encoder_and_distill_head() -> None:
    candidate = _checkpoint_payload(condition="aligned_candidate", encoder_value=1.0)
    null = _checkpoint_payload(condition="cyclic_shift_17_null", encoder_value=1.0)
    with pytest.raises(RuntimeError, match="pretrained encoders are identical"):
        validate_pretrained_pair(candidate, null)

    null["point_model_state_dict"]["distill_head.1.weight"] = torch.tensor([1.0])
    with pytest.raises(RuntimeError, match="contains distill_head"):
        validate_pretrained_pair(candidate, null)


def test_each_arm_replays_the_same_rng_and_epoch_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("cpu", seed)))
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: calls.append(("cuda", seed)),
    )
    assert _reset_downstream_rng(0) == 2_800_000
    assert _reset_downstream_rng(0) == 2_800_000
    assert calls == [
        ("cpu", 2_800_000),
        ("cuda", 2_800_000),
        ("cpu", 2_800_000),
        ("cuda", 2_800_000),
    ]
    assert _downstream_epoch_order(8, seed=0, epoch=3) == _downstream_epoch_order(
        8, seed=0, epoch=3
    )


def test_prediction_rename_is_exact_and_target_free() -> None:
    source: dict[str, np.ndarray] = {}
    for name in EXPECTED_PREDICTION_FIELDS:
        source_name = name
        if name.startswith("candidate_"):
            source_name = f"anchored_{name.removeprefix('candidate_')}"
        elif name.startswith("null_"):
            source_name = f"unanchored_{name.removeprefix('null_')}"
        if name == "schema_version":
            source[source_name] = np.asarray("old.v11.schema")
        elif name in {"keys", "biological_scoring_key"}:
            source[source_name] = np.asarray(["key"], dtype=object)
        elif name == "registered_status":
            source[source_name] = np.asarray(["covered"], dtype=object)
        else:
            source[source_name] = np.ones(1)
    output = _rename_v11_prediction(source)
    assert frozenset(output) == EXPECTED_PREDICTION_FIELDS
    assert str(output["schema_version"].item()) == PREDICTION_SCHEMA
    assert "anchored_point" not in output
    assert "unanchored_point" not in output


def test_cuda_fail_closed_and_fold_overwrite_refusal(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="CUDA_REQUIRED"):
        _assert_tensor_cuda(torch.ones(1), label="test tensor")
    paths = _artifact_paths(tmp_path, fold=0, seed=0)
    paths["result"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _refuse_fold_overwrite(paths)


def test_downstream_encoder_preserves_unobserved_edit_query_state() -> None:
    model = IndependentRNetDistillStudent().eval()
    length = 4
    sequence = torch.eye(4, dtype=torch.float32)
    reactivity = torch.zeros(length)
    precision = torch.zeros(length)
    observed = torch.tensor([1.0, 0.0, 1.0, 1.0])
    position = torch.arange(length, dtype=torch.float32)
    region = torch.zeros(length, 2)
    with torch.no_grad():
        hidden = model.encode(
            (sequence, reactivity, precision, observed, position, region)
        )
    assert hidden.shape == (length, 256)
    assert not torch.equal(hidden[1], torch.zeros_like(hidden[1]))
