from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.reactflow_delta.independent_rnet_distill import (
    FrozenRNet2Binding,
    IndependentRNetDistillStudent,
    RNet2SingleShardStream,
    assert_exact_student_initial_match,
    collate_distill_records,
    downstream_point_state_dict,
    load_downstream_point_state_dict,
    make_exact_student_pair,
    paired_teacher_targets,
)
from scripts.reactflow_delta.run_independent_rnet_distill_pretrain import (
    capture_paired_rng_states,
    require_cuda_training_device,
    restore_paired_rng_states,
)


def _fixture_binding() -> FrozenRNet2Binding:
    return FrozenRNet2Binding(
        record_count=2,
        shard_count=1,
        shard_size=2,
        weights_sha256="fixture-weights",
    )


def _write_fixture(
    root: Path,
    *,
    index_length_delta: int = 0,
    npz_width: int = 384,
    include_pair: bool = False,
) -> None:
    shard = root / "shard_00000"
    shard.mkdir(parents=True)
    content = "fixture-content"
    manifest = {
        "layout": "reactflow-sharded-frozen-v1",
        "model_name": "RibonanzaNet2",
        "model_version": "alpha-v1",
        "record_count": 2,
        "shard_count": 1,
        "shard_size": 2,
        "weights_sha256": "fixture-weights",
        "shards": [
            {
                "path": "shard_00000",
                "record_count": 2,
                "weights_sha256": "fixture-weights",
                "content_sha256": content,
            }
        ],
    }
    (root / "sharded_manifest.json").write_text(json.dumps(manifest))
    provenance = {
        "model_name": "RibonanzaNet2",
        "model_version": "alpha-v1",
        "record_count": 2,
        "weights_sha256": "fixture-weights",
        "content_sha256": content,
        "schema": {"single": {"axes": ["L", 384], "dtype": "<f4"}},
    }
    (shard / "provenance.json").write_text(json.dumps(provenance))
    sequences = ("ACGU", "GCA")
    rows = []
    for row, sequence in enumerate(sequences):
        declared_length = len(sequence) + (index_length_delta if row == 0 else 0)
        rows.append(
            {
                "row": row,
                "record_id": f"record-{row}",
                "sequence": sequence,
                "length": declared_length,
                "arrays": {
                    "single": {
                        "dtype": "<f4",
                        "shape": [declared_length, 384],
                    }
                },
            }
        )
    (shard / "index.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    arrays: dict[str, np.ndarray] = {
        "000000.single": np.arange(4 * npz_width, dtype=np.float32).reshape(
            4, npz_width
        ),
        "000001.single": np.arange(3 * 384, dtype=np.float32).reshape(3, 384),
    }
    if include_pair:
        arrays["000000.pair"] = np.zeros((4, 4, 2), dtype=np.float32)
    np.savez(shard / "features.npz", **arrays)


def test_reader_streams_exact_single_shapes_and_deterministic_batches(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)
    stream = RNet2SingleShardStream(tmp_path, binding=_fixture_binding())
    first = list(stream.iter_batches(batch_size=1, seed=7))
    second = list(stream.iter_batches(batch_size=1, seed=7))
    assert [batch.record_ids for batch in first] == [
        batch.record_ids for batch in second
    ]
    assert sum(len(batch.record_ids) for batch in first) == 2
    assert all(batch.teacher_targets.shape[-1] == 384 for batch in first)
    assert all(batch.teacher_targets.device.type == "cpu" for batch in first)


def test_reader_rejects_index_sequence_length_mismatch(tmp_path: Path) -> None:
    _write_fixture(tmp_path, index_length_delta=1)
    stream = RNet2SingleShardStream(tmp_path, binding=_fixture_binding())
    with pytest.raises(RuntimeError, match="sequence length mismatch"):
        list(stream.iter_records(seed=0))


def test_reader_rejects_npz_width_instead_of_resizing(tmp_path: Path) -> None:
    _write_fixture(tmp_path, npz_width=383)
    stream = RNet2SingleShardStream(tmp_path, binding=_fixture_binding())
    with pytest.raises(RuntimeError, match="shape/dtype mismatch"):
        list(stream.iter_records(seed=0))


def test_reader_rejects_pair_member(tmp_path: Path) -> None:
    _write_fixture(tmp_path, include_pair=True)
    stream = RNet2SingleShardStream(tmp_path, binding=_fixture_binding())
    with pytest.raises(RuntimeError, match="exact single-only"):
        list(stream.iter_records(seed=0))


def test_shift_null_preserves_each_sequence_feature_margins() -> None:
    teacher = torch.zeros(2, 5, 384)
    teacher[0] = torch.arange(5, dtype=torch.float32)[:, None]
    teacher[1, :3] = (10 + torch.arange(3, dtype=torch.float32))[:, None]
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    candidate, null = paired_teacher_targets(teacher, mask)
    assert torch.equal(candidate, teacher)
    for batch_index, length in enumerate((5, 3)):
        assert torch.equal(
            torch.sort(null[batch_index, :length], dim=0).values,
            torch.sort(teacher[batch_index, :length], dim=0).values,
        )
        assert torch.count_nonzero(null[batch_index, length:]) == 0
    assert not torch.equal(null[0, :5], teacher[0, :5])


def test_collation_refuses_variable_length_padding() -> None:
    from scripts.reactflow_delta.independent_rnet_distill import RNet2Record

    records = [
        RNet2Record("a", "AC", np.zeros((2, 384), dtype=np.float32)),
        RNet2Record("b", "ACG", np.zeros((3, 384), dtype=np.float32)),
    ]
    with pytest.raises(ValueError, match="cannot use padding"):
        collate_distill_records(records)


def test_paired_dropout_rng_is_replayed_exactly() -> None:
    torch.manual_seed(101)
    dropout = torch.nn.Dropout(0.5).train()
    values = torch.ones(128)
    cpu_state, cuda_state = capture_paired_rng_states(torch.device("cpu"))
    candidate = dropout(values)
    restore_paired_rng_states(
        cpu_state=cpu_state, cuda_state=cuda_state, device=torch.device("cpu")
    )
    null = dropout(values)
    assert torch.equal(candidate, null)


def test_students_have_exact_common_initialization_and_one_shared_batch_order(
    tmp_path: Path,
) -> None:
    candidate, null = make_exact_student_pair(seed=19, device="cpu")
    assert isinstance(candidate, IndependentRNetDistillStudent)
    assert_exact_student_initial_match(candidate, null)

    _write_fixture(tmp_path)
    stream = RNet2SingleShardStream(tmp_path, binding=_fixture_binding())
    batch = next(stream.iter_batches(batch_size=2, seed=19))
    candidate_targets, null_targets = paired_teacher_targets(
        batch.teacher_targets, batch.mask
    )
    assert candidate_targets.shape == null_targets.shape
    assert len(batch.record_ids) == candidate_targets.shape[0]


def test_downstream_point_state_strictly_excludes_distill_head() -> None:
    candidate, _ = make_exact_student_pair(seed=23, device="cpu")
    state = downstream_point_state_dict(candidate)
    assert state
    assert any(name.startswith("residual_head.") for name in state)
    assert not any(name.startswith("distill_head.") for name in state)
    restored = IndependentRNetDistillStudent()
    load_downstream_point_state_dict(restored, state)
    assert all(
        torch.equal(value, downstream_point_state_dict(restored)[name])
        for name, value in state.items()
    )


def test_downstream_unobserved_query_hidden_is_not_zeroed() -> None:
    torch.manual_seed(29)
    model = IndependentRNetDistillStudent().eval()
    sequence = torch.eye(4, dtype=torch.float32)
    reactivity = torch.zeros(4)
    precision = torch.zeros(4)
    observed = torch.tensor([1.0, 0.0, 1.0, 1.0])
    position = torch.arange(4, dtype=torch.float32)
    region = torch.zeros(4, 2)
    hidden = model.encode(
        (sequence, reactivity, precision, observed, position, region)
    )
    assert hidden.shape == (4, 256)
    assert torch.count_nonzero(hidden[1]).item() > 0


def test_cpu_device_fails_closed_before_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "must_not_exist.pt"
    with pytest.raises(RuntimeError, match="CUDA_REQUIRED"):
        require_cuda_training_device("cpu")
    assert not artifact.exists()
