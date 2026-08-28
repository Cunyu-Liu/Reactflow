#!/usr/bin/env python3
"""Strict single-feature RibonanzaNet2 distillation primitives.

This module deliberately does not use :mod:`reactflow.backbones.foundation`
because that inference backbone may construct ``O(L^2)`` pair features and may
pad or truncate cached singles.  The independent route consumes only the
exported per-nucleotide ``single`` arrays and rejects every length or width
mismatch.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import nn

from scripts.reactflow_delta.model_rescue_v11 import (
    V11ContextBlock,
    mutation_one_hot,
)


ROOT_LAYOUT = "reactflow-sharded-frozen-v1"
MODEL_NAME = "RibonanzaNet2"
MODEL_VERSION = "alpha-v1"
TEACHER_WIDTH = 384
STUDENT_INPUT_CHANNELS = 11
STUDENT_WIDTH = 256
ATTENTION_HEADS = 8
CONTEXT_BLOCKS = 6
FFN_WIDTH = 1024
RELATIVE_DISTANCE_WINDOW = 256
DROPOUT = 0.1
NULL_SHIFT = 17
LEGACY_PARENT_CONTENT_BINDING_MISMATCH_COUNT = 388


@dataclass(frozen=True)
class FrozenRNet2Binding:
    """Frozen source identity required by the independent route."""

    layout: str = ROOT_LAYOUT
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    record_count: int = 208_905
    shard_count: int = 409
    shard_size: int = 512
    weights_sha256: str = (
        "c94031719c8a1c70a9068d5de861f65083cdf0555a15570b3724a8d6d7750e35"
    )


DEFAULT_SOURCE_BINDING = FrozenRNet2Binding()


@dataclass(frozen=True)
class RNet2Record:
    """One exact CPU-resident teacher record."""

    record_id: str
    sequence: str
    teacher: np.ndarray


@dataclass(frozen=True)
class DistillBatch:
    """One exact-common-length CPU batch with no resized teacher rows."""

    record_ids: tuple[str, ...]
    sequences: tuple[str, ...]
    student_inputs: torch.Tensor
    teacher_targets: torch.Tensor
    mask: torch.Tensor


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _exact_single_schema(schema: object) -> bool:
    return schema == {"single": {"axes": ["L", TEACHER_WIDTH], "dtype": "<f4"}}


class RNet2SingleShardStream:
    """Stream exact ``[L,384]`` singles from a frozen sharded export.

    Metadata are checked against the root manifest and the frozen source
    binding at construction.  NPZ members and arrays are checked shard by
    shard while iterating.  Nothing is resized, truncated, or converted.
    """

    def __init__(
        self,
        shard_root: str | Path,
        *,
        binding: FrozenRNet2Binding = DEFAULT_SOURCE_BINDING,
    ) -> None:
        self.shard_root = Path(shard_root)
        self.binding = binding
        manifest_path = self.shard_root / "sharded_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"missing RNet2 root manifest: {manifest_path}")
        self.manifest = _read_json(manifest_path)
        self.runtime_parent_content_binding_mismatch_count = 0
        self._shards = self._validate_root_and_shard_metadata()

    @property
    def record_count(self) -> int:
        return int(self.manifest["record_count"])

    @property
    def shard_count(self) -> int:
        return int(self.manifest["shard_count"])

    def source_summary(self) -> dict[str, object]:
        """Return the already-validated metadata used in audit artifacts."""

        return {
            "layout": self.manifest["layout"],
            "model_name": self.manifest["model_name"],
            "model_version": self.manifest["model_version"],
            "record_count": self.record_count,
            "shard_count": self.shard_count,
            "shard_size": int(self.manifest["shard_size"]),
            "weights_sha256": self.binding.weights_sha256,
            "single_schema": {"axes": ["L", TEACHER_WIDTH], "dtype": "<f4"},
            "legacy_parent_content_binding_mismatch_count": (
                LEGACY_PARENT_CONTENT_BINDING_MISMATCH_COUNT
            ),
            "runtime_parent_content_binding_mismatch_count": (
                self.runtime_parent_content_binding_mismatch_count
            ),
            "root_content_hashes_are_authority": False,
            "full_cache_rehash_performed": False,
        }

    def _validate_root_and_shard_metadata(self) -> list[dict]:
        expected_root = {
            "layout": self.binding.layout,
            "model_name": self.binding.model_name,
            "model_version": self.binding.model_version,
            "record_count": self.binding.record_count,
            "shard_count": self.binding.shard_count,
            "shard_size": self.binding.shard_size,
            "weights_sha256": self.binding.weights_sha256,
        }
        observed_root = {name: self.manifest.get(name) for name in expected_root}
        if observed_root != expected_root:
            raise RuntimeError(
                f"RNet2 root manifest differs from frozen binding: {observed_root}"
            )
        shards = self.manifest.get("shards")
        if not isinstance(shards, list) or len(shards) != self.binding.shard_count:
            raise RuntimeError("RNet2 root manifest shard list is incomplete")

        seen_paths: set[str] = set()
        total_records = 0
        validated: list[dict] = []
        for entry in shards:
            if not isinstance(entry, dict):
                raise RuntimeError("RNet2 root manifest contains a non-object shard")
            path_name = str(entry.get("path", ""))
            if not path_name or path_name in seen_paths:
                raise RuntimeError("RNet2 root manifest has a missing or duplicate shard path")
            seen_paths.add(path_name)
            shard_dir = self.shard_root / path_name
            required = (
                shard_dir / "provenance.json",
                shard_dir / "index.jsonl",
                shard_dir / "features.npz",
            )
            if not shard_dir.is_dir() or not all(path.is_file() for path in required):
                raise RuntimeError(f"RNet2 shard is incomplete: {path_name}")
            provenance = _read_json(shard_dir / "provenance.json")
            shard_records = int(entry.get("record_count", -1))
            expected_provenance = {
                "model_name": self.binding.model_name,
                "model_version": self.binding.model_version,
                "record_count": shard_records,
                "weights_sha256": self.binding.weights_sha256,
            }
            observed_provenance = {
                name: provenance.get(name) for name in expected_provenance
            }
            if observed_provenance != expected_provenance:
                raise RuntimeError(f"RNet2 shard provenance mismatch: {path_name}")
            if provenance.get("content_sha256") != entry.get("content_sha256"):
                # This legacy export is known to have stale parent content
                # bindings.  Parent content hashes are explicitly non-authority;
                # actual index and NPZ records remain strict runtime gates.
                self.runtime_parent_content_binding_mismatch_count += 1
            if entry.get("weights_sha256") != self.binding.weights_sha256:
                raise RuntimeError(f"RNet2 shard weights binding mismatch: {path_name}")
            if not _exact_single_schema(provenance.get("schema")):
                raise RuntimeError(f"RNet2 shard is not exact single-only schema: {path_name}")
            if shard_records < 1 or shard_records > self.binding.shard_size:
                raise RuntimeError(f"RNet2 shard record count is invalid: {path_name}")
            total_records += shard_records
            validated.append(entry)
        if total_records != self.binding.record_count:
            raise RuntimeError("RNet2 shard record counts do not sum to the root manifest")
        return validated

    def _read_index(self, entry: dict) -> list[dict]:
        path_name = str(entry["path"])
        index_path = self.shard_root / path_name / "index.jsonl"
        raw_lines = index_path.read_text(encoding="utf-8").splitlines()
        if any(not line.strip() for line in raw_lines):
            raise RuntimeError(f"RNet2 shard index contains a blank row: {path_name}")
        expected_count = int(entry["record_count"])
        if len(raw_lines) != expected_count:
            raise RuntimeError(f"RNet2 shard index length mismatch: {path_name}")
        rows: list[dict] = []
        for expected_row, raw_line in enumerate(raw_lines):
            row = json.loads(raw_line)
            if not isinstance(row, dict) or int(row.get("row", -1)) != expected_row:
                raise RuntimeError(f"RNet2 shard row numbering mismatch: {path_name}")
            record_id = str(row.get("record_id", ""))
            sequence = str(row.get("sequence", ""))
            length = int(row.get("length", -1))
            if not record_id or length != len(sequence) or length < 1:
                raise RuntimeError(f"RNet2 index sequence length mismatch: {path_name}")
            if set(sequence) - set("ACGU"):
                raise RuntimeError(f"RNet2 index contains a non-ACGU sequence: {record_id}")
            expected_array = {
                "single": {"dtype": "<f4", "shape": [length, TEACHER_WIDTH]}
            }
            if row.get("arrays") != expected_array:
                raise RuntimeError(f"RNet2 index single shape mismatch: {record_id}")
            rows.append(row)
        return rows

    def iter_records(self, *, seed: int) -> Iterator[RNet2Record]:
        """Yield every record once in a deterministic shuffled order."""

        shard_order = list(range(len(self._shards)))
        random.Random(int(seed) * 1_000_003 + 28).shuffle(shard_order)
        seen_record_ids: set[str] = set()
        yielded = 0
        for shard_position in shard_order:
            entry = self._shards[shard_position]
            rows = self._read_index(entry)
            row_order = list(range(len(rows)))
            random.Random(
                int(seed) * 1_000_003 + int(shard_position) * 10_007 + 384
            ).shuffle(row_order)
            shard_dir = self.shard_root / str(entry["path"])
            with np.load(shard_dir / "features.npz", allow_pickle=False) as npz:
                expected_members = {f"{row:06d}.single" for row in range(len(rows))}
                if len(npz.files) != len(expected_members) or set(npz.files) != expected_members:
                    raise RuntimeError(
                        f"RNet2 NPZ is not exact single-only: {entry['path']}"
                    )
                for row_index in row_order:
                    metadata = rows[row_index]
                    record_id = str(metadata["record_id"])
                    if record_id in seen_record_ids:
                        raise RuntimeError(f"duplicate RNet2 record_id: {record_id}")
                    seen_record_ids.add(record_id)
                    array = npz[f"{row_index:06d}.single"]
                    expected_shape = (int(metadata["length"]), TEACHER_WIDTH)
                    if array.shape != expected_shape or array.dtype.str != "<f4":
                        raise RuntimeError(
                            f"RNet2 NPZ single shape/dtype mismatch: {record_id}"
                        )
                    if not bool(np.isfinite(array).all()):
                        raise RuntimeError(f"RNet2 NPZ single is nonfinite: {record_id}")
                    yielded += 1
                    yield RNet2Record(
                        record_id=record_id,
                        sequence=str(metadata["sequence"]),
                        teacher=array,
                    )
        if yielded != self.binding.record_count or len(seen_record_ids) != yielded:
            raise RuntimeError("RNet2 stream did not yield the exact frozen universe")

    def iter_batches(self, *, batch_size: int, seed: int) -> Iterator[DistillBatch]:
        if int(batch_size) < 1:
            raise ValueError("distillation batch_size must be positive")
        # Contract forbids even silent batch padding.  Length buckets preserve
        # exact student-token/teacher-L identity for every forward.
        pending: dict[int, list[RNet2Record]] = {}
        for record in self.iter_records(seed=seed):
            bucket = pending.setdefault(len(record.sequence), [])
            bucket.append(record)
            if len(bucket) == int(batch_size):
                yield collate_distill_records(bucket)
                pending[len(record.sequence)] = []
        for length in sorted(pending):
            if pending[length]:
                yield collate_distill_records(pending[length])


def student_inputs_from_sequences(
    sequences: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create V14-channel-compatible CPU inputs without assay outcomes."""

    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("distillation sequences must be non-empty")
    if len({len(sequence) for sequence in sequences}) != 1:
        raise ValueError("distillation student batches cannot use padding")
    max_length = max(len(sequence) for sequence in sequences)
    local = torch.zeros(
        len(sequences), max_length, STUDENT_INPUT_CHANNELS, dtype=torch.float32
    )
    mask = torch.zeros(len(sequences), max_length, dtype=torch.bool)
    alphabet = {"A": 0, "C": 1, "G": 2, "U": 3}
    for batch_index, sequence in enumerate(sequences):
        if set(sequence) - set(alphabet):
            raise ValueError("distillation student input must be exact ACGU")
        length = len(sequence)
        mask[batch_index, :length] = True
        for position, base in enumerate(sequence):
            local[batch_index, position, alphabet[base]] = 1.0
        # V14 channel layout after sequence one-hot:
        # reactivity=0, precision=0, observed=1, normalized position,
        # two assay-region channels=0 (not applicable), corruption=0.
        local[batch_index, :length, 6] = 1.0
        local[batch_index, :length, 7] = torch.arange(
            length, dtype=torch.float32
        ) / max(length - 1, 1)
    return local, mask


def collate_distill_records(records: Sequence[RNet2Record]) -> DistillBatch:
    if not records:
        raise ValueError("cannot collate an empty distillation batch")
    sequences = tuple(record.sequence for record in records)
    local, mask = student_inputs_from_sequences(sequences)
    teacher = torch.zeros(
        len(records), local.shape[1], TEACHER_WIDTH, dtype=torch.float32
    )
    for batch_index, record in enumerate(records):
        expected_shape = (len(record.sequence), TEACHER_WIDTH)
        if record.teacher.shape != expected_shape or record.teacher.dtype.str != "<f4":
            raise RuntimeError(f"cannot collate malformed RNet2 record: {record.record_id}")
        teacher[batch_index, : len(record.sequence)] = torch.from_numpy(record.teacher)
    if local.device.type != "cpu" or teacher.device.type != "cpu" or mask.device.type != "cpu":
        raise RuntimeError("RNet2 disk reader must remain CPU-resident")
    return DistillBatch(
        record_ids=tuple(record.record_id for record in records),
        sequences=sequences,
        student_inputs=local,
        teacher_targets=teacher,
        mask=mask,
    )


def paired_teacher_targets(
    teacher: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned candidate and fixed cyclic-shift null targets."""

    if teacher.ndim != 3 or teacher.shape[-1] != TEACHER_WIDTH:
        raise ValueError("teacher batch must have shape [B,L,384]")
    if mask.shape != teacher.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("teacher mask is misaligned")
    candidate = teacher.clone()
    null = torch.zeros_like(teacher)
    for batch_index in range(teacher.shape[0]):
        length = int(mask[batch_index].sum().item())
        if not bool(mask[batch_index, :length].all()) or bool(
            mask[batch_index, length:].any()
        ):
            raise ValueError("teacher mask must be contiguous left padding authority")
        shift = min(NULL_SHIFT, max(length - 1, 0))
        null[batch_index, :length] = torch.roll(
            teacher[batch_index, :length], shifts=shift, dims=0
        )
        candidate[batch_index, length:] = 0.0
    return candidate, null


class IndependentRNetDistillStudent(nn.Module):
    """Batch-aware V14-compatible student point model plus distill head."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(STUDENT_INPUT_CHANNELS, STUDENT_WIDTH)
        self.input_norm = nn.LayerNorm(STUDENT_WIDTH)
        self.blocks = nn.ModuleList(
            V11ContextBlock(
                width=STUDENT_WIDTH,
                heads=ATTENTION_HEADS,
                ffn_width=FFN_WIDTH,
                dropout=DROPOUT,
                relative_window=RELATIVE_DISTANCE_WINDOW,
            )
            for _ in range(CONTEXT_BLOCKS)
        )
        self.output_norm = nn.LayerNorm(STUDENT_WIDTH)
        self.distill_head = nn.Sequential(
            nn.LayerNorm(STUDENT_WIDTH),
            nn.Linear(STUDENT_WIDTH, TEACHER_WIDTH),
        )
        point_input_width = 2 * STUDENT_WIDTH + 1 + 8 + 1
        self.residual_head = nn.Sequential(
            nn.Linear(point_input_width, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(DROPOUT),
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(384, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def encode_batch(
        self, local_inputs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if local_inputs.ndim != 3 or local_inputs.shape[-1] != STUDENT_INPUT_CHANNELS:
            raise ValueError("student local inputs must have shape [B,L,11]")
        if mask.shape != local_inputs.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("student mask is misaligned")
        if local_inputs.device != mask.device:
            raise ValueError("student inputs and mask must share a device")
        state = self.input_norm(self.input_projection(local_inputs))
        for block in self.blocks:
            state = block(state, mask)
        # ``mask`` is the attention-key authority, matching V11/V14 semantics.
        # An unobserved query still needs its own source/receiver representation;
        # only the distillation prediction wrapper masks invalid padded tokens.
        return self.output_norm(state)

    def forward(self, local_inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        prediction = self.distill_head(self.encode_batch(local_inputs, mask))
        return prediction * mask.unsqueeze(-1).to(prediction.dtype)

    def encode(self, context: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Encode one downstream V14 context with the pretrained encoder."""

        if len(context) != 6:
            raise ValueError("student context must contain six aligned tensors")
        sequence, reactivity, precision, observed, position, region = context
        length = sequence.shape[0]
        if sequence.shape != (length, 4) or region.shape != (length, 2):
            raise ValueError("student sequence or region tensor has invalid shape")
        for tensor in (reactivity, precision, observed, position):
            if tensor.shape != (length,):
                raise ValueError("student scalar context tensor has invalid shape")
        normalized_position = position / max(length - 1, 1)
        corruption = torch.zeros(length, 1, device=sequence.device, dtype=sequence.dtype)
        local = torch.cat(
            [
                sequence,
                reactivity[:, None],
                precision[:, None],
                observed[:, None],
                normalized_position[:, None],
                region,
                corruption,
            ],
            dim=-1,
        )
        return self.encode_batch(local.unsqueeze(0), observed.bool().unsqueeze(0))[0]

    def encode_context(self, context: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Explicit alias for callers that name the downstream context."""

        return self.encode(context)

    def forward_point_and_features(
        self,
        hidden: torch.Tensor,
        edit_index: torch.Tensor,
        signed_distance: torch.Tensor,
        refs: list[str],
        alts: list[str],
        prediction_mask: torch.Tensor,
        feature41_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the frozen V14 feature41-anchored downstream point head."""

        if hidden.ndim != 2 or hidden.shape[1] != STUDENT_WIDTH:
            raise ValueError("student hidden representation has invalid shape")
        batch = edit_index.shape[0]
        length = hidden.shape[0]
        expected = (batch, length)
        if signed_distance.shape != expected or prediction_mask.shape != expected:
            raise ValueError("student distance or prediction mask has invalid shape")
        if feature41_point.shape != expected:
            raise ValueError("student feature41 point has invalid shape")
        if len(refs) != batch or len(alts) != batch:
            raise ValueError("student mutation identity count is misaligned")
        source = hidden[edit_index][:, None, :].expand(batch, length, -1)
        receiver = hidden[None, :, :].expand(batch, -1, -1)
        normalized_distance = signed_distance / max(length - 1, 1)
        mutation = mutation_one_hot(refs, alts, hidden.device)
        mutation = mutation[:, None, :].expand(batch, length, -1)
        features = torch.cat(
            [
                source,
                receiver,
                normalized_distance[..., None],
                mutation,
                feature41_point[..., None],
            ],
            dim=-1,
        )
        residual = self.residual_head(features).squeeze(-1)
        point = (feature41_point + residual).masked_fill(~prediction_mask, 0.0)
        same = torch.tensor(
            [
                ref.replace("T", "U") == alt.replace("T", "U")
                for ref, alt in zip(refs, alts)
            ],
            dtype=torch.bool,
            device=hidden.device,
        )
        return point.masked_fill(same[:, None], 0.0), features

    def forward_point(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_point_and_features(*args, **kwargs)[0]


def make_exact_student_pair(
    *, seed: int, device: str | torch.device
) -> tuple[IndependentRNetDistillStudent, IndependentRNetDistillStudent]:
    torch.manual_seed(int(seed))
    candidate = IndependentRNetDistillStudent().to(device)
    null = copy.deepcopy(candidate)
    assert_exact_student_initial_match(candidate, null)
    return candidate, null


def assert_exact_student_initial_match(
    candidate: IndependentRNetDistillStudent,
    null: IndependentRNetDistillStudent,
) -> None:
    candidate_state = candidate.state_dict()
    null_state = null.state_dict()
    if candidate_state.keys() != null_state.keys():
        raise RuntimeError("candidate/null student state names differ")
    for name in candidate_state:
        if not torch.equal(candidate_state[name], null_state[name]):
            raise RuntimeError(f"candidate/null common initialization differs at {name}")


def encoder_parameter_count(model: IndependentRNetDistillStudent) -> int:
    return sum(
        parameter.numel()
        for module in (
            model.input_projection,
            model.input_norm,
            model.blocks,
            model.output_norm,
        )
        for parameter in module.parameters()
    )


def total_parameter_count(model: IndependentRNetDistillStudent) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def pretraining_parameters(
    model: IndependentRNetDistillStudent,
) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for module in (
        model.input_projection,
        model.input_norm,
        model.blocks,
        model.output_norm,
        model.distill_head,
    ):
        parameters.extend(module.parameters())
    return parameters


def downstream_point_state_dict(
    model: IndependentRNetDistillStudent,
) -> dict[str, torch.Tensor]:
    """Return the loadable point-model state, excluding ``distill_head``."""

    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("distill_head.")
    }


def load_downstream_point_state_dict(
    model: IndependentRNetDistillStudent,
    state: dict[str, torch.Tensor],
) -> None:
    """Strictly load a point-model state while leaving distill head unused."""

    expected = downstream_point_state_dict(model)
    if state.keys() != expected.keys():
        missing = sorted(expected.keys() - state.keys())
        unexpected = sorted(state.keys() - expected.keys())
        raise RuntimeError(
            f"downstream point state keys differ: missing={missing} unexpected={unexpected}"
        )
    full = model.state_dict()
    full.update(state)
    model.load_state_dict(full, strict=True)


def assert_exact_residual_head_match(
    candidate: IndependentRNetDistillStudent,
    null: IndependentRNetDistillStudent,
) -> None:
    candidate_state = candidate.residual_head.state_dict()
    null_state = null.residual_head.state_dict()
    if candidate_state.keys() != null_state.keys():
        raise RuntimeError("candidate/null residual-head state names differ")
    for name in candidate_state:
        if not torch.equal(candidate_state[name], null_state[name]):
            raise RuntimeError(f"candidate/null residual heads differ at {name}")


def assert_encoder_states_differ(
    candidate: IndependentRNetDistillStudent,
    null: IndependentRNetDistillStudent,
) -> None:
    prefixes = ("input_projection.", "input_norm.", "blocks.", "output_norm.")
    candidate_state = candidate.state_dict()
    null_state = null.state_dict()
    if not any(
        not torch.equal(candidate_state[name], null_state[name])
        for name in candidate_state
        if name.startswith(prefixes)
    ):
        raise RuntimeError("aligned and shift-null encoders did not diverge")
