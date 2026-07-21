"""Portable JSON checkpoints for ReactFlow training runs.

The checkpoint format is intentionally plain JSON so experiment artifacts can be
diffed, archived and inspected without importing tensor libraries.  It stores the
trained denoiser parameters, optional warm-start adapter parameters, the training
configuration, per-epoch history and user-provided metadata/provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Optional, Tuple

from reactflow.features import AdapterParameters
from reactflow.model import DenoiserParameters
from reactflow.train import EpochRecord, TrainConfig, TrainingResult


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainingCheckpoint:
    """A fully restored JSON training checkpoint.

    Complexity: O(1) metadata wrapper; contained parameter/history sizes dominate.
    """

    config: TrainConfig
    result: TrainingResult
    metadata: Mapping[str, object]
    schema_version: int = CHECKPOINT_SCHEMA_VERSION


def _denoiser_to_dict(parameters: DenoiserParameters) -> dict:
    """Serialize denoiser parameters to JSON-compatible nested lists."""

    return {
        "input_weight": parameters.input_weight,
        "input_bias": parameters.input_bias,
        "pair_matrix": parameters.pair_matrix,
        "pair_compat": parameters.pair_compat,
        "unpaired_weight": parameters.unpaired_weight,
        "unpaired_bias": parameters.unpaired_bias,
    }


def _denoiser_from_dict(payload: Mapping[str, object]) -> DenoiserParameters:
    """Restore denoiser parameters from a checkpoint payload."""

    return DenoiserParameters(
        input_weight=[[float(value) for value in row] for row in payload["input_weight"]],  # type: ignore[index]
        input_bias=[float(value) for value in payload["input_bias"]],  # type: ignore[index]
        pair_matrix=[[float(value) for value in row] for row in payload["pair_matrix"]],  # type: ignore[index]
        pair_compat=float(payload["pair_compat"]),
        unpaired_weight=[float(value) for value in payload["unpaired_weight"]],  # type: ignore[index]
        unpaired_bias=float(payload["unpaired_bias"]),
    )


def _adapter_to_dict(parameters: Optional[AdapterParameters]) -> Optional[dict]:
    """Serialize optional adapter parameters."""

    if parameters is None:
        return None
    return {"weight": parameters.weight, "bias": parameters.bias}


def _adapter_from_dict(payload: Optional[Mapping[str, object]]) -> Optional[AdapterParameters]:
    """Restore optional adapter parameters."""

    if payload is None:
        return None
    return AdapterParameters(
        weight=[[float(value) for value in row] for row in payload["weight"]],  # type: ignore[index]
        bias=[float(value) for value in payload["bias"]],  # type: ignore[index]
    )


def write_training_checkpoint(
    path: Path,
    *,
    config: TrainConfig,
    result: TrainingResult,
    metadata: Optional[Mapping[str, object]] = None,
) -> None:
    """Write a deterministic JSON checkpoint.

    Complexity: O(P + E), where P is the number of parameters and E epochs.
    """

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "config": asdict(config),
        "parameters": _denoiser_to_dict(result.parameters),
        "adapter_parameters": _adapter_to_dict(result.adapter_parameters),
        "history": [asdict(record) for record in result.history],
        "profile_summary": result.profile_summary,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_training_checkpoint(path: Path) -> TrainingCheckpoint:
    """Read and validate a JSON training checkpoint.

    Complexity: O(P + E), where P is the number of parameters and E epochs.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema_version {schema_version}")

    raw_config = dict(payload["config"])
    if "length_bucket_boundaries" in raw_config:
        raw_config["length_bucket_boundaries"] = tuple(raw_config["length_bucket_boundaries"])
    config = TrainConfig(**raw_config)
    history: Tuple[EpochRecord, ...] = tuple(EpochRecord(**record) for record in payload["history"])
    result = TrainingResult(
        parameters=_denoiser_from_dict(payload["parameters"]),
        history=history,
        adapter_parameters=_adapter_from_dict(payload.get("adapter_parameters")),
        profile_summary=payload.get("profile_summary"),
    )
    return TrainingCheckpoint(
        config=config,
        result=result,
        metadata=dict(payload.get("metadata") or {}),
        schema_version=schema_version,
    )
