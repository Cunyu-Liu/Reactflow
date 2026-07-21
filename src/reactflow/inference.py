"""Protocol-safe inference for ReactFlow checkpoints.

The default path integrates the learned DFM CTMC.  The historical endpoint
forward pass is retained under an explicit ``legacy_direct`` mode only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import math
from pathlib import Path
import time
from typing import Mapping, Optional, Sequence, Tuple, Union

from reactflow.checkpoint import TrainingCheckpoint, read_training_checkpoint
from reactflow.constraints import (
    PairValidationResult,
    project_greedy_matching,
    project_max_weight_nested,
    validate_pair_matrix,
)
from reactflow.features import FeatureAdapter, FrozenFeatureLookup, build_augmented_features
from reactflow.model import PairwiseDenoiser, marginal_pair_matrix
from reactflow.sampling import (
    SampledStructure,
    ensemble_unpaired_probability,
    pairing_frequency_matrix,
    sample_structures,
)
from reactflow.train import build_features


class InferenceMode(str, Enum):
    LEGACY_DIRECT = "legacy_direct"
    CTMC_SAMPLE = "ctmc_sample"
    CALIBRATED_MARGINAL = "calibrated_marginal"


class MatchingPolicy(str, Enum):
    NESTED_DP = "nested_dp"
    PSEUDOKNOT_ALLOWED_GREEDY = "pseudoknot_allowed_greedy"


@dataclass(frozen=True)
class InferenceConfig:
    mode: InferenceMode = InferenceMode.CALIBRATED_MARGINAL
    seed: int = 20260718
    num_steps: int = 16
    num_samples: int = 8
    frozen_cache_shards: int = 4
    verify_frozen: bool = True

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")


@dataclass(frozen=True)
class DecoderConfig:
    temperature: float = 1.0
    threshold: float = 0.0
    min_loop: int = 3
    matching_policy: MatchingPolicy = MatchingPolicy.NESTED_DP
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.min_loop < 0:
            raise ValueError("min_loop must be non-negative")


@dataclass(frozen=True)
class InferenceResult:
    sequence: str
    mode: InferenceMode
    structure: Tuple[Tuple[int, ...], ...]
    pair_frequency: Tuple[Tuple[float, ...], ...]
    unpaired_probability: Tuple[float, ...]
    ensemble: Tuple[SampledStructure, ...]
    validation: PairValidationResult
    runtime_seconds: float
    provenance: Mapping[str, object]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_vs_unpaired_log_odds(
    pair_probability: Sequence[Sequence[float]],
    unpaired_probability: Sequence[float],
    *,
    temperature: float = 1.0,
    epsilon: float = 1e-8,
) -> Tuple[Tuple[float, ...], ...]:
    """Return symmetric pair-vs-null log odds used by the C0 decoder."""

    if temperature <= 0.0 or epsilon <= 0.0:
        raise ValueError("temperature and epsilon must be positive")
    size = len(pair_probability)
    if len(unpaired_probability) != size or any(len(row) != size for row in pair_probability):
        raise ValueError("pair and unpaired probability shapes differ")
    scores = [[float("-inf") for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            pair = max(0.0, 0.5 * (float(pair_probability[i][j]) + float(pair_probability[j][i])))
            null = math.sqrt(
                max(0.0, float(unpaired_probability[i])) + epsilon
            ) * math.sqrt(max(0.0, float(unpaired_probability[j])) + epsilon)
            value = math.log((pair + epsilon) / null) / temperature
            scores[i][j] = value
            scores[j][i] = value
    return tuple(tuple(row) for row in scores)


def decode_calibrated_marginal(
    sequence: str,
    pair_probability: Sequence[Sequence[float]],
    unpaired_probability: Sequence[float],
    decoder: DecoderConfig,
    *,
    allow_wobble: bool = True,
) -> Tuple[Tuple[int, ...], ...]:
    scores = pair_vs_unpaired_log_odds(
        pair_probability,
        unpaired_probability,
        temperature=decoder.temperature,
        epsilon=decoder.epsilon,
    )
    if decoder.matching_policy is MatchingPolicy.NESTED_DP:
        return project_max_weight_nested(
            sequence,
            scores,
            min_loop=decoder.min_loop,
            allow_wobble=allow_wobble,
            min_score=decoder.threshold,
        )
    if decoder.matching_policy is MatchingPolicy.PSEUDOKNOT_ALLOWED_GREEDY:
        return project_greedy_matching(
            sequence,
            scores,
            min_loop=decoder.min_loop,
            allow_wobble=allow_wobble,
            allow_pseudoknot=True,
            min_score=decoder.threshold,
        )
    raise ValueError(f"unsupported matching policy: {decoder.matching_policy}")


def _checkpoint(value: Union[TrainingCheckpoint, str, Path]) -> tuple[TrainingCheckpoint, dict]:
    if isinstance(value, TrainingCheckpoint):
        return value, {"checkpoint_path": None, "checkpoint_sha256": None}
    path = Path(value)
    return read_training_checkpoint(path), {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": _sha256_path(path),
    }


def predict_structure(
    checkpoint: Union[TrainingCheckpoint, str, Path],
    sequence: str,
    frozen_features: Optional[FrozenFeatureLookup] = None,
    inference_config: Optional[InferenceConfig] = None,
    decoder_config: Optional[DecoderConfig] = None,
) -> InferenceResult:
    """Predict one legal structure through an explicit, auditable mode."""

    started = time.perf_counter()
    config = inference_config or InferenceConfig()
    decoder = decoder_config or DecoderConfig()
    restored, checkpoint_provenance = _checkpoint(checkpoint)
    sequence = str(sequence).upper()
    if not sequence:
        raise ValueError("sequence must be non-empty")
    model = PairwiseDenoiser(
        restored.result.parameters,
        min_loop=decoder.min_loop,
        allow_wobble=True,
    )
    adapter = (
        FeatureAdapter(restored.result.adapter_parameters)
        if restored.result.adapter_parameters is not None
        else None
    )
    single_rows = frozen_features.single_rows(sequence) if frozen_features is not None else None

    def feature_builder(seq: str, t: float, states: Sequence[int]) -> Sequence[Sequence[float]]:
        base = build_features(seq, t, states)
        augmented, _ = build_augmented_features(base, adapter, single_rows)
        return augmented

    ensemble: Tuple[SampledStructure, ...] = tuple()
    if config.mode is InferenceMode.LEGACY_DIRECT:
        features = feature_builder(sequence, 1.0, [0 for _ in sequence])
        marginals = model.forward(sequence, features).marginals
        pair_frequency = marginal_pair_matrix(marginals)
        unpaired = tuple(float(row[0]) for row in marginals)
        structure = project_greedy_matching(
            sequence,
            pair_frequency,
            min_loop=decoder.min_loop,
            allow_wobble=model.allow_wobble,
            allow_pseudoknot=True,
            min_score=1e-6,
        )
        allow_pseudoknot = True
    else:
        allow_pseudoknot = decoder.matching_policy is MatchingPolicy.PSEUDOKNOT_ALLOWED_GREEDY
        ensemble = sample_structures(
            model,
            sequence,
            num_samples=config.num_samples,
            num_steps=config.num_steps,
            seed=config.seed,
            allow_pseudoknot=allow_pseudoknot,
            feature_builder=feature_builder,
        )
        pair_frequency = pairing_frequency_matrix(ensemble)
        unpaired = ensemble_unpaired_probability(ensemble)
        if config.mode is InferenceMode.CTMC_SAMPLE:
            structure = ensemble[0].pair_matrix
        else:
            structure = decode_calibrated_marginal(
                sequence,
                pair_frequency,
                unpaired,
                decoder,
                allow_wobble=model.allow_wobble,
            )
    validation = validate_pair_matrix(
        sequence,
        structure,
        min_loop=decoder.min_loop,
        allow_wobble=model.allow_wobble,
        allow_pseudoknot=allow_pseudoknot,
    )
    provenance = {
        **checkpoint_provenance,
        "inference": {**asdict(config), "mode": config.mode.value},
        "decoder": {**asdict(decoder), "matching_policy": decoder.matching_policy.value},
        "legacy_endpoint_path": config.mode is InferenceMode.LEGACY_DIRECT,
        "frozen_features_present": frozen_features is not None,
    }
    return InferenceResult(
        sequence=sequence,
        mode=config.mode,
        structure=structure,
        pair_frequency=pair_frequency,
        unpaired_probability=unpaired,
        ensemble=ensemble,
        validation=validation,
        runtime_seconds=time.perf_counter() - started,
        provenance=provenance,
    )
