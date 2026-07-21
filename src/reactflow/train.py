"""End-to-end ReactFlow training loop: ``L_DFM + lambda_r * L_react``.

This module wires the discrete-flow-matching denoising objective together with
the reactivity-consistency magnitude term into a single deterministic SGD loop
that runs on the synthetic pilot data.  It is the C3 milestone: prove the
generative core trains end to end and does not collapse to a marginal-only
degenerate solution.

Feature encoding
----------------
For position ``i`` at flow time ``t`` with noised partner class ``x_t_i`` the
feature vector (dimension ``F = 8``) is

    feat_i = [ 1[base=A], 1[base=C], 1[base=G], 1[base=U],
               t,
               1[x_t_i = unpaired], 1[x_t_i = paired],
               (partner(x_t_i) - i) / L  if paired else 0 ].

Loss and gradient
-----------------
The optimized objective per sample is

    L = L_DFM + lambda_r * ell_mag,

where ``L_DFM`` is the mean per-position denoising cross-entropy and ``ell_mag``
is the calibrated weighted reactivity MSE.  The gradient into the denoiser
logit row ``i`` (length ``K = L+1``) is the sum of two exact terms:

    g_i[k] = (1/L) ( pi_i[k] - 1[k = x1_i] )                     # L_DFM
             + lambda_r * d_mag_i * a_i * pi_i[0] * (1[k=0] - pi_i[k])

with

    d_mag_i = 2 w_i alpha (alpha rhat_i + gamma - r_i) / sum_j w_j,
    rhat_i  = a_i q_i + c_i,   q_i = pi_i[0].

The reactivity chain rule uses ``d pi_i[0] / d logit_i[k] =
pi_i[0] (1[k=0] - pi_i[k])`` (softmax Jacobian, class 0 row).  The calibration
pair ``(alpha, gamma)`` is refit each step then held constant for the gradient,
i.e. block-coordinate (alternating) minimization over model and calibration.
The Pearson shape term and joint-structure F1 are monitored, not backpropagated,
in this pilot; full shape-term autodiff is deferred to the tensor backend.

Complexity
----------
Each epoch is ``O(N L^2 H^2)`` for ``N`` samples, sequence length ``L`` and
hidden size ``H``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from reactflow.constraints import matrix_to_pairs, pairs_to_matrix, project_greedy_matching, validate_pair_matrix
from reactflow.contact import contact_denoising_bce, contact_denoising_logit_gradient
from reactflow.data import EfoldRecord, efold_pair_matrix, read_efold_json
from reactflow.dfm import softmax_cross_entropy_gradient, uniform_source, sample_path_index
from reactflow.ensemble import (
    heteroscedastic_reactivity_logit_gradient,
    heteroscedastic_reactivity_nll,
)
from reactflow.features import (
    AdapterGradients,
    AdapterParameters,
    FeatureAdapter,
    FrozenFeatureLookup,
    adapter_sgd_update,
    build_augmented_features,
    split_feature_gradient,
)
from reactflow.metrics import f1_score
from reactflow.model import (
    DenoiserGradients,
    DenoiserParameters,
    PairwiseDenoiser,
    marginal_pair_matrix,
    sgd_update,
    unpaired_probabilities,
)
from reactflow.reactivity import (
    ReactivityForwardOperator,
    fit_weighted_affine_calibration,
    masked_unit_weights,
    weighted_mse,
    weighted_pearson,
)
from reactflow.splits import length_bucket_label
from reactflow.synthetic import SyntheticSample, make_dataset
from reactflow.thermo import monte_carlo_unpaired_prior, thermo_logit_gradient, thermo_unpaired_kl, thermo_unpaired_mse

BASES = ("A", "C", "G", "U")
FEATURE_SIZE = 8


@dataclass
class TrainConfig:
    """Deterministic training configuration.

    Attributes:
        epochs: number of full passes over the dataset.
        hidden_size: denoiser hidden dimension ``H``.
        learning_rate: SGD step size.
        lambda_react: weight ``lambda_r`` on the reactivity magnitude term.
        lambda_thermo: weight ``lambda_td`` on the thermodynamic semi-supervised
            term ``ell_thermo``.  The default ``0.0`` makes the term a no-op so
            the C3 pilot trajectory is bit-for-bit unchanged; C4 experiments
            raise it to inject the Turner prior.
        thermo_mode: ``"mse"`` or ``"kl"`` form of ``ell_thermo``.
        thermo_samples: Monte Carlo samples for the Turner unpaired prior.
        seed: master seed controlling init and path sampling.
        min_loop: minimum hairpin loop for legality masking.
        adapter_dim: output dimension ``d_adapter`` of the C5 frozen-feature
            adapter.  When ``0`` (default) no adapter is created and training is
            bit-for-bit the C3 pilot; when positive, a :class:`FeatureAdapter` is
            trained jointly and its output is concatenated onto the ``FEATURE_SIZE``
            hand-written features.
        adapter_lr: SGD step size for the adapter parameters; when ``None`` the
            denoiser ``learning_rate`` is reused.
        profile_path: optional JSONL path for detailed per-phase training
            timings.  When set, a sibling ``*.summary.json`` file is also
            written with aggregate timing by phase.
        batch_size: optional number of samples per gradient update.  ``None``
            preserves the original full-batch pilot update; smaller values make
            scale-up runs more memory-stable and are the retry knob used after
            OOM or validation instability.
        length_bucket_boundaries: optional sequence-length bucket boundaries used
            to order samples by increasing bucket/length before training.  This
            keeps long-sequence runs deterministic while making per-bucket
            profiling easier to interpret.
        family_balanced_batches: when true, samples are ordered by length bucket
            and then interleaved across ``cluster``/``family`` labels before
            mini-batching.  The default false preserves historical trajectories;
            enabling it is the RF-CF3 cross-family sampler ablation.
        lambda_calib: weight on the variance-aware ensemble-calibration term
            ``ell_calib`` (:mod:`reactflow.ensemble`).  The default ``0.0`` makes
            it a no-op so existing trajectories are bit-for-bit unchanged; raising
            it adds the heteroscedastic reactivity negative log-likelihood that
            couples the mean-field structure confidence to the reactivity
            dispersion (hypothesis H4).
        calib_beta: non-negative scale mapping the Bernoulli structural variance
            ``a_i^2 q_i(1-q_i)`` into reactivity units for ``ell_calib``.
        calib_tau_squared: measurement-noise variance floor ``tau^2`` for
            ``ell_calib`` so the Gaussian log-likelihood stays finite.
        lambda_contact: weight on the contact-map denoising auxiliary.  The term
            uses the DFM-induced soft contact ``0.5(pi_i[j+1]+pi_j[i+1])`` and is
            restricted to legal candidate pairs, so it reinforces pair consistency
            without replacing the partner-class distribution.
        contact_negative_weight: scale on the separately averaged legal non-pair
            BCE term.  Values below 1 prevent the sparse positives from being
            overwhelmed by ``O(L^2)`` negatives.
        contact_long_range_min_distance: RF-CF2 span threshold ``d_min``.  Legal
            candidate pairs with ``|i-j| >= d_min`` are treated as long-range for
            contact auxiliary weighting.
        contact_long_range_weight: RF-CF2 multiplicative weight for long-range
            legal candidate pairs inside the positive/negative weighted means.
            The default ``1.0`` is exactly the historical contact auxiliary.

    Complexity: O(1) configuration storage.
    """

    epochs: int = 40
    hidden_size: int = 8
    learning_rate: float = 0.2
    lambda_react: float = 1.0
    lambda_thermo: float = 0.0
    thermo_mode: str = "mse"
    thermo_samples: int = 128
    seed: int = 0
    min_loop: int = 3
    adapter_dim: int = 0
    adapter_lr: Optional[float] = None
    profile_path: Optional[str] = None
    batch_size: Optional[int] = None
    length_bucket_boundaries: Tuple[int, ...] = ()
    family_balanced_batches: bool = False
    lambda_calib: float = 0.0
    calib_beta: float = 1.0
    calib_tau_squared: float = 0.05
    lambda_contact: float = 0.0
    contact_negative_weight: float = 0.25
    contact_long_range_min_distance: int = 24
    contact_long_range_weight: float = 1.0


@dataclass
class EpochRecord:
    """Per-epoch training diagnostics.

    Complexity: O(1) summary storage.
    """

    epoch: int
    total: float
    dfm: float
    react_magnitude: float
    react_shape: float
    thermo: float
    mean_f1: float
    calib: float = 0.0
    contact: float = 0.0


@dataclass
class TrainingResult:
    """Full training output.

    ``adapter_parameters`` holds the trained C5 frozen-feature adapter when one
    was used (``config.adapter_dim > 0``); it is ``None`` for the C3 pilot path.

    Complexity: O(P + E) storage for parameters P and E epoch records.
    """

    parameters: DenoiserParameters
    history: Tuple[EpochRecord, ...]
    adapter_parameters: Optional[AdapterParameters] = None
    profile_summary: Optional[dict] = None


@dataclass(frozen=True)
class EfoldCacheSummary:
    """Summary produced while materializing an eFold training cache.

    Complexity: O(B) storage for B length-bucket counters.
    """

    output_path: str
    scanned: int
    accepted: int
    skipped_length: int
    skipped_illegal: int
    with_reactivity: int
    min_length: Optional[int]
    max_length: Optional[int]
    windowed_records: int = 0
    windows_emitted: int = 0
    length_buckets: Optional[Dict[str, int]] = None


class TrainingProfiler:
    """Streaming JSONL profiler for the pure-stdlib training loop.

    Each call to :meth:`log` writes one JSON object containing ``phase``,
    ``seconds`` and optional epoch/sample metadata.  Aggregates are kept in
    memory by phase so callers can write a compact summary at the end without
    reading the JSONL file back.

    Complexity: O(P) memory for P phase aggregates; JSONL storage is O(events).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._summary: Dict[str, Dict[str, float]] = {}

    def log(
        self,
        phase: str,
        seconds: float,
        *,
        epoch: Optional[int] = None,
        sample_index: Optional[int] = None,
        length: Optional[int] = None,
    ) -> None:
        """Record one timing event and update phase aggregates.

        Complexity: O(1) per event.
        """

        record = {
            "epoch": epoch,
            "length": length,
            "phase": phase,
            "sample_index": sample_index,
            "seconds": seconds,
        }
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        agg = self._summary.setdefault(phase, {"count": 0.0, "total": 0.0, "max": 0.0})
        agg["count"] += 1.0
        agg["total"] += seconds
        agg["max"] = max(agg["max"], seconds)

    def close(self) -> dict:
        """Flush the JSONL stream and write an aggregate summary JSON file.

        Complexity: O(P) for P recorded phases.
        """

        self._handle.close()
        phases = {}
        for phase, agg in sorted(self._summary.items()):
            count = int(agg["count"])
            total = agg["total"]
            phases[phase] = {
                "count": count,
                "max_seconds": agg["max"],
                "mean_seconds": 0.0 if count == 0 else total / count,
                "total_seconds": total,
            }
        ranked = [
            {"phase": phase, **metrics}
            for phase, metrics in sorted(
                phases.items(),
                key=lambda item: item[1]["total_seconds"],
                reverse=True,
            )
        ]
        step_ranked = [item for item in ranked if not item["phase"].endswith("_total")]
        summary = {
            "events_path": str(self.path),
            "phases": phases,
            "phases_by_total_seconds": ranked,
            "step_phases_by_total_seconds": step_ranked,
            "slowest_phase": ranked[0] if ranked else None,
            "slowest_step_phase": step_ranked[0] if step_ranked else None,
            "summary_path": str(self.path.with_suffix(".summary.json")),
            "total_profiled_seconds": sum(item["total_seconds"] for item in phases.values()),
        }
        self.path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary


def build_features(sequence: str, t: float, noised_classes: Sequence[int]) -> Tuple[Tuple[float, ...], ...]:
    """Construct the per-position feature matrix described in the module docs.

    Complexity: O(L).
    """

    sequence = sequence.upper()
    size = len(sequence)
    if len(noised_classes) != size:
        raise ValueError("noised_classes length must match sequence length")
    features: List[Tuple[float, ...]] = []
    for i, base in enumerate(sequence):
        one_hot = [1.0 if base == b else 0.0 for b in BASES]
        cls = int(noised_classes[i])
        is_unpaired = 1.0 if cls == 0 else 0.0
        is_paired = 1.0 if cls > 0 else 0.0
        rel = ((cls - 1) - i) / size if cls > 0 else 0.0
        features.append(tuple(one_hot + [t, is_unpaired, is_paired, rel]))
    return tuple(features)


def partner_classes_from_pair_matrix(pair_matrix: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    """Convert a pair matrix into DFM clean partner classes.

    Class ``0`` means unpaired; class ``j+1`` means paired to position ``j``.
    The matrix must be a matching (at most one partner per position), which is
    the target representation expected by the per-position DFM denoising loss.

    Complexity: O(L^2).
    """

    size = len(pair_matrix)
    classes: List[int] = []
    for i, row in enumerate(pair_matrix):
        partners = [j for j, value in enumerate(row) if i != j and int(value) > 0]
        if len(partners) > 1:
            raise ValueError(f"position {i} has multiple partners: {partners}")
        classes.append(0 if not partners else partners[0] + 1)
    return tuple(classes)


def sample_from_efold_record(
    record: EfoldRecord,
    *,
    default_probe: str = "2A3",
    min_loop: int = 3,
    allow_wobble: bool = True,
    require_legal: bool = True,
) -> SyntheticSample:
    """Convert one eFold/RNAndria record into a training sample.

    The eFold Dryad files provide a sequence and a hard 2D structure for every
    record, and sometimes a SHAPE/DMS reactivity profile.  This bridge preserves
    the real structure as the DFM target.  If a real profile is present it is used
    as ``sample.reactivity`` with probe-aware masks; otherwise the deterministic
    affine forward operator ``f(S)`` is used only as a monitoring target.  Callers
    should keep ``lambda_react=0`` when training on structure-only files so this
    fallback cannot become fabricated experimental supervision.

    ``require_legal=True`` filters records whose target structure is incompatible
    with the current denoiser legality mask (canonical/wobble, one partner per
    base, and ``min_loop``).  This avoids giving the cross-entropy target a class
    that the model masks out by construction.

    Complexity: O(L^2).
    """

    pair_matrix = efold_pair_matrix(record)
    if require_legal:
        validation = validate_pair_matrix(
            record.sequence,
            pair_matrix,
            min_loop=min_loop,
            allow_wobble=allow_wobble,
            allow_pseudoknot=True,
        )
        if not validation.valid:
            raise ValueError(f"eFold record {record.record_id or '<unknown>'} is not legal: {validation.violations}")
    partner_classes = partner_classes_from_pair_matrix(pair_matrix)
    probe = record.reactivity_probe or default_probe
    operator = ReactivityForwardOperator()
    if record.shape is not None and any(math.isfinite(value) for value in record.shape):
        reactivity = record.shape
    else:
        reactivity = operator.from_structure(record.sequence, pair_matrix, probe)
    weights = masked_unit_weights(record.sequence, probe, reactivity)
    return SyntheticSample(
        sequence=record.sequence,
        dotbracket="." * len(record.sequence),
        pair_matrix=pair_matrix,
        partner_classes=partner_classes,
        reactivity=tuple(float(value) for value in reactivity),
        weights=weights,
        probe=probe,
        source_id=record.record_id,
        family=record.family,
    )


def _window_ranges(length: int, window_size: Optional[int], window_stride: Optional[int]) -> Tuple[Tuple[int, int], ...]:
    """Return deterministic half-open window ranges covering a sequence.

    ``window_size=None`` keeps the sequence intact.  For long sequences the final
    window is anchored at ``length - window_size`` so the tail is never dropped,
    even when ``window_stride`` does not divide the length exactly.

    Complexity: O(W) for W emitted windows.
    """

    if window_size is None:
        return ((0, length),)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    stride = window_stride if window_stride is not None else window_size
    if stride <= 0:
        raise ValueError("window_stride must be positive")
    if length <= window_size:
        return ((0, length),)

    starts = list(range(0, max(1, length - window_size + 1), stride))
    tail_start = length - window_size
    if starts[-1] != tail_start:
        starts.append(tail_start)
    unique_starts = sorted(set(starts))
    return tuple((start, start + window_size) for start in unique_starts)


def window_efold_record(
    record: EfoldRecord,
    *,
    start: int,
    end: int,
    index: int = 0,
) -> Tuple[EfoldRecord, Dict[str, int]]:
    """Slice an eFold record into one local training window.

    Base pairs whose two endpoints fall inside ``[start, end)`` are shifted into
    local coordinates; crossing/outside pairs are intentionally omitted so the
    local target remains a valid partner-class sample.  Reactivity profiles are
    sliced position-wise when present.

    Complexity: O(L_w + P) for window length ``L_w`` and source pair count ``P``.
    """

    parent_length = len(record.sequence)
    if not 0 <= start < end <= parent_length:
        raise ValueError("window bounds must satisfy 0 <= start < end <= length")
    local_pairs = tuple(
        (i - start, j - start)
        for i, j in record.pairs
        if start <= i < end and start <= j < end
    )
    suffix = f"{start}-{end}"
    record_id = f"{record.record_id}:{suffix}" if record.record_id is not None else suffix
    window = EfoldRecord(
        sequence=record.sequence[start:end],
        pairs=local_pairs,
        shape=record.shape[start:end] if record.shape is not None else None,
        reactivity_probe=record.reactivity_probe,
        family=record.family,
        record_id=record_id,
    )
    metadata = {
        "index": index,
        "start": start,
        "end": end,
        "parent_length": parent_length,
    }
    return window, metadata


def iter_efold_record_windows(
    record: EfoldRecord,
    *,
    window_size: Optional[int] = None,
    window_stride: Optional[int] = None,
) -> Tuple[Tuple[EfoldRecord, Optional[Dict[str, int]]], ...]:
    """Return local records for training, windowing only when requested.

    The intact record is returned with ``metadata=None`` when no actual slicing
    occurred, keeping cache files compact for short sequences.

    Complexity: O(W * (L_w + P)).
    """

    ranges = _window_ranges(len(record.sequence), window_size, window_stride)
    if len(ranges) == 1 and ranges[0] == (0, len(record.sequence)):
        return ((record, None),)
    return tuple(
        window_efold_record(record, start=start, end=end, index=index)
        for index, (start, end) in enumerate(ranges)
    )


def _bucket_index(length: int, boundaries: Sequence[int]) -> int:
    """Return the monotone bucket index for ``length``.

    Complexity: O(B) for B boundaries.
    """

    return sum(1 for boundary in boundaries if length > boundary)


def bucket_samples_by_length(
    samples: Sequence[SyntheticSample],
    boundaries: Sequence[int],
) -> Dict[str, Tuple[SyntheticSample, ...]]:
    """Group samples by the same length labels used in split manifests.

    Complexity: O(NB) for N samples and B boundaries.
    """

    buckets: Dict[str, List[SyntheticSample]] = {}
    for sample in samples:
        label = length_bucket_label(len(sample.sequence), boundaries)
        buckets.setdefault(label, []).append(sample)
    return {label: tuple(items) for label, items in sorted(buckets.items())}


def order_samples_by_length_bucket(
    samples: Sequence[SyntheticSample],
    boundaries: Sequence[int],
) -> Tuple[SyntheticSample, ...]:
    """Order samples by length bucket, then length, preserving input ties.

    The pure-stdlib loop still processes one sample at a time, so bucketing is a
    deterministic scheduling/profiling aid there.  Tensor backends can reuse the
    same ordering to minimize padding waste inside future mini-batches.

    Complexity: O(N log N + NB).
    """

    if not boundaries:
        return tuple(samples)
    ordered = sorted(
        enumerate(samples),
        key=lambda item: (_bucket_index(len(item[1].sequence), boundaries), len(item[1].sequence), item[0]),
    )
    return tuple(sample for _index, sample in ordered)


def _sample_family_balance_key(sample: SyntheticSample, index: int) -> str:
    """Return the deterministic balancing group for ``sample``.

    Formula: prefer sequence-identity ``cluster`` when available, then
    ``family``/clan, and finally a singleton key based on ``source_id`` or input
    index.  Singleton fallback makes unlabeled synthetic pilots retain their
    original order under round-robin interleaving.  Complexity: O(1).
    """

    if sample.cluster:
        return f"cluster:{sample.cluster}"
    if sample.family:
        return f"family:{sample.family}"
    if sample.source_id:
        return f"source:{sample.source_id}"
    return f"singleton:{index}"


def order_samples_family_balanced(
    samples: Sequence[SyntheticSample],
    boundaries: Sequence[int],
) -> Tuple[SyntheticSample, ...]:
    """Interleave samples across family/cluster groups within each length bucket.

    For each length bucket, samples are first grouped by
    :func:`_sample_family_balance_key`.  Groups are ordered by first occurrence,
    items inside a group are ordered by ``(length, original_index)``, and the
    output takes one sample per group in round-robin passes.  This approximates a
    per-family balanced mini-batch schedule without changing loss weights or
    duplicating data.

    Formula: for groups ``G_1..G_m`` inside a bucket, emit
    ``G_1[0], G_2[0], ..., G_m[0], G_1[1], ...`` while entries exist.
    Complexity: O(N log N + NB), where B is the number of length boundaries.
    """

    if not samples:
        return tuple()
    bucketed: Dict[str, List[Tuple[int, SyntheticSample]]] = {}
    for index, sample in enumerate(samples):
        label = length_bucket_label(len(sample.sequence), boundaries) if boundaries else "all"
        bucketed.setdefault(label, []).append((index, sample))

    output: List[SyntheticSample] = []
    for _bucket, indexed_samples in sorted(
        bucketed.items(),
        key=lambda item: (_bucket_index(len(item[1][0][1].sequence), boundaries) if boundaries else 0, item[0]),
    ):
        groups: Dict[str, List[Tuple[int, SyntheticSample]]] = {}
        group_order: List[str] = []
        for index, sample in indexed_samples:
            key = _sample_family_balance_key(sample, index)
            if key not in groups:
                group_order.append(key)
                groups[key] = []
            groups[key].append((index, sample))
        for items in groups.values():
            items.sort(key=lambda item: (len(item[1].sequence), item[0]))
        max_len = max(len(items) for items in groups.values())
        for offset in range(max_len):
            for key in group_order:
                items = groups[key]
                if offset < len(items):
                    output.append(items[offset][1])
    return tuple(output)


def order_samples_for_training(
    samples: Sequence[SyntheticSample],
    boundaries: Sequence[int],
    *,
    family_balanced: bool = False,
) -> Tuple[SyntheticSample, ...]:
    """Return the deterministic training schedule.

    Formula: choose
    ``order_samples_family_balanced`` when ``family_balanced`` is true,
    otherwise use the historical ``order_samples_by_length_bucket``.  Complexity:
    O(N log N + NB).
    """

    if family_balanced:
        return order_samples_family_balanced(samples, boundaries)
    return order_samples_by_length_bucket(samples, boundaries)


def _json_reactivity_value(value: float) -> Optional[float]:
    """Return a JSON-safe scalar, mapping NaN/inf to ``None``."""

    return float(value) if math.isfinite(float(value)) else None


def sample_to_cache_obj(
    sample: SyntheticSample,
    *,
    source_id: Optional[str] = None,
    family: Optional[str] = None,
    cluster: Optional[str] = None,
    reactivity_source: Optional[str] = None,
    window: Optional[Mapping[str, int]] = None,
    length_bucket: Optional[str] = None,
) -> dict:
    """Serialize one training sample to the eFold cache JSONL schema.

    The cache stores explicit pairs rather than the full ``L x L`` matrix to keep
    files compact.  Reactivity values use JSON ``null`` for missing positions so
    the cache is strict JSON rather than relying on non-standard ``NaN`` tokens.

    Complexity: O(L^2) due to :func:`matrix_to_pairs`.
    """

    payload = {
        "cluster": cluster if cluster is not None else sample.cluster,
        "family": family if family is not None else sample.family,
        "length_bucket": length_bucket,
        "pairs": [list(pair) for pair in matrix_to_pairs(sample.pair_matrix)],
        "probe": sample.probe,
        "reactivity": [_json_reactivity_value(value) for value in sample.reactivity],
        "reactivity_source": reactivity_source if reactivity_source is not None else sample.reactivity_source,
        "sequence": sample.sequence,
        "source_id": source_id if source_id is not None else sample.source_id,
    }
    effective_window = window
    if effective_window is None and sample.window_start is not None and sample.window_end is not None:
        effective_window = {
            "start": sample.window_start,
            "end": sample.window_end,
            **({"parent_length": sample.parent_length} if sample.parent_length is not None else {}),
        }
    if effective_window is not None:
        payload["window"] = dict(effective_window)
    if sample.reactivity_error is not None:
        payload["reactivity_error"] = [_json_reactivity_value(value) for value in sample.reactivity_error]
    if sample.reactivity_snr is not None:
        payload["snr"] = sample.reactivity_snr
    if sample.reactivity_quality is not None:
        payload["quality"] = sample.reactivity_quality
    if sample.parent_source_id is not None:
        payload["parent_source_id"] = sample.parent_source_id
    return payload


def sample_from_cache_obj(obj: dict) -> SyntheticSample:
    """Deserialize one cached eFold sample JSON object.

    Complexity: O(L^2 + P).
    """

    sequence = str(obj["sequence"]).upper()
    pairs = tuple((int(i), int(j)) for i, j in obj["pairs"])
    pair_matrix = pairs_to_matrix(pairs, len(sequence))
    reactivity = tuple(math.nan if value is None else float(value) for value in obj["reactivity"])
    probe = str(obj.get("probe") or "2A3")
    raw_family = obj.get("family") if obj.get("family") not in (None, "") else obj.get("clan")
    raw_error = obj.get("reactivity_error")
    reactivity_error = None
    if isinstance(raw_error, list):
        reactivity_error = tuple(math.nan if value is None else float(value) for value in raw_error)
    raw_window = obj.get("window") if isinstance(obj.get("window"), Mapping) else {}
    source_id = str(obj.get("source_id")) if obj.get("source_id") not in (None, "") else None
    parent_source_id = obj.get("parent_source_id")
    if parent_source_id in (None, "") and raw_window and source_id:
        parent_source_id = source_id.split(":", 1)[0]
    return SyntheticSample(
        sequence=sequence,
        dotbracket="." * len(sequence),
        pair_matrix=pair_matrix,
        partner_classes=partner_classes_from_pair_matrix(pair_matrix),
        reactivity=reactivity,
        weights=masked_unit_weights(sequence, probe, reactivity),
        probe=probe,
        source_id=source_id,
        family=str(raw_family) if raw_family not in (None, "") else None,
        cluster=str(obj.get("cluster")) if obj.get("cluster") not in (None, "") else None,
        reactivity_source=str(obj.get("reactivity_source") or "unknown"),
        reactivity_error=reactivity_error,
        reactivity_snr=float(obj["snr"]) if obj.get("snr") is not None else None,
        reactivity_quality=str(obj["quality"]) if obj.get("quality") not in (None, "") else None,
        parent_source_id=str(parent_source_id) if parent_source_id not in (None, "") else None,
        window_start=int(raw_window["start"]) if raw_window.get("start") is not None else None,
        window_end=int(raw_window["end"]) if raw_window.get("end") is not None else None,
        parent_length=int(raw_window["parent_length"]) if raw_window.get("parent_length") is not None else None,
    )


def read_sample_cache(path: Path, *, limit: Optional[int] = None) -> Tuple[SyntheticSample, ...]:
    """Read cached training samples from JSONL.

    Complexity: O(N L^2) for N cached samples.
    """

    return tuple(iter_sample_cache(path, limit=limit))


def iter_sample_cache(path: Path, *, limit: Optional[int] = None) -> Iterator[SyntheticSample]:
    """Stream cached training samples from JSONL.

    Streaming avoids a temporary full-cache tuple when callers already append
    accepted samples into their own collection.  Complexity: O(N L^2) time and
    O(L^2) transient memory per sample, excluding caller-owned results.
    """

    emitted = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield sample_from_cache_obj(json.loads(line))
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def build_efold_sample_cache(
    paths: Iterable[Path],
    output_path: Path,
    *,
    limit: Optional[int] = None,
    max_length: Optional[int] = None,
    min_length: int = 1,
    default_probe: str = "2A3",
    min_loop: int = 3,
    allow_wobble: bool = True,
    require_legal: bool = True,
    scan_limit: Optional[int] = None,
    one_based: bool = False,
    window_size: Optional[int] = None,
    window_stride: Optional[int] = None,
    length_bucket_boundaries: Sequence[int] = (),
) -> EfoldCacheSummary:
    """Materialize filtered eFold/RNAndria records into a reusable JSONL cache.

    This is the scale-up bridge between the large Dryad JSON files and repeated
    training/evaluation runs.  The expensive parsing, length filtering and
    legality checking are done once; later CLI calls can read the compact cache
    directly.  ``limit`` caps accepted samples, while ``scan_limit`` caps scanned
    source records for quick exploratory runs.

    ``window_size`` enables long-sequence training by slicing source records into
    local windows before legality checks.  ``length_bucket_boundaries`` stores
    the same monotone labels used by split manifests in each JSONL row and in the
    returned summary.

    Complexity: O(S W L_w^2) over scanned records S and emitted windows W.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scanned = accepted = skipped_length = skipped_illegal = with_reactivity = 0
    windowed_records = windows_emitted = 0
    length_buckets: Dict[str, int] = {}
    min_seen: Optional[int] = None
    max_seen: Optional[int] = None

    def summary() -> EfoldCacheSummary:
        """Freeze streaming cache counters into a summary object.

        Formula: accepted/skipped/window counts are additive sufficient
        statistics over scanned records.  Complexity: O(B), where B is the
        number of length buckets.
        """

        return EfoldCacheSummary(
            output_path=str(output_path),
            scanned=scanned,
            accepted=accepted,
            skipped_length=skipped_length,
            skipped_illegal=skipped_illegal,
            with_reactivity=with_reactivity,
            min_length=min_seen,
            max_length=max_seen,
            windowed_records=windowed_records,
            windows_emitted=windows_emitted,
            length_buckets=dict(sorted(length_buckets.items())),
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            for record in read_efold_json(Path(path), one_based=one_based):
                scanned += 1
                if scan_limit is not None and scanned > scan_limit:
                    break
                length = len(record.sequence)
                min_seen = length if min_seen is None else min(min_seen, length)
                max_seen = length if max_seen is None else max(max_seen, length)
                windows = iter_efold_record_windows(
                    record,
                    window_size=window_size,
                    window_stride=window_stride,
                )
                if any(metadata is not None for _window, metadata in windows):
                    windowed_records += 1
                for window_record, metadata in windows:
                    windows_emitted += 1
                    window_length = len(window_record.sequence)
                    if window_length < min_length or (max_length is not None and window_length > max_length):
                        skipped_length += 1
                        continue
                    try:
                        sample = sample_from_efold_record(
                            window_record,
                            default_probe=default_probe,
                            min_loop=min_loop,
                            allow_wobble=allow_wobble,
                            require_legal=require_legal,
                        )
                    except ValueError:
                        skipped_illegal += 1
                        if require_legal:
                            continue
                        raise
                    has_reactivity = window_record.shape is not None and any(
                        math.isfinite(value) for value in window_record.shape
                    )
                    if has_reactivity:
                        with_reactivity += 1
                    bucket = (
                        length_bucket_label(window_length, length_bucket_boundaries)
                        if length_bucket_boundaries
                        else None
                    )
                    if bucket is not None:
                        length_buckets[bucket] = length_buckets.get(bucket, 0) + 1
                    handle.write(
                        json.dumps(
                            sample_to_cache_obj(
                                sample,
                                source_id=window_record.record_id,
                                family=window_record.family,
                                reactivity_source="real_profile" if has_reactivity else "structure_forward_proxy",
                                window=metadata,
                                length_bucket=bucket,
                            ),
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    accepted += 1
                    if limit is not None and accepted >= limit:
                        return summary()
            if scan_limit is not None and scanned >= scan_limit:
                break
    return summary()


def load_efold_samples(
    paths: Iterable[Path],
    *,
    limit: Optional[int] = None,
    max_length: Optional[int] = None,
    min_length: int = 1,
    default_probe: str = "2A3",
    min_loop: int = 3,
    allow_wobble: bool = True,
    require_legal: bool = True,
    one_based: bool = False,
    window_size: Optional[int] = None,
    window_stride: Optional[int] = None,
    length_bucket_boundaries: Sequence[int] = (),
) -> Tuple[SyntheticSample, ...]:
    """Load eFold/RNAndria JSON files as deterministic training samples.

    Files are read in the supplied order and records are accepted until ``limit``
    samples have passed length and legality filters.  Invalid-for-training
    structures are skipped when ``require_legal=True``; malformed JSON records
    still raise inside :func:`reactflow.data.read_efold_json` so data corruption
    is visible.

    ``window_size`` mirrors :func:`build_efold_sample_cache` for direct raw-JSON
    smoke runs; cached JSONL files are assumed to have already been windowed.
    ``length_bucket_boundaries`` returns samples ordered by bucket/length, which
    makes profiling traces easier to compare across long-sequence jobs.

    Complexity: O(N W L_w^2) over scanned records because pair matrices are built
    and legality-checked.
    """

    samples: List[SyntheticSample] = []

    def finish() -> Tuple[SyntheticSample, ...]:
        """Return samples in deterministic bucket/length order.

        Formula: sorting key is ``(bucket(sequence), length, sequence)`` through
        :func:`order_samples_by_length_bucket`.  Complexity: O(N log N).
        """

        return order_samples_by_length_bucket(samples, length_bucket_boundaries)

    for path in paths:
        path = Path(path)
        if path.suffix == ".jsonl":
            for sample in iter_sample_cache(path):
                length = len(sample.sequence)
                if length < min_length:
                    continue
                if max_length is not None and length > max_length:
                    continue
                samples.append(sample)
                if limit is not None and len(samples) >= limit:
                    return finish()
            continue
        for record in read_efold_json(path, one_based=one_based):
            for window_record, _metadata in iter_efold_record_windows(
                record,
                window_size=window_size,
                window_stride=window_stride,
            ):
                length = len(window_record.sequence)
                if length < min_length:
                    continue
                if max_length is not None and length > max_length:
                    continue
                try:
                    sample = sample_from_efold_record(
                        window_record,
                        default_probe=default_probe,
                        min_loop=min_loop,
                        allow_wobble=allow_wobble,
                        require_legal=require_legal,
                    )
                except ValueError:
                    if require_legal:
                        continue
                    raise
                samples.append(sample)
                if limit is not None and len(samples) >= limit:
                    return finish()
    return finish()


def _reactivity_coefficients(
    operator: ReactivityForwardOperator,
    sequence: str,
    probe: str,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return per-position ``(a_i, c_i)`` for the minimal affine operator.

    With the minimal first-order operator (edge feature set to zero) the forward
    model is ``rhat_i = a_i q_i + c_i``.  This exposes ``a_i`` (needed for the
    reactivity gradient) and ``c_i`` (the bias).

    Complexity: O(L).
    """

    a_values: List[float] = []
    c_values: List[float] = []
    for base in sequence.upper():
        a, _b, c = operator.parameters.coefficient(probe, base)
        a_values.append(a)
        c_values.append(c)
    return tuple(a_values), tuple(c_values)


def _predicted_reactivity(marginals: Sequence[Sequence[float]], a_values: Sequence[float], c_values: Sequence[float]) -> Tuple[float, ...]:
    """Compute ``rhat_i = a_i q_i + c_i`` with ``q_i = pi_i[0]``.

    Complexity: O(L).
    """

    return tuple(a * float(row[0]) + c for row, a, c in zip(marginals, a_values, c_values))


def _reactivity_logit_gradient(
    marginals: Sequence[Sequence[float]],
    predicted: Sequence[float],
    target: Sequence[float],
    weights: Sequence[float],
    a_values: Sequence[float],
    alpha: float,
    gamma: float,
    lambda_react: float,
) -> List[List[float]]:
    """Exact gradient of ``lambda_react * ell_mag`` into every logit row.

    ``ell_mag = sum_i w_i (alpha rhat_i + gamma - r_i)^2 / sum_i w_i`` with
    ``(alpha, gamma)`` held constant.  See module docstring for the derivation.

    Complexity: O(L * K).
    """

    size = len(marginals)
    weight_sum = sum(float(w) for w in weights if float(w) > 0)
    grads: List[List[float]] = []
    for i in range(size):
        row = marginals[i]
        num_classes = len(row)
        grad_row = [0.0 for _ in range(num_classes)]
        w = float(weights[i])
        if weight_sum > 0 and w > 0:
            residual = alpha * float(predicted[i]) + gamma - float(target[i])
            d_mag = 2.0 * w * alpha * residual / weight_sum
            coeff = lambda_react * d_mag * float(a_values[i])
            q0 = float(row[0])
            for k in range(num_classes):
                indicator = 1.0 if k == 0 else 0.0
                grad_row[k] = coeff * q0 * (indicator - float(row[k]))
        grads.append(grad_row)
    return grads


def _accumulate(into: DenoiserGradients, other: DenoiserGradients) -> None:
    """Add ``other`` into ``into`` in place.

    Complexity: O(H*F + H^2).
    """

    for row, orow in zip(into.input_weight, other.input_weight):
        for index in range(len(row)):
            row[index] += orow[index]
    for index in range(len(into.input_bias)):
        into.input_bias[index] += other.input_bias[index]
    for row, orow in zip(into.pair_matrix, other.pair_matrix):
        for index in range(len(row)):
            row[index] += orow[index]
    into.pair_compat += other.pair_compat
    for index in range(len(into.unpaired_weight)):
        into.unpaired_weight[index] += other.unpaired_weight[index]
    into.unpaired_bias += other.unpaired_bias


def _zero_like(params: DenoiserParameters) -> DenoiserGradients:
    """Return a zero gradient container matching ``params``.

    Complexity: O(H*F + H^2).
    """

    return DenoiserGradients(
        input_weight=[[0.0 for _ in row] for row in params.input_weight],
        input_bias=[0.0 for _ in params.input_bias],
        pair_matrix=[[0.0 for _ in row] for row in params.pair_matrix],
        pair_compat=0.0,
        unpaired_weight=[0.0 for _ in params.unpaired_weight],
        unpaired_bias=0.0,
    )


def _sample_loss_and_grad(
    model: PairwiseDenoiser,
    operator: ReactivityForwardOperator,
    sample: SyntheticSample,
    t: float,
    rng: random.Random,
    lambda_react: float,
    lambda_thermo: float = 0.0,
    thermo_mode: str = "mse",
    target_unpaired: Optional[Sequence[float]] = None,
    adapter: Optional[FeatureAdapter] = None,
    frozen: Optional[FrozenFeatureLookup] = None,
    profiler: Optional[TrainingProfiler] = None,
    epoch: Optional[int] = None,
    sample_index: Optional[int] = None,
    lambda_calib: float = 0.0,
    calib_beta: float = 1.0,
    calib_tau_squared: float = 0.05,
    lambda_contact: float = 0.0,
    contact_negative_weight: float = 0.25,
    contact_long_range_min_distance: int = 24,
    contact_long_range_weight: float = 1.0,
) -> Tuple[float, float, float, float, float, float, float, DenoiserGradients, Optional[AdapterGradients]]:
    """Compute per-sample losses and the combined gradient at flow time ``t``.

    Returns ``(dfm, react_mag, react_shape, thermo, calib, contact, f1, gradients,
    adapter_grads)``.  The combined logit gradient injected into the denoiser is
    the sum of the exact per-position terms sharing the same rows:

        g_i[k] = dfm_grad_i[k] + react_grad_i[k] + thermo_grad_i[k]
                 + calib_grad_i[k] + contact_grad_i[k].

    When ``lambda_thermo == 0.0`` the thermodynamic branch is skipped entirely,
    and when ``lambda_calib == 0.0`` / ``lambda_contact == 0.0`` the corresponding
    branches are skipped entirely, so the gradient, the consumed ``rng`` stream
    and the returned diagnostics are bit-for-bit identical to the C3 pilot.

    When ``adapter`` is ``None`` the feature vector is exactly the C3
    ``FEATURE_SIZE`` hand-written encoding and ``adapter_grads`` is ``None``.  When
    an ``adapter`` is supplied, its projected frozen features are concatenated onto
    the base features and the adapter parameter gradient is obtained from the
    trailing block of the denoiser's ``dL/dfeat_i`` via the exact chain rule.

    Complexity: O(L^2 H^2 + L d_adapter d_single).
    """

    sequence = sample.sequence
    size = len(sequence)
    num_classes = size + 1
    phase_start = time.perf_counter()
    source = uniform_source(num_classes)
    noised_classes = [
        sample_path_index(t, sample.partner_classes[i], source, rng=rng) for i in range(size)
    ]
    base_features = build_features(sequence, t, noised_classes)
    if adapter is not None:
        single_rows = frozen.single_rows(sequence) if frozen is not None else None
        features, used_single = build_augmented_features(base_features, adapter, single_rows)
    else:
        features, used_single = base_features, None
    if profiler is not None:
        profiler.log("path_sample_features", time.perf_counter() - phase_start, epoch=epoch, sample_index=sample_index, length=size)

    phase_start = time.perf_counter()
    forward = model.forward(sequence, features)
    if profiler is not None:
        profiler.log("model_forward", time.perf_counter() - phase_start, epoch=epoch, sample_index=sample_index, length=size)

    phase_start = time.perf_counter()
    dfm_grad_rows = [
        list(softmax_cross_entropy_gradient(forward.logits[i], sample.partner_classes[i]))
        for i in range(size)
    ]
    dfm_scale = 1.0 / size
    dfm_loss = 0.0
    for i in range(size):
        target = sample.partner_classes[i]
        dfm_loss += -_safe_log(forward.marginals[i][target])
        for k in range(num_classes):
            dfm_grad_rows[i][k] *= dfm_scale
    dfm_loss /= size
    if profiler is not None:
        profiler.log("dfm_loss_grad", time.perf_counter() - phase_start, epoch=epoch, sample_index=sample_index, length=size)

    phase_start = time.perf_counter()
    a_values, c_values = _reactivity_coefficients(operator, sequence, sample.probe)
    predicted = _predicted_reactivity(forward.marginals, a_values, c_values)
    alpha, gamma = fit_weighted_affine_calibration(predicted, sample.reactivity, sample.weights)
    calibrated = tuple(alpha * value + gamma for value in predicted)
    react_mag = weighted_mse(calibrated, sample.reactivity, sample.weights)
    react_shape = 1.0 - weighted_pearson(predicted, sample.reactivity, sample.weights)

    react_grad_rows = _reactivity_logit_gradient(
        forward.marginals,
        predicted,
        sample.reactivity,
        sample.weights,
        a_values,
        alpha,
        gamma,
        lambda_react,
    )
    if profiler is not None:
        profiler.log("reactivity_loss_grad", time.perf_counter() - phase_start, epoch=epoch, sample_index=sample_index, length=size)

    grad_logits = [
        [dfm_grad_rows[i][k] + react_grad_rows[i][k] for k in range(num_classes)]
        for i in range(size)
    ]

    phase_start = time.perf_counter()
    thermo_loss = 0.0
    if lambda_thermo != 0.0 and target_unpaired is not None:
        model_unpaired = unpaired_probabilities(forward.marginals)
        if thermo_mode == "kl":
            thermo_loss = thermo_unpaired_kl(model_unpaired, target_unpaired)
        else:
            thermo_loss = thermo_unpaired_mse(model_unpaired, target_unpaired)
        thermo_grad_rows = thermo_logit_gradient(
            forward.marginals,
            target_unpaired,
            lambda_thermo,
            mode=thermo_mode,
        )
        for i in range(size):
            for k in range(num_classes):
                grad_logits[i][k] += thermo_grad_rows[i][k]
    if profiler is not None:
        profiler.log("thermo_loss_grad", time.perf_counter() - phase_start, epoch=epoch, sample_index=sample_index, length=size)

    phase_start = time.perf_counter()
    calib_loss = 0.0
    if lambda_calib != 0.0:
        model_unpaired = unpaired_probabilities(forward.marginals)
        calib_loss = heteroscedastic_reactivity_nll(
            model_unpaired,
            sample.reactivity,
            sample.weights,
            a_values,
            c_values,
            alpha=alpha,
            gamma=gamma,
            beta=calib_beta,
            tau_squared=calib_tau_squared,
        )
        calib_grad_rows = heteroscedastic_reactivity_logit_gradient(
            forward.marginals,
            sample.reactivity,
            sample.weights,
            a_values,
            c_values,
            alpha=alpha,
            gamma=gamma,
            beta=calib_beta,
            tau_squared=calib_tau_squared,
            lambda_calib=lambda_calib,
        )
        for i in range(size):
            for k in range(num_classes):
                grad_logits[i][k] += calib_grad_rows[i][k]
    if profiler is not None:
        profiler.log("calib_loss_grad", time.perf_counter() - phase_start, epoch=epoch, sample_index=sample_index, length=size)

    phase_start = time.perf_counter()
    contact_loss = 0.0
    if lambda_contact != 0.0:
        contact_loss = contact_denoising_bce(
            forward.marginals,
            sample.pair_matrix,
            forward.legal_pair,
            negative_weight=contact_negative_weight,
            long_range_min_distance=contact_long_range_min_distance,
            long_range_weight=contact_long_range_weight,
        )
        contact_grad_rows = contact_denoising_logit_gradient(
            forward.marginals,
            sample.pair_matrix,
            forward.legal_pair,
            lambda_contact=lambda_contact,
            negative_weight=contact_negative_weight,
            long_range_min_distance=contact_long_range_min_distance,
            long_range_weight=contact_long_range_weight,
        )
        for i in range(size):
            for k in range(num_classes):
                grad_logits[i][k] += contact_grad_rows[i][k]
    if profiler is not None:
        profiler.log("contact_loss_grad", time.perf_counter() - phase_start, epoch=epoch, sample_index=sample_index, length=size)

    phase_start = time.perf_counter()
    gradients = model.backward(forward, features, grad_logits)
    if profiler is not None:
        profiler.log("model_backward", time.perf_counter() - phase_start, epoch=epoch, sample_index=sample_index, length=size)

    phase_start = time.perf_counter()
    adapter_grads: Optional[AdapterGradients] = None
    if adapter is not None and used_single is not None and gradients.grad_features is not None:
        grad_adapter_output = split_feature_gradient(gradients.grad_features, FEATURE_SIZE)
        adapter_grads = adapter.backward(used_single, grad_adapter_output)
    if profiler is not None:
        profiler.log("adapter_backward", time.perf_counter() - phase_start, epoch=epoch, sample_index=sample_index, length=size)

    phase_start = time.perf_counter()
    soft_matrix = marginal_pair_matrix(forward.marginals)
    projected = project_greedy_matching(
        sequence,
        soft_matrix,
        min_loop=model.min_loop,
        allow_wobble=model.allow_wobble,
        allow_pseudoknot=True,
        min_score=1e-6,
    )
    f1 = f1_score(projected, sample.pair_matrix)
    if profiler is not None:
        profiler.log("projection_f1", time.perf_counter() - phase_start, epoch=epoch, sample_index=sample_index, length=size)
    dfm_loss = _assert_finite_training_scalar(dfm_loss, "dfm_loss", epoch=epoch, sample_index=sample_index)
    react_mag = _assert_finite_training_scalar(react_mag, "react_magnitude", epoch=epoch, sample_index=sample_index)
    react_shape = _assert_finite_training_scalar(react_shape, "react_shape", epoch=epoch, sample_index=sample_index)
    thermo_loss = _assert_finite_training_scalar(thermo_loss, "thermo_loss", epoch=epoch, sample_index=sample_index)
    calib_loss = _assert_finite_training_scalar(calib_loss, "calib_loss", epoch=epoch, sample_index=sample_index)
    contact_loss = _assert_finite_training_scalar(contact_loss, "contact_loss", epoch=epoch, sample_index=sample_index)
    f1 = _assert_finite_training_scalar(f1, "projection_f1", epoch=epoch, sample_index=sample_index)
    return dfm_loss, react_mag, react_shape, thermo_loss, calib_loss, contact_loss, f1, gradients, adapter_grads


def _safe_log(value: float) -> float:
    """Return ``log`` clamped away from zero for stable loss reporting.

    Complexity: O(1).
    """

    import math

    return math.log(max(value, 1e-12))


def _assert_finite_training_scalar(
    value: float,
    label: str,
    *,
    epoch: Optional[int] = None,
    sample_index: Optional[int] = None,
) -> float:
    """Return ``value`` after rejecting NaN/Inf training diagnostics.

    Formula: the guard accepts a scalar ``s`` iff ``isfinite(s)``.  This does not
    change the optimized objective for finite trajectories; it only turns an
    invalid numerical state into a deterministic :class:`FloatingPointError` so
    the outer full-run watcher can reduce batch size and retry.

    Complexity: O(1).
    """

    numeric = float(value)
    if math.isfinite(numeric):
        return numeric
    where = []
    if epoch is not None:
        where.append(f"epoch={epoch}")
    if sample_index is not None:
        where.append(f"sample_index={sample_index}")
    suffix = "" if not where else " " + " ".join(where)
    raise FloatingPointError(f"non-finite training value: {label}={numeric}{suffix}")


def _prefetch_frozen_batch(
    frozen: Optional[FrozenFeatureLookup],
    samples: Sequence[SyntheticSample],
    start: int,
    batch_size: int,
    profiler: Optional[TrainingProfiler],
    epoch: int,
) -> None:
    """Prefetch frozen rows for the next mini-batch when a lazy lookup exists.

    Formula: let ``B = samples[start : start + batch_size]``.  The lookup groups
    ``{sequence_b | b in B}`` by frozen child shard and reads selected
    ``single`` members into its bounded LRU cache.  Subsequent per-sample feature
    construction consumes exactly the cached rows that it would otherwise read
    one sequence at a time, so model outputs and gradients are unchanged.

    Complexity: O(|B| + selected L*d_single), plus one verified shard hash on
    the first access to each child shard.
    """

    if frozen is None or batch_size <= 0 or start >= len(samples):
        return
    stop = min(len(samples), start + batch_size)
    if stop <= start:
        return
    phase_start = time.perf_counter()
    frozen.prefetch(sample.sequence for sample in samples[start:stop])
    if profiler is not None:
        profiler.log(
            "frozen_batch_prefetch",
            time.perf_counter() - phase_start,
            epoch=epoch,
            sample_index=start,
            length=sum(len(sample.sequence) for sample in samples[start:stop]),
        )


def train_pilot(
    samples: Optional[Sequence[SyntheticSample]] = None,
    config: Optional[TrainConfig] = None,
    frozen: Optional[FrozenFeatureLookup] = None,
) -> TrainingResult:
    """Run the deterministic end-to-end pilot training loop.

    The loop performs, per epoch, a full pass over the dataset, drawing one flow
    time per sample and accumulating the combined ``L_DFM + lambda_r * L_react +
    lambda_td * ell_thermo`` gradient before a single SGD step (batch gradient
    descent).  Everything is seeded so the loss trajectory is identical across
    platforms.  When ``config.lambda_thermo == 0.0`` the Turner prior is not even
    computed, so the trajectory is bit-for-bit the C3 pilot.

    When ``config.adapter_dim > 0`` a linear :class:`FeatureAdapter` projects the
    frozen per-nucleotide representations (looked up per sequence in ``frozen``)
    into ``adapter_dim`` extra feature slots, and its parameters are trained
    jointly with the denoiser by the exact hand-written chain rule.  Sequences
    with no frozen record fall back to a zero adapter input.  With ``adapter_dim
    == 0`` (default) the adapter machinery is bypassed entirely and the trajectory
    is bit-for-bit the C3 pilot regardless of ``frozen``.

    Complexity: ``O(epochs * N * L^2 H^2)`` plus a one-time ``O(N L^3)`` Turner
    prior precomputation when ``lambda_thermo != 0.0``.
    """

    config = config or TrainConfig()
    if samples is None:
        samples = make_dataset(count=6, stem=4, loop=4, probe="2A3", seed=1)
    if not samples:
        raise ValueError("training requires at least one sample")
    samples = order_samples_for_training(
        samples,
        config.length_bucket_boundaries,
        family_balanced=config.family_balanced_batches,
    )

    use_adapter = config.adapter_dim > 0
    adapter: Optional[FeatureAdapter] = None
    if use_adapter:
        if frozen is None:
            raise ValueError("adapter_dim > 0 requires a frozen feature lookup")
        adapter_params = AdapterParameters.random_init(
            frozen.d_single, config.adapter_dim, seed=config.seed
        )
        adapter = FeatureAdapter(adapter_params)
        feature_size = FEATURE_SIZE + config.adapter_dim
    else:
        feature_size = FEATURE_SIZE

    params = DenoiserParameters.random_init(feature_size, config.hidden_size, seed=config.seed)
    model = PairwiseDenoiser(params, min_loop=config.min_loop)
    operator = ReactivityForwardOperator()
    rng = random.Random(config.seed + 101)
    adapter_lr = config.adapter_lr if config.adapter_lr is not None else config.learning_rate
    profiler = TrainingProfiler(Path(config.profile_path)) if config.profile_path else None
    batch_size = len(samples) if config.batch_size is None else int(config.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive when set")

    def new_adapter_accumulator() -> Optional[AdapterGradients]:
        """Allocate zero adapter-gradient accumulators for one batch.

        Formula: accumulator tensors match ``d_adapter x d_single`` weights and
        ``d_adapter`` bias entries.  Complexity: O(d_adapter * d_single).
        """

        if not use_adapter:
            return None
        return AdapterGradients(
            weight=[[0.0 for _ in row] for row in adapter.parameters.weight],
            bias=[0.0 for _ in adapter.parameters.bias],
        )

    def apply_batch_update(
        accumulated: DenoiserGradients,
        adapter_accumulated: Optional[AdapterGradients],
        count: int,
        batch_length: int,
        epoch: int,
    ) -> None:
        """Average and apply one accumulated stdlib SGD mini-batch.

        Formula: ``g_bar = (1/count) * sum_b g_b`` and
        ``theta <- theta - lr * g_bar`` for denoiser and optional adapter
        parameters.  Complexity: O(parameter_count).
        """

        phase_start = time.perf_counter()
        scale = 1.0 / count
        for row in accumulated.input_weight:
            for index in range(len(row)):
                row[index] *= scale
        for index in range(len(accumulated.input_bias)):
            accumulated.input_bias[index] *= scale
        for row in accumulated.pair_matrix:
            for index in range(len(row)):
                row[index] *= scale
        accumulated.pair_compat *= scale
        for index in range(len(accumulated.unpaired_weight)):
            accumulated.unpaired_weight[index] *= scale
        accumulated.unpaired_bias *= scale
        sgd_update(params, accumulated, config.learning_rate)

        if adapter is not None and adapter_accumulated is not None:
            for row in adapter_accumulated.weight:
                for index in range(len(row)):
                    row[index] *= scale
            for index in range(len(adapter_accumulated.bias)):
                adapter_accumulated.bias[index] *= scale
            adapter_sgd_update(adapter.parameters, adapter_accumulated, adapter_lr)
        if profiler is not None:
            profiler.log(
                "gradient_average_update",
                time.perf_counter() - phase_start,
                epoch=epoch,
                length=batch_length,
            )

    thermo_priors: List[Optional[Tuple[float, ...]]] = [None for _ in samples]
    if config.lambda_thermo != 0.0:
        phase_start = time.perf_counter()
        thermo_priors = [
            monte_carlo_unpaired_prior(
                sample.sequence,
                samples=config.thermo_samples,
                seed=config.seed,
                min_loop=config.min_loop,
                allow_wobble=model.allow_wobble,
            )
            for sample in samples
        ]
        if profiler is not None:
            profiler.log("thermo_prior_precompute", time.perf_counter() - phase_start, length=sum(len(s.sequence) for s in samples))

    history: List[EpochRecord] = []
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        accumulated = _zero_like(params)
        adapter_accumulated = new_adapter_accumulator()
        batch_count = 0
        batch_length = 0
        dfm_total = react_total = shape_total = thermo_total = calib_total = contact_total = f1_total = 0.0
        for sample_index, (sample, prior) in enumerate(zip(samples, thermo_priors)):
            if use_adapter and batch_count == 0 and batch_size < len(samples):
                _prefetch_frozen_batch(frozen, samples, sample_index, batch_size, profiler, epoch)
            t = 0.05 + 0.9 * rng.random()
            dfm, react_mag, react_shape, thermo, calib, contact, f1, grads, adapter_grads = _sample_loss_and_grad(
                model,
                operator,
                sample,
                t,
                rng,
                config.lambda_react,
                config.lambda_thermo,
                config.thermo_mode,
                prior,
                adapter,
                frozen,
                profiler,
                epoch,
                sample_index,
                config.lambda_calib,
                config.calib_beta,
                config.calib_tau_squared,
                config.lambda_contact,
                config.contact_negative_weight,
                config.contact_long_range_min_distance,
                config.contact_long_range_weight,
            )
            _accumulate(accumulated, grads)
            if adapter_accumulated is not None and adapter_grads is not None:
                for prow, grow in zip(adapter_accumulated.weight, adapter_grads.weight):
                    for index in range(len(prow)):
                        prow[index] += grow[index]
                for index in range(len(adapter_accumulated.bias)):
                    adapter_accumulated.bias[index] += adapter_grads.bias[index]
            dfm_total += dfm
            react_total += react_mag
            shape_total += react_shape
            thermo_total += thermo
            calib_total += calib
            contact_total += contact
            f1_total += f1
            batch_count += 1
            batch_length += len(sample.sequence)
            if batch_count >= batch_size or sample_index == len(samples) - 1:
                apply_batch_update(accumulated, adapter_accumulated, batch_count, batch_length, epoch)
                accumulated = _zero_like(params)
                adapter_accumulated = new_adapter_accumulator()
                batch_count = 0
                batch_length = 0
        count = len(samples)
        scale = 1.0 / count
        total = (
            dfm_total * scale
            + config.lambda_react * react_total * scale
            + config.lambda_thermo * thermo_total * scale
            + config.lambda_calib * calib_total * scale
            + config.lambda_contact * contact_total * scale
        )
        total = _assert_finite_training_scalar(total, "epoch_total", epoch=epoch)
        history.append(
            EpochRecord(
                epoch=epoch,
                total=total,
                dfm=dfm_total * scale,
                react_magnitude=react_total * scale,
                react_shape=shape_total * scale,
                thermo=thermo_total * scale,
                mean_f1=f1_total * scale,
                calib=calib_total * scale,
                contact=contact_total * scale,
            )
        )
        if profiler is not None:
            profiler.log("epoch_total", time.perf_counter() - epoch_start, epoch=epoch, length=sum(len(s.sequence) for s in samples))

    profile_summary = profiler.close() if profiler is not None else None
    return TrainingResult(
        parameters=params,
        history=tuple(history),
        adapter_parameters=adapter.parameters if adapter is not None else None,
        profile_summary=profile_summary,
    )


def train_pilot_torch(
    samples: Optional[Sequence[SyntheticSample]] = None,
    config: Optional[TrainConfig] = None,
    *,
    frozen: Optional[FrozenFeatureLookup] = None,
    device: str = "cpu",
) -> TrainingResult:
    """Run the base training loop with a lazy optional PyTorch backend.

    This backend keeps the public ``TrainConfig``/``TrainingResult`` contract but
    moves the dense denoiser forward, loss, autograd backward and SGD update onto
    torch tensors.  It intentionally does not import torch unless this function
    is called, preserving the package-level standard-library import invariant.

    Current scope: DFM + reactivity loss, with optional frozen-feature adapter
    warm-start.  Turner thermo branches still use :func:`train_pilot`, where
    their hand-written gradients are already validated.

    Complexity: same asymptotic ``O(epochs * N * L^2 H^2)`` model, but dense
    bilinear pair scoring/backward is vectorized by torch.
    """

    try:
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError("torch backend requires optional dependency PyTorch") from exc

    config = config or TrainConfig()
    if config.lambda_thermo != 0.0:
        raise ValueError("torch backend currently supports lambda_thermo=0 only")
    use_adapter = config.adapter_dim > 0
    if use_adapter and frozen is None:
        raise ValueError("torch backend adapter_dim > 0 requires a frozen feature lookup")
    if samples is None:
        samples = make_dataset(count=6, stem=4, loop=4, probe="2A3", seed=1)
    if not samples:
        raise ValueError("training requires at least one sample")
    samples = order_samples_for_training(
        samples,
        config.length_bucket_boundaries,
        family_balanced=config.family_balanced_batches,
    )
    batch_size = len(samples) if config.batch_size is None else int(config.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive when set")

    adapter_params: Optional[AdapterParameters] = None
    if use_adapter:
        adapter_params = AdapterParameters.random_init(frozen.d_single, config.adapter_dim, seed=config.seed)
        feature_size = FEATURE_SIZE + config.adapter_dim
    else:
        feature_size = FEATURE_SIZE

    params = DenoiserParameters.random_init(feature_size, config.hidden_size, seed=config.seed)
    dtype = torch.float64
    torch_device = torch.device(device)
    input_weight = torch.tensor(params.input_weight, dtype=dtype, device=torch_device, requires_grad=True)
    input_bias = torch.tensor(params.input_bias, dtype=dtype, device=torch_device, requires_grad=True)
    pair_matrix_param = torch.tensor(params.pair_matrix, dtype=dtype, device=torch_device, requires_grad=True)
    pair_compat = torch.tensor(float(params.pair_compat), dtype=dtype, device=torch_device, requires_grad=True)
    unpaired_weight = torch.tensor(params.unpaired_weight, dtype=dtype, device=torch_device, requires_grad=True)
    unpaired_bias = torch.tensor(float(params.unpaired_bias), dtype=dtype, device=torch_device, requires_grad=True)
    tensors = [input_weight, input_bias, pair_matrix_param, pair_compat, unpaired_weight, unpaired_bias]
    tensor_names = ["input_weight", "input_bias", "pair_matrix", "pair_compat", "unpaired_weight", "unpaired_bias"]
    adapter_weight = adapter_bias = None
    if adapter_params is not None:
        adapter_weight = torch.tensor(adapter_params.weight, dtype=dtype, device=torch_device, requires_grad=True)
        adapter_bias = torch.tensor(adapter_params.bias, dtype=dtype, device=torch_device, requires_grad=True)
        tensors.extend([adapter_weight, adapter_bias])
        tensor_names.extend(["adapter_weight", "adapter_bias"])

    model_for_masks = PairwiseDenoiser(params, min_loop=config.min_loop)
    operator = ReactivityForwardOperator()
    rng = random.Random(config.seed + 101)
    profiler = TrainingProfiler(Path(config.profile_path)) if config.profile_path else None
    history: List[EpochRecord] = []

    def tensor_to_nested(tensor) -> List[List[float]]:
        """Convert a 2-D torch tensor into JSON-serializable Python lists.

        Formula: identity copy from device tensor to CPU scalar floats.
        Complexity: O(rows * cols).
        """

        return [[float(value) for value in row] for row in tensor.detach().cpu().tolist()]

    def tensor_to_vector(tensor) -> List[float]:
        """Convert a 1-D torch tensor into JSON-serializable Python floats.

        Formula: identity copy from device tensor to CPU scalar floats.
        Complexity: O(n).
        """

        return [float(value) for value in tensor.detach().cpu().tolist()]

    def finite_tensor_scalar(tensor, label: str, epoch: int, sample_index: Optional[int] = None) -> float:
        """Return a detached scalar value after the shared finite-value guard.

        Formula: for scalar torch value ``s``, validate ``isfinite(float(s))`` on
        the detached CPU copy.  The returned float is used only for reporting
        totals; the original tensor remains in the autograd graph for backward.

        Complexity: O(1).
        """

        return _assert_finite_training_scalar(
            float(tensor.detach().cpu()),
            label,
            epoch=epoch,
            sample_index=sample_index,
        )

    def assert_finite_tensor_values(tensor, label: str, epoch: int) -> None:
        """Reject non-finite torch gradients or parameter tensors.

        Formula: for tensor ``T`` with entries ``T_j``, accept iff
        ``prod_j 1[isfinite(T_j)] = 1``.  This is evaluated on a detached view and
        therefore does not alter the autograd graph; it only makes gradient or
        parameter corruption observable to the outer retry watcher.

        Complexity: O(numel(T)).
        """

        finite_mask = torch.isfinite(tensor.detach())
        if bool(finite_mask.all().detach().cpu()):
            return
        bad_count = int((~finite_mask).sum().detach().cpu())
        raise FloatingPointError(f"non-finite training tensor: {label} bad_count={bad_count} epoch={epoch}")

    try:
        for epoch in range(config.epochs):
            epoch_start = time.perf_counter()
            losses = []
            batch_count = 0
            batch_length = 0
            dfm_total = react_total = shape_total = calib_total = contact_total = f1_total = 0.0
            total_length = 0
            for sample_index, sample in enumerate(samples):
                if use_adapter and batch_count == 0 and batch_size < len(samples):
                    _prefetch_frozen_batch(frozen, samples, sample_index, batch_size, profiler, epoch)
                sequence = sample.sequence
                size = len(sequence)
                total_length += size
                batch_length += size
                num_classes = size + 1

                phase_start = time.perf_counter()
                t = 0.05 + 0.9 * rng.random()
                source = uniform_source(num_classes)
                noised_classes = [
                    sample_path_index(t, sample.partner_classes[i], source, rng=rng)
                    for i in range(size)
                ]
                features = build_features(sequence, t, noised_classes)
                feature_tensor = torch.tensor(features, dtype=dtype, device=torch_device)
                if adapter_weight is not None and adapter_bias is not None:
                    single_rows = frozen.single_rows(sequence) if frozen is not None else None
                    if single_rows is None:
                        single_tensor = torch.zeros(
                            (size, adapter_weight.shape[1]),
                            dtype=dtype,
                            device=torch_device,
                        )
                    else:
                        single_tensor = torch.tensor(single_rows, dtype=dtype, device=torch_device)
                    adapter_output = single_tensor.matmul(adapter_weight.t()) + adapter_bias
                    feature_tensor = torch.cat((feature_tensor, adapter_output), dim=1)
                target_tensor = torch.tensor(sample.partner_classes, dtype=torch.long, device=torch_device)
                if profiler is not None:
                    profiler.log(
                        "path_sample_features",
                        time.perf_counter() - phase_start,
                        epoch=epoch,
                        sample_index=sample_index,
                        length=size,
                    )

                phase_start = time.perf_counter()
                hidden = torch.tanh(feature_tensor.matmul(input_weight.t()) + input_bias)
                legal, compat = model_for_masks._legality(sequence)
                legal_tensor = torch.tensor(legal, dtype=torch.bool, device=torch_device)
                compat_tensor = torch.tensor(compat, dtype=dtype, device=torch_device)
                pair_scores = hidden.matmul(pair_matrix_param).matmul(hidden.t()) + pair_compat * compat_tensor
                pair_logits = torch.where(
                    legal_tensor,
                    pair_scores,
                    torch.full((size, size), -1.0e9, dtype=dtype, device=torch_device),
                )
                unpaired_logits = hidden.matmul(unpaired_weight) + unpaired_bias
                logits = torch.cat((unpaired_logits.unsqueeze(1), pair_logits), dim=1)
                marginals = torch.softmax(logits, dim=1)
                if profiler is not None:
                    profiler.log(
                        "model_forward",
                        time.perf_counter() - phase_start,
                        epoch=epoch,
                        sample_index=sample_index,
                        length=size,
                    )

                phase_start = time.perf_counter()
                dfm_loss = torch_functional.cross_entropy(logits, target_tensor, reduction="mean")
                dfm_total += finite_tensor_scalar(dfm_loss, "dfm_loss", epoch, sample_index)
                if profiler is not None:
                    profiler.log(
                        "dfm_loss_grad",
                        time.perf_counter() - phase_start,
                        epoch=epoch,
                        sample_index=sample_index,
                        length=size,
                    )

                phase_start = time.perf_counter()
                a_values, c_values = _reactivity_coefficients(operator, sequence, sample.probe)
                a_tensor = torch.tensor(a_values, dtype=dtype, device=torch_device)
                c_tensor = torch.tensor(c_values, dtype=dtype, device=torch_device)
                predicted_tensor = a_tensor * marginals[:, 0] + c_tensor
                predicted_detached = tuple(float(value) for value in predicted_tensor.detach().cpu().tolist())
                alpha, gamma = fit_weighted_affine_calibration(
                    predicted_detached,
                    sample.reactivity,
                    sample.weights,
                )
                target_values = [0.0 if not math.isfinite(float(value)) else float(value) for value in sample.reactivity]
                target_reactivity = torch.tensor(target_values, dtype=dtype, device=torch_device)
                weights = torch.tensor(sample.weights, dtype=dtype, device=torch_device)
                weight_sum = weights.sum()
                calibrated = float(alpha) * predicted_tensor + float(gamma)
                if float(weight_sum.detach().cpu()) > 0.0:
                    react_loss = torch.sum(weights * (calibrated - target_reactivity) ** 2) / weight_sum
                else:
                    react_loss = torch.zeros((), dtype=dtype, device=torch_device)
                react_total += finite_tensor_scalar(react_loss, "reactivity_loss", epoch, sample_index)
                shape_total += _assert_finite_training_scalar(
                    1.0 - weighted_pearson(predicted_detached, sample.reactivity, sample.weights),
                    "react_shape",
                    epoch=epoch,
                    sample_index=sample_index,
                )
                if profiler is not None:
                    profiler.log(
                        "reactivity_loss_grad",
                        time.perf_counter() - phase_start,
                        epoch=epoch,
                        sample_index=sample_index,
                        length=size,
                    )

                phase_start = time.perf_counter()
                calib_loss = torch.zeros((), dtype=dtype, device=torch_device)
                if config.lambda_calib != 0.0:
                    q_tensor = marginals[:, 0]
                    variance = (
                        float(config.calib_beta) * a_tensor * a_tensor * q_tensor * (1.0 - q_tensor)
                        + max(float(config.calib_tau_squared), 1e-6)
                    )
                    if float(weight_sum.detach().cpu()) > 0.0:
                        calib_terms = (calibrated - target_reactivity) ** 2 / (2.0 * variance) + 0.5 * torch.log(variance)
                        calib_loss = torch.sum(weights * calib_terms) / weight_sum
                    calib_total += finite_tensor_scalar(calib_loss, "calib_loss", epoch, sample_index)
                if profiler is not None:
                    profiler.log(
                        "calib_loss_grad",
                        time.perf_counter() - phase_start,
                        epoch=epoch,
                        sample_index=sample_index,
                        length=size,
                    )

                phase_start = time.perf_counter()
                contact_loss = torch.zeros((), dtype=dtype, device=torch_device)
                if config.lambda_contact != 0.0:
                    row_indices, col_indices = torch.triu_indices(size, size, offset=1, device=torch_device)
                    legal_candidates = legal_tensor[row_indices, col_indices] & legal_tensor[col_indices, row_indices]
                    target_matrix = torch.tensor(sample.pair_matrix, dtype=dtype, device=torch_device)
                    target_contacts = target_matrix[row_indices, col_indices]
                    if bool(((target_contacts > 0.5) & ~legal_candidates).any().detach().cpu()):
                        raise ValueError("target pair is not legal under the current mask")
                    if bool(legal_candidates.any().detach().cpu()):
                        if config.contact_long_range_min_distance < 1:
                            raise ValueError("contact_long_range_min_distance must be positive")
                        if config.contact_long_range_weight < 0.0:
                            raise ValueError("contact_long_range_weight must be non-negative")
                        distances = col_indices - row_indices
                        candidate_weights = torch.where(
                            distances >= int(config.contact_long_range_min_distance),
                            torch.full_like(distances, float(config.contact_long_range_weight), dtype=dtype),
                            torch.ones_like(distances, dtype=dtype),
                        )
                        probs = 0.5 * (
                            marginals[row_indices, col_indices + 1]
                            + marginals[col_indices, row_indices + 1]
                        )
                        probs = torch.clamp(probs[legal_candidates], 1e-6, 1.0 - 1e-6)
                        targets = target_contacts[legal_candidates]
                        weights_for_candidates = candidate_weights[legal_candidates]
                        pos_mask = targets > 0.5
                        neg_mask = ~pos_mask
                        parts = []
                        if bool(pos_mask.any().detach().cpu()):
                            pos_weights = weights_for_candidates[pos_mask]
                            pos_denom = torch.clamp(pos_weights.sum(), min=1e-12)
                            parts.append((pos_weights * -torch.log(probs[pos_mask])).sum() / pos_denom)
                        if bool(neg_mask.any().detach().cpu()) and config.contact_negative_weight != 0.0:
                            neg_weights = weights_for_candidates[neg_mask]
                            neg_denom = torch.clamp(neg_weights.sum(), min=1e-12)
                            parts.append(
                                float(config.contact_negative_weight)
                                * (neg_weights * -torch.log(1.0 - probs[neg_mask])).sum()
                                / neg_denom
                            )
                        if parts:
                            contact_loss = sum(parts)
                    contact_total += finite_tensor_scalar(contact_loss, "contact_loss", epoch, sample_index)
                if profiler is not None:
                    profiler.log(
                        "contact_loss_grad",
                        time.perf_counter() - phase_start,
                        epoch=epoch,
                        sample_index=sample_index,
                        length=size,
                    )

                phase_start = time.perf_counter()
                soft_matrix = marginal_pair_matrix(
                    tuple(tuple(float(value) for value in row) for row in marginals.detach().cpu().tolist())
                )
                projected = project_greedy_matching(
                    sequence,
                    soft_matrix,
                    min_loop=config.min_loop,
                    allow_wobble=model_for_masks.allow_wobble,
                    allow_pseudoknot=True,
                    min_score=1e-6,
                )
                f1_total += _assert_finite_training_scalar(
                    f1_score(projected, sample.pair_matrix),
                    "projection_f1",
                    epoch=epoch,
                    sample_index=sample_index,
                )
                if profiler is not None:
                    profiler.log(
                        "projection_f1",
                        time.perf_counter() - phase_start,
                        epoch=epoch,
                        sample_index=sample_index,
                        length=size,
                    )

                losses.append(
                    dfm_loss
                    + config.lambda_react * react_loss
                    + config.lambda_calib * calib_loss
                    + config.lambda_contact * contact_loss
                )
                batch_count += 1
                if batch_count >= batch_size or sample_index == len(samples) - 1:
                    phase_start = time.perf_counter()
                    batch_loss = sum(losses) / batch_count
                    finite_tensor_scalar(batch_loss, "batch_loss", epoch)
                    batch_loss.backward()
                    with torch.no_grad():
                        for name, tensor in zip(tensor_names, tensors):
                            if tensor.grad is not None:
                                assert_finite_tensor_values(tensor.grad, f"gradient:{name}", epoch)
                                tensor -= config.learning_rate * tensor.grad
                                assert_finite_tensor_values(tensor, f"parameter:{name}", epoch)
                                tensor.grad.zero_()
                    if profiler is not None:
                        profiler.log(
                            "torch_backward_update",
                            time.perf_counter() - phase_start,
                            epoch=epoch,
                            length=batch_length,
                        )
                    losses = []
                    batch_count = 0
                    batch_length = 0

            scale = 1.0 / len(samples)
            total = _assert_finite_training_scalar(
                dfm_total * scale
                + config.lambda_react * react_total * scale
                + config.lambda_calib * calib_total * scale
                + config.lambda_contact * contact_total * scale,
                "epoch_total",
                epoch=epoch,
            )
            history.append(
                EpochRecord(
                    epoch=epoch,
                    total=total,
                    dfm=dfm_total * scale,
                    react_magnitude=react_total * scale,
                    react_shape=shape_total * scale,
                    thermo=0.0,
                    mean_f1=f1_total * scale,
                    calib=calib_total * scale,
                    contact=contact_total * scale,
                )
            )
            if profiler is not None:
                profiler.log("epoch_total", time.perf_counter() - epoch_start, epoch=epoch, length=total_length)
    finally:
        profile_summary = profiler.close() if profiler is not None else None

    trained = DenoiserParameters(
        input_weight=tensor_to_nested(input_weight),
        input_bias=tensor_to_vector(input_bias),
        pair_matrix=tensor_to_nested(pair_matrix_param),
        pair_compat=float(pair_compat.detach().cpu()),
        unpaired_weight=tensor_to_vector(unpaired_weight),
        unpaired_bias=float(unpaired_bias.detach().cpu()),
    )
    trained_adapter = None
    if adapter_weight is not None and adapter_bias is not None:
        trained_adapter = AdapterParameters(
            weight=tensor_to_nested(adapter_weight),
            bias=tensor_to_vector(adapter_bias),
        )
    return TrainingResult(
        parameters=trained,
        history=tuple(history),
        adapter_parameters=trained_adapter,
        profile_summary=profile_summary,
    )
