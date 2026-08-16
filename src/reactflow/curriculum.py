"""Data-diversity curriculum sampler for C1-3 full-scale training.

Spec reference: ReactFlow分阶段执行提示词.md lines 512-527 (data-diversity curriculum).

The curriculum progressively introduces data sources in a fixed order (short
nested ncRNA -> mixed Rfam -> pri-miRNA -> PDB -> viral -> human mRNA -> lncRNA),
keeping a ``replay_ratio`` fraction of samples from earlier stages to avoid
catastrophic forgetting.  Sampling is balanced across the configured
``balance_keys`` (source, family, clan, length bin, pair-distance bin,
structure-complexity bin) via inverse-frequency weighting.

Records are duck-typed objects exposing the same fields as
:class:`reactflow.data_registry.DataRecord` (``source``, ``family``, ``clan``,
``sequence``, ``pairs``), so this module has no hard dependency on the C1-1
registry and can be unit-tested with lightweight stubs.

Complexity
----------
- ``get_stage``: ``O(S)`` where ``S`` is the number of stages.
- ``_compute_weights``: ``O(N * K)`` for ``N`` records and ``K`` balance keys.
- ``get_sampler``: ``O(N * K + N)``.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Sampler, WeightedRandomSampler


# ---------------------------------------------------------------------------
# CurriculumStage
# ---------------------------------------------------------------------------


class CurriculumStage(Enum):
    """Curriculum stages, in the order they are introduced (spec lines 513-520).

    The order matters: earlier stages are seen first, with later stages added
    progressively while replaying earlier ones.
    """

    SHORT_NESTED_NCRNA = "short_nested_ncRNA"
    MIXED_RFAM = "mixed_rfam"
    PRI_MIRNA = "pri_miRNA"
    PDB = "pdb"
    VIRAL = "viral"
    HUMAN_MRNA = "human_mRNA"
    LNCRNA = "lncRNA"


# Default stage order (spec lines 513-520).
DEFAULT_STAGES: List[CurriculumStage] = [
    CurriculumStage.SHORT_NESTED_NCRNA,
    CurriculumStage.MIXED_RFAM,
    CurriculumStage.PRI_MIRNA,
    CurriculumStage.PDB,
    CurriculumStage.VIRAL,
    CurriculumStage.HUMAN_MRNA,
    CurriculumStage.LNCRNA,
]

# Default epochs per stage.
DEFAULT_STAGE_EPOCHS: Dict[CurriculumStage, int] = {
    CurriculumStage.SHORT_NESTED_NCRNA: 5,
    CurriculumStage.MIXED_RFAM: 8,
    CurriculumStage.PRI_MIRNA: 4,
    CurriculumStage.PDB: 6,
    CurriculumStage.VIRAL: 4,
    CurriculumStage.HUMAN_MRNA: 4,
    CurriculumStage.LNCRNA: 4,
}

# Mapping from stage to the canonical record ``source`` values it consumes.
# Short nested ncRNA reuses the ncRNA structured-RNA sources filtered to short
# lengths; pri-miRNA is drawn from Ribonanza (miRNA-enriched) plus Rfam miRNA
# families.  These mappings align with the unified ``DataRecord.source`` values
# produced by the C1-1 registry (see ``data_registry._SOURCE_ALIASES``).
_STAGE_SOURCES: Dict[CurriculumStage, Set[str]] = {
    CurriculumStage.SHORT_NESTED_NCRNA: {
        "efold_train",
        "ArchiveII",
        "bpRNA",
        "RNAStrAlign",
    },
    CurriculumStage.MIXED_RFAM: {"Rfam"},
    CurriculumStage.PRI_MIRNA: {"Ribonanza", "Ribonanza2"},
    CurriculumStage.PDB: {"PDB"},
    CurriculumStage.VIRAL: {"viral"},
    CurriculumStage.HUMAN_MRNA: {"human_mRNA"},
    CurriculumStage.LNCRNA: {"lncRNA"},
}

# Default balance keys (spec lines 522-527).
DEFAULT_BALANCE_KEYS: List[str] = [
    "source",
    "family",
    "clan",
    "length_bin",
    "pair_distance_bin",
    "structure_complexity",
]


# ---------------------------------------------------------------------------
# CurriculumConfig
# ---------------------------------------------------------------------------


@dataclass
class CurriculumConfig:
    """Configuration for :class:`CurriculumSampler`.

    Attributes:
        stages: ordered list of stages (order matters for the curriculum).
        stage_epochs: number of epochs to spend on each stage.
        replay_ratio: fraction of the current stage's records sampled from
            previous stages (default 0.2).
        balance_keys: record attributes used to balance sampling weights.
        seed: RNG seed for reproducible replay sampling.
    """

    stages: List[CurriculumStage] = field(default_factory=lambda: list(DEFAULT_STAGES))
    stage_epochs: Dict[CurriculumStage, int] = field(
        default_factory=lambda: dict(DEFAULT_STAGE_EPOCHS)
    )
    replay_ratio: float = 0.2
    balance_keys: List[str] = field(default_factory=lambda: list(DEFAULT_BALANCE_KEYS))
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("stages must be a non-empty list")
        if not 0.0 <= self.replay_ratio <= 1.0:
            raise ValueError(
                f"replay_ratio must be in [0, 1], got {self.replay_ratio}"
            )
        # Ensure every stage has an epoch budget.
        for stage in self.stages:
            if stage not in self.stage_epochs:
                self.stage_epochs[stage] = DEFAULT_STAGE_EPOCHS.get(stage, 5)
                if self.stage_epochs[stage] <= 0:
                    raise ValueError(
                        f"stage_epochs for {stage.name} must be > 0"
                    )
        for stage, epochs in self.stage_epochs.items():
            if epochs <= 0:
                raise ValueError(
                    f"stage_epochs for {stage.name} must be > 0, got {epochs}"
                )


# ---------------------------------------------------------------------------
# CurriculumSampler
# ---------------------------------------------------------------------------


class CurriculumSampler:
    """Progressive curriculum sampler with replay and diversity balancing.

    Args:
        records: list of data records (duck-typed; needs ``source``,
            ``family``, ``clan``, ``sequence``, ``pairs``).
        config: :class:`CurriculumConfig`.

    Complexity: construction is ``O(N)``; ``get_sampler`` is ``O(N * K)``.
    """

    # Length-bin thresholds (spec line 525).
    LENGTH_SHORT_MAX = 50  # < 50
    LENGTH_MEDIUM_MAX = 200  # 50-200
    # Pair-distance-bin thresholds (spec line 526).
    PAIR_DIST_SHORT_MAX = 11  # 1-11
    PAIR_DIST_MEDIUM_MAX = 23  # 12-23
    # Structure-complexity thresholds (spec line 527).
    COMPLEXITY_MODERATE_MAX = 0.1  # < 0.1 simple
    COMPLEXITY_COMPLEX_MAX = 0.3  # 0.1-0.3 moderate

    def __init__(self, records: Sequence[Any], config: CurriculumConfig) -> None:
        self.config = config
        self.records: List[Any] = list(records)
        # Pre-index records by stage for fast filtering.
        self._stage_index: Dict[CurriculumStage, List[Any]] = {
            stage: [] for stage in CurriculumStage
        }
        for rec in self.records:
            for stage in self.config.stages:
                if self._record_matches_stage(rec, stage):
                    self._stage_index[stage].append(rec)
                    break  # a record belongs to at most one stage

    # ------------------------------------------------------------------
    # Stage selection
    # ------------------------------------------------------------------

    def get_stage(self, epoch: int) -> CurriculumStage:
        """Return the stage active at the given (global) epoch.

        Stages are concatenated in ``config.stages`` order, each lasting
        ``config.stage_epochs[stage]`` epochs.  Epochs beyond the final stage
        map to the last stage.

        Complexity: ``O(S)`` where ``S = len(config.stages)``.
        """
        cumulative = 0
        for stage in self.config.stages:
            cumulative += self.config.stage_epochs.get(stage, 0)
            if epoch < cumulative:
                return stage
        return self.config.stages[-1]

    def get_sampler(self, epoch: int) -> Sampler[int]:
        """Return a :class:`WeightedRandomSampler` for the given epoch.

        The sampler draws from the current stage's records plus replay records
        from earlier stages, with inverse-frequency weights across
        ``config.balance_keys``.  Indices refer to the combined record list
        returned by :meth:`get_epoch_records`.

        Complexity: ``O(N * K + N)`` where ``N`` is the combined record count.
        """
        stage = self.get_stage(epoch)
        stage_records = self._get_stage_records(stage)
        replay_records = self._get_replay_records(stage, self.config.replay_ratio)
        combined = stage_records + replay_records
        weights = self._compute_weights(combined, self.config.balance_keys)
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + epoch)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(combined),
            replacement=True,
            generator=generator,
        )

    def get_epoch_records(self, epoch: int) -> List[Any]:
        """Return the combined (stage + replay) records for an epoch.

        Convenience accessor so callers can build a ``Dataset`` whose indices
        align with the sampler returned by :meth:`get_sampler`.

        Complexity: ``O(N)``.
        """
        stage = self.get_stage(epoch)
        stage_records = self._get_stage_records(stage)
        replay_records = self._get_replay_records(stage, self.config.replay_ratio)
        return stage_records + replay_records

    # ------------------------------------------------------------------
    # Stage / replay record selection
    # ------------------------------------------------------------------

    def _get_stage_records(self, stage: CurriculumStage) -> List[Any]:
        """Return all records belonging to ``stage``.

        Complexity: ``O(1)`` (pre-indexed).
        """
        return list(self._stage_index.get(stage, []))

    def _get_replay_records(
        self, stage: CurriculumStage, ratio: float
    ) -> List[Any]:
        """Return replay samples from stages before ``stage``.

        The number of replay samples is ``ratio * len(current_stage_records)``,
        drawn uniformly (without replacement) from the pool of all earlier
        stages' records.  Returns ``[]`` for the first stage or ``ratio == 0``.

        Complexity: ``O(R)`` where ``R`` is the size of the replay pool.
        """
        if ratio <= 0:
            return []
        idx = self.config.stages.index(stage)
        if idx == 0:
            return []
        replay_pool: List[Any] = []
        for prev_stage in self.config.stages[:idx]:
            replay_pool.extend(self._get_stage_records(prev_stage))
        if not replay_pool:
            return []
        current_size = len(self._get_stage_records(stage))
        n_replay = int(ratio * current_size)
        n_replay = max(0, min(n_replay, len(replay_pool)))
        if n_replay == 0:
            return []
        rng = random.Random(self.config.seed)
        return rng.sample(replay_pool, n_replay)

    def _record_matches_stage(self, record: Any, stage: CurriculumStage) -> bool:
        """Return ``True`` if ``record`` belongs to ``stage``.

        Uses :data:`_STAGE_SOURCES` to match on the record's ``source`` field.
        For :attr:`CurriculumStage.SHORT_NESTED_NCRNA`, an additional length
        filter (``L < 50``) is applied.

        Complexity: ``O(L)`` for the length check (computes ``len(sequence)``).
        """
        sources = _STAGE_SOURCES.get(stage, set())
        record_source = getattr(record, "source", None)
        if record_source not in sources:
            return False
        if stage is CurriculumStage.SHORT_NESTED_NCRNA:
            sequence = getattr(record, "sequence", "")
            if len(sequence) >= self.LENGTH_SHORT_MAX:
                return False
        return True

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _compute_weights(
        self, records: Sequence[Any], balance_keys: Sequence[str]
    ) -> torch.Tensor:
        """Compute inverse-frequency sampling weights.

        For each balance key, each record's weight contribution is
        ``1 / count(group)`` where ``group`` is the record's value for that
        key.  Contributions are averaged across keys, then normalized to a
        probability distribution summing to 1.

        Args:
            records: the records to weight.
            balance_keys: keys to balance across.

        Returns:
            Float tensor of shape ``(N,)`` summing to 1.0 (empty if no records).

        Complexity: ``O(N * K)`` for ``N`` records and ``K`` keys.
        """
        n = len(records)
        if n == 0:
            return torch.zeros(0, dtype=torch.float32)
        keys = list(balance_keys) if balance_keys else ["source"]
        # Accumulate per-key inverse-frequency weights.
        accum = torch.zeros(n, dtype=torch.float32)
        for key in keys:
            groups = [self._get_balance_value(r, key) for r in records]
            counts = Counter(groups)
            for i, g in enumerate(groups):
                accum[i] += 1.0 / float(counts[g])
        accum /= float(len(keys))
        total = accum.sum()
        if total <= 0:
            return torch.full((n,), 1.0 / n, dtype=torch.float32)
        return accum / total

    def _get_balance_value(self, record: Any, key: str) -> str:
        """Return the balance-group label for ``record`` under ``key``.

        Complexity: ``O(L)`` for length/pair-derived keys.
        """
        if key == "source":
            return str(getattr(record, "source", "unknown"))
        if key == "family":
            value = getattr(record, "family", None)
            return str(value) if value else "none"
        if key == "clan":
            value = getattr(record, "clan", None)
            return str(value) if value else "none"
        if key == "length_bin":
            sequence = getattr(record, "sequence", "")
            return self._bin_length(len(sequence))
        if key == "pair_distance_bin":
            pairs = getattr(record, "pairs", ())
            dists = [j - i for (i, j) in pairs if j > i]
            mean_dist = sum(dists) / len(dists) if dists else 0
            return self._bin_pair_distance(mean_dist)
        if key == "structure_complexity":
            pairs = getattr(record, "pairs", ())
            sequence = getattr(record, "sequence", "")
            return self._bin_structure_complexity(len(pairs), len(sequence))
        raise ValueError(f"unknown balance key: {key}")

    # ------------------------------------------------------------------
    # Binning helpers
    # ------------------------------------------------------------------

    @classmethod
    def _bin_length(cls, L: int) -> str:
        """Bin a sequence length.

        Bins: ``short`` (< 50), ``medium`` (50-200), ``long`` (> 200).

        Complexity: ``O(1)``.
        """
        if L < cls.LENGTH_SHORT_MAX:
            return "short"
        if L <= cls.LENGTH_MEDIUM_MAX:
            return "medium"
        return "long"

    @classmethod
    def _bin_pair_distance(cls, dist: float) -> str:
        """Bin a mean pair distance.

        Bins: ``short`` (1-11), ``medium`` (12-23), ``long`` (24+).  A distance
        of 0 (no pairs) is binned as ``short``.

        Complexity: ``O(1)``.
        """
        if dist <= cls.PAIR_DIST_SHORT_MAX:
            return "short"
        if dist <= cls.PAIR_DIST_MEDIUM_MAX:
            return "medium"
        return "long"

    @classmethod
    def _bin_structure_complexity(cls, pair_count: int, L: int) -> str:
        """Bin structure complexity as ``pair_count / L``.

        Bins: ``simple`` (< 0.1), ``moderate`` (0.1-0.3), ``complex`` (> 0.3).
        For ``L == 0``, returns ``simple``.

        Complexity: ``O(1)``.
        """
        if L <= 0:
            return "simple"
        density = pair_count / float(L)
        if density < cls.COMPLEXITY_MODERATE_MAX:
            return "simple"
        if density <= cls.COMPLEXITY_COMPLEX_MAX:
            return "moderate"
        return "complex"
