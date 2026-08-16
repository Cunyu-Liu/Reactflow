"""Unit tests for the C1-3 data-diversity curriculum sampler.

Covers:
- CurriculumStage enum members and ordering
- CurriculumConfig defaults and validation
- get_stage returns the correct stage for a given epoch (and past-last fallback)
- _bin_length / _bin_pair_distance / _bin_structure_complexity boundaries
- _compute_weights produces a valid probability distribution and balances groups
- replay records drawn from previous stages (and none for the first stage)
- get_sampler returns a WeightedRandomSampler aligned with epoch records
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from reactflow.curriculum import (
    DEFAULT_BALANCE_KEYS,
    DEFAULT_STAGES,
    DEFAULT_STAGE_EPOCHS,
    CurriculumConfig,
    CurriculumSampler,
    CurriculumStage,
)


# ---------------------------------------------------------------------------
# Record stub (duck-typed, mirrors DataRecord fields used by the sampler)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Record:
    record_id: str
    sequence: str
    source: str
    family: str = "none"
    clan: str = "none"
    pairs: Tuple[Tuple[int, int], ...] = ()


def _make_record(
    rid: str,
    source: str,
    length: int = 100,
    family: str = "RF00001",
    clan: str = "CL00001",
    pairs: Sequence[Tuple[int, int]] = (),
) -> _Record:
    seq = "A" * length
    return _Record(
        record_id=rid,
        sequence=seq,
        source=source,
        family=family,
        clan=clan,
        pairs=tuple(pairs),
    )


# A fixture record set spanning multiple stages.
def _multi_stage_records() -> list:
    return [
        # SHORT_NESTED_NCRNA: short, from ArchiveII.
        _make_record("a1", "ArchiveII", length=30, pairs=[(0, 10)]),
        _make_record("a2", "ArchiveII", length=40, pairs=[(0, 5), (1, 6)]),
        # MIXED_RFAM
        _make_record("r1", "Rfam", length=120, pairs=[(0, 30)]),
        _make_record("r2", "Rfam", length=150, pairs=[(0, 40), (5, 45)]),
        # PDB
        _make_record("p1", "PDB", length=220, pairs=[(0, 50), (10, 60), (20, 70)]),
        # LNCRNA
        _make_record("l1", "lncRNA", length=300, pairs=[(0, 100)]),
    ]


# ---------------------------------------------------------------------------
# CurriculumStage enum
# ---------------------------------------------------------------------------


class TestCurriculumStage:
    def test_enum_members(self):
        assert len(CurriculumStage) == 7
        names = {s.name for s in CurriculumStage}
        assert names == {
            "SHORT_NESTED_NCRNA",
            "MIXED_RFAM",
            "PRI_MIRNA",
            "PDB",
            "VIRAL",
            "HUMAN_MRNA",
            "LNCRNA",
        }

    def test_default_stage_order_matches_spec(self):
        # Spec lines 513-520: short nested ncRNA -> mixed Rfam -> pri-miRNA ->
        # PDB -> viral -> human mRNA -> lncRNA.
        assert DEFAULT_STAGES[0] is CurriculumStage.SHORT_NESTED_NCRNA
        assert DEFAULT_STAGES[1] is CurriculumStage.MIXED_RFAM
        assert DEFAULT_STAGES[2] is CurriculumStage.PRI_MIRNA
        assert DEFAULT_STAGES[3] is CurriculumStage.PDB
        assert DEFAULT_STAGES[4] is CurriculumStage.VIRAL
        assert DEFAULT_STAGES[5] is CurriculumStage.HUMAN_MRNA
        assert DEFAULT_STAGES[6] is CurriculumStage.LNCRNA


# ---------------------------------------------------------------------------
# CurriculumConfig
# ---------------------------------------------------------------------------


class TestCurriculumConfig:
    def test_defaults(self):
        cfg = CurriculumConfig()
        assert cfg.stages == DEFAULT_STAGES
        assert cfg.replay_ratio == 0.2
        assert cfg.balance_keys == DEFAULT_BALANCE_KEYS
        assert cfg.seed == 42
        # Every default stage has an epoch budget.
        for stage in DEFAULT_STAGES:
            assert cfg.stage_epochs[stage] == DEFAULT_STAGE_EPOCHS[stage]

    def test_custom_config(self):
        cfg = CurriculumConfig(
            stages=[CurriculumStage.PDB, CurriculumStage.VIRAL],
            stage_epochs={CurriculumStage.PDB: 3, CurriculumStage.VIRAL: 2},
            replay_ratio=0.5,
            seed=7,
        )
        assert cfg.stages == [CurriculumStage.PDB, CurriculumStage.VIRAL]
        assert cfg.replay_ratio == 0.5
        assert cfg.seed == 7

    def test_validation_rejects_empty_stages(self):
        with pytest.raises(ValueError, match="non-empty"):
            CurriculumConfig(stages=[])

    def test_validation_rejects_bad_replay_ratio(self):
        with pytest.raises(ValueError, match="replay_ratio"):
            CurriculumConfig(replay_ratio=1.5)

    def test_validation_rejects_nonpositive_stage_epochs(self):
        with pytest.raises(ValueError, match="must be > 0"):
            CurriculumConfig(
                stage_epochs={CurriculumStage.PDB: 0},
            )


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------


class TestBins:
    def test_bin_length_boundaries(self):
        assert CurriculumSampler._bin_length(0) == "short"
        assert CurriculumSampler._bin_length(49) == "short"
        assert CurriculumSampler._bin_length(50) == "medium"
        assert CurriculumSampler._bin_length(200) == "medium"
        assert CurriculumSampler._bin_length(201) == "long"

    def test_bin_pair_distance_boundaries(self):
        assert CurriculumSampler._bin_pair_distance(0) == "short"
        assert CurriculumSampler._bin_pair_distance(1) == "short"
        assert CurriculumSampler._bin_pair_distance(11) == "short"
        assert CurriculumSampler._bin_pair_distance(12) == "medium"
        assert CurriculumSampler._bin_pair_distance(23) == "medium"
        assert CurriculumSampler._bin_pair_distance(24) == "long"
        assert CurriculumSampler._bin_pair_distance(100) == "long"

    def test_bin_structure_complexity_boundaries(self):
        # density = pair_count / L
        assert CurriculumSampler._bin_structure_complexity(0, 100) == "simple"
        assert CurriculumSampler._bin_structure_complexity(9, 100) == "simple"  # 0.09
        assert CurriculumSampler._bin_structure_complexity(10, 100) == "moderate"  # 0.10
        assert CurriculumSampler._bin_structure_complexity(30, 100) == "moderate"  # 0.30
        assert CurriculumSampler._bin_structure_complexity(31, 100) == "complex"  # 0.31
        # L == 0 edge case.
        assert CurriculumSampler._bin_structure_complexity(5, 0) == "simple"


# ---------------------------------------------------------------------------
# get_stage
# ---------------------------------------------------------------------------


class TestGetStage:
    def test_returns_correct_stage_for_epoch(self):
        cfg = CurriculumConfig(
            stages=[CurriculumStage.PDB, CurriculumStage.VIRAL, CurriculumStage.LNCRNA],
            stage_epochs={
                CurriculumStage.PDB: 3,
                CurriculumStage.VIRAL: 2,
                CurriculumStage.LNCRNA: 4,
            },
        )
        sampler = CurriculumSampler([], cfg)
        assert sampler.get_stage(0) is CurriculumStage.PDB
        assert sampler.get_stage(2) is CurriculumStage.PDB
        assert sampler.get_stage(3) is CurriculumStage.VIRAL
        assert sampler.get_stage(4) is CurriculumStage.VIRAL
        assert sampler.get_stage(5) is CurriculumStage.LNCRNA
        assert sampler.get_stage(8) is CurriculumStage.LNCRNA

    def test_past_last_stage_returns_last(self):
        cfg = CurriculumConfig(
            stages=[CurriculumStage.PDB, CurriculumStage.VIRAL],
            stage_epochs={CurriculumStage.PDB: 1, CurriculumStage.VIRAL: 1},
        )
        sampler = CurriculumSampler([], cfg)
        # Epoch 100 is far past the 2-stage budget; should clamp to the last.
        assert sampler.get_stage(100) is CurriculumStage.VIRAL


# ---------------------------------------------------------------------------
# _compute_weights
# ---------------------------------------------------------------------------


class TestComputeWeights:
    def test_valid_probability_distribution(self):
        records = [
            _make_record("a", "PDB", length=100, family="RF1", clan="CL1"),
            _make_record("b", "PDB", length=100, family="RF2", clan="CL1"),
            _make_record("c", "viral", length=100, family="RF1", clan="CL2"),
            _make_record("d", "viral", length=100, family="RF2", clan="CL2"),
        ]
        sampler = CurriculumSampler(records, CurriculumConfig())
        weights = sampler._compute_weights(records, DEFAULT_BALANCE_KEYS)
        assert weights.shape == (4,)
        assert torch.all(weights > 0)
        assert weights.sum().item() == pytest.approx(1.0)

    def test_balances_across_groups(self):
        """A rare group should get a higher per-sample weight than a common one."""
        records = [
            _make_record(f"c{i}", "PDB", length=100, family="common", clan="CL1")
            for i in range(8)
        ] + [
            _make_record(f"r{i}", "PDB", length=100, family="rare", clan="CL1")
            for i in range(2)
        ]
        sampler = CurriculumSampler(records, CurriculumConfig())
        weights = sampler._compute_weights(records, ["family"])
        # rare group (2 members) each get weight 1/2; common (8) get 1/8.
        # After normalization the per-sample rare weight > common weight.
        rare_weights = weights[8:]
        common_weights = weights[:8]
        assert rare_weights.mean().item() > common_weights.mean().item()

    def test_empty_records_returns_empty_tensor(self):
        sampler = CurriculumSampler([], CurriculumConfig())
        weights = sampler._compute_weights([], ["source"])
        assert weights.numel() == 0

    def test_unknown_balance_key_raises(self):
        sampler = CurriculumSampler([_make_record("a", "PDB")], CurriculumConfig())
        with pytest.raises(ValueError, match="unknown balance key"):
            sampler._compute_weights([_make_record("a", "PDB")], ["bogus_key"])


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class TestReplay:
    def test_replay_from_previous_stages(self):
        records = _multi_stage_records()
        cfg = CurriculumConfig(
            stages=[
                CurriculumStage.SHORT_NESTED_NCRNA,
                CurriculumStage.MIXED_RFAM,
                CurriculumStage.PDB,
            ],
            stage_epochs={
                CurriculumStage.SHORT_NESTED_NCRNA: 2,
                CurriculumStage.MIXED_RFAM: 2,
                CurriculumStage.PDB: 2,
            },
            replay_ratio=0.5,
        )
        sampler = CurriculumSampler(records, cfg)
        # PDB stage: previous stages are SHORT_NESTED_NCRNA + MIXED_RFAM.
        replay = sampler._get_replay_records(CurriculumStage.PDB, 0.5)
        # PDB has 1 record; 0.5 * 1 = 0 -> at least 0. Use a larger ratio to
        # guarantee a non-zero replay sample.
        replay = sampler._get_replay_records(CurriculumStage.PDB, 1.0)
        replay_ids = {r.record_id for r in replay}
        # Replay must come only from earlier stages, never the current PDB stage.
        assert "p1" not in replay_ids
        assert replay_ids.issubset({"a1", "a2", "r1", "r2"})
        assert len(replay) == 1  # 1.0 * len(pdb records=1) = 1

    def test_no_replay_for_first_stage(self):
        records = _multi_stage_records()
        cfg = CurriculumConfig(stages=[CurriculumStage.PDB, CurriculumStage.VIRAL])
        sampler = CurriculumSampler(records, cfg)
        replay = sampler._get_replay_records(CurriculumStage.PDB, 0.5)
        assert replay == []

    def test_replay_zero_ratio(self):
        records = _multi_stage_records()
        cfg = CurriculumConfig()
        sampler = CurriculumSampler(records, cfg)
        assert sampler._get_replay_records(CurriculumStage.PDB, 0.0) == []


# ---------------------------------------------------------------------------
# Stage record filtering
# ---------------------------------------------------------------------------


class TestStageRecords:
    def test_stage_records_filtered_by_source(self):
        records = _multi_stage_records()
        sampler = CurriculumSampler(records, CurriculumConfig())
        rfam = sampler._get_stage_records(CurriculumStage.MIXED_RFAM)
        assert all(r.source == "Rfam" for r in rfam)
        assert {r.record_id for r in rfam} == {"r1", "r2"}

    def test_short_nested_ncrna_filters_by_length(self):
        records = [
            _make_record("short", "ArchiveII", length=30),
            _make_record("long", "ArchiveII", length=300),
        ]
        sampler = CurriculumSampler(records, CurriculumConfig())
        short_records = sampler._get_stage_records(CurriculumStage.SHORT_NESTED_NCRNA)
        ids = {r.record_id for r in short_records}
        assert "short" in ids
        assert "long" not in ids


# ---------------------------------------------------------------------------
# get_sampler
# ---------------------------------------------------------------------------


class TestGetSampler:
    def test_returns_weighted_sampler(self):
        records = _multi_stage_records()
        cfg = CurriculumConfig()
        sampler = CurriculumSampler(records, cfg)
        s = sampler.get_sampler(epoch=0)
        assert isinstance(s, WeightedRandomSampler)
        # num_samples matches the combined stage+replay record count for epoch 0.
        epoch_records = sampler.get_epoch_records(epoch=0)
        assert s.num_samples == len(epoch_records)

    def test_sampler_weights_sum_to_num_samples(self):
        """With replacement, the WeightedRandomSampler weights should be a
        valid distribution (sum to 1)."""
        records = _multi_stage_records()
        sampler = CurriculumSampler(records, CurriculumConfig())
        s = sampler.get_sampler(epoch=0)
        assert s.weights.sum().item() == pytest.approx(1.0)

    def test_sampler_is_deterministic_with_seed(self):
        records = _multi_stage_records()
        sampler = CurriculumSampler(records, CurriculumConfig(seed=123))
        s1 = sampler.get_sampler(epoch=0)
        s2 = sampler.get_sampler(epoch=0)
        # Same seed -> identical first draw sequence.
        seq1 = [int(s1.__iter__().__next__()) for _ in range(min(5, s1.num_samples))]
        # Rebuild to reset the generator.
        s1b = sampler.get_sampler(epoch=0)
        seq2 = [int(s1b.__iter__().__next__()) for _ in range(min(5, s1b.num_samples))]
        assert seq1 == seq2
