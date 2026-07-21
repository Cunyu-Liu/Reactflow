"""Unit tests for ``scripts/build_frozen_benchmarks.py`` novel_clan logic (C1-1 Task 4)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

# Import from the script module (scripts/ is not a package, so use importlib).
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_frozen_benchmarks.py"
_spec = importlib.util.spec_from_file_location("build_frozen_benchmarks", _SCRIPT_PATH)
bfb_module = importlib.util.module_from_spec(_spec)
sys.modules["build_frozen_benchmarks"] = bfb_module
_spec.loader.exec_module(bfb_module)

from reactflow.data_registry import DataRecord


def _make_record(
    rid: str,
    family: str | None = None,
    clan: str | None = None,
    source: str = "efold_train",
) -> DataRecord:
    """Helper to make a minimal DataRecord for testing."""
    return DataRecord(
        record_id=rid,
        sequence="ACGU",
        checksum="x" * 64,
        source=source,
        source_version="test",
        source_id=rid,
        family=family,
        clan=clan,
    )


class TestComputeNovelClanSplit:
    def test_empty_assignment(self):
        assignment = {}
        records = {}
        new_assignment, stats = bfb_module.compute_novel_clan_split(assignment, records)
        assert new_assignment == {}
        assert stats["novel_clan_count"] == 0
        assert stats["novel_family_remaining"] == 0
        assert stats["novel_family_no_clan"] == 0
        assert stats["novel_family_clan_in_train"] == 0
        assert stats["train_clan_count"] == 0

    def test_novel_family_record_without_clan_stays(self):
        records = {
            "r1": _make_record("r1", clan=None),
        }
        assignment = {"r1": "novel_family"}
        new_assignment, stats = bfb_module.compute_novel_clan_split(assignment, records)
        assert new_assignment["r1"] == "novel_family"
        assert stats["novel_clan_count"] == 0
        assert stats["novel_family_no_clan"] == 1

    def test_novel_family_record_with_clan_not_in_train_moved(self):
        records = {
            "r1": _make_record("r1", clan="CL00001"),
            "r2": _make_record("r2", clan="CL00002"),  # train clan
        }
        assignment = {"r1": "novel_family", "r2": "train"}
        new_assignment, stats = bfb_module.compute_novel_clan_split(assignment, records)
        assert new_assignment["r1"] == "novel_clan"
        assert new_assignment["r2"] == "train"
        assert stats["novel_clan_count"] == 1
        assert stats["train_clan_count"] == 1

    def test_novel_family_record_with_clan_in_train_stays(self):
        """Defense-in-depth: if a novel_family record's clan is in train, it stays."""
        records = {
            "r1": _make_record("r1", clan="CL00001"),  # clan in train
            "r2": _make_record("r2", clan="CL00001"),  # train
        }
        assignment = {"r1": "novel_family", "r2": "train"}
        new_assignment, stats = bfb_module.compute_novel_clan_split(assignment, records)
        assert new_assignment["r1"] == "novel_family"  # stays
        assert stats["novel_clan_count"] == 0
        assert stats["novel_family_clan_in_train"] == 1

    def test_mixed_records(self):
        records = {
            "r1": _make_record("r1", clan="CL00001"),  # train clan
            "r2": _make_record("r2", clan="CL00002"),  # novel clan
            "r3": _make_record("r3", clan=None),        # no clan
            "r4": _make_record("r4", clan="CL00003"),  # novel clan
            "r5": _make_record("r5", clan="CL00001"),  # train
        }
        assignment = {
            "r1": "train",
            "r2": "novel_family",
            "r3": "novel_family",
            "r4": "novel_family",
            "r5": "train",
        }
        new_assignment, stats = bfb_module.compute_novel_clan_split(assignment, records)
        assert new_assignment["r2"] == "novel_clan"
        assert new_assignment["r3"] == "novel_family"  # no clan
        assert new_assignment["r4"] == "novel_clan"
        assert stats["novel_clan_count"] == 2
        assert stats["novel_family_remaining"] == 1
        assert stats["novel_family_no_clan"] == 1


class TestValidateNovelClanDisjoint:
    def test_no_violations(self):
        records = {
            "r1": _make_record("r1", clan="CL00001"),
            "r2": _make_record("r2", clan="CL00002"),
        }
        assignment = {"r1": "train", "r2": "novel_clan"}
        violations = bfb_module.validate_novel_clan_disjoint(assignment, records)
        assert violations == []

    def test_violation(self):
        records = {
            "r1": _make_record("r1", clan="CL00001"),
            "r2": _make_record("r2", clan="CL00001"),  # same clan
        }
        assignment = {"r1": "train", "r2": "novel_clan"}
        violations = bfb_module.validate_novel_clan_disjoint(assignment, records)
        assert len(violations) == 1
        assert "r2" in violations[0]

    def test_novel_clan_record_without_clan_no_violation(self):
        records = {
            "r1": _make_record("r1", clan="CL00001"),
            "r2": _make_record("r2", clan=None),
        }
        assignment = {"r1": "train", "r2": "novel_clan"}
        violations = bfb_module.validate_novel_clan_disjoint(assignment, records)
        assert violations == []


class TestValidateNovelFamilyDisjoint:
    def test_no_violations(self):
        records = {
            "r1": _make_record("r1", family="RF00001"),
            "r2": _make_record("r2", family="RF00002"),
        }
        assignment = {"r1": "train", "r2": "novel_family"}
        violations = bfb_module.validate_novel_family_disjoint(assignment, records)
        assert violations == []

    def test_violation(self):
        records = {
            "r1": _make_record("r1", family="RF00001"),
            "r2": _make_record("r2", family="RF00001"),
        }
        assignment = {"r1": "train", "r2": "novel_family"}
        violations = bfb_module.validate_novel_family_disjoint(assignment, records)
        assert len(violations) == 1
