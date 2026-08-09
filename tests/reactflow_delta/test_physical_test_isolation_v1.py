#!/usr/bin/env python3
"""Unit tests for verify_physical_test_isolation_v1.py (Task 1D)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "scripts/reactflow_delta"))

import pytest

import verify_physical_test_isolation_v1 as iso


def test_dev_builder_does_not_reference_test_store():
    dev_source = "def build_train(): return load_dev_cache('/data/dev')"
    assert iso.dev_builder_references_test_store(dev_source, "/data/test_outcomes.jsonl") is False


def test_dev_builder_referencing_test_store_detected():
    dev_source = "def build_train(): return open('/data/test_outcomes.jsonl')"
    assert iso.dev_builder_references_test_store(dev_source, "/data/test_outcomes.jsonl") is True


def test_test_store_distinct_from_dev_cache():
    assert iso.is_distinct_path("/data/test_outcomes.jsonl", "/data/dev_cache.pt") is True
    assert iso.is_distinct_path("/data/same.jsonl", "/data/same.jsonl") is False


def test_mixed_cache_load_then_filter_must_fail():
    # loading the full test store then filtering to dev rows is forbidden
    with pytest.raises(ValueError, match="load-then-filter"):
        iso.validate_fixture_purity(loads_full_store=True, filter_after_load=True)
    # loading dev-only is fine
    assert iso.validate_fixture_purity(loads_full_store=False, filter_after_load=True) is True
    assert iso.validate_fixture_purity(loads_full_store=True, filter_after_load=False) is True


def test_append_only_ledger_preserves_events(tmp_path: Path):
    led = tmp_path / "ledger.jsonl"
    iso.append_ledger(led, {"event": "open", "path": "/a"}, seq=1)
    iso.append_ledger(led, {"event": "hash", "path": "/b"}, seq=2)
    records = iso.read_ledger(led)
    assert [r["seq"] for r in records] == [1, 2]
    assert [r["event"] for r in records] == ["open", "hash"]
    # appending is not a rewrite: order preserved, both present
    assert len(records) == 2


def test_duplicate_seq_detected_as_rewrite():
    assert iso.is_append_only_ok([], "new_event") is True
    assert iso.is_append_only_ok([json.dumps({"seq": "dup"})], "dup") is False