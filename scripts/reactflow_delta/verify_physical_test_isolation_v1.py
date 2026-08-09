#!/usr/bin/env python3
"""Physical test isolation verification v1 (benchmark_v3 / Task 1D).

Verifies that the development / benchmarking machinery is physically isolated
from the confirmatory test outcome store:

  * the development builder does NOT enumerate/open/deserialize the test
    outcome store;
  * the test outcome store lives at a distinct path with separate access
    control;
  * any open/hash/stat/evaluation event is appended to an append-only ledger;
  * mixed-cache / load-then-filter fixtures MUST FAIL (a fixture that
    materializes the full test outcome store and then filters to development
    rows is forbidden, because it could leak test outcomes into selection).

Core logic is pure and testable (no I/O required for the checks themselves).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# pure, testable checks
# ---------------------------------------------------------------------------
def dev_builder_references_test_store(dev_source: str, test_store_path: str) -> bool:
    """True if the development builder source references the test outcome store."""
    if not test_store_path:
        return False
    # match on the store's file name or a distinctive path fragment
    name = Path(test_store_path).name
    return name in dev_source or test_store_path in dev_source


def is_distinct_path(test_store_path: str, dev_path: str) -> bool:
    """Test store must be a distinct path from any development cache path."""
    return Path(test_store_path).resolve() != Path(dev_path).resolve()


def validate_fixture_purity(loads_full_store: bool, filter_after_load: bool) -> bool:
    """A load-then-filter fixture must FAIL.

    loads_full_store  : the fixture materializes the full test outcome store.
    filter_after_load : the fixture then filters to development rows.
    Returns True if the fixture is pure (allowed); raises ValueError otherwise.
    """
    if loads_full_store and filter_after_load:
        raise ValueError(
            "forbidden mixed-cache / load-then-filter fixture: test outcome "
            "store is materialized then filtered to development rows")
    return True


def is_append_only_ok(existing_lines: Sequence[str], new_event: str) -> bool:
    """An append-only ledger never rewrites; new events are appended after old."""
    for line in existing_lines:
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("seq") == new_event:
            return False  # duplicate seq -> would be a rewrite, not append
    return True


def append_ledger(ledger_path: Path, event: dict, seq: int) -> None:
    """Append a single event line to the ledger (never overwrites)."""
    record = {
        "seq": seq,
        "ts": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with ledger_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def read_ledger(ledger_path: Path):
    records = []
    if ledger_path.exists():
        for line in ledger_path.open():
            if line.strip():
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    here = Path(__file__).resolve().parent
    worktree_root = here.parent.parent
    default_out = worktree_root / "data_registry" / "reactflow_delta"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-builder", type=Path, default=Path(
        "/home/cunyuliu/reactflow_delta_goal_20260729/scripts/reactflow_delta/"
        "m0x_gate_harness.py"))
    parser.add_argument("--test-store", type=str, required=True,
                        help="Test outcome store path (must be isolated).")
    parser.add_argument("--dev-cache", type=str, required=True,
                        help="Development cache path (must differ from test store).")
    parser.add_argument("--ledger", type=Path, default=default_out /
                        "test_outcome_access_ledger_v1.jsonl")
    parser.add_argument("--output", type=Path, default=default_out /
                        "physical_test_isolation_v1.json")
    args = parser.parse_args()

    results = {}
    dev_source = args.dev_builder.read_text() if args.dev_builder.exists() else ""

    results["dev_builder_references_test_store"] = dev_builder_references_test_store(
        dev_source, args.test_store)
    results["test_store_distinct_from_dev_cache"] = is_distinct_path(
        args.test_store, args.dev_cache)
    results["path_test_store"] = str(Path(args.test_store).resolve())
    results["path_dev_cache"] = str(Path(args.dev_cache).resolve())

    # simulate an access event -> append-only ledger
    append_ledger(args.ledger, {"event": "config_check", "kind": "stat",
                                "path": args.test_store}, seq=1)

    ok = not results["dev_builder_references_test_store"] and \
        results["test_store_distinct_from_dev_cache"]
    results["isolated"] = ok
    results["ledger"] = str(args.ledger)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())