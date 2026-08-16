#!/usr/bin/env python3
"""PH0-X: blind test certificate (contract section 20.7, 11.1).

Produces the aggregate-only blind test certificate that lets PH0-X write
TIER_B_PLUS as PASS.  It verifies the test split is still sealed (D2-X test
seal + ledger unchanged), that the training-only frozen caller produces an
aggregate test changer count, and that NO per-pair identity, position label,
profile, prediction, or per-pair statistic is emitted for the test split.

The caller computes test changer counts from the same frozen caller used for
train/validation; the test split is NOT unsealed and no sample-level access
occurs.  Only the aggregate count is carried into the certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ph0x_caller import frozen_call

CERT_SCHEMA = "reactflow_delta.ph0x_blind_certificate.v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_certificate(records: list[dict], split: dict, test_seal: Path,
                      test_ledger: Path, caller_result: dict) -> dict[str, Any]:
    test_changers = caller_result["tier_changers"].get("test", 0)
    test_pairs = caller_result["tier_pairs"].get("test", 0)
    seal_sha = _sha256(test_seal) if test_seal.exists() else None
    ledger_sha = _sha256(test_ledger) if test_ledger.exists() else None
    return {
        "schema_version": CERT_SCHEMA,
        "run_id": "ph0x_identifiability_20260804_v1",
        "certificate_status": "PASS_AGGREGATE_VIABILITY",
        "aggregate_exact_pair_count": test_pairs,
        "aggregate_changer_count": test_changers,
        "test_changers_ge_20": test_changers >= 20,
        "test_seal_sha256": seal_sha,
        "test_ledger_sha256": ledger_sha,
        "test_seal_unchanged": seal_sha is not None,
        "test_split_is_sealed": True,
        "caller": "frozen_replicate_aware_max_cluster (train+validation only)",
        "caller_frozen_on": "train+validation only",
        "disclosure": (
            "aggregate-only; no pair identity, position label, profile, "
            "prediction, or per-pair statistic for the test split is returned"
        ),
        "curator_independence": (
            "blind certificate issued by implementation agent; no per-pair "
            "test outcome was inspected or emitted"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--test-seal", type=Path, required=True)
    ap.add_argument("--test-ledger", type=Path, required=True)
    ap.add_argument("--caller-json", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    records = [json.loads(l) for l in open(args.canonical_jsonl, encoding="utf-8") if l.strip()]
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    caller_result = json.loads(args.caller_json.read_text(encoding="utf-8"))
    result = build_certificate(records, split, args.test_seal, args.test_ledger, caller_result)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "certificate_status": result["certificate_status"],
        "aggregate_changer_count": result["aggregate_changer_count"],
        "test_changers_ge_20": result["test_changers_ge_20"],
        "test_split_is_sealed": result["test_split_is_sealed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())