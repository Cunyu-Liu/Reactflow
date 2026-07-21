#!/usr/bin/env python3
"""Finalize full-count eFold progress artifacts without copying scores by hand."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from reactflow.protocol import MMSEQS_COMPONENT_HOLDOUT, MMSEQS_COMPONENT_TEST


REQUIRED_ZERO_FIELDS = (
    "missing_count",
    "extra_prediction_count",
    "duplicate_gold_count",
    "sequence_mismatch_count",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_single_tier(path: Path, legacy_tier: str) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tiers"), Mapping):
        raise ValueError(f"{path} does not contain a tiers mapping")
    tier = payload["tiers"].get(legacy_tier)
    if not isinstance(tier, Mapping):
        raise ValueError(f"{path} is missing tier {legacy_tier}")
    tier = dict(tier)
    counts = [int(tier.get(name, -1)) for name in ("matched_count", "gold_count", "count")]
    if len(set(counts)) != 1 or counts[0] <= 0:
        raise ValueError(f"{legacy_tier} is not full-count: {counts}")
    for field in REQUIRED_ZERO_FIELDS:
        if int(tier.get(field, -1)) != 0:
            raise ValueError(f"{legacy_tier} has non-zero {field}")
    provenance = {"progress_artifact": str(path), "progress_sha256": sha256_path(path)}
    for field in ("gold", "predictions"):
        evidence_path = Path(str(tier.get(field, "")))
        if not evidence_path.is_absolute():
            evidence_path = Path.cwd() / evidence_path
        if not evidence_path.is_file():
            raise ValueError(f"{legacy_tier} {field} does not exist: {evidence_path}")
        provenance[f"{field}_path"] = str(evidence_path.resolve())
        provenance[f"{field}_sha256"] = sha256_path(evidence_path)
    tier["legacy_tier"] = legacy_tier
    tier["provenance"] = provenance
    return tier, dict(payload)


def merge_baselines(in_path: Path, holdout_path: Path) -> dict:
    test, test_payload = _load_single_tier(in_path, "in_clan")
    holdout, holdout_payload = _load_single_tier(holdout_path, "novel_clan")
    rows = []
    for label, tier in ((MMSEQS_COMPONENT_TEST, test), (MMSEQS_COMPONENT_HOLDOUT, holdout)):
        rows.append(
            {
                "artifact": tier["provenance"]["progress_artifact"],
                "mean_f1": tier.get("mean_f1"),
                "mean_mcc": tier.get("mean_mcc"),
                "model": "eFold/RNAndria local rerun",
                "protocol": "same_split_local",
                "seed_count": "single_seed",
                "split": f"MMseqs:{label}",
                "status": "ok",
                "tier": label,
            }
        )
    return {
        "schema_version": 2,
        "model": "eFold/RNAndria local rerun",
        "protocol": "same_split_local",
        "seed_count": "single_seed",
        "legacy_aliases": {
            "in_clan": MMSEQS_COMPONENT_TEST,
            "novel_clan": MMSEQS_COMPONENT_HOLDOUT,
        },
        "tiers": {
            MMSEQS_COMPONENT_TEST: test,
            MMSEQS_COMPONENT_HOLDOUT: holdout,
        },
        "rows": rows,
        "source_schema_versions": [test_payload.get("schema_version"), holdout_payload.get("schema_version")],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-test-progress", type=Path, required=True)
    parser.add_argument("--component-holdout-progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = merge_baselines(args.component_test_progress, args.component_holdout_progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tiers": sorted(payload["tiers"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
