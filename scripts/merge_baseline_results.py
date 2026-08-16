#!/usr/bin/env python3
"""Merge individual baseline evaluation results into baseline_same_split_results.json.

Reads all baseline_*_results.json files from evaluation_results/ directory and
merges them into the canonical baseline_same_split_results.json file that the
C1-3 gate audit script expects.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def merge_results(artifacts_dir: Path) -> None:
    """Merge all baseline results into the canonical file."""
    canonical_path = artifacts_dir / "baseline_same_split_results.json"
    eval_dir = artifacts_dir / "baselines" / "evaluation_results"

    # Load canonical file (or create new)
    if canonical_path.exists():
        with open(canonical_path) as f:
            canonical = json.load(f)
    else:
        canonical = {
            "schema_version": 1,
            "protocol": "same_split_local",
            "baselines": [],
            "summary": {},
        }

    if not eval_dir.exists():
        print(f"[merge] No evaluation_results dir at {eval_dir}", file=sys.stderr)
        return

    # Load all individual results
    result_files = sorted(eval_dir.glob("baseline_*_results.json"))
    print(f"[merge] Found {len(result_files)} result files", file=sys.stderr)

    # Build a set of existing model names
    existing_models = {b.get("model", "") for b in canonical.get("baselines", [])}

    for rf in result_files:
        with open(rf) as f:
            data = json.load(f)
        model = data.get("model", rf.stem)
        print(f"[merge] Processing {model} from {rf.name}", file=sys.stderr)

        # If model already exists, replace it; otherwise append
        canonical["baselines"] = [
            b for b in canonical.get("baselines", []) if b.get("model") != model
        ]
        canonical["baselines"].append(data)

        # Update summary
        rows = data.get("rows", [])
        model_key = model
        tier_summary = {}
        for row in rows:
            tier = row.get("tier", "")
            tier_summary[tier] = {
                "mean_f1": row.get("mean_f1", 0.0),
                "mean_mcc": row.get("mean_mcc", 0.0),
                "long_f1": row.get("long_f1", 0.0),
                "long_recall": row.get("long_recall", 0.0),
            }
        canonical["summary"][model_key] = tier_summary

    # Write back
    with open(canonical_path, "w") as f:
        json.dump(canonical, f, indent=2)
    print(f"[merge] Written to {canonical_path}", file=sys.stderr)
    print(f"[merge] Total baselines: {len(canonical['baselines'])}", file=sys.stderr)
    for b in canonical["baselines"]:
        model = b.get("model", "?")
        rows = b.get("rows", [])
        tiers = [r.get("tier", "?") for r in rows]
        print(f"  {model}: {tiers}", file=sys.stderr)


if __name__ == "__main__":
    artifacts_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/c1_3")
    merge_results(artifacts_dir)
