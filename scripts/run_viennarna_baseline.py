#!/usr/bin/env python3
"""Run ViennaRNA MFE baseline on ReactFlow gold splits.

Produces prediction JSONL files in the same format as eFold predictions,
consumable by ``evaluate_external_baseline_predictions.py``.

Usage::

    python scripts/run_viennarna_baseline.py \
        --gold-dir <splits_path> \
        --output-dir <output_dir> \
        --tiers test novel
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import RNA  # ViennaRNA Python bindings


# Tier name mapping: gold split filename -> prediction tier name
TIER_MAP = {
    "test": "in_clan",
    "novel": "novel_clan",
}


def dotbracket_to_pairs(db: str) -> List[List[int]]:
    """Convert dot-bracket notation to a list of [i, j] pairs (0-indexed).

    Complexity: O(L) where L = len(db).
    """
    stack: List[int] = []
    pairs: List[List[int]] = []
    for i, c in enumerate(db):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                pairs.append([j, i])
    return pairs


def fold_sequence(seq: str) -> Tuple[str, float]:
    """Fold a sequence with ViennaRNA MFE.

    Returns (dotbracket, mfe).

    Complexity: O(L^3) where L = len(seq).
    """
    # Use default parameters; convert T to U for ViennaRNA
    seq = seq.replace("T", "U").replace("t", "u")
    fc = RNA.fold_compound(seq)
    ss, mfe = fc.mfe()
    return ss, mfe


def run_tier(
    gold_path: Path,
    output_path: Path,
    max_samples: int = 0,
) -> Dict[str, Any]:
    """Run ViennaRNA on one tier and write predictions.

    Args:
        gold_path: path to the gold JSONL file.
        output_path: path to write the prediction JSONL.
        max_samples: if > 0, only fold this many sequences (for smoke test).

    Returns:
        Summary dict with count, elapsed time, etc.
    """
    records = []
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]

    t0 = time.time()
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for rec in records:
            seq = rec["sequence"]
            sid = rec.get("source_id", "")
            try:
                db, mfe = fold_sequence(seq)
                pairs = dotbracket_to_pairs(db)
            except Exception as e:
                print(f"[WARN] Failed to fold {sid}: {e}", file=sys.stderr)
                db = "." * len(seq)
                pairs = []
            pred = {
                "dotbracket": db,
                "predicted_pairs": pairs,
                "prediction_backend": "viennarna",
                "sequence": seq,
                "source_id": sid,
            }
            out.write(json.dumps(pred) + "\n")
            count += 1
            if count % 2000 == 0:
                elapsed = time.time() - t0
                print(f"  [{count}/{len(records)}] {elapsed:.1f}s", file=sys.stderr)

    elapsed = time.time() - t0
    return {
        "tier": gold_path.stem,
        "count": count,
        "elapsed_seconds": round(elapsed, 2),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ViennaRNA MFE baseline")
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="Directory containing gold JSONL split files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write prediction JSONL files")
    parser.add_argument("--tiers", nargs="+", default=["test", "novel"],
                        help="Gold split names to process (e.g. test novel)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per tier (0 = all)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] ViennaRNA version: {RNA.__version__}", file=sys.stderr)
    print(f"[INFO] Gold dir: {args.gold_dir}", file=sys.stderr)
    print(f"[INFO] Output dir: {args.output_dir}", file=sys.stderr)

    summaries = []
    for tier in args.tiers:
        gold_path = args.gold_dir / f"{tier}.jsonl"
        if not gold_path.exists():
            print(f"[WARN] Gold file not found: {gold_path}", file=sys.stderr)
            continue
        tier_name = TIER_MAP.get(tier, tier)
        output_path = args.output_dir / f"{tier_name}.viennarna.predictions.jsonl"
        print(f"[INFO] Processing tier={tier} -> {tier_name}", file=sys.stderr)
        summary = run_tier(gold_path, output_path, args.max_samples)
        summaries.append(summary)
        print(f"[INFO] Done: {summary}", file=sys.stderr)

    summary_path = args.output_dir / "viennarna_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model": "ViennaRNA MFE", "summaries": summaries}, f, indent=2)
    print(f"[INFO] Summary written to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
