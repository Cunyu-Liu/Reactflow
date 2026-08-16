#!/usr/bin/env python3
"""Run MXfold2 baseline on ReactFlow gold splits.

Produces prediction JSONL files in the same format as ViennaRNA predictions,
consumable by ``evaluate_external_baseline_predictions.py``.

Usage::

    python scripts/run_mxfold2_baseline.py \
        --gold-dir <splits_path> \
        --output-dir <output_dir> \
        --tiers test novel

Note: MXfold2 must be installed in the active environment (``pip install mxfold2``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def fold_batch_mxfold2(records: List[Dict[str, Any]], batch_size: int = 200) -> List[Tuple[str, str, float]]:
    """Fold a batch of sequences with MXfold2.

    Writes all sequences to a single FASTA file and calls mxfold2 predict once,
    loading the model only once per batch.

    Returns list of (source_id, dotbracket, score) tuples.
    """
    import tempfile
    import shutil
    results: List[Tuple[str, str, float]] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="mxfold2_batch_"))

    try:
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start:batch_start + batch_size]
            fa_path = temp_dir / f"batch_{batch_start}.fa"
            with open(fa_path, "w") as f:
                for i, rec in enumerate(batch):
                    seq = rec["sequence"].replace("T", "U").replace("t", "u")
                    sid = rec.get("source_id", f"seq_{batch_start + i}")
                    f.write(f">{sid}\n{seq}\n")

            try:
                result = subprocess.run(
                    ["mxfold2", "predict", str(fa_path)],
                    capture_output=True, text=True, timeout=3600,
                )
                if result.returncode != 0:
                    for rec in batch:
                        seq = rec["sequence"]
                        results.append((rec.get("source_id", ""), "." * len(seq), 0.0))
                    continue

                # Parse output: multiple FASTA records
                lines = result.stdout.strip().split("\n")
                i = 0
                while i < len(lines):
                    if lines[i].startswith(">"):
                        sid = lines[i][1:]
                        seq = lines[i + 1] if i + 1 < len(lines) else ""
                        db_line = lines[i + 2] if i + 2 < len(lines) else ""
                        parts = db_line.split()
                        db = parts[0] if parts else ""
                        score = 0.0
                        if len(parts) > 1:
                            score_str = parts[1].strip("()")
                            try:
                                score = float(score_str)
                            except ValueError:
                                pass
                        results.append((sid, db, score))
                        i += 3
                    else:
                        i += 1
            except subprocess.TimeoutExpired:
                for rec in batch:
                    seq = rec["sequence"]
                    results.append((rec.get("source_id", ""), "." * len(seq), 0.0))
            except Exception as e:
                print(f"[WARN] MXfold2 batch error: {e}", file=sys.stderr)
                for rec in batch:
                    seq = rec["sequence"]
                    results.append((rec.get("source_id", ""), "." * len(seq), 0.0))

            elapsed_so_far = batch_start + len(batch)
            print(f"  [{elapsed_so_far}/{len(records)}]", file=sys.stderr)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def run_tier(
    gold_path: Path,
    output_path: Path,
    max_samples: int = 0,
    batch_size: int = 200,
) -> Dict[str, Any]:
    """Run MXfold2 on one tier and write predictions.

    Args:
        gold_path: path to the gold JSONL file.
        output_path: path to write the prediction JSONL.
        max_samples: if > 0, only fold this many sequences (for smoke test).
        batch_size: number of sequences per mxfold2 invocation.

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
    print(f"[INFO] Folding {len(records)} sequences in batches of {batch_size}", file=sys.stderr)

    batch_results = fold_batch_mxfold2(records, batch_size)

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            sid = rec.get("source_id", "")
            seq = rec["sequence"]
            if i < len(batch_results):
                _, db, score = batch_results[i]
            else:
                db = "." * len(seq)
                score = 0.0
            pairs = dotbracket_to_pairs(db)
            pred = {
                "dotbracket": db,
                "predicted_pairs": pairs,
                "prediction_backend": "mxfold2",
                "sequence": seq,
                "source_id": sid,
            }
            out.write(json.dumps(pred) + "\n")
            count += 1

    elapsed = time.time() - t0
    return {
        "tier": gold_path.stem,
        "count": count,
        "elapsed_seconds": round(elapsed, 2),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MXfold2 baseline")
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="Directory containing gold JSONL split files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write prediction JSONL files")
    parser.add_argument("--tiers", nargs="+", default=["test", "novel"],
                        help="Gold split names to process (e.g. test novel)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per tier (0 = all)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Number of sequences per mxfold2 invocation")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] MXfold2 baseline", file=sys.stderr)
    print(f"[INFO] Gold dir: {args.gold_dir}", file=sys.stderr)
    print(f"[INFO] Output dir: {args.output_dir}", file=sys.stderr)

    summaries = []
    for tier in args.tiers:
        gold_path = args.gold_dir / f"{tier}.jsonl"
        if not gold_path.exists():
            print(f"[WARN] Gold file not found: {gold_path}", file=sys.stderr)
            continue
        tier_name = TIER_MAP.get(tier, tier)
        output_path = args.output_dir / f"{tier_name}.mxfold2.predictions.jsonl"
        print(f"[INFO] Processing tier={tier} -> {tier_name}", file=sys.stderr)
        summary = run_tier(gold_path, output_path, args.max_samples, args.batch_size)
        summaries.append(summary)
        print(f"[INFO] Done: {summary}", file=sys.stderr)

    summary_path = args.output_dir / "mxfold2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model": "MXfold2", "summaries": summaries}, f, indent=2)
    print(f"[INFO] Summary written to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
