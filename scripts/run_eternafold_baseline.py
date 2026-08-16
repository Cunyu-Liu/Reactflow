#!/usr/bin/env python3
"""Run EternaFold baseline on ReactFlow gold splits.

Produces prediction JSONL files in the same format as ViennaRNA predictions,
consumable by ``evaluate_external_baseline_predictions.py``.

Usage::

    python scripts/run_eternafold_baseline.py \
        --gold-dir <splits_path> \
        --output-dir <output_dir> \
        --tiers test novel

Note: EternaFold must be installed (``conda install -c bioconda eternafold``).
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


def fold_sequence_eternafold(seq: str, temp_dir: Path, idx: int) -> str:
    """Fold a single sequence with EternaFold.

    Returns dotbracket string.
    """
    seq = seq.replace("T", "U").replace("t", "u")
    fa_path = temp_dir / f"seq_{idx}.fa"
    with open(fa_path, "w") as f:
        f.write(f">seq_{idx}\n{seq}\n")
    try:
        result = subprocess.run(
            ["eternafold", "predict", str(fa_path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return "." * len(seq)
        # Parse output: >seq_N\nSEQUENCE\n>structure\nDOTBRACKET
        lines = result.stdout.strip().split("\n")
        for i, line in enumerate(lines):
            if line.startswith(">structure") and i + 1 < len(lines):
                return lines[i + 1]
    except Exception:
        pass
    return "." * len(seq)


def fold_batch_eternafold(records: List[Dict[str, Any]], batch_size: int = 5000) -> List[Tuple[str, str]]:
    """Fold sequences with EternaFold.

    EternaFold requires all sequences in one invocation to have the same length,
    so we fold each sequence individually. EternaFold is very fast (~0.01s/seq).

    Returns list of (source_id, dotbracket) tuples.
    """
    import shutil
    results: List[Tuple[str, str]] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="eternafold_"))

    try:
        for i, rec in enumerate(records):
            seq = rec["sequence"]
            sid = rec.get("source_id", f"seq_{i}")
            db = fold_sequence_eternafold(seq, temp_dir, i)
            results.append((sid, db))
            if (i + 1) % 2000 == 0:
                print(f"  [{i + 1}/{len(records)}]", file=sys.stderr)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def run_tier(
    gold_path: Path,
    output_path: Path,
    max_samples: int = 0,
    batch_size: int = 5000,
) -> Dict[str, Any]:
    """Run EternaFold on one tier and write predictions.

    Args:
        gold_path: path to the gold JSONL file.
        output_path: path to write the prediction JSONL.
        max_samples: if > 0, only fold this many sequences (for smoke test).
        batch_size: number of sequences per eternafold invocation.

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

    batch_results = fold_batch_eternafold(records, batch_size)

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            sid = rec.get("source_id", "")
            seq = rec["sequence"]
            if i < len(batch_results):
                _, db = batch_results[i]
            else:
                db = "." * len(seq)
            pairs = dotbracket_to_pairs(db)
            pred = {
                "dotbracket": db,
                "predicted_pairs": pairs,
                "prediction_backend": "eternafold",
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
    parser = argparse.ArgumentParser(description="EternaFold baseline")
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="Directory containing gold JSONL split files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write prediction JSONL files")
    parser.add_argument("--tiers", nargs="+", default=["test", "novel"],
                        help="Gold split names to process (e.g. test novel)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per tier (0 = all)")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Number of sequences per eternafold invocation")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] EternaFold baseline", file=sys.stderr)
    print(f"[INFO] Gold dir: {args.gold_dir}", file=sys.stderr)
    print(f"[INFO] Output dir: {args.output_dir}", file=sys.stderr)

    summaries = []
    for tier in args.tiers:
        gold_path = args.gold_dir / f"{tier}.jsonl"
        if not gold_path.exists():
            print(f"[WARN] Gold file not found: {gold_path}", file=sys.stderr)
            continue
        tier_name = TIER_MAP.get(tier, tier)
        output_path = args.output_dir / f"{tier_name}.eternafold.predictions.jsonl"
        print(f"[INFO] Processing tier={tier} -> {tier_name}", file=sys.stderr)
        summary = run_tier(gold_path, output_path, args.max_samples, args.batch_size)
        summaries.append(summary)
        print(f"[INFO] Done: {summary}", file=sys.stderr)

    summary_path = args.output_dir / "eternafold_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model": "EternaFold", "summaries": summaries}, f, indent=2)
    print(f"[INFO] Summary written to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
