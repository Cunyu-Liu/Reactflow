#!/usr/bin/env python3
"""Run MXfold2 baseline using Python API for fast in-process prediction.

Loads the MXfold2 model once and predicts secondary structure for each
sequence individually, avoiding the batch-mode hang issue.

Produces prediction JSONL files consumable by ``evaluate_external_baseline_predictions.py``.

Usage::

    python scripts/run_mxfold2_baseline_v2.py \
        --gold-dir <splits_path> \
        --output-dir <output_dir> \
        --tiers test novel \
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple


TIER_MAP = {
    "test": "in_clan",
    "novel": "novel_clan",
}

# Global model instance (initialized per worker)
_MODEL_INSTANCE = None


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


class _WorkerModel:
    """Wraps the MXfold2 model loaded in-process for fast prediction.

    Uses the default MixC model with TrainSetAB.pth parameters, matching
    the CLI default configuration (``mxfold2 predict`` with no --param).
    """

    def __init__(self) -> None:
        """Load the MXfold2 MixC model with TrainSetAB parameters."""
        import torch
        from mxfold2 import param_turner2004
        from mxfold2.fold.mix import MixedFold

        # Configuration from TrainSetAB.conf (default MXfold2 config)
        config = {
            'max_helix_length': 30,
            'embed_size': 64,
            'num_filters': (64, 64, 64, 64, 64, 64, 64, 64),
            'filter_size': (5, 3, 5, 3, 5, 3, 5, 3),
            'pool_size': (1,),
            'dilation': 0,
            'num_lstm_layers': 2,
            'num_lstm_units': 32,
            'num_transformer_layers': 0,
            'num_transformer_hidden_units': 2048,
            'num_transformer_att': 8,
            'num_hidden_units': (32,),
            'num_paired_filters': (64, 64, 64, 64, 64, 64, 64, 64),
            'paired_filter_size': 5,  # will be overridden by tuple below
            'dropout_rate': 0.5,
            'fc_dropout_rate': 0.5,
            'num_att': 8,
            'pair_join': 'cat',
            'no_split_lr': False,
        }
        # paired_filter_size is appended 8 times in config -> tuple
        config['paired_filter_size'] = (5, 3, 5, 3, 5, 3, 5, 3)

        # MixC model (model_type='C') with Turner2004 init params
        self.model = MixedFold(
            init_param=param_turner2004,
            model_type='C',
            **config,
        )

        # Load default trained parameters (TrainSetAB.pth)
        import mxfold2
        param_path = os.path.join(
            os.path.dirname(mxfold2.__file__),
            'models', 'TrainSetAB.pth'
        )
        if os.path.exists(param_path):
            state = torch.load(param_path, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            self.model.load_state_dict(state)
            print(f"[Worker] Loaded params from {param_path}", file=sys.stderr, flush=True)
        else:
            print(f"[WARN] Param file not found: {param_path}", file=sys.stderr, flush=True)

        self.model.eval()
        self.torch = torch

    def predict(self, sequence: str) -> str:
        """Predict dot-bracket structure for a single sequence.

        Returns dot-bracket notation string.
        """
        import torch
        seq = sequence.replace('T', 'U').replace('t', 'u')
        length = len(seq)
        if length == 0:
            return ""
        # MXfold2 model.forward expects a list of sequence strings
        # (matching FastaDataset output via DataLoader with batch_size=1)
        try:
            with torch.no_grad():
                scs, preds, bps = self.model([seq])
            return preds[0]
        except Exception as e:
            print(f"[WARN] MXfold2 predict failed (len={length}): {e}", file=sys.stderr)
            return "." * length


def _init_worker() -> None:
    """Initialize the global model instance in each worker process."""
    global _MODEL_INSTANCE
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Limit threads per worker to avoid contention on many-core machines
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    import torch
    torch.set_num_threads(2)
    _MODEL_INSTANCE = _WorkerModel()
    print(f"[Worker {os.getpid()}] MXfold2 model loaded", file=sys.stderr, flush=True)


def _predict_one(args: Tuple[int, str, str]) -> Tuple[int, str, str]:
    """Predict structure for one sequence.

    Args:
        args: (index, source_id, sequence)

    Returns:
        (index, source_id, dotbracket)
    """
    global _MODEL_INSTANCE
    idx, sid, seq = args
    db = _MODEL_INSTANCE.predict(seq)
    return (idx, sid, db)


def _timeout_handler(signum, frame):
    raise TimeoutError("Prediction timed out")


def run_tier(
    gold_path: Path,
    output_path: Path,
    max_samples: int = 0,
    workers: int = 8,
    seq_timeout: int = 30,
) -> Dict[str, Any]:
    """Run MXfold2 on one tier and write predictions.

    Args:
        gold_path: path to the gold JSONL file.
        output_path: path to write the prediction JSONL.
        max_samples: if > 0, only fold this many sequences.
        workers: number of parallel worker processes.
        seq_timeout: per-sequence timeout in seconds.

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
    print(f"[INFO] Folding {len(records)} sequences with {workers} workers", file=sys.stderr)

    # Prepare work items
    work_items = [(i, rec.get("source_id", f"seq_{i}"), rec["sequence"]) for i, rec in enumerate(records)]

    results: List[Tuple[int, str, str]] = [None] * len(records)  # type: ignore

    # Use multiprocessing Pool with initializer
    with Pool(processes=workers, initializer=_init_worker) as pool:
        done = 0
        total = len(work_items)
        for idx, sid, db in pool.imap_unordered(_predict_one, work_items, chunksize=20):
            results[idx] = (idx, sid, db)
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] {elapsed:.1f}s, {rate:.1f} seq/s, ETA {eta:.0f}s", file=sys.stderr, flush=True)

    # Fill in any None results with fallback
    for i in range(len(results)):
        if results[i] is None:
            seq = records[i]["sequence"]
            results[i] = (i, records[i].get("source_id", ""), "." * len(seq))

    # Write output
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            _, sid, db = results[i]
            seq = rec["sequence"]
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
    parser = argparse.ArgumentParser(description="MXfold2 baseline (Python API)")
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="Directory containing gold JSONL split files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write prediction JSONL files")
    parser.add_argument("--tiers", nargs="+", default=["test", "novel"],
                        help="Gold split names to process (e.g. test novel)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per tier (0 = all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes")
    parser.add_argument("--seq-timeout", type=int, default=30,
                        help="Per-sequence timeout in seconds (not yet implemented)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] MXfold2 baseline (Python API v2)", file=sys.stderr)
    print(f"[INFO] Gold dir: {args.gold_dir}", file=sys.stderr)
    print(f"[INFO] Output dir: {args.output_dir}", file=sys.stderr)
    print(f"[INFO] Workers: {args.workers}", file=sys.stderr)

    summaries = []
    for tier in args.tiers:
        gold_path = args.gold_dir / f"{tier}.jsonl"
        if not gold_path.exists():
            print(f"[WARN] Gold file not found: {gold_path}", file=sys.stderr)
            continue
        tier_name = TIER_MAP.get(tier, tier)
        output_path = args.output_dir / f"{tier_name}.mxfold2.predictions.jsonl"
        print(f"[INFO] Processing tier={tier} -> {tier_name}", file=sys.stderr)
        summary = run_tier(gold_path, output_path, args.max_samples, args.workers, args.seq_timeout)
        summaries.append(summary)
        print(f"[INFO] Done: {summary}", file=sys.stderr)

    summary_path = args.output_dir / "mxfold2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model": "MXfold2", "summaries": summaries}, f, indent=2)
    print(f"[INFO] Summary written to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
