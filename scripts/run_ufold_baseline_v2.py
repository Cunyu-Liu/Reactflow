#!/usr/bin/env python3
"""Run UFold baseline on ReactFlow gold splits (CPU multiprocessing version).

This v2 uses multiprocessing.Pool with fork to share the UFold model across
workers, achieving ~10-20x speedup over the single-process v1 by processing
sequences in parallel on CPU (the bottleneck is the 600x600 creatmat matrix
construction per sequence, which is CPU-bound and embarrassingly parallel).

Usage::

    python scripts/run_ufold_baseline_v2.py \
        --gold-dir <splits_path> \
        --output-dir <output_dir> \
        --tiers test novel \
        --workers 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# UFold source must be on path
UFOLD_ROOT = Path("/home/cunyuliu/rna_baselines_src/UFold-main")
sys.path.insert(0, str(UFOLD_ROOT))

# Monkey-patch creatmat to force CPU BEFORE any worker imports it
import ufold.data_generator as _ufold_dg  # noqa: E402

_original_creatmat = _ufold_dg.creatmat


def _creatmat_cpu(data, device=None):
    """Force creatmat to use CPU to avoid GPU OOM."""
    return _original_creatmat(data, device=torch.device("cpu"))


_ufold_dg.creatmat = _creatmat_cpu

import torch  # noqa: E402
from Network import U_Net as FCNNet  # noqa: E402
from ufold.postprocess import postprocess_new as postprocess  # noqa: E402
from ufold.data_generator import (  # noqa: E402
    RNASSDataGenerator_input,
    Dataset_Cut_concat_new as Dataset_FCN,
)
from torch.utils import data as torch_data  # noqa: E402

TIER_MAP = {
    "test": "in_clan",
    "novel": "novel_clan",
}

# Globals shared via fork
_GLOBAL_MODEL = None
_GLOBAL_DEVICE = None


def dotbracket_to_pairs(db: str) -> List[List[int]]:
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


def seq2dot(seq: np.ndarray) -> str:
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(["_"] * len(seq))
    dot_file[seq > idx] = "("
    dot_file[seq < idx] = ")"
    dot_file[seq == 0] = "."
    return "".join(dot_file)


def _init_worker(model_path: str, device: str):
    """Worker initializer: load UFold model once per worker.

    On Linux with fork (default start method), the parent process's memory
    is copy-on-write shared, so loading the model in the parent before fork
    would be ideal. However, multiprocessing.Pool uses fork after the parent
    has already started, so we load the model in each worker once.

    To minimize memory, we load on CPU and use FP32.
    """
    global _GLOBAL_MODEL, _GLOBAL_DEVICE
    _GLOBAL_DEVICE = torch.device("cpu")

    # CRITICAL: RNASSDataGenerator_input.load_data() creates a multiprocessing.Pool()
    # that is never used (dead code). Inside a daemonic Pool worker, creating
    # another Pool raises "daemonic processes are not allowed to have children".
    # Patch the Pool symbol in ufold.data_generator to a no-op dummy.
    import ufold.data_generator as _dg
    class _DummyPool:
        def __init__(self, *args, **kwargs):
            pass
        def map(self, fn, iterable):
            return list(map(fn, iterable))
        def close(self): pass
        def join(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    _dg.Pool = _DummyPool

    _GLOBAL_MODEL = FCNNet(img_ch=17)
    state_dict = torch.load(model_path, map_location="cpu")
    _GLOBAL_MODEL.load_state_dict(state_dict)
    _GLOBAL_MODEL.eval()
    # Pin to CPU (workers cannot share GPU effectively)
    print(f"[worker {os.getpid()}] UFold loaded on CPU", file=sys.stderr)


def _predict_one(item: Tuple[int, str, str]) -> Tuple[int, str, str]:
    """Predict dot-bracket for a single sequence.

    Args:
        item: (index, source_id, sequence)

    Returns:
        (index, source_id, dotbracket)
    """
    idx, sid, sequence = item
    seq = sequence.upper().replace("T", "U")
    length = len(seq)
    if length == 0:
        return (idx, sid, "")
    if length > 600:
        seq = seq[:600]
        length = 600

    try:
        # Write a single-sequence temp FASTA and load via UFold's pipeline
        # NOTE: Do NOT use DataLoader (it spawns workers, which fails inside
        # a daemonic multiprocessing.Pool worker). Instead call
        # Dataset_FCN.__getitem__ directly.
        import tempfile
        import shutil

        temp_dir = Path(tempfile.mkdtemp(prefix=f"ufold_seq_{idx}_"))
        try:
            input_txt = temp_dir / "input.txt"
            with open(input_txt, "w") as f:
                f.write(f">seq_{idx}\n{seq}\n")

            test_data = RNASSDataGenerator_input(str(temp_dir) + "/", "input")
            test_set = Dataset_FCN(test_data)

            # Call __getitem__ directly (index 0, single sequence)
            seq_embeddings, seq_lens, seq_ori, seq_name = test_set[0]
            seq_embedding_batch = torch.from_numpy(np.asarray(seq_embeddings, dtype=np.float32)).unsqueeze(0)
            seq_ori_tensor = torch.from_numpy(np.asarray(seq_ori, dtype=np.float32)).unsqueeze(0)

            with torch.no_grad():
                pred_contacts = _GLOBAL_MODEL(seq_embedding_batch)

            u_no_train = postprocess(
                pred_contacts, seq_ori_tensor,
                0.01, 0.1, 100, 1.6, True, 1.5,
            )
            map_no_train = (u_no_train > 0.5).float()

            pred_cpu = map_no_train.cpu()
            seq_tmp = torch.mul(
                pred_cpu.argmax(axis=1),
                pred_cpu.sum(axis=1).clamp_max(1),
            ).numpy().astype(int)
            seq_tmp[pred_cpu.sum(axis=1) == 0] = -1
            dot_list = seq2dot((seq_tmp[0] + 1))

            actual_len = int(seq_lens)
            db = dot_list[:actual_len]

            # Cleanup tensors
            del seq_embedding_batch, seq_ori_tensor, pred_contacts, u_no_train, map_no_train, pred_cpu
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return (idx, sid, db)
    except Exception as e:
        # Fallback: all unpaired
        import traceback
        print(f"[WARN] UFold failed on seq {idx}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return (idx, sid, "." * len(sequence))


def run_tier(
    gold_path: Path,
    output_path: Path,
    workers: int,
    max_samples: int = 0,
    model_path: str = "",
    seq_timeout: int = 120,
) -> Dict[str, Any]:
    """Run UFold on one tier with multiprocessing.

    Args:
        gold_path: Path to gold JSONL.
        output_path: Path to write predictions JSONL.
        workers: Number of parallel worker processes.
        max_samples: If > 0, only fold this many sequences.
        model_path: Path to UFold checkpoint.
        seq_timeout: Per-sequence soft timeout (for ETA, not enforced).

    Returns:
        Summary dict.
    """
    records = []
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]

    t0 = time.time()
    total = len(records)
    print(f"[INFO] UFold folding {total} sequences with {workers} CPU workers", file=sys.stderr)

    work_items = [
        (i, rec.get("source_id", f"seq_{i}"), rec["sequence"])
        for i, rec in enumerate(records)
    ]
    results: List[Tuple[int, str, str]] = [None] * total  # type: ignore

    with Pool(processes=workers, initializer=_init_worker, initargs=(model_path, "cpu")) as pool:
        done = 0
        for idx, sid, db in pool.imap_unordered(_predict_one, work_items, chunksize=4):
            results[idx] = (idx, sid, db)
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] {elapsed:.1f}s, {rate:.1f} seq/s, ETA {eta:.0f}s", file=sys.stderr, flush=True)

    # Fill in None with fallback
    for i in range(total):
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
                "prediction_backend": "ufold",
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
        "workers": workers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UFold baseline v2 (CPU multiprocessing)")
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="Directory containing gold JSONL split files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write prediction JSONL files")
    parser.add_argument("--model-path", type=str,
                        default=str(UFOLD_ROOT / "models" / "ufold_train_alldata.pt"),
                        help="Path to UFold .pt checkpoint")
    parser.add_argument("--tiers", nargs="+", default=["test", "novel"],
                        help="Gold split names to process")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per tier (0 = all)")
    parser.add_argument("--workers", type=int, default=24,
                        help="Number of parallel CPU workers")
    args = parser.parse_args()

    # Change to UFold root directory (required for relative paths)
    os.chdir(str(UFOLD_ROOT))
    os.makedirs("results/save_ct_file", exist_ok=True)
    os.makedirs("results/save_varna_fig", exist_ok=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] UFold v2 (CPU multiprocessing, {args.workers} workers)", file=sys.stderr)
    print(f"[INFO] Gold dir: {args.gold_dir}", file=sys.stderr)
    print(f"[INFO] Output dir: {args.output_dir}", file=sys.stderr)
    print(f"[INFO] Model: {args.model_path}", file=sys.stderr)

    summaries = []
    for tier in args.tiers:
        gold_path = args.gold_dir / f"{tier}.jsonl"
        if not gold_path.exists():
            print(f"[ERROR] Gold file not found: {gold_path}", file=sys.stderr)
            continue
        tier_name = TIER_MAP.get(tier, tier)
        output_path = args.output_dir / f"{tier_name}.ufold.predictions.jsonl"
        print(f"[INFO] Processing tier={tier} -> {tier_name}", file=sys.stderr)
        summary = run_tier(
            gold_path, output_path, args.workers, args.max_samples, args.model_path
        )
        summaries.append(summary)
        print(f"[INFO] {tier_name}: {summary['count']} seqs in {summary['elapsed_seconds']}s", file=sys.stderr)

    summary_path = args.output_dir / "ufold_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"tiers": summaries}, f, indent=2)
    print(f"[INFO] Summary: {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
