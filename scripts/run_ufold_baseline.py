#!/usr/bin/env python3
"""Run UFold baseline on ReactFlow gold splits.

Loads the pretrained UFold model and predicts RNA secondary structure.
Uses UFold's U-Net architecture with post-processing.

Produces prediction JSONL files consumable by ``evaluate_external_baseline_predictions.py``.

Usage::

    python scripts/run_ufold_baseline.py \
        --gold-dir <splits_path> \
        --output-dir <output_dir> \
        --ufold-root /home/cunyuliu/rna_baselines_src/UFold-main \
        --tiers test novel
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# UFold source must be on path
UFOLD_ROOT = Path("/home/cunyuliu/rna_baselines_src/UFold-main")
sys.path.insert(0, str(UFOLD_ROOT))

from Network import U_Net as FCNNet
from ufold.postprocess import postprocess_new as postprocess

# Monkey-patch creatmat to force CPU (avoids GPU OOM on MIG instances)
# The original creatmat defaults to cuda:0 when CUDA is available, which
# causes OOM because it creates large tensors (n x n x 30) on GPU.
import ufold.data_generator as _ufold_dg
_original_creatmat = _ufold_dg.creatmat


def _creatmat_cpu(data, device=None):
    """Force creatmat to use CPU to avoid GPU OOM."""
    return _original_creatmat(data, device=torch.device('cpu'))


_ufold_dg.creatmat = _creatmat_cpu


TIER_MAP = {
    "test": "in_clan",
    "novel": "novel_clan",
}


def dotbracket_to_pairs(db: str) -> List[List[int]]:
    """Convert dot-bracket notation to a list of [i, j] pairs (0-indexed)."""
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
    """Convert pair index array to dot-bracket notation.

    Args:
        seq: 1-indexed pair positions (0 = unpaired).

    Returns:
        Dot-bracket string.
    """
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    return ''.join(dot_file)


def one_hot_600(seq_item: str) -> np.ndarray:
    """One-hot encode a sequence to 600 length (UFold's fixed input size)."""
    seq_item = seq_item.upper().replace('T', 'U')
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES
         else np.array([[-1] * len(BASES)]) for base in seq_item])
    one_hot_matrix_600 = np.zeros((600, 4))
    one_hot_matrix_600[:len(seq_item)] = feat
    return one_hot_matrix_600


class UFoldPredictor:
    """Wraps the UFold model for batch prediction.

    The model runs on GPU for inference, but data loading (including the
    memory-intensive ``creatmat`` function) runs on CPU to avoid GPU OOM
    on MIG instances with limited memory.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """Load UFold model.

        Args:
            model_path: Path to the .pt checkpoint.
            device: "auto", "cuda", or "cpu".
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Create model: U-Net with 17 input channels
        # Model is created on CPU, then moved to GPU for inference only
        self.model = FCNNet(img_ch=17)

        # Load pretrained weights
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if self.device.type == "cuda":
            # Move model to GPU for inference
            self.model.to(self.device)

        print(f"[INFO] UFold loaded on {self.device} (data loading on CPU)", file=sys.stderr)

    @torch.no_grad()
    def predict(self, sequence: str) -> str:
        """Predict dot-bracket structure for a single sequence.

        Args:
            sequence: RNA sequence (A/C/G/U/T/N).

        Returns:
            Dot-bracket notation string.
        """
        seq = sequence.upper().replace('T', 'U')
        length = len(seq)
        if length == 0:
            return ""
        if length > 600:
            # UFold max length is 600; truncate
            seq = seq[:600]
            length = 600

        # One-hot encode
        one_hot = one_hot_600(seq)  # (600, 4)
        # Create seq_ori: the one-hot padded to 600x600 for the model
        # UFold's Dataset_FCN creates a 600x600 matrix from the one-hot
        # The model expects input shape (batch, 17, 600, 600)
        # Let me check what Dataset_FCN produces

        # Actually, looking at the UFold code more carefully:
        # seq_embeddings is (600, 4) one-hot
        # seq_ori is (1, 600, 600) - original sequence as a matrix
        # The model takes seq_embedding_batch which is (batch, 17, 600, 600)

        # Let me reconstruct the input pipeline from Dataset_FCN
        # Dataset_FCN.__getitem__ returns: seq_embedding, seq_lens, seq_ori, seq_name
        # where seq_embedding is the 17-channel input

        # For now, use the simpler approach: create the 17-channel input manually
        # The 17 channels come from: one_hot(4) + pairing_matrix(1) + other features(12)
        # Actually, looking at data_generator more carefully...

        # Simpler approach: use the same encoding as RNASSDataGenerator_input
        # and Dataset_FCN
        seq_embedding = self._prepare_input(seq)
        seq_ori = self._prepare_seq_ori(seq)

        seq_embedding_batch = torch.Tensor(seq_embedding).unsqueeze(0).to(self.device)
        seq_ori_tensor = torch.Tensor(seq_ori).unsqueeze(0).to(self.device)

        # Run model
        pred_contacts = self.model(seq_embedding_batch)

        # Post-process
        u_no_train = postprocess(
            pred_contacts, seq_ori_tensor,
            0.01, 0.1, 100, 1.6, True, 1.5
        )
        map_no_train = (u_no_train > 0.5).float()

        # Extract dot-bracket from contact map
        # Use the same logic as get_ct_dict_fast
        pred_cpu = map_no_train.cpu()
        seq_tmp = torch.mul(
            pred_cpu.argmax(axis=1),
            pred_cpu.sum(axis=1).clamp_max(1)
        ).numpy().astype(int)
        seq_tmp[pred_cpu.sum(axis=1) == 0] = -1
        dot_list = seq2dot((seq_tmp[0] + 1))

        # Truncate to actual sequence length
        return dot_list[:length]

    def _prepare_input(self, seq: str) -> np.ndarray:
        """Prepare 17-channel input for UFold model.

        UFold uses 17 channels: 4 (one-hot) + 1 (pairing matrix) + 12 (position features).
        """
        # This is complex; let me use the data_generator's approach
        # Actually, let me check what Dataset_FCN does
        # For now, create a minimal 17-channel input
        length = len(seq)
        # Channel 0-3: one-hot encoding
        one_hot = one_hot_600(seq)  # (600, 4)

        # Channel 4: pairing matrix (from creatmat)
        # Channels 5-16: position features
        # This is complex; let me use the actual data_generator

        # Actually, the simplest approach is to use Dataset_FCN directly
        # But that requires setting up the data in UFold's format
        # Let me use the data_generator's encoding instead

        raise NotImplementedError("Use _predict_via_generator instead")

    def _prepare_seq_ori(self, seq: str) -> np.ndarray:
        """Prepare seq_ori matrix."""
        length = len(seq)
        matrix = np.zeros((1, 600, 600))
        for i in range(min(length, 600)):
            matrix[0, i, i] = 1
        return matrix

    @torch.no_grad()
    def predict_batch(self, sequences: List[str]) -> List[str]:
        """Predict structures for a batch of sequences using UFold's data pipeline.

        This uses UFold's Dataset_FCN to properly create the 17-channel input.
        """
        import collections
        RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

        # Create a temporary input file in UFold's format
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="ufold_"))
        input_file = temp_dir / "input.txt"
        with open(input_file, "w") as f:
            for i, seq in enumerate(sequences):
                seq = seq.upper().replace('T', 'U')
                f.write(f">seq_{i}\n{seq}\n")

        # Load using UFold's data generator
        from ufold.data_generator import RNASSDataGenerator_input, Dataset_Cut_concat_new as Dataset_FCN
        from torch.utils import data as torch_data

        # Create a temporary data directory
        ufold_data_dir = temp_dir
        input_txt = ufold_data_dir / "input.txt"
        with open(input_txt, "w") as f:
            for i, seq in enumerate(sequences):
                seq = seq.upper().replace('T', 'U')
                f.write(f">seq_{i}\n{seq}\n")

        test_data = RNASSDataGenerator_input(str(ufold_data_dir) + '/', 'input')
        test_set = Dataset_FCN(test_data)
        params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
        test_generator = torch_data.DataLoader(test_set, **params)

        results = []
        seq_idx = 0
        for seq_embeddings, seq_lens, seq_ori, seq_name in test_generator:
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(self.device)
            seq_ori_tensor = torch.Tensor(seq_ori.float()).to(self.device)

            pred_contacts = self.model(seq_embedding_batch)
            u_no_train = postprocess(
                pred_contacts, seq_ori_tensor,
                0.01, 0.1, 100, 1.6, True, 1.5
            )
            map_no_train = (u_no_train > 0.5).float()

            # Extract dot-bracket
            pred_cpu = map_no_train.cpu()
            seq_tmp = torch.mul(
                pred_cpu.argmax(axis=1),
                pred_cpu.sum(axis=1).clamp_max(1)
            ).numpy().astype(int)
            seq_tmp[pred_cpu.sum(axis=1) == 0] = -1
            dot_list = seq2dot((seq_tmp[0] + 1))

            # Get actual sequence length
            actual_len = int(seq_lens[0])
            results.append(dot_list[:actual_len])
            seq_idx += 1

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return results


def run_tier(
    predictor: UFoldPredictor,
    gold_path: Path,
    output_path: Path,
    max_samples: int = 0,
    batch_size: int = 1,
    start_index: int = 0,
    append: bool = False,
) -> Dict[str, Any]:
    """Run UFold on one tier and write predictions.

    Creates one temp file with all sequences and loads the data generator once
    to avoid per-sequence overhead.

    Args:
        start_index: Skip first N records (for chunked processing).
        max_samples: Process at most N records after start_index.
        append: If True, append to output file instead of overwriting.
    """
    import collections
    import tempfile
    import shutil
    from torch.utils import data as torch_data
    from ufold.data_generator import RNASSDataGenerator_input, Dataset_Cut_concat_new as Dataset_FCN

    records = []
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Apply start_index and max_samples for chunked processing
    total_in_tier = len(records)
    records = records[start_index:]
    if max_samples > 0:
        records = records[:max_samples]

    total = len(records)
    if total == 0:
        print(f"[INFO] No records to process (start_index={start_index})", file=sys.stderr)
        return {
            "tier": gold_path.stem,
            "count": 0,
            "elapsed_seconds": 0,
            "output": str(output_path),
            "start_index": start_index,
        }

    t0 = time.time()

    # Create one temp file with ALL sequences for this tier
    temp_dir = Path(tempfile.mkdtemp(prefix="ufold_tier_"))
    try:
        input_txt = temp_dir / "input.txt"
        with open(input_txt, "w") as f:
            for i, rec in enumerate(records):
                seq = rec["sequence"].upper().replace('T', 'U')
                f.write(f">seq_{i}\n{seq}\n")

        # Load data generator once
        test_data = RNASSDataGenerator_input(str(temp_dir) + '/', 'input')
        test_set = Dataset_FCN(test_data)
        params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
        test_generator = torch_data.DataLoader(test_set, **params)

        count = 0
        out_mode = "a" if append else "w"
        with open(output_path, out_mode, encoding="utf-8") as out:
            for seq_embeddings, seq_lens, seq_ori, seq_name in test_generator:
                try:
                    # Data is on CPU (creatmat monkey-patched to CPU)
                    # Move only model input to GPU for inference
                    seq_embedding_batch = torch.Tensor(seq_embeddings.float())
                    seq_ori_tensor = torch.Tensor(seq_ori.float())

                    if predictor.device.type == "cuda":
                        seq_embedding_batch = seq_embedding_batch.to(predictor.device)
                        # Use FP16 to reduce GPU memory
                        seq_embedding_batch = seq_embedding_batch.half()
                        with torch.amp.autocast('cuda', dtype=torch.float16):
                            pred_contacts = predictor.model(seq_embedding_batch)
                        pred_contacts = pred_contacts.float()
                    else:
                        pred_contacts = predictor.model(seq_embedding_batch)

                    # Move output back to CPU for postprocessing
                    pred_contacts_cpu = pred_contacts.cpu()
                    seq_ori_cpu = seq_ori_tensor.cpu()

                    # Free GPU memory
                    del seq_embedding_batch, pred_contacts
                    if predictor.device.type == "cuda":
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()

                    u_no_train = postprocess(
                        pred_contacts_cpu, seq_ori_cpu,
                        0.01, 0.1, 100, 1.6, True, 1.5
                    )
                    map_no_train = (u_no_train > 0.5).float()

                    # Extract dot-bracket
                    pred_cpu = map_no_train.cpu()
                    seq_tmp = torch.mul(
                        pred_cpu.argmax(axis=1),
                        pred_cpu.sum(axis=1).clamp_max(1)
                    ).numpy().astype(int)
                    seq_tmp[pred_cpu.sum(axis=1) == 0] = -1
                    dot_list = seq2dot((seq_tmp[0] + 1))

                    actual_len = int(seq_lens[0])
                    db = dot_list[:actual_len]

                    # Free CPU tensors
                    del pred_contacts_cpu, seq_ori_cpu, u_no_train, map_no_train, pred_cpu
                except Exception as e:
                    if count < len(records):
                        db = "." * len(records[count]["sequence"])
                    else:
                        db = ""
                    print(f"[WARN] UFold predict failed on seq {count}: {e}", file=sys.stderr)
                    # Clear GPU cache on error
                    if predictor.device.type == "cuda":
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()

                if count < len(records):
                    rec = records[count]
                    seq = rec["sequence"]
                    sid = rec.get("source_id", "")
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
                if count % 500 == 0:
                    elapsed = time.time() - t0
                    rate = count / elapsed if elapsed > 0 else 0
                    eta = (total - count) / rate if rate > 0 else 0
                    actual_idx = start_index + count
                    print(f"  [{actual_idx}/{total_in_tier}] (chunk {count}/{total}) {elapsed:.1f}s, {rate:.1f} seq/s, ETA {eta:.0f}s", file=sys.stderr, flush=True)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    return {
        "tier": gold_path.stem,
        "count": count,
        "start_index": start_index,
        "total_in_tier": total_in_tier,
        "elapsed_seconds": round(elapsed, 2),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UFold baseline")
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
                        help="Max samples per tier (0 = all, after start-index)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip first N records (for chunked processing)")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (UFold processes one at a time)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, or cpu")
    args = parser.parse_args()

    # Change to UFold root directory (required for relative paths)
    os.chdir(str(UFOLD_ROOT))

    # Create results directories (UFold expects them)
    os.makedirs("results/save_ct_file", exist_ok=True)
    os.makedirs("results/save_varna_fig", exist_ok=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] UFold baseline", file=sys.stderr)
    print(f"[INFO] Model: {args.model_path}", file=sys.stderr)

    predictor = UFoldPredictor(args.model_path, args.device)

    summaries = []
    for tier in args.tiers:
        gold_path = args.gold_dir / f"{tier}.jsonl"
        if not gold_path.exists():
            print(f"[WARN] Gold file not found: {gold_path}", file=sys.stderr)
            continue
        tier_name = TIER_MAP.get(tier, tier)
        output_path = args.output_dir / f"{tier_name}.ufold.predictions.jsonl"
        print(f"[INFO] Processing tier={tier} -> {tier_name} (start_index={args.start_index}, append={args.append})", file=sys.stderr)
        summary = run_tier(predictor, gold_path, output_path, args.max_samples, args.batch_size,
                           start_index=args.start_index, append=args.append)
        summaries.append(summary)
        print(f"[INFO] Done: {summary}", file=sys.stderr)

    summary_path = args.output_dir / "ufold_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model": "UFold", "summaries": summaries}, f, indent=2)
    print(f"[INFO] Summary written to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
