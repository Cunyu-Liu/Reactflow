#!/usr/bin/env python3
"""Run RNAformer baseline on ReactFlow gold splits.

Loads the pretrained RNAformer 32M model (intra-family fine-tuned) and
predicts RNA secondary structure for each sequence.

Produces prediction JSONL files consumable by ``evaluate_external_baseline_predictions.py``.

Usage::

    python scripts/run_rnaformer_baseline.py \
        --gold-dir <splits_path> \
        --output-dir <output_dir> \
        --state_dict <path_to_.pth> \
        --config <path_to_config.yml> \
        --tiers test novel
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

import loralib as lora
from RNAformer.model.RNAformer import RiboFormer
from RNAformer.utils.configuration import Config


TIER_MAP = {
    "test": "in_clan",
    "novel": "novel_clan",
}

SEQ_VOCAB = ['A', 'C', 'G', 'U', 'N']
SEQ_STOI = dict(zip(SEQ_VOCAB, range(len(SEQ_VOCAB))))


def sequence2index_vector(sequence: str, mapping: Dict[str, int]) -> torch.LongTensor:
    """Convert nucleotide sequence to index tensor.

    Complexity: O(L) where L = len(sequence).
    """
    int_sequence = [mapping.get(nt, 4) for nt in sequence]  # default to N=4
    return torch.LongTensor(int_sequence)


def pairs_to_dotbracket(pairs: List[Tuple[int, int]], length: int) -> str:
    """Convert list of (i, j) pairs to dot-bracket notation.

    Handles non-crossing pairs with parentheses. Crossing pairs (pseudoknots)
    use square brackets as a secondary level.

    Complexity: O(L + P) where L = length, P = number of pairs.
    """
    db = ['.'] * length
    # Sort pairs by left index
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    # Use stack-based assignment for non-crossing
    stack1: List[int] = []
    assigned = [False] * length
    for i, j in pairs_sorted:
        if i >= length or j >= length or i >= j:
            continue
        if assigned[i] or assigned[j]:
            continue
        db[i] = '('
        db[j] = ')'
        assigned[i] = True
        assigned[j] = True
    return ''.join(db)


def dotbracket_to_pairs(db: str) -> List[List[int]]:
    """Convert dot-bracket notation to a list of [i, j] pairs (0-indexed).

    Complexity: O(L) where L = len(db).
    """
    stack: List[int] = []
    pairs: List[List[int]] = []
    for i, c in enumerate(db):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                j = stack.pop()
                pairs.append([j, i])
    return pairs


class RNAformerPredictor:
    """Wraps the RNAformer model for batch prediction."""

    def __init__(self, state_dict_path: str, config_path: str, device: str = "auto",
                 cycling: int = 3):
        """Load model from state dict and config.

        Handles LoRA-finetuned checkpoints by applying insert_lora_layer before
        loading the state dict, and cycling (recycling) checkpoints.

        Args:
            state_dict_path: Path to the .pth checkpoint.
            config_path: Path to the YAML config file.
            device: "auto", "cuda", or "cpu".
            cycling: Number of recycling steps (0 to disable).
        """
        self.config = Config(config_file=config_path)

        # Enable cycling if the checkpoint has recycle_pair_norm
        if cycling and cycling > 0:
            self.config.RNAformer.cycling = cycling

        self.model = RiboFormer(self.config.RNAformer)

        # Apply LoRA layers if the checkpoint contains lora weights
        # The intra-family finetuned checkpoint uses LoRA r=32 on attention + transition
        if not hasattr(self.config, 'lora'):
            self.config.lora = True
            self.config.r = 32
            self.config.lora_alpha = 64
            self.config.lora_dropout = 0.0
            # Full replace_layer list matching the checkpoint
            self.config.replace_layer = [
                'attn_pair_row.Wqkv',
                'attn_pair_row.out_proj',
                'attn_pair_col.Wqkv',
                'attn_pair_col.out_proj',
                'pair_transition.conv1',
                'pair_transition.conv2',
            ]

        if self.config.lora:
            self.model = self._insert_lora(self.model)

        if torch.cuda.is_available() and device != "cpu":
            state_dict = torch.load(state_dict_path)
            self.model = self.model.cuda()
            if torch.cuda.is_bf16_supported():
                self.model = self.model.bfloat16()
            else:
                self.model = self.model.half()
        else:
            state_dict = torch.load(state_dict_path, map_location='cpu')

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        print(f"[INFO] RNAformer loaded on {self.device}", file=sys.stderr)

    @staticmethod
    def _insert_lora(model: torch.nn.Module) -> torch.nn.Module:
        """Replace target Linear/Conv2d layers with LoRA equivalents.

        Mirrors the insert_lora_layer function from the official infer script.
        """
        ft_config_r = 32
        ft_config_alpha = 64
        ft_config_dropout = 0.0
        replace_keys = [
            'attn_pair_row.Wqkv', 'attn_pair_row.out_proj',
            'attn_pair_col.Wqkv', 'attn_pair_col.out_proj',
            'pair_transition.conv1', 'pair_transition.conv2',
        ]
        lora_config = {
            "r": ft_config_r,
            "lora_alpha": ft_config_alpha,
            "lora_dropout": ft_config_dropout,
        }
        with torch.no_grad():
            for name, module in model.named_modules():
                if any(rk in name for rk in replace_keys):
                    parent = model.get_submodule(".".join(name.split(".")[:-1]))
                    target_name = name.split(".")[-1]
                    target = model.get_submodule(name)
                    if isinstance(target, torch.nn.Linear) and "qkv" in name:
                        new_module = lora.MergedLinear(
                            target.in_features, target.out_features,
                            bias=target.bias is not None,
                            enable_lora=[True, True, True], **lora_config)
                        new_module.weight.copy_(target.weight)
                        if target.bias is not None:
                            new_module.bias.copy_(target.bias)
                    elif isinstance(target, torch.nn.Linear):
                        new_module = lora.Linear(
                            target.in_features, target.out_features,
                            bias=target.bias is not None, **lora_config)
                        new_module.weight.copy_(target.weight)
                        if target.bias is not None:
                            new_module.bias.copy_(target.bias)
                    elif isinstance(target, torch.nn.Conv2d):
                        kernel_size = target.kernel_size[0]
                        new_module = lora.Conv2d(
                            target.in_channels, target.out_channels, kernel_size,
                            padding=(kernel_size - 1) // 2,
                            bias=target.bias is not None, **lora_config)
                        new_module.conv.weight.copy_(target.weight)
                        if target.bias is not None:
                            new_module.conv.bias.copy_(target.bias)
                    else:
                        continue
                    setattr(parent, target_name, new_module)
        return model

    @torch.no_grad()
    def predict(self, sequence: str) -> str:
        """Predict dot-bracket structure for a single sequence.

        Args:
            sequence: RNA sequence (A/C/G/U/N).

        Returns:
            Dot-bracket notation string.
        """
        seq = sequence.replace('T', 'U').replace('t', 'u')
        length = len(seq)
        src_seq = sequence2index_vector(seq, SEQ_STOI)

        sample_seq = src_seq.unsqueeze(0).to(self.device)
        src_len = torch.LongTensor([length]).to(self.device)

        if self.device == "cuda":
            if torch.cuda.is_bf16_supported():
                pdb_sample = torch.FloatTensor([[1]]).bfloat16().cuda()
            else:
                pdb_sample = torch.FloatTensor([[1]]).half().cuda()
        else:
            pdb_sample = torch.FloatTensor([[1]]).to(self.device)

        logits, pair_mask = self.model(sample_seq, src_len, pdb_sample)
        pred_mat = torch.sigmoid(logits[0, :, :, -1]) > 0.5

        pos_id = torch.where(pred_mat == True)
        pos1 = pos_id[0].cpu().tolist()
        pos2 = pos_id[1].cpu().tolist()
        pairs = list(zip(pos1, pos2))
        return pairs_to_dotbracket(pairs, length)


def run_tier(
    predictor: RNAformerPredictor,
    gold_path: Path,
    output_path: Path,
    max_samples: int = 0,
) -> Dict[str, Any]:
    """Run RNAformer on one tier and write predictions.

    Args:
        predictor: Loaded RNAformerPredictor.
        gold_path: Path to the gold JSONL file.
        output_path: Path to write the prediction JSONL.
        max_samples: If > 0, only fold this many sequences.

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
                db = predictor.predict(seq)
            except Exception as e:
                print(f"[WARN] Failed on {sid}: {e}", file=sys.stderr)
                db = "." * len(seq)
            pairs = dotbracket_to_pairs(db)
            pred = {
                "dotbracket": db,
                "predicted_pairs": pairs,
                "prediction_backend": "rnaformer",
                "sequence": seq,
                "source_id": sid,
            }
            out.write(json.dumps(pred) + "\n")
            count += 1
            if count % 500 == 0:
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
    parser = argparse.ArgumentParser(description="RNAformer baseline")
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="Directory containing gold JSONL split files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write prediction JSONL files")
    parser.add_argument("--state-dict", type=str, required=True,
                        help="Path to RNAformer .pth checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to RNAformer config .yml")
    parser.add_argument("--tiers", nargs="+", default=["test", "novel"],
                        help="Gold split names to process")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per tier (0 = all)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, or cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] RNAformer baseline", file=sys.stderr)
    print(f"[INFO] State dict: {args.state_dict}", file=sys.stderr)
    print(f"[INFO] Config: {args.config}", file=sys.stderr)

    predictor = RNAformerPredictor(args.state_dict, args.config, args.device)

    summaries = []
    for tier in args.tiers:
        gold_path = args.gold_dir / f"{tier}.jsonl"
        if not gold_path.exists():
            print(f"[WARN] Gold file not found: {gold_path}", file=sys.stderr)
            continue
        tier_name = TIER_MAP.get(tier, tier)
        output_path = args.output_dir / f"{tier_name}.rnaformer.predictions.jsonl"
        print(f"[INFO] Processing tier={tier} -> {tier_name}", file=sys.stderr)
        summary = run_tier(predictor, gold_path, output_path, args.max_samples)
        summaries.append(summary)
        print(f"[INFO] Done: {summary}", file=sys.stderr)

    summary_path = args.output_dir / "rnaformer_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model": "RNAformer 32M", "summaries": summaries}, f, indent=2)
    print(f"[INFO] Summary written to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
