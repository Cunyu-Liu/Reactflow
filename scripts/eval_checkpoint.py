#!/usr/bin/env python3
"""Standalone evaluation script for C1-3 checkpoints.

Loads a checkpoint (DDP or FSDP) on a single GPU and evaluates on val/test/novel splits.

Usage::

    PYTHONPATH=src python scripts/eval_checkpoint.py \
        --config configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_ddp.yaml \
        --checkpoint artifacts/c1_3/runs/.../best.pt \
        --device cuda:0 \
        --eval-decoders threshold,mea \
        --output-dir artifacts/c1_3/eval_results
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from reactflow.backbones.embeddings import PAD_INDEX, encode_sequence
from reactflow.decoders import DecoderConfig, decode
from reactflow.models.static_pairformer import PairFormerConfig, StaticPairFormer

try:
    from reactflow.constraints import validate_pair_matrix
    _HAS_CONSTRAINTS = True
except ImportError:
    _HAS_CONSTRAINTS = False
    validate_pair_matrix = None

# Reuse from train_c1_3
sys.path.insert(0, str(Path(__file__).parent))
from train_c1_3 import (
    DataRecord, FrozenFeatureStore, C1_3Dataset, collate_fn,
    BatchPairFormer, _filter_fields, _load_jsonl, _source_from_id,
    _build_records, _indices_to_seq, _dilate, _accumulate, _finalize,
    DIST_BINS, _NPZ_CACHE,
)


def evaluate_split(model: nn.Module, loader: DataLoader, decoder_cfg: DecoderConfig,
                   device: torch.device, max_samples: int = 0,
                   eval_modes: Sequence[str] = ("threshold", "mea")) -> Dict[str, Any]:
    """Evaluate with specified decoders."""
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    cache: List[Tuple] = []
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(bd)
            cache.append((out.logits.cpu(), out.bpp.cpu(), out.unpaired_prob.cpu(),
                          out.temperature.cpu(), batch["indices"].cpu(),
                          out.mask.cpu() if out.mask is not None else None,
                          batch["targets"].cpu(), batch["lengths"]))
            n_seen += len(batch["lengths"])
            if max_samples > 0 and n_seen >= max_samples:
                break
    results: Dict[str, Any] = {}
    for mode in eval_modes:
        counts: Dict[str, int] = {
            "tp": 0, "fp": 0, "fn": 0, "tn": 0, "tp_shifted": 0, "fp_shifted": 0,
            "fn_shifted": 0, "empty_pred": 0, "total_samples": 0,
        }
        for bn in DIST_BINS:
            counts[f"{bn}_tp"] = counts[f"{bn}_fp"] = counts[f"{bn}_fn"] = 0
        legal = 0
        for (logits, bpp, unp, temp, indices, mask, targets, lengths) in cache:
            decoded = decode(logits, indices=indices, mask=mask, bpp=bpp,
                             unpaired_prob=unp, temperature=temp, config=decoder_cfg, mode=mode)
            for i in range(len(lengths)):
                L = lengths[i]
                _accumulate(decoded[i, :L, :L], targets[i, :L, :L], counts)
                if _HAS_CONSTRAINTS and validate_pair_matrix is not None:
                    seq = _indices_to_seq(indices[i], L)
                    if validate_pair_matrix(seq, decoded[i, :L, :L].tolist(),
                                            min_loop=decoder_cfg.min_loop,
                                            allow_wobble=decoder_cfg.allow_wobble).valid:
                        legal += 1
        metrics = _finalize(counts)
        if _HAS_CONSTRAINTS:
            metrics["legal_rate"] = legal / max(counts["total_samples"], 1)
        results[mode] = metrics
    results["_runtime_sec"] = time.time() - t0
    results["_peak_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate C1-3 checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-decoders", type=str, default="threshold,mea")
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}", file=sys.stderr)

    # --- Build model ---
    model_cfg_dict = _filter_fields(cfg["model"], PairFormerConfig)
    fusion_type = cfg.get("fusion", {}).get("fusion_type", "single_only")
    model_cfg_dict["frozen_pair_fusion"] = (fusion_type == "pair_feature")
    model_cfg = PairFormerConfig(**model_cfg_dict)
    base_model = StaticPairFormer(model_cfg)
    model = BatchPairFormer(base_model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {n_params}", file=sys.stderr)

    # --- Load checkpoint ---
    print(f"[INFO] Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # Handle checkpoint payload format (saved by TrainingEngine.save_checkpoint)
    # The checkpoint is a dict with keys: model_state_dict, optimizer_state_dict, etc.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("[INFO] Detected checkpoint payload format, extracting model_state_dict", file=sys.stderr)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    # Handle FSDP/DDP checkpoints: keys might have prefixes
    # Try direct load first, then try stripping prefixes
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[INFO] Checkpoint loaded (strict)", file=sys.stderr)
    except RuntimeError as e:
        print(f"[WARN] Strict load failed: {e}", file=sys.stderr)
        # Try stripping common prefixes (module., _module., etc.)
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            for prefix in ["module.", "_module.", "_fsdp_wrapped_module.",
                           "flat_param.", "_fsdp_wrapped_module.flat_param."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            new_sd[nk] = v
        try:
            model.load_state_dict(new_sd, strict=False)
            print("[INFO] Checkpoint loaded (non-strict, stripped prefixes)", file=sys.stderr)
        except Exception as e2:
            print(f"[ERROR] Failed to load checkpoint: {e2}", file=sys.stderr)
            sys.exit(1)

    model = model.to(device)
    model.eval()

    # --- Load data ---
    data_cfg = cfg.get("data", {})
    min_len = data_cfg.get("min_length", 1)
    max_len = data_cfg.get("max_length", 512)
    frozen_dim = model_cfg_dict.get("frozen_feature_dim", 0)
    frozen_path = cfg.get("frozen_features_path", data_cfg.get("frozen_features_path", ""))
    feature_store = FrozenFeatureStore(frozen_path) if frozen_path and frozen_dim > 0 else None
    if feature_store:
        print(f"[INFO] Frozen feature index: {len(feature_store)} records", file=sys.stderr)

    splits_cfg = cfg.get("splits", {})
    eval_splits: Dict[str, C1_3Dataset] = {}
    for name in ("val", "test", "novel"):
        path = splits_cfg.get(f"{name}_path", "")
        if path and Path(path).exists():
            raw = _load_jsonl(Path(path))
            records = _build_records(raw, min_len, max_len)
            eval_splits[name] = C1_3Dataset(records, feature_store, frozen_dim)
            print(f"[INFO] {name}: {len(eval_splits[name])} samples", file=sys.stderr)

    if not eval_splits:
        print("[ERROR] No evaluation splits found", file=sys.stderr)
        sys.exit(1)

    # --- Evaluate ---
    decoder_cfg = DecoderConfig(**_filter_fields(cfg.get("decoder", {}), DecoderConfig))
    eval_decoders = tuple(args.eval_decoders.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_results: Dict[str, Any] = {}
    for name, ds in eval_splits.items():
        print(f"[INFO] Evaluating on {name} ({len(ds)} samples)", file=sys.stderr)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=collate_fn)
        eval_results[name] = evaluate_split(model, loader, decoder_cfg, device,
                                            args.max_eval_samples, eval_modes=eval_decoders)
        # Print summary
        for mode in eval_decoders:
            m = eval_results[name][mode]
            print(f"  {name}/{mode}: F1={m['f1']:.4f} F1_shifted={m['f1_shifted']:.4f} "
                  f"P={m['precision']:.4f} R={m['recall']:.4f} MCC={m['mcc']:.4f}", file=sys.stderr)

    with open(args.output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"[INFO] Results saved to {args.output_dir / 'evaluation_results.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
