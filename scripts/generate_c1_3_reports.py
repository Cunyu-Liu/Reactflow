#!/usr/bin/env python3
"""Generate calibration, legality, and runtime reports for C1-3 gate.

Spec requirement (line 591):
    calibration、legality 和 runtime 有完整报告

This script:
1. Loads a trained checkpoint
2. Computes calibration metrics (ECE, Brier score, reliability diagram)
3. Extracts legality rate from evaluation results
4. Measures runtime and peak memory
5. Saves three JSON reports:
   - artifacts/c1_3/calibration_report.json
   - artifacts/c1_3/legality_report.json
   - artifacts/c1_3/runtime_report.json

Usage::

    PYTHONPATH=src python scripts/generate_c1_3_reports.py \
        --config configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_ddp.yaml \
        --checkpoint artifacts/c1_3/runs/.../best.pt \
        --device cuda:6 \
        --eval-results artifacts/c1_3/eval_results/evaluation_results.json \
        --output-dir artifacts/c1_3 \
        --max-samples 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from reactflow.backbones.embeddings import PAD_INDEX, encode_sequence
from reactflow.models.static_pairformer import PairFormerConfig, StaticPairFormer

sys.path.insert(0, str(Path(__file__).parent))
from train_c1_3 import (
    DataRecord, FrozenFeatureStore, C1_3Dataset, collate_fn,
    BatchPairFormer, _filter_fields, _load_jsonl, _source_from_id,
    _build_records, _indices_to_seq,
)

try:
    from reactflow.constraints import validate_pair_matrix
    _HAS_CONSTRAINTS = True
except ImportError:
    _HAS_CONSTRAINTS = False
    validate_pair_matrix = None


def compute_ece(pred_probs: np.ndarray, true_labels: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    """Expected Calibration Error.

    ECE = sum over bins of |accuracy - confidence| * (bin_count / total)
    """
    if len(pred_probs) == 0:
        return {"ece": 0.0, "bins": [], "n_samples": 0}

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bins_info = []
    total = len(pred_probs)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (pred_probs >= bin_lower) & (pred_probs < bin_upper)
        bin_count = in_bin.sum()
        if bin_count > 0:
            bin_acc = true_labels[in_bin].mean()
            bin_conf = pred_probs[in_bin].mean()
            ece += abs(bin_acc - bin_conf) * (bin_count / total)
            bins_info.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": int(bin_count),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "gap": float(abs(bin_acc - bin_conf)),
            })
        else:
            bins_info.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "gap": 0.0,
            })

    return {"ece": float(ece), "bins": bins_info, "n_samples": total}


def compute_brier_score(pred_probs: np.ndarray, true_labels: np.ndarray) -> float:
    """Brier score = mean((prob - label)^2)."""
    if len(pred_probs) == 0:
        return 0.0
    return float(np.mean((pred_probs - true_labels) ** 2))


def compute_calibration(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 1000,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Compute calibration metrics from BPP predictions."""
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    n_seen = 0

    with torch.no_grad():
        for batch in loader:
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(bd)
            bpp = out.bpp.cpu().numpy()  # (B, L, L)
            targets = batch["targets"].cpu().numpy()  # (B, L, L)
            mask = out.mask.cpu().numpy() if out.mask is not None else None

            for i in range(bpp.shape[0]):
                L = batch["lengths"][i]
                # Extract upper triangle (i < j)
                for ii in range(L):
                    for jj in range(ii + 1, L):
                        prob = float(bpp[i, ii, jj])
                        label = int(targets[i, ii, jj])
                        all_probs.append(prob)
                        all_labels.append(label)
                n_seen += 1
                if max_samples > 0 and n_seen >= max_samples:
                    break
            if max_samples > 0 and n_seen >= max_samples:
                break

    probs = np.array(all_probs, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)

    # Subsample if too many points (for memory)
    if len(probs) > 500000:
        idx = np.random.choice(len(probs), 500000, replace=False)
        probs = probs[idx]
        labels = labels[idx]

    ece_result = compute_ece(probs, labels, n_bins)
    brier = compute_brier_score(probs, labels)

    # Mean confidence and accuracy
    mean_conf = float(probs.mean()) if len(probs) > 0 else 0.0
    mean_acc = float(labels.mean()) if len(labels) > 0 else 0.0

    return {
        "ece": ece_result["ece"],
        "brier_score": brier,
        "mean_confidence": mean_conf,
        "mean_accuracy": mean_acc,
        "confidence_gap": float(abs(mean_conf - mean_acc)),
        "reliability_diagram": ece_result["bins"],
        "n_pairs": int(len(probs)),
        "n_sequences": n_seen,
        "n_bins": n_bins,
    }


def check_legality(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    decoder_cfg: Any,
    max_samples: int = 500,
) -> Dict[str, Any]:
    """Check legality of predicted structures."""
    from reactflow.decoders import decode, DecoderConfig

    model.eval()
    total = 0
    legal = 0
    illegal_examples: List[Dict] = []
    n_seen = 0

    with torch.no_grad():
        for batch in loader:
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(bd)
            decoded = decode(
                out.logits, indices=batch["indices"].to(device),
                mask=out.mask, bpp=out.bpp, unpaired_prob=out.unpaired_prob,
                temperature=out.temperature, config=decoder_cfg, mode="mea",
            )

            for i in range(len(batch["lengths"])):
                L = batch["lengths"][i]
                pred = decoded[i, :L, :L].cpu().numpy()
                total += 1

                if _HAS_CONSTRAINTS and validate_pair_matrix is not None:
                    seq = _indices_to_seq(batch["indices"][i], L)
                    result = validate_pair_matrix(
                        seq, pred.tolist(),
                        min_loop=decoder_cfg.min_loop,
                        allow_wobble=decoder_cfg.allow_wobble,
                    )
                    if result.valid:
                        legal += 1
                    else:
                        if len(illegal_examples) < 10:
                            illegal_examples.append({
                                "sample_id": int(batch["indices"][i].sum().item()),
                                "length": int(L),
                                "reason": str(getattr(result, "reason", "unknown")),
                            })
                else:
                    # Basic legality: check symmetry and min_loop
                    is_legal = True
                    for ii in range(L):
                        for jj in range(ii + 1, L):
                            if pred[ii, jj] != pred[jj, ii]:
                                is_legal = False
                                break
                            if pred[ii, jj] > 0 and (jj - ii) < decoder_cfg.min_loop:
                                is_legal = False
                                break
                        if not is_legal:
                            break
                    if is_legal:
                        legal += 1

                n_seen += 1
                if max_samples > 0 and n_seen >= max_samples:
                    break
            if max_samples > 0 and n_seen >= max_samples:
                break

    return {
        "total_samples": total,
        "legal_samples": legal,
        "legal_rate": float(legal / max(total, 1)),
        "illegal_examples": illegal_examples,
        "min_loop_length": decoder_cfg.min_loop,
        "allow_wobble": decoder_cfg.allow_wobble,
    }


def measure_runtime(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> Dict[str, Any]:
    """Measure inference runtime and peak memory."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(loader):
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            _ = model(bd)
            if i >= n_warmup:
                break

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Timed runs
    times: List[float] = []
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            _ = model(bd)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1 - t0)
            total_samples += len(batch["lengths"])
            if i >= n_runs:
                break

    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    return {
        "mean_inference_time_sec": float(np.mean(times)) if times else 0.0,
        "std_inference_time_sec": float(np.std(times)) if times else 0.0,
        "min_inference_time_sec": float(np.min(times)) if times else 0.0,
        "max_inference_time_sec": float(np.max(times)) if times else 0.0,
        "mean_samples_per_batch": float(total_samples / max(len(times), 1)),
        "throughput_samples_per_sec": float(total_samples / max(sum(times), 1e-6)),
        "peak_memory_bytes": int(peak_mem),
        "peak_memory_mb": float(peak_mem / 1024 / 1024),
        "n_runs": len(times),
        "n_warmup": n_warmup,
        "device": str(device),
    }


def load_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, Any]:
    """Load model from checkpoint."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = PairFormerConfig(**config["model"])
    model = StaticPairFormer(model_cfg)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Handle FSDP checkpoint
    if any(k.startswith("_fsdp") for k in state_dict.keys()):
        # Strip FSDP wrapper keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate C1-3 calibration/legality/runtime reports")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-results", type=Path, default=None,
                        help="Existing evaluation_results.json for extracting legality/runtime")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="Directory containing test.jsonl, val.jsonl, novel.jsonl")
    parser.add_argument("--frozen-features", type=Path, required=True,
                        help="Path to frozen RibonanzaNet features")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[INFO] Loading model from {args.checkpoint}", file=sys.stderr)
    model, config = load_model(args.config, args.checkpoint, device)

    # Load data
    from reactflow.decoders import DecoderConfig
    decoder_cfg = DecoderConfig(
        min_loop=config.get("decoder", {}).get("min_loop", 3),
        allow_wobble=config.get("decoder", {}).get("allow_wobble", True),
    )

    # Build dataset
    test_path = args.gold_dir / "test.jsonl"
    records = _build_records(test_path, source="test")
    feature_store = FrozenFeatureStore(args.frozen_features)
    dataset = C1_3Dataset(records, feature_store, max_len=args.max_len, training=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # 1. Calibration report
    print("[INFO] Computing calibration metrics...", file=sys.stderr)
    t0 = time.time()
    calibration = compute_calibration(model, loader, device, max_samples=args.max_samples)
    calibration["computation_time_sec"] = time.time() - t0
    calibration["checkpoint"] = str(args.checkpoint)
    calibration["config"] = str(args.config)

    cal_path = output_dir / "calibration_report.json"
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"[INFO] Calibration report: {cal_path}", file=sys.stderr)
    print(f"  ECE={calibration['ece']:.4f}, Brier={calibration['brier_score']:.4f}", file=sys.stderr)

    # 2. Legality report
    print("[INFO] Checking legality...", file=sys.stderr)
    t0 = time.time()
    legality = check_legality(model, loader, device, decoder_cfg, max_samples=min(args.max_samples, 500))
    legality["computation_time_sec"] = time.time() - t0

    # Also extract legality from existing eval results if available
    if args.eval_results and args.eval_results.exists():
        with open(args.eval_results) as f:
            eval_data = json.load(f)
        for split_name, split_data in eval_data.items():
            if isinstance(split_data, dict):
                for mode, mode_data in split_data.items():
                    if isinstance(mode_data, dict) and "legal_rate" in mode_data:
                        legality.setdefault("eval_results", {})[f"{split_name}_{mode}"] = {
                            "legal_rate": mode_data["legal_rate"],
                            "total_samples": mode_data.get("total_samples", 0),
                        }

    leg_path = output_dir / "legality_report.json"
    with open(leg_path, "w") as f:
        json.dump(legality, f, indent=2)
    print(f"[INFO] Legality report: {leg_path}", file=sys.stderr)
    print(f"  Legal rate: {legality['legal_rate']:.4f} ({legality['legal_samples']}/{legality['total_samples']})", file=sys.stderr)

    # 3. Runtime report
    print("[INFO] Measuring runtime...", file=sys.stderr)
    t0 = time.time()
    runtime = measure_runtime(model, loader, device)
    runtime["computation_time_sec"] = time.time() - t0

    # Also extract runtime from existing eval results
    if args.eval_results and args.eval_results.exists():
        with open(args.eval_results) as f:
            eval_data = json.load(f)
        for split_name, split_data in eval_data.items():
            if isinstance(split_data, dict):
                if "_runtime_sec" in split_data:
                    runtime.setdefault("eval_runtimes", {})[split_name] = {
                        "runtime_sec": split_data["_runtime_sec"],
                        "peak_memory_bytes": split_data.get("_peak_memory_bytes", 0),
                    }

    rt_path = output_dir / "runtime_report.json"
    with open(rt_path, "w") as f:
        json.dump(runtime, f, indent=2)
    print(f"[INFO] Runtime report: {rt_path}", file=sys.stderr)
    print(f"  Mean inference: {runtime['mean_inference_time_sec']:.4f}s, Peak memory: {runtime['peak_memory_mb']:.1f}MB", file=sys.stderr)

    print("\n[DONE] All reports generated:", file=sys.stderr)
    print(f"  - {cal_path}", file=sys.stderr)
    print(f"  - {leg_path}", file=sys.stderr)
    print(f"  - {rt_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
