#!/usr/bin/env python3
"""Training script for the C1-2 static PairFormer pilot.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 396-410.

Trains one of the four matched-capacity models (compact PairFormer, bilinear
baseline, CNN baseline, UNet baseline) on the frozen C1-1 splits filtered to
``L <= max_length``, with the combined :func:`reactflow.losses.pairformer_loss`.

Usage::

    python scripts/train_static_pairformer.py \\
        --config configs/models/pairformer_compact.yaml \\
        --output artifacts/c1_2/runs/pairformer_compact_seed0 \\
        --seed 0 --device cuda:0

Outputs:
- ``checkpoint_best.pt``: best (lowest val loss) checkpoint
- ``checkpoint_last.pt``: final checkpoint
- ``training_log.jsonl``: per-epoch metrics
- ``config_snapshot.yaml``: copy of the resolved config
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reactflow.models.static_pairformer import StaticPairFormer, PairFormerConfig
from reactflow.models.bilinear_pair_head import BilinearPairHead, BilinearPairHeadConfig
from reactflow.models.cnn_pair_head import CNNPairHead, CNNPairHeadConfig, UNetPairHead, UNetPairHeadConfig
from reactflow.losses import LossConfig, pairformer_loss
from reactflow.pilot_data import build_pilot_dataloaders


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_type = config["model_type"]
    model_cfg = config["model"]
    if model_type == "static_pairformer":
        cfg = PairFormerConfig(**model_cfg)
        return StaticPairFormer(cfg)
    if model_type == "bilinear_pair_head":
        cfg = BilinearPairHeadConfig(**model_cfg)
        return BilinearPairHead(cfg)
    if model_type == "cnn_pair_head":
        cfg = CNNPairHeadConfig(**model_cfg)
        return CNNPairHead(cfg)
    if model_type == "unet_pair_head":
        cfg = UNetPairHeadConfig(**model_cfg)
        return UNetPairHead(cfg)
    raise ValueError(f"unknown model_type: {model_type!r}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def evaluate_model(
    model: torch.nn.Module,
    loader,
    loss_cfg: LossConfig,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate the model on a DataLoader."""
    model.eval()
    total_loss = 0.0
    total_parts: Dict[str, float] = {}
    n_batches = 0
    with torch.no_grad():
        for b_idx, (indices, targets, masks) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            indices = indices.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            output = model(indices, mask=masks)
            parts = pairformer_loss(output, targets, config=loss_cfg)
            total_loss += parts["total"].item()
            for k, v in parts.items():
                if k == "total":
                    continue
                total_parts[k] = total_parts.get(k, 0.0) + v.item()
            n_batches += 1
    if n_batches == 0:
        return {"loss": 0.0}
    result = {"loss": total_loss / n_batches}
    for k, v in total_parts.items():
        result[f"loss_{k}"] = v / n_batches
    return result


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_cfg: LossConfig,
    device: torch.device,
    *,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    log_every: int = 20,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    amp_dtype = torch.float16 if use_amp else torch.float32

    for indices, targets, masks in loader:
        indices = indices.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                output = model(indices, mask=masks)
                parts = pairformer_loss(output, targets, config=loss_cfg)
                loss = parts["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(indices, mask=masks)
            parts = pairformer_loss(output, targets, config=loss_cfg)
            loss = parts["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        if n_batches % log_every == 0:
            print(
                f"    step {n_batches}: loss={loss.item():.4f}",
                file=sys.stderr,
            )

    return {"loss": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Train C1-2 static PairFormer pilot")
    parser.add_argument("--config", type=Path, required=True, help="Path to model YAML config")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--max-per-split", type=int, default=None, help="Cap records per split")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Override manifest path")
    parser.add_argument("--cache-root", type=Path, default=None, help="Override cache root")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    if args.manifest_path is not None:
        config["data"]["manifest_path"] = str(args.manifest_path)
    if args.cache_root is not None:
        config["data"]["cache_root"] = str(args.cache_root)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cuda:0")
    print(f"[C1-2 train] device={device} seed={seed}", file=sys.stderr)

    # Output dir
    args.output.mkdir(parents=True, exist_ok=True)
    # Save config snapshot
    shutil.copy(args.config, args.output / "config_snapshot.yaml")

    # Build model
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[C1-2 train] model: {config['model_type']}, params={n_params:,}", file=sys.stderr)

    # Build dataloaders
    data_cfg = config["data"]
    manifest_path = Path(data_cfg["manifest_path"])
    cache_root = Path(data_cfg["cache_root"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root

    print(f"[C1-2 train] loading data from {manifest_path}", file=sys.stderr)
    train_cfg = config["training"]
    loaders = build_pilot_dataloaders(
        manifest_path, cache_root,
        max_length=train_cfg["max_len"],
        min_length=data_cfg["min_length"],
        batch_size=train_cfg["batch_size"],
        eval_batch_size=train_cfg["eval_batch_size"],
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        max_per_split=args.max_per_split,
        seed=seed,
    )

    if "train" not in loaders:
        print("ERROR: no train loader", file=sys.stderr)
        return 1

    # Loss config
    loss_cfg = LossConfig(**config["loss"])

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = epochs * len(loaders["train"])
    warmup_steps = train_cfg["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if train_cfg["use_amp"] and device.type == "cuda" else None

    # Training loop
    log_path = args.output / "training_log.jsonl"
    best_val_loss = float("inf")
    start_time = time.time()

    with open(log_path, "w", encoding="utf-8") as log_f:
        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"[C1-2 train] epoch {epoch + 1}/{epochs}", file=sys.stderr)

            train_metrics = train_one_epoch(
                model, loaders["train"], optimizer, scheduler, loss_cfg, device,
                max_grad_norm=train_cfg["max_grad_norm"],
                use_amp=train_cfg["use_amp"],
                log_every=train_cfg["log_every"],
                scaler=scaler,
            )
            val_metrics = evaluate_model(
                model, loaders.get("val"), loss_cfg, device,
            ) if "val" in loaders else {"loss": 0.0}

            epoch_time = time.time() - epoch_start
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "epoch_time_sec": epoch_time,
                "lr": optimizer.param_groups[0]["lr"],
            }
            metrics.update({f"val_{k}": v for k, v in val_metrics.items() if k != "loss"})
            log_f.write(json.dumps(metrics) + "\n")
            log_f.flush()
            print(
                f"  epoch {epoch + 1}: train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} time={epoch_time:.1f}s",
                file=sys.stderr,
            )

            # Save best
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "config": config,
                    "n_params": n_params,
                }, args.output / "checkpoint_best.pt")
                print(f"  saved best checkpoint (val_loss={val_metrics['loss']:.4f})", file=sys.stderr)

            # Save last every epoch
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "config": config,
                "n_params": n_params,
            }, args.output / "checkpoint_last.pt")

    total_time = time.time() - start_time
    summary = {
        "model_type": config["model_type"],
        "config_name": config["config_name"],
        "seed": seed,
        "n_params": n_params,
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "total_time_sec": total_time,
        "device": str(device),
    }
    with open(args.output / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[C1-2 train] done. best_val_loss={best_val_loss:.4f} time={total_time:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
