#!/usr/bin/env python3
"""Debug NaN in training - check first batch forward + backward."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import yaml
from reactflow.models.static_pairformer import StaticPairFormer, PairFormerConfig
from reactflow.losses import LossConfig, pairformer_loss
from reactflow.pilot_data import build_pilot_dataloaders

with open(ROOT / "configs/models/pairformer_compact.yaml") as f:
    config = yaml.safe_load(f)

cfg = PairFormerConfig(**config["model"])
model = StaticPairFormer(cfg).cuda()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

loss_cfg = LossConfig(**config["loss"])
loaders = build_pilot_dataloaders(
    ROOT / config["data"]["manifest_path"],
    ROOT / config["data"]["cache_root"],
    max_length=config["training"]["max_len"],
    min_length=config["data"]["min_length"],
    batch_size=2,
    eval_batch_size=2,
    num_workers=0,
    max_per_split=10,
    seed=0,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

for b_idx, (indices, targets, masks) in enumerate(loaders["train"]):
    print(f"\n=== batch {b_idx} ===")
    print(f"  indices: {indices.shape} dtype={indices.dtype}")
    print(f"  targets: {targets.shape} sum={targets.sum().item()}")
    print(f"  masks: {masks.shape} sum={masks.sum().item()}")
    indices = indices.cuda()
    targets = targets.cuda()
    masks = masks.cuda()

    optimizer.zero_grad()
    output = model(indices, mask=masks)
    print(f"  logits: has_nan={output.logits.isnan().any().item()} has_inf={output.logits.isinf().any().item()}")
    print(f"  logits range: [{output.logits[torch.isfinite(output.logits)].min().item():.4f}, {output.logits[torch.isfinite(output.logits)].max().item():.4f}]")
    print(f"  bpp: has_nan={output.bpp.isnan().any().item()} range=[{output.bpp.min().item():.4f}, {output.bpp.max().item():.4f}]")

    parts = pairformer_loss(output, targets, config=loss_cfg)
    for k, v in parts.items():
        print(f"  loss[{k}]: {v.item():.6f} has_nan={v.isnan().item()}")

    if parts["total"].isnan():
        print("  NaN detected! Stopping.")
        break

    parts["total"].backward()
    grad_max = 0.0
    grad_nan = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if p.grad.isnan().any():
                print(f"  GRAD NaN in {name}")
                grad_nan = True
            else:
                grad_max = max(grad_max, p.grad.abs().max().item())
    print(f"  grad_max={grad_max:.6f} grad_nan={grad_nan}")
    optimizer.step()
    if b_idx >= 3:
        break
