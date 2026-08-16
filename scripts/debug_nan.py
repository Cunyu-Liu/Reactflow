#!/usr/bin/env python3
"""Debug NaN in StaticPairFormer forward pass."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from reactflow.models.static_pairformer import StaticPairFormer, PairFormerConfig
from reactflow.losses import LossConfig, pairformer_loss

cfg = PairFormerConfig(single_dim=256, pair_dim=64, max_len=128, num_blocks=2, outer_product_mean_dim=16)
model = StaticPairFormer(cfg).cuda()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

B, L = 2, 16
indices = torch.randint(0, 4, (B, L)).cuda()
mask = torch.ones(B, L, dtype=torch.bool).cuda()
targets = torch.zeros(B, L, L).cuda()
targets[0, 0, 1] = targets[0, 1, 0] = 1.0
targets[0, 2, 3] = targets[0, 3, 2] = 1.0

with torch.no_grad():
    out = model(indices, mask=mask)
    print(f"logits: shape={out.logits.shape} has_nan={out.logits.isnan().any().item()} has_inf={out.logits.isinf().any().item()}")
    print(f"logits range: [{out.logits.min().item():.4f}, {out.logits.max().item():.4f}]")
    print(f"bpp range: [{out.bpp.min().item():.4f}, {out.bpp.max().item():.4f}]")
    print(f"temperature: {out.temperature.item():.4f}")

    loss_cfg = LossConfig()
    parts = pairformer_loss(out, targets, config=loss_cfg)
    for k, v in parts.items():
        print(f"  {k}: {v.item():.6f}")
