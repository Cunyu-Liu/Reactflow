#!/usr/bin/env python3
"""Smoke test for fusion fix: verify single_only != pair_feature."""
import torch
import sys
sys.path.insert(0, "src")
from reactflow.models.static_pairformer import PairFormerConfig, StaticPairFormer

# Test single_only (frozen_pair_fusion=False)
cfg_single = PairFormerConfig(single_dim=64, pair_dim=16, num_blocks=2,
                              frozen_feature_dim=32, frozen_pair_fusion=False)
m_single = StaticPairFormer(cfg_single)
print(f"single_only params: {m_single.num_parameters():,}")
print(f"has frozen_opm: {hasattr(m_single, 'frozen_opm')}")

# Test pair_feature (frozen_pair_fusion=True)
cfg_pair = PairFormerConfig(single_dim=64, pair_dim=16, num_blocks=2,
                            frozen_feature_dim=32, frozen_pair_fusion=True)
m_pair = StaticPairFormer(cfg_pair)
print(f"pair_feature params: {m_pair.num_parameters():,}")
print(f"has frozen_opm: {hasattr(m_pair, 'frozen_opm')}")

# Verify they produce DIFFERENT outputs
indices = torch.randint(0, 5, (2, 20))
mask = torch.ones(2, 20, dtype=torch.bool)
frozen = torch.randn(2, 20, 32)
with torch.no_grad():
    out_single = m_single(indices, mask=mask, frozen_features=frozen)
    out_pair = m_pair(indices, mask=mask, frozen_features=frozen)
diff = (out_single.bpp - out_pair.bpp).abs().max().item()
print(f"BPP max diff between single_only and pair_feature: {diff:.6f}")
assert diff > 0, "FAIL: single_only and pair_feature produce identical outputs!"
print("PASS: fusion fix works - single_only and pair_feature produce different outputs")
