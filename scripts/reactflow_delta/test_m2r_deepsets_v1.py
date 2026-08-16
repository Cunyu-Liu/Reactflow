#!/usr/bin/env python3
"""Tests for m2r_deepsets_v1 (full-profile attention model).

Runs CPU-only on a tiny synthetic batch: checks forward shape, zero-init
head output near zero, dataset builder shape/mask validity, and the fixed
clipping bounds in build_dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_deepsets_v1 as md

M2R_CSV = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"


def test_forward_shape_and_zero_init():
    torch.manual_seed(0)
    model = md.FullProfileAttention(8, hidden=16, nhead=4, nlayers=1,
                                    dropout=0.0, head_hidden=8, seed=0)
    model._pe = md._pos_encoding(10, 16)
    x = torch.randn(4, 10, 8)
    m = torch.ones(4, 10, dtype=torch.bool)
    with torch.no_grad():
        y = model(x, m)
    assert y.shape == (4,)
    assert torch.all(torch.abs(y) < 1e-6), "zero-init head should output ~0"


def test_forward_masked():
    torch.manual_seed(0)
    model = md.FullProfileAttention(8, hidden=16, nhead=4, nlayers=1,
                                    dropout=0.0, head_hidden=8, seed=0)
    model._pe = md._pos_encoding(10, 16)
    x = torch.randn(4, 10, 8)
    m = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * 4, dtype=torch.bool)
    with torch.no_grad():
        y = model(x, m)
    assert y.shape == (4,)
    assert torch.all(torch.isfinite(y))


def test_build_dataset_shapes_and_clip():
    designs, meta = m2r.parse_m2r_csv(M2R_CSV)
    samples = [s for s in m2r.build_all_pair_samples(designs)
               if s.rescue_factor is not None]
    X, M, y, keys = md.build_dataset(samples, 400)
    assert X.shape[0] == len(samples)
    assert X.shape[2] == 8
    assert M.shape == (len(samples), X.shape[1])
    assert y.shape[0] == len(samples)
    assert len(keys) == len(samples)
    # fixed clip bounds must hold for react/err channels
    assert X[:, :, 0].max() <= 10.0 and X[:, :, 0].min() >= -3.0
    assert X[:, :, 1].min() >= 0.0 and X[:, :, 1].max() <= 10.0
    # every row must have at least one valid position
    assert M.sum(dim=1).min() > 0


def test_pos_encoding_cache():
    pe = md._pos_encoding(21, 32)
    assert pe.shape == (21, 32)
    assert torch.all(torch.isfinite(pe))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
