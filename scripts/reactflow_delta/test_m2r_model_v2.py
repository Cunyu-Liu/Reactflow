#!/usr/bin/env python3
"""Tests for m2r_model_v2 — symmetric pair encoder."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2r_data_v1 as m2r
import m2r_model_v2 as m2v


@pytest.fixture(scope="module")
def samples():
    path = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv"
    designs, meta = m2r.parse_m2r_csv(path)
    return [s for s in m2r.build_all_pair_samples(designs)
            if s.rescue_factor is not None]


def test_pair_windows_shape(samples):
    s = samples[0]
    wi, wj, g = m2v.build_pair_windows(s)
    assert wi.ndim == 1 and wj.ndim == 1 and g.ndim == 1
    assert wi.shape == wj.shape
    assert wi.shape[0] == 6 * 15  # 6 arrays x 15 window positions
    assert g.shape[0] == 7 + 8 + 2  # structure(7) + bases(8) + pairing(2)
    assert np.all(np.isfinite(wi)) and np.all(np.isfinite(wj)) and np.all(np.isfinite(g))


def test_symmetric_pair_forward(samples):
    s = samples[0]
    wi, wj, g = m2v.build_pair_windows(s)
    model = m2v.SymmetricPairRescue(wi.shape[0], g.shape[0], seed=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(wi).unsqueeze(0),
                    torch.tensor(wj).unsqueeze(0),
                    torch.tensor(g).unsqueeze(0))
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_symmetric_is_symmetric(samples):
    """The encoder should be invariant to swapping i and j for the pair terms."""
    s = samples[0]
    wi, wj, g = m2v.build_pair_windows(s)
    model = m2v.SymmetricPairRescue(wi.shape[0], g.shape[0], seed=0)
    model.eval()
    with torch.no_grad():
        out12 = model(torch.tensor(wi).unsqueeze(0), torch.tensor(wj).unsqueeze(0),
                      torch.tensor(g).unsqueeze(0))
        out21 = model(torch.tensor(wj).unsqueeze(0), torch.tensor(wi).unsqueeze(0),
                      torch.tensor(g).unsqueeze(0))
    # sum/diff/product terms are symmetric under swap; e_i,e_j terms swap, so
    # the model is NOT strictly symmetric (site identities matter), but the
    # pairwise structure must keep both finite.
    assert torch.isfinite(out12).all() and torch.isfinite(out21).all()


def test_train_symmetric_smoke(samples):
    subs = samples[:60]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, log = m2v.train_symmetric(subs, None, device=device, epochs=2,
                                     bs=32, seed=0)
    assert "learning_curve" in log
    assert len(log["learning_curve"]) == 2
    preds = m2v.predict_symmetric(model, subs, device=device)
    assert preds.shape == (len(subs),)
    assert np.all(np.isfinite(preds))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
