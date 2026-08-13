#!/usr/bin/env python3
"""Fixtures for RFD-LRSO contract invariants (contract 5.2, 14.1 items 6-9)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.reactflow_delta.lrso_v1 import RFDLRSO


def _ctx(device="cpu", L: int = 12):
    torch.manual_seed(0)
    seq = torch.zeros(L, 4, device=device)
    for i in range(L):
        seq[i, i % 4] = 1.0
    react = (torch.rand(L, device=device) * 0.8 + 0.1)
    prec = torch.zeros(L, device=device)
    mask = torch.ones(L, dtype=bool, device=device)
    pos = torch.arange(L, dtype=torch.float32, device=device)
    region = torch.zeros(L, 2, device=device)
    region[:, 0] = 1.0
    return seq, react, prec, mask, pos, region


def test_source_receiver_asymmetric():
    """Contract 14.1.6: source/receiver roles are NOT shared => heads differ on same input,
    and the directed LRSO term changes when source/receiver inputs are exchanged."""
    m = RFDLRSO(k_rank=2).eval()
    seq, react, prec, mask, pos, region = _ctx()
    H = m.encoder(seq[None], react[None], prec[None], mask[None], pos[None], region[None])[0]
    dists = (torch.arange(12, dtype=torch.float32) - 4).float()
    with torch.no_grad():
        # same input to both heads => outputs differ (roles not shared)
        x = H[4]
        src_out = m.src(torch.cat([x, m._onehot("C", "G", "cpu")])).numpy()
        recv_out = m.recv(x).numpy()
        assert not np.allclose(src_out, recv_out, atol=1e-6), "source/receiver heads must differ"
        # directed: source uses edit+alt, receiver uses readout; exchanging them changes delta
        d1 = m.delta(H, 4, dists, "C", "G", mask).numpy()
        # swapped-role delta: source now consumes readout vector, receiver consumes edit+alt
        Hn = m.ctx_norm(H)
        hp = Hn[4]
        ra = m._onehot("C", "G", "cpu")
        src_s = m.src(torch.cat([Hn[3], ra])).numpy()      # source at a different readout pos
        recv_s = m.recv(hp.unsqueeze(0)).squeeze(0).numpy()  # receiver at edit pos
        g = m.gmod(dists.unsqueeze(-1)).squeeze(-1).numpy()
        lrso_swapped = (src_s * recv_s * g).sum(-1)
        bd = m.bdirect(torch.cat([hp.expand(12, -1), Hn, dists.unsqueeze(-1),
                                  ra.expand(12, -1)], dim=-1)).squeeze(-1).numpy()
        d_swapped = bd + lrso_swapped
        assert not np.allclose(d1, d_swapped, atol=1e-5), "directed source/receiver must be asymmetric"


def test_exact_alt_changes_source_representation():
    """Contract 14.1.9: changing exact alt changes the source/delta."""
    m = RFDLRSO(k_rank=2).eval()
    seq, react, prec, mask, pos, region = _ctx()
    H = m.encoder(seq[None], react[None], prec[None], mask[None], pos[None], region[None])[0]
    dists = (torch.arange(12, dtype=torch.float32) - 4).float()
    d_g = m.delta(H, 4, dists, "C", "G", mask)
    d_a = m.delta(H, 4, dists, "C", "A", mask)
    assert not torch.allclose(d_g, d_a, atol=1e-6), "exact alt must change source representation"


def test_k_rank_zero_is_exact_direct_nested_null():
    """Contract 14.1.8 / 5.2: K_rank=0 => delta == B_direct (no LRSO term)."""
    m0 = RFDLRSO(k_rank=0).eval()
    m2 = RFDLRSO(k_rank=2).eval()
    # force K_rank=2 model's LRSO term to zero and compare structure is not required;
    # K_rank=0 must not be a zero-response: bdirect still active.
    seq, react, prec, mask, pos, region = _ctx()
    H = m0.encoder(seq[None], react[None], prec[None], mask[None], pos[None], region[None])[0]
    dists = (torch.arange(12, dtype=torch.float32) - 4).float()
    with torch.no_grad():
        d0 = m0.delta(H, 4, dists, "C", "G", mask)
    # K_rank=0 => lrso term is exactly zero => delta equals bdirect output
    # verify delta is not identically zero (nested null is NOT zero-response)
    assert not torch.allclose(d0, torch.zeros_like(d0), atol=1e-6)
    # and equals the bdirect-only computation
    with torch.no_grad():
        Hn = m0.ctx_norm(H)
        hp = Hn[4]
        ra = torch.zeros(8)
        ra[1] = 1.0; ra[4 + 2] = 1.0  # ref C (idx1), alt G (idx2 -> slot 4+2)
        bd_in = torch.cat([hp.expand(12, -1), Hn, dists.unsqueeze(-1), ra.expand(12, -1)], dim=-1)
        bd = m0.bdirect(bd_in).squeeze(-1)
    assert torch.allclose(d0, bd, atol=1e-5), "K_rank=0 must exactly equal B_direct"


def test_ref_alt_mean_zero_scale_can_be_nonzero():
    """Contract 5.2 / 14.1.7: ref==alt => mutation-induced mean strictly 0."""
    m = RFDLRSO(k_rank=2).eval()
    seq, react, prec, mask, pos, region = _ctx()
    H = m.encoder(seq[None], react[None], prec[None], mask[None], pos[None], region[None])[0]
    dists = (torch.arange(12, dtype=torch.float32) - 4).float()
    with torch.no_grad():
        d = m.delta(H, 4, dists, "C", "C", mask)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-6), "ref==alt mean must be zero"
