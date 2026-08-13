#!/usr/bin/env python3
"""lrso_v1: RFD-LRSO model (contract 5.2, 10.2).

DeltaA_{p,i} = B_direct(i-p, a->b, H_p, H_i)
               + sum_k S_k(H_p, a->b) * R_k(H_i) * G_k(i-p)

  - WT context encoder: 2 relative-position attention blocks (hidden 96, 4 heads)
  - source head S_k / receiver head R_k: hidden 64, roles NOT shared (asymmetric)
  - G_k: continuous signed-distance modulation
  - K_rank=0 => exactly B_direct (nested direct null)
  - ref=alt => mutation-induced mean forced to 0 (mask applied downstream)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = {"A": 0, "C": 1, "G": 2, "U": 3}


class RelativeAttentionBlock(nn.Module):
    def __init__(self, d: int = 96, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        # relative position bias per head over a max window (small init for stability)
        self.rel = nn.Parameter(torch.randn(heads, 401) * 0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d), mask: (B, L)
        B, L, d = x.shape
        xn = self.norm1(x)
        qkv = self.qkv(xn).reshape(B, L, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,L,dh)
        att = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5)
        rel = torch.stack([self.rel[h, torch.clamp(torch.arange(L, device=x.device)[None, :]
                                                   - torch.arange(L, device=x.device)[:, None] + 200, 0, 400)]
                           for h in range(self.heads)])
        att = att + rel.unsqueeze(0)
        pad = (~mask).unsqueeze(1).unsqueeze(-1)
        att = att.masked_fill(pad, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(pad, 0.0)
        ctx = (att @ v).transpose(1, 2).reshape(B, L, d)
        return self.norm2(x + self.out(ctx))


class WTContextEncoder(nn.Module):
    """2 relative-position attention blocks; hidden 96, 4 heads."""

    def __init__(self, d: int = 96, heads: int = 4) -> None:
        super().__init__()
        self.seq_emb = nn.Linear(4, d)
        self.react_emb = nn.Linear(1, d)
        self.prec_emb = nn.Linear(1, d)
        self.pos_emb = nn.Linear(1, d)
        self.region_emb = nn.Linear(2, d)
        self.blocks = nn.Sequential(
            RelativeAttentionBlock(d, heads),
            RelativeAttentionBlock(d, heads),
        )

    def forward(self, seq: torch.Tensor, react: torch.Tensor, prec: torch.Tensor,
                mask: torch.Tensor, pos: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        x = (self.seq_emb(seq) + self.react_emb(react.unsqueeze(-1))
             + self.prec_emb(prec.unsqueeze(-1)) + self.pos_emb(pos.unsqueeze(-1))
             + self.region_emb(region))
        for blk in self.blocks:
            x = blk(x, mask)
        return x


class RFDLRSO(nn.Module):
    def __init__(self, d: int = 96, heads: int = 4, hidden: int = 64, k_rank: int = 2) -> None:
        super().__init__()
        self.k_rank = k_rank
        self.encoder = WTContextEncoder(d, heads)
        self.ctx_norm = nn.LayerNorm(d)
        self.src = nn.Sequential(nn.Linear(d + 8, hidden), nn.ReLU(), nn.Linear(hidden, max(k_rank, 1)))
        self.recv = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, max(k_rank, 1)))
        self.gmod = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, max(k_rank, 1)))
        self.bdirect = nn.Sequential(nn.Linear(d + d + 1 + 8, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))

    def _onehot(self, base: str, device: torch.device) -> torch.Tensor:
        v = torch.zeros(8, device=device)
        v[ALPHA.get(base, 3)] = 1.0
        v[4 + ALPHA.get(base, 3)] = 0.0
        return v

    def delta(self, H: torch.Tensor, edit_idx: int, dists: torch.Tensor,
              ref: str, alt: str, mask: torch.Tensor) -> torch.Tensor:
        """Delta for all positions i of one construct given an edit at edit_idx."""
        # H: (L, d); dists: (L,); mask: (L,)
        device = H.device
        H = self.ctx_norm(H)
        hp = H[edit_idx]
        ra = self._onehot(ref, device)
        src = self.src(torch.cat([hp, ra]))          # (k,) or (1,)
        recv = self.recv(H)                           # (L, k)
        g = self.gmod(dists.unsqueeze(-1))            # (L, k)
        bd_in = torch.cat([hp.expand(H.shape[0], -1), H, dists.unsqueeze(-1),
                           ra.expand(H.shape[0], -1)], dim=-1)
        bd = self.bdirect(bd_in).squeeze(-1)          # (L,)
        if self.k_rank == 0:
            lrso = torch.zeros_like(bd)
        else:
            lrso = (src * recv * g).sum(-1)           # (L,)
        delta = bd + lrso
        delta = delta.masked_fill(~mask, 0.0)
        # ref==alt => mean forced 0
        if ref == alt:
            delta = delta * 0.0
        return delta

    def delta_batch(self, H: torch.Tensor, edit_idx: torch.Tensor, dists: torch.Tensor,
                    refs: list[str], alts: list[str], masks: torch.Tensor) -> torch.Tensor:
        """Batched delta for B mutants of one construct: (B, L)."""
        device = H.device
        B = edit_idx.shape[0]
        L = H.shape[0]
        H = self.ctx_norm(H)
        hp = H[edit_idx]  # (B, d)
        ref_idx = torch.tensor([ALPHA.get(x, 3) for x in refs], device=device)
        alt_idx = torch.tensor([ALPHA.get(x, 3) for x in alts], device=device)
        ra = torch.zeros(B, 8, device=device)
        ra.scatter_(1, ref_idx[:, None], 1.0)
        ra.scatter_(1, alt_idx[:, None] + 4, 1.0)
        src = self.src(torch.cat([hp, ra], dim=-1))            # (B, k)
        recv = self.recv(H)                                     # (L, k)
        g = self.gmod(dists.unsqueeze(-1))                      # (B, L, k)
        hp_e = hp.unsqueeze(1).expand(B, L, -1)                 # (B, L, d)
        H_e = H.unsqueeze(0).expand(B, -1, -1)                  # (B, L, d)
        ra_e = ra.unsqueeze(1).expand(B, L, -1)                 # (B, L, 8)
        bd = self.bdirect(torch.cat([hp_e, H_e, dists.unsqueeze(-1), ra_e], dim=-1)).squeeze(-1)  # (B, L)
        if self.k_rank == 0:
            lrso = torch.zeros_like(bd)
        else:
            lrso = (src.unsqueeze(1) * recv.unsqueeze(0) * g).sum(-1)  # (B, L)
        delta = bd + lrso
        delta = delta.masked_fill(~masks, 0.0)
        for bi in range(B):
            if refs[bi] == alts[bi]:
                delta[bi] = 0.0
        return delta
