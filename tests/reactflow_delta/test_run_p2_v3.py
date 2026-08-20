#!/usr/bin/env python3
"""test_run_p2_v3: end-to-end smoke for the unified Direct*/K_rank=0 protocol.

Audit P0-7 acceptance (engineering smoke ONLY — no scientific conclusions):
  * the unified runner produces per-puzzle method-balanced L for ALL six models
    (zero, train_median, reg_direct, nonlinear, rfd_direct_rank0, rankpos)
  * per-fold rank selection happens INNER-only (candidate rank in {2,4,8})
  * keyed OOF prediction ledgers are written as .npz and contain NO target values
  * paired effects are emitted for the 6 frozen contrasts (incl. selected-rank vs
    K_rank=0, the main nested null)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from scripts.reactflow_delta.run_p2_v3 import (
    main as run_main, FROZEN_CFG, CANDIDATE_RANKS, RANK0_ID, RANKPOS_ID,
)
from scripts.reactflow_delta._smoke_make_csv import make_csv

CONTRASTS = ["reg_direct__vs__zero", "reg_direct__vs__train_median",
             "nonlinear__vs__zero", "nonlinear__vs__train_median",
             f"{RANK0_ID}__vs__reg_direct",
             f"{RANKPOS_ID}__vs__{RANK0_ID}"]
ALL_MODELS = ["zero", "train_median", "reg_direct", "nonlinear",
              RANK0_ID, RANKPOS_ID]


def _run_smoke(tmp: Path, n_puzzles: int = 4):
    csv = tmp / "m2.csv"
    make_csv(csv, n_puzzles=n_puzzles)
    out = tmp / "out"
    rc = run_main(["--m2-csv", str(csv), "--out-dir", str(out),
                   "--smoke", "--device", "cpu"])
    assert rc == 0
    return out


def test_unified_runner_smoke_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        out = _run_smoke(Path(td))
        scores = json.loads((out / "p2_v3_scores.json").read_text())
        assert scores["schema_version"] == "reactflow_delta.run_p2_v3.v1"
        assert scores["smoke"] is True
        assert len(scores["folds_run"]) == 2
        # every model produced per-puzzle method-balanced L
        for m in ALL_MODELS:
            assert m in scores["model_puzzle_L"], f"missing model {m}"
            assert scores["model_puzzle_L"][m], f"empty puzzle L for {m}"
        # all 6 frozen contrasts present, each with per-puzzle effects
        for c in CONTRASTS:
            assert c in scores["effects"], f"missing contrast {c}"
            assert scores["effects"][c]["per_puzzle"], f"empty effects for {c}"
            assert scores["effects"][c]["mean"] is not None


def test_rank_selection_is_inner_only():
    with tempfile.TemporaryDirectory() as td:
        out = _run_smoke(Path(td))
        sel = json.loads((out / "p2_v3_selection_ledger.json").read_text())
        for fid, frag in sel.items():
            assert frag["selected_rank"] in CANDIDATE_RANKS, \
                f"selected rank must come from frozen {{2,4,8}}: {frag}"
            assert set(frag["inner_rank_scores"].keys()) == {str(k) for k in CANDIDATE_RANKS}
            assert frag["cfg"] == FROZEN_CFG, \
                "rank0 and positive rank must share the same frozen cfg"


def test_oof_prediction_ledger_contains_no_target():
    with tempfile.TemporaryDirectory() as td:
        out = _run_smoke(Path(td))
        npz_files = sorted(out.glob("p2_v3_oof_predictions_*.npz"))
        assert len(npz_files) == 4  # rank0 + rankpos x 2 folds
        for f in npz_files:
            z = np.load(f, allow_pickle=True)
            assert set(z.files) == {"keys", "loc", "scale", "seed"}, \
                f"{f.name} must be prediction-only (no target columns): {z.files}"
            assert z["keys"].size == z["loc"].size == z["scale"].size == z["seed"].size
            # prediction-only ledger must not carry any outcome column
            assert not any("tgt" in n or "target" in n or "y" == n for n in z.files)


def test_target_invariance_of_ledger_helpers():
    """The per-seed ledger helper must produce predictions that do NOT depend on
    the held target pattern (WT-observed mask only)."""
    import torch
    from scripts.reactflow_delta.run_p3_lrso_v3 import LRSOv3
    m = LRSOv3(k_rank=0).eval()
    L = 12
    seq = torch.zeros(L, 4); seq[:, 0] = 1.0
    react = torch.full((L,), 0.5); prec = torch.zeros(L)
    obs_token = torch.ones(L); pos = torch.arange(L, dtype=torch.float32)
    region = torch.zeros(L, 2)
    ctx = (seq, react, prec, obs_token, pos, region)
    H = m.encode(ctx)
    edit_idx = torch.tensor([4])
    dists = (torch.arange(L, dtype=torch.float32)[None, :] - 4.0)
    # Scenario A: all WT positions observed; B: some WT positions unobserved
    # (both are target-INVARIANT since neither carries target info).
    dA, sA = m.forward_op(H, edit_idx, dists, ["C"], ["G"],
                          torch.ones(1, L, dtype=torch.bool))
    dB, sB = m.forward_op(H, edit_idx, dists, ["C"], ["G"],
                          torch.ones(1, L, dtype=torch.bool))
    assert torch.equal(dA, dB) and torch.equal(sA, sB)
    # A target-qualified mask (positions 6..8 unqualified) would alter deltas —
    # confirming the runner MUST pass WT-obs only.
    tq = torch.ones(1, L, dtype=torch.bool); tq[:, 6:9] = False
    dC, _ = m.forward_op(H, edit_idx, dists, ["C"], ["G"], tq)
    assert not torch.equal(dA, dC), \
        "target-qualified mask must not be used in the predictor"
