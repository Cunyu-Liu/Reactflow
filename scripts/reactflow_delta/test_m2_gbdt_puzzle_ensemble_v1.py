#!/usr/bin/env python3
"""test_m2_gbdt_puzzle_ensemble_v1.py — tests for the puzzle-level LOPO
GBDT + deep ensemble pipeline."""
import json
from pathlib import Path

import numpy as np

import m2_gbdt_puzzle_ensemble_v1 as pe


def _mk_pred(tmp_path, n_pairs=6, n_seeds=5, win=21):
    """Synthetic puzzle OOF jsonl: per pair, prior seed0 + attn seeds 0..4."""
    lines = []
    for i in range(n_pairs):
        pid = f"OK7a_M2_P{i % 3 + 1:02d}_design{i}"
        y = np.linspace(0.0, 1.0, win)
        w = np.ones(win)
        for k in range(win):
            y[k] = float(np.sin(k) * 0.3 + i * 0.01)
        prior = y + 0.1
        lines.append({"pair_id": pid, "task": "magnitude_spectrum",
                      "model_variant": "wmed_spectrum", "seed": 0,
                      "coverage_status": "CALLED",
                      "source_accession": pid.split(":")[0],
                      "raw_prediction": prior.tolist(),
                      "y": y.tolist(), "weight": w.tolist()})
        for s in range(n_seeds):
            p = y - 0.05 - s * 0.01
            lines.append({"pair_id": pid, "task": "magnitude_spectrum",
                          "model_variant": "wmae_resid_attn_spectrum", "seed": s,
                          "coverage_status": "CALLED",
                          "source_accession": pid.split(":")[0],
                          "raw_prediction": p.tolist(),
                          "y": y.tolist(), "weight": w.tolist()})
    p = Path(tmp_path) / "puzzle_preds.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")
    return str(p)


def test_puzzle_of():
    assert pe.puzzle_of("OK7a_M2_P01_Eterna") == "P01"
    assert pe.puzzle_of("OK7a_M2_P20_MPNN-fixbb") == "P20"


def test_load_puzzle_preds(tmp_path):
    p = _mk_pred(tmp_path)
    attn, prior, designs, n_inc = pe._load_puzzle_preds(p)
    assert len(attn) == 6 and len(prior) == 6
    pid = "OK7a_M2_P01_design0"
    assert len(attn[pid]) == 21 and len(prior[pid]) == 21
    assert designs[pid] == pid
    assert n_inc == 0


def test_analyze_puzzle_block(tmp_path):
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.normal(0.0, 1.0, n)
    w = np.ones(n)
    pz = np.array([f"P{i % 20 + 1:02d}" for i in range(n)])
    pred = y * 0.8
    base = y * 1.5
    sig = pe.analyze_puzzle(y, w, pz, pred, base, n_perm=30, n_boot=30, seed=1)
    assert sig["n_puzzles"] == 20
    assert sig["ci_low"] <= sig["ci_high"]
    assert 0.0 <= sig["permutation_p"] <= 1.0
    assert sig["skill"] > 0.0  # pred better than base


def _run_synthetic(tmp_path, monkeypatch):
    """Run run_m2_gbdt_puzzle_ensemble with tiny synthetic X/y/deep/prior."""
    rng = np.random.default_rng(1)
    n = 600
    X = rng.normal(0, 1, (n, 4))
    y = rng.normal(0.0, 1.0, n)
    w = np.ones(n)
    keys = np.array([f"OK7a_M2_P{i % 3 + 1:02d}_design{i}" for i in range(n)],
                    dtype="U64")
    # pair_id = "design_id:mutA"; pid row = "design_id:mutA:k"
    pids = np.array([f"{k}:m{i}:{i % 21}" for i, k in enumerate(keys)],
                    dtype="U96")
    deep = {f"{k}:m{i}": rng.normal(0.0, 1.0, 21)
            for i, k in enumerate(keys)}
    prior = {f"{k}:m{i}": rng.normal(0.0, 1.0, 21)
             for i, k in enumerate(keys)}

    def fake_loo(X, y, sample_puzzles, puzzles):
        return 0.5 * y + rng.normal(0, 0.05, len(y))

    monkeypatch.setattr(pe, "_loo_puzzle_lgb", fake_loo)

    class A:
        out = str(tmp_path / "out")
        alpha = 0.5
        n_perm = 30
        n_boot = 30

    rep = pe.run_m2_gbdt_puzzle_ensemble(X, y, w, keys, pids, deep, prior, A())
    return rep


def test_run_puzzle_ensemble_structure(tmp_path, monkeypatch):
    rep = _run_synthetic(tmp_path, monkeypatch)
    assert rep["fold_unit"] == "puzzle"
    assert "blend" in rep["results"]
    assert "sig" in rep["results"]["blend"]
    assert rep["results"]["blend"]["sig"]["n_puzzles"] == 3
    assert "blend_vs_deep" in rep
    assert "loo_exclusion" in rep["blend_vs_deep"]
    # report json written
    rp = Path(tmp_path) / "out" / "m2_gbdt_puzzle_ensemble_report.json"
    assert rp.exists()
    j = json.loads(rp.read_text(encoding="utf-8"))
    assert j["schema"].endswith("m2_gbdt_puzzle_ensemble.v1")
