#!/usr/bin/env python3
"""Tests for m2_gbdt_ensemble_v1 — M2 GBDT + attn cross-architecture ensemble."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_gbdt_ensemble_v1 as ge


def _make_rows():
    rng = np.random.default_rng(0)
    n_des, per = 4, 120
    y = rng.normal(0.0, 1.0, n_des * per)
    w = np.ones(n_des * per)
    keys = np.concatenate([np.full(per, f"D{i}") for i in range(n_des)]).astype("U64")
    pids = np.array([f"D{i}:m{k % 21}:{k % 21}" for i in range(n_des)
                     for k in range(per)], dtype="U96")
    return y, w, keys, pids


def test_run_m2_gbdt_ensemble_structure(tmp_path, monkeypatch):
    y, w, keys, pids = _make_rows()
    X = np.random.default_rng(1).normal(size=(len(y), 12))

    def fake_loo(X, y, keys, des_list):
        return 0.5 * y + np.random.default_rng(3).normal(0, 0.05, len(y))

    monkeypatch.setattr(ge, "_loo_lgb", fake_loo)

    attn_pred = {}
    prior_pred = {}
    for i in range(4):
        for k in range(120):
            pid = f"D{i}:m{k % 21}"
            attn_pred[pid] = np.full(21, 0.4 * y[i * 120 + k])
            prior_pred[pid] = np.zeros(21)

    class A:
        pass
    A.out = str(tmp_path / "out")
    A.n_perm = 50
    A.n_boot = 50
    A.alpha = 0.5
    A.deep_component = "attn_v5"
    rep = ge.run_m2_gbdt_ensemble(X, y, w, keys, pids, attn_pred, prior_pred, A)

    assert "results" in rep
    for k in ("gbdt_3way", "deep_mu", "blend"):
        assert "skill" in rep["results"][k]
    assert "blend_curve" in rep
    assert "blend_vs_deep" in rep
    assert "loo_exclusion" in rep["blend_vs_deep"]
    assert rep["blend_vs_deep"]["loo_exclusion"]["n_folds"] >= 3
    assert (tmp_path / "out" / "m2_gbdt_ensemble_report.json").exists()
    assert (tmp_path / "out" / "m2_gbdt_ensemble_oof.npz").exists()
    d = json.loads((tmp_path / "out" / "m2_gbdt_ensemble_report.json").read_text())
    assert d["schema"] == "reactflow_delta.response_spectrum.m2_gbdt_ensemble.v1"


def test_load_preds_from_file(tmp_path):
    import json as _json
    rows = []
    for pid in ("OK7a_M2_P01_Eterna:1", "OK7a_M2_P01_Eterna:2"):
        rows.append({"task": "magnitude_spectrum", "coverage_status": "CALLED",
                     "pair_id": pid, "source_accession": "OK7a_M2_P01_Eterna",
                     "model_variant": "wmed_spectrum", "seed": 0,
                     "raw_prediction": [0.1] * 21})
        for s in range(5):
            rows.append({"task": "magnitude_spectrum", "coverage_status": "CALLED",
                         "pair_id": pid, "source_accession": "OK7a_M2_P01_Eterna",
                         "model_variant": "wmae_resid_attn_spectrum", "seed": s,
                         "raw_prediction": [0.2 + 0.01 * s] * 21})
    f = tmp_path / "preds.jsonl"
    f.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    attn, prior, designs = ge._load_preds(str(f))
    assert len(attn) == 2 and len(prior) == 2
    # mu-ensemble = mean over 5 seeds = 0.2 + 0.01*2 = 0.22
    assert abs(attn["OK7a_M2_P01_Eterna:1"][0] - 0.22) < 1e-9
    assert abs(prior["OK7a_M2_P01_Eterna:1"][0] - 0.1) < 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
