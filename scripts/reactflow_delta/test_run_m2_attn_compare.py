#!/usr/bin/env python3
"""test_run_m2_attn_compare — unit tests for the M2 attention comparison poller."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_m2_attn_compare as rc  # noqa: E402

W = 21
SEEDS = [0, 1, 2, 3, 4]


def _synth_rows(designs, variant, n_pairs=8):
    rng = np.random.default_rng(0)
    rows = []
    for d in designs:
        for _ in range(n_pairs):
            pid = f"{d}:A{d}"
            y = rng.normal(size=W).tolist()
            w = [1.0] * W
            prior = [0.0] * W
            if variant == "wmed_spectrum":
                rows.append({"pair_id": pid, "task": "magnitude_spectrum",
                             "fold_id": d, "seed": 0, "model_variant": "wmed_spectrum",
                             "model_id": "x", "publication_id": d,
                             "source_accession": d, "split_role": "development",
                             "endpoint_version": "m2", "caller_version": "m2_caller_v1",
                             "caller_mode": "PER_POS_ERROR", "y": y, "weight": w,
                             "raw_prediction": prior, "transformed_prediction": prior,
                             "coverage_status": "CALLED"})
            else:
                for s in SEEDS:
                    rows.append({"pair_id": pid, "task": "magnitude_spectrum",
                                 "fold_id": d, "seed": s, "model_variant": variant,
                                 "model_id": "x", "publication_id": d,
                                 "source_accession": d, "split_role": "development",
                                 "endpoint_version": "m2", "caller_version": "m2_caller_v1",
                                 "caller_mode": "PER_POS_ERROR", "y": y, "weight": w,
                                 "raw_prediction": [float(rng.normal(0, 0.1)) for _ in range(W)],
                                 "transformed_prediction": [0.0] * W,
                                 "coverage_status": "CALLED"})
    return rows


def test_progress_parse(tmp_path):
    p = tmp_path / "fold_progress.json"
    p.write_text(json.dumps({"completed_folds": ["A", "B"]}), encoding="utf-8")
    assert rc._progress(str(p)) == {"A", "B"}
    assert rc._progress(str(tmp_path / "missing.json")) == set()


def test_filter_to_designs(tmp_path):
    rows = _synth_rows(["D1", "D2", "D3"], "wmae_resid_spectrum")
    src = tmp_path / "pred.jsonl"
    src.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
                   encoding="utf-8")
    dst = tmp_path / "filtered.jsonl"
    n = rc._filter_to_designs(str(src), {"D1", "D3"}, str(dst))
    assert n == 2 * 8 * len(SEEDS)  # 2 designs x 8 pairs x 5 seeds
    got = [json.loads(l) for l in dst.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {r["pair_id"].split(":")[0] for r in got} == {"D1", "D3"}


def test_rows_summary_shape():
    report = {
        "mu_ensemble": {
            "skill": 0.1, "ci_low": 0.05, "ci_high": 0.15, "permutation_p": 0.01,
            "n_designs": 5, "n_positions": 100,
            "per_design": {"mean": 0.1, "median": 0.1, "pct_positive": 1.0},
        },
        "n_seed_single_skill": {"seed_0": {"skill": 0.08}},
    }
    s = rc._rows_summary(report)
    assert s["skill"] == 0.1
    assert s["per_seed_skill"]["seed_0"] == 0.08
