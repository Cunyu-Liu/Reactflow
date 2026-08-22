from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.reactflow_delta import qualify_model_rescue_v3 as Q
from scripts.reactflow_delta.model_rescue_v3 import (
    CANDIDATE,
    INNER_PREDICTION_SCHEMA,
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.run_model_rescue_v2 import BASELINE


def _write_prediction(path: Path, *, corrupt_delta: bool = False) -> None:
    b1 = np.asarray([0.0, 0.0, 1.0])
    mean = np.asarray([1.0, 2.0, -1.0])
    disagreement = np.abs(b1 - mean)
    threshold = np.ones(3)
    alpha_low = np.full(3, 0.8)
    alpha_high = np.full(3, 0.25)
    alpha = np.where(disagreement > threshold, alpha_high, alpha_low)
    delta = b1 + alpha * (mean - b1)
    if corrupt_delta:
        delta[0] += 0.1
    point = delta + 5.0
    np.savez_compressed(
        path,
        schema_version=np.asarray(PREDICTION_SCHEMA),
        keys=np.asarray(["k0", "k1", "k2"], dtype=object),
        biological_scoring_key=np.asarray(["k0", "k1", "k2"], dtype=object),
        candidate_id=np.full(3, CANDIDATE, dtype=object),
        outer_fold=np.zeros(3, dtype=int),
        seed=np.zeros(3, dtype=int),
        b1_delta_mean=b1,
        meanaligned_delta_mean=mean,
        expert_disagreement=disagreement,
        gate_threshold=threshold,
        gate_alpha_low=alpha_low,
        gate_alpha_high=alpha_high,
        gate_alpha_applied=alpha,
        delta_mean=delta,
        point_mean=point,
        locations=np.repeat(point[:, None], 2, axis=1),
        scales=np.full((3, 2), 0.2),
        weights=np.full((3, 2), 0.5),
        registered_status=np.full(3, "covered", dtype=object),
        b1_checkpoint_path=np.full(3, "b1.pt", dtype=object),
        meanaligned_checkpoint_path=np.full(3, "mean.pt", dtype=object),
        calibration_checkpoint_path=np.full(3, "cal.pt", dtype=object),
        inner_crossfit_ledger_path=np.full(3, "inner.json", dtype=object),
    )


def test_prediction_checks_replay_gate_and_reject_blend_tampering(tmp_path: Path) -> None:
    valid = tmp_path / "valid.npz"
    _write_prediction(valid)
    assert all(Q._prediction_checks(valid).values())
    corrupt = tmp_path / "corrupt.npz"
    _write_prediction(corrupt, corrupt_delta=True)
    checks = Q._prediction_checks(corrupt)
    assert checks["blended_delta_replays"] is False


def test_inner_ledger_checks_complete_disjoint_crossfit(tmp_path: Path) -> None:
    puzzles = [f"P{i:02d}" for i in range(1, 20)]
    groups = [puzzles[index::4] for index in range(4)]
    rows = []
    for inner_fold, held in enumerate(groups):
        b1 = tmp_path / f"b1_{inner_fold}.pt"
        mean = tmp_path / f"mean_{inner_fold}.pt"
        b1.touch()
        mean.touch()
        prediction = tmp_path / f"pred_{inner_fold}.npz"
        keys = np.asarray([f"k{inner_fold}"], dtype=object)
        np.savez_compressed(
            prediction,
            schema_version=np.asarray(INNER_PREDICTION_SCHEMA),
            keys=keys,
            b1_delta_mean=np.asarray([0.0]),
            meanaligned_delta_mean=np.asarray([0.1]),
            outer_fold=np.asarray([19]),
            inner_fold=np.asarray([inner_fold]),
            seed=np.asarray([0]),
        )
        rows.append(
            {
                "inner_fold": inner_fold,
                "held_puzzles": held,
                "train_puzzles": sorted(set(puzzles) - set(held)),
                "b1_checkpoint": str(b1),
                "meanaligned_checkpoint": str(mean),
                "prediction_artifact": str(prediction),
            }
        )
    gate = {"threshold": 0.1, "alpha_low": 0.8, "alpha_high": 0.3, "quantile": 0.95}
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "inner_folds": rows,
                "gate": gate,
                "coverage": {"hierarchy_weight_sum": 1.0},
                "target_values_stored": False,
                "method_used_as_gate_input": False,
            }
        ),
        encoding="utf-8",
    )
    assert all(Q._inner_ledger_checks(ledger, "P20", gate).values())


def _score(crps: float, delta: float) -> dict:
    return {
        "crps": crps,
        "signed_delta_mae": delta,
        "registered_prediction_coverage": 1.0,
        "failure_rate": 0.0,
        "n_unexpected_prediction_keys": 0,
    }


def _screen_rows(candidate_delta: float) -> list[dict]:
    return [
        {
            "outer_fold": fold,
            "held_puzzle": f"P{fold + 1:02d}",
            "seed": 0,
            "baseline": {"model_id": BASELINE, "score": _score(0.2, 0.2)},
            "candidate": {
                "candidate_id": CANDIDATE,
                "score": _score(0.195, candidate_delta),
            },
        }
        for fold in range(20)
    ]


def test_screen_preserves_one_percent_mean_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        Q,
        "_fold_checks",
        lambda row, smoke: {"passed": True},
    )
    passed = Q.qualify_screen(_screen_rows(0.197))
    assert passed["overall_status"] == "R3M3_SCREEN_PASS"
    assert passed["r3m4_authorized"] is True
    failed = Q.qualify_screen(_screen_rows(0.1982))
    assert failed["mean_gate"]["status"] == "MEAN_GATE_FAIL"
    assert failed["overall_status"] == "MODEL_RESCUE_V3_FAIL"
    assert failed["r3m4_authorized"] is False
