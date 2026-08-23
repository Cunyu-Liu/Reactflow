from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.reactflow_delta import score_model_rescue_v4 as S


def _prediction(path: Path, seed: int, keys=("a", "b")) -> None:
    point = np.array([1.0 + seed, 2.0 + seed], dtype=float)
    n = len(keys)
    np.savez_compressed(
        path,
        keys=np.asarray(keys, dtype=object),
        biological_scoring_key=np.asarray(keys, dtype=object),
        candidate_id=np.full(n, "model", dtype=object),
        outer_fold=np.zeros(n, dtype=int),
        seed=np.full(n, seed, dtype=int),
        delta_mean=point - 0.5,
        point_mean=point,
        locations=np.stack([point, point], axis=1),
        scales=np.full((n, 2), 0.2),
        weights=np.full((n, 2), 0.5),
        registered_status=np.full(n, "covered", dtype=object),
        mean_checkpoint_path=np.full(n, "mean.pt", dtype=object),
        calibration_checkpoint_path=np.full(n, "cal.pt", dtype=object),
    )


def test_seed_assembler_preserves_equal_seed_mass_and_averages_point_mean(tmp_path) -> None:
    paths = [tmp_path / "seed0.npz", tmp_path / "seed1.npz"]
    for seed, path in enumerate(paths):
        _prediction(path, seed)
    combined = S.combine_seed_predictions(paths, "model")
    assert combined["locations"].shape == (2, 4)
    assert np.allclose(combined["weights"].sum(-1), 1.0)
    assert np.allclose(combined["weights"][:, :2].sum(-1), 0.5)
    assert np.allclose(combined["weights"][:, 2:].sum(-1), 0.5)
    assert np.allclose(combined["point_mean"], np.array([1.5, 2.5]))


def test_seed_assembler_rejects_key_universe_mismatch(tmp_path) -> None:
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    _prediction(first, 0)
    _prediction(second, 1, keys=("a", "c"))
    with pytest.raises(ValueError, match="different key universes"):
        S.combine_seed_predictions([first, second], "model")


def test_v4m1_authority_cannot_join_development_targets() -> None:
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(RuntimeError, match="closed outside active V4M3"):
        S.assert_score_authority(root, "V4M3")
