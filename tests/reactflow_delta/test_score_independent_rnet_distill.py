from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.reactflow_delta.score_independent_rnet_distill as scorer
from scripts.reactflow_delta.run_p2_v3 import _bio_key


def _score_active() -> dict:
    return {
        "project_task_id": scorer.PROJECT_TASK_ID,
        "authority": {
            "screen_prediction_dir": str(scorer.SCREEN_DIR),
            "complete_unscored_merge_path": str(scorer.MERGED_PATH),
            "m2_csv_path": str(scorer.M2_PATH),
            "historical_v14_score_path": str(scorer.V14_SCORE_PATH),
            "complete_score_path": str(scorer.SCORE_PATH),
            "qualification_path": str(scorer.QUALIFICATION_PATH),
        },
    }


def test_score_authority_uses_independent_validator_and_exact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "configs/reactflow_delta"
    config.mkdir(parents=True)
    (config / "active_contract.yaml").write_text(
        yaml.safe_dump(_score_active()), encoding="utf-8"
    )
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        scorer,
        "assert_run_authority",
        lambda root, phase: calls.append((root, phase)),
    )
    scorer.assert_score_authority(
        tmp_path,
        merged_json=scorer.MERGED_PATH,
        m2_csv=scorer.M2_PATH,
        historical_v14_score_json=scorer.V14_SCORE_PATH,
        out_json=scorer.SCORE_PATH,
    )
    assert calls == [(tmp_path, "RND4")]

    with pytest.raises(RuntimeError, match="CLI complete_score_path differs"):
        scorer.assert_score_authority(
            tmp_path,
            merged_json=scorer.MERGED_PATH,
            m2_csv=scorer.M2_PATH,
            historical_v14_score_json=scorer.V14_SCORE_PATH,
            out_json=tmp_path / "wrong.json",
        )

    changed = _score_active()
    changed["authority"]["qualification_path"] = "/mnt/cunyuliu/wrong.json"
    (config / "active_contract.yaml").write_text(
        yaml.safe_dump(changed), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="qualification_path is not exact"):
        scorer.assert_score_authority(
            tmp_path,
            merged_json=scorer.MERGED_PATH,
            m2_csv=scorer.M2_PATH,
            historical_v14_score_json=scorer.V14_SCORE_PATH,
            out_json=scorer.SCORE_PATH,
        )


def test_complete_merge_integrity_is_exact_and_target_free() -> None:
    integrity = {
        **{name: True for name in scorer.MERGE_TRUE_INVARIANTS},
        **{name: False for name in scorer.MERGE_FALSE_INVARIANTS},
    }
    assert scorer.merged_integrity_pass(integrity)
    integrity["target_free_all_runs"] = False
    assert not scorer.merged_integrity_pass(integrity)


def _write_prediction(path: Path, *, include_target: bool = False) -> None:
    n_rows = 2
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(scorer.PREDICTION_SCHEMA),
        "keys": np.asarray(["k0", "k1"], dtype=object),
        "biological_scoring_key": np.asarray(["k0", "k1"], dtype=object),
        "outer_fold": np.zeros(n_rows, dtype=np.int64),
        "seed": np.zeros(n_rows, dtype=np.int64),
        "registered_status": np.full(n_rows, "covered", dtype=object),
    }
    for name in scorer.POINT_NAMES:
        arrays[f"{name}_point"] = np.zeros(n_rows)
    for name in scorer.MIXTURE_NAMES:
        arrays[f"{name}_weights"] = np.ones((n_rows, 1))
        arrays[f"{name}_locations"] = np.zeros((n_rows, 1))
        arrays[f"{name}_scales"] = np.ones((n_rows, 1))
        arrays[f"{name}_expected_absolute_delta"] = np.ones(n_rows)
    if include_target:
        arrays["target"] = np.zeros(n_rows)
    np.savez(path, **arrays)


def test_prediction_loader_rejects_any_target_field(tmp_path: Path) -> None:
    valid = tmp_path / "rnet_distill_predictions_fold0_seed0.npz"
    _write_prediction(valid)
    result = scorer._load_prediction(valid, fold=0, prediction_root=tmp_path)
    assert set(result) == scorer.EXPECTED_PREDICTION_FIELDS

    leaked = tmp_path / "leaked" / valid.name
    leaked.parent.mkdir()
    _write_prediction(leaked, include_target=True)
    with pytest.raises(scorer.ScoreIntegrityError, match="field universe"):
        scorer._load_prediction(leaked, fold=0, prediction_root=leaked.parent)


@dataclass
class Record:
    puzzle: str
    method: str
    construct_id: str
    wt_id: str
    design_pos: int
    ref: str
    alt: str


@dataclass
class Construct:
    sequence: str
    wt_observed: np.ndarray
    wt_reactivity: np.ndarray


class FakeUniverse:
    def __init__(self, records: list[Record]) -> None:
        self.constructs = {
            row.construct_id: Construct(
                sequence="AA",
                wt_observed=np.asarray([True, True]),
                wt_reactivity=np.asarray([0.0, 0.0]),
            )
            for row in records
        }

    def get_construct(self, construct_id: str) -> Construct:
        return self.constructs[construct_id]

    def mutant_full_profile(self, _wt_id: str, *_args):
        return np.asarray([1.0, 1.0]), np.asarray([0.1, 0.1])


def _fold_prediction(univ: FakeUniverse, records: list[Record]) -> dict[str, np.ndarray]:
    keys = [_bio_key(univ, row, position) for row in records for position in range(2)]
    n_rows = len(keys)
    candidate_point = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    prediction: dict[str, np.ndarray] = {
        "keys": np.asarray(keys, dtype=object),
        "feature41_point": np.zeros(n_rows),
        "candidate_point": candidate_point,
        "null_point": np.zeros(n_rows),
    }
    for name, locations in (
        ("feature41", np.zeros(n_rows)),
        ("candidate", candidate_point),
        ("null", np.zeros(n_rows)),
    ):
        prediction[f"{name}_weights"] = np.ones((n_rows, 1))
        prediction[f"{name}_locations"] = locations[:, None]
        prediction[f"{name}_scales"] = np.ones((n_rows, 1))
        prediction[f"{name}_expected_absolute_delta"] = np.abs(locations)
    prediction["historical_v10_expected_absolute_delta"] = np.zeros(n_rows)
    return prediction


def test_score_fold_uses_puzzle_method_macro_and_all_frozen_comparators() -> None:
    records = [
        Record("P01", "M1", "P01_M1", "wt1", 0, "A", "C"),
        Record("P01", "M2", "P01_M2", "wt2", 0, "A", "C"),
        Record("P01", "M2", "P01_M2", "wt2", 1, "A", "G"),
    ]
    univ = FakeUniverse(records)
    result = scorer.score_fold(univ, records, _fold_prediction(univ, records))
    # M1 candidate loss=0 and M2 mean candidate loss=1, so equal-method macro=0.5.
    assert result["candidate_signed_delta_mae"] == pytest.approx(0.5)
    assert result["candidate_point_absolute_delta_mae"] == pytest.approx(0.5)
    assert result["null_signed_delta_mae"] == pytest.approx(1.0)
    assert result["historical_v10_distribution_absolute_delta_mae"] == pytest.approx(1.0)
    assert result["registered_prediction_coverage"] == 1.0
    assert result["score_integrity_pass"] is True


def test_score_fold_refuses_incomplete_registered_universe() -> None:
    record = Record("P01", "M1", "P01_M1", "wt1", 0, "A", "C")
    univ = FakeUniverse([record])
    prediction = {"keys": np.asarray([_bio_key(univ, record, 0)], dtype=object)}
    with pytest.raises(scorer.ScoreIntegrityError, match="not exact"):
        scorer.score_fold(univ, [record], prediction)
