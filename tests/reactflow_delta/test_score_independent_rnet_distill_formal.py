from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.reactflow_delta.score_independent_rnet_distill_formal as scorer


def _active() -> dict:
    return {
        "project_task_id": scorer.PROJECT_TASK_ID,
        "authority": {
            "formal_prediction_dir": str(scorer.FORMAL_DIR),
            "formal_complete_unscored_merge_path": str(scorer.MERGED_PATH),
            "formal_assembly_dir": str(scorer.ASSEMBLY_DIR),
            "formal_assembly_path": str(scorer.ASSEMBLY_PATH),
            "m2_csv_path": str(scorer.M2_PATH),
            "historical_v14_score_path": str(scorer.V14_SCORE_PATH),
            "formal_complete_score_path": str(scorer.FORMAL_SCORE_PATH),
            "formal_qualification_path": str(scorer.FORMAL_QUALIFICATION_PATH),
            "screen_qualification_path": str(scorer.SCREEN_QUALIFICATION_PATH),
        },
    }


def test_formal_score_authority_requires_exact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "configs/reactflow_delta"
    config.mkdir(parents=True)
    (config / "active_contract.yaml").write_text(
        yaml.safe_dump(_active()), encoding="utf-8"
    )
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        scorer, "assert_run_authority", lambda root, phase: calls.append((root, phase))
    )
    scorer.assert_formal_score_authority(
        tmp_path,
        merged_json=scorer.MERGED_PATH,
        assembly_json=scorer.ASSEMBLY_PATH,
        m2_csv=scorer.M2_PATH,
        historical_v14_score_json=scorer.V14_SCORE_PATH,
        out_json=scorer.FORMAL_SCORE_PATH,
    )
    assert calls == [(tmp_path, "RND6S")]

    with pytest.raises(RuntimeError, match="CLI formal_complete_score_path differs"):
        scorer.assert_formal_score_authority(
            tmp_path,
            merged_json=scorer.MERGED_PATH,
            assembly_json=scorer.ASSEMBLY_PATH,
            m2_csv=scorer.M2_PATH,
            historical_v14_score_json=scorer.V14_SCORE_PATH,
            out_json=tmp_path / "wrong.json",
        )


def _prediction(*, fold: int, seed: int, assembled: bool) -> dict[str, np.ndarray]:
    n_rows = 2
    fields = (
        scorer.EXPECTED_FORMAL_PREDICTION_FIELDS
        if assembled
        else scorer.EXPECTED_PREDICTION_FIELDS
    )
    result: dict[str, np.ndarray] = {
        "schema_version": np.asarray(
            scorer.FORMAL_PREDICTION_SCHEMA if assembled else scorer.PREDICTION_SCHEMA
        ),
        "keys": np.asarray(["k0", "k1"], dtype=object),
        "biological_scoring_key": np.asarray(["k0", "k1"], dtype=object),
        "outer_fold": np.full(n_rows, fold, dtype=np.int64),
        "seed": np.full(n_rows, seed, dtype=np.int64),
        "registered_status": np.full(n_rows, "covered", dtype=object),
    }
    if assembled:
        result["assembled_seed_count"] = np.full(n_rows, 5, dtype=np.int64)
    for name in scorer.POINT_NAMES:
        result[f"{name}_point"] = np.zeros(n_rows)
    for name in scorer.MIXTURE_NAMES:
        components = 2 if (not assembled or name in {"feature41", "historical_v10"}) else 10
        result[f"{name}_weights"] = np.full((n_rows, components), 1.0 / components)
        result[f"{name}_locations"] = np.zeros((n_rows, components))
        result[f"{name}_scales"] = np.ones((n_rows, components))
        result[f"{name}_expected_absolute_delta"] = np.ones(n_rows)
    assert set(result) == set(fields)
    return result


def test_formal_prediction_validation_accepts_equal_seed_mixture() -> None:
    scorer._validate_prediction_arrays(
        _prediction(fold=3, seed=-1, assembled=True),
        fields=scorer.EXPECTED_FORMAL_PREDICTION_FIELDS,
        schema=scorer.FORMAL_PREDICTION_SCHEMA,
        fold=3,
        seed=-1,
        source_components=False,
    )


def test_formal_prediction_validation_rejects_non_equal_component_weights() -> None:
    prediction = _prediction(fold=3, seed=-1, assembled=True)
    prediction["candidate_weights"][0, 0] = 0.9
    with pytest.raises(scorer.ScoreIntegrityError, match="candidate distribution"):
        scorer._validate_prediction_arrays(
            prediction,
            fields=scorer.EXPECTED_FORMAL_PREDICTION_FIELDS,
            schema=scorer.FORMAL_PREDICTION_SCHEMA,
            fold=3,
            seed=-1,
            source_components=False,
        )
