from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.reactflow_delta.assemble_puzzle_set_meta_context_formal import (
    assert_assembly_authority,
)
from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import (
    assert_merge_authority,
)
from scripts.reactflow_delta.puzzle_set_score_chain import (
    EXPECTED_PROJECT_TASK,
    V13_HISTORICAL_COUNT_FIELDS,
    V13_HISTORICAL_METRIC_FIELDS,
    V13_HISTORICAL_ROW_FIELDS,
    validate_v13_historical_bundle,
)
from scripts.reactflow_delta.puzzle_set_safe_sources import (
    EXPECTED_CONTRACT_ID,
    FOLD_SCOPED_INPUT_SOURCES,
    FROZEN_INPUT_SOURCE_SPEC,
    SOURCE_BINDING_STATUS,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_MANIFEST_STATUS,
)
from scripts.reactflow_delta.qualify_puzzle_set_meta_context_smoke import (
    assert_smoke_qualifier_authority,
)
from scripts.reactflow_delta.qualify_puzzle_set_meta_context import (
    assert_qualifier_authority as assert_screen_qualifier_authority,
)
from scripts.reactflow_delta.qualify_puzzle_set_meta_context_formal import (
    assert_qualifier_authority as assert_formal_qualifier_authority,
)
from scripts.reactflow_delta.score_model_rescue_v13 import SCHEMA as V13_SCORE_SCHEMA
from scripts.reactflow_delta.score_puzzle_set_meta_context import (
    assert_score_authority as assert_screen_score_authority,
)
from scripts.reactflow_delta.score_puzzle_set_meta_context_formal import (
    assert_score_authority as assert_formal_score_authority,
)


def _v13_bundle() -> dict:
    rows = []
    for fold in range(20):
        row = {
            "outer_fold": fold,
            "held_puzzle": f"P{fold + 1:02d}",
            **{field: 0.5 for field in V13_HISTORICAL_METRIC_FIELDS},
            **{field: 1 for field in V13_HISTORICAL_COUNT_FIELDS},
        }
        row.update(
            {
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
            }
        )
        assert set(row) == V13_HISTORICAL_ROW_FIELDS
        rows.append(row)
    return {
        "schema_version": V13_SCORE_SCHEMA,
        "phase": "V13M3",
        "status": "V13M3_COMPLETE_SCORE_PASS",
        "scores": rows,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "terminal_parent_metrics_from_frozen_complete_v12_score": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }


def test_v13_historical_bundle_requires_exact_complete_safe_lineage() -> None:
    rows = validate_v13_historical_bundle(_v13_bundle())
    assert tuple(rows) == tuple(range(20))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("phase", "V13M2"),
        lambda value: value.__setitem__("partial_fold_scores_inspected", True),
        lambda value: value["scores"][4].__setitem__("held_puzzle", "P06"),
        lambda value: value["scores"][4].__setitem__(
            "candidate_signed_delta_mae", float("nan")
        ),
        lambda value: value["scores"][4].__setitem__("candidate_crps", -0.1),
        lambda value: value["scores"][4].__setitem__("candidate_crps", True),
        lambda value: value["scores"][4].__setitem__("candidate_crps", "0.5"),
        lambda value: value["scores"][4].__setitem__(
            "registered_prediction_coverage", 0.99
        ),
        lambda value: value["scores"][4].__setitem__("failure_rate", 0.01),
        lambda value: value["scores"][4].__setitem__("n_unexpected_prediction_keys", 1),
        lambda value: value["scores"][4].__setitem__("n_registered_observed", 2),
        lambda value: value["scores"][4].__setitem__("n_qualified_positions", 0),
        lambda value: value["scores"].__setitem__(4, copy.deepcopy(value["scores"][3])),
    ],
)
def test_v13_historical_bundle_rejects_protocol_or_fold_tampering(mutation) -> None:
    value = _v13_bundle()
    mutation(value)
    with pytest.raises(ValueError, match="V13 historical bundle|protocol"):
        validate_v13_historical_bundle(value)


def _write_active(
    repo_root: Path,
    *,
    phase: str,
    held_score: str | bool,
    paths: dict[str, Path],
    training_allowed: bool,
) -> None:
    config = repo_root / "configs/reactflow_delta"
    config.mkdir(parents=True, exist_ok=True)
    source_manifest = _write_bound_source_manifest(repo_root)
    active = {
        "project_task_id": EXPECTED_PROJECT_TASK,
        "authority": {
            "current_phase": phase,
            "source_manifest_path": str(source_manifest),
            "source_binding_status": SOURCE_BINDING_STATUS,
            **{name: str(path) for name, path in paths.items()},
        },
        "runnable_phases": [phase],
        "training_allowed": training_allowed,
        "candidate_model_training_allowed": training_allowed,
        "held_score_read_allowed": held_score,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
    }
    (config / "active_contract.yaml").write_text(
        yaml.safe_dump(active), encoding="utf-8"
    )


def _write_bound_source_manifest(repo_root: Path) -> Path:
    manifest_path = (repo_root / "source_binding/source_manifest.json").resolve()
    if manifest_path.is_file():
        return manifest_path
    source_root = (repo_root / "source_fixture").resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    global_paths = {
        "tic2a_merged_registry": source_root / "tic2a_merged.json",
        "unconstrained_feature_cache": source_root / "unconstrained.h5",
        "constrained_feature_cache": source_root / "constrained.h5",
    }
    for path in global_paths.values():
        path.touch()
    rows = []
    for fold in range(20):
        fold_paths = {
            "v13_point_checkpoint": (
                source_root / f"v13_candidate_point_fold{fold}_seed0.pt"
            ),
            "v14_encoder_checkpoint": (
                source_root / f"v14_candidate_point_fold{fold}_seed0.pt"
            ),
            "v8_meanaligned_checkpoint": (
                source_root / f"v8_corrected_mean_fold{fold}_seed0.pt"
            ),
            "tic2a_feature41_model_artifact": (
                source_root / f"tic2a_corrected_models_fold{fold}.json"
            ),
        }
        for path in fold_paths.values():
            path.touch()
        sources = {}
        for source_id, expected in FROZEN_INPUT_SOURCE_SPEC.items():
            path = (
                fold_paths[source_id]
                if source_id in FOLD_SCOPED_INPUT_SOURCES
                else global_paths[source_id]
            )
            sources[source_id] = {
                "path": str(path),
                "role": expected["role"],
                "used_in_candidate_prediction": expected[
                    "used_in_candidate_prediction"
                ],
                "outer_fold": (
                    fold if source_id in FOLD_SCOPED_INPUT_SOURCES else None
                ),
                "seed": expected["seed"],
                "realized_parameter_count": expected["realized_parameter_count"],
                "trainable_in_p1": expected["trainable_in_p1"],
            }
        rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "seed": 0,
                "sources": sources,
            }
        )
    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "status": SOURCE_MANIFEST_STATUS,
        "contract_id": EXPECTED_CONTRACT_ID,
        "binding_status": SOURCE_BINDING_STATUS,
        "folds": rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def test_screen_merge_score_and_qualifier_bind_one_exact_authority_universe(
    tmp_path: Path,
) -> None:
    paths = {
        "prediction_dir": tmp_path / "screen",
        "complete_unscored_merge_path": tmp_path / "screen" / "merged.json",
        "tic2a_merged_registry_path": tmp_path / "tic2a.json",
        "v13_historical_bundle_path": tmp_path / "v13_score.json",
        "m2_csv_path": tmp_path / "m2.csv",
        "complete_score_path": tmp_path / "screen" / "score.json",
        "qualification_path": tmp_path / "screen" / "qualification.json",
    }
    _write_active(
        tmp_path,
        phase="P1M3",
        held_score=False,
        paths=paths,
        training_allowed=True,
    )
    assert_merge_authority(
        tmp_path,
        phase="P1M3",
        input_dir=paths["prediction_dir"],
        out_json=paths["complete_unscored_merge_path"],
    )
    with pytest.raises(RuntimeError, match="differs from active authority"):
        assert_merge_authority(
            tmp_path,
            phase="P1M3",
            input_dir=tmp_path / "other",
            out_json=paths["complete_unscored_merge_path"],
        )

    _write_active(
        tmp_path,
        phase="P1M3",
        held_score="PUZZLE_SET_COMPLETE_MERGE_SCORE_ONCE_ONLY",
        paths=paths,
        training_allowed=False,
    )
    assert_screen_score_authority(
        tmp_path,
        merged_json=paths["complete_unscored_merge_path"],
        tic2a_merged_json=paths["tic2a_merged_registry_path"],
        v13_historical_bundle=paths["v13_historical_bundle_path"],
        m2_csv=paths["m2_csv_path"],
        out_json=paths["complete_score_path"],
    )
    assert_screen_qualifier_authority(
        tmp_path,
        score_json=paths["complete_score_path"],
        out_json=paths["qualification_path"],
    )
    with pytest.raises(RuntimeError, match="differs from active authority"):
        assert_screen_qualifier_authority(
            tmp_path,
            score_json=paths["complete_score_path"],
            out_json=tmp_path / "other_qualification.json",
        )


def test_real_smoke_merge_is_bound_to_the_p1m2_authority_paths(
    tmp_path: Path,
) -> None:
    paths = {
        "prediction_dir": tmp_path / "p1m2_real_smoke",
        "complete_unscored_merge_path": (
            tmp_path / "p1m2_real_smoke" / "p1m2_complete_unscored_merge.json"
        ),
    }
    _write_active(
        tmp_path,
        phase="P1M2",
        held_score=False,
        paths=paths,
        training_allowed=True,
    )
    assert_merge_authority(
        tmp_path,
        phase="P1M2",
        input_dir=paths["prediction_dir"],
        out_json=paths["complete_unscored_merge_path"],
    )
    with pytest.raises(RuntimeError, match="differs from active authority"):
        assert_merge_authority(
            tmp_path,
            phase="P1M2",
            input_dir=paths["prediction_dir"],
            out_json=tmp_path / "unbound_smoke_merge.json",
        )


@pytest.mark.parametrize("manifest_state", ["pending", "missing", "invalid"])
def test_p1m2_merge_and_smoke_qualifier_require_valid_bound_source_manifest(
    tmp_path: Path, manifest_state: str
) -> None:
    paths = {
        "prediction_dir": tmp_path / "p1m2_real_smoke",
        "complete_unscored_merge_path": (
            tmp_path / "p1m2_real_smoke" / "p1m2_complete_unscored_merge.json"
        ),
        "qualification_path": (
            tmp_path / "p1m2_real_smoke" / "p1m2_qualification.json"
        ),
    }
    _write_active(
        tmp_path,
        phase="P1M2",
        held_score=False,
        paths=paths,
        training_allowed=True,
    )
    active_path = tmp_path / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text(encoding="utf-8"))
    manifest_path = Path(active["authority"]["source_manifest_path"])
    if manifest_state == "pending":
        active["authority"][
            "source_binding_status"
        ] = "REALIZED_PATHS_ROLES_AND_COUNTS_PENDING"
        active_path.write_text(yaml.safe_dump(active), encoding="utf-8")
        expected_error = RuntimeError
    elif manifest_state == "missing":
        manifest_path.unlink()
        expected_error = FileNotFoundError
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["binding_status"] = "INVALID"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        expected_error = RuntimeError

    with pytest.raises(expected_error):
        assert_merge_authority(
            tmp_path,
            phase="P1M2",
            input_dir=paths["prediction_dir"],
            out_json=paths["complete_unscored_merge_path"],
        )
    with pytest.raises(expected_error):
        assert_smoke_qualifier_authority(
            tmp_path,
            merged_json=paths["complete_unscored_merge_path"],
            out_json=paths["qualification_path"],
        )


def test_formal_assembly_score_and_qualifier_bind_screen_prerequisite(
    tmp_path: Path,
) -> None:
    paths = {
        "prediction_dir": tmp_path / "formal",
        "complete_unscored_merge_path": tmp_path / "formal" / "merged.json",
        "formal_assembly_prediction_dir": tmp_path / "formal" / "assembled",
        "formal_assembly_path": tmp_path / "formal" / "assembly.json",
        "tic2a_merged_registry_path": tmp_path / "tic2a.json",
        "v13_historical_bundle_path": tmp_path / "v13_score.json",
        "m2_csv_path": tmp_path / "m2.csv",
        "complete_score_path": tmp_path / "formal" / "score.json",
        "screen_qualification_path": tmp_path / "screen" / "qualification.json",
        "qualification_path": tmp_path / "formal" / "qualification.json",
    }
    _write_active(
        tmp_path,
        phase="P1M4",
        held_score=False,
        paths=paths,
        training_allowed=True,
    )
    assert_assembly_authority(
        tmp_path,
        merged_json=paths["complete_unscored_merge_path"],
        out_dir=paths["formal_assembly_prediction_dir"],
        out_json=paths["formal_assembly_path"],
    )
    with pytest.raises(RuntimeError, match="differs from active authority"):
        assert_assembly_authority(
            tmp_path,
            merged_json=paths["complete_unscored_merge_path"],
            out_dir=tmp_path / "unbound_assembled",
            out_json=paths["formal_assembly_path"],
        )

    _write_active(
        tmp_path,
        phase="P1M4",
        held_score="PUZZLE_SET_FORMAL_COMPLETE_SCORE_ONCE_ONLY",
        paths=paths,
        training_allowed=False,
    )
    assert_formal_score_authority(
        tmp_path,
        assembly_json=paths["formal_assembly_path"],
        merged_json=paths["complete_unscored_merge_path"],
        tic2a_merged_json=paths["tic2a_merged_registry_path"],
        v13_historical_bundle=paths["v13_historical_bundle_path"],
        m2_csv=paths["m2_csv_path"],
        out_json=paths["complete_score_path"],
    )
    assert_formal_qualifier_authority(
        tmp_path,
        score_json=paths["complete_score_path"],
        screen_qualification_json=paths["screen_qualification_path"],
        out_json=paths["qualification_path"],
    )
    with pytest.raises(RuntimeError, match="differs from active authority"):
        assert_formal_qualifier_authority(
            tmp_path,
            score_json=paths["complete_score_path"],
            screen_qualification_json=tmp_path / "unfrozen_screen.json",
            out_json=paths["qualification_path"],
        )
