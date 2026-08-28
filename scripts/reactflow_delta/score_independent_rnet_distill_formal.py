#!/usr/bin/env python3
"""Score the complete fixed five-seed independent-RNet formal model once.

All one hundred source predictions and the twenty equal-seed assemblies are
validated without targets before the frozen V14 score or M2 outcomes are read.
The canonical score contains the equal-weight mixture and every constituent
seed; it does not qualify or select a seed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.reactflow_delta.assemble_independent_rnet_distill_formal import (
    ASSEMBLY_DIR,
    ASSEMBLY_PATH,
    ASSEMBLY_STATUS,
    EXPECTED_FORMAL_PREDICTION_FIELDS,
    FORMAL_DIR,
    FORMAL_PREDICTION_SCHEMA,
    MERGED_PATH,
    SCHEMA as ASSEMBLY_SCHEMA,
    _expected_absolute,
)
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_independent_rnet_distill import (
    EXPECTED_MERGED_FIELDS,
    MERGE_INTEGRITY,
    SCHEMA as MERGED_SCHEMA,
    STATUS as MERGED_STATUS,
)
from scripts.reactflow_delta.run_independent_rnet_distill_downstream import (
    EXPECTED_PREDICTION_FIELDS,
    FORBIDDEN_PREDICTION_FIELDS,
    PREDICTION_SCHEMA,
)
from scripts.reactflow_delta.score_independent_rnet_distill import (
    EXPECTED_FOLDS,
    MIXTURE_NAMES,
    M2_PATH,
    POINT_NAMES,
    SCORE_ROW_FIELDS,
    ScoreIntegrityError,
    V14_SCORE_PATH,
    _v14_parent_rows,
    score_fold,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4
from scripts.reactflow_delta.validate_independent_rnet_distill_contract import (
    PROJECT_TASK_ID,
    assert_run_authority,
)


FORMAL_SCORE_PHASE = "RND6S"
FORMAL_SCORE_SCHEMA = "reactflow_delta.independent_rnet_distill_formal_score.v1"
FORMAL_SCORE_STATUS = "RND6S_COMPLETE_FORMAL_SCORE_PASS"
FORMAL_SCORE_PATH = FORMAL_DIR / "rnet_distill_complete_formal_score.json"
FORMAL_QUALIFICATION_PATH = FORMAL_DIR / "rnet_distill_formal_qualification.json"
SCREEN_QUALIFICATION_PATH = (
    FORMAL_DIR.parent / "rnd3_screen_seed0" / "rnet_distill_qualification.json"
)
EXPECTED_SEEDS = tuple(range(5))
EXPECTED_FOLD_SEED_COUNT = len(EXPECTED_FOLDS) * len(EXPECTED_SEEDS)
EVIDENCE_STATUS = "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY"

EXPECTED_ASSEMBLY_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "folds",
    "equal_seed_mixture",
    "equal_seed_weight",
    "best_seed_selection_performed",
    "score_computed",
    "target_accessed",
    "external_outcome_accessed",
}
EXPECTED_ASSEMBLY_FOLD_FIELDS = {
    "outer_fold",
    "seeds",
    "prediction_artifact",
    "n_registered_prediction_rows",
    "candidate_components_per_distribution",
    "null_components_per_distribution",
    "feature41_components_per_distribution",
    "historical_v10_components_per_distribution",
}


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a mapping at {path}")
    return value


def assert_formal_score_authority(
    repo_root: Path,
    *,
    merged_json: Path,
    assembly_json: Path,
    m2_csv: Path,
    historical_v14_score_json: Path,
    out_json: Path,
) -> dict[str, Any]:
    """Require exact RND6S authority and canonical CLI paths."""

    assert_run_authority(repo_root, FORMAL_SCORE_PHASE)
    active = _load_yaml(repo_root / "configs/reactflow_delta/active_contract.yaml")
    if active.get("project_task_id") != PROJECT_TASK_ID:
        raise RuntimeError("formal scorer is not under the independent RNet project")
    authority = active.get("authority", {})
    expected = {
        "formal_prediction_dir": FORMAL_DIR,
        "formal_complete_unscored_merge_path": MERGED_PATH,
        "formal_assembly_dir": ASSEMBLY_DIR,
        "formal_assembly_path": ASSEMBLY_PATH,
        "m2_csv_path": M2_PATH,
        "historical_v14_score_path": V14_SCORE_PATH,
        "formal_complete_score_path": FORMAL_SCORE_PATH,
        "formal_qualification_path": FORMAL_QUALIFICATION_PATH,
        "screen_qualification_path": SCREEN_QUALIFICATION_PATH,
    }
    for name, path in expected.items():
        if name not in authority or _resolved(authority[name]) != path.resolve():
            raise RuntimeError(f"RND6S active authority {name} is not exact")
    provided = {
        "formal_complete_unscored_merge_path": merged_json,
        "formal_assembly_path": assembly_json,
        "m2_csv_path": m2_csv,
        "historical_v14_score_path": historical_v14_score_json,
        "formal_complete_score_path": out_json,
    }
    for name, path in provided.items():
        if _resolved(path) != expected[name].resolve():
            raise RuntimeError(f"RND6S CLI {name} differs from active authority")
    return active


def _read_npz(path: Path, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=True) as handle:
            fields = set(handle.files)
            if fields & FORBIDDEN_PREDICTION_FIELDS:
                raise ScoreIntegrityError(f"{label} contains forbidden target/score fields")
            return {name: np.asarray(handle[name]) for name in fields}
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, ScoreIntegrityError):
            raise
        raise ScoreIntegrityError(f"{label} cannot be read") from error


def _validate_prediction_arrays(
    prediction: dict[str, np.ndarray],
    *,
    fields: set[str] | frozenset[str],
    schema: str,
    fold: int,
    seed: int,
    source_components: bool,
) -> None:
    if set(prediction) != set(fields):
        raise ScoreIntegrityError(f"fold{fold} seed{seed} prediction fields changed")
    if str(prediction["schema_version"].item()) != schema:
        raise ScoreIntegrityError(f"fold{fold} seed{seed} prediction schema changed")
    keys = list(map(str, prediction["keys"]))
    n_rows = len(keys)
    if (
        n_rows < 1
        or len(keys) != len(set(keys))
        or prediction["biological_scoring_key"].shape != (n_rows,)
        or keys != list(map(str, prediction["biological_scoring_key"]))
        or prediction["outer_fold"].shape != (n_rows,)
        or set(map(int, prediction["outer_fold"])) != {fold}
        or prediction["seed"].shape != (n_rows,)
        or set(map(int, prediction["seed"])) != {seed}
        or prediction["registered_status"].shape != (n_rows,)
        or set(map(str, prediction["registered_status"])) != {"covered"}
    ):
        raise ScoreIntegrityError(f"fold{fold} seed{seed} registered identity changed")
    if seed == -1 and (
        prediction["assembled_seed_count"].shape != (n_rows,)
        or set(map(int, prediction["assembled_seed_count"])) != {5}
    ):
        raise ScoreIntegrityError(f"fold{fold} assembled seed count changed")
    for name in POINT_NAMES:
        point = prediction[f"{name}_point"]
        if point.shape != (n_rows,) or not np.isfinite(point).all():
            raise ScoreIntegrityError(f"fold{fold} seed{seed} {name} point is invalid")
    for name in MIXTURE_NAMES:
        weights = prediction[f"{name}_weights"]
        locations = prediction[f"{name}_locations"]
        scales = prediction[f"{name}_scales"]
        expected_absolute = prediction[f"{name}_expected_absolute_delta"]
        expected_components = 2 if source_components or name in {"feature41", "historical_v10"} else 10
        if (
            weights.shape != (n_rows, expected_components)
            or locations.shape != weights.shape
            or scales.shape != weights.shape
            or expected_absolute.shape != (n_rows,)
            or not np.isfinite(weights).all()
            or not np.isfinite(locations).all()
            or not np.isfinite(scales).all()
            or not np.isfinite(expected_absolute).all()
            or np.any(weights < 0.0)
            or np.any(scales <= 0.0)
            or np.any(expected_absolute < 0.0)
            or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-7, rtol=0.0)
        ):
            raise ScoreIntegrityError(
                f"fold{fold} seed{seed} {name} distribution is invalid"
            )


def _validated_merge_rows(merged: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    if (
        set(merged) != set(EXPECTED_MERGED_FIELDS)
        or merged.get("schema_version") != MERGED_SCHEMA
        or merged.get("phase") != "RND6P"
        or merged.get("status") != MERGED_STATUS["RND6P"]
        or merged.get("merge_integrity") != MERGE_INTEGRITY
    ):
        raise ScoreIntegrityError("RND6S requires the exact target-free RND6P merge")
    rows = merged.get("folds")
    if not isinstance(rows, list) or len(rows) != EXPECTED_FOLD_SEED_COUNT:
        raise ScoreIntegrityError("RND6S requires exactly 100 fold-seed rows")
    try:
        by_pair = {
            (int(row["outer_fold"]), int(row["seed"])): row for row in rows
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ScoreIntegrityError("RND6P merge rows are malformed") from error
    expected = {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
    if len(by_pair) != len(rows) or set(by_pair) != expected:
        raise ScoreIntegrityError("RND6P fold-seed universe is not exact")
    for (fold, seed), row in by_pair.items():
        if (
            row.get("phase") != "RND6P"
            or row.get("held_puzzle") != f"P{fold + 1:02d}"
            or int(row.get("point_epochs", -1)) != 40
            or int(row.get("calibration_epochs", -1)) != 40
        ):
            raise ScoreIntegrityError(f"RND6P fold{fold} seed{seed} identity changed")
    return by_pair


def _validated_assembly_rows(assembly: dict[str, Any]) -> dict[int, dict[str, Any]]:
    expected_top = {
        "schema_version": ASSEMBLY_SCHEMA,
        "phase": "RND6P",
        "status": ASSEMBLY_STATUS,
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "score_computed": False,
        "target_accessed": False,
        "external_outcome_accessed": False,
    }
    if set(assembly) != EXPECTED_ASSEMBLY_FIELDS or any(
        assembly.get(name) != value for name, value in expected_top.items()
    ):
        raise ScoreIntegrityError("RND6S requires the exact target-free assembly")
    rows = assembly.get("folds")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_FOLDS):
        raise ScoreIntegrityError("RND6S requires exactly twenty assembled folds")
    try:
        by_fold = {int(row["outer_fold"]): row for row in rows}
    except (KeyError, TypeError, ValueError) as error:
        raise ScoreIntegrityError("RND6P assembly rows are malformed") from error
    if len(by_fold) != len(rows) or sorted(by_fold) != EXPECTED_FOLDS:
        raise ScoreIntegrityError("RND6P assembly fold universe is not exact")
    for fold, row in by_fold.items():
        if (
            set(row) != EXPECTED_ASSEMBLY_FOLD_FIELDS
            or row.get("seeds") != list(EXPECTED_SEEDS)
            or int(row.get("n_registered_prediction_rows", 0)) < 1
            or int(row.get("candidate_components_per_distribution", -1)) != 10
            or int(row.get("null_components_per_distribution", -1)) != 10
            or int(row.get("feature41_components_per_distribution", -1)) != 2
            or int(row.get("historical_v10_components_per_distribution", -1)) != 2
        ):
            raise ScoreIntegrityError(f"RND6P assembly fold{fold} metadata changed")
    return by_fold


def _validate_assembly_against_sources(
    *,
    fold: int,
    sources: list[dict[str, np.ndarray]],
    assembled: dict[str, np.ndarray],
) -> None:
    """Prove the scored payload is the frozen equal-seed assembly."""

    if len(sources) != len(EXPECTED_SEEDS):
        raise ScoreIntegrityError(f"RND6P assembly fold{fold} source count changed")
    reference = sources[0]
    fixed_fields = (
        "feature41_point",
        "v8_point",
        *(f"feature41_{suffix}" for suffix in ("weights", "locations", "scales", "expected_absolute_delta")),
        *(f"historical_v10_{suffix}" for suffix in ("weights", "locations", "scales", "expected_absolute_delta")),
    )
    for field in fixed_fields:
        if any(not np.array_equal(source[field], reference[field]) for source in sources[1:]):
            raise ScoreIntegrityError(
                f"RND6P assembly fold{fold} fixed source {field} differs by seed"
            )
        if not np.array_equal(assembled[field], reference[field]):
            raise ScoreIntegrityError(
                f"RND6P assembly fold{fold} fixed field {field} differs from sources"
            )

    for name in ("candidate", "null"):
        expected_point = np.mean(
            [np.asarray(source[f"{name}_point"], dtype=np.float64) for source in sources],
            axis=0,
        )
        expected_weights = np.concatenate(
            [
                np.asarray(source[f"{name}_weights"], dtype=np.float64)
                / len(EXPECTED_SEEDS)
                for source in sources
            ],
            axis=1,
        )
        expected_locations = np.concatenate(
            [np.asarray(source[f"{name}_locations"], dtype=np.float64) for source in sources],
            axis=1,
        )
        expected_scales = np.concatenate(
            [np.asarray(source[f"{name}_scales"], dtype=np.float64) for source in sources],
            axis=1,
        )
        expected_absolute = _expected_absolute(
            expected_weights, expected_locations, expected_scales
        )
        expected = {
            f"{name}_point": expected_point,
            f"{name}_weights": expected_weights,
            f"{name}_locations": expected_locations,
            f"{name}_scales": expected_scales,
            f"{name}_expected_absolute_delta": expected_absolute,
        }
        for field, values in expected.items():
            if not np.array_equal(assembled[field], values):
                raise ScoreIntegrityError(
                    f"RND6P assembly fold{fold} {field} is not the exact equal-seed value"
                )


def load_complete_formal_predictions(
    merged: dict[str, Any], assembly: dict[str, Any]
) -> tuple[
    dict[tuple[int, int], dict[str, np.ndarray]],
    dict[int, dict[str, np.ndarray]],
]:
    """Validate every target-free source and assembly before outcome access."""

    merge_rows = _validated_merge_rows(merged)
    assembly_rows = _validated_assembly_rows(assembly)
    source_predictions: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for fold in EXPECTED_FOLDS:
        reference_keys: list[str] | None = None
        for seed in EXPECTED_SEEDS:
            row = merge_rows[(fold, seed)]
            path = Path(str(row.get("prediction_artifact", ""))).resolve()
            expected_path = FORMAL_DIR / f"rnet_distill_predictions_fold{fold}_seed{seed}.npz"
            if path != expected_path.resolve():
                raise ScoreIntegrityError(f"RND6P fold{fold} seed{seed} path is not canonical")
            prediction = _read_npz(path, label=f"RND6P fold{fold} seed{seed}")
            _validate_prediction_arrays(
                prediction,
                fields=EXPECTED_PREDICTION_FIELDS,
                schema=PREDICTION_SCHEMA,
                fold=fold,
                seed=seed,
                source_components=True,
            )
            keys = list(map(str, prediction["keys"]))
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise ScoreIntegrityError(f"RND6P fold{fold} key order differs by seed")
            source_predictions[(fold, seed)] = prediction

    assembled_predictions: dict[int, dict[str, np.ndarray]] = {}
    for fold in EXPECTED_FOLDS:
        row = assembly_rows[fold]
        path = Path(str(row.get("prediction_artifact", ""))).resolve()
        expected_path = ASSEMBLY_DIR / f"rnet_distill_formal_predictions_fold{fold}_seeds0_4.npz"
        if path != expected_path.resolve():
            raise ScoreIntegrityError(f"RND6P assembly fold{fold} path is not canonical")
        prediction = _read_npz(path, label=f"RND6P assembly fold{fold}")
        _validate_prediction_arrays(
            prediction,
            fields=EXPECTED_FORMAL_PREDICTION_FIELDS,
            schema=FORMAL_PREDICTION_SCHEMA,
            fold=fold,
            seed=-1,
            source_components=False,
        )
        if list(map(str, prediction["keys"])) != list(
            map(str, source_predictions[(fold, 0)]["keys"])
        ):
            raise ScoreIntegrityError(f"RND6P assembly fold{fold} keys differ from sources")
        _validate_assembly_against_sources(
            fold=fold,
            sources=[source_predictions[(fold, seed)] for seed in EXPECTED_SEEDS],
            assembled=prediction,
        )
        assembled_predictions[fold] = prediction
    return source_predictions, assembled_predictions


def _score_prediction(
    *,
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
    parent: dict[str, Any],
    fold: int,
    held_puzzle: str,
) -> dict[str, Any]:
    row = score_fold(univ, held_records, prediction)
    row.update(
        {
            "outer_fold": fold,
            "held_puzzle": held_puzzle,
            "historical_v14_signed_delta_mae": float(
                parent["candidate_signed_delta_mae"]
            ),
            "historical_v14_point_absolute_delta_mae": float(
                parent["candidate_point_absolute_delta_mae"]
            ),
            "historical_v14_crps": float(parent["candidate_crps"]),
        }
    )
    if set(row) != SCORE_ROW_FIELDS:
        raise ScoreIntegrityError(f"RND6S fold{fold} score schema is incomplete")
    return row


def score_formal(
    *,
    historical_v14_score: dict[str, Any],
    m2_csv: Path,
    source_predictions: dict[tuple[int, int], dict[str, np.ndarray]],
    assembled_predictions: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Score already target-free-validated predictions."""

    expected_pairs = {
        (fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS
    }
    if set(source_predictions) != expected_pairs or sorted(assembled_predictions) != EXPECTED_FOLDS:
        raise ScoreIntegrityError("RND6S validated prediction universe changed")
    parent_rows = _v14_parent_rows(historical_v14_score)
    univ = M2Universe(m2_csv)
    identity = univ.build()
    if (
        identity.get("n_canonical_mutant_full_profiles") != 13976
        or identity.get("canonical_mutant_full_profile_identity")
        != "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise ScoreIntegrityError("RND6S requires the exact canonical M2 target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = {int(row.outer_fold): row for row in split["folds"]}
    if sorted(folds) != EXPECTED_FOLDS:
        raise ScoreIntegrityError("RND6S split_v4 fold universe changed")

    mixture_scores: list[dict[str, Any]] = []
    individual_seed_scores: dict[str, list[dict[str, Any]]] = {
        str(seed): [] for seed in EXPECTED_SEEDS
    }
    for fold in EXPECTED_FOLDS:
        held_puzzle = str(folds[fold].held_puzzle)
        parent = parent_rows[fold]
        if held_puzzle != str(parent.get("held_puzzle")):
            raise ScoreIntegrityError(f"RND6S/V14 held puzzle differs at fold{fold}")
        held_records = [record for record in records if record.puzzle == held_puzzle]
        mixture_scores.append(
            _score_prediction(
                univ=univ,
                held_records=held_records,
                prediction=assembled_predictions[fold],
                parent=parent,
                fold=fold,
                held_puzzle=held_puzzle,
            )
        )
        for seed in EXPECTED_SEEDS:
            individual_seed_scores[str(seed)].append(
                _score_prediction(
                    univ=univ,
                    held_records=held_records,
                    prediction=source_predictions[(fold, seed)],
                    parent=parent,
                    fold=fold,
                    held_puzzle=held_puzzle,
                )
            )

    return {
        "schema_version": FORMAL_SCORE_SCHEMA,
        "phase": FORMAL_SCORE_PHASE,
        "status": FORMAL_SCORE_STATUS,
        "mixture_scores": mixture_scores,
        "individual_seed_scores": individual_seed_scores,
        "integrity_errors": [],
        "complete_valid_score": True,
        "complete_source_fold_seed_universe": True,
        "complete_assembly_fold_universe": True,
        "expected_fold_seed_count": EXPECTED_FOLD_SEED_COUNT,
        "actual_fold_seed_count": EXPECTED_FOLD_SEED_COUNT,
        "expected_fold_count": 20,
        "actual_fold_count": 20,
        "expected_seed_count": 5,
        "actual_seed_count": 5,
        "failed_rows": 0,
        "duplicate_or_unexpected_artifacts": 0,
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge_and_assembly": True,
        "aggregation": "POSITION_TO_MUTANT_TO_METHOD_TO_PUZZLE",
        "independent_units": "20_PUZZLES_NOT_100_FOLD_SEEDS",
        "attribution_null": "RNET2_SHIFT17_SINGLE_FEATURE_DISTILLATION",
        "feature41_comparator": "AUTHORITATIVE_FEATURE41_SEED0_REPLAY_FIXED_ACROSS_SEEDS",
        "historical_parent_source": "FROZEN_V14_CANONICAL_COMPLETE_SCORE",
        "historical_distribution_comparator": (
            "FROZEN_V10_COMPARATOR_FIXED_ACROSS_SEEDS"
        ),
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "partial_seed_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "source_exposure_status": EVIDENCE_STATUS,
    }


def _write_json_once(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("RND6S refuses to overwrite its canonical formal score")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--assembly-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--historical-v14-score-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    merged_json = args.merged_json.resolve()
    assembly_json = args.assembly_json.resolve()
    m2_csv = args.m2_csv.resolve()
    historical_v14_score_json = args.historical_v14_score_json.resolve()
    out_json = args.out_json.resolve()
    try:
        assert_formal_score_authority(
            repo_root,
            merged_json=merged_json,
            assembly_json=assembly_json,
            m2_csv=m2_csv,
            historical_v14_score_json=historical_v14_score_json,
            out_json=out_json,
        )
        if out_json.exists() or FORMAL_QUALIFICATION_PATH.exists():
            raise FileExistsError(
                "RND6S formal score or downstream qualification exists; refusing rerun"
            )
        merged = json.loads(merged_json.read_text(encoding="utf-8"))
        assembly = json.loads(assembly_json.read_text(encoding="utf-8"))
        source_predictions, assembled_predictions = load_complete_formal_predictions(
            merged, assembly
        )
        # The historical score and M2 outcomes are opened only after the entire
        # source and assembled target-free universe has passed above.
        historical_v14_score = json.loads(
            historical_v14_score_json.read_text(encoding="utf-8")
        )
        result = score_formal(
            historical_v14_score=historical_v14_score,
            m2_csv=m2_csv,
            source_predictions=source_predictions,
            assembled_predictions=assembled_predictions,
        )
        _write_json_once(out_json, result)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {
                    "status": "RND6S_COMPLETE_FORMAL_SCORE_ENGINEERING_INDETERMINATE",
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
