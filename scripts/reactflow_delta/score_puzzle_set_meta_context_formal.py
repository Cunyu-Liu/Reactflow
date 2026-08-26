#!/usr/bin/env python3
"""Score the frozen Puzzle-Set five-seed mixture and all constituent seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.reactflow_delta.assemble_puzzle_set_meta_context_formal import (
    ASSEMBLY_STATUS,
    EXPECTED_SEEDS,
    FORMAL_PREDICTION_FIELDS,
    FORMAL_PREDICTION_SCHEMA,
    SCHEMA as ASSEMBLY_SCHEMA,
    assemble_fold_prediction_arrays,
)
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_puzzle_set_meta_context_probe import (
    MERGED_SCHEMA,
    PREDICTION_FIELDS,
)
from scripts.reactflow_delta.puzzle_set_meta_context_data import PREDICTION_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v10 import _load_tic2a_absolute
from scripts.reactflow_delta.score_model_rescue_v9 import TIC2A_MERGED_SCHEMA
from scripts.reactflow_delta.score_puzzle_set_meta_context import (
    _assert_parent_and_baseline_replay,
    _v13_reference_rows,
    merged_integrity_pass,
    score_fold,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.puzzle_set_meta_context_formal_score.proposed.v2"
EXPECTED_PROJECT_TASK = "reactflow_delta_puzzle_set_meta_context"
EXPECTED_PHASE = "P1M4"
EXPECTED_SCORE_TOKEN = "PUZZLE_SET_FORMAL_COMPLETE_SCORE_ONCE_ONLY"


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active.get("project_task_id") != EXPECTED_PROJECT_TASK:
        raise RuntimeError("puzzle-set formal scorer is not the active project task")
    authority = active.get("authority", {})
    if authority.get("current_phase") != EXPECTED_PHASE or active.get(
        "runnable_phases"
    ) != [EXPECTED_PHASE]:
        raise RuntimeError("puzzle-set formal scorer is closed outside P1M4")
    if (
        active.get("training_allowed") is not False
        or active.get("candidate_model_training_allowed") is not False
    ):
        raise RuntimeError("puzzle-set formal training must be closed before scoring")
    if active.get("held_score_read_allowed") != EXPECTED_SCORE_TOKEN:
        raise RuntimeError("puzzle-set formal score-once authority is closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("puzzle-set formal partial score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError(
            "puzzle-set formal scoring requires external outcomes locked"
        )


def _load_prediction(
    path: Path,
    *,
    schema: str,
    fold: int,
    seed: int | None,
    expected_fields: set[str] | None = None,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        names = set(handle.files)
        if expected_fields is not None and names != expected_fields:
            raise ValueError(
                f"puzzle-set formal source field universe mismatch in {path}"
            )
        prediction = {name: np.asarray(handle[name]).copy() for name in handle.files}
    if str(prediction.get("schema_version", np.asarray("")).item()) != schema:
        raise ValueError(f"invalid puzzle-set formal prediction schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold}:
        raise ValueError(f"puzzle-set formal source fold mismatch in {path}")
    expected_seed = {-1} if seed is None else {seed}
    if set(map(int, prediction["seed"])) != expected_seed:
        raise ValueError(f"puzzle-set formal source seed mismatch in {path}")
    keys = list(map(str, prediction["keys"]))
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate puzzle-set formal source keys in {path}")
    return prediction


def _assert_exact_formal_reconstruction(
    observed: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    *,
    fold: int,
) -> None:
    if set(observed) != set(expected):
        raise ValueError(
            f"formal fold {fold} does not exactly derive from merged source artifacts: "
            "field universe differs"
        )
    for name in sorted(expected):
        actual_array = np.asarray(observed[name])
        expected_array = np.asarray(expected[name])
        if (
            actual_array.shape != expected_array.shape
            or actual_array.dtype != expected_array.dtype
            or not np.array_equal(actual_array, expected_array)
        ):
            raise ValueError(
                f"formal fold {fold} does not exactly derive from merged source "
                f"artifacts at {name}"
            )


def _reconstruct_and_validate_formal_predictions(
    assembly_rows: dict[int, dict[str, Any]],
    source_rows: dict[tuple[int, int], dict[str, Any]],
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    dict[tuple[int, int], dict[str, np.ndarray]],
]:
    mixtures: dict[int, dict[str, np.ndarray]] = {}
    sources_by_pair: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for fold in range(20):
        ordered_sources = []
        for seed in EXPECTED_SEEDS:
            source_row = source_rows[(fold, seed)]
            source = _load_prediction(
                Path(source_row["prediction_artifact"]),
                schema=PREDICTION_SCHEMA,
                fold=fold,
                seed=seed,
                expected_fields=PREDICTION_FIELDS,
            )
            if int(source_row.get("n_registered_prediction_rows", -1)) != len(
                source["keys"]
            ):
                raise ValueError(
                    f"formal fold {fold} seed{seed} merged source row count differs"
                )
            sources_by_pair[(fold, seed)] = source
            ordered_sources.append(source)
        reconstructed = assemble_fold_prediction_arrays(ordered_sources, fold=fold)
        assembly_row = assembly_rows[fold]
        observed = _load_prediction(
            Path(assembly_row["prediction_artifact"]),
            schema=FORMAL_PREDICTION_SCHEMA,
            fold=fold,
            seed=None,
            expected_fields=FORMAL_PREDICTION_FIELDS,
        )
        expected_metadata = {
            "seeds": list(EXPECTED_SEEDS),
            "n_registered_prediction_rows": len(reconstructed["keys"]),
            "components_per_seed": 2,
            "components_per_distribution": 10,
            "equal_seed_weight": 0.2,
            "parent_point_policy": "fixed_seed0_parent_not_averaged",
            "candidate_null_point_policy": "five_seed_arithmetic_mean",
        }
        if any(
            assembly_row.get(name) != value for name, value in expected_metadata.items()
        ):
            raise ValueError(
                f"formal fold {fold} does not exactly derive from merged source "
                "artifacts: assembly metadata differs"
            )
        _assert_exact_formal_reconstruction(observed, reconstructed, fold=fold)
        mixtures[fold] = observed
    return mixtures, sources_by_pair


def _add_frozen_references(observed: dict[str, Any], reference: dict[str, Any]) -> None:
    _assert_parent_and_baseline_replay(observed, reference)
    observed.update(
        {
            "feature41_crps": float(reference["feature41_crps"]),
            "feature41_coverage68": float(reference["feature41_coverage68"]),
            "feature41_coverage95": float(reference["feature41_coverage95"]),
            "historical_v13_signed_delta_mae": float(
                reference["candidate_signed_delta_mae"]
            ),
            "historical_v13_point_absolute_delta_mae": float(
                reference["candidate_point_absolute_delta_mae"]
            ),
            "terminal_v12_signed_delta_mae": float(
                reference["terminal_v12_signed_delta_mae"]
            ),
            "terminal_v11_point_absolute_delta_mae": float(
                reference["terminal_v11_point_absolute_delta_mae"]
            ),
            "terminal_v12_crps": float(reference["terminal_v12_crps"]),
            "terminal_v10_distribution_absolute_delta_mae": float(
                reference["terminal_v10_distribution_absolute_delta_mae"]
            ),
        }
    )


def score_formal(
    assembly: dict[str, Any],
    merged: dict[str, Any],
    tic2a_merged: dict[str, Any],
    v13_score: dict[str, Any],
    m2_csv: Path,
) -> dict[str, Any]:
    if (
        assembly.get("schema_version") != ASSEMBLY_SCHEMA
        or assembly.get("status") != ASSEMBLY_STATUS
        or assembly.get("phase") != EXPECTED_PHASE
    ):
        raise ValueError("puzzle-set formal scorer requires complete assembly")
    if (
        merged.get("schema_version") != MERGED_SCHEMA
        or merged.get("status") != "PUZZLE_SET_COMPLETE_UNSCORED_MERGE_PASS"
        or merged.get("phase") != EXPECTED_PHASE
        or merged.get("expected_folds") != list(range(20))
        or merged.get("expected_seeds") != list(range(5))
        or int(merged.get("expected_pretraining_epochs", -1)) != 200
        or int(merged.get("expected_point_epochs", -1)) != 40
        or int(merged.get("expected_calibration_epochs", -1)) != 40
    ):
        raise ValueError("puzzle-set formal scorer requires complete 100-run merge")
    if not merged_integrity_pass(merged.get("merge_integrity", {})):
        raise ValueError("puzzle-set formal merged integrity is not qualified")
    if not (
        assembly.get("equal_seed_mixture") is True
        and int(assembly.get("source_run_count", -1)) == 100
        and assembly.get("best_seed_selection_performed") is False
        and assembly.get("score_computed") is False
        and assembly.get("partial_scores_inspected") is False
        and assembly.get("external_outcome_accessed") is False
    ):
        raise ValueError("puzzle-set formal assembly violates score blindness")
    if (
        tic2a_merged.get("schema_version") != TIC2A_MERGED_SCHEMA
        or tic2a_merged.get("status") != "TIC2A_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("puzzle-set formal scorer requires corrected TIC2A merge")

    assembly_fold_rows = assembly.get("folds", [])
    source_fold_rows = merged.get("folds", [])
    tic_fold_rows = tic2a_merged.get("folds", [])
    if not all(
        isinstance(rows, list)
        for rows in (assembly_fold_rows, source_fold_rows, tic_fold_rows)
    ):
        raise ValueError("puzzle-set formal score universes are malformed")
    assembly_rows = {int(row["outer_fold"]): row for row in assembly_fold_rows}
    source_rows = {
        (int(row["outer_fold"]), int(row["seed"])): row for row in source_fold_rows
    }
    tic_rows = {int(row["outer_fold"]): row for row in tic_fold_rows}
    expected = {(fold, seed) for fold in range(20) for seed in range(5)}
    if (
        len(assembly_fold_rows) != 20
        or sorted(assembly_rows) != list(range(20))
        or len(source_fold_rows) != 100
        or set(source_rows) != expected
        or len(tic_fold_rows) != 20
        or sorted(tic_rows) != list(range(20))
    ):
        raise ValueError("puzzle-set formal score universes are incomplete")

    mixture_predictions, source_predictions = (
        _reconstruct_and_validate_formal_predictions(assembly_rows, source_rows)
    )

    reference_rows = _v13_reference_rows(v13_score)
    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("puzzle-set formal scorer requires exact target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = {int(fold.outer_fold): fold for fold in split["folds"]}

    mixture_scores: list[dict[str, Any]] = []
    individual_seed_scores: dict[str, list[dict[str, Any]]] = {
        str(seed): [] for seed in range(5)
    }
    for fold_id in range(20):
        fold = folds[fold_id]
        reference = reference_rows[fold_id]
        if str(reference.get("held_puzzle")) != str(fold.held_puzzle):
            raise ValueError("puzzle-set formal/V13 held puzzle alignment differs")
        held_records = [
            record for record in records if record.puzzle == fold.held_puzzle
        ]
        absolute = _load_tic2a_absolute(
            Path(tic_rows[fold_id]["prediction_artifact"]), fold_id
        )
        mixture = score_fold(
            univ,
            held_records,
            mixture_predictions[fold_id],
            absolute,
        )
        _add_frozen_references(mixture, reference)
        mixture.update({"outer_fold": fold_id, "held_puzzle": str(fold.held_puzzle)})
        mixture_scores.append(mixture)

        for seed in range(5):
            score = score_fold(
                univ,
                held_records,
                source_predictions[(fold_id, seed)],
                absolute,
            )
            _add_frozen_references(score, reference)
            score.update({"outer_fold": fold_id, "held_puzzle": str(fold.held_puzzle)})
            individual_seed_scores[str(seed)].append(score)

    return {
        "schema_version": SCHEMA,
        "phase": EXPECTED_PHASE,
        "status": "PUZZLE_SET_M4_COMPLETE_FORMAL_SCORE_PASS",
        "mixture_scores": mixture_scores,
        "individual_seed_scores": individual_seed_scores,
        "context_retention_summary": merged["context_retention_summary"],
        "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
        "target_join_after_complete_merge": True,
        "v13_parent_and_feature41_replay_at_5e_7": True,
        "feature41_reference_fixed_across_seeds": True,
        "formal_assembly_reconstructed_exactly_from_merged_sources": True,
        "equal_seed_mixture": True,
        "best_seed_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--assembly-json", type=Path, required=True)
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--v13-score-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_score_authority(args.repo_root.resolve())
    if args.out_json.exists():
        raise FileExistsError("puzzle-set refuses to overwrite its one formal score")
    result = score_formal(
        json.loads(args.assembly_json.read_text(encoding="utf-8")),
        json.loads(args.merged_json.read_text(encoding="utf-8")),
        json.loads(args.tic2a_merged_json.read_text(encoding="utf-8")),
        json.loads(args.v13_score_json.read_text(encoding="utf-8")),
        args.m2_csv,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "result": str(args.out_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
