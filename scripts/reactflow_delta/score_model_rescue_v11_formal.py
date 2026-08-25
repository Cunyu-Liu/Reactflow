#!/usr/bin/env python3
"""Score the frozen V11 five-seed mixture and each constituent seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.reactflow_delta.assemble_model_rescue_v11_formal import (
    FORMAL_PREDICTION_SCHEMA,
    SCHEMA as ASSEMBLY_SCHEMA,
)
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_model_rescue_v11 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.model_rescue_v11 import PREDICTION_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v10 import _load_tic2a_absolute
from scripts.reactflow_delta.score_model_rescue_v11 import (
    merged_integrity_pass,
    score_fold,
)
from scripts.reactflow_delta.score_model_rescue_v9 import TIC2A_MERGED_SCHEMA
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v11_formal_score.v1"


def assert_score_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V11M4":
        raise RuntimeError("V11 formal scorer is closed outside V11M4")
    if active.get("runnable_phases") != ["V11M4"]:
        raise RuntimeError("V11M4 must be the only runnable phase")
    if active.get("training_allowed") is not False:
        raise RuntimeError("V11M4 training must be closed before scoring")
    if active.get("held_score_read_allowed") is not True:
        raise RuntimeError("V11M4 complete score access is closed")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("V11M4 partial score access must remain closed")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("V11M4 requires external outcomes locked")


def _load(path: Path, schema: str, fold: int, seed: int | None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if str(prediction["schema_version"].item()) != schema:
        raise ValueError(f"invalid V11 formal source schema in {path}")
    if set(map(int, prediction["outer_fold"])) != {fold}:
        raise ValueError(f"V11 formal source fold mismatch in {path}")
    expected_seed = {-1} if seed is None else {seed}
    if set(map(int, prediction["seed"])) != expected_seed:
        raise ValueError(f"V11 formal source seed mismatch in {path}")
    return prediction


def score_formal(
    assembly: dict[str, Any],
    merged: dict[str, Any],
    tic2a: dict[str, Any],
    m2_csv: Path,
) -> dict[str, Any]:
    if assembly.get("schema_version") != ASSEMBLY_SCHEMA or assembly.get("status") != (
        "V11M4_FIVE_SEED_PREDICTION_ONLY_ASSEMBLY_PASS"
    ):
        raise ValueError("V11 formal scorer requires complete assembly")
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != (
        "V11M4_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("V11 formal scorer requires complete fold-seed merge")
    if not merged_integrity_pass(merged.get("merge_integrity", {})):
        raise ValueError("V11 formal merged integrity is not qualified")
    if not (
        assembly.get("equal_seed_mixture") is True
        and assembly.get("best_seed_selection_performed") is False
        and assembly.get("score_computed") is False
        and assembly.get("external_outcome_accessed") is False
    ):
        raise ValueError("V11 formal assembly violates the frozen score-blind protocol")
    if tic2a.get("schema_version") != TIC2A_MERGED_SCHEMA or tic2a.get("status") != (
        "TIC2A_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("V11 formal scorer requires corrected TIC2A merge")
    assembly_rows = {int(row["outer_fold"]): row for row in assembly["folds"]}
    source_rows = {
        (int(row["outer_fold"]), int(row["seed"])): row for row in merged["folds"]
    }
    tic_rows = {int(row["outer_fold"]): row for row in tic2a["folds"]}
    expected_sources = {(fold, seed) for fold in range(20) for seed in range(5)}
    if sorted(assembly_rows) != list(range(20)) or set(source_rows) != expected_sources:
        raise ValueError("V11 formal score universes are incomplete")

    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise RuntimeError("V11 formal scorer requires exact target identity")
    records = univ.get_records()
    split = build_split_v4(
        sorted({record.puzzle for record in records}), seed=20260813
    )
    folds = {int(fold.outer_fold): fold for fold in split["folds"]}
    mixture_scores = []
    individual_seed_scores = {str(seed): [] for seed in range(5)}
    for fold_id in range(20):
        fold = folds[fold_id]
        held_records = [
            record for record in records if record.puzzle == fold.held_puzzle
        ]
        absolute = _load_tic2a_absolute(
            Path(tic_rows[fold_id]["prediction_artifact"]), fold_id
        )
        mixture = score_fold(
            univ,
            held_records,
            _load(
                Path(assembly_rows[fold_id]["prediction_artifact"]),
                FORMAL_PREDICTION_SCHEMA,
                fold_id,
                None,
            ),
            absolute,
        )
        mixture.update({"outer_fold": fold_id, "held_puzzle": str(fold.held_puzzle)})
        mixture_scores.append(mixture)
        for seed in range(5):
            score = score_fold(
                univ,
                held_records,
                _load(
                    Path(source_rows[(fold_id, seed)]["prediction_artifact"]),
                    PREDICTION_SCHEMA,
                    fold_id,
                    seed,
                ),
                absolute,
            )
            score.update({"outer_fold": fold_id, "held_puzzle": str(fold.held_puzzle)})
            individual_seed_scores[str(seed)].append(score)
    return {
        "schema_version": SCHEMA,
        "phase": "V11M4",
        "status": "V11M4_COMPLETE_FORMAL_SCORE_PASS",
        "mixture_scores": mixture_scores,
        "individual_seed_scores": individual_seed_scores,
        "equal_seed_mixture": True,
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
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    assert_score_authority(args.repo_root.resolve())
    result = score_formal(
        json.loads(args.assembly_json.read_text(encoding="utf-8")),
        json.loads(args.merged_json.read_text(encoding="utf-8")),
        json.loads(args.tic2a_merged_json.read_text(encoding="utf-8")),
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
