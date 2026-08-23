#!/usr/bin/env python3
"""Join targets and score v4 only after the complete fold universe is merged."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_model_rescue_v4 import EXPECTED_MODELS, SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.run_model_rescue_m2_v1 import score_predictions
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.model_rescue_v4_complete_score.v1"
PUBLISHED = "task_matched_published_comparator"
REQUIRED_PREDICTION_FIELDS = {
    "keys",
    "biological_scoring_key",
    "candidate_id",
    "outer_fold",
    "seed",
    "delta_mean",
    "point_mean",
    "locations",
    "scales",
    "weights",
    "registered_status",
    "mean_checkpoint_path",
    "calibration_checkpoint_path",
}
FORBIDDEN_PREDICTION_FIELDS = {
    "target",
    "target_error",
    "target_mask",
    "qualified_target_mask",
    "score",
}


def assert_score_authority(repo_root: Path, phase: str) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != phase:
        raise RuntimeError(f"v4 scorer is closed outside active {phase}")
    if active.get("held_score_read_allowed") is not True:
        raise RuntimeError("complete v4 score access has not been opened")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("partial fold score access must remain prohibited")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("v4 complete scorer requires external outcomes locked")


def load_prediction(path: Path, model_id: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        prediction = {name: handle[name] for name in handle.files}
    if not REQUIRED_PREDICTION_FIELDS <= set(prediction):
        raise ValueError(f"prediction {path} is missing required v4 fields")
    if not FORBIDDEN_PREDICTION_FIELDS.isdisjoint(prediction):
        raise ValueError(f"prediction {path} contains target-side fields")
    keys = prediction["keys"]
    n = len(keys)
    if n == 0 or len(set(map(str, keys))) != n:
        raise ValueError(f"prediction {path} has an invalid key universe")
    if not np.array_equal(keys, prediction["biological_scoring_key"]):
        raise ValueError(f"prediction {path} key fields disagree")
    if set(map(str, prediction["candidate_id"])) != {model_id}:
        raise ValueError(f"prediction {path} candidate id is not {model_id}")
    for field in ("locations", "scales", "weights"):
        if prediction[field].shape != (n, 2):
            raise ValueError(f"prediction {path} {field} must contain two components")
    if not np.allclose(prediction["weights"].sum(-1), 1.0, atol=1e-7, rtol=0):
        raise ValueError(f"prediction {path} weights do not sum to one")
    if not np.all(prediction["scales"] > 0) or not np.isfinite(prediction["scales"]).all():
        raise ValueError(f"prediction {path} has invalid scales")
    if not np.array_equal(prediction["locations"][:, 0], prediction["point_mean"]):
        raise ValueError(f"prediction {path} first location changed point mean")
    if not np.array_equal(prediction["locations"][:, 1], prediction["point_mean"]):
        raise ValueError(f"prediction {path} second location changed point mean")
    return prediction


def combine_seed_predictions(
    paths: list[Path], model_id: str
) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError("at least one seed prediction is required")
    rows = [load_prediction(path, model_id) for path in paths]
    keys = rows[0]["keys"]
    if any(not np.array_equal(keys, row["keys"]) for row in rows[1:]):
        raise ValueError(f"{model_id} seed predictions have different key universes")
    seed_values = [int(np.unique(row["seed"])[0]) for row in rows]
    if len(set(seed_values)) != len(seed_values):
        raise ValueError(f"{model_id} seed predictions contain duplicate seeds")
    weight = 1.0 / len(rows)
    return {
        "keys": keys,
        "locations": np.concatenate([row["locations"] for row in rows], axis=1),
        "scales": np.concatenate([row["scales"] for row in rows], axis=1),
        "weights": np.concatenate([row["weights"] * weight for row in rows], axis=1),
        "point_mean": np.mean(
            np.stack([row["point_mean"] for row in rows], axis=1), axis=1
        ),
        "seed_universe": np.asarray(sorted(seed_values), dtype=np.int64),
    }


def score_complete_merged(
    merged: dict[str, Any], m2_csv: Path, published_scores: dict[str, Any] | None
) -> dict[str, Any]:
    if merged.get("schema_version") != MERGED_SCHEMA:
        raise ValueError("v4 scorer requires the complete merged schema")
    integrity = merged.get("merge_integrity", {})
    if integrity.get("complete_fold_seed_universe") is not True:
        raise ValueError("v4 scorer cannot access an incomplete fold universe")
    rows = merged["folds"]
    by_fold: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_fold.setdefault(int(row["outer_fold"]), []).append(row)
    if set(by_fold) != set(range(20)):
        raise ValueError("v4 merged artifact does not contain folds 0 through 19")

    universe = M2Universe(m2_csv)
    universe.build()
    records = universe.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    fold_map = {int(fold.outer_fold): fold for fold in split["folds"]}
    scores: dict[str, list[dict[str, Any]]] = {model: [] for model in EXPECTED_MODELS}
    for fold_id in range(20):
        fold = fold_map[fold_id]
        held_records = [record for record in records if record.puzzle == fold.held_puzzle]
        seed_rows = sorted(by_fold[fold_id], key=lambda row: int(row["seed"]))
        for model_id in sorted(EXPECTED_MODELS):
            prediction = combine_seed_predictions(
                [Path(row["models"][model_id]["prediction_artifact"]) for row in seed_rows],
                model_id,
            )
            score = score_predictions(prediction, universe, held_records)
            score["outer_fold"] = fold_id
            score["held_puzzle"] = fold.held_puzzle
            scores[model_id].append(score)
    if published_scores is not None:
        published = published_scores.get("scores")
        if not isinstance(published, list) or len(published) != 20:
            raise ValueError("task-matched published comparator requires 20 puzzle scores")
        if published_scores.get("task_matched") is not True:
            raise ValueError("published comparator is not qualified as task matched")
        scores[PUBLISHED] = published
    return {
        "schema_version": SCHEMA,
        "phase": merged["phase"],
        "evidence_status": "DEVELOPMENT_CONSUMED_COMPLETE_UNIVERSE_SCORED",
        "scores": scores,
        "target_join_after_complete_merge": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--published-score-json", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--phase", choices=["V4M3", "V4M4"], required=True)
    args = parser.parse_args(argv)
    assert_score_authority(args.repo_root.resolve(), args.phase)
    merged = json.loads(args.merged_json.read_text(encoding="utf-8"))
    if merged.get("phase") != args.phase:
        raise ValueError("merged artifact phase does not match scorer phase")
    published = (
        json.loads(args.published_score_json.read_text(encoding="utf-8"))
        if args.published_score_json
        else None
    )
    result = score_complete_merged(merged, args.m2_csv, published)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": f"{args.phase}_COMPLETE_SCORE_PASS", "result": str(args.out_json)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
