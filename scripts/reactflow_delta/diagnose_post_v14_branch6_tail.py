#!/usr/bin/env python3
"""Run the pre-score-frozen post-V14 branch-6 tail diagnostic once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.special import ndtr

from scripts.reactflow_delta.diagnose_model_rescue_v11 import (
    directional_summary,
    method_balanced_weights,
)
from scripts.reactflow_delta.m2_universe_v1 import M2Universe
from scripts.reactflow_delta.merge_model_rescue_v14 import SCHEMA as MERGED_SCHEMA
from scripts.reactflow_delta.route_post_v14_model_contingency import (
    SCHEMA as ROUTER_SCHEMA,
    STATUS as ROUTER_STATUS,
    _assert_bound_paths,
    _load_canonical_active_contract,
    route,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key
from scripts.reactflow_delta.score_model_rescue_v14 import (
    _load_prediction,
    merged_integrity_pass,
)
from scripts.reactflow_delta.split_v4_lopo_puzzle import build_split_v4


SCHEMA = "reactflow_delta.post_v14_branch6_tail_diagnostic.v1"
AUTHORITY_TOKEN = "POST_V14_BRANCH6_TAIL_DIAGNOSTIC_ONCE_ONLY"
AUTHORITY_ACTION = "RUN_SINGLE_POST_V14_BRANCH6_TAIL_DIAGNOSTIC"
AUTHORITY_MAPPING = "post_v14_branch6_diagnostic_authority"
PRIMARY_STATISTIC = "LOWER_MINUS_UPPER_TAIL_MISS90"


def assert_diagnostic_authority(
    active_contract: Path,
    *,
    merged_path: Path,
    score_path: Path,
    qualification_path: Path,
    router_path: Path,
    m2_csv_path: Path,
    output_path: Path,
) -> None:
    active = _load_canonical_active_contract(active_contract)
    if active.get("authority", {}).get("current_phase") != "V14M3":
        raise RuntimeError("post-V14 branch-6 diagnostic requires terminal V14M3")
    if active.get("training_allowed") is not False or active.get(
        "candidate_model_training_allowed"
    ) is not False:
        raise RuntimeError("post-V14 branch-6 diagnostic requires training closed")
    if active.get("held_score_read_allowed") != AUTHORITY_TOKEN:
        raise RuntimeError("post-V14 branch-6 diagnostic token is not issued")
    if active.get("partial_fold_score_read_allowed") is not False:
        raise RuntimeError("partial score access remains prohibited")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("external outcome access remains prohibited")
    if active.get("next_allowed_action") != AUTHORITY_ACTION:
        raise RuntimeError("post-V14 branch-6 diagnostic action is not bound")
    _assert_bound_paths(
        active.get(AUTHORITY_MAPPING),
        mapping_name=AUTHORITY_MAPPING,
        authority_token=AUTHORITY_TOKEN,
        actual_paths={
            "complete_unscored_merge_path": merged_path,
            "complete_score_path": score_path,
            "qualification_path": qualification_path,
            "router_path": router_path,
            "m2_csv_path": m2_csv_path,
            "diagnostic_output_path": output_path,
        },
    )


def candidate_tail_difference(
    *,
    target: np.ndarray,
    mixture_weights: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
    methods: np.ndarray,
    mutants: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    mixture_weights = np.asarray(mixture_weights, dtype=np.float64)
    locations = np.asarray(locations, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if target.ndim != 1 or mixture_weights.ndim != 2:
        raise ValueError("branch-6 diagnostic arrays have invalid dimensions")
    if locations.shape != mixture_weights.shape or scales.shape != mixture_weights.shape:
        raise ValueError("branch-6 mixture arrays are misaligned")
    if mixture_weights.shape[0] != len(target):
        raise ValueError("branch-6 target and mixture rows are misaligned")
    if not all(
        np.isfinite(array).all()
        for array in (target, mixture_weights, locations, scales)
    ):
        raise ValueError("branch-6 diagnostic requires finite arrays")
    if (mixture_weights < 0.0).any() or not np.allclose(
        mixture_weights.sum(axis=1), 1.0, atol=1e-7, rtol=0.0
    ):
        raise ValueError("branch-6 mixture weights are invalid")
    if (scales <= 0.0).any():
        raise ValueError("branch-6 mixture scales must be positive")
    balanced = method_balanced_weights(methods, mutants)
    cdf = np.sum(
        mixture_weights * ndtr((target[:, None] - locations) / scales), axis=1
    )
    lower = float(np.sum(balanced * (cdf < 0.05)))
    upper = float(np.sum(balanced * (cdf > 0.95)))
    return {
        "lower_tail_miss90": lower,
        "upper_tail_miss90": upper,
        "lower_minus_upper_tail_miss90": lower - upper,
    }


def summarize_tail_differences(values: list[float]) -> dict[str, Any]:
    if len(values) != 20 or not all(_is_finite_number(value) for value in values):
        raise ValueError(
            "branch-6 diagnostic requires exactly twenty finite puzzle statistics"
        )
    summary = directional_summary(values)
    ci95 = summary["ci95"]
    same_direction_count_passed = bool(
        (float(ci95[0]) > 0.0 and int(summary["positive_puzzles"]) >= 14)
        or (float(ci95[1]) < 0.0 and int(summary["negative_puzzles"]) >= 14)
    )
    eligible = bool(
        summary.get("confirmatory") is True
        and same_direction_count_passed
    )
    return {
        **summary,
        "stable_nonzero_direction": eligible,
        "primary_statistic": PRIMARY_STATISTIC,
        "ci95_excludes_zero_and_same_direction_puzzles_ge_14": eligible,
        "p2_route_eligible": eligible,
    }


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and bool(np.isfinite(value))


def _fold_tail_diagnostic(
    univ: M2Universe,
    held_records: list[Any],
    prediction: dict[str, np.ndarray],
) -> dict[str, float]:
    index = {str(key): row for row, key in enumerate(prediction["keys"])}
    expected = {
        _bio_key(univ, record, position)
        for record in held_records
        for position in range(len(univ.get_construct(record.construct_id).sequence))
    }
    if set(index) != expected:
        raise ValueError("branch-6 prediction key universe is not exact")
    methods: list[str] = []
    mutants: list[str] = []
    targets: list[float] = []
    weights: list[np.ndarray] = []
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for record in held_records:
        construct = univ.get_construct(record.construct_id)
        target, _error = univ.mutant_full_profile(
            record.wt_id, record.design_pos, record.ref, record.alt
        )
        if target is None:
            continue
        positions = np.flatnonzero(construct.wt_observed & np.isfinite(target))
        mutant = f"{record.method}|{record.mutation_key}"
        for position in positions:
            row = index[_bio_key(univ, record, int(position))]
            methods.append(str(record.method))
            mutants.append(mutant)
            targets.append(float(target[position] - construct.wt_reactivity[position]))
            weights.append(
                np.asarray(prediction["candidate_weights"][row], dtype=np.float64)
            )
            locations.append(
                np.asarray(prediction["candidate_locations"][row], dtype=np.float64)
            )
            scales.append(
                np.asarray(prediction["candidate_scales"][row], dtype=np.float64)
            )
    return candidate_tail_difference(
        target=np.asarray(targets, dtype=np.float64),
        mixture_weights=np.stack(weights),
        locations=np.stack(locations),
        scales=np.stack(scales),
        methods=np.asarray(methods, dtype=object),
        mutants=np.asarray(mutants, dtype=object),
    )


def diagnose(
    *,
    merged: dict[str, Any],
    score: dict[str, Any],
    qualification: dict[str, Any],
    router: dict[str, Any],
    m2_csv: Path,
    merged_path: str,
    score_path: str,
    qualification_path: str,
    router_path: str,
) -> dict[str, Any]:
    expected_route = route(
        score,
        qualification,
        score_path=score_path,
        qualification_path=qualification_path,
    )
    if expected_route["selected_router_branch_id"] != "6":
        raise ValueError("branch-6 diagnostic requires exact router branch 6")
    if router.get("schema_version") != ROUTER_SCHEMA or router.get(
        "status"
    ) != ROUTER_STATUS:
        raise ValueError("branch-6 diagnostic requires the canonical router artifact")
    if router != expected_route:
        raise ValueError("canonical router differs from the fully recomputed route")
    if merged.get("schema_version") != MERGED_SCHEMA or merged.get("status") != (
        "V14M3_COMPLETE_UNSCORED_MERGE_PASS"
    ):
        raise ValueError("branch-6 diagnostic requires the complete V14 merge")
    if not merged_integrity_pass(merged.get("merge_integrity", {})):
        raise ValueError("branch-6 diagnostic requires qualified merge integrity")
    fold_rows = {int(row["outer_fold"]): row for row in merged.get("folds", [])}
    if sorted(fold_rows) != list(range(20)):
        raise ValueError("branch-6 diagnostic requires folds0-19")

    univ = M2Universe(m2_csv)
    identity = univ.build()
    if identity.get("canonical_mutant_full_profile_identity") != (
        "EXACT_PUZZLE_METHOD_MUTATION"
    ):
        raise ValueError("branch-6 diagnostic requires exact target identity")
    records = univ.get_records()
    split = build_split_v4(sorted({record.puzzle for record in records}), seed=20260813)
    folds = {int(fold.outer_fold): fold for fold in split["folds"]}
    rows = []
    for fold_id in range(20):
        held_puzzle = str(folds[fold_id].held_puzzle)
        if str(fold_rows[fold_id].get("held_puzzle")) != held_puzzle:
            raise ValueError("branch-6 merge held-puzzle mapping changed")
        held_records = [record for record in records if record.puzzle == held_puzzle]
        current = _fold_tail_diagnostic(
            univ,
            held_records,
            _load_prediction(Path(fold_rows[fold_id]["prediction_artifact"]), fold_id),
        )
        rows.append(
            {
                "outer_fold": fold_id,
                "held_puzzle": held_puzzle,
                **current,
            }
        )
    summary = summarize_tail_differences(
        [row["lower_minus_upper_tail_miss90"] for row in rows]
    )
    passed = summary["p2_route_eligible"] is True
    return {
        "schema_version": SCHEMA,
        "phase": "POST_V14D1",
        "status": (
            "POST_V14_BRANCH6_TAIL_DIAGNOSTIC_PASS"
            if passed
            else "POST_V14_BRANCH6_TAIL_DIAGNOSTIC_FAIL"
        ),
        "authority_token": AUTHORITY_TOKEN,
        "router_branch_id": "6",
        "route_classification": "DISTRIBUTION_ONLY_FAILURE",
        "primary_statistic": PRIMARY_STATISTIC,
        "source_implementation_commit": (
            "7468f1e066c7e1f80aae326bcadc41f0349f172a"
        ),
        "source_artifacts": {
            "complete_merge": merged_path,
            "complete_score": score_path,
            "qualification": qualification_path,
            "router": router_path,
            "m2_csv": str(m2_csv),
        },
        "folds": rows,
        "summary": summary,
        "p2_route_eligible": passed,
        "next_action": (
            "OPEN_FOCUSED_P2_AMENDMENT_AUTHORITY"
            if passed
            else "P3_STOP_MODEL_RESCUE"
        ),
        "training_performed": False,
        "point_or_prediction_updated": False,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
        "evidence_status": "POST_HOC_DEVELOPMENT_ROUTING_DIAGNOSTIC_ONLY",
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError("branch-6 diagnostic refuses to overwrite its result")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-contract", type=Path, required=True)
    parser.add_argument("--merged-json", type=Path, required=True)
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--qualification-json", type=Path, required=True)
    parser.add_argument("--router-json", type=Path, required=True)
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    active_contract = args.active_contract.resolve()
    merged_json = args.merged_json.resolve()
    score_json = args.score_json.resolve()
    qualification_json = args.qualification_json.resolve()
    router_json = args.router_json.resolve()
    m2_csv = args.m2_csv.resolve()
    out_json = args.out_json.resolve()
    assert_diagnostic_authority(
        active_contract,
        merged_path=merged_json,
        score_path=score_json,
        qualification_path=qualification_json,
        router_path=router_json,
        m2_csv_path=m2_csv,
        output_path=out_json,
    )
    if out_json.exists():
        raise FileExistsError("branch-6 diagnostic refuses to overwrite its result")
    result = diagnose(
        merged=json.loads(merged_json.read_text(encoding="utf-8")),
        score=json.loads(score_json.read_text(encoding="utf-8")),
        qualification=json.loads(qualification_json.read_text(encoding="utf-8")),
        router=json.loads(router_json.read_text(encoding="utf-8")),
        m2_csv=m2_csv,
        merged_path=str(merged_json),
        score_path=str(score_json),
        qualification_path=str(qualification_json),
        router_path=str(router_json),
    )
    _atomic_write_json(out_json, result)
    print(json.dumps({"status": result["status"], "result": str(out_json)}))
    return 0 if result["p2_route_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
