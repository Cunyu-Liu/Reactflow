#!/usr/bin/env python3
"""Outcome-blind data assembly for the proposed puzzle-set model family."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from scripts.reactflow_delta.puzzle_set_meta_context import (
    EXPECTED_CONSTRUCTS_PER_PUZZLE,
    PuzzleSetMetaContextPointModel,
)
from scripts.reactflow_delta.puzzle_set_meta_context_calibration import (
    calibrated_distribution,
)
from scripts.reactflow_delta.run_p2_v3 import _bio_key


POINT_PREDICTION_SCHEMA = (
    "reactflow_delta.puzzle_set_meta_context_point_prediction.proposed.v1"
)
PREDICTION_SCHEMA = "reactflow_delta.puzzle_set_meta_context_prediction.proposed.v2"
FORBIDDEN_PREDICTION_FIELDS = {
    "target",
    "target_error",
    "qualified_mask",
    "qualified_target_mask",
    "target_mask",
    "loss",
    "score",
}


def _construct_to_puzzle(records: Sequence[Any]) -> dict[str, str]:
    observed: dict[str, set[str]] = defaultdict(set)
    for record in records:
        observed[str(record.construct_id)].add(str(record.puzzle))
    if any(len(puzzles) != 1 for puzzles in observed.values()):
        raise ValueError("one construct cannot belong to multiple puzzles")
    return {
        construct_id: next(iter(puzzles))
        for construct_id, puzzles in observed.items()
    }


def assemble_puzzle_training_batches(
    records: Sequence[Any],
    cells: Sequence[dict[str, Any]],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
) -> list[dict[str, Any]]:
    """Convert focal construct cells into equal eight-cell puzzle batches."""

    construct_puzzle = _construct_to_puzzle(records)
    constructs_by_puzzle: dict[str, list[str]] = defaultdict(list)
    for construct_id, puzzle in construct_puzzle.items():
        constructs_by_puzzle[puzzle].append(construct_id)
    cells_by_construct: dict[str, dict[str, Any]] = {}
    seen_constructs: set[str] = set()
    for cell in cells:
        construct_id = str(cell["construct_id"])
        if construct_id in seen_constructs:
            raise ValueError("puzzle-set assembler received duplicate construct cells")
        seen_constructs.add(construct_id)
        if construct_id not in construct_puzzle or construct_id not in context_cache:
            raise ValueError("puzzle-set cell lacks record or WT context metadata")
        cells_by_construct[construct_id] = cell

    batches = []
    for puzzle in sorted(constructs_by_puzzle):
        construct_ids = sorted(constructs_by_puzzle[puzzle])
        if len(construct_ids) != EXPECTED_CONSTRUCTS_PER_PUZZLE:
            raise ValueError(
                f"puzzle {puzzle} has {len(construct_ids)} constructs instead of eight"
            )
        converted = []
        for focal_index, construct_id in enumerate(construct_ids):
            cell = cells_by_construct.get(construct_id)
            if cell is None:
                # P20_Eterna is a real registered construct without a qualified
                # supervised cell.  Its outcome-blind WT context remains in the
                # eight-construct set; no target or loss is fabricated for it.
                continue
            converted.append(
                {
                    "focal_construct_index": focal_index,
                    "construct_id": construct_id,
                    "edit_index": cell["edit"],
                    "signed_distance": cell["distance"],
                    "refs": list(cell["refs"]),
                    "alts": list(cell["alts"]),
                    "feature41_point": cell["feature41_point"],
                    "prediction_mask": cell["prediction_mask"],
                    "target": cell["target"],
                    "qualified_mask": cell["qualified_mask"],
                    "wt": cell["wt"],
                    "feature41_basis": cell["feature41_basis"],
                }
            )
        if not converted:
            raise ValueError(f"puzzle {puzzle} has no qualified supervised cells")
        batches.append(
            {
                "puzzle": puzzle,
                "construct_ids": construct_ids,
                "contexts": [context_cache[value] for value in construct_ids],
                "cells": converted,
            }
        )
    if not batches:
        raise ValueError("puzzle-set assembler produced no training puzzles")
    return batches


def predict_held_puzzle_points(
    *,
    univ: Any,
    held_records: Sequence[Any],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    feature41_by_construct: dict[str, np.ndarray],
    candidate: PuzzleSetMetaContextPointModel,
    null: PuzzleSetMetaContextPointModel,
    outer_fold: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate full registered point predictions without reading outcomes."""

    puzzles = {str(record.puzzle) for record in held_records}
    if len(puzzles) != 1:
        raise ValueError("held puzzle prediction requires exactly one puzzle")
    by_construct: dict[str, list[Any]] = defaultdict(list)
    for record in held_records:
        by_construct[str(record.construct_id)].append(record)
    construct_ids = sorted(by_construct)
    if len(construct_ids) != EXPECTED_CONSTRUCTS_PER_PUZZLE:
        raise ValueError("held puzzle prediction requires exactly eight constructs")
    if set(construct_ids) != set(context_cache) or set(construct_ids) != set(
        feature41_by_construct
    ):
        raise ValueError("held puzzle context or feature41 universe is not exact")

    contexts = [context_cache[construct_id] for construct_id in construct_ids]
    candidate.eval()
    null.eval()
    keys: list[str] = []
    feature41_values = []
    candidate_values = []
    null_values = []
    with torch.no_grad():
        candidate_hidden = candidate.encode_puzzle_set(contexts)
        null_hidden = null.encode_puzzle_set(contexts)
        observed = [context[3].bool() for context in contexts]
        candidate_mixed = candidate.meta_context.mix_construct_tokens(
            candidate_hidden, observed
        )
        null_mixed = null.meta_context.mix_construct_tokens(null_hidden, observed)
        for focal_index, construct_id in enumerate(construct_ids):
            construct_records = sorted(
                by_construct[construct_id],
                key=lambda record: (
                    int(record.design_pos),
                    str(record.ref),
                    str(record.alt),
                ),
            )
            construct = univ.get_construct(construct_id)
            length = len(construct.sequence)
            feature41 = np.asarray(
                feature41_by_construct[construct_id], dtype=np.float32
            )
            expected = (len(construct_records), length)
            if feature41.shape != expected:
                raise ValueError("held puzzle feature41 matrix is misaligned")
            device = next(candidate.parameters()).device
            if next(null.parameters()).device != device:
                raise ValueError("candidate and null must use the same device")
            edit = torch.tensor(
                [int(record.full_pos) for record in construct_records], device=device
            )
            distance = (
                torch.arange(length, device=device)[None, :] - edit[:, None]
            ).float()
            prediction_mask = torch.tensor(
                np.tile(
                    np.asarray(construct.wt_observed, dtype=bool),
                    (len(construct_records), 1),
                ),
                device=device,
            )
            feature41_tensor = torch.tensor(feature41, device=device)
            refs = [str(record.ref) for record in construct_records]
            alts = [str(record.alt) for record in construct_records]
            candidate_point, _ = candidate.forward_from_encoded(
                candidate_hidden,
                candidate_mixed,
                focal_index,
                edit,
                distance,
                refs,
                alts,
                feature41_tensor,
                prediction_mask,
            )
            null_point, _ = null.forward_from_encoded(
                null_hidden,
                null_mixed,
                focal_index,
                edit,
                distance,
                refs,
                alts,
                feature41_tensor,
                prediction_mask,
            )
            for mutant, record in enumerate(construct_records):
                keys.extend(
                    _bio_key(univ, record, position) for position in range(length)
                )
                feature41_values.append(feature41[mutant].astype(np.float64))
                candidate_values.append(
                    candidate_point[mutant].cpu().numpy().astype(np.float64)
                )
                null_values.append(null_point[mutant].cpu().numpy().astype(np.float64))

    if len(keys) != len(set(keys)):
        raise RuntimeError("puzzle-set held prediction contains duplicate keys")
    output = {
        "schema_version": np.asarray(POINT_PREDICTION_SCHEMA),
        "keys": np.asarray(keys, dtype=object),
        "biological_scoring_key": np.asarray(keys, dtype=object),
        "outer_fold": np.full(len(keys), int(outer_fold), dtype=np.int64),
        "seed": np.full(len(keys), int(seed), dtype=np.int64),
        "registered_status": np.full(len(keys), "covered", dtype=object),
        "feature41_point": np.concatenate(feature41_values),
        "candidate_point": np.concatenate(candidate_values),
        "null_point": np.concatenate(null_values),
    }
    if set(output) & FORBIDDEN_PREDICTION_FIELDS:
        raise RuntimeError("puzzle-set prediction contains target-side fields")
    if not all(
        np.isfinite(value).all()
        for value in output.values()
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiu"
    ):
        raise RuntimeError("puzzle-set prediction contains nonfinite values")
    return output


def _flatten_held_calibration_features(
    *,
    held_records: Sequence[Any],
    feature41_basis_by_construct: dict[str, np.ndarray],
    direct_features_by_construct: dict[str, np.ndarray],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    by_construct: dict[str, list[Any]] = defaultdict(list)
    for record in held_records:
        by_construct[str(record.construct_id)].append(record)
    if set(by_construct) != set(feature41_basis_by_construct) or set(
        by_construct
    ) != set(direct_features_by_construct) or set(by_construct) != set(context_cache):
        raise ValueError("held calibration feature universe is not exact")
    feature_rows = []
    direct_rows = []
    for construct_id in sorted(by_construct):
        records = sorted(
            by_construct[construct_id],
            key=lambda record: (
                int(record.design_pos),
                str(record.ref),
                str(record.alt),
            ),
        )
        length = int(context_cache[construct_id][0].shape[0])
        feature41 = np.asarray(
            feature41_basis_by_construct[construct_id], dtype=np.float32
        )
        direct = np.asarray(
            direct_features_by_construct[construct_id], dtype=np.float32
        )
        if feature41.shape != (len(records), length, 41) or direct.shape != (
            len(records),
            length,
            201,
        ):
            raise ValueError("held calibration feature matrices are misaligned")
        feature_rows.append(feature41.reshape(-1, 41))
        direct_rows.append(direct.reshape(-1, 201))
    return np.concatenate(feature_rows), np.concatenate(direct_rows)


def predict_held_puzzle_distributions(
    *,
    univ: Any,
    held_records: Sequence[Any],
    context_cache: dict[str, tuple[torch.Tensor, ...]],
    feature41_by_construct: dict[str, np.ndarray],
    feature41_basis_by_construct: dict[str, np.ndarray],
    direct_features_by_construct: dict[str, np.ndarray],
    candidate: PuzzleSetMetaContextPointModel,
    null: PuzzleSetMetaContextPointModel,
    residual_heads: dict[str, Any],
    standardizers: dict[str, Any],
    outer_fold: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Emit target-free point and median-preserving probability predictions."""

    if set(residual_heads) != {"candidate", "null"} or set(standardizers) != {
        "candidate",
        "null",
    }:
        raise ValueError("puzzle-set held calibration requires both exact arms")
    output = predict_held_puzzle_points(
        univ=univ,
        held_records=held_records,
        context_cache=context_cache,
        feature41_by_construct=feature41_by_construct,
        candidate=candidate,
        null=null,
        outer_fold=outer_fold,
        seed=seed,
    )
    output["schema_version"] = np.asarray(PREDICTION_SCHEMA)
    feature41, direct = _flatten_held_calibration_features(
        held_records=held_records,
        feature41_basis_by_construct=feature41_basis_by_construct,
        direct_features_by_construct=direct_features_by_construct,
        context_cache=context_cache,
    )
    if len(feature41) != len(output["keys"]):
        raise RuntimeError("held calibration rows do not match biological keys")
    for name in ("candidate", "null"):
        distribution = calibrated_distribution(
            point=np.asarray(output[f"{name}_point"], dtype=np.float64),
            feature41=feature41,
            direct_features=direct,
            head=residual_heads[name],
            standardizer=standardizers[name],
        )
        for suffix, values in distribution.items():
            output[f"{name}_{suffix}"] = values
    if set(output) & FORBIDDEN_PREDICTION_FIELDS:
        raise RuntimeError("puzzle-set distribution contains target-side fields")
    if not all(
        np.isfinite(value).all()
        for value in output.values()
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiu"
    ):
        raise RuntimeError("puzzle-set distribution contains nonfinite values")
    return output
