#!/usr/bin/env python3
"""Read-only realized-source preflight for the inactive Puzzle-Set V5 draft.

The report produced here is deliberately not an authority transition.  It reads
only frozen model state, prediction-free registries and source identity fields;
it never opens an M2 target table or a scientific score artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from scripts.reactflow_delta.model_rescue_v2 import MeanAlignedModel
from scripts.reactflow_delta.model_rescue_v13 import SECOND_PASS_EXACT, V13PointModel
from scripts.reactflow_delta.model_rescue_v5_probe import EnsembleFeatureCache
from scripts.reactflow_delta.model_rescue_v6_probe import (
    ConstrainedFeatureCache,
    validate_cache_alignment,
)
from scripts.reactflow_delta.puzzle_set_meta_context import (
    OutcomeBlindWTEncoder,
    load_frozen_v14_encoder,
)
from scripts.reactflow_delta.puzzle_set_safe_sources import (
    load_tic2a_safe_registry,
    validate_feature41_ridge,
)


SCHEMA = "reactflow_delta.puzzle_set_meta_context_source_preflight.v1"
EXPECTED_FOLDS = tuple(range(20))
EXPECTED_CONTRACT_ID = "reactflow_delta_puzzle_set_meta_context_v5_20260827"
EXPECTED_CONTRACT_STATUS = "DRAFT_FROZEN_INACTIVE_V14_SOLE_ACTIVE"
EXPECTED_PARAMETER_COUNTS = {
    "v13_point_checkpoint": 2_064_737,
    "v14_encoder_checkpoint": 4_767_280,
    "v8_meanaligned_checkpoint": 109_581,
}
P1_POINT_PLUS_RESIDUAL_PARAMETERS = 6_235_445
CONTRACT_SOURCE_ROLES = {
    "v13_point_checkpoint": "IMMUTABLE_POINT_ANCHOR",
    "v14_encoder_checkpoint": "IMMUTABLE_OUTCOME_BLIND_WT_REPRESENTATION",
    "v8_meanaligned_checkpoint": (
        "FROZEN_TRAINED_DIRECT_FEATURE_SOURCE_FOR_CALIBRATION"
    ),
    "tic2a_feature41_model_artifact": (
        "OUTER_FOLD_FROZEN_FEATURE41_WEIGHTED_RIDGE_AND_41D_FEATURE_BASIS"
    ),
    "tic2a_merged_registry": ("COMPLETE_TWENTY_FOLD_SOURCE_REGISTRY_PROVENANCE_ONLY"),
    "unconstrained_feature_cache": "FROZEN_FEATURE41_CONSTRUCTION_INPUT",
    "constrained_feature_cache": "FROZEN_FEATURE41_CONSTRUCTION_INPUT",
}
ARTIFACT_SOURCE_ROLES = {
    "v13_point_checkpoint": "FROZEN_SAME_FOLD_POINT_ANCHOR",
    "v14_encoder_checkpoint": "FROZEN_SAME_FOLD_OUTCOME_BLIND_ENCODER",
    "v8_meanaligned_checkpoint": (
        "FROZEN_SAME_FOLD_201D_CALIBRATION_FEATURE_GENERATOR"
    ),
    "tic2a_feature41_model_artifact": (
        "FROZEN_OUTER_FOLD_FEATURE41_RIDGE_AND_41D_BASIS"
    ),
    "tic2a_merged_registry": "FROZEN_COMPLETE_TWENTY_FOLD_SOURCE_REGISTRY",
    "unconstrained_feature_cache": "FROZEN_OUTCOME_BLIND_ENSEMBLE_FEATURE_CACHE",
    "constrained_feature_cache": ("FROZEN_OUTCOME_BLIND_CONSTRAINED_FEATURE_CACHE"),
}
CONTRACT_SOURCE_IDS = {
    "v13_point_checkpoint": "v13_seed0_point",
    "v14_encoder_checkpoint": "v14_seed0_encoder",
    "v8_meanaligned_checkpoint": "v8_seed0_meanaligned",
    "tic2a_feature41_model_artifact": "tic2a_feature41_ridge",
    "tic2a_merged_registry": "tic2a_merged_registry",
    "unconstrained_feature_cache": "unconstrained_feature_cache",
    "constrained_feature_cache": "constrained_feature_cache",
}
EXPECTED_CONTRACT_SOURCE_SPECS = {
    "v13_seed0_point": {
        "role": CONTRACT_SOURCE_ROLES["v13_point_checkpoint"],
        "source": "SAME_OUTER_FOLD_V13_CANDIDATE_SEED0_POINT_CHECKPOINT",
        "expected_filename_pattern": "v13_candidate_point_fold{outer_fold}_seed0.pt",
        "parameter_count": 2_064_737,
        "trainable_in_p1": False,
    },
    "v14_seed0_encoder": {
        "role": CONTRACT_SOURCE_ROLES["v14_encoder_checkpoint"],
        "source": (
            "SAME_OUTER_FOLD_V14_CANDIDATE_SEED0_POINT_CHECKPOINT_ENCODER_SUBSET"
        ),
        "expected_filename_pattern": "v14_candidate_point_fold{outer_fold}_seed0.pt",
        "imported_parameter_count": 4_767_280,
        "trainable_in_p1": False,
    },
    "v8_seed0_meanaligned": {
        "role": CONTRACT_SOURCE_ROLES["v8_meanaligned_checkpoint"],
        "source": "SAME_OUTER_FOLD_V8_SEED0_MEANALIGNED_CHECKPOINT",
        "expected_checkpoint_filename_pattern": (
            "v8_corrected_mean_fold{outer_fold}_seed0.pt"
        ),
        "runtime_path_construction": (
            "DIRECT_FROZEN_V8_DIRECTORY_PLUS_EXPECTED_FILENAME_PATTERN"
        ),
        "wide_fold_result_content_may_be_read": False,
        "calibration_direct_feature_width": 201,
        "parameter_count": 109_581,
        "trainable_in_p1": False,
    },
    "tic2a_feature41_ridge": {
        "role": CONTRACT_SOURCE_ROLES["tic2a_feature41_model_artifact"],
        "source": "TIC2A_COMPLETE_CORRECTED_OUTER_FOLD_MODEL_ARTIFACT_V6_FEATURE41",
        "expected_model_artifact_filename_pattern": (
            "tic2a_corrected_models_fold{outer_fold}.json"
        ),
        "feature_width": 41,
        "trainable_in_p1": False,
    },
    "tic2a_merged_registry": {
        "role": CONTRACT_SOURCE_ROLES["tic2a_merged_registry"],
        "source": "TIC2A_COMPLETE_CORRECTED_MERGED_UNSCORED_JSON",
        "feature_source": False,
        "trainable_in_p1": False,
    },
    "unconstrained_feature_cache": {
        "role": CONTRACT_SOURCE_ROLES["unconstrained_feature_cache"],
        "source": "UNCONSTRAINED_ENSEMBLE_FEATURE_CACHE",
        "trainable_in_p1": False,
    },
    "constrained_feature_cache": {
        "role": CONTRACT_SOURCE_ROLES["constrained_feature_cache"],
        "source": "CONSTRAINED_ENSEMBLE_FEATURE_CACHE",
        "trainable_in_p1": False,
    },
}


def _numel(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a state dict")
    return state


def inspect_checkpoint_parameter_counts(
    *,
    v8_checkpoint: Path,
    v13_checkpoint: Path,
    v14_checkpoint: Path | None,
) -> dict[str, int | None]:
    """Strictly replay checkpoint architecture and return inference counts."""

    v8 = MeanAlignedModel()
    v8.load_state_dict(_load_state(v8_checkpoint), strict=True)
    v13 = V13PointModel(second_pass_mode=SECOND_PASS_EXACT)
    v13.load_state_dict(_load_state(v13_checkpoint), strict=True)
    counts: dict[str, int | None] = {
        "v8_meanaligned_checkpoint": _numel(v8),
        "v13_point_checkpoint": _numel(v13),
        "v14_encoder_checkpoint": None,
    }
    if v14_checkpoint is not None:
        encoder = OutcomeBlindWTEncoder()
        load_frozen_v14_encoder(encoder, _load_state(v14_checkpoint))
        counts["v14_encoder_checkpoint"] = _numel(encoder)
    return counts


def inspect_feature_caches(
    *, unconstrained_cache: Path, constrained_cache: Path
) -> dict[str, int | bool]:
    unconstrained = EnsembleFeatureCache(unconstrained_cache)
    constrained = ConstrainedFeatureCache(constrained_cache)
    try:
        return validate_cache_alignment(unconstrained, constrained)
    finally:
        unconstrained.close()
        constrained.close()


def ridge_parameter_counts(value: dict[str, Any]) -> dict[str, int]:
    """Count predictive and fitted-state scalars for the 41D two-target ridge."""

    _normalized, counts = validate_feature41_ridge(value)
    return counts


def _source_record(
    *,
    outer_fold: int | None,
    source_id: str,
    path: Path,
    seed: int | None,
    realized_parameter_count: int,
    trainable_in_p1: bool,
    provenance_status: str = "FROZEN_PATH_AND_ARCHITECTURE_VERIFIED",
    same_fold_provenance_bound: bool = True,
) -> dict[str, Any]:
    return {
        "contract_source_id": CONTRACT_SOURCE_IDS[source_id],
        "contract_role": CONTRACT_SOURCE_ROLES[source_id],
        "artifact_role": ARTIFACT_SOURCE_ROLES[source_id],
        "observed_path": str(path),
        "outer_fold": outer_fold,
        "seed": seed,
        "realized_parameter_count": int(realized_parameter_count),
        "trainable_in_p1": bool(trainable_in_p1),
        "provenance_status": provenance_status,
        "same_fold_provenance_bound": bool(same_fold_provenance_bound),
    }


def _identity(value: dict[str, Any], *, fold: int, seed: int | None) -> str:
    try:
        observed_fold = int(value["outer_fold"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"fold {fold} source lacks an outer-fold identity") from None
    if observed_fold != fold or (
        seed is not None and int(value.get("seed", -1)) != seed
    ):
        raise ValueError(f"fold {fold} source identity changed")
    held = str(value.get("held_puzzle", ""))
    canonical_held = f"P{fold + 1:02d}"
    if held != canonical_held:
        raise ValueError(
            f"fold {fold} source held puzzle changed: {held!r} != {canonical_held!r}"
        )
    return held


def _validate_contract_source_specs(contract: dict[str, Any]) -> None:
    frozen = contract.get("frozen_input_sources")
    if not isinstance(frozen, dict):
        raise ValueError("Puzzle-Set V5 frozen-input source contract is absent")
    for source_id, expected in EXPECTED_CONTRACT_SOURCE_SPECS.items():
        observed = frozen.get(source_id)
        if not isinstance(observed, dict):
            raise ValueError(f"inactive contract source {source_id} is absent")
        for field, expected_value in expected.items():
            if observed.get(field) != expected_value:
                raise ValueError(
                    f"inactive contract source {source_id}.{field} changed"
                )


def build_preflight(
    *,
    inactive_contract: Path,
    v8_dir: Path,
    v13_dir: Path,
    v14_dir: Path,
    tic2a_merged_json: Path,
    unconstrained_cache: Path,
    constrained_cache: Path,
    checkpoint_inspector: Callable[..., dict[str, int | None]] = (
        inspect_checkpoint_parameter_counts
    ),
    cache_inspector: Callable[..., dict[str, int | bool]] = inspect_feature_caches,
) -> dict[str, Any]:
    """Inspect the exact source universe without granting runtime authority."""

    contract = yaml.safe_load(inactive_contract.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("inactive contract must contain one mapping")
    if (
        contract.get("contract_id") != EXPECTED_CONTRACT_ID
        or contract.get("contract_status") != EXPECTED_CONTRACT_STATUS
        or contract.get("inactive_authority", {}).get("activation_allowed_now")
        is not False
        or contract.get("inactive_authority", {}).get("training_allowed") is not False
    ):
        raise ValueError("Puzzle-Set V5 inactive contract identity or status changed")
    _validate_contract_source_specs(contract)
    for path in (tic2a_merged_json, unconstrained_cache, constrained_cache):
        if not path.is_file():
            raise FileNotFoundError(f"required global source is absent: {path}")
    cache_alignment = cache_inspector(
        unconstrained_cache=unconstrained_cache,
        constrained_cache=constrained_cache,
    )
    if cache_alignment != {
        "biological_key_universe_equal": True,
        "registered_mutants": 13_976,
        "receiver_length": 177,
        "unconstrained_width": 12,
        "constrained_cache_width": 12,
        "constrained_probe_width": 11,
    }:
        raise ValueError("frozen feature-cache identity or alignment changed")

    tic_by_fold = load_tic2a_safe_registry(tic2a_merged_json)

    global_records = {
        "tic2a_merged_registry": _source_record(
            outer_fold=None,
            source_id="tic2a_merged_registry",
            path=tic2a_merged_json,
            seed=None,
            realized_parameter_count=0,
            trainable_in_p1=False,
        ),
        "unconstrained_feature_cache": _source_record(
            outer_fold=None,
            source_id="unconstrained_feature_cache",
            path=unconstrained_cache,
            seed=None,
            realized_parameter_count=0,
            trainable_in_p1=False,
        ),
        "constrained_feature_cache": _source_record(
            outer_fold=None,
            source_id="constrained_feature_cache",
            path=constrained_cache,
            seed=None,
            realized_parameter_count=0,
            trainable_in_p1=False,
        ),
    }

    folds = []
    missing_v14_folds = []
    ridge_counts = set()
    for fold in EXPECTED_FOLDS:
        held = f"P{fold + 1:02d}"
        v8_checkpoint = v8_dir / f"v8_corrected_mean_fold{fold}_seed0.pt"
        v13_checkpoint = v13_dir / f"v13_candidate_point_fold{fold}_seed0.pt"
        v14_checkpoint = v14_dir / f"v14_candidate_point_fold{fold}_seed0.pt"
        for path in (v8_checkpoint, v13_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(
                    f"fold {fold} required source is absent: {path}"
                )

        tic_source = tic_by_fold[fold]
        tic_row = tic_source.row
        if _identity(tic_row, fold=fold, seed=None) != held:
            raise ValueError(f"fold {fold} TIC2A held-puzzle identity differs")
        tic_model_path = tic_source.model_path
        counts = tic_source.ridge_parameter_counts
        ridge_counts.add(tuple(sorted(counts.items())))

        v14_exists = v14_checkpoint.is_file()
        if not v14_exists:
            missing_v14_folds.append(fold)
        checkpoint_counts = checkpoint_inspector(
            v8_checkpoint=v8_checkpoint,
            v13_checkpoint=v13_checkpoint,
            v14_checkpoint=v14_checkpoint if v14_exists else None,
        )
        for source_id, expected in EXPECTED_PARAMETER_COUNTS.items():
            observed = checkpoint_counts[source_id]
            if source_id == "v14_encoder_checkpoint" and not v14_exists:
                if observed is not None:
                    raise ValueError("absent V14 checkpoint unexpectedly has a count")
            elif observed != expected:
                raise ValueError(
                    f"fold {fold} {source_id} parameter count changed: "
                    f"{observed} != {expected}"
                )

        source_records = {
            "v13_point_checkpoint": _source_record(
                outer_fold=fold,
                source_id="v13_point_checkpoint",
                path=v13_checkpoint,
                seed=0,
                realized_parameter_count=EXPECTED_PARAMETER_COUNTS[
                    "v13_point_checkpoint"
                ],
                trainable_in_p1=False,
                provenance_status="FILENAME_AND_ARCHITECTURE_ONLY",
                same_fold_provenance_bound=False,
            ),
            "v8_meanaligned_checkpoint": _source_record(
                outer_fold=fold,
                source_id="v8_meanaligned_checkpoint",
                path=v8_checkpoint,
                seed=0,
                realized_parameter_count=EXPECTED_PARAMETER_COUNTS[
                    "v8_meanaligned_checkpoint"
                ],
                trainable_in_p1=False,
                provenance_status="FILENAME_AND_ARCHITECTURE_ONLY",
                same_fold_provenance_bound=False,
            ),
            "tic2a_feature41_model_artifact": _source_record(
                outer_fold=fold,
                source_id="tic2a_feature41_model_artifact",
                path=tic_model_path,
                seed=None,
                realized_parameter_count=counts["predictive_parameter_count"],
                trainable_in_p1=False,
            ),
        }
        if v14_exists:
            source_records["v14_encoder_checkpoint"] = _source_record(
                outer_fold=fold,
                source_id="v14_encoder_checkpoint",
                path=v14_checkpoint,
                seed=0,
                realized_parameter_count=EXPECTED_PARAMETER_COUNTS[
                    "v14_encoder_checkpoint"
                ],
                trainable_in_p1=False,
                provenance_status=("FILENAME_AND_P1_ENCODER_SUBSET_ARCHITECTURE_ONLY"),
                same_fold_provenance_bound=False,
            )
        folds.append(
            {
                "outer_fold": fold,
                "held_puzzle": held,
                "v14_checkpoint_filename_and_architecture_verified": v14_exists,
                "same_fold_checkpoint_provenance_bound": False,
                "observed_source_records": source_records,
            }
        )

    if len(ridge_counts) != 1:
        raise ValueError("TIC2A feature41 ridge parameterization differs across folds")
    ridge = dict(next(iter(ridge_counts)))
    full_footprint = (
        P1_POINT_PLUS_RESIDUAL_PARAMETERS
        + EXPECTED_PARAMETER_COUNTS["v13_point_checkpoint"]
        + EXPECTED_PARAMETER_COUNTS["v8_meanaligned_checkpoint"]
        + ridge["predictive_parameter_count"]
    )
    all_consumed_frozen_source_parameters = (
        EXPECTED_PARAMETER_COUNTS["v13_point_checkpoint"]
        + EXPECTED_PARAMETER_COUNTS["v14_encoder_checkpoint"]
        + EXPECTED_PARAMETER_COUNTS["v8_meanaligned_checkpoint"]
        + ridge["predictive_parameter_count"]
    )
    external_to_p1_module_upstream_parameters = (
        EXPECTED_PARAMETER_COUNTS["v13_point_checkpoint"]
        + EXPECTED_PARAMETER_COUNTS["v8_meanaligned_checkpoint"]
        + ridge["predictive_parameter_count"]
    )
    v14_filename_architecture_complete = not missing_v14_folds
    status = (
        "SOURCE_CHECKPOINT_FILENAME_AND_ARCHITECTURE_UNIVERSE_COMPLETE_"
        "TERMINAL_SAFE_PROVENANCE_REQUIRED"
        if v14_filename_architecture_complete
        else "SOURCE_PREFLIGHT_WAITING_FOR_V14_CHECKPOINT_FILENAME_AND_"
        "ARCHITECTURE_UNIVERSE"
    )
    return {
        "schema_version": SCHEMA,
        "inactive_contract_source_projection_verified": {
            "contract_id": EXPECTED_CONTRACT_ID,
            "contract_status": EXPECTED_CONTRACT_STATUS,
            "observed_path": str(inactive_contract),
        },
        "status": status,
        "active_contract_path_accepted_or_read": False,
        "terminal_safe_provenance_manifest_accepted_or_read": False,
        "non_v14_checkpoint_filename_and_architecture_folds": list(EXPECTED_FOLDS),
        "v14_checkpoint_filename_and_architecture_folds": [
            fold for fold in EXPECTED_FOLDS if fold not in missing_v14_folds
        ],
        "missing_v14_checkpoint_folds": missing_v14_folds,
        "checkpoint_same_fold_provenance_binding": {
            "status": "TERMINAL_SAFE_PROVENANCE_MANIFEST_REQUIRED",
            "v8": False,
            "v13": False,
            "v14": False,
            "tic2a": True,
        },
        "observed_global_source_records": global_records,
        "feature_cache_alignment": cache_alignment,
        "folds": folds,
        "parameter_accounting": {
            "expected_at_complete": {
                "tic2a_feature41_ridge_predictive_parameters": ridge[
                    "predictive_parameter_count"
                ],
                "tic2a_feature41_ridge_stored_fitted_scalars": ridge[
                    "stored_fitted_scalar_count"
                ],
                "p1_point_plus_residual_modules_total": (
                    P1_POINT_PLUS_RESIDUAL_PARAMETERS
                ),
                "frozen_v13_point_upstream": EXPECTED_PARAMETER_COUNTS[
                    "v13_point_checkpoint"
                ],
                "frozen_v8_meanaligned_upstream": EXPECTED_PARAMETER_COUNTS[
                    "v8_meanaligned_checkpoint"
                ],
                "feature_caches_learned_parameters": 0,
                "all_consumed_frozen_source_parameters": (
                    all_consumed_frozen_source_parameters
                ),
                "external_to_p1_module_upstream_parameters": (
                    external_to_p1_module_upstream_parameters
                ),
                "candidate_full_prediction_learned_parameter_footprint": (
                    full_footprint
                ),
                "v14_encoder_already_included_in_p1_point_module": True,
                "temporary_pretraining_decoder_in_inference_footprint": False,
            },
            "realized_checkpoint_filename_and_architecture_universe": (
                {
                    "all_consumed_frozen_source_parameters": (
                        all_consumed_frozen_source_parameters
                    ),
                    "candidate_full_prediction_learned_parameter_footprint": (
                        full_footprint
                    ),
                }
                if v14_filename_architecture_complete
                else None
            ),
            "activation_bound_same_fold_parameter_accounting": None,
        },
        "wide_v8_v13_result_content_read": False,
        "scientific_score_fields_read": False,
        "m2_target_table_read": False,
        "external_outcome_accessed": False,
        "authority_modified": False,
        "activation_ready": False,
        "binding_candidate": None,
        "scientific_status": "NOT_EVALUATED_SOURCE_PROVENANCE_ONLY",
        "training_authorized": False,
        "activation_authorized_by_report": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inactive-contract", type=Path, required=True)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--v13-dir", type=Path, required=True)
    parser.add_argument("--v14-dir", type=Path, required=True)
    parser.add_argument("--tic2a-merged-json", type=Path, required=True)
    parser.add_argument("--unconstrained-cache", type=Path, required=True)
    parser.add_argument("--constrained-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_preflight(
        inactive_contract=args.inactive_contract,
        v8_dir=args.v8_dir,
        v13_dir=args.v13_dir,
        v14_dir=args.v14_dir,
        tic2a_merged_json=args.tic2a_merged_json,
        unconstrained_cache=args.unconstrained_cache,
        constrained_cache=args.constrained_cache,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
