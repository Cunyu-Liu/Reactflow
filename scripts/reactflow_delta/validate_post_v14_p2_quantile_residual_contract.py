#!/usr/bin/env python3
"""Validate the frozen inactive post-V14 P2 declaration without artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
AMENDMENT_PATH = (
    REPO_ROOT
    / "configs/reactflow_delta/post_v14_p2_quantile_residual_amendment.yaml"
)
LEDGER_PATH = (
    REPO_ROOT
    / "docs/prospective_v2/post_v14_p2_quantile_residual_decision_ledger.yaml"
)
ACTIVE_POINTER_PATH = REPO_ROOT / "configs/reactflow_delta/active_contract.yaml"

EXPECTED_SCHEMA = "reactflow_delta.post_v14_p2_quantile_residual_amendment.v1"
EXPECTED_LEDGER_SCHEMA = (
    "reactflow_delta.post_v14_p2_quantile_residual_decision_ledger.draft.v1"
)
EXPECTED_CONTRACT_ID = "reactflow_delta_post_v14_p2_quantile_residual_20260827"
EXPECTED_STATUS = "DRAFT_FROZEN_INACTIVE"
PROJECT_TASK_ID = "reactflow_delta_post_v14_p2_quantile_residual"
PENDING = "PENDING_TERMINAL_BINDING"

TAUS = [
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
]
WEIGHTS = [
    0.0375,
    0.0375,
    0.075,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.075,
    0.0375,
    0.0375,
]
FOLDS = list(range(20))

TOP_LEVEL_DENIALS = (
    "authority_issued_now",
    "activation_allowed",
    "source_projection_allowed",
    "training_allowed",
    "candidate_model_training_allowed",
    "prediction_allowed",
    "smoke_allowed",
    "screen_allowed",
    "held_score_read_allowed",
    "partial_fold_score_read_allowed",
    "scoring_allowed",
    "qualification_allowed",
    "formal_confirmation_allowed",
    "new_external_outcome_access_allowed",
)
NESTED_DENIALS = TOP_LEVEL_DENIALS

EXPECTED_DOCUMENTS = {
    "human_contract_path": (
        "docs/prospective_v2/"
        "post_v14_p2_quantile_residual_amendment_20260827.md"
    ),
    "decision_ledger_path": (
        "docs/prospective_v2/"
        "post_v14_p2_quantile_residual_decision_ledger.yaml"
    ),
    "design_path": (
        "docs/plans/2026-08-27-post-v14-p2-quantile-residual-design.md"
    ),
    "implementation_plan_path": (
        "docs/plans/2026-08-27-post-v14-p2-quantile-residual-implementation.md"
    ),
}
EXPECTED_PARENT_ROUTE = {
    "binding_status": PENDING,
    "selected_router_branch_id": "6",
    "route_classification": "DISTRIBUTION_ONLY_FAILURE",
    "diagnostic_schema": "reactflow_delta.post_v14_branch6_tail_diagnostic.v1",
    "diagnostic_status": "POST_V14_BRANCH6_TAIL_DIAGNOSTIC_PASS",
    "diagnostic_primary_statistic": "LOWER_MINUS_UPPER_TAIL_MISS90",
    "diagnostic_interval_requirement": (
        "PUZZLE_LEVEL_95_PERCENT_INTERVAL_WHOLLY_ONE_SIDE_OF_ZERO"
    ),
    "diagnostic_direction_agreement_min_puzzles": 14,
    "diagnostic_total_puzzles": 20,
    "diagnostic_next_action": "OPEN_FOCUSED_P2_AMENDMENT_AUTHORITY",
    "route_eligibility_only": True,
}
EXPECTED_PENDING_BINDINGS = {
    "binding_status": PENDING,
    "v14_terminal_handoff_path": PENDING,
    "post_v14_router_output_path": PENDING,
    "branch6_diagnostic_output_path": PENDING,
    "source_manifest_path": PENDING,
    "artifact_root": PENDING,
    "smoke_output_path": PENDING,
    "screen_output_path": PENDING,
    "formal_output_path": PENDING,
    "copied_v14_gate_values": PENDING,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"static contract YAML root must be a mapping: {path}")
    return value


def _require(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise RuntimeError(message)


def _mapping(value: Any, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(message)
    return value


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _registered_initial_target_gaps() -> list[float]:
    """Derive the fixed input-independent V10 initialization gaps in float64."""

    # Existing V10 initialization: weight 0.5, allocation 0, SCALE_FLOOR 1e-3,
    # narrow softplus 0.08 and an additional wide gap of 0.20. Both component
    # locations equal the frozen point, so quantiles can be derived at point 0.
    narrow_scale = 0.001 + 0.08
    wide_scale = narrow_scale + 0.20

    def mixture_cdf(value: float) -> float:
        return 0.5 * _normal_cdf(value / narrow_scale) + 0.5 * _normal_cdf(
            value / wide_scale
        )

    quantiles: list[float] = []
    for tau in TAUS:
        low, high = -4.0, 4.0
        for _ in range(256):
            midpoint = (low + high) / 2.0
            if mixture_cdf(midpoint) < tau:
                low = midpoint
            else:
                high = midpoint
        quantiles.append((low + high) / 2.0)
    return [right - left for left, right in zip(quantiles, quantiles[1:])]


def _assert_inactive_authority(
    amendment: dict[str, Any], active: dict[str, Any]
) -> None:
    for field in TOP_LEVEL_DENIALS:
        if amendment.get(field) is not False:
            raise RuntimeError(f"top-level inactive authority reopened: {field}")

    nested = _mapping(
        amendment.get("inactive_authority"),
        "nested inactive authority is missing",
    )
    for field in NESTED_DENIALS:
        if nested.get(field) is not False:
            raise RuntimeError(f"nested inactive authority reopened: {field}")
    _require(
        nested.get("canonical_active_pointer"),
        "configs/reactflow_delta/active_contract.yaml",
        "canonical active pointer changed",
    )
    _require(nested.get("authority_state"), EXPECTED_STATUS, "inactive state changed")
    _require(nested.get("runnable_phases"), [], "P2 runnable phase opened")
    _require(
        nested.get("phase_tokens_issued_now"), False, "P2 phase token was issued"
    )
    _require(
        nested.get("generic_training_token_allowed"),
        False,
        "generic training token was allowed",
    )
    _require(
        nested.get("next_allowed_action_now"),
        "WAIT_FOR_EXACT_TERMINAL_BINDING",
        "inactive next action changed",
    )

    active_authority = _mapping(
        active.get("authority"), "canonical active authority is missing"
    )
    active_phases = active.get("runnable_phases", [])
    if not isinstance(active_phases, list):
        raise RuntimeError("canonical runnable phases are malformed")
    active_fields = (
        active.get("project_task_id"),
        active_authority.get("current_phase"),
        active_authority.get("current_runnable_phase"),
        active_authority.get("machine_contract_path"),
        *active_phases,
    )
    if any(
        value == PROJECT_TASK_ID
        or (isinstance(value, str) and value.startswith("P2M"))
        or value == str(AMENDMENT_PATH.relative_to(REPO_ROOT))
        for value in active_fields
    ):
        raise RuntimeError("pending P2 declaration became active or runnable")


def _assert_parent_and_pending(amendment: dict[str, Any]) -> None:
    _require(
        amendment.get("parent_route"),
        EXPECTED_PARENT_ROUTE,
        "branch-6 parent route changed",
    )
    _require(
        amendment.get("pending_terminal_binding"),
        EXPECTED_PENDING_BINDINGS,
        "pending terminal/source/output binding changed",
    )


def _assert_input_and_point(amendment: dict[str, Any]) -> None:
    frozen_input = _mapping(amendment.get("frozen_input"), "frozen input is missing")
    expected_order = [
        {"name": "FEATURE41_BASIS", "width": 41},
        {"name": "FROZEN_V14_POINT", "width": 1},
        {"name": "ABSOLUTE_FROZEN_V14_POINT", "width": 1},
        {"name": "FROZEN_TRAINED_V8_DIRECT_FEATURES", "width": 201},
    ]
    expected_v8 = [
        {"name": "SOURCE_HIDDEN", "width": 96},
        {"name": "RECEIVER_HIDDEN", "width": 96},
        {"name": "SIGNED_DISTANCE", "width": 1},
        {"name": "MUTATION_ONE_HOT", "width": 8},
    ]
    _require(frozen_input.get("total_width"), 244, "frozen input width changed")
    _require(frozen_input.get("order"), expected_order, "frozen input order changed")
    _require(
        frozen_input.get("v8_direct_feature_order"),
        expected_v8,
        "V8 direct feature construction changed",
    )
    if sum(item["width"] for item in expected_order) != 244:
        raise RuntimeError("frozen input component widths do not sum to 244")
    if sum(item["width"] for item in expected_v8) != 201:
        raise RuntimeError("V8 direct feature widths do not sum to 201")
    _require(
        frozen_input.get("builder"),
        "scripts.reactflow_delta.model_rescue_v10.calibration_input",
        "V10 calibration input reuse changed",
    )
    _require(
        frozen_input.get("standardizer"),
        "scripts.reactflow_delta.model_rescue_v10.TrainOnlyStandardizer",
        "V10 standardizer reuse changed",
    )
    _require(
        (
            frozen_input.get("standardizer_fit_rows"),
            frozen_input.get("standardizer_scale_below"),
            frozen_input.get("standardizer_replacement_scale"),
            frozen_input.get("same_rows_order_and_standardizer_for_both_arms"),
        ),
        ("OUTER_TRAIN_ONLY", 1.0e-6, 1.0, True),
        "input standardization or matched-row rule changed",
    )
    _require(
        frozen_input.get("forbidden_inputs"),
        [
            "METHOD_ID",
            "PUZZLE_ID",
            "DATASET_ID",
            "HELD_TARGET",
            "HELD_ERROR",
            "HELD_QUALIFIED_TARGET_MASK",
            "EXTERNAL_OUTCOME",
            "SCORE_DERIVED_FIELD",
        ],
        "target-free input boundary changed",
    )

    point = _mapping(
        amendment.get("frozen_point_anchor"), "frozen V14 point anchor is missing"
    )
    _require(
        (
            point.get("model"),
            point.get("seed"),
            point.get("trainable"),
            point.get("outer_train_mode"),
            point.get("held_gpu_recomputation_is_authority"),
            point.get("raw_point_passed_separately_to_both_heads"),
            point.get("candidate_tau_0_50_assignment"),
        ),
        (
            "SAME_OUTER_FOLD_V14_CANDIDATE",
            0,
            False,
            "EVAL_NO_GRAD_FULL_STATE_SNAPSHOT",
            False,
            True,
            "DETACHED_FLOAT64_RAW_POINT",
        ),
        "frozen V14 point anchor changed",
    )
    replay = _mapping(point.get("point_replay"), "point replay is missing")
    _require(
        replay,
        {
            "invariant": "FROZEN_V14_POINT_REPLAY",
            "atol": 1.0e-7,
            "rtol": 0.0,
            "candidate_median_exact_array_equality_required": True,
            "initialization_grid_tolerance_applies": False,
        },
        "point replay or exact median rule changed",
    )


def _assert_distribution_and_models(amendment: dict[str, Any]) -> None:
    candidate = _mapping(
        amendment.get("candidate_predictive_distribution"),
        "candidate predictive distribution is missing",
    )
    _require(candidate.get("definition_from_phase"), "P2M3", "atom phase changed")
    _require(
        candidate.get("family"),
        "FIXED_THIRTEEN_ATOM_MONOTONE_QUANTILE_DISTRIBUTION",
        "candidate predictive distribution changed",
    )
    _require(candidate.get("taus"), TAUS, "frozen tau array changed")
    _require(candidate.get("weights"), WEIGHTS, "frozen weight array changed")
    if len(candidate["taus"]) != 13 or candidate["taus"].count(0.5) != 1:
        raise RuntimeError("frozen taus must contain one median among 13 nodes")
    if not math.isclose(sum(candidate["weights"]), 1.0, abs_tol=1.0e-12):
        raise RuntimeError("predictive atom weights do not sum to one")
    _require(
        (
            candidate.get("total_mass"),
            candidate.get("mass_below_median"),
            candidate.get("median_mass"),
            candidate.get("mass_above_median"),
        ),
        (1.0, 0.45, 0.10, 0.45),
        "predictive mass split changed",
    )
    if not (
        math.isclose(sum(WEIGHTS[:6]), 0.45, abs_tol=1.0e-12)
        and WEIGHTS[6] == 0.10
        and math.isclose(sum(WEIGHTS[7:]), 0.45, abs_tol=1.0e-12)
    ):
        raise RuntimeError("registered weights violate the 0.45/0.10/0.45 split")
    _require(
        (
            candidate.get("learned_atom_masses_allowed"),
            candidate.get("interpolation_or_extrapolation_allowed"),
            candidate.get("result_dependent_tail_refinement_allowed"),
        ),
        (False, False, False),
        "fixed predictive atom boundary changed",
    )
    _require(
        candidate.get("training_surrogate"),
        {
            "name": "TWO_TIMES_WEIGHTED_PINBALL",
            "formula": "2_SUM_I_WEIGHT_I_RHO_TAU_I_OF_Y_MINUS_Q_I",
            "scientific_crps": False,
            "allowed_in_scientific_score_or_gate_fields": False,
        },
        "weighted pinball was relabeled or changed",
    )
    scores = _mapping(
        candidate.get("scientific_scores"), "candidate scientific scores missing"
    )
    _require(
        scores.get("crps"),
        {
            "name": "EXACT_FINITE_DISTRIBUTION_CRPS",
            "formula": (
                "SUM_I_W_I_ABS_Y_MINUS_Q_I_MINUS_HALF_SUM_IJ_W_I_W_J_"
                "ABS_Q_I_MINUS_Q_J"
            ),
            "exact_for_declared_distribution": True,
        },
        "candidate scientific CRPS changed",
    )
    _require(
        scores.get("expected_absolute_delta"),
        {
            "name": "EXACT_WEIGHTED_ABSOLUTE_ATOM_MEAN",
            "formula": "SUM_I_W_I_ABS_Q_I",
            "exact_for_declared_distribution": True,
        },
        "candidate expected-absolute estimand changed",
    )

    model = _mapping(amendment.get("candidate_model"), "candidate model missing")
    expected_candidate_count = (244 + 1) * 248 + (248 + 1) * 12
    _require(
        (
            model.get("architecture"),
            model.get("input_width"),
            model.get("hidden_width"),
            model.get("output_width"),
            model.get("adjacent_gap_formula"),
            model.get("adjacent_gap_floor"),
            model.get("median_index"),
            model.get("exact_parameter_count"),
        ),
        (
            "LINEAR_244_248_RELU_LINEAR_248_12",
            244,
            248,
            12,
            "1E_4_PLUS_SOFTPLUS_RAW",
            1.0e-4,
            6,
            expected_candidate_count,
        ),
        "candidate architecture, monotonicity, or parameter count changed",
    )

    comparator = _mapping(
        amendment.get("matched_v10_replay"), "matched V10 replay is missing"
    )
    expected_comparator_count = (244 + 1) * 256 + (256 + 1) * 4
    _require(
        (
            comparator.get("class"),
            comparator.get("architecture"),
            comparator.get("input_width"),
            comparator.get("hidden_width"),
            comparator.get("output_width"),
            comparator.get("exact_parameter_count"),
            comparator.get("newly_trained_each_authorized_fold_and_seed"),
            comparator.get("historical_v10_predictions_allowed"),
            comparator.get("predictive_distribution"),
            comparator.get("training_objective"),
            comparator.get("same_proper_scoring_rule_estimand_as_candidate"),
        ),
        (
            "scripts.reactflow_delta.model_rescue_v10.MedianAsymmetricResidual",
            "LINEAR_244_256_RELU_LINEAR_256_4",
            244,
            256,
            4,
            expected_comparator_count,
            True,
            False,
            "EXACT_TWO_GAUSSIAN_MEDIAN_PRESERVING_MIXTURE",
            "EXACT_GAUSSIAN_MIXTURE_CRPS",
            "CRPS",
        ),
        "matched V10 model, fairness, or parameter count changed",
    )
    _require(
        comparator.get("scientific_crps"),
        {
            "name": "EXACT_GAUSSIAN_MIXTURE_CRPS",
            "exact_for_declared_distribution": True,
        },
        "matched V10 scientific CRPS changed",
    )
    if expected_candidate_count != 63748 or expected_comparator_count != 63748:
        raise RuntimeError("candidate and comparator are no longer parameter matched")


def _assert_initialization(amendment: dict[str, Any]) -> None:
    init = _mapping(
        amendment.get("input_independent_initialization"),
        "input-independent initialization is missing",
    )
    _require(
        init.get("comparator"),
        {
            "construct_existing_model_first": True,
            "entire_output_layer_weight": "ALL_ZERO",
            "output_biases": {
                "mixture_weight_logit": 0.0,
                "narrow_scale_raw": "INVERSE_SOFTPLUS_0_08",
                "wide_gap_raw": "INVERSE_SOFTPLUS_0_20",
                "allocation_raw": 0.0,
            },
        },
        "input-independent V10 initialization changed",
    )
    _require(
        init.get("comparator_initial_distribution_input_independent"),
        True,
        "V10 initialization is no longer input-independent",
    )
    target = _mapping(init.get("target_grid"), "initial target grid is missing")
    _require(
        target,
        {
            "dtype": "FLOAT64",
            "method": "FIXED_BOUNDED_FLOAT64_INVERSE_CDF_BISECTION",
            "taus_source": "candidate_predictive_distribution.taus",
            "adjacent_gap_required_gt": 1.0e-4,
        },
        "initial target-grid construction changed",
    )
    gaps = _registered_initial_target_gaps()
    if len(gaps) != 12 or any(
        not math.isfinite(gap) or gap <= target["adjacent_gap_required_gt"]
        for gap in gaps
    ):
        raise RuntimeError("registered target adjacent gap is at or below 1e-4")
    _require(
        init.get("candidate"),
        {
            "entire_output_layer_weight": "ALL_ZERO",
            "output_bias_formula": (
                "INVERSE_SOFTPLUS_TARGET_ADJACENT_GAP_MINUS_1E_4"
            ),
        },
        "candidate initialization changed",
    )
    _require(
        init.get("initial_grid_replay"),
        {
            "invariant": "INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0",
            "comparison": (
                "NP_ALLCLOSE_CANDIDATE_FLOAT32_GRID_VS_REGISTERED_"
                "COMPARATOR_FLOAT64_GRID"
            ),
            "atol": 1.0e-6,
            "rtol": 0.0,
            "tolerance_scope": (
                "FIXED_FLOAT64_BISECTION_TO_FLOAT32_BIAS_AND_FORWARD_"
                "ROUND_TRIP_ONLY"
            ),
            "applies_to_point_replay": False,
            "applies_to_scientific_crps": False,
            "applies_to_any_scientific_score": False,
        },
        "INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0 mapping changed",
    )
    _require(
        (
            init.get("initial_quantile_grids_match_within_registered_tolerance"),
            init.get("full_initial_predictive_distributions_identical"),
        ),
        (True, False),
        "initial-grid or full-distribution identity claim changed",
    )


def _assert_schedule_gates_and_claim(amendment: dict[str, Any]) -> None:
    training = _mapping(amendment.get("training"), "training declaration missing")
    _require(
        training,
        {
            "both_arms_same_input_point_rows_standardizer_seed_epochs_and_puzzle_order": True,
            "optimizer": "ADAM",
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "gradient_clip": 5.0,
            "early_stopping_allowed": False,
            "best_epoch_selection_allowed": False,
            "puzzle_order_formula": "SEED_TIMES_100003_PLUS_EPOCH",
            "cuda_required_for_future_training": True,
            "cpu_fallback_allowed": False,
            "gpu_memory_gate_allowed": False,
        },
        "optimizer, pairing, epoch-order, or GPU rule changed",
    )
    _require(
        amendment.get("phase_state"),
        {
            "P2M0": "DRAFT_FROZEN_INACTIVE_PREPARATION_ONLY",
            "P2M1": "NOT_AUTHORIZED",
            "P2M2": "NOT_AUTHORIZED",
            "P2M3": "NOT_AUTHORIZED",
            "P2M4": "NOT_AUTHORIZED",
            "P2M5": "NOT_RUN",
        },
        "P2M0-P2M5 phase state changed",
    )
    _require(
        amendment.get("future_phase_tokens"),
        {
            "tokens_issued_now": False,
            "P2M1": PENDING,
            "P2M2": PENDING,
            "P2M3": PENDING,
            "P2M4": PENDING,
            "generic_training_token_allowed": False,
        },
        "future P2 phase token declaration changed",
    )
    phases = _mapping(amendment.get("phase_universes"), "phase universes missing")
    _require(
        phases,
        {
            "P2M1": {
                "purpose": "SOURCE_PROJECTION_ONLY",
                "folds": 20,
                "training_allowed": False,
                "prediction_allowed": False,
                "scoring_allowed": False,
            },
            "P2M2": {
                "purpose": "ENGINEERING_SMOKE_ONLY",
                "folds": [0, 1],
                "seeds": [0],
                "epochs": 3,
                "scientific_scorer_allowed": False,
            },
            "P2M3": {
                "purpose": "TWENTY_FOLD_SCREEN",
                "folds": FOLDS,
                "seeds": [0],
                "epochs": 40,
                "complete_prediction_only_merge_before_score": True,
                "partial_fold_score_allowed": False,
            },
            "P2M4": {
                "purpose": "FIXED_FIVE_SEED_FORMAL_CONFIRMATION",
                "folds": FOLDS,
                "seeds": [0, 1, 2, 3, 4],
                "epochs": 40,
                "seed0_retrained": True,
                "screen_predictions_reused": False,
                "expected_runs": 100,
                "failed_seed_removal_allowed": False,
                "seed_subset_or_best_seed_selection_allowed": False,
            },
        },
        "smoke, screen, or formal universe changed",
    )

    schemas = _mapping(
        amendment.get("artifact_schemas"), "artifact schema declaration missing"
    )
    _require(
        schemas,
        {
            "status": "FROZEN_INACTIVE_SCHEMA_ONLY",
            "source_manifest": (
                "reactflow_delta.post_v14_p2_quantile_residual_source_manifest.v1"
            ),
            "fold": "reactflow_delta.post_v14_p2_quantile_residual_fold.v1",
            "prediction": (
                "reactflow_delta.post_v14_p2_quantile_residual_prediction.v1"
            ),
            "merged": "reactflow_delta.post_v14_p2_quantile_residual_merged.v1",
            "score": "reactflow_delta.post_v14_p2_quantile_residual_score.v1",
            "qualification": (
                "reactflow_delta.post_v14_p2_quantile_residual_qualification.v1"
            ),
            "formal_assembly": (
                "reactflow_delta.post_v14_p2_quantile_residual_formal_assembly.v1"
            ),
            "prediction_only_forbidden_fields": [
                "HELD_TARGET",
                "HELD_ERROR",
                "HELD_QUALIFIED_TARGET_MASK",
                "SCORE",
                "PER_PUZZLE_EFFECT",
                "GATE",
                "EXTERNAL_OUTCOME",
            ],
        },
        "artifact schemas or prediction-only boundary changed",
    )

    screen = _mapping(amendment.get("p2m3_screen_gates"), "screen Gates missing")
    _require(
        screen.get("integrity"),
        {
            "registered_prediction_coverage": 1.0,
            "failure_rate": 0.0,
            "unexpected_keys": 0,
            "complete_folds": 20,
            "frozen_point_state_unchanged": True,
            "point_gradients_absent": True,
            "finite_strictly_increasing_quantiles": True,
            "candidate_median_exactly_equals_v14_point": True,
            "point_replay_atol": 1.0e-7,
            "point_replay_rtol": 0.0,
            "held_target_prediction_input_allowed": False,
            "partial_score_allowed": False,
            "external_outcome_allowed": False,
        },
        "screen integrity Gate changed",
    )
    _require(
        screen.get("copied_v14_gates"),
        {
            "binding_status": PENDING,
            "feature41_and_terminal_crps_distribution_absolute_coverage_calibration": PENDING,
            "signed_and_point_absolute_replay": PENDING,
            "relaxation_allowed": False,
        },
        "pending copied-V14 Gate binding changed",
    )
    _require(
        screen.get("matched_v10_replay"),
        {
            "crps_relative_gain_min": 0.015,
            "distribution_absolute_mae_relative_gain_min": 0.01,
            "paired_ci_level": 0.95,
            "paired_ci_lower_each_gt": 0.0,
            "t_0_975_df19": 2.093024054408263,
            "positive_puzzles_min": 14,
            "total_puzzles": 20,
            "leave_one_puzzle_positive_all_headline_comparisons": True,
            "max_single_puzzle_effect_fraction": 0.20,
            "candidate_crps": "EXACT_FINITE_DISTRIBUTION_CRPS",
            "comparator_crps": "EXACT_GAUSSIAN_MIXTURE_CRPS",
            "weighted_pinball_allowed_in_scientific_score_or_gate_fields": False,
        },
        "matched V10 replay screen Gate changed",
    )

    _require(
        amendment.get("p2m4_formal"),
        {
            "activation_requires_exact_p2m3_pass": True,
            "candidate_distribution": "EQUAL_SEED_SIXTY_FIVE_ATOM_MIXTURE",
            "candidate_atom_count": 65,
            "candidate_atom_weight_formula": "SCREEN_WEIGHT_DIVIDED_BY_5",
            "candidate_scientific_crps": "EXACT_FINITE_DISTRIBUTION_CRPS",
            "comparator_distribution": "EQUAL_FIVE_SEED_GAUSSIAN_MIXTURE",
            "comparator_scientific_crps": "EXACT_GAUSSIAN_MIXTURE_CRPS",
            "same_crps_estimand_as_screen": True,
            "average_quantile_curves_allowed": False,
            "average_per_seed_crps_allowed": False,
            "repeat_all_screen_gates": True,
            "positive_seeds_min_each_metric": 4,
            "total_seeds": 5,
            "all_runs_reported": True,
        },
        "formal 65-atom mixture or Gate changed",
    )
    _require(
        amendment.get("qualification_and_claim"),
        {
            "valid_complete_gate_failure": "SCIENTIFIC_FAIL",
            "integrity_or_provenance_failure": "INDETERMINATE",
            "smoke_proxy_or_training_loss_is_scientific_result": False,
            "maximum_pass_claim": "POST_HOC_DEVELOPMENT_FORMAL_PASS",
            "external_replication_claim_allowed": False,
            "sota_claim_allowed": False,
            "mechanism_claim_allowed": False,
            "practical_utility_claim_allowed": False,
            "publication_ready_claim_allowed": False,
            "historical_qualification_restored": False,
        },
        "qualification policy or claim ceiling changed",
    )
    forbidden = _mapping(amendment.get("forbidden_now"), "forbidden actions missing")
    if not forbidden or any(value is not True for value in forbidden.values()):
        raise RuntimeError("an inactive forbidden action was reopened")


def _assert_ledger(ledger: dict[str, Any]) -> None:
    _require(
        (
            ledger.get("schema_version"),
            ledger.get("contract_id"),
            ledger.get("status"),
        ),
        (EXPECTED_LEDGER_SCHEMA, EXPECTED_CONTRACT_ID, EXPECTED_STATUS),
        "decision ledger identity or status changed",
    )
    authority = _mapping(ledger.get("authority"), "ledger authority missing")
    expected_false = (
        "p2_authority_issued_now",
        "p2_activation_allowed",
        "p2_source_projection_allowed",
        "p2_training_allowed",
        "p2_candidate_training_allowed",
        "p2_prediction_allowed",
        "p2_smoke_allowed",
        "p2_screen_allowed",
        "p2_held_score_read_allowed",
        "p2_partial_fold_score_read_allowed",
        "p2_scoring_allowed",
        "p2_qualification_allowed",
        "p2_formal_confirmation_allowed",
        "p2_external_outcome_access_allowed",
        "p2_phase_tokens_issued_now",
        "generic_training_token_allowed",
        "active_pointer_modified_by_this_amendment",
    )
    for field in expected_false:
        if authority.get(field) is not False:
            if field == "generic_training_token_allowed":
                raise RuntimeError("ledger generic training token was allowed")
            raise RuntimeError(f"ledger inactive authority reopened: {field}")
    _require(authority.get("p2_runnable_phases"), [], "ledger opened a P2 phase")

    entry = _mapping(ledger.get("unique_entry"), "ledger parent entry missing")
    _require(
        (
            entry.get("terminal_binding_status"),
            entry.get("branch_id"),
            entry.get("classification"),
            entry.get("diagnostic_schema"),
            entry.get("diagnostic_status"),
            entry.get("diagnostic_primary_statistic"),
            entry.get("next_action"),
            entry.get("diagnostic_pass_is_route_eligibility_only"),
        ),
        (
            PENDING,
            "6",
            "DISTRIBUTION_ONLY_FAILURE",
            "reactflow_delta.post_v14_branch6_tail_diagnostic.v1",
            "POST_V14_BRANCH6_TAIL_DIAGNOSTIC_PASS",
            "LOWER_MINUS_UPPER_TAIL_MISS90",
            "OPEN_FOCUSED_P2_AMENDMENT_AUTHORITY",
            True,
        ),
        "ledger branch-6 entry changed",
    )
    pending = _mapping(ledger.get("pending_bindings"), "ledger pending bindings missing")
    if pending.get("invention_allowed") is not False or any(
        pending.get(field) != PENDING
        for field in (
            "terminal_artifact_paths",
            "realized_source_paths",
            "realized_output_paths",
            "copied_v14_feature41_terminal_gates",
            "phase_specific_tokens",
        )
    ):
        raise RuntimeError("ledger pending binding was realized or invented")

    decisions = _mapping(
        ledger.get("frozen_scientific_decisions"),
        "ledger scientific decisions missing",
    )
    _require(decisions.get("taus"), TAUS, "ledger tau array changed")
    _require(decisions.get("weights"), WEIGHTS, "ledger weight array changed")
    _require(
        (
            decisions.get("input_width"),
            decisions.get("candidate_training_surrogate"),
            decisions.get("candidate_scientific_crps"),
            decisions.get("comparator_scientific_crps"),
            decisions.get("same_proper_scoring_rule_estimand"),
            decisions.get("weighted_pinball_is_scientific_crps"),
            decisions.get("candidate_parameter_count"),
            decisions.get("comparator_parameter_count"),
        ),
        (
            244,
            "TWO_TIMES_WEIGHTED_PINBALL_ONLY",
            "EXACT_FINITE_DISTRIBUTION_CRPS",
            "EXACT_GAUSSIAN_MIXTURE_CRPS",
            "CRPS",
            False,
            63748,
            63748,
        ),
        "ledger estimand, CRPS separation, or parameter count changed",
    )
    initialization = _mapping(
        ledger.get("initialization_decision"),
        "ledger initialization decision missing",
    )
    _require(
        (
            initialization.get("comparator_entire_output_layer_weight"),
            initialization.get("comparator_output_biases"),
            initialization.get("target_adjacent_gap_required_gt"),
            initialization.get("candidate_entire_output_layer_weight"),
            initialization.get("candidate_bias_formula"),
            initialization.get("replay_invariant"),
            initialization.get("replay_atol"),
            initialization.get("replay_rtol"),
            initialization.get("point_replay_atol"),
            initialization.get("point_replay_rtol"),
            initialization.get("full_initial_predictive_distributions_identical"),
        ),
        (
            "ALL_ZERO",
            [0, "INVERSE_SOFTPLUS_0_08", "INVERSE_SOFTPLUS_0_20", 0],
            1.0e-4,
            "ALL_ZERO",
            "INVERSE_SOFTPLUS_TARGET_ADJACENT_GAP_MINUS_1E_4",
            "INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0",
            1.0e-6,
            0.0,
            1.0e-7,
            0.0,
            False,
        ),
        "ledger input-independent initialization or tolerance changed",
    )
    _require(
        ledger.get("phase_state"),
        {
            "P2M0": "DRAFT_FROZEN_INACTIVE_PREPARATION_ONLY",
            "P2M1": "NOT_AUTHORIZED",
            "P2M2": "NOT_AUTHORIZED",
            "P2M3": "NOT_AUTHORIZED",
            "P2M4": "NOT_AUTHORIZED",
            "P2M5": "NOT_RUN",
        },
        "ledger phase state changed",
    )
    screen = _mapping(
        ledger.get("screen_gate_decisions"), "ledger screen Gate decisions missing"
    )
    _require(
        (
            screen.get("copied_v14_gates"),
            screen.get("matched_v10_crps_relative_gain_min"),
            screen.get("matched_v10_distribution_absolute_mae_relative_gain_min"),
            screen.get("paired_ci_lower_each_gt"),
            screen.get("positive_puzzles_min"),
            screen.get("leave_one_puzzle_positive_all_headline_comparisons"),
            screen.get("max_single_puzzle_effect_fraction"),
        ),
        (PENDING, 0.015, 0.01, 0.0, 14, True, 0.20),
        "ledger screen Gate changed",
    )
    formal = _mapping(
        ledger.get("formal_gate_decisions"), "ledger formal Gate decisions missing"
    )
    _require(
        (
            formal.get("candidate_distribution"),
            formal.get("candidate_scientific_crps"),
            formal.get("comparator_scientific_crps"),
            formal.get("repeats_same_estimand_and_all_screen_gates"),
            formal.get("positive_seeds_min_each_metric"),
            formal.get("total_seeds"),
            formal.get("seed_selection_allowed"),
        ),
        (
            "EQUAL_SEED_SIXTY_FIVE_ATOM_MIXTURE",
            "EXACT_FINITE_DISTRIBUTION_CRPS",
            "EXACT_GAUSSIAN_MIXTURE_CRPS",
            True,
            4,
            5,
            False,
        ),
        "ledger formal mixture or Gate changed",
    )
    claim = _mapping(ledger.get("claim_boundary"), "ledger claim boundary missing")
    _require(
        claim.get("maximum_pass_claim"),
        "POST_HOC_DEVELOPMENT_FORMAL_PASS",
        "ledger claim ceiling changed",
    )
    if any(value is not False for key, value in claim.items() if key != "maximum_pass_claim"):
        raise RuntimeError("ledger claim boundary was broadened")


def validate_static_contract(
    amendment: dict[str, Any], ledger: dict[str, Any], active: dict[str, Any]
) -> dict[str, Any]:
    """Validate in-memory declarations only; perform no filesystem operations."""

    _require(
        (
            amendment.get("schema_version"),
            amendment.get("contract_id"),
            amendment.get("status"),
            amendment.get("project_task_id"),
        ),
        (EXPECTED_SCHEMA, EXPECTED_CONTRACT_ID, EXPECTED_STATUS, PROJECT_TASK_ID),
        "P2 amendment identity or status changed",
    )
    _require(amendment.get("documents"), EXPECTED_DOCUMENTS, "document paths changed")
    _require(
        amendment.get("design_provenance"),
        {
            "classification": "NEW_FOCUSED_PRE_SCORE_DESIGN_JUDGMENT",
            "historical_contract_fact": False,
            "scientific_result": False,
            "route_eligibility_is_p2_result": False,
        },
        "pre-score design provenance changed",
    )

    _assert_inactive_authority(amendment, active)
    _assert_parent_and_pending(amendment)
    _assert_input_and_point(amendment)
    _assert_distribution_and_models(amendment)
    _assert_initialization(amendment)
    _assert_schedule_gates_and_claim(amendment)
    _assert_ledger(ledger)

    return {
        "status": "POST_V14_P2_QUANTILE_INACTIVE_CONTRACT_VALIDATION_PASS",
        "contract_status": EXPECTED_STATUS,
        "branch_id": "6",
        "activation_allowed": False,
        "training_allowed": False,
        "held_score_read_allowed": False,
        "external_outcome_access_allowed": False,
        "runnable_phases": [],
        "terminal_binding_status": PENDING,
        "scientific_result": False,
    }


def validate_contract() -> dict[str, Any]:
    """Load only the three canonical repository YAML declarations."""

    amendment = _load_yaml(AMENDMENT_PATH)
    ledger = _load_yaml(LEDGER_PATH)
    active = _load_yaml(ACTIVE_POINTER_PATH)
    return validate_static_contract(amendment, ledger, active)


def main() -> int:
    result = validate_contract()
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
