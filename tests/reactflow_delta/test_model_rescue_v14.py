from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import torch

from scripts.reactflow_delta.model_rescue_v14 import (
    EXPECTED_TOTAL_PARAMETERS,
    V14PointModel,
    assert_exact_initial_match,
    assert_exact_parameter_contract,
    assert_snapshot_equal,
    deterministic_corruption_mask,
    make_exact_matched_pair,
    module_snapshot,
    parameter_count,
    pretrain_wt_encoder,
)
from scripts.reactflow_delta.merge_model_rescue_v14 import (
    prediction_checks,
    recorded_invariants_pass,
)
from scripts.reactflow_delta.assemble_model_rescue_v14_formal import assemble_fold
from scripts.reactflow_delta.qualify_model_rescue_v14 import qualify
from scripts.reactflow_delta.score_model_rescue_v14 import SCHEMA as SCORE_SCHEMA
from scripts.reactflow_delta.validate_model_rescue_v14_contract import (
    assert_run_authority,
)


ROOT = Path(__file__).resolve().parents[2]


def _context(length: int = 8) -> tuple[torch.Tensor, ...]:
    sequence = torch.eye(4).repeat((length + 3) // 4, 1)[:length]
    reactivity = torch.linspace(-1.0, 1.0, length)
    precision = torch.linspace(0.5, 1.0, length)
    observed = torch.ones(length)
    position = torch.arange(length, dtype=torch.float32)
    region = torch.zeros(length, 2)
    region[:, 0] = 1.0
    return sequence, reactivity, precision, observed, position, region


def test_v14_exact_parameter_contract_and_common_initialization() -> None:
    candidate, null = make_exact_matched_pair(seed=14, device="cpu")
    assert parameter_count(candidate) == EXPECTED_TOTAL_PARAMETERS
    assert parameter_count(null) == EXPECTED_TOTAL_PARAMETERS
    assert_exact_parameter_contract(candidate)
    assert_exact_initial_match(candidate, null)


def test_v14_mask_is_deterministic_observed_only_and_removes_target_value() -> None:
    context = _context()
    observed = context[3].clone()
    observed[-1] = 0.0
    context = (*context[:3], observed, *context[4:])
    mask = deterministic_corruption_mask(
        observed, seed=3, epoch=2, construct_index=1
    )
    repeated = deterministic_corruption_mask(
        observed, seed=3, epoch=2, construct_index=1
    )
    assert torch.equal(mask, repeated)
    assert int(mask.sum()) == 3
    assert not bool(mask[-1])

    model = V14PointModel().eval()
    modified = list(context)
    modified_reactivity = modified[1].clone()
    modified_reactivity[mask] += 1000.0
    modified[1] = modified_reactivity
    with torch.no_grad():
        original_hidden = model.encode(context, mask)
        modified_hidden = model.encode(tuple(modified), mask)
    assert torch.equal(original_hidden, modified_hidden)


def test_v14_pretraining_changes_only_candidate_encoder_and_decoder() -> None:
    candidate, null = make_exact_matched_pair(seed=7, device="cpu")
    null_before = module_snapshot(null)
    residual_before = module_snapshot(candidate.residual_head)
    encoder_before = module_snapshot(candidate.input_projection)
    history = pretrain_wt_encoder(
        candidate, {"construct-0": _context()}, epochs=1, seed=7
    )
    assert len(history) == 1
    assert_snapshot_equal(null_before, null, "from-scratch null")
    assert_snapshot_equal(residual_before, candidate.residual_head, "candidate residual")
    assert any(
        not torch.equal(value, candidate.input_projection.state_dict()[name])
        for name, value in encoder_before.items()
    )
    # The null is still the original common state, while only the candidate
    # encoder/decoder was permitted to move.
    residual_null = copy.deepcopy(null.residual_head).state_dict()
    for name, value in candidate.residual_head.state_dict().items():
        assert torch.equal(value, residual_null[name])


def test_v14_active_authority_opens_only_implementation() -> None:
    assert_run_authority(ROOT, "V14M1")


def test_v14_prediction_path_has_no_target_or_identity_shortcut_inputs() -> None:
    signature = inspect.signature(V14PointModel.forward_point_and_features)
    for forbidden in (
        "target",
        "target_error",
        "qualified_mask",
        "method_id",
        "puzzle_id",
        "dataset_id",
    ):
        assert forbidden not in signature.parameters


def _prediction_payload(rows: int = 2) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray("reactflow_delta.model_rescue_v14_prediction.v1"),
        "keys": np.asarray([f"k{i}" for i in range(rows)], dtype=object),
        "biological_scoring_key": np.asarray(
            [f"k{i}" for i in range(rows)], dtype=object
        ),
        "outer_fold": np.zeros(rows, dtype=np.int64),
        "seed": np.zeros(rows, dtype=np.int64),
        "registered_status": np.full(rows, "covered", dtype=object),
        "feature41_point": np.zeros(rows),
        "candidate_point": np.zeros(rows),
        "null_point": np.zeros(rows),
    }
    for name in ("feature41", "candidate", "null"):
        payload[f"{name}_weights"] = np.full((rows, 2), 0.5)
        payload[f"{name}_locations"] = np.zeros((rows, 2))
        payload[f"{name}_scales"] = np.ones((rows, 2))
    payload["candidate_expected_absolute_delta"] = np.ones(rows)
    payload["null_expected_absolute_delta"] = np.ones(rows)
    return payload


def test_v14_prediction_checks_reject_target_bearing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "prediction.npz"
    payload = _prediction_payload()
    payload["target"] = np.zeros(2)
    np.savez_compressed(path, **payload)
    checks = prediction_checks(path, fold=0, seed=0, expected_rows=2)
    assert checks["target_free"] is False


def test_v14_merge_invariants_require_pretraining_isolation() -> None:
    invariants = {
        "target_profile_identity_exact": True,
        "outer_train_wt_only_pretraining": True,
        "held_puzzle_wt_excluded_from_pretraining": True,
        "mutant_outcome_excluded_from_pretraining": True,
        "exact_initial_parameter_match": True,
        "exact_total_and_downstream_parameter_match": True,
        "candidate_encoder_changed_during_pretraining": True,
        "null_state_unchanged_before_supervised_training": True,
        "residual_heads_identical_before_supervised_step_one": True,
        "pretraining_decoder_frozen_downstream": True,
        "same_point_training_order_and_dropout_stream": True,
        "point_frozen_during_calibration": True,
        "v10_residual_family_reused": True,
        "feature41_replay_at_1e_7": True,
        "median_constraint_all_held_rows": True,
        "held_score_computed": False,
        "prediction_contains_target_fields": False,
        "external_outcome_accessed": False,
    }
    assert recorded_invariants_pass(invariants)
    invariants["held_puzzle_wt_excluded_from_pretraining"] = False
    assert not recorded_invariants_pass(invariants)


def test_v14_qualifier_reuses_strict_top_journal_gates() -> None:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
                "feature41_signed_delta_mae": 1.0,
                "terminal_v12_signed_delta_mae": 0.9,
                "null_signed_delta_mae": 0.8,
                "candidate_signed_delta_mae": 0.5,
                "feature41_absolute_delta_mae": 1.0,
                "terminal_v11_point_absolute_delta_mae": 0.9,
                "null_point_absolute_delta_mae": 0.8,
                "candidate_point_absolute_delta_mae": 0.5,
                "terminal_v10_distribution_absolute_delta_mae": 0.9,
                "null_distribution_absolute_delta_mae": 0.8,
                "candidate_distribution_absolute_delta_mae": 0.5,
                "feature41_crps": 1.0,
                "terminal_v12_crps": 0.9,
                "null_crps": 0.8,
                "candidate_crps": 0.5,
                "feature41_coverage68": 0.68,
                "candidate_coverage68": 0.68,
                "feature41_coverage95": 0.95,
                "candidate_coverage95": 0.95,
            }
        )
    result = qualify(
        {
            "schema_version": SCORE_SCHEMA,
            "status": "V14M3_COMPLETE_SCORE_PASS",
            "scores": rows,
        }
    )
    assert result["status"] == "V14M3_TOP_JOURNAL_SCREEN_PASS"
    assert result["v14m4_authorized"] is True


def test_v14_formal_assembly_uses_all_five_seeds_equally(tmp_path: Path) -> None:
    rows = []
    for seed in range(5):
        payload = _prediction_payload(rows=1)
        payload["seed"] = np.full(1, seed, dtype=np.int64)
        payload["candidate_point"] = np.asarray([float(seed)])
        path = tmp_path / f"seed{seed}.npz"
        np.savez_compressed(path, **payload)
        rows.append({"seed": seed, "prediction_artifact": str(path)})
    result = assemble_fold(rows, fold=0, out_dir=tmp_path)
    assert result["components_per_distribution"] == 10
    with np.load(result["prediction_artifact"], allow_pickle=True) as handle:
        assert float(handle["candidate_point"][0]) == 2.0
        assert np.allclose(handle["candidate_weights"].sum(axis=1), 1.0)
        assert handle["candidate_weights"].shape == (1, 10)
