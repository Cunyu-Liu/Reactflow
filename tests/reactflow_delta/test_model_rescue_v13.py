from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.reactflow_delta.model_rescue_v13 import (
    MATCHED_NULL,
    PRIMARY_CANDIDATE,
    SECOND_PASS_EXACT,
    SECOND_PASS_WT_REPLAY,
    V13PointModel,
    assert_exact_trainable_match,
    build_second_pass_sequences,
    freeze_point_model,
    make_exact_matched_pair,
    method_cell_balanced_l1,
    trainable_parameter_count,
)
from scripts.reactflow_delta.assemble_model_rescue_v13_formal import assemble_fold
from scripts.reactflow_delta.merge_model_rescue_v13 import (
    merge_folds,
    prediction_checks,
    recorded_invariants_pass,
)
from scripts.reactflow_delta.qualify_model_rescue_v13 import qualify
from scripts.reactflow_delta.qualify_model_rescue_v13_smoke import (
    qualify as qualify_smoke,
)
from scripts.reactflow_delta.qualify_model_rescue_v13_formal import (
    qualify as qualify_formal,
)
from scripts.reactflow_delta.score_model_rescue_v13 import SCHEMA as SCORE_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v13_formal import (
    SCHEMA as FORMAL_SCORE_SCHEMA,
)
from scripts.reactflow_delta.run_model_rescue_v9 import (
    _feature41_replay_max_difference,
)
from scripts.reactflow_delta.validate_model_rescue_v13_contract import (
    assert_outcome_authority_is_narrow,
    assert_run_authority,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def _context(length: int = 7) -> tuple[torch.Tensor, ...]:
    sequence = torch.nn.functional.one_hot(
        torch.arange(length) % 4, num_classes=4
    ).float()
    reactivity = torch.linspace(-0.2, 0.8, length)
    precision = torch.linspace(0.1, 1.0, length)
    observed = torch.ones(length)
    position = torch.arange(length).float()
    region = torch.zeros(length, 2)
    region[:, 0] = 1.0
    return sequence, reactivity, precision, observed, position, region


def _mutation_fixture(length: int = 7) -> dict[str, object]:
    edit = torch.tensor([1, 4])
    return {
        "edit": edit,
        "distance": (torch.arange(length)[None, :] - edit[:, None]).float(),
        "refs": ["C", "A"],
        "alts": ["G", "U"],
        "mask": torch.ones(2, length, dtype=torch.bool),
        "feature41": torch.linspace(-0.2, 0.3, length)[None, :].repeat(2, 1),
    }


def test_v13_contract_preserves_v12_and_keeps_training_closed() -> None:
    result = validate_contract(ROOT)
    assert result["status"] == "V13_CONTRACT_VALIDATION_PASS"
    assert result["phase"] in {"V13M1", "V13M2", "V13M3", "V13M4", "M6"}
    active = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    contract = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/model_rescue_v13_amendment.yaml").read_text()
    )
    assert active["parent_state"]["v12_status"] == (
        "TERMINAL_V12M3_TOP_JOURNAL_SCREEN_FAIL_DIAGNOSTICS_COMPLETE"
    )
    assert active["parent_state"]["shrinkage_gate_route"] == "TERMINATED"
    assert active["held_score_read_allowed"] in {
        False,
        "V13_COMPLETE_MERGE_SCORE_ONCE_ONLY",
        "V13_FORMAL_COMPLETE_SCORE_ONCE_ONLY",
    }
    assert contract["screen"]["signed_delta"][
        "relative_gain_vs_feature41_min"
    ] == 0.12
    assert contract["screen"]["task_crps"][
        "relative_gain_vs_feature41_min"
    ] == 0.05
    if result["phase"] == "V13M2":
        assert result["training_allowed"] == "V13_REAL_DATA_ENGINEERING_SMOKE_ONLY"
        assert active["candidate_model_training_allowed"] == (
            "V13_REAL_DATA_ENGINEERING_SMOKE_ONLY"
        )
        assert_run_authority(ROOT, "V13M2")
    elif result["phase"] == "V13M1":
        assert result["training_allowed"] is False
        assert active["authorization"]["implementation_allowed"] is True


@pytest.mark.parametrize(
    ("phase", "token"),
    (
        ("V13M3", "V13_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY"),
        ("V13M4", "V13_FIXED_FIVE_SEED_FORMAL_ONLY"),
    ),
)
def test_future_training_authority_requires_matching_primary_and_candidate_tokens(
    tmp_path: Path, phase: str, token: str
) -> None:
    config_dir = tmp_path / "configs" / "reactflow_delta"
    config_dir.mkdir(parents=True)
    active_path = config_dir / "active_contract.yaml"
    active = {
        "authority": {"current_phase": phase},
        "runnable_phases": [phase],
        "training_allowed": token,
        "candidate_model_training_allowed": token,
        "held_score_read_allowed": False,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
    }
    active_path.write_text(yaml.safe_dump(active), encoding="utf-8")
    assert_run_authority(tmp_path, phase)

    active["candidate_model_training_allowed"] = False
    active_path.write_text(yaml.safe_dump(active), encoding="utf-8")
    with pytest.raises(RuntimeError, match="candidate training authority is absent"):
        assert_run_authority(tmp_path, phase)


@pytest.mark.parametrize(
    ("phase", "token"),
    (
        ("V13M3", "V13_COMPLETE_MERGE_SCORE_ONCE_ONLY"),
        ("V13M4", "V13_FORMAL_COMPLETE_SCORE_ONCE_ONLY"),
    ),
)
def test_complete_score_authority_requires_training_closed_and_exact_token(
    phase: str, token: str
) -> None:
    active = {
        "authority": {"current_phase": phase},
        "runnable_phases": [phase],
        "training_allowed": False,
        "candidate_model_training_allowed": False,
        "held_score_read_allowed": token,
        "partial_fold_score_read_allowed": False,
        "new_external_outcome_access_allowed": False,
        "v12_terminal_verdict_change_allowed": False,
    }
    assert_outcome_authority_is_narrow(active)

    active["training_allowed"] = "V13_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY"
    with pytest.raises(RuntimeError, match="not the frozen training-closed step"):
        assert_outcome_authority_is_narrow(active)

    active["training_allowed"] = False
    active["held_score_read_allowed"] = f"{token}_CHANGED"
    with pytest.raises(RuntimeError, match="not the frozen training-closed step"):
        assert_outcome_authority_is_narrow(active)


def test_candidate_and_null_are_exact_parameter_matches() -> None:
    candidate, null = make_exact_matched_pair(seed=17, device="cpu")
    assert candidate.second_pass_mode == SECOND_PASS_EXACT
    assert null.second_pass_mode == SECOND_PASS_WT_REPLAY
    assert_exact_trainable_match(candidate, null)
    assert trainable_parameter_count(candidate) == trainable_parameter_count(null)
    assert PRIMARY_CANDIDATE != MATCHED_NULL


def test_exact_mutant_changes_one_registered_token_and_null_changes_none() -> None:
    context = _context()
    fixture = _mutation_fixture()
    exact = build_second_pass_sequences(
        context[0],
        fixture["edit"],
        fixture["refs"],
        fixture["alts"],
        mode=SECOND_PASS_EXACT,
    )
    replay = build_second_pass_sequences(
        context[0],
        fixture["edit"],
        fixture["refs"],
        fixture["alts"],
        mode=SECOND_PASS_WT_REPLAY,
    )
    wt = context[0].unsqueeze(0).expand(2, -1, -1)
    assert torch.equal((exact != wt).any(dim=-1).sum(dim=-1), torch.ones(2, dtype=torch.long))
    assert torch.equal(replay, wt)
    assert exact[0, 1].argmax().item() == 2
    assert exact[1, 4].argmax().item() == 3


def test_registered_ref_mismatch_fails_closed() -> None:
    context = _context()
    fixture = _mutation_fixture()
    with pytest.raises(ValueError, match="registered ref"):
        build_second_pass_sequences(
            context[0],
            fixture["edit"],
            ["A", "A"],
            fixture["alts"],
            mode=SECOND_PASS_EXACT,
        )


def test_wt_replay_hidden_delta_is_zero_in_evaluation_mode() -> None:
    _candidate, null = make_exact_matched_pair(seed=23, device="cpu")
    null.eval()
    context = _context()
    fixture = _mutation_fixture()
    with torch.no_grad():
        wt, replay = null.encode_paired_passes(
            context,
            fixture["edit"],
            fixture["refs"],
            fixture["alts"],
        )
    assert torch.allclose(replay, wt, atol=1e-7, rtol=0.0)


def test_paired_encoder_dropout_cannot_create_a_counterfactual_difference() -> None:
    candidate, _null = make_exact_matched_pair(seed=27, device="cpu")
    candidate.train()
    context = _context()
    fixture = _mutation_fixture()
    refs = fixture["refs"]
    with torch.no_grad():
        wt, identical = candidate.encode_paired_passes(
            context,
            fixture["edit"],
            refs,
            refs,
        )
    assert torch.equal(identical, wt)


def test_pre_shared_dropout_fold_invariants_are_not_qualified() -> None:
    invariants = {
        "target_profile_identity_exact": True,
        "exact_point_parameter_and_initial_state_match": True,
        "second_pass_sequence_is_only_candidate_null_difference": True,
        "candidate_exact_mutant_null_wt_replay": True,
        "null_hidden_delta_at_most_1e_7": True,
        "same_point_training_order_and_dropout_seed": True,
        "paired_encoder_dropout_mask_shared": True,
        "point_frozen_during_calibration": True,
        "v10_residual_family_reused": True,
        "feature41_replay_at_1e_7": True,
        "median_constraint_all_held_rows": True,
        "held_score_computed": False,
        "prediction_contains_target_fields": False,
        "external_outcome_accessed": False,
    }
    assert recorded_invariants_pass(invariants)
    invariants.pop("paired_encoder_dropout_mask_shared")
    assert not recorded_invariants_pass(invariants)


def test_exact_mutant_produces_receiver_shaped_counterfactual_signal() -> None:
    candidate, _null = make_exact_matched_pair(seed=29, device="cpu")
    candidate.eval()
    context = _context()
    fixture = _mutation_fixture()
    with torch.no_grad():
        wt, mutant = candidate.encode_paired_passes(
            context,
            fixture["edit"],
            fixture["refs"],
            fixture["alts"],
        )
    delta = mutant - wt
    assert delta.shape == (2, 7, 192)
    assert torch.count_nonzero(delta).item() > 0
    assert torch.isfinite(delta).all()


def test_zero_initialized_heads_make_feature41_the_shared_initial_point() -> None:
    candidate, null = make_exact_matched_pair(seed=31, device="cpu")
    candidate.eval()
    null.eval()
    context = _context()
    fixture = _mutation_fixture()
    with torch.no_grad():
        candidate_point = candidate.forward_point(
            context,
            fixture["edit"],
            fixture["distance"],
            fixture["refs"],
            fixture["alts"],
            fixture["mask"],
            fixture["feature41"],
        )
        null_point = null.forward_point(
            context,
            fixture["edit"],
            fixture["distance"],
            fixture["refs"],
            fixture["alts"],
            fixture["mask"],
            fixture["feature41"],
        )
    assert torch.equal(candidate_point, fixture["feature41"])
    assert torch.equal(null_point, fixture["feature41"])


def test_prediction_path_cannot_accept_target_bearing_inputs() -> None:
    signature = inspect.signature(V13PointModel.forward_point_and_features)
    for forbidden in (
        "target",
        "target_error",
        "qualified_mask",
        "method_id",
        "puzzle_id",
    ):
        assert forbidden not in signature.parameters


def test_reused_feature41_replay_helper_has_the_frozen_seven_argument_interface() -> None:
    assert tuple(inspect.signature(_feature41_replay_max_difference).parameters) == (
        "univ",
        "held_records",
        "feature41_model",
        "unconstrained",
        "constrained",
        "reference_path",
        "fold_id",
    )


def test_method_balanced_loss_does_not_pool_positions() -> None:
    point = torch.zeros(2, 3)
    target = torch.tensor([[1.0, 1.0, 1.0], [3.0, 0.0, 0.0]])
    qualified = torch.tensor([[True, True, True], [True, False, False]])
    loss = method_cell_balanced_l1(point, target, qualified, torch.zeros(3))
    assert torch.allclose(loss, torch.tensor(2.0))


def test_point_freeze_clears_gradients_and_disables_updates() -> None:
    candidate, _null = make_exact_matched_pair(seed=37, device="cpu")
    context = _context()
    fixture = _mutation_fixture()
    point = candidate.forward_point(
        context,
        fixture["edit"],
        fixture["distance"],
        fixture["refs"],
        fixture["alts"],
        fixture["mask"],
        fixture["feature41"],
    )
    point.abs().mean().backward()
    assert any(parameter.grad is not None for parameter in candidate.parameters())
    freeze_point_model(candidate)
    assert candidate.training is False
    assert all(parameter.requires_grad is False for parameter in candidate.parameters())
    assert all(parameter.grad is None for parameter in candidate.parameters())


def test_prediction_checks_reject_target_bearing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "prediction.npz"
    rows = 2
    common = {
        "schema_version": np.asarray("reactflow_delta.model_rescue_v13_prediction.v1"),
        "keys": np.asarray(["a", "b"], dtype=object),
        "biological_scoring_key": np.asarray(["a", "b"], dtype=object),
        "outer_fold": np.zeros(rows, dtype=np.int64),
        "seed": np.zeros(rows, dtype=np.int64),
        "registered_status": np.full(rows, "covered", dtype=object),
        "feature41_point": np.zeros(rows),
        "candidate_point": np.zeros(rows),
        "null_point": np.zeros(rows),
        "null_hidden_delta_max_abs": np.zeros(rows),
    }
    for name in ("candidate", "null"):
        common[f"{name}_weights"] = np.full((rows, 2), 0.5)
        common[f"{name}_locations"] = np.zeros((rows, 2))
        common[f"{name}_scales"] = np.ones((rows, 2))
        common[f"{name}_expected_absolute_delta"] = np.ones(rows)
    common["target"] = np.zeros(rows)
    np.savez_compressed(path, **common)
    checks = prediction_checks(path, fold=0, seed=0, expected_rows=rows)
    assert checks["target_free"] is False


def test_complete_merger_rejects_incomplete_v13_universe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="universe is incomplete"):
        merge_folds(tmp_path, "V13M3")


def _passing_complete_score() -> dict[str, object]:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "feature41_signed_delta_mae": 1.0,
                "terminal_v12_signed_delta_mae": 0.90,
                "null_signed_delta_mae": 0.88,
                "candidate_signed_delta_mae": 0.80,
                "feature41_absolute_delta_mae": 1.0,
                "terminal_v11_point_absolute_delta_mae": 0.90,
                "null_point_absolute_delta_mae": 0.88,
                "candidate_point_absolute_delta_mae": 0.80,
                "terminal_v10_distribution_absolute_delta_mae": 0.90,
                "null_distribution_absolute_delta_mae": 0.88,
                "candidate_distribution_absolute_delta_mae": 0.80,
                "feature41_crps": 1.0,
                "terminal_v12_crps": 0.90,
                "null_crps": 0.88,
                "candidate_crps": 0.80,
                "feature41_coverage68": 0.68,
                "candidate_coverage68": 0.68,
                "feature41_coverage95": 0.95,
                "candidate_coverage95": 0.95,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
            }
        )
    return {
        "schema_version": SCORE_SCHEMA,
        "status": "V13M3_COMPLETE_SCORE_PASS",
        "scores": rows,
    }


def test_qualifier_requires_all_high_point_probability_and_attribution_gates() -> None:
    result = qualify(_passing_complete_score())
    assert result["status"] == "V13M3_TOP_JOURNAL_SCREEN_PASS"
    assert result["gate_passed"] is True
    assert all(result["gates"].values())

    failing = _passing_complete_score()
    for row in failing["scores"]:
        row["candidate_crps"] = 0.96
    result = qualify(failing)
    assert result["status"] == "V13M3_TOP_JOURNAL_SCREEN_FAIL"
    assert result["gates"]["task_crps_gain_vs_feature41_ge_5pct"] is False
    assert result["v13m4_authorized"] is False


def test_smoke_qualifier_cannot_create_scientific_pass() -> None:
    merged = {
        "schema_version": "reactflow_delta.model_rescue_v13_merged.v1",
        "status": "V13M2_COMPLETE_UNSCORED_SMOKE_MERGE_PASS",
        "folds": [
            {
                "outer_fold": fold,
                "seed": 0,
                "evidence_status": "ENGINEERING_SMOKE_ONLY",
                "history_lengths": {"candidate_point": 3, "null_point": 3},
            }
            for fold in (0, 1)
        ],
        "merge_integrity": {
            "complete_fold_seed_universe": True,
            "prediction_only_schema": True,
            "exact_point_parameter_and_initial_state_match_all_runs": True,
            "second_pass_only_difference_all_runs": True,
            "null_hidden_delta_at_most_1e_7_all_runs": True,
            "paired_encoder_dropout_mask_shared_all_runs": True,
            "point_frozen_during_calibration_all_runs": True,
            "median_constraint_all_runs": True,
            "partial_scores_inspected": False,
            "external_outcome_accessed": False,
        },
    }
    result = qualify_smoke(merged)
    assert result["status"] == "V13M2_ENGINEERING_SMOKE_PASS"
    assert result["scientific_score_computed"] is False


def test_formal_assembler_uses_all_five_seeds_with_equal_mass(tmp_path: Path) -> None:
    rows = []
    for seed in range(5):
        path = tmp_path / f"seed{seed}.npz"
        payload = {
            "schema_version": np.asarray("reactflow_delta.model_rescue_v13_prediction.v1"),
            "keys": np.asarray(["a", "b"], dtype=object),
            "outer_fold": np.zeros(2, dtype=np.int64),
            "seed": np.full(2, seed, dtype=np.int64),
            "feature41_point": np.asarray([0.1, -0.2]),
            "candidate_point": np.asarray([0.2 + seed * 0.01, -0.1]),
            "null_point": np.asarray([0.15, -0.15]),
        }
        for name in ("feature41", "candidate", "null"):
            payload[f"{name}_weights"] = np.full((2, 2), 0.5)
            payload[f"{name}_locations"] = np.zeros((2, 2))
            payload[f"{name}_scales"] = np.ones((2, 2))
            payload[f"{name}_expected_absolute_delta"] = np.ones(2)
        np.savez_compressed(path, **payload)
        rows.append({"seed": seed, "prediction_artifact": str(path)})
    result = assemble_fold(rows, fold=0, out_dir=tmp_path)
    with np.load(result["prediction_artifact"], allow_pickle=True) as handle:
        assert handle["candidate_weights"].shape == (2, 10)
        assert np.allclose(handle["candidate_weights"].sum(axis=1), 1.0)
        assert np.allclose(handle["candidate_point"], [0.22, -0.1])


def test_formal_qualifier_requires_screen_pass_and_four_positive_seeds() -> None:
    screen = qualify(_passing_complete_score())
    mixture = _passing_complete_score()["scores"]
    formal = {
        "schema_version": FORMAL_SCORE_SCHEMA,
        "status": "V13M4_COMPLETE_FORMAL_SCORE_PASS",
        "mixture_scores": mixture,
        "individual_seed_scores": {str(seed): mixture for seed in range(5)},
        "equal_seed_mixture": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }
    result = qualify_formal(formal, screen)
    assert result["status"] == "V13M4_TOP_JOURNAL_FORMAL_PASS"
    assert result["gate_passed"] is True

    formal["individual_seed_scores"]["4"] = [
        {**row, "candidate_crps": 1.1, "candidate_signed_delta_mae": 1.1}
        for row in mixture
    ]
    formal["individual_seed_scores"]["3"] = formal["individual_seed_scores"]["4"]
    result = qualify_formal(formal, screen)
    assert result["gate_passed"] is False
    assert result["gates"]["signed_positive_individual_seeds_ge_4"] is False
