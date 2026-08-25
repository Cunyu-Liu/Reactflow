from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.reactflow_delta.assemble_model_rescue_v11_formal import assemble_fold
from scripts.reactflow_delta.diagnose_model_rescue_v11 import (
    assert_diagnostic_authority,
    convergence_diagnostic,
    directional_summary,
    distribution_diagnostic,
    method_balanced_weights,
    puzzle_macro_from_mutant_losses,
    summarize_train_held_gap,
)
from scripts.reactflow_delta.model_rescue_v11 import (
    ATTENTION_HEADS,
    CONTEXT_BLOCKS,
    CONTEXT_WIDTH,
    FFN_WIDTH,
    HEAD_WIDTH,
    V11PointModel,
    assert_exact_trainable_match,
    make_exact_matched_pair,
    method_cell_balanced_l1,
    trainable_parameter_count,
)
from scripts.reactflow_delta.merge_model_rescue_v11 import (
    authoritative_comparator_invariant_pass,
)
from scripts.reactflow_delta.qualify_model_rescue_v11 import (
    SCHEMA as SCREEN_QUALIFICATION_SCHEMA,
)
from scripts.reactflow_delta.qualify_model_rescue_v11 import qualify
from scripts.reactflow_delta.qualify_model_rescue_v11_formal import (
    qualify as qualify_formal,
)
from scripts.reactflow_delta.run_model_rescue_v11 import (
    _apply_authoritative_feature41_distribution,
    assert_run_authority,
    fit_point_model,
)
from scripts.reactflow_delta.score_model_rescue_v11 import SCHEMA as SCORE_SCHEMA
from scripts.reactflow_delta.score_model_rescue_v11_formal import (
    SCHEMA as FORMAL_SCORE_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[2]


def _context(length: int) -> tuple[torch.Tensor, ...]:
    sequence = torch.nn.functional.one_hot(
        torch.arange(length) % 4, num_classes=4
    ).float()
    reactivity = torch.linspace(-1.0, 1.0, length)
    precision = torch.linspace(0.1, 0.8, length)
    observed = torch.ones(length)
    position = torch.arange(length).float()
    region = torch.zeros(length, 2)
    region[:, 0] = 1.0
    return sequence, reactivity, precision, observed, position, region


def test_v11_contract_keeps_parent_terminal_and_enforces_active_authority_exclusion() -> None:
    active = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    contract = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/model_rescue_v11_amendment.yaml").read_text()
    )
    phase = active["authority"]["current_phase"]
    assert phase in {"V11M3", "V11M4"}
    assert active["runnable_phases"] == [phase]
    assert active["gate_state"]["V11M2"] == "V11M2_ENGINEERING_SMOKE_PASS"
    if active["training_allowed"] is False:
        assert active["candidate_model_training_allowed"] is False
    else:
        expected_training = {
            "V11M3": "V11_TWENTY_FOLD_PREDICTION_ONLY_SCREEN_ONLY",
            "V11M4": "V11_FIXED_FIVE_SEED_FORMAL_ONLY",
        }[phase]
        assert active["training_allowed"] == expected_training
        assert active["held_score_read_allowed"] is False
        assert_run_authority(ROOT, phase)
    assert active["partial_fold_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
    assert contract["parent"]["v10_terminal_status"] == (
        "TERMINAL_TOP_JOURNAL_TASK_CRPS_MARGIN_FAIL"
    )
    assert contract["parent"]["v10_formal_opened"] is False
    assert contract["comparators"][
        "seed0_feature41_authoritative_checkpoint_and_prediction_reuse"
    ] is True
    assert contract["comparators"]["seed0_feature41_retraining_required"] is False
    gates = contract["v11m3_screen"]["gates"]
    assert gates["signed_delta_relative_gain_vs_feature41_min"] == 0.10
    assert gates["task_crps_relative_gain_vs_feature41_asymmetric_min"] == 0.05
    assert gates["task_crps_relative_gain_vs_terminal_v10_min"] == 0.015


def test_primary_and_null_have_exact_trainable_state_and_parameter_count() -> None:
    primary, null = make_exact_matched_pair(seed=17, device="cpu")
    assert_exact_trainable_match(primary, null)
    assert trainable_parameter_count(primary) == trainable_parameter_count(null)
    assert len(primary.blocks) == CONTEXT_BLOCKS
    assert primary.blocks[0].heads == ATTENTION_HEADS
    assert primary.input_projection.out_features == CONTEXT_WIDTH
    assert primary.blocks[0].ffn[0].out_features == FFN_WIDTH
    assert primary.residual_head[0].out_features == HEAD_WIDTH


def test_zero_initialized_residual_makes_skip_the_only_initial_output_difference() -> None:
    primary, null = make_exact_matched_pair(seed=23, device="cpu")
    primary.eval()
    null.eval()
    length = 7
    feature41 = torch.linspace(-0.2, 0.3, length)[None, :].repeat(2, 1)
    edit = torch.tensor([1, 4])
    distance = torch.arange(length)[None, :] - edit[:, None]
    mask = torch.ones(2, length, dtype=torch.bool)
    with torch.no_grad():
        primary_hidden = primary.encode(_context(length))
        null_hidden = null.encode(_context(length))
        anchored = primary.forward_point(
            primary_hidden,
            edit,
            distance.float(),
            ["A", "C"],
            ["G", "U"],
            mask,
            feature41,
        )
        unanchored = null.forward_point(
            null_hidden,
            edit,
            distance.float(),
            ["A", "C"],
            ["G", "U"],
            mask,
            feature41,
        )
    assert torch.equal(primary_hidden, null_hidden)
    assert torch.allclose(anchored, feature41, atol=0.0, rtol=0.0)
    assert torch.count_nonzero(unanchored) == 0


def test_seed0_feature41_distribution_replays_authoritative_v10_exactly() -> None:
    output = {
        f"feature41_{suffix}": np.zeros((2, 2), dtype=np.float64)
        for suffix in ("weights", "locations", "scales", "expected_absolute_delta")
    }
    historical = {
        "feature41_weights": np.asarray([[0.25, 0.75], [0.4, 0.6]]),
        "feature41_locations": np.asarray([[-0.1, 0.2], [-0.2, 0.1]]),
        "feature41_scales": np.asarray([[0.1, 0.3], [0.2, 0.4]]),
        "feature41_expected_absolute_delta": np.asarray([0.2, 0.3]),
    }
    _apply_authoritative_feature41_distribution(output, historical)
    for suffix in ("weights", "locations", "scales", "expected_absolute_delta"):
        assert np.array_equal(
            output[f"feature41_{suffix}"], historical[f"feature41_{suffix}"]
        )
        assert output[f"feature41_{suffix}"] is not historical[f"feature41_{suffix}"]


def test_scientific_merge_rejects_old_retrained_comparator_artifacts() -> None:
    old = {"feature41_asymmetric_screen_replay_or_not_applicable": True}
    current = {
        "feature41_asymmetric_seed0_uses_authoritative_v10_or_not_applicable": True
    }
    assert authoritative_comparator_invariant_pass("V11M2", old) is True
    assert authoritative_comparator_invariant_pass("V11M3", old) is False
    assert authoritative_comparator_invariant_pass("V11M3", current) is True
    assert authoritative_comparator_invariant_pass("V11M4", current) is True


def test_point_prediction_does_not_accept_target_error_or_target_mask() -> None:
    parameters = V11PointModel.forward_point_and_features.__annotations__
    assert "target" not in parameters
    assert "target_error" not in parameters
    assert "qualified_mask" not in parameters


def test_method_cell_loss_weights_mutants_not_positions() -> None:
    point = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([[1.0, 1.0, 1.0], [3.0, 0.0, 0.0]])
    qualified = torch.tensor(
        [[True, True, True], [True, False, False]], dtype=torch.bool
    )
    wt = torch.zeros(3)
    loss = method_cell_balanced_l1(point, target, qualified, wt)
    # Mutant means are 1 and 3.  Equal-mutant weighting is 2, whereas pooled
    # positions would be 1.5.
    assert torch.allclose(loss, torch.tensor(2.0))


def test_point_training_path_has_finite_gradients_and_complete_history() -> None:
    primary, _null = make_exact_matched_pair(seed=5, device="cpu")
    length = 5
    edit = torch.tensor([1, 3])
    distance = torch.arange(length)[None, :] - edit[:, None]
    cells = [
        {
            "construct_id": "fixture",
            "edit": edit,
            "distance": distance.float(),
            "refs": ["A", "C"],
            "alts": ["G", "U"],
            "target": torch.tensor(
                [[0.1, 0.0, -0.1, 0.2, 0.0], [0.0, 0.2, 0.1, -0.2, 0.3]]
            ),
            "prediction_mask": torch.ones(2, length, dtype=torch.bool),
            "qualified_mask": torch.ones(2, length, dtype=torch.bool),
            "wt": torch.zeros(length),
            "feature41_point": torch.zeros(2, length),
        }
    ]
    history = fit_point_model(
        primary,
        cells,
        {"fixture": _context(length)},
        epochs=1,
        seed=5,
    )
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history)).all()


def test_zero_initialized_head_allows_backbone_updates_after_first_cell() -> None:
    primary, _null = make_exact_matched_pair(seed=41, device="cpu")
    before = {
        name: parameter.detach().clone()
        for name, parameter in primary.named_parameters()
    }
    length = 5
    edit = torch.tensor([1, 3])
    distance = torch.arange(length)[None, :] - edit[:, None]
    cells = []
    contexts = {}
    for construct_id, shift in (("a", 0.0), ("b", 0.15)):
        cells.append(
            {
                "construct_id": construct_id,
                "edit": edit,
                "distance": distance.float(),
                "refs": ["A", "C"],
                "alts": ["G", "U"],
                "target": torch.tensor(
                    [
                        [0.3 + shift, 0.1, -0.2, 0.2, 0.0],
                        [0.0, 0.2, 0.3 + shift, -0.1, 0.25],
                    ]
                ),
                "prediction_mask": torch.ones(2, length, dtype=torch.bool),
                "qualified_mask": torch.ones(2, length, dtype=torch.bool),
                "wt": torch.zeros(length),
                "feature41_point": torch.zeros(2, length),
            }
        )
        contexts[construct_id] = _context(length)
    fit_point_model(primary, cells, contexts, epochs=1, seed=41)
    changed = {
        name: not torch.equal(before[name], parameter.detach())
        for name, parameter in primary.named_parameters()
    }
    assert any(
        value
        for name, value in changed.items()
        if name.startswith(("input_projection", "blocks", "output_norm"))
    )
    assert all(changed.values())


def test_unobserved_positions_and_same_base_mutations_are_zeroed() -> None:
    model = V11PointModel(feature41_skip_multiplier=1.0).eval()
    length = 5
    feature41 = torch.ones(2, length)
    edit = torch.tensor([1, 2])
    distance = torch.arange(length)[None, :] - edit[:, None]
    mask = torch.ones(2, length, dtype=torch.bool)
    mask[1, 4] = False
    with torch.no_grad():
        hidden = model.encode(_context(length))
        point = model.forward_point(
            hidden,
            edit,
            distance.float(),
            ["A", "C"],
            ["A", "U"],
            mask,
            feature41,
        )
    assert torch.count_nonzero(point[0]) == 0
    assert point[1, 4] == 0.0


def _complete_score(*, anchored_crps: float = 0.12) -> dict:
    rows = []
    for fold in range(20):
        rows.append(
            {
                "outer_fold": fold,
                "held_puzzle": f"P{fold + 1:02d}",
                "feature41_signed_delta_mae": 0.20,
                "v8_signed_delta_mae": 0.18,
                "anchored_signed_delta_mae": 0.17,
                "unanchored_signed_delta_mae": 0.19,
                "feature41_absolute_delta_mae": 0.15,
                "anchored_point_absolute_delta_mae": 0.14,
                "unanchored_point_absolute_delta_mae": 0.145,
                "anchored_distribution_absolute_delta_mae": 0.13,
                "unanchored_distribution_absolute_delta_mae": 0.14,
                "historical_v10_distribution_absolute_delta_mae": 0.131,
                "feature41_crps": 0.13,
                "anchored_crps": anchored_crps,
                "unanchored_crps": 0.128,
                "historical_v10_crps": 0.125,
                "feature41_coverage68": 0.68,
                "anchored_coverage68": 0.68,
                "feature41_coverage95": 0.95,
                "anchored_coverage95": 0.95,
                "registered_prediction_coverage": 1.0,
                "failure_rate": 0.0,
                "n_unexpected_prediction_keys": 0,
            }
        )
    return {
        "schema_version": SCORE_SCHEMA,
        "status": "V11M3_COMPLETE_SCORE_PASS",
        "scores": rows,
    }


def test_qualifier_requires_every_prefrozen_top_journal_gate() -> None:
    passed = qualify(_complete_score())
    assert passed["status"] == "V11M3_TOP_JOURNAL_SCREEN_PASS"
    assert passed["gate_passed"] is True
    failed = qualify(_complete_score(anchored_crps=0.125))
    assert failed["status"] == "V11M3_TOP_JOURNAL_SCREEN_FAIL"
    assert failed["gate_passed"] is False
    assert failed["gates"]["task_crps_gain_vs_feature41_ge_5pct"] is False


def _write_formal_source_prediction(path: Path, *, seed: int) -> None:
    n_rows = 2
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(
            "reactflow_delta.model_rescue_v11_prediction.v1"
        ),
        "keys": np.asarray(["key0", "key1"], dtype=object),
        "outer_fold": np.zeros(n_rows, dtype=np.int64),
        "seed": np.full(n_rows, seed, dtype=np.int64),
        "feature41_point": np.asarray([0.1, -0.1]),
        "v8_point": np.asarray([0.2, -0.2]),
        "anchored_point": np.full(n_rows, float(seed)),
        "unanchored_point": np.full(n_rows, float(seed + 10)),
    }
    for name, offset in (("feature41", 0.0), ("anchored", 1.0), ("unanchored", 2.0)):
        output[f"{name}_weights"] = np.tile([[0.25, 0.75]], (n_rows, 1))
        output[f"{name}_locations"] = np.full(
            (n_rows, 2), seed + offset, dtype=np.float64
        )
        output[f"{name}_scales"] = np.tile([[0.1, 0.3]], (n_rows, 1))
        output[f"{name}_expected_absolute_delta"] = np.full(n_rows, 0.2)
    output["historical_v10_weights"] = np.tile([[0.4, 0.6]], (n_rows, 1))
    output["historical_v10_locations"] = np.zeros((n_rows, 2))
    output["historical_v10_scales"] = np.tile([[0.1, 0.2]], (n_rows, 1))
    output["historical_v10_expected_absolute_delta"] = np.full(n_rows, 0.1)
    np.savez_compressed(path, **output)


def test_formal_assembler_builds_equal_seed_ten_component_mixtures(
    tmp_path: Path,
) -> None:
    rows = []
    for seed in range(5):
        path = tmp_path / f"seed{seed}.npz"
        _write_formal_source_prediction(path, seed=seed)
        rows.append({"seed": seed, "prediction_artifact": str(path)})
    result = assemble_fold(rows, fold=0, out_dir=tmp_path)
    with np.load(result["prediction_artifact"], allow_pickle=True) as handle:
        assert handle["anchored_weights"].shape == (2, 10)
        assert handle["anchored_locations"].shape == (2, 10)
        assert handle["anchored_scales"].shape == (2, 10)
        assert np.allclose(handle["anchored_weights"].sum(axis=1), 1.0)
        assert np.allclose(handle["anchored_point"], 2.0)
        assert np.allclose(handle["unanchored_point"], 12.0)
        assert np.all(handle["seed"] == -1)


def _formal_score(*, anchored_crps: float = 0.12) -> dict:
    rows = _complete_score(anchored_crps=anchored_crps)["scores"]
    return {
        "schema_version": FORMAL_SCORE_SCHEMA,
        "status": "V11M4_COMPLETE_FORMAL_SCORE_PASS",
        "mixture_scores": rows,
        "individual_seed_scores": {
            str(seed): [dict(row) for row in rows] for seed in range(5)
        },
        "equal_seed_mixture": True,
        "partial_fold_scores_inspected": False,
        "external_outcome_accessed": False,
        "model_or_threshold_selection_performed": False,
    }


def _screen_pass() -> dict:
    return {
        "schema_version": SCREEN_QUALIFICATION_SCHEMA,
        "status": "V11M3_TOP_JOURNAL_SCREEN_PASS",
        "gate_passed": True,
    }


def test_formal_qualifier_repeats_gates_and_requires_seed_stability() -> None:
    passed = qualify_formal(_formal_score(), _screen_pass())
    assert passed["status"] == "V11M4_TOP_JOURNAL_FORMAL_PASS"
    assert passed["gate_passed"] is True

    failed = qualify_formal(_formal_score(anchored_crps=0.125), _screen_pass())
    assert failed["status"] == "V11M4_TOP_JOURNAL_FORMAL_FAIL"
    assert failed["gate_passed"] is False
    assert failed["gates"]["task_crps_gain_vs_feature41_ge_5pct"] is False


def test_post_v11_diagnostic_weights_methods_then_mutants_then_positions() -> None:
    methods = np.asarray(["A", "A", "A", "A", "B", "B"], dtype=object)
    mutants = np.asarray(["A1", "A2", "A2", "A2", "B1", "B1"], dtype=object)
    weights = method_balanced_weights(methods, mutants)
    assert np.isclose(weights[methods == "A"].sum(), 0.5)
    assert np.isclose(weights[methods == "B"].sum(), 0.5)
    assert np.isclose(weights[mutants == "A1"].sum(), 0.25)
    assert np.isclose(weights[mutants == "A2"].sum(), 0.25)
    assert np.allclose(weights[mutants == "A2"], 1.0 / 12.0)


def test_post_v11_diagnostics_are_closed_while_screen_training_runs() -> None:
    with pytest.raises(RuntimeError, match="training closed"):
        assert_diagnostic_authority(ROOT)


def test_post_v11_convergence_rule_requires_one_percent_in_fourteen_folds() -> None:
    folds = []
    for fold in range(20):
        anchored = [1.2] * 30 + [1.0] * 5 + ([0.98] * 5 if fold < 14 else [1.0] * 5)
        folds.append(
            {
                "outer_fold": fold,
                "training_histories": {
                    "anchored_point": anchored,
                    "unanchored_point": [1.0] * 40,
                },
            }
        )
    diagnostic = convergence_diagnostic(folds)
    assert diagnostic["anchored_folds_ge_1pct"] == 14
    assert diagnostic["schedule_visibly_unfinished"] is True

    folds[13]["training_histories"]["anchored_point"] = [1.0] * 40
    diagnostic = convergence_diagnostic(folds)
    assert diagnostic["anchored_folds_ge_1pct"] == 13
    assert diagnostic["schedule_visibly_unfinished"] is False


def test_post_v11_direction_requires_all_twenty_independent_puzzles() -> None:
    complete = directional_summary([0.1] * 20)
    assert complete["confirmatory"] is True
    assert complete["stable_nonzero_direction"] is True
    incomplete = directional_summary([0.1] * 19)
    assert incomplete == {
        "n_puzzles": 19,
        "confirmatory": False,
        "reason": "REQUIRES_ALL_TWENTY_PUZZLES",
    }


def test_post_v11_distribution_diagnostic_uses_fixed_point_allocation() -> None:
    observations: dict[str, np.ndarray] = {
        "method": np.asarray(["A", "A", "B", "B"], dtype=object),
        "mutant": np.asarray(["A1", "A1", "B1", "B1"], dtype=object),
        "target": np.asarray([-0.2, -0.05, 0.1, 0.3]),
    }
    locations = np.asarray(
        [[-0.1, 0.1], [-0.2, 0.1], [-0.1, 0.2], [-0.3, 0.1]]
    )
    scales = np.asarray(
        [[0.1, 0.2], [0.2, 0.3], [0.15, 0.25], [0.3, 0.4]]
    )
    for name in ("feature41", "anchored", "unanchored"):
        observations[f"{name}_point"] = np.zeros(4)
        observations[f"{name}_weights"] = np.full((4, 2), 0.5)
        observations[f"{name}_locations"] = locations
        observations[f"{name}_scales"] = scales
    result = distribution_diagnostic(observations)
    assert set(result) == {"feature41", "anchored", "unanchored"}
    assert "coverage68" in result["anchored"]
    assert "lower_tail_miss90" in result["anchored"]
    assert "median_allocation_absolute_error_association" in result["anchored"]


def test_post_v11_outer_train_aggregation_does_not_pool_mutants_or_methods() -> None:
    result = puzzle_macro_from_mutant_losses(
        {"P01": {"M1": [1.0, 3.0], "M2": [5.0]}}
    )
    assert result == {"P01": 3.5}


def test_post_v11_train_held_gap_is_descriptive_and_requires_fourteen_folds() -> None:
    rows = [
        {
            "outer_fold": fold,
            "anchored_train_minus_held_gain": 0.06 if fold < 14 else 0.0,
        }
        for fold in range(20)
    ]
    result = summarize_train_held_gap(rows)
    assert result["folds_with_gap_ge_5_percentage_points"] == 14
    assert result["large_train_to_held_gap"] is True
    assert result["independent_effect_interval_computed"] is False
