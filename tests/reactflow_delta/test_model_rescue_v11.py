from __future__ import annotations

from pathlib import Path

import torch
import yaml

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
from scripts.reactflow_delta.qualify_model_rescue_v11 import qualify
from scripts.reactflow_delta.run_model_rescue_v11 import (
    assert_run_authority,
    fit_point_model,
)
from scripts.reactflow_delta.score_model_rescue_v11 import SCHEMA as SCORE_SCHEMA


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


def test_v11_contract_opens_only_engineering_smoke_and_keeps_parent_terminal() -> None:
    active = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/active_contract.yaml").read_text()
    )
    contract = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/model_rescue_v11_amendment.yaml").read_text()
    )
    assert active["authority"]["current_phase"] == "V11M2"
    assert active["training_allowed"] == "V11_REAL_DATA_ENGINEERING_SMOKE_ONLY"
    assert active["candidate_model_training_allowed"] == "ENGINEERING_SMOKE_ONLY"
    assert active["held_score_read_allowed"] is False
    assert active["new_external_outcome_access_allowed"] is False
    assert contract["parent"]["v10_terminal_status"] == (
        "TERMINAL_TOP_JOURNAL_TASK_CRPS_MARGIN_FAIL"
    )
    assert contract["parent"]["v10_formal_opened"] is False
    gates = contract["v11m3_screen"]["gates"]
    assert gates["signed_delta_relative_gain_vs_feature41_min"] == 0.10
    assert gates["task_crps_relative_gain_vs_feature41_asymmetric_min"] == 0.05
    assert gates["task_crps_relative_gain_vs_terminal_v10_min"] == 0.015
    assert_run_authority(ROOT, "V11M2")


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
