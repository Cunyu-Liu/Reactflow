from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.reactflow_delta import (
    validate_post_v14_p2_quantile_residual_contract as validator,
)


ROOT = Path(__file__).resolve().parents[2]
MACHINE_PATH = (
    ROOT / "configs/reactflow_delta/post_v14_p2_quantile_residual_amendment.yaml"
)
LEDGER_PATH = (
    ROOT / "docs/prospective_v2/post_v14_p2_quantile_residual_decision_ledger.yaml"
)
ACTIVE_PATH = ROOT / "configs/reactflow_delta/active_contract.yaml"


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _declarations() -> tuple[dict, dict, dict]:
    return _read_yaml(MACHINE_PATH), _read_yaml(LEDGER_PATH), _read_yaml(ACTIVE_PATH)


def _set_path(mapping: dict, dotted_path: str, value: object) -> None:
    parts = dotted_path.split(".")
    target = mapping
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _validate_changed(
    document: str, dotted_path: str, value: object, message: str
) -> None:
    amendment, ledger, active = _declarations()
    target = {"amendment": amendment, "ledger": ledger, "active": active}[document]
    changed = copy.deepcopy(target)
    _set_path(changed, dotted_path, value)
    values = {
        "amendment": amendment,
        "ledger": ledger,
        "active": active,
    }
    values[document] = changed
    with pytest.raises(RuntimeError, match=message):
        validator.validate_static_contract(
            values["amendment"], values["ledger"], values["active"]
        )


def test_committed_inactive_contract_passes() -> None:
    assert validator.validate_contract() == {
        "status": "POST_V14_P2_QUANTILE_INACTIVE_CONTRACT_VALIDATION_PASS",
        "contract_status": "DRAFT_FROZEN_INACTIVE",
        "branch_id": "6",
        "activation_allowed": False,
        "training_allowed": False,
        "held_score_read_allowed": False,
        "external_outcome_access_allowed": False,
        "runnable_phases": [],
        "terminal_binding_status": "PENDING_TERMINAL_BINDING",
        "scientific_result": False,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("parent_route.selected_router_branch_id", "5"),
        ("parent_route.route_classification", "POINT_FAILURE"),
        ("parent_route.diagnostic_schema", "wrong.schema"),
        ("parent_route.diagnostic_status", "NOT_RUN"),
        ("parent_route.diagnostic_primary_statistic", "UPPER_TAIL_ONLY"),
        ("parent_route.diagnostic_next_action", "RUN_TRAINING"),
    ],
)
def test_validator_rejects_wrong_parent_route(path: str, value: object) -> None:
    _validate_changed("amendment", path, value, "branch-6 parent route changed")


@pytest.mark.parametrize(
    "path",
    [
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
    ],
)
def test_validator_rejects_reopened_top_level_authority(path: str) -> None:
    _validate_changed("amendment", path, True, "top-level inactive authority reopened")


@pytest.mark.parametrize(
    "field",
    [
        "activation_allowed",
        "source_projection_allowed",
        "training_allowed",
        "prediction_allowed",
        "held_score_read_allowed",
        "partial_fold_score_read_allowed",
        "scoring_allowed",
        "new_external_outcome_access_allowed",
    ],
)
def test_validator_rejects_reopened_nested_authority(field: str) -> None:
    _validate_changed(
        "amendment",
        f"inactive_authority.{field}",
        True,
        "nested inactive authority reopened",
    )


@pytest.mark.parametrize(
    ("document", "path", "value", "message"),
    [
        (
            "amendment",
            "inactive_authority.runnable_phases",
            ["P2M1"],
            "P2 runnable phase opened",
        ),
        (
            "active",
            "project_task_id",
            "reactflow_delta_post_v14_p2_quantile_residual",
            "became active or runnable",
        ),
        (
            "active",
            "authority.current_runnable_phase",
            "P2M1",
            "became active or runnable",
        ),
        (
            "active",
            "authority.machine_contract_path",
            "configs/reactflow_delta/post_v14_p2_quantile_residual_amendment.yaml",
            "became active or runnable",
        ),
    ],
)
def test_validator_rejects_active_pending_inconsistency(
    document: str, path: str, value: object, message: str
) -> None:
    _validate_changed(document, path, value, message)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("pending_terminal_binding.v14_terminal_handoff_path", "/mnt/v14.json"),
        ("pending_terminal_binding.source_manifest_path", "/mnt/sources.json"),
        ("pending_terminal_binding.screen_output_path", "/mnt/screen"),
        ("pending_terminal_binding.copied_v14_gate_values", {"crps": 0.015}),
    ],
)
def test_validator_rejects_realized_binding_in_inactive_draft(
    path: str, value: object
) -> None:
    _validate_changed(
        "amendment",
        path,
        value,
        "pending terminal/source/output binding changed",
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("frozen_input.total_width", 245, "frozen input width changed"),
        (
            "frozen_input.standardizer_scale_below",
            1.0e-5,
            "input standardization or matched-row rule changed",
        ),
        (
            "frozen_point_anchor.point_replay.atol",
            1.0e-6,
            "point replay or exact median rule changed",
        ),
        (
            "frozen_point_anchor.point_replay.initialization_grid_tolerance_applies",
            True,
            "point replay or exact median rule changed",
        ),
        (
            "candidate_predictive_distribution.taus",
            [0.05] * 13,
            "frozen tau array changed",
        ),
        (
            "candidate_predictive_distribution.weights",
            [1.0 / 13.0] * 13,
            "frozen weight array changed",
        ),
        (
            "candidate_model.hidden_width",
            256,
            "candidate architecture, monotonicity, or parameter count changed",
        ),
        (
            "candidate_model.exact_parameter_count",
            63749,
            "candidate architecture, monotonicity, or parameter count changed",
        ),
        (
            "matched_v10_replay.exact_parameter_count",
            63747,
            "matched V10 model, fairness, or parameter count changed",
        ),
    ],
)
def test_validator_rejects_changed_input_point_grid_or_count(
    path: str, value: object, message: str
) -> None:
    _validate_changed("amendment", path, value, message)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            "candidate_predictive_distribution.training_surrogate.scientific_crps",
            True,
            "weighted pinball was relabeled or changed",
        ),
        (
            "candidate_predictive_distribution.training_surrogate.allowed_in_scientific_score_or_gate_fields",
            True,
            "weighted pinball was relabeled or changed",
        ),
        (
            "candidate_predictive_distribution.scientific_scores.crps.name",
            "TWO_TIMES_WEIGHTED_PINBALL",
            "candidate scientific CRPS changed",
        ),
        (
            "matched_v10_replay.scientific_crps.name",
            "WEIGHTED_PINBALL",
            "matched V10 scientific CRPS changed",
        ),
        (
            "p2m3_screen_gates.matched_v10_replay.weighted_pinball_allowed_in_scientific_score_or_gate_fields",
            True,
            "matched V10 replay screen Gate changed",
        ),
    ],
)
def test_validator_keeps_training_surrogate_out_of_scientific_crps(
    path: str, value: object, message: str
) -> None:
    _validate_changed("amendment", path, value, message)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            "input_independent_initialization.comparator.entire_output_layer_weight",
            "EXISTING_RANDOM_WEIGHT",
            "input-independent V10 initialization changed",
        ),
        (
            "input_independent_initialization.comparator.output_biases.narrow_scale_raw",
            "INVERSE_SOFTPLUS_0_10",
            "input-independent V10 initialization changed",
        ),
        (
            "input_independent_initialization.candidate.entire_output_layer_weight",
            "NONZERO",
            "candidate initialization changed",
        ),
        (
            "input_independent_initialization.candidate.output_bias_formula",
            "INVERSE_SOFTPLUS_TARGET_GAP",
            "candidate initialization changed",
        ),
        (
            "input_independent_initialization.full_initial_predictive_distributions_identical",
            True,
            "full-distribution identity claim changed",
        ),
    ],
)
def test_validator_rejects_changed_input_independent_initialization(
    path: str, value: object, message: str
) -> None:
    _validate_changed("amendment", path, value, message)


def test_validator_rejects_target_gap_at_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    amendment, ledger, active = _declarations()
    monkeypatch.setattr(
        validator,
        "_registered_initial_target_gaps",
        lambda: [0.01] * 11 + [1.0e-4],
    )
    with pytest.raises(RuntimeError, match="adjacent gap is at or below 1e-4"):
        validator.validate_static_contract(amendment, ledger, active)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("invariant", "INITIAL_GRID_REPLAY_EXACT"),
        ("atol", 1.0e-7),
        ("rtol", 1.0e-6),
        ("applies_to_point_replay", True),
        ("applies_to_scientific_crps", True),
        ("applies_to_any_scientific_score", True),
    ],
)
def test_validator_locks_initial_grid_replay_tolerance(
    field: str, value: object
) -> None:
    _validate_changed(
        "amendment",
        f"input_independent_initialization.initial_grid_replay.{field}",
        value,
        "INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0 mapping changed",
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("training.learning_rate", 5.0e-4, "optimizer, pairing, epoch-order"),
        ("phase_universes.P2M2.epochs", 4, "smoke, screen, or formal universe"),
        ("phase_universes.P2M3.epochs", 39, "smoke, screen, or formal universe"),
        (
            "phase_universes.P2M4.seeds",
            [0, 1, 2, 3],
            "smoke, screen, or formal universe",
        ),
        (
            "p2m3_screen_gates.matched_v10_replay.crps_relative_gain_min",
            0.014,
            "matched V10 replay screen Gate changed",
        ),
        (
            "p2m3_screen_gates.matched_v10_replay.positive_puzzles_min",
            13,
            "matched V10 replay screen Gate changed",
        ),
        (
            "p2m3_screen_gates.matched_v10_replay.max_single_puzzle_effect_fraction",
            0.25,
            "matched V10 replay screen Gate changed",
        ),
        (
            "p2m4_formal.candidate_atom_count",
            13,
            "formal 65-atom mixture or Gate changed",
        ),
        (
            "p2m4_formal.positive_seeds_min_each_metric",
            3,
            "formal 65-atom mixture or Gate changed",
        ),
        (
            "qualification_and_claim.maximum_pass_claim",
            "PUBLICATION_READY",
            "claim ceiling changed",
        ),
    ],
)
def test_validator_rejects_changed_schedule_gate_or_claim(
    path: str, value: object, message: str
) -> None:
    _validate_changed("amendment", path, value, message)


@pytest.mark.parametrize(
    ("document", "path"),
    [
        ("amendment", "inactive_authority.generic_training_token_allowed"),
        ("amendment", "future_phase_tokens.generic_training_token_allowed"),
        ("ledger", "authority.generic_training_token_allowed"),
    ],
)
def test_validator_rejects_generic_training_token(
    document: str, path: str
) -> None:
    _validate_changed(document, path, True, "generic training token|phase token")


def test_in_memory_validation_performs_no_filesystem_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    amendment, ledger, active = _declarations()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("static in-memory validation attempted filesystem I/O")

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    result = validator.validate_static_contract(amendment, ledger, active)
    assert result["status"].endswith("VALIDATION_PASS")


def test_canonical_validation_reads_only_declared_yaml_and_creates_no_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [MACHINE_PATH.resolve(), LEDGER_PATH.resolve(), ACTIVE_PATH.resolve()]
    calls: list[Path] = []
    original_read_text = Path.read_text

    def guarded_read(path: Path, *args, **kwargs):
        resolved = path.resolve()
        if resolved not in expected:
            raise AssertionError(f"unexpected artifact read: {resolved}")
        calls.append(resolved)
        return original_read_text(path, *args, **kwargs)

    def forbidden_output(*_args, **_kwargs):
        raise AssertionError("contract validator attempted output creation")

    monkeypatch.setattr(Path, "read_text", guarded_read)
    monkeypatch.setattr(Path, "write_text", forbidden_output)
    monkeypatch.setattr(Path, "mkdir", forbidden_output)
    result = validator.validate_contract()
    assert result["activation_allowed"] is False
    assert calls == expected
