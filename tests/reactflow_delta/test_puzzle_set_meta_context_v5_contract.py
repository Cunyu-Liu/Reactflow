from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.reactflow_delta.validate_puzzle_set_meta_context_v5_contract import (
    EXPECTED_DOCUMENTS,
    EXPECTED_EXECUTABLE_PATHS,
    EXPECTED_ROUTER_PATH,
    EXPECTED_V14_MACHINE_CONTRACT,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]
MACHINE_PATH = "configs/reactflow_delta/puzzle_set_meta_context_v5_amendment.yaml"
ACTIVE_PATH = "configs/reactflow_delta/active_contract.yaml"
LEDGER_PATH = EXPECTED_DOCUMENTS["decision_ledger_path"]


def _copy_validation_universe(tmp_path: Path) -> Path:
    copied = tmp_path / "repo"
    relative_paths = {
        MACHINE_PATH,
        ACTIVE_PATH,
        EXPECTED_V14_MACHINE_CONTRACT,
        EXPECTED_ROUTER_PATH,
        *EXPECTED_DOCUMENTS.values(),
        *EXPECTED_EXECUTABLE_PATHS.values(),
    }
    for relative in relative_paths:
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8"
        )
    return copied


def _rewrite_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _tamper_yaml(copied: Path, relative_path: str, edit) -> None:
    path = copied / relative_path
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    bad = copy.deepcopy(value)
    edit(bad)
    _rewrite_yaml(path, bad)


def test_frozen_inactive_puzzle_set_v5_declaration_passes() -> None:
    result = validate_contract(ROOT)
    assert result == {
        "status": "PUZZLE_SET_V5_INACTIVE_DECLARATION_VALIDATION_PASS",
        "contract_status": "DRAFT_FROZEN_INACTIVE_V14_SOLE_ACTIVE",
        "active_project_task_id": "reactflow_delta_model_rescue_v14",
        "active_phase": "V14M3",
        "activation_allowed_now": False,
        "training_allowed": False,
        "held_score_read_allowed": False,
        "external_outcome_access_allowed": False,
        "runtime_execution_authorized": False,
        "validation_scope": (
            "INACTIVE_DECLARATION_AND_RUNTIME_CONSTANT_ALIGNMENT_ONLY"
        ),
    }


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda active: active["authority"].__setitem__("current_phase", "V14M4"),
            "active execution state changed: current_phase",
        ),
        (
            lambda active: active["authority"].__setitem__(
                "artifact_root", "/mnt/cunyuliu/wrong_v14_root"
            ),
            "active execution state changed: artifact_root",
        ),
        (
            lambda active: active.__setitem__("held_score_read_allowed", True),
            "active execution state changed: held_score_read_allowed",
        ),
    ],
)
def test_validator_rejects_changed_active_v14_execution_state(
    tmp_path: Path, edit, message: str
) -> None:
    copied = _copy_validation_universe(tmp_path)
    _tamper_yaml(copied, ACTIVE_PATH, edit)
    with pytest.raises(RuntimeError, match=message):
        validate_contract(copied)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda contract: contract["frozen_parents"]["point_anchor"].__setitem__(
                "seed", 1
            ),
            "frozen parent identity changed",
        ),
        (
            lambda contract: contract["scope"].__setitem__(
                "split", "RANDOM_MUTANT_SPLIT"
            ),
            "task, data, split, or estimand changed",
        ),
        (
            lambda contract: contract["artifact_schemas_and_provenance"].__setitem__(
                "fold_schema", "reactflow_delta.puzzle_set_meta_context_fold.bad"
            ),
            "artifact provenance declaration changed",
        ),
    ],
)
def test_validator_rejects_parent_split_or_artifact_identity_change(
    tmp_path: Path, edit, message: str
) -> None:
    copied = _copy_validation_universe(tmp_path)
    _tamper_yaml(copied, MACHINE_PATH, edit)
    with pytest.raises(RuntimeError, match=message):
        validate_contract(copied)


def test_validator_rejects_formal_source_provenance_gate_change(
    tmp_path: Path,
) -> None:
    copied = _copy_validation_universe(tmp_path)
    _tamper_yaml(
        copied,
        MACHINE_PATH,
        lambda contract: contract["p1m4_formal"].__setitem__(
            "formal_assembly_reconstructed_exactly_from_same_100_run_merged_sources",
            False,
        ),
    )
    with pytest.raises(RuntimeError, match="formal universe or source Gate changed"):
        validate_contract(copied)


def test_validator_rejects_ledger_authorizing_p1_early(tmp_path: Path) -> None:
    copied = _copy_validation_universe(tmp_path)
    _tamper_yaml(
        copied,
        LEDGER_PATH,
        lambda ledger: ledger["authority"].__setitem__("p1_training_allowed", True),
    )
    with pytest.raises(RuntimeError, match="ledger authority changed"):
        validate_contract(copied)


def test_validator_rejects_ledger_phase_activation_while_v14_is_active(
    tmp_path: Path,
) -> None:
    copied = _copy_validation_universe(tmp_path)
    _tamper_yaml(
        copied,
        LEDGER_PATH,
        lambda ledger: ledger["phase_state"].__setitem__("P1M2", "AUTHORIZED"),
    )
    with pytest.raises(RuntimeError, match="ledger phase state changed"):
        validate_contract(copied)


def test_validator_rejects_missing_post_v14_router(tmp_path: Path) -> None:
    copied = _copy_validation_universe(tmp_path)
    (copied / EXPECTED_ROUTER_PATH).unlink()
    with pytest.raises(RuntimeError, match="post-V14 router is missing"):
        validate_contract(copied)


def test_validator_rejects_missing_declared_runtime(tmp_path: Path) -> None:
    copied = _copy_validation_universe(tmp_path)
    (copied / EXPECTED_EXECUTABLE_PATHS["fold_runner"]).unlink()
    with pytest.raises(RuntimeError, match="declared runtime is missing: fold_runner"):
        validate_contract(copied)


def test_validator_rejects_generic_or_changed_phase_training_token(
    tmp_path: Path,
) -> None:
    copied = _copy_validation_universe(tmp_path)
    _tamper_yaml(
        copied,
        MACHINE_PATH,
        lambda contract: contract["future_phase_training_tokens"].__setitem__(
            "generic_training_token_allowed", True
        ),
    )
    with pytest.raises(RuntimeError, match="future phase token declaration changed"):
        validate_contract(copied)


def test_validator_rejects_puzzle_set_v5_operator_change(tmp_path: Path) -> None:
    copied = _copy_validation_universe(tmp_path)
    _tamper_yaml(
        copied,
        MACHINE_PATH,
        lambda contract: contract["models"]["operator"].__setitem__(
            "id", "POSITION_ALIGNED_CROSS_CONSTRUCT_CONSENSUS_V4"
        ),
    )
    with pytest.raises(RuntimeError, match="operator changed"):
        validate_contract(copied)


def test_validator_rejects_puzzle_set_v5_gate_change(tmp_path: Path) -> None:
    copied = _copy_validation_universe(tmp_path)
    _tamper_yaml(
        copied,
        MACHINE_PATH,
        lambda contract: contract["p1m3_screen"]["gates"]["task_crps"].__setitem__(
            "relative_gain_vs_feature41_min", 0.04
        ),
    )
    with pytest.raises(RuntimeError, match="screen Gate changed"):
        validate_contract(copied)
