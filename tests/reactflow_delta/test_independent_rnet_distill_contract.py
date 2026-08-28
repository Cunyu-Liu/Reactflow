from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.reactflow_delta import validate_independent_rnet_distill_contract as validator
from scripts.reactflow_delta.merge_independent_rnet_distill import STATUS as MERGE_STATUS
from scripts.reactflow_delta.project_independent_rnet_distill_source import inspect_shard
from scripts.reactflow_delta.score_independent_rnet_distill import SCORE_STATUS
from scripts.reactflow_delta.score_independent_rnet_distill_formal import (
    assert_formal_score_authority,
)
from scripts.reactflow_delta.qualify_independent_rnet_distill_formal import (
    assert_formal_qualifier_authority,
    load_frozen_formal_gates,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_shard(root: Path, *, bad_length: bool = False, outcome_key: bool = False) -> tuple[Path, dict]:
    shard = root / "shard_00000"
    shard.mkdir()
    np.savez(
        shard / "features.npz",
        **{"000000.single": np.zeros((3, 384), dtype=np.float32)},
    )
    provenance = {
        "model_name": "RibonanzaNet2",
        "model_version": "alpha-v1",
        "weights_sha256": "weights",
        "content_sha256": "content",
        "record_count": 1,
        "schema": {"single": {"axes": ["L", 384], "dtype": "<f4"}},
    }
    (shard / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    entry = {
        "arrays": {"single": {"shape": [3 if not bad_length else 2, 384], "dtype": "<f4"}},
        "family": "fixture",
        "length": 3,
        "record_id": "fixture-0",
        "row": 0,
        "sequence": "ACG",
    }
    if outcome_key:
        entry["target"] = [0.0]
    (shard / "index.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return shard, {
        "path": shard.name,
        "record_count": 1,
        "content_sha256": "content",
        "weights_sha256": "weights",
    }


def test_inspect_shard_accepts_exact_single_only_schema(tmp_path: Path) -> None:
    shard, root_entry = _write_shard(tmp_path)
    count, record_ids, content_binding_matches = inspect_shard(
        shard, root_entry, expected_weights="weights", expected_width=384
    )
    assert count == 1
    assert record_ids == ["fixture-0"]
    assert content_binding_matches is True


def test_inspect_shard_rejects_silent_length_resize(tmp_path: Path) -> None:
    shard, root_entry = _write_shard(tmp_path, bad_length=True)
    with pytest.raises(RuntimeError, match="teacher shape mismatch"):
        inspect_shard(shard, root_entry, expected_weights="weights", expected_width=384)


def test_inspect_shard_rejects_outcome_field(tmp_path: Path) -> None:
    shard, root_entry = _write_shard(tmp_path, outcome_key=True)
    with pytest.raises(RuntimeError, match="index schema changed"):
        inspect_shard(shard, root_entry, expected_weights="weights", expected_width=384)


def _persist_authority_fixture(fixture: dict) -> None:
    repo_root = fixture["repo_root"]
    for path, key in (
        (validator.ACTIVE_PATH, "active"),
        (validator.CONTRACT_PATH, "contract"),
        (validator.LEDGER_PATH, "ledger"),
    ):
        destination = repo_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(fixture[key], sort_keys=False), encoding="utf-8"
        )
    research_path = repo_root / validator.RESEARCH_PATH
    research_path.parent.mkdir(parents=True, exist_ok=True)
    research_path.write_text(
        "---\n"
        + yaml.safe_dump(fixture["research"], sort_keys=False)
        + "---\n\n# Authority fixture\n",
        encoding="utf-8",
    )


def _authority_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: str,
    terminal_status: str = validator.RND6_QUALIFICATION_PASS,
) -> dict:
    token = validator.TOKENS[phase]
    active = copy.deepcopy(_read_yaml(ROOT / validator.ACTIVE_PATH))
    contract = copy.deepcopy(_read_yaml(ROOT / validator.CONTRACT_PATH))
    ledger = copy.deepcopy(_read_yaml(ROOT / validator.LEDGER_PATH))
    research = {
        "slug": "orchestrator-260828-independent-rnet-distill",
        "date": "2026-08-28",
        "status": token,
        "parent": "orchestrator-260827-v14-wt-profile",
        "formal_chain_status": validator.FORMAL_INACTIVE_STATUS,
        "formal_phase_order": validator.FORMAL_PHASE_ORDER,
        "formal_activation_allowed": False,
        "formal_score_accessed": False,
        "formal_qualification_accessed": False,
    }
    if phase in {"RND2", "RND3", "RND4", "RND5", *validator.FORMAL_PHASES}:
        contract["contract_status"] = token
        authority = active["authority"]
        authority["current_phase"] = phase
        authority["current_runnable_phase"] = "NONE" if phase == "RND6T" else phase
        authority["current_authority_state"] = token
        authority["binding_status"] = token
        active["runnable_phases"] = [] if phase == "RND6T" else [phase]
        active["partial_fold_score_read_allowed"] = False
        active["new_external_outcome_access_allowed"] = False
        ledger["current_phase"] = phase
        ledger["current_status"] = token
    if phase in validator.FORMAL_PHASES:
        lifecycle_status, activation_allowed = validator._formal_lifecycle(phase)
        contract["formal_chain"]["lifecycle_status"] = lifecycle_status
        contract["formal_chain"]["activation_allowed"] = activation_allowed
        active["inactive_formal_chain"]["lifecycle_status"] = lifecycle_status
        active["inactive_formal_chain"]["activation_allowed"] = activation_allowed
        ledger["formal_chain_status"] = lifecycle_status
        research["formal_chain_status"] = lifecycle_status
        research["formal_activation_allowed"] = activation_allowed
        active["formal_output_state"] = copy.deepcopy(
            validator.FORMAL_OUTPUT_STATE_BY_PHASE[phase]
        )
        ledger["score_accessed"] = True
        formal_score_accessed = phase in {"RND6Q", "RND6T"}
        formal_qualification_accessed = phase == "RND6T"
        ledger["formal_score_accessed"] = formal_score_accessed
        ledger["formal_qualification_accessed"] = formal_qualification_accessed
        research["formal_score_accessed"] = formal_score_accessed
        research["formal_qualification_accessed"] = formal_qualification_accessed
        for name, path in validator.RND6_CANONICAL_PATHS.items():
            authority[name] = str(path)
    if phase == "RND2":
        active["authorization"].update(validator.RND2_AUTHORIZATION)
        active["training_allowed"] = True
        active["candidate_model_training_allowed"] = True
        active["held_score_read_allowed"] = False
        active["gate_state"] = copy.deepcopy(validator.RND2_GATE_STATE)
        active["next_allowed_action"] = validator.RND2_ACTION
        ledger["next_action"] = validator.RND2_ACTION
        ledger["decisions"].append(
            {
                "time": "2026-08-28T13:53:00+08:00",
                "event": validator.RND1_PASS,
                "decision": validator.RND2_DECISION,
                "exit_code": 0,
                "artifact_count": 3,
                "device_actual": "cuda:0",
                "cpu_fallback": False,
                "outcome_accessed": False,
                "training_loss_accessed": False,
                "scientific_metric_accessed": False,
                "residual_heads_identical": True,
                "pretrained_encoders_different": True,
                "authority_token": token,
            }
        )
    elif phase == "RND3":
        active["authorization"].update(validator.RND3_AUTHORIZATION)
        active["training_allowed"] = True
        active["candidate_model_training_allowed"] = True
        active["held_score_read_allowed"] = False
        active["gate_state"] = copy.deepcopy(validator.RND3_GATE_STATE)
        active["next_allowed_action"] = validator.RND3_ACTION
        ledger["next_action"] = validator.RND3_ACTION
        ledger["score_accessed"] = False
        for name, path in validator.RND3_AUTHORITY_PATHS.items():
            authority[name] = str(path)
        ledger["decisions"].append(
            {
                "time": "2026-08-28T14:10:00+08:00",
                "event": validator.RND2_MERGE_PASS,
                "decision": validator.RND3_DECISION,
                "experiment_id": "RND2_RNET_DISTILL_TWO_FOLD_GPU_ENGINEERING_SMOKE",
                "folds": [0, 1],
                "seed": 0,
                "point_epochs": 3,
                "calibration_epochs": 3,
                "controller_exit_code": 0,
                "runner_exit_codes": [0, 0],
                "cuda_only": True,
                "cpu_fallback": False,
                "held_target_accessed": False,
                "score_accessed": False,
                "partial_score_accessed": False,
                "new_external_outcome_accessed": False,
                "canonical_merge_path": str(validator.RND2_MERGED_PATH),
                "canonical_merge_status": validator.RND2_MERGE_PASS,
                "authority_token": token,
            }
        )
    elif phase == "RND4":
        active["authorization"].update(validator.RND4_AUTHORIZATION)
        active["training_allowed"] = False
        active["candidate_model_training_allowed"] = False
        active["held_score_read_allowed"] = True
        active["gate_state"] = copy.deepcopy(validator.RND4_GATE_STATE)
        active["next_allowed_action"] = validator.RND4_ACTION
        ledger["next_action"] = validator.RND4_ACTION
        ledger["score_accessed"] = False
        for name, path in validator.RND4_AUTHORITY_PATHS.items():
            authority[name] = str(path)
        ledger["decisions"].append(
            {
                "time": "2026-08-28T16:00:00+08:00",
                "event": validator.RND3_MERGE_PASS,
                "decision": validator.RND4_DECISION,
                "experiment_id": "RND3_RNET_DISTILL_COMPLETE_SEED0_PREDICTION_ONLY",
                "folds": list(range(20)),
                "seed": 0,
                "artifact_count": 20,
                "held_target_accessed": False,
                "score_accessed": False,
                "partial_score_accessed": False,
                "new_external_outcome_accessed": False,
                "canonical_merge_path": str(validator.RND3_MERGED_PATH),
                "canonical_merge_status": validator.RND3_MERGE_PASS,
                "authority_token": token,
            }
        )
    elif phase == "RND5":
        active["authorization"].update(validator.RND5_AUTHORIZATION)
        active["training_allowed"] = False
        active["candidate_model_training_allowed"] = False
        active["held_score_read_allowed"] = False
        active["gate_state"] = copy.deepcopy(validator.RND5_GATE_STATE)
        active["next_allowed_action"] = validator.RND5_ACTION
        ledger["next_action"] = validator.RND5_ACTION
        ledger["score_accessed"] = True
        for name, path in validator.RND5_AUTHORITY_PATHS.items():
            authority[name] = str(path)
        ledger["decisions"].append(
            {
                "time": "2026-08-28T16:10:00+08:00",
                "event": validator.RND4_SCORE_PASS,
                "decision": validator.RND5_DECISION,
                "canonical_score_path": str(validator.RND4_SCORE_PATH),
                "canonical_score_status": validator.RND4_SCORE_PASS,
                "exit_code": 0,
                "complete_valid_score": True,
                "actual_fold_count": 20,
                "score_accessed": True,
                "partial_score_accessed": False,
                "new_external_outcome_accessed": False,
                "model_or_threshold_selection_performed": False,
                "authority_token": token,
            }
        )
    elif phase == "RND6P":
        active["authorization"].update(validator.RND6P_AUTHORIZATION)
        active["training_allowed"] = True
        active["candidate_model_training_allowed"] = True
        active["held_score_read_allowed"] = False
        active["gate_state"] = copy.deepcopy(validator.RND6P_GATE_STATE)
        active["next_allowed_action"] = validator.RND6P_ACTION
        ledger["next_action"] = validator.RND6P_ACTION
        ledger["decisions"].append(
            {
                "time": "2026-08-28T17:00:00+08:00",
                "event": validator.RND5_SCREEN_PASS,
                "decision": validator.RND6P_DECISION,
                "canonical_qualification_path": str(validator.RND5_QUALIFICATION_PATH),
                "canonical_qualification_status": validator.RND5_SCREEN_PASS,
                "exit_code": 0,
                "gate_passed": True,
                "integrity_passed": True,
                "rnd6_authorized": True,
                "evidence_status": "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
                "score_accessed": True,
                "formal_score_accessed": False,
                "formal_qualification_accessed": False,
                "partial_score_accessed": False,
                "new_external_outcome_accessed": False,
                "model_or_threshold_selection_performed": False,
                "authority_token": token,
            }
        )
    elif phase == "RND6S":
        active["authorization"].update(validator.RND6S_AUTHORIZATION)
        active["training_allowed"] = False
        active["candidate_model_training_allowed"] = False
        active["held_score_read_allowed"] = True
        active["gate_state"] = copy.deepcopy(validator.RND6S_GATE_STATE)
        active["next_allowed_action"] = validator.RND6S_ACTION
        ledger["next_action"] = validator.RND6S_ACTION
        ledger["decisions"].append(
            {
                "time": "2026-08-28T18:00:00+08:00",
                "event": validator.RND6_ASSEMBLY_PASS,
                "decision": validator.RND6S_DECISION,
                "experiment_id": "RND6P_RNET_DISTILL_FIXED_SEEDS_0_TO_4_FORMAL_PREDICTION_ONLY",
                "folds": validator.FORMAL_FOLDS,
                "seeds": validator.FORMAL_SEEDS,
                "expected_fold_seed_pairs": validator.FORMAL_PAIR_COUNT,
                "actual_fold_seed_pairs": validator.FORMAL_PAIR_COUNT,
                "point_epochs": validator.FORMAL_POINT_EPOCHS,
                "calibration_epochs": validator.FORMAL_CALIBRATION_EPOCHS,
                "controller_exit_code": 0,
                "runner_exit_codes_all_zero": True,
                "cuda_only": True,
                "cpu_fallback": False,
                "held_target_accessed": False,
                "score_accessed": True,
                "formal_score_accessed": False,
                "formal_qualification_accessed": False,
                "partial_score_accessed": False,
                "new_external_outcome_accessed": False,
                "best_seed_selection_performed": False,
                "equal_seed_mixture": True,
                "equal_seed_weight": validator.FORMAL_EQUAL_SEED_WEIGHT,
                "canonical_merge_path": str(validator.RND6_MERGED_PATH),
                "canonical_merge_status": validator.RND6_MERGE_PASS,
                "canonical_assembly_path": str(validator.RND6_ASSEMBLY_MANIFEST_PATH),
                "canonical_assembly_status": validator.RND6_ASSEMBLY_PASS,
                "authority_token": token,
            }
        )
    elif phase == "RND6Q":
        active["authorization"].update(validator.RND6Q_AUTHORIZATION)
        active["training_allowed"] = False
        active["candidate_model_training_allowed"] = False
        active["held_score_read_allowed"] = False
        active["gate_state"] = copy.deepcopy(validator.RND6Q_GATE_STATE)
        active["next_allowed_action"] = validator.RND6Q_ACTION
        ledger["next_action"] = validator.RND6Q_ACTION
        ledger["decisions"].append(
            {
                "time": "2026-08-28T18:10:00+08:00",
                "event": validator.RND6_SCORE_PASS,
                "decision": validator.RND6Q_DECISION,
                "canonical_formal_score_path": str(validator.RND6_SCORE_PATH),
                "canonical_formal_score_status": validator.RND6_SCORE_PASS,
                "exit_code": 0,
                "complete_valid_score": True,
                "actual_fold_count": 20,
                "actual_seed_count": 5,
                "actual_fold_seed_pairs": validator.FORMAL_PAIR_COUNT,
                "equal_seed_mixture": True,
                "equal_seed_weight": validator.FORMAL_EQUAL_SEED_WEIGHT,
                "best_seed_selection_performed": False,
                "formal_score_accessed": True,
                "formal_qualification_accessed": False,
                "partial_score_accessed": False,
                "new_external_outcome_accessed": False,
                "model_or_threshold_selection_performed": False,
                "authority_token": token,
            }
        )
    elif phase == "RND6T":
        assert terminal_status in validator.RND6_QUALIFICATION_STATUSES
        active["authorization"].update(validator.RND6T_AUTHORIZATION)
        active["training_allowed"] = False
        active["candidate_model_training_allowed"] = False
        active["held_score_read_allowed"] = False
        active["gate_state"] = {
            **validator.RND6Q_GATE_STATE,
            "RND6Q": terminal_status,
            "RND6T": f"TERMINAL_{terminal_status}",
        }
        active["next_allowed_action"] = validator.RND6T_ACTION
        ledger["next_action"] = validator.RND6T_ACTION
        exit_code, gate_passed, integrity_passed, complete_valid = {
            validator.RND6_QUALIFICATION_PASS: (0, True, True, True),
            validator.RND6_QUALIFICATION_FAIL: (1, False, True, True),
            validator.RND6_QUALIFICATION_INDETERMINATE: (2, False, False, False),
        }[terminal_status]
        ledger["decisions"].append(
            {
                "time": "2026-08-28T18:20:00+08:00",
                "event": terminal_status,
                "decision": validator.RND6T_DECISION,
                "canonical_formal_qualification_path": str(
                    validator.RND6_QUALIFICATION_PATH
                ),
                "canonical_formal_qualification_status": terminal_status,
                "exit_code": exit_code,
                "gate_passed": gate_passed,
                "integrity_passed": integrity_passed,
                "complete_valid_qualification": complete_valid,
                "formal_score_accessed": True,
                "formal_qualification_accessed": True,
                "partial_score_accessed": False,
                "new_external_outcome_accessed": False,
                "model_or_threshold_selection_performed": False,
                "evidence_status": "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY",
                "authority_token": token,
            }
        )
    fixture = {
        "repo_root": tmp_path / "repo",
        "active": active,
        "contract": contract,
        "ledger": ledger,
        "research": research,
    }
    _persist_authority_fixture(fixture)

    def fake_git(_repo_root: Path, *args: str) -> str:
        if args == ("branch", "--show-current"):
            return validator.BRANCH
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(validator, "_git", fake_git)
    return fixture


def _set_nested(fixture: dict, path: tuple[object, ...], value: object) -> None:
    current = fixture
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def test_authority_validator_accepts_current_rnd1_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase="RND1")
    result = validator.validate_contract(fixture["repo_root"])
    assert result["status"] == "INDEPENDENT_RNET_DISTILL_AUTHORITY_EXACT_PASS"
    assert result["phase"] == "RND1"


def test_authority_validator_accepts_exact_rnd2_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase="RND2")
    result = validator.validate_contract(fixture["repo_root"])
    assert result["status"] == "INDEPENDENT_RNET_DISTILL_AUTHORITY_EXACT_PASS"
    assert result["phase"] == "RND2"


@pytest.mark.parametrize("phase", ("RND3", "RND4", "RND5"))
def test_authority_validator_accepts_exact_later_phase_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase=phase)
    result = validator.validate_contract(fixture["repo_root"])
    assert result["status"] == "INDEPENDENT_RNET_DISTILL_AUTHORITY_EXACT_PASS"
    assert result["phase"] == phase


@pytest.mark.parametrize("phase", ("RND6P", "RND6S", "RND6Q"))
def test_authority_validator_accepts_exact_formal_runnable_phase_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase=phase)
    result = validator.validate_contract(fixture["repo_root"])
    assert result["status"] == "INDEPENDENT_RNET_DISTILL_AUTHORITY_EXACT_PASS"
    assert result["phase"] == phase


def test_exact_rnd6s_fixture_is_accepted_by_formal_scorer_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase="RND6S")
    active = assert_formal_score_authority(
        fixture["repo_root"],
        merged_json=validator.RND6_MERGED_PATH,
        assembly_json=validator.RND6_ASSEMBLY_MANIFEST_PATH,
        m2_csv=validator.M2_PATH,
        historical_v14_score_json=validator.HISTORICAL_V14_SCORE_PATH,
        out_json=validator.RND6_SCORE_PATH,
    )
    assert active["authority"]["formal_assembly_path"] == str(
        validator.RND6_ASSEMBLY_MANIFEST_PATH
    )


def test_exact_rnd6q_fixture_is_accepted_by_formal_qualifier_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase="RND6Q")
    active = assert_formal_qualifier_authority(
        fixture["repo_root"],
        screen_qualification_json=validator.RND5_QUALIFICATION_PATH,
        score_json=validator.RND6_SCORE_PATH,
        out_json=validator.RND6_QUALIFICATION_PATH,
    )
    _, formal_gates = load_frozen_formal_gates(fixture["repo_root"])
    assert active["authority"]["formal_qualification_path"] == str(
        validator.RND6_QUALIFICATION_PATH
    )
    assert formal_gates == validator.FORMAL_GATES


@pytest.mark.parametrize("terminal_status", validator.RND6_QUALIFICATION_STATUSES)
def test_authority_validator_accepts_exact_formal_terminal_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    fixture = _authority_fixture(
        tmp_path,
        monkeypatch,
        phase="RND6T",
        terminal_status=terminal_status,
    )
    result = validator.validate_contract(fixture["repo_root"])
    assert result["status"] == "INDEPENDENT_RNET_DISTILL_AUTHORITY_EXACT_PASS"
    assert result["phase"] == "RND6T"
    assert fixture["active"]["runnable_phases"] == []


@pytest.mark.parametrize(
    "path",
    (
        ("contract", "contract_status"),
        ("active", "authority", "binding_status"),
        ("active", "authority", "current_authority_state"),
        ("ledger", "current_status"),
        ("research", "status"),
    ),
)
def test_authority_validator_rejects_phase_token_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[object, ...],
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase="RND2")
    _set_nested(fixture, path, "WRONG")
    _persist_authority_fixture(fixture)
    with pytest.raises(RuntimeError, match="diverged"):
        validator.validate_contract(fixture["repo_root"])


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("active", "authorization", "scope"), "WRONG"),
        (("active", "authorization", "implementation_allowed"), True),
        (("active", "authorization", "neural_training_allowed"), False),
        (("active", "authorization", "smoke_allowed"), False),
        (("active", "authorization", "screen_allowed"), True),
        (("active", "authorization", "score_allowed"), True),
        (("active", "authorization", "qualification_allowed"), True),
        (("active", "authorization", "formal_confirmation_allowed"), True),
        (("active", "authorization", "new_external_outcome_access_allowed"), True),
        (("active", "runnable_phases"), ["RND2", "RND3"]),
        (("active", "training_allowed"), False),
        (("active", "candidate_model_training_allowed"), False),
        (("active", "held_score_read_allowed"), True),
        (("active", "partial_fold_score_read_allowed"), True),
        (("active", "new_external_outcome_access_allowed"), True),
        (("active", "gate_state", "RND1"), "WRONG"),
        (("active", "gate_state", "RND2"), "WRONG"),
        (("active", "gate_state", "RND6P"), "AUTHORIZED"),
        (("active", "next_allowed_action"), "WRONG"),
        (("ledger", "next_action"), "WRONG"),
        (("ledger", "decisions", -1, "event"), "WRONG"),
        (("ledger", "decisions", -1, "decision"), "WRONG"),
        (("ledger", "decisions", -1, "authority_token"), "WRONG"),
        (("ledger", "decisions", -1, "exit_code"), 1),
        (("ledger", "decisions", -1, "artifact_count"), 2),
        (("ledger", "decisions", -1, "device_actual"), "cpu"),
        (("ledger", "decisions", -1, "cpu_fallback"), True),
        (("ledger", "decisions", -1, "outcome_accessed"), True),
        (("ledger", "decisions", -1, "training_loss_accessed"), True),
        (("ledger", "decisions", -1, "scientific_metric_accessed"), True),
        (("ledger", "decisions", -1, "residual_heads_identical"), False),
        (("ledger", "decisions", -1, "pretrained_encoders_different"), False),
    ),
)
def test_authority_validator_rejects_incomplete_rnd2_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[object, ...],
    value: object,
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase="RND2")
    _set_nested(fixture, path, value)
    _persist_authority_fixture(fixture)
    with pytest.raises(RuntimeError):
        validator.validate_contract(fixture["repo_root"])


@pytest.mark.parametrize(
    ("phase", "path", "value"),
    (
        ("RND3", ("active", "authorization", "score_allowed"), True),
        ("RND3", ("active", "gate_state", "RND2"), "WRONG"),
        ("RND3", ("active", "next_allowed_action"), "WRONG"),
        ("RND3", ("ledger", "next_action"), "WRONG"),
        ("RND3", ("ledger", "decisions", -1, "experiment_id"), "WRONG"),
        ("RND3", ("ledger", "decisions", -1, "controller_exit_code"), 1),
        ("RND3", ("ledger", "decisions", -1, "canonical_merge_status"), "WRONG"),
        ("RND3", ("active", "authority", "screen_prediction_dir"), "/wrong"),
        ("RND4", ("active", "authorization", "score_allowed"), False),
        ("RND4", ("active", "gate_state", "RND3"), "WRONG"),
        ("RND4", ("active", "next_allowed_action"), "WRONG"),
        ("RND4", ("ledger", "next_action"), "WRONG"),
        ("RND4", ("ledger", "decisions", -1, "artifact_count"), 19),
        ("RND4", ("ledger", "decisions", -1, "partial_score_accessed"), True),
        ("RND4", ("ledger", "decisions", -1, "canonical_merge_path"), "/wrong"),
        ("RND4", ("active", "authority", "complete_score_path"), "/wrong"),
        ("RND5", ("active", "authorization", "qualification_allowed"), False),
        ("RND5", ("active", "gate_state", "RND4"), "WRONG"),
        ("RND5", ("active", "next_allowed_action"), "WRONG"),
        ("RND5", ("ledger", "next_action"), "WRONG"),
        ("RND5", ("ledger", "score_accessed"), False),
        ("RND5", ("ledger", "decisions", -1, "complete_valid_score"), False),
        ("RND5", ("ledger", "decisions", -1, "actual_fold_count"), 19),
        ("RND5", ("ledger", "decisions", -1, "canonical_score_path"), "/wrong"),
        (
            "RND5",
            ("ledger", "decisions", -1, "model_or_threshold_selection_performed"),
            True,
        ),
    ),
)
def test_authority_validator_rejects_incomplete_later_phase_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    path: tuple[object, ...],
    value: object,
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase=phase)
    _set_nested(fixture, path, value)
    _persist_authority_fixture(fixture)
    with pytest.raises(RuntimeError):
        validator.validate_contract(fixture["repo_root"])


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("contract", "formal_chain", "activation_allowed"), True),
        (("active", "inactive_formal_chain", "activation_allowed"), True),
        (("active", "inactive_formal_chain", "lifecycle_status"), "ACTIVE_RND6P"),
        (("active", "gate_state", "RND6P"), "AUTHORIZED_FIXED_SEEDS_0_TO_4_FORMAL_PREDICTION_ONLY"),
        (("active", "formal_output_state", "complete_unscored_merge_exists"), True),
        (("ledger", "formal_chain_status"), "ACTIVE_RND6P"),
        (("ledger", "formal_score_accessed"), True),
        (("research", "formal_activation_allowed"), True),
        (("research", "formal_score_accessed"), True),
        (
            ("active", "inactive_formal_chain", "canonical_paths", "formal_prediction_dir"),
            "/wrong",
        ),
        (
            ("contract", "formal_gates", "individual_seed_positive_vs_matched_null_minimum", "signed_delta"),
            3,
        ),
        (
            ("ledger", "decisions", -1, "activation_allowed"),
            True,
        ),
    ),
)
def test_authority_validator_rejects_premature_formal_activation_from_current_rnd1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[object, ...],
    value: object,
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase="RND1")
    _set_nested(fixture, path, value)
    _persist_authority_fixture(fixture)
    with pytest.raises(RuntimeError):
        validator.validate_contract(fixture["repo_root"])


@pytest.mark.parametrize(
    ("phase", "path", "value"),
    (
        ("RND6P", ("active", "authorization", "score_allowed"), True),
        ("RND6P", ("active", "authorization", "qualification_allowed"), True),
        ("RND6P", ("active", "held_score_read_allowed"), True),
        ("RND6P", ("ledger", "formal_score_accessed"), True),
        ("RND6P", ("active", "formal_output_state", "complete_formal_score_exists"), True),
        ("RND6P", ("ledger", "decisions", -1, "canonical_qualification_status"), "WRONG"),
        ("RND6S", ("active", "authorization", "neural_training_allowed"), True),
        ("RND6S", ("active", "authorization", "qualification_allowed"), True),
        ("RND6S", ("active", "training_allowed"), True),
        ("RND6S", ("active", "formal_output_state", "equal_seed_assembly_exists"), False),
        ("RND6S", ("ledger", "decisions", -1, "actual_fold_seed_pairs"), 99),
        ("RND6S", ("ledger", "decisions", -1, "canonical_merge_path"), "/wrong"),
        ("RND6Q", ("active", "authorization", "score_allowed"), True),
        ("RND6Q", ("active", "authorization", "neural_training_allowed"), True),
        ("RND6Q", ("active", "held_score_read_allowed"), True),
        ("RND6Q", ("ledger", "formal_score_accessed"), False),
        ("RND6Q", ("active", "formal_output_state", "formal_qualification_exists"), True),
        ("RND6Q", ("ledger", "decisions", -1, "canonical_formal_score_path"), "/wrong"),
        ("RND6T", ("active", "authorization", "qualification_allowed"), True),
        ("RND6T", ("active", "runnable_phases"), ["RND6T"]),
        ("RND6T", ("active", "formal_output_state", "formal_qualification_exists"), False),
        ("RND6T", ("ledger", "formal_qualification_accessed"), False),
        ("RND6T", ("ledger", "decisions", -1, "exit_code"), 1),
        ("RND6T", ("ledger", "decisions", -1, "canonical_formal_qualification_path"), "/wrong"),
    ),
)
def test_authority_validator_rejects_formal_phase_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    path: tuple[object, ...],
    value: object,
) -> None:
    fixture = _authority_fixture(tmp_path, monkeypatch, phase=phase)
    _set_nested(fixture, path, value)
    _persist_authority_fixture(fixture)
    with pytest.raises(RuntimeError):
        validator.validate_contract(fixture["repo_root"])


@pytest.mark.parametrize(
    ("terminal_status", "bad_exit_code"),
    (
        (validator.RND6_QUALIFICATION_PASS, 1),
        (validator.RND6_QUALIFICATION_FAIL, 0),
        (validator.RND6_QUALIFICATION_INDETERMINATE, 1),
    ),
)
def test_authority_validator_rejects_terminal_qualifier_exit_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
    bad_exit_code: int,
) -> None:
    fixture = _authority_fixture(
        tmp_path,
        monkeypatch,
        phase="RND6T",
        terminal_status=terminal_status,
    )
    fixture["ledger"]["decisions"][-1]["exit_code"] = bad_exit_code
    _persist_authority_fixture(fixture)
    with pytest.raises(RuntimeError, match="terminal qualification evidence"):
        validator.validate_contract(fixture["repo_root"])


@pytest.mark.parametrize(
    ("phase", "runtime_status", "validator_status"),
    (
        ("RND3", MERGE_STATUS["RND2"], validator.RND2_MERGE_PASS),
        ("RND4", MERGE_STATUS["RND3"], validator.RND3_MERGE_PASS),
        ("RND5", SCORE_STATUS, validator.RND4_SCORE_PASS),
    ),
)
def test_later_phase_predecessor_is_canonical_runtime_status(
    phase: str, runtime_status: str, validator_status: str
) -> None:
    contract = copy.deepcopy(_read_yaml(ROOT / validator.CONTRACT_PATH))
    predecessor = contract["phase_contract"][phase]["required_predecessor"]
    assert predecessor == runtime_status == validator_status
    contract["phase_contract"][phase]["required_predecessor"] = "WRONG"
    with pytest.raises(RuntimeError, match=f"{phase} predecessor"):
        validator._check_frozen_scientific_contract(contract)


def test_formal_phase_predecessors_and_exact_gates_are_frozen() -> None:
    contract = copy.deepcopy(_read_yaml(ROOT / validator.CONTRACT_PATH))
    assert contract["phase_contract"]["RND6P"]["required_predecessor"] == validator.RND5_SCREEN_PASS
    assert contract["phase_contract"]["RND6S"]["required_predecessors"] == [
        validator.RND6_MERGE_PASS,
        validator.RND6_ASSEMBLY_PASS,
    ]
    assert contract["phase_contract"]["RND6Q"]["required_predecessor"] == validator.RND6_SCORE_PASS
    assert contract["phase_contract"]["RND6T"]["required_predecessor_one_of"] == list(
        validator.RND6_QUALIFICATION_STATUSES
    )
    assert contract["formal_gates"] == validator.FORMAL_GATES


@pytest.mark.parametrize(
    ("phase", "predecessor_key"),
    (
        ("RND6P", "required_predecessor"),
        ("RND6Q", "required_predecessor"),
    ),
)
def test_frozen_contract_rejects_formal_single_predecessor_drift(
    phase: str, predecessor_key: str
) -> None:
    contract = copy.deepcopy(_read_yaml(ROOT / validator.CONTRACT_PATH))
    contract["phase_contract"][phase][predecessor_key] = "WRONG"
    with pytest.raises(RuntimeError, match=f"{phase} predecessor"):
        validator._check_frozen_scientific_contract(contract)


@pytest.mark.parametrize("phase", ("RND6S", "RND6T"))
def test_frozen_contract_rejects_formal_multi_predecessor_drift(phase: str) -> None:
    contract = copy.deepcopy(_read_yaml(ROOT / validator.CONTRACT_PATH))
    key = "required_predecessors" if phase == "RND6S" else "required_predecessor_one_of"
    contract["phase_contract"][phase][key] = ["WRONG"]
    with pytest.raises(RuntimeError, match=f"{phase} predecessor"):
        validator._check_frozen_scientific_contract(contract)
