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
    }
    if phase in {"RND2", "RND3", "RND4", "RND5"}:
        contract["contract_status"] = token
        authority = active["authority"]
        authority["current_phase"] = phase
        authority["current_runnable_phase"] = phase
        authority["current_authority_state"] = token
        authority["binding_status"] = token
        active["runnable_phases"] = [phase]
        active["partial_fold_score_read_allowed"] = False
        active["new_external_outcome_access_allowed"] = False
        ledger["current_phase"] = phase
        ledger["current_status"] = token
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
        (("active", "gate_state", "RND6"), "AUTHORIZED"),
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
