from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import scripts.reactflow_delta.finalize_independent_rnet_distill_result as finalizer
from scripts.reactflow_delta.assemble_independent_rnet_distill_formal import (
    ASSEMBLY_STATUS,
    EXPECTED_ASSEMBLY_FIELDS,
    EXPECTED_FOLD_MANIFEST_FIELDS,
    SCHEMA as ASSEMBLY_SCHEMA,
)
from scripts.reactflow_delta.merge_independent_rnet_distill import (
    EXPECTED_FOLD_FIELDS,
    EXPECTED_INVARIANTS,
    EXPECTED_MERGED_FIELDS,
    MERGE_INTEGRITY,
    SCHEMA as MERGE_SCHEMA,
    STATUS as MERGE_STATUS,
)
from scripts.reactflow_delta.qualify_independent_rnet_distill import (
    SCORE_TOP_FIELDS,
)
from scripts.reactflow_delta.qualify_independent_rnet_distill_formal import (
    FORMAL_SCORE_TOP_FIELDS,
)
import scripts.reactflow_delta.validate_independent_rnet_distill_contract as authority


ROOT = Path(__file__).resolve().parents[2]
RECORDED_AT = "2026-08-28T19:00:00+08:00"
FINALIZER_COMMIT = "d" * 40
REAL_VALIDATE_CONTRACT = authority.validate_contract


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _merge(phase: str) -> dict:
    folds = range(20)
    seeds = range(5) if phase == "RND6P" else (0,)
    rows = []
    for seed in seeds:
        for fold in folds:
            row = {name: None for name in EXPECTED_FOLD_FIELDS}
            row.update(
                {
                    "schema_version": "fixture.fold.v1",
                    "experiment_id": finalizer.EXPECTED_EXPERIMENT_ID[phase],
                    "phase": phase,
                    "evidence_status": finalizer.PREDICTION_EVIDENCE_STATUS[phase],
                    "metric_eligibility": finalizer.PREDICTION_EVIDENCE_STATUS[phase],
                    "started_at_utc": f"2026-08-28T08:{fold:02d}:00Z",
                    "finished_at_utc": f"2026-08-28T09:{fold:02d}:00Z",
                    "git_commit": ("a" if phase == "RND3" else "b") * 40,
                    "command": [
                        "runner.py",
                        "--phase",
                        phase,
                        "--folds",
                        str(fold),
                        "--seed",
                        str(seed),
                    ],
                    "outer_fold": fold,
                    "held_puzzle": f"P{fold + 1:02d}",
                    "seed": seed,
                    "point_epochs": finalizer.EXPECTED_SCHEDULE[phase][0],
                    "calibration_epochs": finalizer.EXPECTED_SCHEDULE[phase][1],
                    "training_device": "cuda:0",
                    "gpu_name": "Fixture CUDA GPU",
                    "invariants": copy.deepcopy(EXPECTED_INVARIANTS),
                    "exit_code": 0,
                }
            )
            rows.append(row)
    merged = {
        "schema_version": MERGE_SCHEMA,
        "phase": phase,
        "status": MERGE_STATUS[phase],
        "folds": rows,
        "merge_integrity": copy.deepcopy(MERGE_INTEGRITY),
    }
    assert set(merged) == set(EXPECTED_MERGED_FIELDS)
    return merged


def _screen_score() -> dict:
    score = {name: "fixture" for name in SCORE_TOP_FIELDS}
    score.update(
        {
            "schema_version": finalizer.SCORE_SCHEMA,
            "phase": "RND4",
            "status": finalizer.SCORE_STATUS,
            "scores": [{"outer_fold": fold} for fold in range(20)],
            "integrity_errors": [],
            "complete_valid_score": True,
            "complete_fold_artifact_universe": True,
            "expected_fold_count": 20,
            "actual_fold_count": 20,
            "failed_rows": 0,
            "duplicate_or_unexpected_artifacts": 0,
            "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
            "target_join_after_complete_merge": True,
            "independent_units": "20_PUZZLES",
            "partial_fold_scores_inspected": False,
            "external_outcome_accessed": False,
            "model_or_threshold_selection_performed": False,
            "source_exposure_status": finalizer.EVIDENCE_STATUS,
        }
    )
    return score


def _formal_score() -> dict:
    score = {name: "fixture" for name in FORMAL_SCORE_TOP_FIELDS}
    score.update(
        {
            "schema_version": finalizer.FORMAL_SCORE_SCHEMA,
            "phase": finalizer.FORMAL_SCORE_PHASE,
            "status": finalizer.FORMAL_SCORE_STATUS,
            "mixture_scores": [{"outer_fold": fold} for fold in range(20)],
            "individual_seed_scores": {
                str(seed): [{"outer_fold": fold} for fold in range(20)]
                for seed in range(5)
            },
            "integrity_errors": [],
            "complete_valid_score": True,
            "complete_source_fold_seed_universe": True,
            "complete_assembly_fold_universe": True,
            "expected_fold_seed_count": 100,
            "actual_fold_seed_count": 100,
            "expected_fold_count": 20,
            "actual_fold_count": 20,
            "expected_seed_count": 5,
            "actual_seed_count": 5,
            "failed_rows": 0,
            "duplicate_or_unexpected_artifacts": 0,
            "target_profile_identity": "EXACT_PUZZLE_METHOD_MUTATION",
            "target_join_after_complete_merge_and_assembly": True,
            "independent_units": "20_PUZZLES_NOT_100_FOLD_SEEDS",
            "equal_seed_mixture": True,
            "equal_seed_weight": 0.2,
            "best_seed_selection_performed": False,
            "partial_fold_scores_inspected": False,
            "partial_seed_scores_inspected": False,
            "external_outcome_accessed": False,
            "model_or_threshold_selection_performed": False,
            "source_exposure_status": finalizer.EVIDENCE_STATUS,
        }
    )
    return score


def _comparisons() -> dict:
    comparisons = {}
    for metric, fields in finalizer.METRIC_FIELDS.items():
        for comparator in finalizer.COMPARATORS:
            comparisons[f"{metric}_vs_{comparator}"] = {
                "comparator_field": fields[comparator],
                "candidate_field": fields["candidate"],
                "comparator_mean": 2.0,
                "candidate_mean": 1.0,
                "mean_gain": 1.0,
                "relative_gain": 0.5,
                "ci95": [0.5, 1.5],
                "positive_puzzles": 20,
                "per_puzzle": [1.0] * 20,
                "leave_one_puzzle_out": [1.0] * 20,
                "leave_one_puzzle_out_all_positive": True,
                "max_single_puzzle_effect_fraction": 0.05,
            }
    return comparisons


def _calibration() -> dict:
    return {
        "coverage95": {
            "candidate": 0.95,
            "frozen_interval": copy.deepcopy(
                finalizer.FROZEN_SCREEN_GATES["coverage_95_interval"]
            ),
            "within_interval": True,
        }
    }


def _screen_gates(*, failed: bool = False) -> dict[str, bool]:
    gates = {name: True for name in finalizer.SCREEN_GATE_NAMES}
    if failed:
        gates["signed_delta_gain_vs_matched_null_ge_frozen_minimum"] = False
    return gates


def _screen_qualification(status: str) -> dict:
    _, gate_passed, integrity_passed, _, rnd6_authorized = finalizer.SCREEN_SEMANTICS[
        status
    ]
    indeterminate = status == finalizer.SCREEN_INDETERMINATE_STATUS
    base = {
        "schema_version": finalizer.SCREEN_QUALIFICATION_SCHEMA,
        "phase": "RND5",
        "status": status,
        "gate_passed": gate_passed,
        "integrity_passed": integrity_passed,
        "integrity_errors": ["fixture_integrity"] if indeterminate else [],
        "gates": {},
        "comparisons": {},
        "calibration": {},
        "rnd6_authorized": rnd6_authorized,
        **finalizer.CLAIM_BOUNDARY,
    }
    if indeterminate:
        return base
    return {
        **base,
        "gates": _screen_gates(failed=status == finalizer.SCREEN_FAIL_STATUS),
        "comparisons": _comparisons(),
        "calibration": _calibration(),
        "frozen_gate_values": copy.deepcopy(finalizer.FROZEN_SCREEN_GATES),
        "model_or_threshold_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "new_external_outcome_accessed": False,
    }


def _formal_qualification(status: str) -> dict:
    _, gate_passed, integrity_passed, _ = finalizer.FORMAL_SEMANTICS[status]
    indeterminate = status == finalizer.FORMAL_INDETERMINATE_STATUS
    base = {
        "schema_version": finalizer.FORMAL_QUALIFICATION_SCHEMA,
        "phase": finalizer.FORMAL_QUALIFICATION_PHASE,
        "status": status,
        "gate_passed": gate_passed,
        "integrity_passed": integrity_passed,
        "integrity_errors": ["fixture_formal_integrity"] if indeterminate else [],
        "gates": {},
        "mixture_gates": {},
        "mixture_comparisons": {},
        "mixture_calibration": {},
        "individual_seed_directions": {},
        "positive_seed_counts": {},
        **finalizer.CLAIM_BOUNDARY,
        "terminal_closure_required": True,
    }
    if indeterminate:
        return base

    failed_metric = (
        "signed_delta" if status == finalizer.FORMAL_FAIL_STATUS else None
    )
    directions = {}
    counts = {metric: 0 for metric in finalizer.METRICS}
    for seed in range(5):
        by_metric = {}
        for metric in finalizer.METRICS:
            positive = not (metric == failed_metric and seed >= 3)
            mean_gain = 1.0 if positive else -1.0
            by_metric[metric] = {
                "mean_gain_vs_matched_null": mean_gain,
                "relative_gain_vs_matched_null": 0.5 if positive else -0.5,
                "positive": positive,
            }
            counts[metric] += int(positive)
        directions[str(seed)] = by_metric
    gates = {name: True for name in finalizer.FORMAL_GATE_NAMES}
    if failed_metric is not None:
        minimum = finalizer.FROZEN_FORMAL_GATES[
            "individual_seed_positive_vs_matched_null_minimum"
        ][failed_metric]
        gates[
            f"{failed_metric}_positive_individual_seeds_ge_{minimum}"
        ] = False
    return {
        **base,
        "gates": gates,
        "mixture_gates": _screen_gates(),
        "mixture_comparisons": _comparisons(),
        "mixture_calibration": _calibration(),
        "individual_seed_directions": directions,
        "positive_seed_counts": counts,
        "frozen_screen_gate_values": copy.deepcopy(finalizer.FROZEN_SCREEN_GATES),
        "frozen_formal_gate_values": copy.deepcopy(finalizer.FROZEN_FORMAL_GATES),
        "screen_prerequisite_status": finalizer.SCREEN_PASS_STATUS,
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "model_or_threshold_selection_performed": False,
        "partial_fold_scores_inspected": False,
        "partial_seed_scores_inspected": False,
        "new_external_outcome_accessed": False,
    }


def _assembly() -> dict:
    rows = []
    for fold in range(20):
        row = {name: 2 for name in EXPECTED_FOLD_MANIFEST_FIELDS}
        row.update(
            {
                "outer_fold": fold,
                "seeds": list(range(5)),
                "prediction_artifact": f"/fixture/formal_fold{fold}.npz",
                "n_registered_prediction_rows": 2,
            }
        )
        rows.append(row)
    assembly = {
        "schema_version": ASSEMBLY_SCHEMA,
        "phase": "RND6P",
        "status": ASSEMBLY_STATUS,
        "folds": rows,
        "equal_seed_mixture": True,
        "equal_seed_weight": 0.2,
        "best_seed_selection_performed": False,
        "score_computed": False,
        "target_accessed": False,
        "external_outcome_accessed": False,
    }
    assert set(assembly) == set(EXPECTED_ASSEMBLY_FIELDS)
    return assembly


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, Path]:
    paths = {
        "screen_merge": root / "artifacts/screen/rnet_distill_complete_unscored_merge.json",
        "screen_score": root / "artifacts/screen/rnet_distill_complete_score.json",
        "screen_qualification": root / "artifacts/screen/rnet_distill_qualification.json",
        "formal_merge": root / "artifacts/formal/rnet_distill_complete_unscored_merge.json",
        "formal_assembly": root / "artifacts/formal/assembled/rnet_distill_five_seed_prediction_only_assembly.json",
        "formal_score": root / "artifacts/formal/rnet_distill_complete_formal_score.json",
        "formal_qualification": root / "artifacts/formal/rnet_distill_formal_qualification.json",
    }
    patch_values = {
        "SCREEN_MERGE_PATH": paths["screen_merge"],
        "SCREEN_SCORE_PATH": paths["screen_score"],
        "SCREEN_QUALIFICATION_PATH": paths["screen_qualification"],
        "FORMAL_MERGE_PATH": paths["formal_merge"],
        "FORMAL_ASSEMBLY_PATH": paths["formal_assembly"],
        "FORMAL_SCORE_PATH": paths["formal_score"],
        "FORMAL_QUALIFICATION_PATH": paths["formal_qualification"],
    }
    for name, value in patch_values.items():
        monkeypatch.setattr(finalizer, name, value)
    monkeypatch.setattr(authority, "RND3_MERGED_PATH", paths["screen_merge"])
    monkeypatch.setattr(authority, "RND4_SCORE_PATH", paths["screen_score"])
    monkeypatch.setattr(authority, "RND5_QUALIFICATION_PATH", paths["screen_qualification"])
    screen_authority = {
        "screen_prediction_dir": paths["screen_merge"].parent,
        "complete_unscored_merge_path": paths["screen_merge"],
        "complete_score_path": paths["screen_score"],
        "qualification_path": paths["screen_qualification"],
    }
    monkeypatch.setattr(authority, "RND5_AUTHORITY_PATHS", screen_authority)
    monkeypatch.setattr(authority, "RND5T_AUTHORITY_PATHS", screen_authority)
    formal_authority = {
        "formal_prediction_dir": paths["formal_merge"].parent,
        "formal_complete_unscored_merge_path": paths["formal_merge"],
        "formal_assembly_dir": paths["formal_assembly"].parent,
        "formal_assembly_path": paths["formal_assembly"],
        "formal_complete_score_path": paths["formal_score"],
        "formal_qualification_path": paths["formal_qualification"],
        "screen_qualification_path": paths["screen_qualification"],
    }
    monkeypatch.setattr(authority, "RND6_CANONICAL_PATHS", formal_authority)
    monkeypatch.setattr(authority, "RND6_MERGED_PATH", paths["formal_merge"])
    monkeypatch.setattr(authority, "RND6_ASSEMBLY_MANIFEST_PATH", paths["formal_assembly"])
    monkeypatch.setattr(authority, "RND6_SCORE_PATH", paths["formal_score"])
    monkeypatch.setattr(authority, "RND6_QUALIFICATION_PATH", paths["formal_qualification"])
    return paths


def _repo_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: str,
    screen_status: str = finalizer.SCREEN_PASS_STATUS,
    formal_status: str = finalizer.FORMAL_PASS_STATUS,
) -> tuple[Path, dict[str, Path]]:
    repo = tmp_path / "repo"
    paths = _configure_paths(monkeypatch, repo)
    active = copy.deepcopy(_read_yaml(ROOT / authority.ACTIVE_PATH))
    contract = copy.deepcopy(_read_yaml(ROOT / authority.CONTRACT_PATH))
    ledger = copy.deepcopy(_read_yaml(ROOT / authority.LEDGER_PATH))
    for event in ledger["decisions"]:
        if event.get("event") == "RND6_FORMAL_CHAIN_FROZEN_INACTIVE":
            event["canonical_paths"] = {
                name: str(path) for name, path in authority.RND6_CANONICAL_PATHS.items()
            }
    active["authority"]["current_phase"] = phase
    active["authority"]["current_authority_state"] = authority.TOKENS[phase]
    active["authority"]["binding_status"] = authority.TOKENS[phase]
    contract["contract_status"] = authority.TOKENS[phase]
    ledger["current_phase"] = phase
    ledger["current_status"] = authority.TOKENS[phase]
    for name, path in authority.RND5_AUTHORITY_PATHS.items():
        active["authority"][name] = str(path)
    _write_json(paths["screen_merge"], _merge("RND3"))
    _write_json(paths["screen_score"], _screen_score())
    screen_qualification = _screen_qualification(screen_status)
    _write_json(paths["screen_qualification"], screen_qualification)
    if phase == "RND6Q":
        for name, path in authority.RND6_CANONICAL_PATHS.items():
            active["authority"][name] = str(path)
        _write_json(paths["formal_merge"], _merge("RND6P"))
        _write_json(paths["formal_assembly"], _assembly())
        _write_json(paths["formal_score"], _formal_score())
        _write_json(paths["formal_qualification"], _formal_qualification(formal_status))
        screen_provenance = finalizer._validate_merge(
            _merge("RND3"), phase="RND3", folds=list(range(20)), seeds=[0]
        )
        screen_registry = finalizer._registry_entry(
            phase="RND5",
            status=finalizer.SCREEN_PASS_STATUS,
            report_path=finalizer.SCREEN_REPORT_PATH,
            merge_path=paths["screen_merge"],
            score_path=paths["screen_score"],
            qualification_path=paths["screen_qualification"],
            provenance=screen_provenance,
            qualification=screen_qualification,
            recorded_at="2026-08-28T17:00:00+08:00",
            finalizer_source_commit="c" * 40,
        )
        ledger["result_registry"] = {
            "screen": screen_registry,
            "formal": copy.deepcopy(authority._PENDING_FORMAL_REGISTRY),
        }
        screen_report = finalizer._render_report(
            kind="screen",
            qualification=screen_qualification,
            registry=screen_registry,
            canonical_paths=[
                ("Complete target-free merge", paths["screen_merge"]),
                ("Complete score", paths["screen_score"]),
                ("Qualification", paths["screen_qualification"]),
            ],
        )
        screen_report_path = repo / authority.SCREEN_REPORT_PATH
        screen_report_path.parent.mkdir(parents=True, exist_ok=True)
        screen_report_path.write_text(screen_report, encoding="utf-8")
    for path, payload in (
        (authority.ACTIVE_PATH, active),
        (authority.CONTRACT_PATH, contract),
        (authority.LEDGER_PATH, ledger),
    ):
        destination = repo / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    research_source = ROOT / authority.RESEARCH_PATH
    research_destination = repo / authority.RESEARCH_PATH
    research_destination.parent.mkdir(parents=True, exist_ok=True)
    research_destination.write_text(
        research_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr(
        authority,
        "validate_contract",
        lambda observed_repo: {
            "status": "INDEPENDENT_RNET_DISTILL_AUTHORITY_EXACT_PASS",
            "phase": phase,
        },
    )
    monkeypatch.setattr(finalizer, "_git", lambda *_args: FINALIZER_COMMIT)
    return repo, paths


@pytest.mark.parametrize(
    ("status", "expected_phase", "formal_registry_status"),
    (
        (finalizer.SCREEN_PASS_STATUS, "RND6P", authority.FORMAL_PENDING_STATUS),
        (finalizer.SCREEN_FAIL_STATUS, "RND5T", authority.FORMAL_NOT_RUN_STATUS),
        (
            finalizer.SCREEN_INDETERMINATE_STATUS,
            "RND5T",
            authority.FORMAL_NOT_RUN_STATUS,
        ),
    ),
)
def test_screen_finalization_routes_three_states_without_recomputing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_phase: str,
    formal_registry_status: str,
) -> None:
    repo, _ = _repo_fixture(
        tmp_path, monkeypatch, phase="RND5", screen_status=status
    )
    result = finalizer.finalize(repo, recorded_at=RECORDED_AT)
    active = _read_yaml(repo / authority.ACTIVE_PATH)
    ledger = _read_yaml(repo / authority.LEDGER_PATH)
    report = (repo / authority.SCREEN_REPORT_PATH).read_text(encoding="utf-8")
    assert result["next_phase"] == expected_phase
    assert result["commit_or_push_performed"] is False
    assert active["authority"]["current_phase"] == expected_phase
    assert active["runnable_phases"] == (["RND6P"] if expected_phase == "RND6P" else [])
    assert ledger["result_registry"]["screen"]["status"] == status
    assert ledger["result_registry"]["screen"]["experiment_id"] == (
        finalizer.EXPECTED_EXPERIMENT_ID["RND3"]
    )
    assert ledger["result_registry"]["screen"]["authority_branch"] == authority.BRANCH
    assert ledger["result_registry"]["formal"]["status"] == formal_registry_status
    assert ledger["result_registry"]["screen"]["publication_ready"] is False
    assert "EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY" in report
    assert "Clean out-of-distribution evidence is not established" in report
    assert "Publication readiness is false" in report
    assert "## Canonical calibration" in report
    if status != finalizer.SCREEN_INDETERMINATE_STATUS:
        assert '"coverage95"' in report
    assert "folds[*].command" in report
    monkeypatch.setattr(
        authority,
        "_git",
        lambda _root, *args: authority.BRANCH
        if args == ("branch", "--show-current")
        else "",
    )
    assert REAL_VALIDATE_CONTRACT(repo)["phase"] == expected_phase


@pytest.mark.parametrize("status", finalizer.FORMAL_STATUSES)
def test_formal_finalization_routes_three_states_to_rnd6t(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    repo, _ = _repo_fixture(
        tmp_path,
        monkeypatch,
        phase="RND6Q",
        screen_status=finalizer.SCREEN_PASS_STATUS,
        formal_status=status,
    )
    result = finalizer.finalize(repo, recorded_at=RECORDED_AT)
    active = _read_yaml(repo / authority.ACTIVE_PATH)
    ledger = _read_yaml(repo / authority.LEDGER_PATH)
    report = (repo / authority.FORMAL_REPORT_PATH).read_text(encoding="utf-8")
    assert result["next_phase"] == "RND6T"
    assert active["runnable_phases"] == []
    assert not any(active["authorization"][name] for name in (
        "neural_training_allowed",
        "score_allowed",
        "qualification_allowed",
        "formal_confirmation_allowed",
    ))
    assert ledger["result_registry"]["formal"]["status"] == status
    assert ledger["result_registry"]["formal"]["experiment_id"] == (
        finalizer.EXPECTED_EXPERIMENT_ID["RND6P"]
    )
    assert ledger["result_registry"]["formal"]["authority_branch"] == authority.BRANCH
    assert ledger["result_registry"]["formal"]["folds"] == list(range(20))
    assert ledger["result_registry"]["formal"]["seeds"] == list(range(5))
    assert "Publication readiness is false" in report
    assert "Independent external replication is not established" in report
    assert "## Canonical calibration" in report
    if status != finalizer.FORMAL_INDETERMINATE_STATUS:
        assert '"coverage95"' in report
    assert "folds[*].command" in report
    monkeypatch.setattr(
        authority,
        "_git",
        lambda _root, *args: authority.BRANCH
        if args == ("branch", "--show-current")
        else "",
    )
    assert REAL_VALIDATE_CONTRACT(repo)["phase"] == "RND6T"


def test_finalizer_rejects_wrong_canonical_path_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _repo_fixture(tmp_path, monkeypatch, phase="RND5")
    active_path = repo / authority.ACTIVE_PATH
    active = _read_yaml(active_path)
    active["authority"]["qualification_path"] = "/wrong/qualification.json"
    active_path.write_text(yaml.safe_dump(active, sort_keys=False), encoding="utf-8")
    before = active_path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical path"):
        finalizer.finalize(repo, recorded_at=RECORDED_AT)
    assert active_path.read_text(encoding="utf-8") == before
    assert not (repo / authority.SCREEN_REPORT_PATH).exists()


def test_finalizer_rejects_mixed_source_commit_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, paths = _repo_fixture(tmp_path, monkeypatch, phase="RND5")
    merged = _load_json(paths["screen_merge"])
    merged["folds"][1]["git_commit"] = "e" * 40
    _write_json(paths["screen_merge"], merged)
    with pytest.raises(RuntimeError, match="source commit is not exact and single"):
        finalizer.finalize(repo, recorded_at=RECORDED_AT)
    assert not (repo / authority.SCREEN_REPORT_PATH).exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("training_device", "cuda:1", "training device is not exact cuda:0"),
        ("command", [], "exact per-fold runner command is missing"),
        ("command", "runner --fold 0", "exact per-fold runner command is missing"),
    ),
)
def test_finalizer_rejects_noncanonical_gpu_or_missing_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    repo, paths = _repo_fixture(tmp_path, monkeypatch, phase="RND5")
    merged = _load_json(paths["screen_merge"])
    merged["folds"][0][field] = value
    _write_json(paths["screen_merge"], merged)
    with pytest.raises(RuntimeError, match=message):
        finalizer.finalize(repo, recorded_at=RECORDED_AT)
    assert not (repo / authority.SCREEN_REPORT_PATH).exists()


def test_finalizer_rejects_arbitrary_pass_gate_universe() -> None:
    qualification = _screen_qualification(finalizer.SCREEN_PASS_STATUS)
    qualification["gates"] = {"arbitrary_gate": True}
    with pytest.raises(RuntimeError, match="Gate.*name universe changed"):
        finalizer._validate_qualification(qualification, formal=False)


def test_finalizer_rejects_changed_frozen_gate_values() -> None:
    qualification = _screen_qualification(finalizer.SCREEN_PASS_STATUS)
    qualification["frozen_gate_values"]["matched_null_positive_puzzles_minimum"] = 13
    with pytest.raises(RuntimeError, match="frozen Gate values changed"):
        finalizer._validate_qualification(qualification, formal=False)


def test_finalizer_rejects_formal_seed_metric_universe_drift() -> None:
    qualification = _formal_qualification(finalizer.FORMAL_PASS_STATUS)
    qualification["individual_seed_directions"]["0"].pop("task_crps")
    with pytest.raises(RuntimeError, match="seed 0 metric universe changed"):
        finalizer._validate_qualification(qualification, formal=True)


def test_finalizer_rejects_arbitrary_formal_pass_gate_universe() -> None:
    qualification = _formal_qualification(finalizer.FORMAL_PASS_STATUS)
    qualification["gates"] = {"arbitrary_gate": True}
    with pytest.raises(RuntimeError, match="Gate.*name universe changed"):
        finalizer._validate_qualification(qualification, formal=True)


def test_finalizer_rejects_changed_frozen_formal_gate_values() -> None:
    qualification = _formal_qualification(finalizer.FORMAL_PASS_STATUS)
    qualification["frozen_formal_gate_values"][
        "individual_seed_positive_vs_matched_null_minimum"
    ]["signed_delta"] = 3
    with pytest.raises(RuntimeError, match="frozen Gate values changed"):
        finalizer._validate_qualification(qualification, formal=True)


def test_finalizer_rejects_formal_complete_result_with_failed_mixture_integrity_gate() -> None:
    qualification = _formal_qualification(finalizer.FORMAL_FAIL_STATUS)
    qualification["mixture_gates"]["prediction_and_score_integrity"] = False
    qualification["gates"]["mixture_repeats_every_frozen_screen_gate"] = False
    with pytest.raises(
        RuntimeError, match="mixture complete-result engineering Gates changed"
    ):
        finalizer._validate_qualification(qualification, formal=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("status", "WRONG_STATUS", "status is not canonical"),
        ("publication_ready", True, "claim boundary changed"),
    ),
)
def test_finalizer_rejects_wrong_status_or_widened_claim_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    repo, paths = _repo_fixture(tmp_path, monkeypatch, phase="RND5")
    qualification = _load_json(paths["screen_qualification"])
    qualification[field] = value
    _write_json(paths["screen_qualification"], qualification)
    active_path = repo / authority.ACTIVE_PATH
    before = active_path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        finalizer.finalize(repo, recorded_at=RECORDED_AT)
    assert active_path.read_text(encoding="utf-8") == before
    assert not (repo / authority.SCREEN_REPORT_PATH).exists()


def test_finalizer_refuses_different_existing_report_without_state_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _repo_fixture(tmp_path, monkeypatch, phase="RND5")
    report_path = repo / authority.SCREEN_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("different canonical report\n", encoding="utf-8")
    active_path = repo / authority.ACTIVE_PATH
    before = active_path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="different content"):
        finalizer.finalize(repo, recorded_at=RECORDED_AT)
    assert active_path.read_text(encoding="utf-8") == before
    assert report_path.read_text(encoding="utf-8") == "different canonical report\n"


def test_finalizer_accepts_identical_preexisting_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, paths = _repo_fixture(tmp_path, monkeypatch, phase="RND5")
    qualification = _load_json(paths["screen_qualification"])
    provenance = finalizer._validate_merge(
        _load_json(paths["screen_merge"]),
        phase="RND3",
        folds=list(range(20)),
        seeds=[0],
    )
    entry = finalizer._registry_entry(
        phase="RND5",
        status=qualification["status"],
        report_path=finalizer.SCREEN_REPORT_PATH,
        merge_path=paths["screen_merge"],
        score_path=paths["screen_score"],
        qualification_path=paths["screen_qualification"],
        provenance=provenance,
        qualification=qualification,
        recorded_at=RECORDED_AT,
        finalizer_source_commit=FINALIZER_COMMIT,
    )
    expected = finalizer._render_report(
        kind="screen",
        qualification=qualification,
        registry=entry,
        canonical_paths=[
            ("Complete target-free merge", paths["screen_merge"]),
            ("Complete score", paths["screen_score"]),
            ("Qualification", paths["screen_qualification"]),
        ],
    )
    report_path = repo / authority.SCREEN_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(expected, encoding="utf-8")
    result = finalizer.finalize(repo, recorded_at=RECORDED_AT)
    assert result["next_phase"] == "RND6P"
    assert report_path.read_text(encoding="utf-8") == expected


def test_finalizer_full_rerun_is_exact_noop_after_authority_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _repo_fixture(tmp_path, monkeypatch, phase="RND5")
    first = finalizer.finalize(repo, recorded_at=RECORDED_AT)
    assert first["next_phase"] == "RND6P"
    tracked = [
        repo / authority.ACTIVE_PATH,
        repo / authority.CONTRACT_PATH,
        repo / authority.LEDGER_PATH,
        repo / authority.RESEARCH_PATH,
        repo / authority.SCREEN_REPORT_PATH,
    ]
    before = {path: path.read_text(encoding="utf-8") for path in tracked}
    monkeypatch.setattr(
        authority,
        "validate_contract",
        lambda observed_repo: {
            "status": "INDEPENDENT_RNET_DISTILL_AUTHORITY_EXACT_PASS",
            "phase": _read_yaml(observed_repo / authority.ACTIVE_PATH)["authority"][
                "current_phase"
            ],
        },
    )
    second = finalizer.finalize(
        repo, recorded_at="2026-08-28T20:00:00+08:00"
    )
    assert second["status"].endswith("ALREADY_EXACT_NO_CHANGES")
    assert second["next_phase"] == "RND6P"
    assert {path: path.read_text(encoding="utf-8") for path in tracked} == before


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
