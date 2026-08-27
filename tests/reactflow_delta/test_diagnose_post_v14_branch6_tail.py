from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

import scripts.reactflow_delta.diagnose_post_v14_branch6_tail as diagnostic_module
import scripts.reactflow_delta.route_post_v14_model_contingency as router_module
from scripts.reactflow_delta.diagnose_post_v14_branch6_tail import (
    AUTHORITY_ACTION,
    AUTHORITY_TOKEN,
    assert_diagnostic_authority,
    candidate_tail_difference,
    diagnose,
    main,
    summarize_tail_differences,
)
from scripts.reactflow_delta.route_post_v14_model_contingency import (
    SCHEMA as ROUTER_SCHEMA,
    STATUS as ROUTER_STATUS,
)


def _toy_inputs(target_value: float) -> dict:
    return {
        "target": np.full(8, target_value, dtype=np.float64),
        "mixture_weights": np.ones((8, 1), dtype=np.float64),
        "locations": np.zeros((8, 1), dtype=np.float64),
        "scales": np.ones((8, 1), dtype=np.float64),
        "methods": np.asarray(["A"] * 4 + ["B"] * 4, dtype=object),
        "mutants": np.asarray(
            ["A|m1"] * 2
            + ["A|m2"] * 2
            + ["B|m1"] * 2
            + ["B|m2"] * 2,
            dtype=object,
        ),
    }


def _write_diagnostic_authority(
    path: Path,
    *,
    merged: Path,
    score: Path,
    qualification: Path,
    router: Path,
    m2_csv: Path,
    output: Path,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "reactflow_delta.active_contract.v14",
                "project_task_id": "reactflow_delta_model_rescue_v14",
                "authority": {"current_phase": "V14M3"},
                "runnable_phases": ["V14M3"],
                "training_allowed": False,
                "candidate_model_training_allowed": False,
                "held_score_read_allowed": AUTHORITY_TOKEN,
                "partial_fold_score_read_allowed": False,
                "new_external_outcome_access_allowed": False,
                "next_allowed_action": AUTHORITY_ACTION,
                "post_v14_branch6_diagnostic_authority": {
                    "runtime_authority_token": AUTHORITY_TOKEN,
                    "complete_unscored_merge_path": str(merged.resolve()),
                    "complete_score_path": str(score.resolve()),
                    "qualification_path": str(qualification.resolve()),
                    "router_path": str(router.resolve()),
                    "m2_csv_path": str(m2_csv.resolve()),
                    "diagnostic_output_path": str(output.resolve()),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_branch6_tail_statistic_preserves_lower_minus_upper_direction() -> None:
    lower = candidate_tail_difference(**_toy_inputs(-10.0))
    upper = candidate_tail_difference(**_toy_inputs(10.0))
    assert lower == {
        "lower_tail_miss90": 1.0,
        "upper_tail_miss90": 0.0,
        "lower_minus_upper_tail_miss90": 1.0,
    }
    assert upper == {
        "lower_tail_miss90": 0.0,
        "upper_tail_miss90": 1.0,
        "lower_minus_upper_tail_miss90": -1.0,
    }


def test_branch6_tail_gate_requires_twenty_stable_puzzles() -> None:
    passed = summarize_tail_differences([0.1] * 20)
    assert passed["confirmatory"] is True
    assert passed["p2_route_eligible"] is True

    try:
        summarize_tail_differences([0.1] * 19)
    except ValueError as error:
        assert "exactly twenty finite" in str(error)
    else:
        raise AssertionError("branch-6 diagnostic treated missing evidence as P3 science")

    only_thirteen = summarize_tail_differences([1.0] * 13 + [-0.1] * 7)
    assert only_thirteen["positive_puzzles"] == 13
    assert only_thirteen["p2_route_eligible"] is False

    opposite_majority = summarize_tail_differences([1.0] * 6 + [-0.01] * 14)
    assert opposite_majority["ci95"][0] > 0.0
    assert opposite_majority["negative_puzzles"] == 14
    assert opposite_majority["p2_route_eligible"] is False


def test_branch6_tail_statistic_rejects_invalid_mixtures() -> None:
    invalid_weights = _toy_inputs(-10.0)
    invalid_weights["mixture_weights"][:] = 0.5
    try:
        candidate_tail_difference(**invalid_weights)
    except ValueError as error:
        assert "weights are invalid" in str(error)
    else:
        raise AssertionError("branch-6 diagnostic accepted invalid mixture weights")

    invalid_scales = _toy_inputs(-10.0)
    invalid_scales["scales"][0, 0] = 0.0
    try:
        candidate_tail_difference(**invalid_scales)
    except ValueError as error:
        assert "scales must be positive" in str(error)
    else:
        raise AssertionError("branch-6 diagnostic accepted a zero scale")


def test_branch6_diagnostic_compares_the_full_recomputed_router(monkeypatch) -> None:
    expected_router = {
        "schema_version": ROUTER_SCHEMA,
        "status": ROUTER_STATUS,
        "selected_router_branch_id": "6",
        "route_classification": "DISTRIBUTION_ONLY_FAILURE",
        "next_action": "RUN_FROZEN_BRANCH6_TAIL_DIAGNOSTIC",
        "branch6_residual_diagnostic": {"primary_statistic": "frozen"},
    }
    monkeypatch.setattr(
        diagnostic_module,
        "route",
        lambda *_args, **_kwargs: deepcopy(expected_router),
    )
    tampered_router = deepcopy(expected_router)
    tampered_router["next_action"] = "UNBOUND_ALTERNATE_ACTION"
    try:
        diagnose(
            merged={},
            score={},
            qualification={},
            router=tampered_router,
            m2_csv=Path("/m2.csv"),
            merged_path="/merge.json",
            score_path="/score.json",
            qualification_path="/qualification.json",
            router_path="/router.json",
        )
    except ValueError as error:
        assert "fully recomputed route" in str(error)
    else:
        raise AssertionError("branch-6 diagnostic accepted a partially matching router")


def test_branch6_authority_binds_every_resolved_cli_path(
    tmp_path: Path, monkeypatch
) -> None:
    paths = {
        "merged": tmp_path / "merge.json",
        "score": tmp_path / "score.json",
        "qualification": tmp_path / "qualification.json",
        "router": tmp_path / "router.json",
        "m2_csv": tmp_path / "m2.csv",
        "output": tmp_path / "diagnostic.json",
    }
    active_contract = tmp_path / "active-contract.yaml"
    _write_diagnostic_authority(active_contract, **paths)
    monkeypatch.setattr(
        router_module, "CANONICAL_ACTIVE_CONTRACT", active_contract.resolve()
    )
    assert_diagnostic_authority(
        active_contract.resolve(),
        merged_path=paths["merged"].resolve(),
        score_path=paths["score"].resolve(),
        qualification_path=paths["qualification"].resolve(),
        router_path=paths["router"].resolve(),
        m2_csv_path=paths["m2_csv"].resolve(),
        output_path=paths["output"].resolve(),
    )
    try:
        assert_diagnostic_authority(
            active_contract.resolve(),
            merged_path=paths["merged"].resolve(),
            score_path=paths["score"].resolve(),
            qualification_path=paths["qualification"].resolve(),
            router_path=(tmp_path / "alternate-router.json").resolve(),
            m2_csv_path=paths["m2_csv"].resolve(),
            output_path=paths["output"].resolve(),
        )
    except RuntimeError as error:
        assert "router_path differs from the CLI path" in str(error)
    else:
        raise AssertionError("branch-6 diagnostic accepted an unbound router path")


def test_branch6_refuses_authority_output_overwrite_before_input_reads(
    tmp_path: Path, monkeypatch,
) -> None:
    paths = {
        "merged": tmp_path / "missing-merge.json",
        "score": tmp_path / "missing-score.json",
        "qualification": tmp_path / "missing-qualification.json",
        "router": tmp_path / "missing-router.json",
        "m2_csv": tmp_path / "missing-m2.csv",
        "output": tmp_path / "diagnostic.json",
    }
    active_contract = tmp_path / "active-contract.yaml"
    original = b"canonical diagnostic already exists\n"
    paths["output"].write_bytes(original)
    _write_diagnostic_authority(active_contract, **paths)
    monkeypatch.setattr(
        router_module, "CANONICAL_ACTIVE_CONTRACT", active_contract.resolve()
    )
    try:
        main(
            [
                "--active-contract",
                str(active_contract),
                "--merged-json",
                str(paths["merged"]),
                "--score-json",
                str(paths["score"]),
                "--qualification-json",
                str(paths["qualification"]),
                "--router-json",
                str(paths["router"]),
                "--m2-csv",
                str(paths["m2_csv"]),
                "--out-json",
                str(paths["output"]),
            ]
        )
    except FileExistsError as error:
        assert "refuses to overwrite" in str(error)
    else:
        raise AssertionError("branch-6 diagnostic overwrote its authority output")
    assert paths["output"].read_bytes() == original


def test_branch6_rejects_alternate_active_contract_before_input_reads(
    tmp_path: Path, monkeypatch
) -> None:
    paths = {
        "merged": tmp_path / "missing-merge.json",
        "score": tmp_path / "missing-score.json",
        "qualification": tmp_path / "missing-qualification.json",
        "router": tmp_path / "missing-router.json",
        "m2_csv": tmp_path / "missing-m2.csv",
        "output": tmp_path / "diagnostic.json",
    }
    alternate = tmp_path / "alternate-active-contract.yaml"
    canonical = tmp_path / "repo/configs/reactflow_delta/active_contract.yaml"
    _write_diagnostic_authority(alternate, **paths)
    monkeypatch.setattr(
        router_module, "CANONICAL_ACTIVE_CONTRACT", canonical.resolve()
    )
    try:
        main(
            [
                "--active-contract",
                str(alternate),
                "--merged-json",
                str(paths["merged"]),
                "--score-json",
                str(paths["score"]),
                "--qualification-json",
                str(paths["qualification"]),
                "--router-json",
                str(paths["router"]),
                "--m2-csv",
                str(paths["m2_csv"]),
                "--out-json",
                str(paths["output"]),
            ]
        )
    except RuntimeError as error:
        assert "canonical active contract" in str(error)
    else:
        raise AssertionError("branch-6 diagnostic accepted an alternate authority file")
    assert not paths["output"].exists()
