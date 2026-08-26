from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.reactflow_delta.validate_post_v13_route_diagnostics import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_post_v13_diagnostic_contract_passes() -> None:
    result = validate_contract(ROOT)
    assert result == {
        "status": "POST_V13_ROUTE_DIAGNOSTIC_CONTRACT_VALIDATION_PASS",
        "phase": "M6",
        "training_allowed": False,
        "held_score_read_allowed": False,
        "external_outcome_access_allowed": False,
    }


def test_contract_freezes_two_distinct_route_gates() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/post_v13_route_diagnostics.yaml").read_text()
    )
    assert set(contract["diagnostic_arms"]) == {
        "baseline",
        "noise_aware",
        "coherent_sign_magnitude",
    }
    assert contract["route_gates"]["noise_aware_supported"][
        "signed_relative_gain_min"
    ] == 0.005
    assert contract["route_gates"]["coherent_factorization_supported"][
        "point_absolute_relative_gain_min"
    ] == 0.01
    assert contract["boundaries"]["score_before_complete_merge_allowed"] is False
    assert contract["boundaries"]["same_family_v14_allowed"] is False


def test_validator_rejects_broader_score_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active_path = ROOT / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text())
    bad = copy.deepcopy(active)
    bad["held_score_read_allowed"] = True
    copied = tmp_path / "repo"
    for relative in (
        "configs/reactflow_delta",
        "docs/prospective_v2",
    ):
        (copied / relative).mkdir(parents=True, exist_ok=True)
    needed = (
        "configs/reactflow_delta/post_v13_route_diagnostics.yaml",
        "configs/reactflow_delta/model_rescue_v13_amendment.yaml",
        "docs/prospective_v2/post_v13_route_diagnostics_ledger.yaml",
        "docs/prospective_v2/model_rescue_v13_decision_ledger.yaml",
    )
    for relative in needed:
        source = ROOT / relative
        (copied / relative).write_text(source.read_text())
    (copied / "configs/reactflow_delta/active_contract.yaml").write_text(
        yaml.safe_dump(bad, sort_keys=False)
    )
    with pytest.raises(RuntimeError, match="held-score authority"):
        validate_contract(copied)
