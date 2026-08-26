from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.reactflow_delta.validate_model_rescue_v14_contract import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_v14_contract_passes() -> None:
    assert validate_contract(ROOT) == {
        "status": "V14_CONTRACT_VALIDATION_PASS",
        "phase": "V14M1",
        "training_allowed": False,
        "held_score_read_allowed": False,
        "external_outcome_access_allowed": False,
    }


def test_v14_freezes_matched_null_and_top_journal_gates() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/model_rescue_v14_amendment.yaml").read_text()
    )
    assert contract["models"]["exact_parameter_counts"]["total_each"] == 5_117_874
    assert contract["models"]["matched_null"]["id"] == (
        "v14_from_scratch_feature41_anchor"
    )
    assert contract["pretraining"]["data"] == "OUTER_TRAIN_WT_CONSTRUCTS_ONLY"
    assert contract["v14m3_screen"]["gates"]["task_crps"][
        "relative_gain_vs_from_scratch_null_min"
    ] == 0.015
    assert contract["v14m3_screen"]["gates"]["signed_delta"][
        "relative_gain_vs_feature41_min"
    ] == 0.12


def test_validator_rejects_broader_v14_score_authority(tmp_path: Path) -> None:
    copied = tmp_path / "repo"
    relative_files = (
        "configs/reactflow_delta/active_contract.yaml",
        "configs/reactflow_delta/model_rescue_v14_amendment.yaml",
        "docs/prospective_v2/model_rescue_v14_amendment_20260827.md",
        "docs/prospective_v2/model_rescue_v14_decision_ledger.yaml",
        "docs/plans/2026-08-27-model-rescue-v14.md",
        "autoresearch/orchestrator-260827-v14-wt-profile/research.md",
    )
    for relative in relative_files:
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text())
    active_path = copied / "configs/reactflow_delta/active_contract.yaml"
    active = yaml.safe_load(active_path.read_text())
    bad = copy.deepcopy(active)
    bad["held_score_read_allowed"] = True
    active_path.write_text(yaml.safe_dump(bad, sort_keys=False))
    with pytest.raises(RuntimeError, match="held-score authority"):
        validate_contract(copied)
