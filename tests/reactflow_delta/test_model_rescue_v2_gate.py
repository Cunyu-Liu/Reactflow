from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_v2_terminal_contract_is_preserved_by_v8() -> None:
    v2 = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/model_rescue_v2_amendment.yaml").read_text()
    )
    v8 = yaml.safe_load(
        (ROOT / "configs/reactflow_delta/model_rescue_v8_amendment.yaml").read_text()
    )
    assert v2["contract_status"] == (
        "TERMINAL_R2M3_MEAN_GATE_FAIL_CALIBRATION_BASELINE_ONLY"
    )
    assert v2["r2m3_result"]["overall_status"] == "MODEL_RESCUE_V2_FAIL"
    assert v8["parent"]["v2_terminal_status"] == v2["contract_status"]
