"""R1: Frozen endpoint_v2 spec + information-permission invariants.

Validates configs/reactflow_delta/endpoint_v2.yaml against the contract
(ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md
§3.2/§3.5/§3.6/§8.7/§13.3).

Scope: spec-freeze only (no data, no training). Synthetic degenerate-policy
checks mirror the R1 acceptance criteria:
  - exactly one primary unit/label/score/metric per task
  - changes require a new version
  - synthetic hotspot/majority, signed/abs, pair-any-degenerate, missing-info
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT_PATH = ROOT / "configs" / "reactflow_delta" / "endpoint_v2.yaml"


@pytest.fixture(scope="module")
def spec():
    assert ENDPOINT_PATH.is_file(), f"missing {ENDPOINT_PATH}"
    return yaml.safe_load(ENDPOINT_PATH.read_text())


def test_endpoint_version_and_identity(spec):
    assert spec["endpoint_id"] == "RFD_ENDPOINT_V2"
    assert spec["endpoint_version"] == 2
    assert spec["status"] == "FROZEN"
    assert spec["authority_epoch"] == 14


def test_single_primary_estimand(spec):
    # §8.7: one versioned endpoint ID -> unique mask/caller/unit/score/metric/resampling
    primary = spec["primary"]
    for field in ("unit", "label_definition", "score", "metric", "resampling"):
        assert isinstance(primary[field], str) and primary[field].strip(), f"primary.{field} empty"
    assert isinstance(primary["metric"], str)
    assert isinstance(primary["score"], str)


def test_primary_not_max_aggregation(spec):
    # §3.2: primary score must be direct pair probability, not max(position prob)
    assert "禁止" in spec["primary"]["score"]


def test_primary_metric_is_publication_macro_auprc(spec):
    metric = spec["primary"]["metric"].lower()
    assert "auprc" in metric
    assert "publication" in metric


def test_conditional_and_secondary_distinct_metrics(spec):
    cond = spec["conditional"]["metric"].lower()
    sec = spec["secondary"]["metric"].lower()
    assert cond != spec["primary"]["metric"].lower()
    assert sec != spec["primary"]["metric"].lower()
    assert cond != sec


def test_change_control_requires_new_version(spec):
    cc = spec["change_control"]
    assert "endpoint_v3" in cc["rule"]
    assert "requires" in cc
    assert "PRIMARY_ENDPOINT_NEVER_SILENT_CHANGE" in cc["primary_endpoint_change_policy"]


def test_information_permission_forbids_mutant_profile(spec):
    ip = spec["information_permission"]
    forbidden = " ".join(ip["forbidden_inputs"]).lower()
    assert "mutant" in forbidden and "profile" in forbidden
    allowed = " ".join(ip["allowed_inputs"]).lower()
    assert "wt" in allowed
    assert "ref_allele" in allowed


def test_target_mask_not_prospective_input(spec):
    ip = spec["information_permission"]
    assert "target mask" in " ".join(ip["forbidden_inputs"]).lower()
    assert "target_mask_policy" in ip


def test_mask_eligibility_reason_codes(spec):
    mask = spec["mask"]
    codes = mask["eligibility_reason_codes"]
    assert "ELIGIBLE" in codes
    for code in ("EDITED_SITE", "ALIGNMENT_CHANGE", "PROBE_ELIGIBILITY_CHANGE",
                 "MISSING_REACTIVITY", "LENGTH_MISMATCH"):
        assert code in codes
    assert "明确 eligibility reason code" in mask["mask_policy"]


def test_caller_contract_outer_unseen(spec):
    caller = spec["caller_contract"]
    assert "train-fold" in caller["scope"]
    assert "NO_CALL" in caller["reliability"]
    assert "deterministic" in caller["determinism"]


def test_degenerate_policies_present(spec):
    dp = spec["degenerate_policies"]
    for key in ("single_hotspot_not_majority", "signed_abs_same_rank", "pair_any_all_positive",
                "missing_info_not_zero", "tied_ap_row_order_invariant", "constant_label",
                "publication_lt_3"):
        assert key in dp and dp[key], f"degenerate_policies.{key} missing/empty"


def test_unidentifiable_not_numeric_for_degenerate(spec):
    # §13.3: degenerate/constant-label/publication<3 -> UNIDENTIFIABLE, not a number
    dp = spec["degenerate_policies"]
    assert "UNIDENTIFIABLE" in " ".join(str(v) for v in dp.values())


def test_detached_hash_ledger_bound(spec):
    ledger_rel = spec["integrity"]["detached_ledger_path"]
    ledger = ROOT / ledger_rel
    assert ledger.is_file(), f"missing detached ledger {ledger_rel}"
    exp = hashlib.sha256(ENDPOINT_PATH.read_bytes()).hexdigest()
    found = False
    for line in ledger.read_text().splitlines():
        if line.split(None, 1)[-1] == "configs/reactflow_delta/endpoint_v2.yaml":
            assert line.split(None, 1)[0] == exp, "endpoint sha256 drift vs ledger"
            found = True
    assert found, "endpoint_v2.yaml not present in detached ledger"


# ---------------------------------------------------------------------------
# Synthetic degenerate-policy checks (acceptance: each encoded in frozen spec)
# ---------------------------------------------------------------------------
def test_synthetic_tied_ap_row_order_spec_policy(spec):
    # §13.3/§13.4: tied AP must be row-order invariant (actual AP impl is R4 scope)
    assert spec["degenerate_policies"]["tied_ap_row_order_invariant"]


def test_synthetic_hotspot_not_majority(spec):
    assert spec["degenerate_policies"]["single_hotspot_not_majority"]


def test_synthetic_signed_abs_same_rank(spec):
    assert spec["degenerate_policies"]["signed_abs_same_rank"]


def test_synthetic_pair_any_all_positive_degenerate(spec):
    assert spec["degenerate_policies"]["pair_any_all_positive"]


def test_synthetic_missing_info_not_zero(spec):
    assert spec["degenerate_policies"]["missing_info_not_zero"]
