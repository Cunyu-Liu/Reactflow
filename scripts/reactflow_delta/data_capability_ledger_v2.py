#!/usr/bin/env python3
"""data_capability_ledger_v2 (contract 7.2, 11.4).

Per dataset/checkpoint records on separate axes:
  historical analytic exposure, sequence exposure, WT profile exposure,
  mutant outcome exposure, publication/study/batch, parent/lineage, homology,
  probe/platform/normalization, license/release, role, evidence status.
"""

from __future__ import annotations

from typing import Any

import yaml

EXPOSURE_AXES = [
    "historical_analytic_exposure",
    "sequence_exposure",
    "wt_profile_exposure",
    "mutant_outcome_exposure",
]

ROLES = {"development", "pretraining", "direct_external", "adjacent", "oracle", "unqualified"}


def register_asset(ledger: dict[str, Any], asset: dict[str, Any]) -> dict[str, Any]:
    """Validate and append one dataset/checkpoint capability record."""
    required = {"asset_id", "role", "probe", "platform", "normalization",
                "license", "release", "exposure", "evidence_status"}
    missing = required - set(asset)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if asset["role"] not in ROLES:
        raise ValueError(f"invalid role {asset['role']}")
    if not set(EXPOSURE_AXES).issubset(set(asset["exposure"])):
        raise ValueError("exposure must cover all four axes")
    if "assets" not in ledger:
        ledger["assets"] = []
    ledger["assets"].append(asset)
    return ledger


def load_ledger(path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    problems = []
    ids = set()
    for a in ledger.get("assets", []):
        if a["asset_id"] in ids:
            problems.append(f"duplicate asset_id {a['asset_id']}")
        ids.add(a["asset_id"])
    return {"all_pass": len(problems) == 0, "n_assets": len(ledger.get("assets", [])),
            "problems": problems}
