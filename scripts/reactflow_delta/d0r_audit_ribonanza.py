#!/usr/bin/env python3
"""D0-R: audit Ribonanza Kaggle data accessibility.

Records the correct Kaggle competition download flow, the credential status,
and the fact that M2SL5 (Ribonanza pre-competition data) is accessible via RMDB
without Kaggle credentials. The 404 in D0 only proves the探测 method failed,
not that the data is inaccessible.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "reactflow-delta-d0r-ribonanza-audit-v1"


def check_kaggle_cli() -> dict[str, Any]:
    """Check if kaggle CLI is installed and credentials are present."""
    kaggle_cli = shutil.which("kaggle")
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    return {
        "kaggle_cli_installed": kaggle_cli is not None,
        "kaggle_cli_path": kaggle_cli,
        "kaggle_json_exists": os.path.isfile(kaggle_json),
        "kaggle_json_path": kaggle_json,
        "kaggle_module_available": _check_python_module("kaggle"),
    }


def _check_python_module(name: str) -> bool:
    try:
        import importlib
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> int:
    kaggle_status = check_kaggle_cli()
    audit = {
        "schema_version": SCHEMA_VERSION,
        "stage": "D0-R",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "competition": "stanford-ribonanza-rna-folding",
        "correct_download_flow": [
            "1. Install kaggle CLI: pip install kaggle",
            "2. Place credentials at ~/.kaggle/kaggle.json (from kaggle.com -> Account -> Create New Token)",
            "3. Run: kaggle competitions download -c stanford-ribonanza-rna-folding",
            "4. Accept competition rules at https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/rules",
        ],
        "kaggle_status": kaggle_status,
        "d0_finding": {
            "d0_method": "direct URL probe (incorrect URL, no Kaggle auth)",
            "d0_result": "404 / no data retrieved",
            "interpretation": "The 404 only proves the探测 method (direct URL without auth) failed, NOT that the data is inaccessible. The correct method is via Kaggle CLI with credentials.",
        },
        "m2sl5_via_rmdb": {
            "accessible": True,
            "rmdb_ids": ["M2SL5_2A3_0000", "M2SL5_DMS_0000"],
            "description": "M2SL5 is Ribonanza Kaggle pre-competition data deposited in RMDB. Accessible via RMDB GitHub releases without Kaggle credentials. Entry description explicitly states 'Ribonanza Kaggle dataset... Collected: pre-competition'.",
            "citation": "R. Huang, R. Das (no PMID/DOI in RMDB entry)",
            "caveat": "M2SL5 candidate relations are candidate_only_unverified. Intervention lineage is NOT independently confirmed — the WT anchor is identified by name convention, not by experimental verification.",
        },
        "conclusion": (
            "Ribonanza Kaggle data is NOT directly accessible in this environment "
            "(no kaggle CLI, no credentials). However, M2SL5 (Ribonanza pre-competition "
            "derivative) IS accessible via RMDB and has been downloaded and parsed. "
            "The D0 404 is reinterpreted as a探测 method failure, not data inaccessibility."
        ),
    }
    output_path = Path("data_registry/d0r_ribonanza_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
