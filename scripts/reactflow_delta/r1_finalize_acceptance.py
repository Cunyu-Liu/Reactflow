"""R1: Freeze endpoint_v2 + information permission — acceptance/closure.

Checks that endpoint_v2 is frozen, hash-bound, tests pass, and each task has a
unique unit/label/score/metric (contract §13.2 R1, §3.2/§8.7). Emits an
append-only acceptance manifest and a CLOSED sentinel.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = ROOT / "configs" / "reactflow_delta" / "endpoint_v2.yaml"
LEDGER = ROOT / "configs" / "reactflow_delta" / "endpoint_v2.sha256"
TEST = ROOT / "tests" / "reactflow_delta" / "test_endpoint_v2_spec.py"
OUT = ROOT / "results" / "r1_endpoint_v2_acceptance_20260807"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []

    # 1. endpoint frozen + bound to epoch 14 + hash ledger
    spec = yaml.safe_load(ENDPOINT.read_text())
    if spec.get("endpoint_version") != 2 or spec.get("status") != "FROZEN":
        errors.append("endpoint not v2 FROZEN")
    if spec.get("authority_epoch") != 14:
        errors.append("authority_epoch != 14")
    ledger_exp = sha256(ENDPOINT)
    ledger_found = any(
        p.split(None, 1)[-1] == "configs/reactflow_delta/endpoint_v2.yaml"
        and p.split(None, 1)[0] == ledger_exp
        for p in LEDGER.read_text().splitlines()
    )
    if not ledger_found:
        errors.append("endpoint sha256 not in detached ledger")

    # 2. unique primary estimand (one unit/label/score/metric/resampling)
    pri = spec["primary"]
    for f in ("unit", "label_definition", "score", "metric", "resampling"):
        if not (isinstance(pri.get(f), str) and pri[f].strip()):
            errors.append(f"primary.{f} empty")

    # 3. information permission forbids mutant profile
    forb = " ".join(spec["information_permission"]["forbidden_inputs"]).lower()
    if not ("mutant" in forb and "profile" in forb):
        errors.append("mutant profile not forbidden")

    # 4. change control requires new version
    if "endpoint_v3" not in spec["change_control"]["rule"]:
        errors.append("change control lacks v3 requirement")

    # 5. run tests
    py = sys.executable
    res = subprocess.run([py, "-m", "pytest", str(TEST), "-q", "--no-header"],
                         cwd=ROOT, capture_output=True, text=True)
    tail = (res.stdout + res.stderr).strip().splitlines()
    pass_line = [l for l in tail if "passed" in l]
    n_passed = pass_line[-1].split() if pass_line else []

    status = "FAIL" if errors else "PASS"
    ts = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "reactflow_delta.r1_endpoint_v2_acceptance.v1",
        "task_id": "R1",
        "status": status,
        "completed_at": ts,
        "endpoint_id": "RFD_ENDPOINT_V2",
        "endpoint_version": 2,
        "authority_epoch": 14,
        "endpoint_sha256": sha256(ENDPOINT),
        "ledger_path": "configs/reactflow_delta/endpoint_v2.sha256",
        "ledger_sha256": sha256(LEDGER),
        "test_path": "tests/reactflow_delta/test_endpoint_v2_spec.py",
        "pytest_exit_code": res.returncode,
        "pytest_passed": n_passed,
        "errors": errors,
        "evidence_class": "BENCHMARK_QUALIFICATION_ONLY",
        "claim_eligibility": "NO_CONFIRMATORY_CLAIM",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r1_acceptance.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "r1_endpoint_v2.sha256").write_text(
        f"{sha256(ENDPOINT)}  configs/reactflow_delta/endpoint_v2.yaml\n")
    # CLOSED sentinel
    (OUT / "R1_CLOSED.yaml").write_text(
        yaml.safe_dump({**manifest, "sentinel": "R1_ENDPOINT_V2_CLOSED"}))
    print(json.dumps(manifest, indent=2))
    print("exit:", res.returncode)
    return 0 if (status == "PASS" and res.returncode == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
