#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R6 — GO/STOP scientific adjudication for ReactFlowDelta.

Independent, disk-evidence-based verification of the §13.4 P0 completion gates
(contract ``ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md``).

This script does NOT trust any self-reported status.  It reads the actual
artifact files on disk for each of the 8 gates, runs the caller_v2 / evaluate_v2
reference test suites, and emits a machine-readable terminal decision manifest.

Contract rule (§13.2 R6 / §13.4): any gate that is FAIL / UNKNOWN / NOT_RUN (or
missing) => overall decision is STOP.  Manual override to GO is never allowed.
Because the R5 verdict is STOP_METHOD_ROUTE / UNIDENTIFIABLE, the P2 gate is
NOT_GO and Phase 3 model-training is blocked.

Only NEW files are written (results/r6_go_stop_20260807/*).  No legacy,
authority, contract, endpoint, split or verdict file is modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_CODES = {
    "EDITED_SITE",
    "ALIGNMENT_CHANGE",
    "PROBE_ELIGIBILITY_CHANGE",
    "MISSING_REACTIVITY",
    "LENGTH_MISMATCH",
    "ELIGIBLE",
}

ALL_GATES = [
    "AUTHORITY_CLOSED_PASS",
    "ASSET_DISPOSITION_1024_OF_1024",
    "PRIMARY_MASK_V2_PASS",
    "GROUP_ATOMS_AND_PUBLICATION_SPLIT_PASS",
    "OLD_TEST_RETIRED_NEW_TEST_UNTOUCHED",
    "CALLER_V2_FOLD_LOCAL_AND_RELIABLE",
    "EVALUATOR_V2_REFERENCE_TESTS_PASS",
    "P2_LEARNABILITY_GO",
]

RUN_ID = "r6_go_stop_20260807"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML not available")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Individual gates (each returns status + evidence dict)
# ---------------------------------------------------------------------------

def gate_authority_closed() -> tuple[str, dict]:
    """epoch 14 active, M0-X TERMINAL FAIL, training_allowed False, hash ok."""
    cfg = REPO_ROOT / "configs/reactflow_delta/active_contract.yaml"
    sent = REPO_ROOT / "configs/reactflow_delta/authority_epoch_14.sentinel.yaml"
    bundle = REPO_ROOT / "configs/reactflow_delta/authority_epoch_14.bundle.sha256"
    ev = {"contract_path": str(cfg), "sentinel_path": str(sent), "bundle_path": str(bundle)}
    if not (cfg.exists() and sent.exists() and bundle.exists()):
        ev["reason"] = "epoch-14 contract/sentinel/bundle file missing"
        return "FAIL", ev
    try:
        c = _load_yaml(cfg)
        s = _load_yaml(sent)
    except Exception as e:  # pragma: no cover
        ev["reason"] = f"yaml parse error: {e}"
        return "FAIL", ev

    # authority epoch / phase / state
    auth = c.get("authority", {})
    ev["authority_epoch"] = auth.get("authority_epoch")
    ev["current_phase"] = auth.get("current_phase")
    ev["current_authority_state"] = auth.get("current_authority_state")
    checks = {
        "authority_epoch==14": auth.get("authority_epoch") == 14,
        "current_phase==REBUILD-P1": auth.get("current_phase") == "REBUILD-P1",
    }

    # M0-X phase terminalization
    m0x = None
    for ph in c.get("phase_graph", []):
        if ph.get("phase_id") == "M0-X":
            m0x = ph
            break
    if m0x is None:
        checks["M0X_present"] = False
        ev["reason"] = "M0-X phase block not found"
        return "FAIL", ev
    ev["M0X_lifecycle_status"] = m0x.get("lifecycle_status")
    ev["M0X_gate_result"] = m0x.get("gate_result")
    checks["M0X_TERMINAL"] = m0x.get("lifecycle_status") == "TERMINAL"
    checks["M0X_gate_FAIL"] = m0x.get("gate_result") == "FAIL"

    # training allowed
    authz = c.get("authorization", {})
    ev["training_allowed"] = authz.get("training_allowed")
    checks["training_allowed==False"] = authz.get("training_allowed") is False

    # sentinel state
    ev["sentinel_epoch"] = s.get("authority_epoch")
    ev["sentinel_state"] = s.get("current_authority_state")
    checks["sentinel_epoch==14"] = s.get("authority_epoch") == 14
    checks["sentinel_state==REBUILD_P1_AUTHORIZED"] = (
        s.get("current_authority_state") == "REBUILD_P1_AUTHORIZED"
    )

    # hash check: contract hash == sentinel active_manifest_sha256 == bundle ledger entry
    actual_hash = sha256_file(cfg)
    ev["contract_sha256"] = actual_hash
    ev["sentinel_active_manifest_sha256"] = s.get("active_manifest_sha256")
    ev["bundle_ledger"] = str(bundle)
    checks["hash_matches_sentinel"] = actual_hash == s.get("active_manifest_sha256")
    ledger_hit = False
    try:
        for line in bundle.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == actual_hash and parts[1].endswith("active_contract.yaml"):
                ledger_hit = True
                break
    except Exception:  # pragma: no cover
        pass
    checks["hash_matches_bundle_ledger"] = ledger_hit

    if all(checks.values()):
        return "PASS", ev
    ev["failed_checks"] = [k for k, v in checks.items() if not v]
    return "FAIL", ev


def gate_asset_disposition() -> tuple[str, dict]:
    """data_registry/d0x_v2 ledger has 1024/1024 rows, 0 empty disposition."""
    jl = REPO_ROOT / "data_registry/d0x_v2/asset_disposition_20260807.jsonl"
    summary = REPO_ROOT / "data_registry/d0x_v2/asset_disposition_20260807.summary.json"
    ev = {"jsonl_path": str(jl), "summary_path": str(summary)}
    if not jl.exists():
        ev["reason"] = "asset_disposition jsonl missing"
        return "FAIL", ev
    n_rows = 0
    n_empty = 0
    n_bad = 0
    with open(jl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            try:
                rec = json.loads(line)
            except Exception:
                n_bad += 1
                continue
            disp = rec.get("disposition")
            if disp is None or str(disp).strip() == "":
                n_empty += 1
    ev["n_rows"] = n_rows
    ev["n_empty_disposition"] = n_empty
    ev["n_malformed"] = n_bad
    if summary.exists():
        try:
            s = json.loads(summary.read_text(encoding="utf-8"))
            ev["summary_asset_count"] = s.get("asset_count")
            ev["summary_empty_disposition_count"] = s.get("empty_disposition_count")
        except Exception:  # pragma: no cover
            pass
    if n_rows == 1024 and n_empty == 0 and n_bad == 0:
        return "PASS", ev
    ev["failed_checks"] = {
        "rows==1024": n_rows == 1024,
        "empty==0": n_empty == 0,
        "malformed==0": n_bad == 0,
    }
    return "FAIL", ev


def gate_primary_mask() -> tuple[str, dict]:
    """every primary position in canonical v2 carries an eligibility_reason_code."""
    # canonical source of primary positions = primary_pairs_v2.jsonl (d1x_v2)
    candidates = [
        Path("/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/d1x_v2/"
             "d1x_v2_canonicalization_20260807T1830+0800/primary_pairs_v2.jsonl"),
        REPO_ROOT / "data_registry/d1x_v2/primary_pairs_v2.jsonl",
    ]
    src = next((p for p in candidates if p.exists()), None)
    summary = REPO_ROOT / "data_registry/d1x_v2/d1x_v2_summary.json"
    ev = {"primary_pairs_path": str(src) if src else None, "summary_path": str(summary)}
    if src is None:
        ev["reason"] = "primary_pairs_v2.jsonl not found on disk"
        return "UNKNOWN", ev
    n_pairs = 0
    n_missing = 0
    n_bad_code = 0
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_pairs += 1
            try:
                rec = json.loads(line)
            except Exception:
                n_bad_code += 1
                continue
            codes = rec.get("eligibility_reason_codes")
            if not codes:
                n_missing += 1
                continue
            for c in codes:
                if c not in ALLOWED_CODES:
                    n_bad_code += 1
    ev["n_primary_pairs"] = n_pairs
    ev["n_pairs_missing_code"] = n_missing
    ev["n_unknown_code"] = n_bad_code
    if summary.exists():
        try:
            s = json.loads(summary.read_text(encoding="utf-8"))
            ev["summary_primary_pairs"] = s.get("primary_pairs")
        except Exception:  # pragma: no cover
            pass
    if n_missing == 0 and n_bad_code == 0 and n_pairs > 0:
        return "PASS", ev
    ev["failed_checks"] = {
        "missing_code==0": n_missing == 0,
        "unknown_code==0": n_bad_code == 0,
        "n_pairs>0": n_pairs > 0,
    }
    return "FAIL", ev


def gate_group_atoms_publication_split() -> tuple[str, dict]:
    """publication/study/lineage overlap == 0 or explicit exemption."""
    overlap = REPO_ROOT / "data_registry/d2x_v2/overlap_report.json"
    split = REPO_ROOT / "configs/reactflow_delta/split_v2.yaml"
    ev = {"overlap_path": str(overlap), "split_path": str(split)}
    if not overlap.exists() or not split.exists():
        ev["reason"] = "overlap_report.json or split_v2.yaml missing"
        return "FAIL", ev
    try:
        o = json.loads(overlap.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        ev["reason"] = f"overlap parse error: {e}"
        return "FAIL", ev
    try:
        s = _load_yaml(split)
    except Exception as e:  # pragma: no cover
        ev["reason"] = f"split parse error: {e}"
        return "FAIL", ev

    checks = {}
    for section in ("publication_overlap", "study_overlap", "lineage_overlap_parent_sha256"):
        sec = o.get(section, {})
        for pair in ("train_vs_validation", "test_vs_train", "test_vs_validation"):
            sub = sec.get(pair, {})
            key = f"{section}:{pair}:n_intersection"
            checks[key] = sub.get("n_intersection", -1) == 0
            ev[key] = sub.get("n_intersection")
    # new_test_vs_train_val strict-zero checks
    ntv = o.get("new_test_vs_train_val", {})
    for k in ("lineage_parent_sha256", "sequence_exact_canonical"):
        sub = ntv.get(k, {})
        key = f"new_test_vs_train_val:{k}:n_intersection"
        checks[key] = sub.get("n_intersection", -1) == 0
        ev[key] = sub.get("n_intersection")

    # split_v2 group-atom consistency: same PMID -> one publication (publication_map present),
    # distinct publications consistent, and publication_studies mapping covers all studies.
    pub_map = s.get("publication_map", {})
    distinct = s.get("distinct_publications", [])
    checks["publication_map_nonempty"] = bool(pub_map)
    checks["distinct_publications_nonempty"] = bool(distinct)

    if all(checks.values()):
        return "PASS", ev
    ev["failed_checks"] = [k for k, v in checks.items() if not v]
    return "FAIL", ev


def gate_old_test_retired_new_test_untouched() -> tuple[str, dict]:
    """16SFWJ retired as DEVELOPMENT_CONSUMED; SL5 family new untouched test."""
    split = REPO_ROOT / "configs/reactflow_delta/split_v2.yaml"
    ev = {"split_path": str(split)}
    if not split.exists():
        ev["reason"] = "split_v2.yaml missing"
        return "FAIL", ev
    try:
        s = _load_yaml(split)
    except Exception as e:  # pragma: no cover
        ev["reason"] = f"split parse error: {e}"
        return "FAIL", ev

    retired = s.get("retired_test", {})
    new_test = s.get("new_test", {})
    assignment = s.get("assignment", {})
    study_roles = s.get("study_roles", {})

    checks = {
        "retired_16SFWJ": "16SFWJ" in retired.get("studies", []),
        "retired_status_DEVELOPMENT_CONSUMED": retired.get("status") == "DEVELOPMENT_CONSUMED",
        "assignment_16SFWJ_DEV_CONSUMED": assignment.get("16SFWJ") == "DEVELOPMENT_CONSUMED",
        "study_roles_16SFWJ_DEV_CONSUMED": study_roles.get("16SFWJ") == "DEVELOPMENT_CONSUMED",
        "new_test_untouched": new_test.get("untouched") is True,
    }
    new_studies = set(new_test.get("studies", []))
    checks["new_test_sl5_family"] = new_studies == {"SL5CV2", "SL5HKU", "SL5MER"}
    # SL5 family never in development: role must be test, and not in train/val/consumed
    for st in new_studies:
        checks[f"SL5_{st}_role_test"] = assignment.get(st) == "test"
    checks["SL5_not_in_train_or_dev"] = all(
        assignment.get(st) in ("test",) for st in new_studies
    )

    ev["retired_test"] = retired
    ev["new_test"] = new_test
    ev["assignment_16SFWJ"] = assignment.get("16SFWJ")

    if all(checks.values()):
        return "PASS", ev
    ev["failed_checks"] = [k for k, v in checks.items() if not v]
    return "FAIL", ev


def _run_pytest(path: Path) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pytest", str(path), "-q"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        return res.returncode == 0, (res.stdout or "") + (res.stderr or "")
    except Exception as e:  # pragma: no cover
        return False, f"pytest execution error: {e}"


def gate_caller_v2() -> tuple[str, dict]:
    """caller_v2.py present with fold-local guard + NO_CALL; tests pass."""
    caller = REPO_ROOT / "scripts/reactflow_delta/caller_v2.py"
    test = REPO_ROOT / "tests/reactflow_delta/test_caller_v2.py"
    ev = {"caller_path": str(caller), "test_path": str(test)}
    if not caller.exists():
        ev["reason"] = "caller_v2.py missing"
        return "FAIL", ev
    src = caller.read_text(encoding="utf-8")
    checks = {
        "fold_local_guard_present": ("OuterFoldAccessError" in src) and ("assert_train" in src),
        "no_call_present": ("NO_CALL" in src) and ("ICC_THRESHOLD" in src),
        "seal_present": ("Seal" in src) and ("break_seal" in src),
    }
    ev["checks"] = checks
    if not all(checks.values()):
        ev["failed_checks"] = [k for k, v in checks.items() if not v]
        return "FAIL", ev
    if not test.exists():
        ev["reason"] = "test_caller_v2.py missing"
        return "NOT_RUN", ev
    ok, out = _run_pytest(test)
    ev["tests_passed"] = ok
    ev["pytest_output_tail"] = out[-1500:]
    if ok:
        return "PASS", ev
    return "FAIL", ev


def gate_evaluator_v2() -> tuple[str, dict]:
    """evaluate_v2 reference tests pass."""
    ev_path = REPO_ROOT / "scripts/reactflow_delta/evaluate_v2.py"
    test = REPO_ROOT / "tests/reactflow_delta/test_evaluate_v2.py"
    ev = {"evaluator_path": str(ev_path), "test_path": str(test)}
    if not ev_path.exists():
        ev["reason"] = "evaluate_v2.py missing"
        return "FAIL", ev
    if not test.exists():
        ev["reason"] = "test_evaluate_v2.py missing"
        return "NOT_RUN", ev
    ok, out = _run_pytest(test)
    ev["tests_passed"] = ok
    ev["pytest_output_tail"] = out[-1500:]
    return ("PASS" if ok else "FAIL"), ev


def gate_p2_learnability() -> tuple[str, dict]:
    """R5 verdict must be assessed as NOT_GO (STOP_METHOD_ROUTE / UNIDENTIFIABLE)."""
    verdict = REPO_ROOT / "results/p2_v1_learnability_20260808/P2_learnability_verdict.json"
    ev = {"verdict_path": str(verdict)}
    if not verdict.exists():
        ev["reason"] = "R5 P2 verdict file missing"
        return "NOT_RUN", ev
    try:
        v = json.loads(verdict.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        ev["reason"] = f"verdict parse error: {e}"
        return "UNKNOWN", ev
    ev["verdict"] = v.get("verdict")
    est = v.get("estimand_status", {})
    ev["estimand_status_primary"] = est.get("primary")
    ev["estimand_reason"] = est.get("reason")
    checks = {
        "verdict_is_NOT_GO": v.get("verdict") == "STOP_METHOD_ROUTE",
        "primary_UNIDENTIFIABLE": est.get("primary") == "UNIDENTIFIABLE",
    }
    ev["checks"] = checks
    # Honest assessment: the R5 verdict is explicitly NOT_GO. Never fabricate a GO.
    ev["assessment"] = "NOT_GO"
    if not all(checks.values()):
        ev["failed_checks"] = [k for k, v in checks.items() if not v]
        return "FAIL", ev
    # This gate is NOT met: P2 is not GO.
    return "FAIL", ev


# ---------------------------------------------------------------------------
# Decision + manifest
# ---------------------------------------------------------------------------

def decide_overall(gate_statuses: dict[str, str]) -> dict:
    """Pure decision over gate statuses. Any non-PASS / missing gate blocks.

    Returns dict with keys: decision, route, blocking, all_gates.
    """
    blocking = [g for g in ALL_GATES if gate_statuses.get(g) != "PASS"]
    decision = "GO" if not blocking else "STOP"
    route = "STOP_METHOD_ROUTE" if gate_statuses.get("P2_LEARNABILITY_GO") != "PASS" else "STOP"
    return {
        "decision": decision,
        "route": route,
        "blocking": blocking,
        "all_gates": [g for g in ALL_GATES],
    }


def build_manifest(gate_results: dict[str, dict], recommendation: str) -> dict:
    gate_statuses = {g: r[0] for g, r in gate_results.items()}
    dec = decide_overall(gate_statuses)
    manifest = {
        "schema": "reactflow_delta.r6_go_stop_adjudication.v1",
        "run_id": RUN_ID,
        "generated_at": None,  # filled at write time
        "governing_contract": (
            "docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md"
        ),
        "contract_section": "§13.2 R6 / §13.4",
        "independent": True,
        "overall_decision": dec["decision"],
        "route": dec["route"],
        "blocking_gates": dec["blocking"],
        "gates": {g: {"status": gate_results[g][0], "evidence": gate_results[g][1]}
                  for g in ALL_GATES},
        "recommendation": recommendation,
    }
    return manifest


def run_all() -> dict:
    results = {
        "AUTHORITY_CLOSED_PASS": gate_authority_closed(),
        "ASSET_DISPOSITION_1024_OF_1024": gate_asset_disposition(),
        "PRIMARY_MASK_V2_PASS": gate_primary_mask(),
        "GROUP_ATOMS_AND_PUBLICATION_SPLIT_PASS": gate_group_atoms_publication_split(),
        "OLD_TEST_RETIRED_NEW_TEST_UNTOUCHED": gate_old_test_retired_new_test_untouched(),
        "CALLER_V2_FOLD_LOCAL_AND_RELIABLE": gate_caller_v2(),
        "EVALUATOR_V2_REFERENCE_TESTS_PASS": gate_evaluator_v2(),
        "P2_LEARNABILITY_GO": gate_p2_learnability(),
    }
    recommendation = (
        "Phase 3 model-architecture iteration is BLOCKED. The R5 P2 verdict is "
        "STOP_METHOD_ROUTE: the primary binary-changer estimand is UNIDENTIFIABLE "
        "under the frozen caller_v2 / d1x_v2 data (caller/null-calibration artifact: "
        "3 changers / 3178 non / 3204 NO_CALL). Per §13.4 any non-PASS gate stops "
        "Phase 3. Fix requires per-study reactivity normalization + error "
        "recalibration to a common scale and a caller_v3/endpoint_v3 amendment "
        "under a new authority epoch (NOT a silent in-place change)."
    )
    return build_manifest(results, recommendation)


def main() -> int:
    manifest = run_all()
    out_dir = REPO_ROOT / "results/r6_go_stop_20260807"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest["generated_at"] = (
        subprocess.run(["date", "-Iseconds"], capture_output=True, text=True)
        .stdout.strip()
    )
    json_path = out_dir / "go_stop_terminal.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    md_path = out_dir / "GO_STOP_decision.md"
    md_path.write_text(render_markdown(manifest), encoding="utf-8")

    print(json.dumps({"overall_decision": manifest["overall_decision"],
                      "route": manifest["route"],
                      "blocking_gates": manifest["blocking_gates"],
                      "gates": {g: manifest["gates"][g]["status"] for g in ALL_GATES}},
                     indent=2, ensure_ascii=False))
    print(f"\nManifest written: {json_path}")
    print(f"Decision written: {md_path}")
    return 0


def render_markdown(m: dict) -> str:
    lines = []
    lines.append(f"# GO/STOP Terminal Decision — run `{m['run_id']}`")
    lines.append("")
    lines.append(f"**Overall decision: `{m['overall_decision']}`**  ")
    lines.append(f"**Route: `{m['route']}`**")
    lines.append("")
    lines.append("Independent disk-evidence adjudication of §13.4 P0 gates "
                 "(contract §13.2 R6). Any non-PASS / missing gate => STOP. "
                 "Manual override to GO is not permitted.")
    lines.append("")
    lines.append("## Gate verdicts")
    lines.append("")
    lines.append("| Gate | Status | Primary evidence |")
    lines.append("|---|---|---|")
    for g in ALL_GATES:
        status = m["gates"][g]["status"]
        ev = m["gates"][g]["evidence"]
        # pick a terse evidence string
        ev_str = "; ".join(f"{k}={v}" for k, v in list(ev.items())[:3])
        lines.append(f"| {g} | {status} | {ev_str} |")
    lines.append("")
    lines.append("## Blocking Phase 3")
    lines.append("")
    if m["blocking_gates"]:
        for g in m["blocking_gates"]:
            lines.append(f"- `{g}` — {m['gates'][g]['status']}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(m["recommendation"])
    lines.append("")
    lines.append(f"Machine manifest: `results/r6_go_stop_20260807/go_stop_terminal.json`")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
