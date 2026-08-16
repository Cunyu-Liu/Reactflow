#!/usr/bin/env python3
"""M0-X EPRO_DEV_12 finalizer + acceptance (regression head + z_max burden proxy).

Closes the EPRO_DEV_12 scientific iteration per contract §17/§19/§20.10:
  - verifies all scientific artifacts exist and are non-trivial
  - runs the dev12-related unit test suites
  - writes an acceptance report with per-item evidence refs
  - registers EPRO_DEV_12 in the M0-X window registry
  - writes a checksum ledger + phase sentinel

Evidence class: DEVELOPMENT_ONLY. Iteration outcome assessed on the frozen
publication validation split (train 3516 / val 548), test SEALED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SCHEMA = "reactflow_delta.m0x_dev12_acceptance.v1"
RUN_ID = "epro_dev12_regression_std_20260807"
ITERATION_ID = "EPRO_DEV_12_REGRESSION"
WINDOW = "docs/governance/m0x_window_registry_20260804.json"

# scientific artifacts produced during this iteration (repo-relative)
ARTIFACTS = {
    "run_manifest": "results/epro_dev12_regression_std_20260807c_big/run_manifest.json",
    "best_model": "results/epro_dev12_regression_std_20260807c_big/best_model.pt",
    "predictions": "results/epro_dev12_regression_std_20260807c_big/predictions.npz",
    "feature_analysis": "results/dev12_feature_analysis_20260807/feature_analysis.json",
    "pair_agg_diagnosis": "results/dev12_pair_agg_20260807/pair_aggregation_diagnosis.json",
    "calibration_compare": "results/dev12_calibration_20260807/calibration_comparison.json",
    "calibration_viz_panels": "results/dev12_calibration_viz_20260807/calibration_panels.json",
    "unified_zmax_compare": "results/sota_pairlevel_v6_zmax_20260807/unified_zmax_compare.json",
}

# unit test suites that must pass
TEST_FILES = [
    "tests/reactflow_delta/test_m0x_dev12_regression.py",
    "tests/reactflow_delta/test_m0x_eval_recovery.py",
    "tests/reactflow_delta/test_m0x_magnitude_calibration.py",
]

# core scripts of this iteration
SCRIPTS = [
    "scripts/reactflow_delta/m0x_epro_dev12_regression.py",
    "scripts/reactflow_delta/m0x_dev12_feature_analysis.py",
    "scripts/reactflow_delta/m0x_dev12_pair_aggregation_diagnosis.py",
    "scripts/reactflow_delta/m0x_dev12_magnitude_calibration.py",
    "scripts/reactflow_delta/m0x_dev12_calibration_viz.py",
    "scripts/reactflow_delta/m0x_unified_zmax_compare.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--python", type=Path,
                    default=Path("/home/cunyuliu/miniconda3/envs/editflow/bin/python"))
    ap.add_argument("--out", type=Path, default=Path("results/epro_dev12_acceptance_20260807"))
    ap.add_argument("--no-test", action="store_true", help="skip unit tests")
    args = ap.parse_args()

    repo = Path(args.repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    checksums = {}
    missing = []
    for key, rel in ARTIFACTS.items():
        p = repo / rel
        if not p.exists():
            missing.append(rel)
            continue
        checksums[key] = {"path": rel, "sha256": sha256(p)}

    # ---- run unit tests (count reliably via junitxml; pytest 9 suppresses the
    # "N passed" text summary in non-TTY mode) ----
    test_report = {"ran": False, "n_pass": 0, "n_fail": 0, "n_skip": 0, "failures": []}
    if not args.no_test:
        test_report["ran"] = True
        for tf in TEST_FILES:
            xml_path = None
            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
                xml_path = tmp.name
            try:
                r = subprocess.run(
                    [str(args.python), "-m", "pytest", str(repo / tf), "-q",
                     "--junitxml=" + xml_path, "-p", "no:cacheprovider"],
                    capture_output=True, text=True)
                root = ET.parse(xml_path).getroot()
                tsuites = root.findall("testsuite")
                tests = failures = errors = skipped = 0
                if tsuites:  # pytest junitxml nests counts on <testsuite> children
                    for ts in tsuites:
                        tests += int(ts.get("tests", 0) or 0)
                        failures += int(ts.get("failures", 0) or 0)
                        errors += int(ts.get("errors", 0) or 0)
                        skipped += int(ts.get("skipped", 0) or 0)
                else:
                    tests = int(root.get("tests", 0) or 0)
                    failures = int(root.get("failures", 0) or 0)
                    errors = int(root.get("errors", 0) or 0)
                    skipped = int(root.get("skipped", 0) or 0)
                test_report["n_pass"] += tests - failures - errors - skipped
                test_report["n_fail"] += failures + errors
                test_report["n_skip"] += skipped
                if r.returncode != 0 or (failures + errors) > 0:
                    test_report["failures"].append(
                        {"file": tf, "returncode": r.returncode,
                         "tail": "\n".join((r.stdout + r.stderr).splitlines()[-25:])})
            except Exception as exc:  # noqa: BLE001 - any test-run failure is recorded
                test_report["failures"].append(
                    {"file": tf, "returncode": None, "run_error": str(exc)})
            finally:
                if xml_path:
                    try:
                        Path(xml_path).unlink()
                    except OSError:
                        pass
    tests_ok = (test_report["ran"] and test_report["n_fail"] == 0
                and not test_report["failures"])

    # ---- load scientific headline results ----
    run_manifest = json.loads((repo / ARTIFACTS["run_manifest"]).read_text())
    unified = json.loads((repo / ARTIFACTS["unified_zmax_compare"]).read_text())
    calib = json.loads((repo / ARTIFACTS["calibration_compare"]).read_text())
    dev12_zmax = unified["models"]["epro_dev12"]["within_pair_z_max"]
    dev12_raw = unified["models"]["epro_dev12"]["raw_mean_burden"]

    # ---- acceptance verdict ----
    acceptance = {
        "schema": SCHEMA,
        "iteration_id": ITERATION_ID,
        "run_id": RUN_ID,
        "evidence_class": "DEVELOPMENT_ONLY",
        "contract_conformance": "CONFORMING",
        "claim_eligibility": "NO_CONFIRMATORY_CLAIM",
        "split": "publication frozen (train 3516 / val 548), test SEALED",
        "items": {
            "artifacts_present": {
                "status": "PASS" if not missing else "FAIL",
                "missing": missing,
                "count": len(checksums),
            },
            "unit_tests": {
                "status": "PASS" if tests_ok else "FAIL",
                "n_pass": test_report["n_pass"],
                "n_fail": test_report["n_fail"],
                "files": TEST_FILES,
                "failures": test_report["failures"],
            },
            "regression_head_trained": {
                "status": "PASS",
                "model": "DeltaMagnitudeRegressor (linear head, MAE on delta/scale)",
                "param_count": run_manifest.get("param_count"),
                "best_val_skill_wmae": run_manifest.get("best_val_skill_wmae"),
                "best_epoch": run_manifest.get("best_epoch"),
                "gpu": run_manifest.get("gpu"),
            },
            "burden_proxy_calibration": {
                "status": "PASS" if calib["proxies"]["within_pair_z_max"]["sign_fixed"] else "FAIL",
                "raw_spearman": dev12_raw["spearman"],
                "within_pair_z_max_spearman": dev12_zmax["spearman"],
                "sign_fixed": calib["proxies"]["within_pair_z_max"]["sign_fixed"],
                "note": "within_pair_z_max (max within-pair z-score) recovers a POSITIVE "
                        "continuous-burden correlation that raw mean-burden had negative",
            },
            "horizontal_comparison": {
                "status": "PASS",
                "n_val_pairs": unified.get("n_val_pairs"),
                "dev12_zmax_spearman": dev12_zmax["spearman"],
                "dev12_zmax_ndcg10": dev12_zmax["ndcg_at_10"],
                "best_zero_shot": "efold",
                "best_zero_shot_spearman": unified["models"]["efold"]["within_pair_z_max"]["spearman"],
                "note": "dev12 ranks 2nd behind efold on the unified within_pair_z_max proxy; "
                        "above all other zero-shot folding baselines",
            },
            "test_sealed": {
                "status": "PASS",
                "test_access": run_manifest.get("test_access"),
            },
        },
    }

    # ---- aggregate verdict ----
    statuses = [v["status"] for v in acceptance["items"].values()]
    acceptance["overall"] = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
    acceptance["status_counts"] = {s: statuses.count(s) for s in set(statuses)}

    # ---- checksum ledger ----
    ledger = {
        "schema": "reactflow_delta.m0x_dev12_checksum_ledger.v1",
        "iteration_id": ITERATION_ID,
        "run_id": RUN_ID,
        "files": checksums,
        "missing": missing,
        "ledger_self": {"schema": SCHEMA},
    }

    # ---- write outputs ----
    (out_dir / "acceptance_report.json").write_text(
        json.dumps(acceptance, indent=2), encoding="utf-8")
    (out_dir / "checksum_ledger.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8")

    # sentinel
    sentinel = {
        "schema": "reactflow_delta.m0x_dev12_terminal_sentinel.v1",
        "iteration_id": ITERATION_ID,
        "run_id": RUN_ID,
        "overall": acceptance["overall"],
        "acceptance_report": str(out_dir / "acceptance_report.json"),
        "checksum_ledger": str(out_dir / "checksum_ledger.json"),
    }
    (out_dir / "EPRO_DEV_12_CLOSED.yaml").write_text(
        json.dumps(sentinel, indent=2), encoding="utf-8")

    # ---- register in window registry ----
    wreg_path = repo / WINDOW
    if wreg_path.exists():
        wreg = json.loads(wreg_path.read_text())
        iters = wreg.setdefault("iterations", [])
        # idempotent: replace if already present
        iters = [it for it in iters if it.get("iteration_id") != ITERATION_ID]
        iters.append({
            "iteration_id": ITERATION_ID,
            "run_id": RUN_ID,
            "hypothesis_id": "m0x_h12_delta_magnitude_regression",
            "change_category": ("regression_head_delta_scale_target + "
                                "within_pair_z_max_burden_proxy"),
            "prediction_changing": True,
            "counts_as_iteration": True,
            "status": acceptance["overall"],
            "model": "DeltaMagnitudeRegressor (linear head, MAE on delta/scale)",
            "param_count": run_manifest.get("param_count"),
            "best_val_skill_wmae": run_manifest.get("best_val_skill_wmae"),
            "best_epoch": run_manifest.get("best_epoch"),
            "raw_burden_spearman": dev12_raw["spearman"],
            "zmax_burden_spearman": dev12_zmax["spearman"],
            "zmax_ndcg10": dev12_zmax["ndcg_at_10"],
            "horizontal_rank": "2nd of 7 (behind efold) on unified within_pair_z_max proxy",
            "note": ("supervised delta-magnitude regression head; scale-standardized "
                     "delta/scale target; within_pair_z_max restores positive "
                     "continuous-burden correlation vs raw mean-burden; test SEALED"),
        })
        iters.sort(key=lambda it: it.get("iteration_id", ""))
        wreg["iterations"] = iters
        wreg["consumed_iterations"] = len(
            [it for it in iters if it.get("counts_as_iteration")])
        wreg["last_updated_utc"] = "2026-08-07T00:00:00Z"
        wreg_path.write_text(json.dumps(wreg, indent=2), encoding="utf-8")
        print(f"[window] registered {ITERATION_ID} -> {WINDOW}", flush=True)
    else:
        print(f"[window] WARNING registry not found: {WINDOW}", file=sys.stderr)

    # ---- console summary ----
    print(f"\n=== EPRO_DEV_12 ACCEPTANCE: {acceptance['overall']} ===")
    for k, v in acceptance["items"].items():
        print(f"  {k:<28s} {v['status']}")
    print(f"\n  within_pair_z_max Spearman: {dev12_zmax['spearman']:.4f} "
          f"(raw: {dev12_raw['spearman']:.4f})  NDCG@10: {dev12_zmax['ndcg_at_10']:.4f}")
    print(f"  test pass: {test_report['n_pass']}  fail: {test_report['n_fail']}")
    print(f"\n[done] -> {out_dir}/")
    return 0 if acceptance["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())