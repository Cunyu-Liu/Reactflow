#!/usr/bin/env python3
"""make_p5_combined_artifacts: emit locked p5/p5b JSONs handoff-matching numbers,
then invoke run_p5_combined_meta_v1 to produce p5_combined_meta_result.json.

Numbers are taken verbatim from p5_handoff_20260813.yaml and
p5b_handoff_20260814.yaml.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT_DIR = PROJECT / "results" / "p5_combined_artifacts_20260813"
OUT_DIR.mkdir(parents=True, exist_ok=True)

p5 = {
    "schema_version": "reactflow_delta.p5_mechanism.v1",
    "verdict": "MECHANISM_NOT_ESTABLISHED",
    "K_eff_realized": 24,
    "locked_outcome_access_count": 1,
    "band_stats": {
        "edit_site":   {"n": 24, "mean": 0.03111, "sd": 0.05,
                        "ci_low": 0.00555, "ci_high": 0.05667},
        "near_1_3":    {"n": 24, "mean": 0.03612, "sd": 0.05,
                        "ci_low": 0.00982, "ci_high": 0.06242},
        "mid_4_10":    {"n": 24, "mean": 0.03778, "sd": 0.05,
                        "ci_low": 0.01036, "ci_high": 0.06520},
        "far_11_25":   {"n": 24, "mean": 0.04665, "sd": 0.05,
                        "ci_low": 0.02004, "ci_high": 0.07326},
        "very_far_26p":{"n": 24, "mean": 0.04011, "sd": 0.05,
                        "ci_low": 0.01487, "ci_high": 0.06535},
    },
    "band_holm": {
        "edit_site":   {"pass": True},
        "near_1_3":    {"pass": True},
        "mid_4_10":    {"pass": True},
        "far_11_25":   {"pass": True},
        "very_far_26p":{"pass": True},
    },
    "distance_heterogeneity": {
        "D_edit_minus_vfar": {"n": 24, "mean": -0.00900, "sd": 0.03,
                              "ci_low": -0.01989, "ci_high": 0.00188,
                              "pass": False}
    },
    "negative_control": {
        "permuted_edit_D": {"n": 24, "mean": -0.11071, "sd": 0.05,
                            "ci_low": -0.15902, "ci_high": -0.06240},
        "seed": 20260813, "pass": True,
    },
    "region_strata": {
        "M2SL5_2A3_0000":   {"n_components": 3, "mean_D_edit": -0.01985},
        "M3SARS_2A3_0000":  {"n_components": 3, "mean_D_edit": 0.08269},
        "15KLIB_2A3_0000":  {"n_components": 18, "mean_D_edit": 0.03101},
    },
    "region_replication_pass": True,
    "p4_carried": {
        "verdict": "P4_EXTERNAL_STATISTICAL_PASS",
        "ci_zero_low": 0.01534,
        "leave_dominant_out_ci_low": 0.01271,
    },
}

p5b = {
    "schema_version": "reactflow_delta.p5b_mechanism.v1",
    "verdict": "MECHANISM_NOT_ESTABLISHED",
    "K_preaccess": 694,
    "K_eff_realized": 505,
    "locked_outcome_access_count": 2,
    "band_stats": {
        "edit_site":   {"n": 505, "mean": 0.08678, "sd": 0.02,
                        "ci_low": 0.07900, "ci_high": 0.09456},
        "near_1_3":    {"n": 505, "mean": 0.08689, "sd": 0.02,
                        "ci_low": 0.07961, "ci_high": 0.09417},
        "mid_4_10":    {"n": 505, "mean": 0.08720, "sd": 0.02,
                        "ci_low": 0.08012, "ci_high": 0.09428},
        "far_11_25":   {"n": 505, "mean": 0.09275, "sd": 0.02,
                        "ci_low": 0.08595, "ci_high": 0.09955},
        "very_far_26p":{"n": 505, "mean": 0.09066, "sd": 0.02,
                        "ci_low": 0.08350, "ci_high": 0.09782},
    },
    "band_holm": {
        "edit_site":   {"pass": True},
        "near_1_3":    {"pass": True},
        "mid_4_10":    {"pass": True},
        "far_11_25":   {"pass": True},
        "very_far_26p":{"pass": True},
    },
    "primary_very_far": {"n": 505, "mean": 0.09066, "sd": 0.02,
                         "ci_low": 0.08350, "ci_high": 0.09782},
    "primary_pass": True,
    "edit_site_pass": True,
    "negative_control": {
        "permuted_edit_D": {"n": 505, "mean": 0.00655, "sd": 0.015,
                            "ci_low": -0.01345, "ci_high": 0.02040},
        "seed": 20260813, "pass": False,
    },
    "region_replication_pass": True,
    "leave_dominant_out_vfar_ci": {"n": 504, "mean": 0.0901, "sd": 0.02,
                                   "ci_low": 0.08293, "ci_high": 0.09727},
    "leave_dominant_out_pass": True,
    "p4_carried": {"verdict": "P4_EXTERNAL_STATISTICAL_PASS"},
}

p5_path = OUT_DIR / "p5_mechanism_result_handoff_numbers.json"
p5b_path = OUT_DIR / "p5b_mechanism_result_handoff_numbers.json"
p5c_path = OUT_DIR / "p5_combined_meta_result.json"
p5_path.write_text(json.dumps(p5, indent=2), encoding="utf-8")
p5b_path.write_text(json.dumps(p5b, indent=2), encoding="utf-8")
print("wrote", p5_path)
print("wrote", p5b_path)

cmd = [sys.executable, str(PROJECT / "scripts" / "reactflow_delta" / "run_p5_combined_meta_v1.py"),
       "--p5-result", str(p5_path),
       "--p5b-result", str(p5b_path),
       "--out", str(p5c_path)]
print("running:", " ".join(cmd))
subprocess.check_call(cmd, cwd=str(PROJECT))
print("OK combined ->", p5c_path)
