#!/usr/bin/env bash
# run_submission_final.sh - regenerate the definitive M2R submission table
# including rD + formula-blend audit reports.
set -euo pipefail
BASE=/mnt/cunyuliu
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
OUT="$BASE/m2r_submission_final_20260817"
cd /home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/scripts/reactflow_delta

"$PY" m2r_submission_horizontal_table_v1.py \
  --transfer-report "$BASE/m2r_transfer_20260816/m2r_transfer_report.json" \
  --robust-report "$BASE/m2r_robust_objective_20260817/m2r_robust_objective_report.json" \
  --robust-permtest "$BASE/m2r_robust_objective_20260817/m2r_robust_permtest.json" \
  --puzzle-report "$BASE/m2r_transfer_puzzle_20260817/m2r_transfer_puzzle_report.json" \
  --noise-floor "$BASE/m2r_noise_floor_20260817/m2r_noise_floor.json" \
  --threeway-report "$BASE/m2r_3way_ensemble_20260817/m2r_3way_ensemble_report.json" \
  --threeway-permtest "$BASE/m2r_3way_ensemble_20260817/m2r_3way_permtest.json" \
  --threeway-puzzle-report "$BASE/m2r_3way_puzzle_20260817/m2r_3way_puzzle_report.json" \
  --threeway-strong-report "$BASE/m2r_3way_strong_20260817/m2r_3way_strong_report.json" \
  --threeway-strong-permtest "$BASE/m2r_3way_strong_20260817/m2r_3way_strong_permtest.json" \
  --threeway-strong-puzzle-report "$BASE/m2r_3way_strong_puzzle_20260817/m2r_3way_strong_puzzle_report.json" \
  --ceiling-audit-report "$BASE/m2r_ceiling_audit_lean_20260817/m2r_ceiling_audit_report.json" \
  --features-v2-report "$BASE/m2r_features_v2_ablation_20260817/m2r_features_v2_ablation_report.json" \
  --features-v2-permtest "$BASE/m2r_features_v2_ablation_20260817/m2r_features_v2_permtest.json" \
  --features-v2-puzzle-report "$BASE/m2r_features_v2_ablation_20260817/m2r_features_v2_puzzle_report.json" \
  --doublemut-report "$BASE/m2r_doublemut_pred_20260817/m2r_doublemut_pred_report.json" \
  --formula-blend-report "$BASE/m2r_formula_blend_20260817/m2r_formula_blend_report.json" \
  --multiseed-report "$BASE/m2r_multiseed_20260817/m2r_multiseed_report.json" \
  --multiseed-permtest "$BASE/m2r_multiseed_20260817/m2r_multiseed_permtest.json" \
  --multiseed-puzzle-report "$BASE/m2r_multiseed_puzzle_20260817/m2r_multiseed_puzzle_report.json" \
  --out "$OUT"
echo "EXIT=$?"
