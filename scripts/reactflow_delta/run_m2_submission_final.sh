#!/usr/bin/env bash
# run_m2_submission_final.sh - regenerate the definitive M2 response-spectrum
# submission horizontal table (GBDT cross-arch + 3-way deep headline).
set -euo pipefail
BASE=/mnt/cunyuliu
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
OUT="$BASE/m2_spectrum_submission_final_20260818"
cd /home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/scripts/reactflow_delta

"$PY" m2_spectrum_submission_table_v1.py \
  --attn-summary "$BASE/m2_response_spectrum_attn_gpu3_20260815/compare_m2_attn/m2_attn_method_summary_full.json" \
  --crossarch-report "$BASE/m2_response_spectrum_attn_gpu3_20260815/compare_m2_attn/m2_crossarch_ensemble_report.json" \
  --threeway-report "$BASE/m2_response_spectrum_attn_gpu3_20260815/compare_m2_attn/m2_three_way_ensemble_report.json" \
  --fourway-report "$BASE/m2_response_spectrum_attn_gpu3_20260815/compare_m2_attn/ens4/m2_four_way_ensemble_report.json" \
  --gbdt-report "$BASE/m2_gbdt_3way_ensemble_matched_20260818/m2_masked_eval_report.json" \
  --out "$OUT"
echo "EXIT=$?"
