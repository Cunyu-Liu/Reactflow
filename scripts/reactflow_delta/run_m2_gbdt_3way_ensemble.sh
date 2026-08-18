#!/usr/bin/env bash
# run_m2_gbdt_ensemble.sh - cross-architecture GBDT + 3-way attention ensemble for M2
set -euo pipefail
M2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
ATTN=/mnt/cunyuliu/m2_response_spectrum_attn_v5_deep_20260815/keyed_predictions_m2_attn.jsonl
V3=/mnt/cunyuliu/m2_response_spectrum_posaware_20260813/keyed_predictions_m2_posaware.jsonl
V4=/mnt/cunyuliu/m2_response_spectrum_attn_gpu3_20260815/keyed_predictions_m2_attn.jsonl
V5=/mnt/cunyuliu/m2_response_spectrum_attn_v5_deep_20260815/keyed_predictions_m2_attn.jsonl
OUT=/mnt/cunyuliu/m2_gbdt_3way_ensemble_20260818
cd /home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/scripts/reactflow_delta
/home/cunyuliu/miniconda3/envs/editflow/bin/python m2_gbdt_ensemble_v1.py \
  --m2-csv "$M2" --attn-pred "$ATTN" \
  --pred-v3 "$V3" --pred-v4 "$V4" --pred-v5 "$V5" \
  --out "$OUT"
echo "EXIT=$?"
