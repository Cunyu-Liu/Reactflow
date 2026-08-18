#!/usr/bin/env bash
# run_mfe_multiseed.sh - design-level multi-seed 3-way + MFE features audit
set -euo pipefail
M2R=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv
M2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
M2PRED=/mnt/cunyuliu/m2_response_spectrum_attn_v5_deep_20260815/keyed_predictions_m2_attn.jsonl
BASE=/mnt/cunyuliu/m2r_multiseed_20260817/m2r_multiseed_oof.npz
OUT=/mnt/cunyuliu/m2r_mfe_multiseed_20260818
cd /home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/scripts/reactflow_delta
/home/cunyuliu/miniconda3/envs/editflow/bin/python m2r_mfe_multiseed_v1.py \
  --m2r-csv "$M2R" --m2-csv "$M2" --m2-pred "$M2PRED" \
  --base-npz "$BASE" --out "$OUT" --n-perm 500
echo "EXIT=$?"
