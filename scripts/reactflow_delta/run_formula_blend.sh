#!/usr/bin/env bash
# run_formula_blend.sh - physics-constrained 4th blend member (reuses rD OOF npz)
set -euo pipefail
NPZ=/mnt/cunyuliu/m2r_doublemut_pred_20260817/m2r_doublemut_pred_oof.npz
M2R=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv
OUT=/mnt/cunyuliu/m2r_formula_blend_20260817
cd /home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/scripts/reactflow_delta
/home/cunyuliu/miniconda3/envs/editflow/bin/python m2r_formula_blend_v1.py \
  --npz "$NPZ" --m2r-csv "$M2R" --out "$OUT"
echo "EXIT=$?"
