#!/usr/bin/env bash
# run_multiseed_puzzle.sh - puzzle-level leak-free multi-seed averaging audit
set -euo pipefail
M2R=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2R_data.v4.5.1.csv
M2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
M2PZ=/mnt/cunyuliu/m2_attn_puzzle_20260817/keyed_predictions_m2_attn_puzzle.jsonl
OUT=/mnt/cunyuliu/m2r_multiseed_puzzle_20260817
cd /home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/scripts/reactflow_delta
/home/cunyuliu/miniconda3/envs/editflow/bin/python m2r_multiseed_puzzle_v1.py \
  --m2r-csv "$M2R" --m2-csv "$M2" --m2-pred-puzzle "$M2PZ" \
  --out "$OUT" --n-perm 500 --k-seeds 5
echo "EXIT=$?"
