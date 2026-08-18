#!/usr/bin/env bash
# run_m2_gbdt_puzzle_ensemble.sh - PUZZLE-level LOPO audit of the M2 GBDT + deep
# ensemble (train 19 puzzles -> predict held-out puzzle, all components leak-free).
set -euo pipefail
M2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
PUZZLE=/mnt/cunyuliu/m2_attn_puzzle_20260817/keyed_predictions_m2_attn_puzzle.jsonl
OUT=/mnt/cunyuliu/m2_gbdt_puzzle_ensemble_20260818
cd /home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/scripts/reactflow_delta
/home/cunyuliu/miniconda3/envs/editflow/bin/python m2_gbdt_puzzle_ensemble_v1.py \
  --m2-csv "$M2" --puzzle-pred "$PUZZLE" --out "$OUT"
echo "EXIT=$?"
