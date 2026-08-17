#!/usr/bin/env bash
# run_multiseed_audit.sh - significance of the multi-seed averaging gain
set -euo pipefail
NPZ=/mnt/cunyuliu/m2r_multiseed_20260817/m2r_multiseed_oof.npz
OUT=/mnt/cunyuliu/m2r_multiseed_20260817
cd /home/cunyuliu/reactflow_delta_worktrees/benchmark_v3_20260809/scripts/reactflow_delta
/home/cunyuliu/miniconda3/envs/editflow/bin/python m2r_multiseed_permtest_v1.py \
  --npz "$NPZ" --out "$OUT" --n-perm 500 --n-boot 500
echo "EXIT=$?"
