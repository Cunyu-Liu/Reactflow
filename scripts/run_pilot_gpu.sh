#!/bin/bash
# Pilot runner: runs 3 seeds for one config on one GPU.
# Usage: run_pilot_gpu.sh <gpu_id> <config_name> <config_file>
set -euo pipefail
GPU_ID=$1
CONFIG_NAME=$2
CONFIG_FILE=$3
STAGE_DIR=/home/cunyuliu/reactflow_c1_3_stage_20260722
LOG_DIR=/tmp/c1_3_results/pilot_logs
mkdir -p "$LOG_DIR"

source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
conda activate editflow
cd "$STAGE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for SEED in 0 1 2; do
    OUT_DIR=/tmp/c1_3_results/${CONFIG_NAME}_seed${SEED}
    CKPT_DIR=/dev/shm/c1_3/${CONFIG_NAME}_seed${SEED}
    mkdir -p "$OUT_DIR" "$CKPT_DIR"
    LOG_FILE=$LOG_DIR/${CONFIG_NAME}_seed${SEED}.log
    echo "[$(date)] Starting ${CONFIG_NAME} seed=${SEED} on GPU ${GPU_ID}" | tee "$LOG_FILE"
    PYTHONPATH=src python scripts/train_c1_3.py \
        --config "$CONFIG_FILE" \
        --seed "$SEED" \
        --device "cuda:${GPU_ID}" \
        --max-train-samples 10000 \
        --max-eval-samples 1000 \
        --epochs 5 \
        --batch-size 4 \
        --max-len 400 \
        --output-dir "$OUT_DIR" \
        --checkpoint-dir "$CKPT_DIR" \
        >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    echo "[$(date)] Finished ${CONFIG_NAME} seed=${SEED} (exit ${EXIT_CODE})" | tee -a "$LOG_FILE"
done
echo "[$(date)] All seeds complete for ${CONFIG_NAME}" | tee -a "$LOG_FILE"
