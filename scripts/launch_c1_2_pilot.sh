#!/usr/bin/env bash
# Launch the C1-2 3-seed pilot for all 4 matched-capacity models.
#
# Spec reference: ReactFlow分阶段执行提示词.md lines 396-430.
#
# Runs 3 seeds (0, 1, 2) for each of the 4 models:
#   - pairformer_compact (the new symmetric PairFormer)
#   - bilinear_baseline  (no pair stack)
#   - cnn_baseline       (2D CNN pair stack, no triangle updates)
#   - unet_baseline      (UNet encoder-decoder pair stack)
#
# Total: 4 models x 3 seeds = 12 training runs + 12 evaluation runs.
#
# Usage:
#   bash scripts/launch_c1_2_pilot.sh [GPU_ID] [MAX_PER_SPLIT]
#
# Defaults: GPU_ID=5, MAX_PER_SPLIT=500 (for fast pilot)
# Note: GPU_ID sets CUDA_VISIBLE_DEVICES, and the script uses cuda:0 internally.
#       Avoid GPU 4 (calibrate PID 2544995) and GPUs 6/7 (MIG enabled, 5GB slices).

set -euo pipefail

GPU_ID=${1:-5}
MAX_PER_SPLIT=${2:-500}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
DEVICE="cuda:0"
STAGE_DIR="/home/cunyuliu/reactflow_c1_2_stage_20260721"
RUNS_DIR="${STAGE_DIR}/artifacts/c1_2/runs"
LOG_DIR="${STAGE_DIR}/artifacts/c1_2/logs"

mkdir -p "${RUNS_DIR}" "${LOG_DIR}"

cd "${STAGE_DIR}"

# Activate conda env
source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
conda activate editflow

MODELS=(
    "pairformer_compact"
    "bilinear_baseline"
    "cnn_baseline"
    "unet_baseline"
)
SEEDS=(0 1 2)

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${MODEL}_seed${SEED}"
        RUN_DIR="${RUNS_DIR}/${RUN_NAME}"
        LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

        if [ -f "${RUN_DIR}/evaluation_results.json" ]; then
            echo "[skip] ${RUN_NAME} already has evaluation results"
            continue
        fi

        echo "=========================================="
        echo "[train] ${RUN_NAME} on GPU ${GPU_ID} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
        echo "=========================================="

        CONFIG="configs/models/${MODEL}.yaml"

        if [ -f "${RUN_DIR}/checkpoint_best.pt" ]; then
            echo "[skip train] ${RUN_NAME} checkpoint already exists, reusing"
        else
            python scripts/train_static_pairformer.py \
                --config "${CONFIG}" \
                --output "${RUN_DIR}" \
                --seed "${SEED}" \
                --device "${DEVICE}" \
                --max-per-split "${MAX_PER_SPLIT}" \
                2>&1 | tee "${LOG_FILE}"
        fi

        echo "=========================================="
        echo "[eval] ${RUN_NAME}"
        echo "=========================================="

        python scripts/evaluate_static_pairformer.py \
            --checkpoint "${RUN_DIR}/checkpoint_best.pt" \
            --output "${RUN_DIR}/evaluation_results.json" \
            --device "${DEVICE}" \
            --max-samples 500 \
            --batch-size 8 \
            --num-workers 2 \
            2>&1 | tee -a "${LOG_FILE}"

        echo "[done] ${RUN_NAME}"
    done
done

echo "=========================================="
echo "[C1-2 pilot] all runs complete"
echo "=========================================="
echo "Results: ${RUNS_DIR}"
ls -la "${RUNS_DIR}"/*/evaluation_results.json 2>/dev/null || true
