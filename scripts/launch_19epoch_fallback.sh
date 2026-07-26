#!/bin/bash
# Fallback: Launch multi-epoch training if 1-epoch does not beat ViennaRNA
# Usage: bash scripts/launch_19epoch_fallback.sh [epochs]
# Default: 5 epochs (~3.5 days). Use 19 for full objective (~13.5 days).
# Note: FSDP skips in-script validation, so early_stop_patience has no effect.
set -euo pipefail
cd /home/cunyuliu/reactflow_c1_3_stage_20260722

NUM_EPOCHS="${1:-5}"
CONFIG="configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_ddp.yaml"
RESULTS_DIR="artifacts/c1_3/runs/pairformer_ribonanza_frozen_small_pair_fsdp_seed0_${NUM_EPOCHS}ep"
mkdir -p "${RESULTS_DIR}"

echo "[$(date)] Launching ${NUM_EPOCHS}-epoch training"
echo "[$(date)] Config: ${CONFIG}"
echo "[$(date)] Output: ${RESULTS_DIR}"

torchrun --nproc_per_node=3 --master_port=29516 \
    scripts/train_c1_3.py \
    --config "${CONFIG}" \
    --seed 0 \
    --batch-size 4 \
    --max-len 512 \
    --eval-decoders threshold,mea \
    --output-dir "${RESULTS_DIR}" \
    --epochs "${NUM_EPOCHS}" \
    2>&1 | tee /tmp/c1_3_${NUM_EPOCHS}epoch.log

echo "[$(date)] Done. Eval with:"
echo "  python3 scripts/eval_and_generate_grid.py --config ${CONFIG} --checkpoint ${RESULTS_DIR}/latest.pt --device cuda:0"
