#!/bin/bash
# Launch multi-seed training for C1-3 gate.
# Usage: bash scripts/launch_multiseed.sh <num_seeds> [epochs_per_seed] [max_train_samples]
# Default: 10 seeds, 1 epoch each, 0 (full dataset)
#
# Examples:
#   # Full dataset, 1 epoch (slowest, most accurate)
#   bash scripts/launch_multiseed.sh 10 1 0
#
#   # 50K samples, 1 epoch (fast, ~4h per seed)
#   bash scripts/launch_multiseed.sh 10 1 50000
#
#   # 100K samples, 1 epoch (medium, ~8h per seed)
#   bash scripts/launch_multiseed.sh 10 1 100000
#
# This script runs seeds sequentially using FSDP on 3 GPUs (0, 2, 5).
# Each seed trains for the specified number of epochs and evaluates.

set -euo pipefail

NUM_SEEDS="${1:-10}"
EPOCHS_PER_SEED="${2:-1}"
MAX_TRAIN_SAMPLES="${3:-0}"
PROJECT_DIR="/home/cunyuliu/reactflow_c1_3_stage_20260722"
CONFIG="configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_ddp.yaml"

cd "${PROJECT_DIR}"
source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
conda activate editflow

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Create a modified config for multi-seed
MULTISEED_CONFIG="configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_multiseed.yaml"
python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['training']['epochs'] = ${EPOCHS_PER_SEED}
cfg['training']['early_stop_patience'] = 0  # disable early stop for multi-seed
with open('${MULTISEED_CONFIG}', 'w') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)
print(f'Created ${MULTISEED_CONFIG} with epochs=${EPOCHS_PER_SEED}')
"

if [ "${MAX_TRAIN_SAMPLES}" -gt 0 ]; then
    log "Using max_train_samples=${MAX_TRAIN_SAMPLES} (reduced dataset for faster multi-seed)"
    STEPS_PER_EPOCH=$((MAX_TRAIN_SAMPLES / 12))  # batch_size=4 * world_size=3
    ETA_HOURS=$(python3 -c "print(f'{${STEPS_PER_EPOCH} / 0.29 / 3600:.1f}')")
    log "Estimated steps per epoch: ${STEPS_PER_EPOCH}, ETA per seed: ${ETA_HOURS}h"
fi

for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
    log "=== Training seed ${SEED}/${NUM_SEEDS} ==="

    CKPT_DIR="artifacts/c1_3/runs/pairformer_ribonanza_frozen_small_pair_fsdp_seed${SEED}"
    RESULTS_DIR="artifacts/c1_3/results_fsdp_seed${SEED}"

    # Skip if already trained
    if [ -f "${CKPT_DIR}/best.pt" ] && [ -s "${CKPT_DIR}/best.pt" ]; then
        log "Seed ${SEED} already has checkpoint, skipping training"
    else
        log "Launching training for seed ${SEED}..."
        export CUDA_VISIBLE_DEVICES=0,2,5
        export PYTHONPATH=src
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        MAX_TRAIN_FLAG=""
        if [ "${MAX_TRAIN_SAMPLES}" -gt 0 ]; then
            MAX_TRAIN_FLAG="--max-train-samples ${MAX_TRAIN_SAMPLES}"
        fi

        torchrun --nproc_per_node=3 --master_port=29516 \
            scripts/train_c1_3.py \
            --config "${MULTISEED_CONFIG}" \
            --seed ${SEED} \
            --batch-size 4 \
            --max-len 512 \
            --eval-decoders threshold,mea \
            --output-dir "${RESULTS_DIR}" \
            ${MAX_TRAIN_FLAG} \
            2>&1 | tee /tmp/c1_3_multiseed_seed${SEED}.log

        if [ ! -f "${CKPT_DIR}/best.pt" ] || [ ! -s "${CKPT_DIR}/best.pt" ]; then
            log "ERROR: Seed ${SEED} training failed - no checkpoint produced"
            continue
        fi
    fi

    # Evaluate the checkpoint
    log "Evaluating seed ${SEED}..."
    EVAL_GPU="cuda:0"
    GPU4_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | head -1)
    if [ "${GPU4_USED}" -gt 30000 ]; then
        log "GPU 4 busy, using cuda:0 for eval"
        EVAL_GPU="cuda:0"
    fi

    python3 scripts/eval_and_generate_grid.py \
        --config "${MULTISEED_CONFIG}" \
        --checkpoint "${CKPT_DIR}/best.pt" \
        --device "${EVAL_GPU}" \
        --eval-decoders threshold,mea \
        --decoder mea \
        --output-dir "artifacts/c1_3" \
        --config-name "pairformer_ribonanza_frozen_small_pair_fsdp_seed${SEED}" \
        --batch-size 4 \
        2>&1 | tee /tmp/c1_3_multiseed_eval_seed${SEED}.log

    log "Seed ${SEED} complete"
done

log "=== All seeds complete, generating multiseed_results.json ==="
python3 scripts/generate_multiseed_results.py --artifacts-dir artifacts/c1_3 --num-seeds ${NUM_SEEDS}

log "=== Generating significance report ==="
python3 scripts/generate_significance_report.py --artifacts-dir artifacts/c1_3

log "Multi-seed training and evaluation complete!"
log "Run gate audit: python3 scripts/audit_c1_3_gate.py --artifacts-dir artifacts/c1_3 --docs-dir docs"
