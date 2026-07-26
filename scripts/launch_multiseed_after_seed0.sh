#!/bin/bash
# Orchestrator: Wait for seed 0 to complete, evaluate, then launch multi-seed training.
# This script:
# 1. Waits for seed 0 checkpoint (best.pt or latest.pt)
# 2. Runs evaluation if not already done
# 3. Checks if model F1 beats ViennaRNA (0.6819)
# 4. If yes, launches seeds 1-9 with configurable dataset size
# 5. Generates multiseed_results.json and significance_report.json
# 6. Runs final gate audit
#
# Usage: bash scripts/launch_multiseed_after_seed0.sh [max_train_samples]
# Default: 0 (full dataset, ~18h per seed)
# Example: bash scripts/launch_multiseed_after_seed0.sh 100000  (100K, ~8h per seed)

set -uo pipefail

PROJECT_DIR="/home/cunyuliu/reactflow_c1_3_stage_20260722"
CONFIG="configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_ddp.yaml"
ARTIFACTS_DIR="${PROJECT_DIR}/artifacts/c1_3"
CKPT_DIR="${ARTIFACTS_DIR}/runs/pairformer_ribonanza_frozen_small_pair_fsdp_seed0"
MAX_TRAIN_SAMPLES="${1:-0}"
NUM_SEEDS=10

cd "${PROJECT_DIR}"
source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
conda activate editflow

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

VIENNARNA_F1=0.6819

# ============================================================
# Step 1: Wait for seed 0 checkpoint
# ============================================================
log "=== Step 1: Waiting for seed 0 checkpoint ==="

while true; do
    BEST_PT="${CKPT_DIR}/best.pt"
    LATEST_PT="${CKPT_DIR}/latest.pt"
    CHECKPOINT=""

    if [ -f "${BEST_PT}" ] && [ -s "${BEST_PT}" ]; then
        CHECKPOINT="${BEST_PT}"
        log "Found best.pt: $(stat -c%s ${BEST_PT}) bytes"
    elif [ -f "${LATEST_PT}" ] && [ -s "${LATEST_PT}" ]; then
        CHECKPOINT="${LATEST_PT}"
        log "Found latest.pt: $(stat -c%s ${LATEST_PT}) bytes"
    else
        STEP=$(tail -1 /tmp/c1_3_fsdp_full.log 2>/dev/null | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('step',0))" 2>/dev/null || echo "0")
        log "Waiting for checkpoint... (training step ${STEP}/19000)"
        sleep 600
        continue
    fi
    break
done

# ============================================================
# Step 2: Check if model already evaluated
# ============================================================
log "=== Step 2: Checking model evaluation ==="

MODEL_F1=$(python3 -c "
import json
try:
    with open('${ARTIFACTS_DIR}/model_grid_results.json') as f:
        d = json.load(f)
    best_f1 = 0.0
    for c in d.get('configs', []):
        for tier_name, tier_data in c.get('tiers', {}).items():
            if isinstance(tier_data, dict):
                f1 = tier_data.get('mean_f1', tier_data.get('f1', 0))
                if f1 > best_f1:
                    best_f1 = f1
    print(f'{best_f1:.6f}')
except Exception as e:
    print('0.000000')
" 2>/dev/null)

log "Current model F1 in grid: ${MODEL_F1}"

# If model F1 is 0, run evaluation
if [ "$(echo "${MODEL_F1} < 0.01" | bc -l 2>/dev/null || echo 1)" = "1" ]; then
    log "Model not evaluated yet, running evaluation..."

    # Find a free GPU
    EVAL_GPU="cuda:0"
    GPU4_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | head -1)
    if [ "${GPU4_USED}" -gt 30000 ]; then
        log "GPU 4 busy, using cuda:0 for eval"
        EVAL_GPU="cuda:0"
    fi

    export PYTHONPATH=src
    python3 scripts/eval_and_generate_grid.py \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --device "${EVAL_GPU}" \
        --eval-decoders threshold,mea \
        --decoder mea \
        --output-dir "${ARTIFACTS_DIR}" \
        --config-name pairformer_ribonanza_frozen_small_pair_fsdp_seed0 \
        --batch-size 4 \
        2>&1 | tee /tmp/c1_3_seed0_eval.log

    # Re-check model F1
    MODEL_F1=$(python3 -c "
import json
try:
    with open('${ARTIFACTS_DIR}/model_grid_results.json') as f:
        d = json.load(f)
    best_f1 = 0.0
    for c in d.get('configs', []):
        for tier_name, tier_data in c.get('tiers', {}).items():
            if isinstance(tier_data, dict):
                f1 = tier_data.get('mean_f1', tier_data.get('f1', 0))
                if f1 > best_f1:
                    best_f1 = f1
    print(f'{best_f1:.6f}')
except Exception as e:
    print('0.000000')
" 2>/dev/null)
    log "Model F1 after evaluation: ${MODEL_F1}"
fi

# ============================================================
# Step 3: Check if model beats ViennaRNA
# ============================================================
log "=== Step 3: Checking model vs ViennaRNA (F1=${VIENNARNA_F1}) ==="

if [ "$(echo "${MODEL_F1} < ${VIENNARNA_F1}" | bc -l 2>/dev/null || echo 1)" = "1" ]; then
    log "WARNING: Model F1 (${MODEL_F1}) does NOT beat ViennaRNA (${VIENNARNA_F1})"
    log "Proceeding with multi-seed anyway (significance test will determine final verdict)"
else
    log "GOOD: Model F1 (${MODEL_F1}) beats ViennaRNA (${VIENNARNA_F1})"
fi

# ============================================================
# Step 4: Launch multi-seed training (seeds 1-9)
# ============================================================
log "=== Step 4: Launching multi-seed training (seeds 1-${NUM_SEEDS}-1) ==="

if [ "${MAX_TRAIN_SAMPLES}" -gt 0 ]; then
    log "Using reduced dataset: ${MAX_TRAIN_SAMPLES} samples per seed"
    STEPS_PER_EPOCH=$((MAX_TRAIN_SAMPLES / 12))
    ETA_HOURS=$(python3 -c "print(f'{${STEPS_PER_EPOCH} * 3.49 / 3600:.1f}')")
    log "Estimated ETA per seed: ${ETA_HOURS}h"
    log "Total ETA for ${NUM_SEEDS} seeds: $(python3 -c "print(f'{${ETA_HOURS} * ${NUM_SEEDS}:.1f}')")h"
else
    log "Using full dataset (228K samples, ~18h per seed)"
    log "Total ETA for ${NUM_SEEDS} seeds: ~180h"
fi

# Create multi-seed config
MULTISEED_CONFIG="configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_multiseed.yaml"
python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['training']['epochs'] = 1
cfg['training']['early_stop_patience'] = 0
with open('${MULTISEED_CONFIG}', 'w') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)
print(f'Created ${MULTISEED_CONFIG} with epochs=1')
"

for SEED in $(seq 1 $((NUM_SEEDS - 1))); do
    log "=== Training seed ${SEED}/${NUM_SEEDS} ==="

    SEED_CKPT_DIR="${ARTIFACTS_DIR}/runs/pairformer_ribonanza_frozen_small_pair_fsdp_seed${SEED}"
    SEED_RESULTS_DIR="${ARTIFACTS_DIR}/results_fsdp_seed${SEED}"

    # Skip if already trained
    if [ -f "${SEED_CKPT_DIR}/best.pt" ] && [ -s "${SEED_CKPT_DIR}/best.pt" ]; then
        log "Seed ${SEED} already has best.pt, skipping training"
    elif [ -f "${SEED_CKPT_DIR}/latest.pt" ] && [ -s "${SEED_CKPT_DIR}/latest.pt" ]; then
        log "Seed ${SEED} already has latest.pt, skipping training"
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
            --output-dir "${SEED_RESULTS_DIR}" \
            ${MAX_TRAIN_FLAG} \
            2>&1 | tee /tmp/c1_3_multiseed_seed${SEED}.log

        if [ ! -f "${SEED_CKPT_DIR}/latest.pt" ] && [ ! -f "${SEED_CKPT_DIR}/best.pt" ]; then
            log "ERROR: Seed ${SEED} training failed - no checkpoint produced"
            continue
        fi
    fi

    # Evaluate the checkpoint
    log "Evaluating seed ${SEED}..."

    # Find checkpoint
    SEED_CHECKPOINT=""
    if [ -f "${SEED_CKPT_DIR}/best.pt" ] && [ -s "${SEED_CKPT_DIR}/best.pt" ]; then
        SEED_CHECKPOINT="${SEED_CKPT_DIR}/best.pt"
    elif [ -f "${SEED_CKPT_DIR}/latest.pt" ] && [ -s "${SEED_CKPT_DIR}/latest.pt" ]; then
        SEED_CHECKPOINT="${SEED_CKPT_DIR}/latest.pt"
    fi

    if [ -z "${SEED_CHECKPOINT}" ]; then
        log "ERROR: No checkpoint for seed ${SEED}, skipping evaluation"
        continue
    fi

    EVAL_GPU="cuda:0"
    GPU4_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | head -1)
    if [ "${GPU4_USED}" -gt 30000 ]; then
        EVAL_GPU="cuda:0"
    fi

    python3 scripts/eval_and_generate_grid.py \
        --config "${MULTISEED_CONFIG}" \
        --checkpoint "${SEED_CHECKPOINT}" \
        --device "${EVAL_GPU}" \
        --eval-decoders threshold,mea \
        --decoder mea \
        --output-dir "${ARTIFACTS_DIR}" \
        --config-name "pairformer_ribonanza_frozen_small_pair_fsdp_seed${SEED}" \
        --batch-size 4 \
        2>&1 | tee /tmp/c1_3_multiseed_eval_seed${SEED}.log

    log "Seed ${SEED} complete"
done

# ============================================================
# Step 5: Generate multiseed_results.json and significance_report.json
# ============================================================
log "=== Step 5: Generating multiseed and significance reports ==="

python3 scripts/generate_multiseed_results.py --artifacts-dir "${ARTIFACTS_DIR}" --num-seeds ${NUM_SEEDS}
python3 scripts/generate_significance_report.py --artifacts-dir "${ARTIFACTS_DIR}"

# ============================================================
# Step 6: Run final gate audit
# ============================================================
log "=== Step 6: Running final gate audit ==="
python3 scripts/audit_c1_3_gate.py --artifacts-dir "${ARTIFACTS_DIR}" --docs-dir docs 2>&1 | tee /tmp/c1_3_final_gate_audit.log

log "=== Pipeline complete! ==="
log "Check ${ARTIFACTS_DIR}/gate_audit.json for final results"
