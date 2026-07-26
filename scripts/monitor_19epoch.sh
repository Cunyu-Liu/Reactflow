#!/bin/bash
# Monitor 19-epoch FSDP training and evaluate when complete.
# This script:
# 1. Monitors training progress (checks log for step/epoch)
# 2. When training process exits, evaluates the best checkpoint
# 3. Runs gate audit

set -uo pipefail

PROJECT_DIR="/home/cunyuliu/reactflow_c1_3_stage_20260722"
GOLD_DIR="/home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012/splits/rfam_current_mmseqs_seed0"
ARTIFACTS_DIR="${PROJECT_DIR}/artifacts/c1_3"
CONFIG="${PROJECT_DIR}/configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_ddp.yaml"
CKPT_DIR="${ARTIFACTS_DIR}/runs/pairformer_ribonanza_frozen_small_pair_fsdp_seed0"
LOG_FILE="/tmp/c1_3_fsdp_19epoch.log"
TOTAL_EPOCHS=19

cd "${PROJECT_DIR}"
source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
conda activate editflow

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ============================================================
# PHASE 1: Monitor training progress
# ============================================================
log "=== Phase 1: Monitoring 19-epoch training ==="
log "Config: ${CONFIG}"
log "Checkpoint dir: ${CKPT_DIR}"
log "Log file: ${LOG_FILE}"

TRAINING_PID=""
EPOCH0_EVALUATED=false

while true; do
    # Check if training process is still running
    # The training was launched with nohup torchrun
    CURRENT_PID=$(pgrep -f "train_c1_3.py.*pairformer_ribonanza_frozen_small_pair_ddp" | head -1)

    if [ -z "${CURRENT_PID}" ]; then
        log "Training process not found. Checking if it completed..."
        # Check if the last log line indicates completion
        LAST_LOG=$(tail -5 "${LOG_FILE}" 2>/dev/null)
        if echo "${LAST_LOG}" | grep -qE "(Training complete|Epoch 18/19|Saved checkpoint|Error|Traceback)"; then
            log "Training appears to have completed or errored."
            break
        else
            log "Training process gone but no completion marker. Waiting 60s..."
            sleep 60
            CURRENT_PID=$(pgrep -f "train_c1_3.py.*pairformer_ribonanza_frozen_small_pair_ddp" | head -1)
            if [ -z "${CURRENT_PID}" ]; then
                log "Training process definitively gone. Proceeding to evaluation."
                break
            fi
        fi
    fi

    if [ -n "${CURRENT_PID}" ] && [ "${TRAINING_PID}" != "${CURRENT_PID}" ]; then
        TRAINING_PID="${CURRENT_PID}"
        log "Training PID: ${TRAINING_PID}"
    fi

    # Get current step and epoch from log
    STEP_INFO=$(tail -1 "${LOG_FILE}" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print(f\"epoch={d.get('epoch', '?')} step={d.get('step', '?')} loss={d.get('loss', '?'):.4f}\")
except:
    print('parsing...')
" 2>/dev/null || echo "waiting...")

    # Check for epoch 0 checkpoint (early evaluation)
    if [ "${EPOCH0_EVALUATED}" = false ]; then
        BEST_PT="${CKPT_DIR}/best.pt"
        LATEST_PT="${CKPT_DIR}/latest.pt"
        if ([ -f "${BEST_PT}" ] && [ -s "${BEST_PT}" ]) || ([ -f "${LATEST_PT}" ] && [ -s "${LATEST_PT}" ]); then
            log "=== Epoch 0 checkpoint detected! Running early evaluation ==="
            CHECKPOINT="${BEST_PT}"
            [ ! -f "${BEST_PT}" ] && CHECKPOINT="${LATEST_PT}"

            # Check if already evaluated
            ALREADY_EVAL=false
            if [ -f "${ARTIFACTS_DIR}/model_grid_results.json" ]; then
                HAS_CONFIG=$(python3 -c "
import json
with open('${ARTIFACTS_DIR}/model_grid_results.json') as f:
    d = json.load(f)
configs = d.get('configs', [])
print(any(c.get('config') == 'pairformer_ribonanza_frozen_small_pair_fsdp_seed0' and c.get('tiers', {}).get('in_clan', {}).get('mean_f1', 0) > 0.1 for c in configs))
" 2>/dev/null || echo "False")
                [ "${HAS_CONFIG}" = "True" ] && ALREADY_EVAL=true
            fi

            if [ "${ALREADY_EVAL}" = false ]; then
                log "Running early evaluation with checkpoint: ${CHECKPOINT}"
                export PYTHONPATH=src
                EVAL_GPU="cuda:0"
                GPU4_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | head -1)
                [ "${GPU4_USED}" -gt 30000 ] 2>/dev/null && EVAL_GPU="cuda:6"

                python3 scripts/eval_and_generate_grid.py \
                    --config "${CONFIG}" \
                    --checkpoint "${CHECKPOINT}" \
                    --device "${EVAL_GPU}" \
                    --eval-decoders threshold,mea \
                    --decoder mea \
                    --output-dir "${ARTIFACTS_DIR}" \
                    --config-name pairformer_ribonanza_frozen_small_pair_fsdp_seed0 \
                    --batch-size 4 \
                    2>&1 | tee /tmp/c1_3_early_eval.log

                if [ ${PIPESTATUS[0]} -eq 0 ]; then
                    log "Early evaluation complete!"
                    # Extract F1 score
                    F1=$(python3 -c "
import json
with open('${ARTIFACTS_DIR}/model_grid_results.json') as f:
    d = json.load(f)
for c in d.get('configs', []):
    if c.get('config') == 'pairformer_ribonanza_frozen_small_pair_fsdp_seed0':
        f1 = c.get('tiers', {}).get('in_clan', {}).get('mean_f1', 0)
        print(f'{f1:.4f}')
        break
" 2>/dev/null || echo "unknown")
                    log "Early F1 (epoch 0): ${F1} (ViennaRNA baseline: 0.6819)"
                    EPOCH0_EVALUATED=true
                else
                    log "Early evaluation FAILED! Will retry after training completes."
                    EPOCH0_EVALUATED=true  # Don't keep retrying
                fi
            else
                log "Model already evaluated, skipping early evaluation"
                EPOCH0_EVALUATED=true
            fi
        fi
    fi

    log "Training: ${STEP_INFO} (PID: ${TRAINING_PID})"
    log "Sleeping 600s..."
    sleep 600
done

# ============================================================
# PHASE 2: Final evaluation with best checkpoint
# ============================================================
log "=== Phase 2: Final evaluation after training completion ==="

BEST_PT="${CKPT_DIR}/best.pt"
LATEST_PT="${CKPT_DIR}/latest.pt"
CHECKPOINT=""

if [ -f "${BEST_PT}" ] && [ -s "${BEST_PT}" ]; then
    SIZE=$(stat -c%s "${BEST_PT}")
    log "Best checkpoint: ${BEST_PT} (${SIZE} bytes)"
    CHECKPOINT="${BEST_PT}"
elif [ -f "${LATEST_PT}" ] && [ -s "${LATEST_PT}" ]; then
    SIZE=$(stat -c%s "${LATEST_PT}")
    log "Latest checkpoint: ${LATEST_PT} (${SIZE} bytes)"
    CHECKPOINT="${LATEST_PT}"
else
    log "ERROR: No checkpoint found! Training may have failed."
    log "Last 20 lines of log:"
    tail -20 "${LOG_FILE}"
    exit 1
fi

# Remove old evaluation to force re-evaluation with final checkpoint
log "Removing old model_grid_results.json to force re-evaluation..."
python3 -c "
import json, sys
path = '${ARTIFACTS_DIR}/model_grid_results.json'
try:
    with open(path) as f:
        d = json.load(f)
    # Remove old fsdp_seed0 entries
    d['configs'] = [c for c in d.get('configs', []) if c.get('config') != 'pairformer_ribonanza_frozen_small_pair_fsdp_seed0']
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
    print(f'Removed old entries, {len(d[\"configs\"])} configs remaining')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
" 2>&1

log "Running final evaluation with best checkpoint..."
export PYTHONPATH=src
EVAL_GPU="cuda:0"
GPU4_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | head -1)
[ "${GPU4_USED}" -gt 30000 ] 2>/dev/null && EVAL_GPU="cuda:6"

python3 scripts/eval_and_generate_grid.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --device "${EVAL_GPU}" \
    --eval-decoders threshold,mea \
    --decoder mea \
    --output-dir "${ARTIFACTS_DIR}" \
    --config-name pairformer_ribonanza_frozen_small_pair_fsdp_seed0 \
    --batch-size 4 \
    2>&1 | tee /tmp/c1_3_final_eval.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "ERROR: Final evaluation failed!"
    exit 1
fi

# Extract final F1
F1=$(python3 -c "
import json
with open('${ARTIFACTS_DIR}/model_grid_results.json') as f:
    d = json.load(f)
for c in d.get('configs', []):
    if c.get('config') == 'pairformer_ribonanza_frozen_small_pair_fsdp_seed0':
        f1 = c.get('tiers', {}).get('in_clan', {}).get('mean_f1', 0)
        novel_f1 = c.get('tiers', {}).get('novel_clan', {}).get('mean_f1', 0)
        print(f'in_clan={f1:.4f} novel_clan={novel_f1:.4f}')
        break
" 2>/dev/null || echo "unknown")

log "Final F1: ${F1}"
log "ViennaRNA baseline: in_clan=0.6819 novel_clan=0.6822"
log "EternaFold baseline: in_clan=0.7039 novel_clan=0.7036"

# Generate placeholder multiseed_results.json (single seed)
python3 << 'PYEOF'
import json
from pathlib import Path
artifacts = Path("artifacts/c1_3")
multiseed = {
    "schema_version": 1,
    "seeds": [0],
    "configs": [
        {
            "config": "pairformer_ribonanza_frozen_small_pair_fsdp",
            "num_seeds": 1,
            "note": "Single seed result (19 epochs). Multi-seed training (10 seeds) pending for gate compliance.",
        }
    ],
    "note": "Multi-seed training not yet started. Gate audit will show INCOMPLETE for multi-seed check."
}
with open(artifacts / "multiseed_results.json", "w") as f:
    json.dump(multiseed, f, indent=2)
print(f"[INFO] Written multiseed_results.json (placeholder, single seed)")
PYEOF

# ============================================================
# PHASE 3: Merge baselines and run gate audit
# ============================================================
log "=== Phase 3: Merging baselines and running gate audit ==="

python3 scripts/merge_baseline_results.py "${ARTIFACTS_DIR}" 2>/dev/null

log "Running gate audit..."
python3 scripts/audit_c1_3_gate.py --artifacts-dir "${ARTIFACTS_DIR}" --docs-dir docs 2>&1 | tee /tmp/c1_3_gate_audit.log

log "=== Monitoring complete ==="
log "Check ${ARTIFACTS_DIR}/gate_audit.json for results"
log ""
log "If F1 > ViennaRNA (0.6819), proceed to multi-seed training:"
log "  bash scripts/launch_multiseed.sh 10 1 50000"
log ""
log "If F1 < ViennaRNA, consider:"
log "  1. Training more epochs (resume from checkpoint)"
log "  2. Adjusting learning rate or other hyperparameters"
