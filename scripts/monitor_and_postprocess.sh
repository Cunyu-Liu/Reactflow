#!/bin/bash
# Monitor all C1-3 tasks and run post-processing when complete.
# This script checks:
# 1. eFold (in_clan + novel_clan) completion
# 2. RNAformer (test + novel) completion
# 3. Training epoch 0 completion (checkpoint save)
#
# When each task completes, it evaluates and merges results.
# When ALL tasks are complete, it runs the gate audit.

set -uo pipefail

PROJECT_DIR="/home/cunyuliu/reactflow_c1_3_stage_20260722"
GOLD_DIR="/home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012/splits/rfam_current_mmseqs_seed0"
ARTIFACTS_DIR="${PROJECT_DIR}/artifacts/c1_3"
BASELINES_DIR="${ARTIFACTS_DIR}/baselines"
RESULTS_DIR="${BASELINES_DIR}/evaluation_results"

cd "${PROJECT_DIR}"
source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
conda activate editflow

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ============================================================
# EFOLD: Merge worker predictions and evaluate
# ============================================================
process_efold() {
    log "=== Processing eFold ==="

    # Check if already evaluated
    if [ -f "${RESULTS_DIR}/baseline_efold_results.json" ]; then
        log "eFold already evaluated, skipping"
        return 0
    fi

    # Check if eFold in_clan is complete (original workers)
    local EFOLD_INCLAN_ORIG="${BASELINES_DIR}/efold_same_split/splits_in_clan"
    local EFOLD_INCLAN_GPU4="/tmp/efold_test_gpu4"
    local EFOLD_NOVEL_MIG="/tmp/efold_novel_mig"

    # Count predictions from both in_clan worker sets
    local inclan_orig_count=0
    local inclan_gpu4_count=0
    local novel_mig_count=0

    if [ -d "${EFOLD_INCLAN_ORIG}" ]; then
        inclan_orig_count=$(cat "${EFOLD_INCLAN_ORIG}"/worker_*_output/in_clan.efold.predictions.jsonl 2>/dev/null | wc -l)
    fi
    if [ -d "${EFOLD_INCLAN_GPU4}" ]; then
        inclan_gpu4_count=$(cat "${EFOLD_INCLAN_GPU4}"/worker_*_output/in_clan.efold.predictions.jsonl 2>/dev/null | wc -l)
    fi
    if [ -d "${EFOLD_NOVEL_MIG}" ]; then
        novel_mig_count=$(cat "${EFOLD_NOVEL_MIG}"/worker_*/novel_clan.efold.predictions.jsonl 2>/dev/null | wc -l)
    fi

    log "eFold in_clan: original=${inclan_orig_count}, gpu4=${inclan_gpu4_count} (target: 16606)"
    log "eFold novel_clan: mig=${novel_mig_count} (target: 34610)"

    # Use whichever in_clan set has more predictions
    local best_inclan_dir=""
    local best_inclan_count=0
    if [ "${inclan_orig_count}" -gt "${best_inclan_count}" ]; then
        best_inclan_dir="${EFOLD_INCLAN_ORIG}"
        best_inclan_count="${inclan_orig_count}"
    fi
    if [ "${inclan_gpu4_count}" -gt "${best_inclan_count}" ]; then
        best_inclan_dir="${EFOLD_INCLAN_GPU4}"
        best_inclan_count="${inclan_gpu4_count}"
    fi

    # Check completeness
    local inclan_complete=false
    local novel_complete=false

    if [ "${best_inclan_count}" -ge 16606 ]; then
        inclan_complete=true
        log "eFold in_clan: COMPLETE (${best_inclan_count} predictions)"
    fi
    if [ "${novel_mig_count}" -ge 34610 ]; then
        novel_complete=true
        log "eFold novel_clan: COMPLETE (${novel_mig_count} predictions)"
    fi

    # Merge and evaluate if complete
    if [ "${inclan_complete}" = true ] && [ "${novel_complete}" = true ]; then
        if [ ! -f "${RESULTS_DIR}/baseline_efold_results.json" ] || [ "${FORCE_REEVAL}" = true ]; then
            log "Merging eFold predictions..."

            # Merge in_clan (use best set)
            local inclan_worker_dirs=""
            if [ "${best_inclan_dir}" = "${EFOLD_INCLAN_ORIG}" ]; then
                for d in "${EFOLD_INCLAN_ORIG}"/worker_*_output; do
                    inclan_worker_dirs="${inclan_worker_dirs} ${d}"
                done
            else
                for d in "${EFOLD_INCLAN_GPU4}"/worker_*_output; do
                    inclan_worker_dirs="${inclan_worker_dirs} ${d}"
                done
            fi
            python3 scripts/merge_efold_predictions.py \
                --tier in_clan \
                --worker-dirs ${inclan_worker_dirs} \
                --output "${BASELINES_DIR}/efold_same_split/in_clan.efold.predictions.jsonl" 2>&1

            # Merge novel_clan
            local novel_worker_dirs=""
            for d in "${EFOLD_NOVEL_MIG}"/worker_*; do
                novel_worker_dirs="${novel_worker_dirs} ${d}"
            done
            python3 scripts/merge_efold_predictions.py \
                --tier novel_clan \
                --worker-dirs ${novel_worker_dirs} \
                --output "${BASELINES_DIR}/efold_same_split/novel_clan.efold.predictions.jsonl" 2>&1

            # Evaluate
            log "Evaluating eFold..."
            python3 scripts/evaluate_external_baseline_predictions.py \
                --model efold \
                --gold-json "in_clan=${GOLD_DIR}/test.jsonl" \
                --gold-json "novel_clan=${GOLD_DIR}/novel.jsonl" \
                --prediction-json "in_clan=${BASELINES_DIR}/efold_same_split/in_clan.efold.predictions.jsonl" \
                --prediction-json "novel_clan=${BASELINES_DIR}/efold_same_split/novel_clan.efold.predictions.jsonl" \
                --output "${RESULTS_DIR}/baseline_efold_results.json" \
                --protocol same_split_local \
                --seed-count single_seed 2>&1

            log "eFold evaluation complete"
            return 0
        else
            log "eFold already evaluated, skipping"
            return 0
        fi
    fi
    return 1
}

# ============================================================
# RNAFORMER: Check and evaluate
# ============================================================
process_rnaformer() {
    log "=== Processing RNAformer ==="

    # Check if already evaluated
    if [ -f "${RESULTS_DIR}/baseline_rnaformer_results.json" ]; then
        log "RNAformer already evaluated, skipping"
        return 0
    fi

    local RF_TEST_PRED="${BASELINES_DIR}/rnaformer_same_split/in_clan.rnaformer.predictions.jsonl"
    local RF_NOVEL_GPU4="/tmp/rnaformer_novel_gpu4/novel_clan.rnaformer.predictions.jsonl"
    local RF_NOVEL_GPU7="/tmp/rnaformer_novel_only/novel_clan.rnaformer.predictions.jsonl"

    # Check if RNAformer processes are still running
    local rf_running=$(ps aux | grep "run_rnaformer" | grep -v grep | wc -l)

    local test_count=0
    local novel_gpu4_count=0
    local novel_gpu7_count=0
    [ -f "${RF_TEST_PRED}" ] && test_count=$(wc -l < "${RF_TEST_PRED}")
    [ -f "${RF_NOVEL_GPU4}" ] && novel_gpu4_count=$(wc -l < "${RF_NOVEL_GPU4}")
    [ -f "${RF_NOVEL_GPU7}" ] && novel_gpu7_count=$(wc -l < "${RF_NOVEL_GPU7}")

    # Use whichever novel location has more predictions (they're duplicates)
    local novel_count=${novel_gpu4_count}
    local best_novel_pred="${RF_NOVEL_GPU4}"
    if [ "${novel_gpu7_count}" -gt "${novel_gpu4_count}" ]; then
        novel_count=${novel_gpu7_count}
        best_novel_pred="${RF_NOVEL_GPU7}"
    fi

    log "RNAformer test: ${test_count}/16606, novel: ${novel_count}/46147 (gpu4=${novel_gpu4_count}, gpu7=${novel_gpu7_count}), processes: ${rf_running}"

    if [ "${test_count}" -ge 16606 ] && [ "${novel_count}" -ge 46147 ]; then
        if [ ! -f "${RESULTS_DIR}/baseline_rnaformer_results.json" ] || [ "${FORCE_REEVAL}" = true ]; then
            log "Evaluating RNAformer..."

            local novel_pred="${best_novel_pred}"

            python3 scripts/evaluate_external_baseline_predictions.py \
                --model rnaformer \
                --gold-json "in_clan=${GOLD_DIR}/test.jsonl" \
                --gold-json "novel_clan=${GOLD_DIR}/novel.jsonl" \
                --prediction-json "in_clan=${RF_TEST_PRED}" \
                --prediction-json "novel_clan=${novel_pred}" \
                --output "${RESULTS_DIR}/baseline_rnaformer_results.json" \
                --protocol same_split_local \
                --seed-count single_seed 2>&1

            log "RNAformer evaluation complete"
            return 0
        else
            log "RNAformer already evaluated, skipping"
            return 0
        fi
    fi
    return 1
}

# ============================================================
# TRAINING: Check checkpoint and run model evaluation
# ============================================================
check_training() {
    log "=== Checking training ==="

    local CKPT_DIR="${ARTIFACTS_DIR}/runs/pairformer_ribonanza_frozen_small_pair_fsdp_seed0"
    local CONFIG="${PROJECT_DIR}/configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_ddp.yaml"

    # Check for non-empty checkpoint
    local best_pt="${CKPT_DIR}/best.pt"
    local latest_pt="${CKPT_DIR}/latest.pt"
    local checkpoint=""

    if [ -f "${best_pt}" ] && [ -s "${best_pt}" ]; then
        local size=$(stat -c%s "${best_pt}")
        log "Training: best.pt exists (${size} bytes)"
        checkpoint="${best_pt}"
    elif [ -f "${latest_pt}" ] && [ -s "${latest_pt}" ]; then
        local size=$(stat -c%s "${latest_pt}")
        log "Training: latest.pt exists (${size} bytes)"
        checkpoint="${latest_pt}"
    else
        # Check training progress
        local step=$(tail -1 /tmp/c1_3_fsdp_full.log 2>/dev/null | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('step',0))" 2>/dev/null || echo "0")
        log "Training: step ${step}/19000 (epoch 0), no checkpoint yet"
        return 1
    fi

    # Check if model_grid_results.json already has this config
    if [ -f "${ARTIFACTS_DIR}/model_grid_results.json" ]; then
        local has_config=$(python3 -c "
import json
with open('${ARTIFACTS_DIR}/model_grid_results.json') as f:
    d = json.load(f)
configs = d.get('configs', [])
print(any(c.get('config') == 'pairformer_ribonanza_frozen_small_pair_fsdp_seed0' and c.get('tiers', {}).get('in_clan', {}).get('mean_f1', 0) > 0.1 for c in configs))
" 2>/dev/null || echo "False")
        if [ "${has_config}" = "True" ]; then
            log "Model already evaluated, skipping"
            return 0
        fi
    fi

    # Find a free GPU for evaluation
    local eval_gpu="cuda:0"
    local gpu4_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | head -1)
    if [ "${gpu4_used}" -gt 30000 ]; then
        log "GPU 4 busy (${gpu4_used} MiB used), trying GPU 6..."
        eval_gpu="cuda:6"
    fi

    # Run model evaluation
    log "Running model evaluation with checkpoint: ${checkpoint}"
    cd "${PROJECT_DIR}"
    export PYTHONPATH=src
    python3 scripts/eval_and_generate_grid.py \
        --config "${CONFIG}" \
        --checkpoint "${checkpoint}" \
        --device "${eval_gpu}" \
        --eval-decoders threshold,mea \
        --decoder mea \
        --output-dir "${ARTIFACTS_DIR}" \
        --config-name pairformer_ribonanza_frozen_small_pair_fsdp_seed0 \
        --batch-size 4 \
        2>&1 | tee /tmp/c1_3_model_eval.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "Model evaluation complete!"

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
            "note": "Single seed result. Multi-seed training (3+ seeds) pending.",
        }
    ],
    "note": "Multi-seed training not yet started. Gate audit will show INCOMPLETE for multi-seed check."
}
with open(artifacts / "multiseed_results.json", "w") as f:
    json.dump(multiseed, f, indent=2)
print(f"[INFO] Written multiseed_results.json (placeholder, single seed)")
PYEOF
        return 0
    else
        log "Model evaluation FAILED!"
        return 1
    fi
}

# ============================================================
# MAIN LOOP
# ============================================================
log "Starting C1-3 monitoring loop..."

EFOLD_DONE=false
RNAFORMER_DONE=false
TRAINING_DONE=false

while true; do
    # Check eFold
    if [ "${EFOLD_DONE}" = false ]; then
        if process_efold; then
            EFOLD_DONE=true
        fi
    fi

    # Check RNAformer
    if [ "${RNAFORMER_DONE}" = false ]; then
        if process_rnaformer; then
            RNAFORMER_DONE=true
        fi
    fi

    # Check training
    if [ "${TRAINING_DONE}" = false ]; then
        if check_training; then
            TRAINING_DONE=true
        fi
    fi

    # Merge baseline results if any new evaluations
    python3 scripts/merge_baseline_results.py "${ARTIFACTS_DIR}" 2>/dev/null

    # Check if all done
    if [ "${EFOLD_DONE}" = true ] && [ "${RNAFORMER_DONE}" = true ] && [ "${TRAINING_DONE}" = true ]; then
        log "ALL TASKS COMPLETE!"
        log "Running gate audit..."
        python3 scripts/audit_c1_3_gate.py --artifacts-dir "${ARTIFACTS_DIR}" --docs-dir docs 2>&1
        log "Gate audit complete. Check ${ARTIFACTS_DIR}/gate_audit.json"
        break
    fi

    # Status summary
    log "Status: eFold=${EFOLD_DONE}, RNAformer=${RNAFORMER_DONE}, Training=${TRAINING_DONE}"
    log "Sleeping 300s..."
    sleep 300
done
