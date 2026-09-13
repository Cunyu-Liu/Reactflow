#!/usr/bin/env bash
# DeltaFlow multi-seed grid controller (Task 6.6).
#
# Schedules the seeds x folds DFLOW3 grid across GPUs with dynamic
# sharding: each worker claims the next missing fold of its seed from
# the artifact directory, so folds already published are never rerun
# (fill-missing-only), and a worker that finishes its shard immediately
# steals work from slower shards.
#
# Discipline (frozen):
#   - CUDA fail-fast: workers require CUDA before any artifact;
#   - MIG/small GPUs (<10 GiB) are excluded via mem_get_info probing;
#   - artifacts under /mnt/cunyuliu/<experiment dir>;
#   - logs under /home/cunyuliu/<experiment>_worker*.log;
#   - the runner itself refuses overwrite; the controller never deletes.
#
# Usage:
#   bash run_deltaflow_grid_controller.sh \
#       --phase DFLOW3 --seeds 1,2,3,4 --out-dir /mnt/.../dflow3 \
#       --gpus 0,1,3,4,5 [--workers-per-gpu 1] [--point-epochs 40 ...]
#
# The controller exits when every (seed, fold) pair in the grid has a
# published fold_result json.  It performs no metric reads.

set -euo pipefail

REPO_ROOT="/home/cunyuliu/reactflow_delta_worktrees/deltaflow_impl_20260908"
PYTHON="/home/cunyuliu/miniconda3/envs/editflow/bin/python"
COMMON_ARGS=(
    --repo-root "$REPO_ROOT"
    --m2-csv /mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
    --v8-dir /mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
    --v10-dir /mnt/cunyuliu/reactflow_delta_model_rescue_v10/v10m2_screen_seed0
    --tic2a-merged-json /mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
    --unconstrained-cache /mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
    --constrained-cache /mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
    --teacher-npz /home/cunyuliu/deltaflow_m2_teacher/m2_wt_teacher_single.npz
    --device cuda:0
)

PHASE="DFLOW3"
SEEDS="1,2,3,4"
OUT_DIR=""
GPUS="0,1,3,4,5"
POINT_EPOCHS=40
CALIBRATION_EPOCHS=40
FLOW_EPOCHS=100
EXPERIMENT_ID="DFLOW3_FIXED_SEEDS_FORMAL_PREDICTION_ONLY"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --point-epochs) POINT_EPOCHS="$2"; shift 2 ;;
        --calibration-epochs) CALIBRATION_EPOCHS="$2"; shift 2 ;;
        --flow-epochs) FLOW_EPOCHS="$2"; shift 2 ;;
        --experiment-id) EXPERIMENT_ID="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    echo "--out-dir is required" >&2
    exit 2
fi
if [[ ! "$OUT_DIR" == /mnt/cunyuliu/* ]]; then
    echo "artifacts must live under /mnt/cunyuliu (got $OUT_DIR)" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

# Probe each requested GPU: usable only if CUDA reports >= 10 GiB free.
USABLE_GPUS=()
for gpu in "${GPU_LIST[@]}"; do
    free_gb=$(
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" - <<'PY' 2>/dev/null || echo "-1"
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
free, _total = torch.cuda.mem_get_info()
print(f"{free / 2**30:.1f}")
PY
    ) || true
    if awk "BEGIN {exit !($free_gb >= 10.0)}" 2>/dev/null; then
        USABLE_GPUS+=("$gpu")
    else
        echo "[controller] GPU $gpu skipped (free=${free_gb}GiB < 10 or CUDA unavailable)"
    fi
done
if [[ ${#USABLE_GPUS[@]} -eq 0 ]]; then
    echo "[controller] no usable GPU with >=10GiB free; aborting before any artifact" >&2
    exit 3
fi
echo "[controller] usable GPUs: ${USABLE_GPUS[*]}"

grid_complete() {
    for seed in "${SEED_LIST[@]}"; do
        for fold in $(seq 0 19); do
            [[ -f "$OUT_DIR/deltaflow_fold_result_fold${fold}_seed${seed}.json" ]] || return 1
        done
    done
    return 0
}

worker() {
    local gpu="$1"
    local worker_id="$2"
    local log="/home/cunyuliu/deltaflow_${PHASE,,}_worker${worker_id}_gpu${gpu}.log"
    while ! grid_complete; do
        local claimed=0
        for seed in "${SEED_LIST[@]}"; do
            for fold in $(seq 0 19); do
                local marker="$OUT_DIR/deltaflow_fold_result_fold${fold}_seed${seed}.json"
                local lock="$OUT_DIR/.claim_fold${fold}_seed${seed}"
                if [[ -f "$marker" || -f "$lock" ]]; then
                    continue
                fi
                if ( set -o noclobber; echo "$worker_id" > "$lock" ) 2>/dev/null; then
                    claimed=1
                    echo "[worker${worker_id}] gpu=${gpu} seed=${seed} fold=${fold} start" >> "$log"
                    if CUDA_VISIBLE_DEVICES="$gpu" \
                       PYTHONPATH="$REPO_ROOT" \
                       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
                       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                       "$PYTHON" "$REPO_ROOT/scripts/reactflow_delta/run_deltaflow_fold.py" \
                            "${COMMON_ARGS[@]}" \
                            --phase "$PHASE" \
                            --experiment-id "$EXPERIMENT_ID" \
                            --out-dir "$OUT_DIR" \
                            --folds "$fold" \
                            --seed "$seed" \
                            --point-epochs "$POINT_EPOCHS" \
                            --calibration-epochs "$CALIBRATION_EPOCHS" \
                            --flow-epochs "$FLOW_EPOCHS" \
                            >> "$log" 2>&1; then
                        echo "[worker${worker_id}] gpu=${gpu} seed=${seed} fold=${fold} complete" >> "$log"
                    else
                        echo "[worker${worker_id}] gpu=${gpu} seed=${seed} fold=${fold} FAILED (see log)" >> "$log"
                    fi
                    rm -f "$lock"
                fi
            done
            [[ $claimed -eq 1 ]] && break
        done
        if [[ $claimed -eq 0 ]]; then
            sleep 120
        fi
    done
    echo "[worker${worker_id}] grid complete" >> "$log"
}

worker_id=0
for gpu in "${USABLE_GPUS[@]}"; do
    worker "$gpu" "$worker_id" &
    worker_id=$((worker_id + 1))
done
wait
grid_complete && echo "[controller] ALL ${#SEED_LIST[@]} seeds x 20 folds present under $OUT_DIR"
