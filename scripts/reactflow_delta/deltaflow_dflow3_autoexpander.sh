#!/usr/bin/env bash
# DeltaFlow DFLOW3 auto-expander: add grid workers when GPUs free up.
#
# Launches one additional controller instance per 30-minute cycle whenever
# a big GPU (0-5, never the 4.8GB MIG slices 6/7) reports >= 20 GiB free
# (room for one ~8.4GB flow worker besides existing load).  Each instance
# is a PID-tagged copy of the committed controller (running bash scripts
# must never be edited in place), so worker logs never collide and the
# fold-claim locks keep instances duplicate-free.  Exits when the grid
# (seeds 1-4 x 20 folds) is complete.

set -uo pipefail

REPO_ROOT="/home/cunyuliu/reactflow_delta_worktrees/deltaflow_impl_20260908"
CONTROLLER="$REPO_ROOT/scripts/reactflow_delta/run_deltaflow_grid_controller.sh"
OUT_DIR="/mnt/cunyuliu/reactflow_delta_deltaflow_dflow3_formal"
PYTHON="/home/cunyuliu/miniconda3/envs/editflow/bin/python"
EXPANDER_LOG="/home/cunyuliu/deltaflow_dflow3_expander.log"
THRESH_GB=11
CYCLE_SECONDS=1800

grid_complete() {
    local count=0 seed
    for seed in 1 2 3 4; do
        count=$((count + $(ls "$OUT_DIR" 2>/dev/null | grep -c "fold_result.*_seed${seed}\.json")))
    done
    [ "$count" -ge 80 ]
}

echo "$(date -u '+%F %T UTC') expander started (pid $$, threshold ${THRESH_GB}GB)" >> "$EXPANDER_LOG"
while ! grid_complete; do
    launched=0
    for gpu in 0 1 2 3 4 5; do
        free_gb=$(
            CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -c \
                "import torch; f,_=torch.cuda.mem_get_info(); print(f'{f/2**30:.1f}')" \
                2>/dev/null || echo "-1"
        )
        if awk "BEGIN {exit !($free_gb >= $THRESH_GB)}" 2>/dev/null; then
            tag="exp$$"
            instance="/tmp/deltaflow_dflow3_${tag}_gpu${gpu}.sh"
            sed "s|_worker\${worker_id}|_${tag}_worker\${worker_id}|g" \
                "$CONTROLLER" > "$instance"
            nohup bash "$instance" \
                --phase DFLOW3 --seeds 1,2,3,4 --out-dir "$OUT_DIR" \
                --gpus "$gpu" --point-epochs 40 --calibration-epochs 40 \
                --flow-epochs 100 \
                --experiment-id DFLOW3_FIXED_SEEDS_FORMAL_PREDICTION_ONLY \
                >> "$EXPANDER_LOG" 2>&1 &
            echo "$(date -u '+%F %T UTC') launched instance on GPU $gpu (free=${free_gb}GB)" >> "$EXPANDER_LOG"
            launched=1
            break
        fi
    done
    [ "$launched" -eq 0 ] && echo "$(date -u '+%F %T UTC') cycle: no eligible GPU (waiting)" >> "$EXPANDER_LOG"
    sleep "$CYCLE_SECONDS"
done
echo "$(date -u '+%F %T UTC') grid complete; expander exits" >> "$EXPANDER_LOG"
