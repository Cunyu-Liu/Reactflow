#!/bin/bash
# Parallel eFold runner: splits sequences into N chunks and runs eFold in parallel.
# Each chunk runs in a separate process with its own model instance.
#
# Usage:
#   bash scripts/run_efold_parallel.sh <gold_dir> <output_dir> <device> <n_workers> [tiers...]
#
# Example:
#   bash scripts/run_efold_parallel.sh /path/to/splits artifacts/c1_3/baselines/efold_same_split cuda 8 test novel

set -euo pipefail

GOLD_DIR="$1"
OUTPUT_DIR="$2"
DEVICE="${3:-cuda}"
N_WORKERS="${4:-8}"
GPU_DEVICE="${5:-}"
shift 5 2>/dev/null || shift $#
TIERS=("$@")

if [ ${#TIERS[@]} -eq 0 ]; then
    TIERS=("test" "novel")
fi

echo "[parallel] gold_dir=$GOLD_DIR output_dir=$OUTPUT_DIR device=$DEVICE workers=$N_WORKERS gpu_device=$GPU_DEVICE tiers=${TIERS[*]}"

mkdir -p "$OUTPUT_DIR"

# TIER_MAP: test -> in_clan, novel -> novel_clan
tier_to_name() {
    case "$1" in
        test) echo "in_clan" ;;
        novel) echo "novel_clan" ;;
        *) echo "$1" ;;
    esac
}

for tier in "${TIERS[@]}"; do
    TIER_NAME=$(tier_to_name "$tier")
    GOLD_FILE="$GOLD_DIR/${tier}.jsonl"
    FINAL_OUTPUT="$OUTPUT_DIR/${TIER_NAME}.efold.predictions.jsonl"

    if [ ! -f "$GOLD_FILE" ]; then
        echo "[parallel] ERROR: Gold file not found: $GOLD_FILE"
        continue
    fi

    TOTAL=$(wc -l < "$GOLD_FILE")
    echo "[parallel] tier=$tier ($TIER_NAME) total=$TOTAL workers=$N_WORKERS"

    # Split gold file into N_WORKERS parts
    SPLIT_DIR="$OUTPUT_DIR/splits_${TIER_NAME}"
    mkdir -p "$SPLIT_DIR"

    # Use Python to split the JSONL file evenly
    cd /home/cunyuliu/reactflow_c1_3_stage_20260722
    source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
    conda activate editflow

    python3 -c "
import json, os, math
gold_file = '$GOLD_FILE'
n_workers = $N_WORKERS
split_dir = '$SPLIT_DIR'
records = []
with open(gold_file) as f:
    for line in f:
        if line.strip():
            records.append(line)
chunk_size = math.ceil(len(records) / n_workers)
for i in range(n_workers):
    start = i * chunk_size
    end = min(start + chunk_size, len(records))
    part_file = os.path.join(split_dir, f'part_{i:03d}.jsonl')
    with open(part_file, 'w') as f:
        f.writelines(records[start:end])
    print(f'  Part {i}: {end - start} sequences ({start}:{end})')
print(f'Total: {len(records)} sequences in {n_workers} parts')
"

    # Launch N_WORKERS parallel eFold processes
    # GPU_DEVICE is a comma-separated list of MIG UUIDs or GPU IDs to cycle through
    if [ -n "$GPU_DEVICE" ]; then
        GPU_DEVICES=(${GPU_DEVICE//,/ })
        NUM_GPUS=${#GPU_DEVICES[@]}
    else
        GPU_DEVICES=()
        NUM_GPUS=0
    fi

    WORKER_PIDS=()
    for i in $(seq 0 $((N_WORKERS - 1))); do
        PART_FILE="$SPLIT_DIR/part_$(printf '%03d' $i).jsonl"
        WORKER_OUT="$SPLIT_DIR/worker_${i}_output"
        mkdir -p "$WORKER_OUT"

        # Cycle through GPUs
        if [ "$NUM_GPUS" -gt 0 ]; then
            WORKER_GPU="${GPU_DEVICES[$((i % NUM_GPUS))]}"
            WORKER_DEVICE="cuda"
            CUDA_VAR="CUDA_VISIBLE_DEVICES=$WORKER_GPU"
        else
            WORKER_GPU="none"
            WORKER_DEVICE="cpu"
            CUDA_VAR="CUDA_VISIBLE_DEVICES="
        fi

        echo "[parallel] Launching worker $i for $TIER_NAME (GPU=$WORKER_GPU, device=$WORKER_DEVICE)..."
        env $CUDA_VAR \
        PYTHONPATH=/home/cunyuliu/reactflow_external_envs/efold_py310/lib/python3.10/site-packages:${PYTHONPATH:-} \
        nohup python3 scripts/run_efold_same_split_baseline.py \
            --gold-json "${TIER_NAME}=$PART_FILE" \
            --output-dir "$WORKER_OUT" \
            --results-json "$WORKER_OUT/efold_results.json" \
            --backend module \
            --device "$WORKER_DEVICE" \
            --progress-every 200 \
            > "$SPLIT_DIR/worker_${i}.log" 2>&1 &
        WORKER_PIDS+=($!)
    done

    echo "[parallel] Waiting for $N_WORKERS workers (PIDs: ${WORKER_PIDS[*]})..."
    FAILED=0
    for pid in "${WORKER_PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "[parallel] WARNING: Worker PID $pid failed"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ $FAILED -gt 0 ]; then
        echo "[parallel] $FAILED workers failed for $TIER_NAME"
    fi

    # Merge all worker outputs into final file
    echo "[parallel] Merging outputs for $TIER_NAME..."
    > "$FINAL_OUTPUT"
    for i in $(seq 0 $((N_WORKERS - 1))); do
        WORKER_OUT="$SPLIT_DIR/worker_${i}_output/${TIER_NAME}.efold.predictions.jsonl"
        if [ -f "$WORKER_OUT" ]; then
            cat "$WORKER_OUT" >> "$FINAL_OUTPUT"
            COUNT=$(wc -l < "$WORKER_OUT")
            echo "  Worker $i: $COUNT sequences"
        else
            echo "  Worker $i: NO OUTPUT (check $SPLIT_DIR/worker_${i}.log)"
        fi
    done

    FINAL_COUNT=$(wc -l < "$FINAL_OUTPUT")
    echo "[parallel] $TIER_NAME complete: $FINAL_COUNT / $TOTAL sequences"

    # Clean up worker outputs (keep logs for debugging)
    for i in $(seq 0 $((N_WORKERS - 1))); do
        rm -rf "$SPLIT_DIR/worker_${i}_output"
    done
done

echo "[parallel] All tiers complete."
