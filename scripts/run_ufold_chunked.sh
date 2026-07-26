#!/bin/bash
# Chunked UFold runner: processes sequences in chunks to avoid GPU memory leaks.
# Each chunk runs in a fresh Python process, clearing all GPU memory.
#
# Usage:
#   bash scripts/run_ufold_chunked.sh <gold_dir> <output_dir> <gpu> [chunk_size] [tiers...]
#
# Example:
#   bash scripts/run_ufold_chunked.sh /path/to/splits artifacts/c1_3/baselines/ufold_same_split 6 300 test novel

set -euo pipefail

GOLD_DIR="$1"
OUTPUT_DIR="$2"
GPU="${3:-6}"
CHUNK_SIZE="${4:-300}"
shift 4 2>/dev/null || shift $#
TIERS=("$@")

if [ ${#TIERS[@]} -eq 0 ]; then
    TIERS=("test" "novel")
fi

echo "[chunked] gold_dir=$GOLD_DIR"
echo "[chunked] output_dir=$OUTPUT_DIR"
echo "[chunked] gpu=$GPU chunk_size=$CHUNK_SIZE tiers=${TIERS[*]}"

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
    OUTPUT_FILE="$OUTPUT_DIR/${TIER_NAME}.ufold.predictions.jsonl"
    GOLD_FILE="$GOLD_DIR/${tier}.jsonl"

    if [ ! -f "$GOLD_FILE" ]; then
        echo "[chunked] ERROR: Gold file not found: $GOLD_FILE"
        continue
    fi

    TOTAL=$(wc -l < "$GOLD_FILE")

    while true; do
        if [ -f "$OUTPUT_FILE" ]; then
            DONE=$(wc -l < "$OUTPUT_FILE")
        else
            DONE=0
        fi

        if [ "$DONE" -ge "$TOTAL" ]; then
            echo "[chunked] Tier $tier ($TIER_NAME) complete: $DONE/$TOTAL"
            break
        fi

        REMAINING=$((TOTAL - DONE))
        THIS_CHUNK=$CHUNK_SIZE
        if [ "$REMAINING" -lt "$THIS_CHUNK" ]; then
            THIS_CHUNK=$REMAINING
        fi

        echo "[chunked] tier=$tier start=$DONE chunk=$THIS_CHUNK remaining=$REMAINING total=$TOTAL"

        # Run UFold for this chunk in a fresh process
        cd /home/cunyuliu/reactflow_c1_3_stage_20260722
        source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
        conda activate editflow

        CUDA_VISIBLE_DEVICES=$GPU \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python3 scripts/run_ufold_baseline.py \
            --gold-dir "$GOLD_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --tiers "$tier" \
            --start-index "$DONE" \
            --max-samples "$THIS_CHUNK" \
            --append \
            --device cuda \
            2>&1 | tail -5 || true

        # Verify progress was made
        NEW_DONE=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
        if [ "$NEW_DONE" -le "$DONE" ]; then
            echo "[chunked] WARNING: No progress ($DONE -> $NEW_DONE). Retrying after 5s..."
            sleep 5
        fi
    done
done

echo "[chunked] All tiers complete."
