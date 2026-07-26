#!/bin/bash
# Evaluate all completed external baselines against ReactFlow gold splits.
#
# Usage: bash scripts/run_all_baseline_evaluations.sh [baselines...]
# Default: evaluate all baselines that have prediction files.
#
# Produces baseline_*_results.json files in artifacts/c1_3/baselines/

set -euo pipefail

GOLD_DIR="/home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012/splits/rfam_current_mmseqs_seed0"
BASELINES_DIR="/home/cunyuliu/reactflow_c1_3_stage_20260722/artifacts/c1_3/baselines"
RESULTS_DIR="${BASELINES_DIR}/evaluation_results"
mkdir -p "${RESULTS_DIR}"

# Default baselines (skip if prediction files don't exist)
DEFAULT_BASELINES="viennarna eternafold mxfold2 ufold efold rnaformer"

# Override with command line args if provided
BASELINES="${*:-${DEFAULT_BASELINES}}"

echo "[eval] Gold dir: ${GOLD_DIR}"
echo "[eval] Baselines dir: ${BASELINES_DIR}"
echo "[eval] Results dir: ${RESULTS_DIR}"
echo "[eval] Baselines: ${BASELINES}"
echo ""

for MODEL in ${BASELINES}; do
    MODEL_DIR="${BASELINES_DIR}/${MODEL}_same_split"
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "[eval] SKIP ${MODEL}: directory not found"
        continue
    fi

    # Find prediction files for in_clan and novel_clan
    IN_CLAN_PRED="${MODEL_DIR}/in_clan.${MODEL}.predictions.jsonl"
    NOVEL_CLAN_PRED="${MODEL_DIR}/novel_clan.${MODEL}.predictions.jsonl"

    # Handle RNAformer novel tier from separate GPU7 run
    if [ "${MODEL}" = "rnaformer" ] && [ ! -f "${NOVEL_CLAN_PRED}" ]; then
        NOVEL_CLAN_PRED_GPU7="/tmp/rnaformer_novel_only/novel_clan.${MODEL}.predictions.jsonl"
        if [ -f "${NOVEL_CLAN_PRED_GPU7}" ]; then
            echo "[eval] Using RNAformer novel tier from GPU7 run"
            NOVEL_CLAN_PRED="${NOVEL_CLAN_PRED_GPU7}"
        fi
    fi

    # Build --gold-json and --prediction-json args
    GOLD_ARGS=""
    PRED_ARGS=""

    if [ -f "${IN_CLAN_PRED}" ]; then
        IN_CLAN_LINES=$(wc -l < "${IN_CLAN_PRED}")
        echo "[eval] ${MODEL}: in_clan=${IN_CLAN_LINES} predictions"
        if [ "${IN_CLAN_LINES}" -gt 0 ]; then
            GOLD_ARGS="${GOLD_ARGS} --gold-json in_clan=${GOLD_DIR}/test.jsonl"
            PRED_ARGS="${PRED_ARGS} --prediction-json in_clan=${IN_CLAN_PRED}"
        fi
    else
        echo "[eval] ${MODEL}: in_clan predictions NOT FOUND"
    fi

    if [ -f "${NOVEL_CLAN_PRED}" ]; then
        NOVEL_CLAN_LINES=$(wc -l < "${NOVEL_CLAN_PRED}")
        echo "[eval] ${MODEL}: novel_clan=${NOVEL_CLAN_LINES} predictions"
        if [ "${NOVEL_CLAN_LINES}" -gt 0 ]; then
            GOLD_ARGS="${GOLD_ARGS} --gold-json novel_clan=${GOLD_DIR}/novel.jsonl"
            PRED_ARGS="${PRED_ARGS} --prediction-json novel_clan=${NOVEL_CLAN_PRED}"
        fi
    else
        echo "[eval] ${MODEL}: novel_clan predictions NOT FOUND"
    fi

    if [ -z "${GOLD_ARGS}" ]; then
        echo "[eval] SKIP ${MODEL}: no prediction files found"
        echo ""
        continue
    fi

    OUTPUT="${RESULTS_DIR}/baseline_${MODEL}_results.json"
    echo "[eval] Running evaluation for ${MODEL}..."
    cd /home/cunyuliu/reactflow_c1_3_stage_20260722
    source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
    conda activate editflow

    python3 scripts/evaluate_external_baseline_predictions.py \
        --model "${MODEL}" \
        ${GOLD_ARGS} \
        ${PRED_ARGS} \
        --output "${OUTPUT}" \
        --protocol same_split_local \
        --seed-count single_seed \
        2>&1 | tail -20

    if [ -f "${OUTPUT}" ]; then
        echo "[eval] ${MODEL}: results saved to ${OUTPUT}"
        # Print summary
        python3 -c "
import json
with open('${OUTPUT}') as f:
    data = json.load(f)
for tier_name, tier_data in data.get('tiers', {}).items():
    metrics = tier_data.get('metrics', {})
    print(f'  {tier_name}: F1={metrics.get(\"macro_f1\", \"N/A\"):.4f} MCC={metrics.get(\"macro_mcc\", \"N/A\"):.4f} N={tier_data.get(\"count\", 0)}')
" 2>/dev/null || echo "  (summary parse failed)"
    else
        echo "[eval] ${MODEL}: FAILED - no output file"
    fi
    echo ""
done

echo "[eval] All evaluations complete."
echo "[eval] Results in: ${RESULTS_DIR}"
ls -la "${RESULTS_DIR}" 2>/dev/null
