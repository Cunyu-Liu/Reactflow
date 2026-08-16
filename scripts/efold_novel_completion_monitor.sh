#!/bin/bash
# Monitor eFold novel MIG workers and re-evaluate when all complete.
# This script checks every 5 minutes if workers 6 and 7 have finished,
# and if so, re-merges and re-evaluates eFold novel_clan.

set -uo pipefail

PROJECT_DIR="/home/cunyuliu/reactflow_c1_3_stage_20260722"
GOLD_DIR="/home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012/splits/rfam_current_mmseqs_seed0"
ARTIFACTS_DIR="${PROJECT_DIR}/artifacts/c1_3"
BASELINES_DIR="${ARTIFACTS_DIR}/baselines"
RESULTS_DIR="${BASELINES_DIR}/evaluation_results"
EFOLD_NOVEL_MIG="/tmp/efold_novel_mig"

cd "${PROJECT_DIR}"
source /home/cunyuliu/miniconda3/etc/profile.d/conda.sh
conda activate editflow

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

TARGET=46147

while true; do
    # Count total predictions
    total=$(cat ${EFOLD_NOVEL_MIG}/worker_*/novel_clan.efold.predictions.jsonl 2>/dev/null | wc -l)

    # Check if workers 6 and 7 are done
    w6_done=$([ -f ${EFOLD_NOVEL_MIG}/worker_6/efold_results.json ] && echo YES || echo NO)
    w7_done=$([ -f ${EFOLD_NOVEL_MIG}/worker_7/efold_results.json ] && echo YES || echo NO)

    log "eFold novel: ${total}/${TARGET} predictions, worker_6=${w6_done}, worker_7=${w7_done}"

    if [ "${w6_done}" = "YES" ] && [ "${w7_done}" = "YES" ]; then
        log "All eFold novel workers complete! Re-merging and re-evaluating..."

        # Re-merge novel_clan
        python3 scripts/merge_efold_predictions.py \
            --tier novel_clan \
            --worker-dirs ${EFOLD_NOVEL_MIG}/worker_0 ${EFOLD_NOVEL_MIG}/worker_1 \
                           ${EFOLD_NOVEL_MIG}/worker_2 ${EFOLD_NOVEL_MIG}/worker_3 \
                           ${EFOLD_NOVEL_MIG}/worker_4 ${EFOLD_NOVEL_MIG}/worker_5 \
                           ${EFOLD_NOVEL_MIG}/worker_6 ${EFOLD_NOVEL_MIG}/worker_7 \
            --output ${BASELINES_DIR}/efold_same_split/novel_clan.efold.predictions.jsonl \
            2>&1 | tail -3

        # Re-evaluate eFold
        python3 scripts/evaluate_external_baseline_predictions.py \
            --model efold \
            --gold-json "in_clan=${GOLD_DIR}/test.jsonl" \
            --gold-json "novel_clan=${GOLD_DIR}/novel.jsonl" \
            --prediction-json "in_clan=${BASELINES_DIR}/efold_same_split/in_clan.efold.predictions.jsonl" \
            --prediction-json "novel_clan=${BASELINES_DIR}/efold_same_split/novel_clan.efold.predictions.jsonl" \
            --output ${RESULTS_DIR}/baseline_efold_results.json \
            --protocol same_split_local \
            --seed-count single_seed \
            --emit-partial-rows \
            2>&1 | tail -3

        # Re-merge baseline results
        python3 scripts/merge_baseline_results.py ${ARTIFACTS_DIR} 2>&1 | tail -3

        # Update SOTA table
        python3 /tmp/update_sota_table.py 2>/dev/null || true

        log "eFold re-evaluation complete!"
        break
    fi

    sleep 300
done
