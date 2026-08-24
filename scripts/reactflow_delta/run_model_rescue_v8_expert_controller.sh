#!/usr/bin/env bash
set -euo pipefail

REPO=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v8_20260824
PYTHON=/home/cunyuliu/miniconda3/envs/editflow/bin/python
M2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
OUT=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0

mkdir -p "$OUT/logs"
cd "$REPO"

run_shard() {
  local shard="$1"
  local physical_gpu="$2"
  shift 2
  local requested=("$@")
  local missing=()
  local fold
  for fold in "${requested[@]}"; do
    if [[ ! -f "$OUT/v8_corrected_expert_fold_result_fold${fold}_seed0.json" ]]; then
      missing+=("$fold")
    fi
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi
  local csv
  csv="$(IFS=,; echo "${missing[*]}")"
  printf '%s shard=%s physical_gpu=%s folds=%s start\n' \
    "$(date --iso-8601=seconds)" "$shard" "$physical_gpu" "$csv" \
    >> "$OUT/logs/shard${shard}.log"
  CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYTHON" -m \
    scripts.reactflow_delta.run_model_rescue_v8_expert_rebuild \
      --repo-root "$REPO" \
      --m2-csv "$M2" \
      --out-dir "$OUT" \
      --device cuda:0 \
      --folds "$csv" \
      --epochs 40 \
      --learning-rate 0.001 \
      --weight-decay 0.0 \
      --seed 0 \
      >> "$OUT/logs/shard${shard}.log" 2>&1
}

run_shard 0 6 0 4 8 12 16 &
pid0=$!
run_shard 1 3 1 5 9 13 17 &
pid1=$!
run_shard 2 0 2 6 10 14 18 &
pid2=$!
run_shard 3 7 3 7 11 15 19 &
pid3=$!

failed=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more V8M1 corrected expert shards failed"
  exit 1
fi

complete=0
for fold in $(seq 0 19); do
  if [[ -f "$OUT/v8_corrected_expert_fold_result_fold${fold}_seed0.json" ]]; then
    complete=$((complete + 1))
  fi
done
if [[ "$complete" -ne 20 ]]; then
  echo "V8M1 corrected expert universe is incomplete: $complete/20"
  exit 1
fi

"$PYTHON" -m scripts.reactflow_delta.qualify_model_rescue_v8_expert_rebuild \
  --input-dir "$OUT" \
  --m2-csv "$M2" \
  --out-json "$OUT/v8m1_corrected_expert_qualification.json"
touch "$OUT/v8m1_corrected_expert_rebuild_pass"
printf '%s V8M1_CORRECTED_EXPERT_REBUILD_PASS\n' "$(date --iso-8601=seconds)"
