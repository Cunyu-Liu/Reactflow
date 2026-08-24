#!/usr/bin/env bash
set -euo pipefail

REPO=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v7_20260824
PYTHON=/home/cunyuliu/miniconda3/envs/editflow/bin/python
M2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
V5=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
V6=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
V7=/mnt/cunyuliu/reactflow_delta_model_rescue_v7/v7m1_rinalmo_dependency_cache/dependency_cache.h5
V7_QUAL=/mnt/cunyuliu/reactflow_delta_model_rescue_v7/v7m1_rinalmo_dependency_cache/qualification.json
CORRECTED=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines
OUT=/mnt/cunyuliu/reactflow_delta_model_rescue_v7/v7m2_corrected_probe
EXPECTED_V7M1_STATUS=V7M1_OUTCOME_BLIND_RINALMO_DEPENDENCY_CACHE_PASS

mkdir -p "$OUT/logs"
cd "$REPO"

status="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$V7_QUAL")"
if [[ "$status" != "$EXPECTED_V7M1_STATUS" ]]; then
  echo "V7M2 controller requires exact qualified V7M1 cache"
  exit 1
fi

run_shard() {
  local shard="$1"
  shift
  local requested=("$@")
  local missing=()
  local fold
  for fold in "${requested[@]}"; do
    if [[ ! -f "$OUT/v7_probe_fold_result_fold${fold}.json" ]]; then
      missing+=("$fold")
    fi
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi
  local csv
  csv="$(IFS=,; echo "${missing[*]}")"
  "$PYTHON" -m scripts.reactflow_delta.run_model_rescue_v7_probe \
    --repo-root "$REPO" \
    --m2-csv "$M2" \
    --unconstrained-cache "$V5" \
    --constrained-cache "$V6" \
    --dependency-cache "$V7" \
    --corrected-baseline-dir "$CORRECTED" \
    --out-dir "$OUT" \
    --folds "$csv" \
    > "$OUT/logs/shard${shard}.log" 2>&1
}

run_shard 0 0 4 8 12 16 &
pid0=$!
run_shard 1 1 5 9 13 17 &
pid1=$!
run_shard 2 2 6 10 14 18 &
pid2=$!
run_shard 3 3 7 11 15 19 &
pid3=$!

failed=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more V7M2 prediction-only shards failed"
  exit 1
fi

complete=0
for fold in $(seq 0 19); do
  if [[ -f "$OUT/v7_probe_fold_result_fold${fold}.json" ]]; then
    complete=$((complete + 1))
  fi
done
if [[ "$complete" -ne 20 ]]; then
  echo "V7M2 prediction-only universe is incomplete: $complete/20"
  exit 1
fi

"$PYTHON" -m scripts.reactflow_delta.merge_model_rescue_v7_probe \
  --input-dir "$OUT" \
  --out-json "$OUT/v7m2_complete_merged_unscored.json"
touch "$OUT/v7m2_complete_unscored_merge_pass"
printf '%s V7M2_COMPLETE_UNSCORED_MERGE_PASS\n' "$(date --iso-8601=seconds)"
