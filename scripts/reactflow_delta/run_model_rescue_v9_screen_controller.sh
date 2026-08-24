#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v9_20260824
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v8_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
out=/mnt/cunyuliu/reactflow_delta_model_rescue_v9/v9m2_screen_seed0

mkdir -p "${out}/logs"
cd "${repo}"

run_shard() {
  local shard=$1
  local gpu=$2
  shift 2
  local requested=("$@")
  local missing=()
  local fold
  for fold in "${requested[@]}"; do
    if [[ ! -f "${out}/v9_fold_result_fold${fold}_seed0.json" ]]; then
      missing+=("${fold}")
    fi
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi
  local csv
  csv=$(IFS=,; echo "${missing[*]}")
  printf '%s shard=%s physical_gpu=%s folds=%s start\n' \
    "$(date --iso-8601=seconds)" "${shard}" "${gpu}" "${csv}" \
    >> "${out}/logs/shard${shard}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_model_rescue_v9 \
      --repo-root "${repo}" \
      --phase V9M2 \
      --m2-csv "${m2}" \
      --v8-dir "${v8_dir}" \
      --tic2a-merged-json "${tic2a}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --out-dir "${out}" \
      --device cuda:0 \
      --folds "${csv}" \
      --epochs 40 \
      --seed 0 \
      >> "${out}/logs/shard${shard}.log" 2>&1
}

run_shard 0 6 0 7 14 & pid0=$!
run_shard 1 0 1 8 15 & pid1=$!
run_shard 2 3 2 9 16 & pid2=$!
run_shard 3 7 3 10 17 & pid3=$!
run_shard 4 4 4 11 18 & pid4=$!
run_shard 5 1 5 12 19 & pid5=$!
run_shard 6 2 6 13 & pid6=$!

failed=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}" "${pid4}" "${pid5}" "${pid6}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more V9M2 prediction shards failed"
  exit 1
fi

"${python_bin}" -m scripts.reactflow_delta.merge_model_rescue_v9 \
  --input-dir "${out}" \
  --out-json "${out}/v9m2_complete_unscored_merge.json"
touch "${out}/v9m2_complete_prediction_only_merge_pass"
