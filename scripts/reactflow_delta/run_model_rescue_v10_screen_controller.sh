#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v10_20260825
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v8_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
v9=/mnt/cunyuliu/reactflow_delta_model_rescue_v9/v9m2_screen_seed0/v9m2_complete_unscored_merge.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
out=/mnt/cunyuliu/reactflow_delta_model_rescue_v10/v10m2_screen_seed0

if [[ "$#" -lt 1 || "$#" -gt 8 ]]; then
  echo "usage: $0 PHYSICAL_GPU [PHYSICAL_GPU ...]" >&2
  exit 2
fi

gpus=("$@")
mkdir -p "${out}/logs"
cd "${repo}"

run_worker() {
  local worker=$1
  local gpu=$2
  shift 2
  local requested=("$@")
  local missing=()
  local fold
  for fold in "${requested[@]}"; do
    if [[ ! -f "${out}/v10_fold_result_fold${fold}_seed0.json" ]]; then
      missing+=("${fold}")
    fi
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi
  local csv
  csv=$(IFS=,; echo "${missing[*]}")
  printf '%s worker=%s physical_gpu=%s folds=%s start\n' \
    "$(date --iso-8601=seconds)" "${worker}" "${gpu}" "${csv}" \
    >> "${out}/logs/worker${worker}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_model_rescue_v10 \
      --repo-root "${repo}" \
      --phase V10M2 \
      --m2-csv "${m2}" \
      --v8-dir "${v8_dir}" \
      --tic2a-merged-json "${tic2a}" \
      --v9-merged-json "${v9}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --out-dir "${out}" \
      --device cuda:0 \
      --folds "${csv}" \
      --epochs 40 \
      --seed 0 \
      >> "${out}/logs/worker${worker}.log" 2>&1
}

worker_pids=()
worker_count=${#gpus[@]}
for ((worker=0; worker<worker_count; worker++)); do
  assigned=()
  for ((fold=worker; fold<20; fold+=worker_count)); do
    assigned+=("${fold}")
  done
  run_worker "${worker}" "${gpus[$worker]}" "${assigned[@]}" &
  worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more V10M2 prediction workers failed"
  exit 1
fi

"${python_bin}" -m scripts.reactflow_delta.merge_model_rescue_v10 \
  --input-dir "${out}" \
  --out-json "${out}/v10m2_complete_unscored_merge.json"
touch "${out}/v10m2_complete_prediction_only_merge_pass"
