#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
source_manifest=/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/source_binding/post_v14_branch5_safe_source_manifest.json
out=/mnt/cunyuliu/reactflow_delta_post_v14_branch5_route_probe/b5rp1_seed0
merged=${out}/puzzle_set_branch5_probe_complete_unscored_merge.json

if [[ "$#" -lt 1 || "$#" -gt 8 ]]; then
  printf 'usage: %s CUDA_VISIBLE_DEVICES_VALUE [CUDA_VISIBLE_DEVICES_VALUE ...]\n' "$0" >&2
  exit 2
fi
gpus=("$@")
cd "${repo}"

require_gpu() {
  local cuda_visible_devices_value=$1
  local status
  if CUDA_VISIBLE_DEVICES="${cuda_visible_devices_value}" "${python_bin}" -c \
    "from scripts.reactflow_delta.gpu_runtime import require_cuda_device; require_cuda_device('cuda:0')"; then
    return 0
  else
    status=$?
  fi
  printf 'CUDA_REQUIRED: cuda_visible_devices_value=%s logical_device=cuda:0 preflight_status=%s\n' \
    "${cuda_visible_devices_value}" "${status}" >&2
  return "${status}"
}

fold_result() {
  printf '%s/puzzle_set_branch5_probe_fold%s_seed0.json' "${out}" "$1"
}

fold_prediction() {
  printf '%s/puzzle_set_branch5_probe_predictions_fold%s_seed0.npz' "${out}" "$1"
}

fold_ridge() {
  printf '%s/puzzle_set_branch5_probe_ridge_fold%s_seed0.json' "${out}" "$1"
}

fold_is_complete() {
  local fold=$1
  [[ -f "$(fold_result "${fold}")" && \
     -f "$(fold_prediction "${fold}")" && \
     -f "$(fold_ridge "${fold}")" ]]
}

archive_incomplete_fold() {
  local fold=$1
  if fold_is_complete "${fold}"; then
    return 0
  fi
  local present=()
  local path
  for path in \
    "$(fold_result "${fold}")" \
    "$(fold_prediction "${fold}")" \
    "$(fold_ridge "${fold}")"; do
    if [[ -e "${path}" ]]; then
      present+=("${path}")
    fi
  done
  if [[ "${#present[@]}" -gt 0 ]]; then
    local interrupted
    interrupted="${out}/interrupted_attempts/fold${fold}_$(date +%Y%m%dT%H%M%S)_$$"
    mkdir -p "${interrupted}"
    mv "${present[@]}" "${interrupted}/"
  fi
}

run_worker() {
  local worker=$1
  local gpu=$2
  shift 2
  local requested=("$@")
  local missing=()
  local fold
  for fold in "${requested[@]}"; do
    if ! fold_is_complete "${fold}"; then
      archive_incomplete_fold "${fold}"
      missing+=("${fold}")
    fi
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi
  local csv
  csv=$(IFS=,; printf '%s' "${missing[*]}")
  local log="${out}/logs/worker${worker}_gpu${gpu}.log"
  printf '%s worker=%s cuda_visible_devices_value=%s logical_device=cuda:0 folds=%s start\n' \
    "$(date --iso-8601=seconds)" "${worker}" "${gpu}" "${csv}" >> "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_post_v14_branch5_route_probe \
      --repo-root "${repo}" \
      --m2-csv "${m2}" \
      --source-manifest "${source_manifest}" \
      --tic2a-merged-json "${tic2a}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --out-dir "${out}" \
      --device cuda:0 \
      --folds "${csv}" \
      >> "${log}" 2>&1
}

has_missing_training=0
for fold in {0..19}; do
  if ! fold_is_complete "${fold}"; then
    has_missing_training=1
    break
  fi
done
if [[ "${has_missing_training}" -ne 0 ]]; then
  for gpu in "${gpus[@]}"; do
    require_gpu "${gpu}"
  done
fi

mkdir -p "${out}/logs" "${out}/interrupted_attempts"

pids=()
worker_count=${#gpus[@]}
for ((worker=0; worker<worker_count; worker++)); do
  assigned=()
  for ((fold=worker; fold<20; fold+=worker_count)); do
    assigned+=("${fold}")
  done
  run_worker "${worker}" "${gpus[$worker]}" "${assigned[@]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  printf 'one or more branch5 prediction-only workers failed\n' >&2
  exit 1
fi

for fold in {0..19}; do
  if ! fold_is_complete "${fold}"; then
    printf 'branch5 fold %s is incomplete after workers exited\n' "${fold}" >&2
    exit 1
  fi
done

if [[ -e "${merged}" ]]; then
  printf 'branch5 complete unscored merge already exists; refusing overwrite: %s\n' \
    "${merged}"
  exit 0
fi
"${python_bin}" -m scripts.reactflow_delta.merge_post_v14_branch5_route_probe \
  --input-dir "${out}" \
  --out-json "${merged}"
