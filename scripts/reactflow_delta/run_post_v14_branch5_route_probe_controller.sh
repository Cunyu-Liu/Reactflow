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

run_task() {
  local gpu=$1
  local fold=$2
  archive_incomplete_fold "${fold}"
  local log="${out}/logs/fold${fold}_seed0_gpu${gpu}.log"
  printf '%s cuda_visible_devices_value=%s logical_device=cuda:0 fold=%s start\n' \
    "$(date --iso-8601=seconds)" "${gpu}" "${fold}" >> "${log}"
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
      --folds "${fold}" \
      >> "${log}" 2>&1
}

tasks=()
for fold in {0..19}; do
  if ! fold_is_complete "${fold}"; then
    tasks+=("${fold}")
  fi
done
if ((${#tasks[@]} > 0)); then
  for gpu in "${gpus[@]}"; do
    require_gpu "${gpu}"
  done
  mkdir -p "${out}/logs" "${out}/interrupted_attempts"
fi

declare -A gpu_by_pid=()
declare -A task_by_pid=()
active_pids=()
next_task=0

launch_next() {
  local gpu=$1
  local fold pid
  while ((next_task < ${#tasks[@]})); do
    fold=${tasks[$next_task]}
    ((next_task += 1))
    if fold_is_complete "${fold}"; then
      continue
    fi
    run_task "${gpu}" "${fold}" &
    pid=$!
    active_pids+=("${pid}")
    gpu_by_pid["${pid}"]=${gpu}
    task_by_pid["${pid}"]=${fold}
    return 0
  done
  return 1
}

remove_active_pid() {
  local finished_pid=$1
  local pid
  local remaining=()
  for pid in "${active_pids[@]}"; do
    if [[ "${pid}" != "${finished_pid}" ]]; then
      remaining+=("${pid}")
    fi
  done
  active_pids=("${remaining[@]}")
}

for gpu in "${gpus[@]}"; do
  if ! launch_next "${gpu}"; then
    break
  fi
done

failed=0
while ((${#active_pids[@]} > 0)); do
  finished_pid=
  if wait -n -p finished_pid "${active_pids[@]}"; then
    task_status=0
  else
    task_status=$?
  fi
  gpu=${gpu_by_pid["${finished_pid}"]}
  fold=${task_by_pid["${finished_pid}"]}
  remove_active_pid "${finished_pid}"
  unset "gpu_by_pid[${finished_pid}]"
  unset "task_by_pid[${finished_pid}]"

  if ((task_status != 0)); then
    failed=1
    printf 'branch5 fold %s with cuda_visible_devices_value=%s failed with status %s\n' \
      "${fold}" "${gpu}" "${task_status}" >&2
  elif ((failed == 0)); then
    if launch_next "${gpu}"; then
      :
    fi
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  printf 'one or more branch5 prediction-only tasks failed; merge was not run\n' >&2
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
