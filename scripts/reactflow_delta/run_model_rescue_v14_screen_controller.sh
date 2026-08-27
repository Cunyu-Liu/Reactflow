#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v14_20260827
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v8_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
v10_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v10/v10m2_screen_seed0
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
out=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0

if [[ "$#" -gt 8 ]]; then
  printf 'usage: %s [PHYSICAL_GPU ...]\n' "$0" >&2
  exit 2
fi

gpus=("$@")
cd "${repo}"

preflight_gpu() {
  local gpu=$1
  local status
  if CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -c \
    "from scripts.reactflow_delta.gpu_runtime import require_cuda_device; require_cuda_device('cuda:0')"; then
    return 0
  else
    status=$?
    printf 'V14M3 CUDA preflight failed: cuda_visible_devices_value=%s logical_device=cuda:0 status=%s\n' \
      "${gpu}" "${status}" >&2
    return "${status}"
  fi
}

result_path() {
  local seed=$1
  local fold=$2
  printf '%s/v14_fold_result_fold%s_seed%s.json' "${out}" "${fold}" "${seed}"
}

task_is_complete() {
  local seed=$1
  local fold=$2
  [[ -f "$(result_path "${seed}" "${fold}")" ]]
}

run_task() {
  local gpu=$1
  local seed=$2
  local fold=$3
  local log="${out}/logs/fold${fold}_seed${seed}_gpu${gpu}.log"
  printf '%s cuda_visible_devices_value=%s seed=%s fold=%s start\n' \
    "$(date --iso-8601=seconds)" "${gpu}" "${seed}" "${fold}" >> "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_model_rescue_v14 \
      --repo-root "${repo}" \
      --phase V14M3 \
      --m2-csv "${m2}" \
      --v8-dir "${v8_dir}" \
      --v10-dir "${v10_dir}" \
      --tic2a-merged-json "${tic2a}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --out-dir "${out}" \
      --device cuda:0 \
      --folds "${fold}" \
      --pretraining-epochs 200 \
      --point-epochs 40 \
      --calibration-epochs 40 \
      --seed "${seed}" \
      >> "${log}" 2>&1
}

tasks=()
for fold in {0..19}; do
  if ! task_is_complete 0 "${fold}"; then
    tasks+=("0:${fold}")
  fi
done

if ((${#tasks[@]} > 0)); then
  if ((${#gpus[@]} == 0)); then
    printf 'V14M3 has missing tasks but no CUDA_VISIBLE_DEVICES value was supplied; logical_device=cuda:0\n' >&2
    exit 2
  fi
  preflight_failed=0
  for gpu in "${gpus[@]}"; do
    if ! preflight_gpu "${gpu}"; then
      preflight_failed=1
    fi
  done
  if ((preflight_failed != 0)); then
    exit 1
  fi
  mkdir -p "${out}/logs"
fi

declare -A gpu_by_pid=()
declare -A task_by_pid=()
active_pids=()
next_task=0

launch_next() {
  local gpu=$1
  local task seed fold pid
  while ((next_task < ${#tasks[@]})); do
    task=${tasks[$next_task]}
    ((next_task += 1))
    IFS=: read -r seed fold <<< "${task}"
    if task_is_complete "${seed}" "${fold}"; then
      continue
    fi
    run_task "${gpu}" "${seed}" "${fold}" &
    pid=$!
    active_pids+=("${pid}")
    gpu_by_pid["${pid}"]=${gpu}
    task_by_pid["${pid}"]=${task}
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
  task=${task_by_pid["${finished_pid}"]}
  remove_active_pid "${finished_pid}"
  unset "gpu_by_pid[${finished_pid}]"
  unset "task_by_pid[${finished_pid}]"

  if ((task_status != 0)); then
    failed=1
    printf 'V14M3 task %s with cuda_visible_devices_value=%s failed with status %s\n' \
      "${task}" "${gpu}" "${task_status}" >&2
  elif ((failed == 0)); then
    if launch_next "${gpu}"; then
      :
    fi
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  printf 'one or more V14M3 prediction tasks failed; merge was not run\n' >&2
  exit 1
fi

"${python_bin}" -m scripts.reactflow_delta.merge_model_rescue_v14 \
  --input-dir "${out}" \
  --phase V14M3 \
  --out-json "${out}/v14m3_complete_unscored_merge.json"
