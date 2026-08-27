#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v8_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
v13_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v13/v13m3_screen_seed0
v14_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
source_manifest=/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/source_binding/puzzle_set_source_manifest.json
out=/mnt/cunyuliu/reactflow_delta_puzzle_set_meta_context/p1m3_screen_seed0

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

archive_incomplete_fold() {
  local fold=$1
  local result="${out}/puzzle_set_fold_result_fold${fold}_seed0.json"
  if [[ -f "${result}" ]]; then
    return 0
  fi
  local partial=(
    "${out}/puzzle_set_predictions_fold${fold}_seed0.npz"
    "${out}/puzzle_set_candidate_point_fold${fold}_seed0.pt"
    "${out}/puzzle_set_null_point_fold${fold}_seed0.pt"
    "${out}/puzzle_set_candidate_wt_decoder_fold${fold}_seed0.pt"
    "${out}/puzzle_set_null_wt_decoder_fold${fold}_seed0.pt"
    "${out}/puzzle_set_candidate_residual_fold${fold}_seed0.pt"
    "${out}/puzzle_set_null_residual_fold${fold}_seed0.pt"
  )
  local present=()
  local path
  for path in "${partial[@]}"; do
    if [[ -e "${path}" ]]; then
      present+=("${path}")
    fi
  done
  if [[ "${#present[@]}" -gt 0 ]]; then
    local interrupted="${out}/interrupted_attempts/fold${fold}_$(date +%Y%m%dT%H%M%S)"
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
    scripts.reactflow_delta.run_puzzle_set_meta_context_probe \
      --repo-root "${repo}" \
      --phase P1M3 \
      --m2-csv "${m2}" \
      --v8-dir "${v8_dir}" \
      --v13-dir "${v13_dir}" \
      --v14-dir "${v14_dir}" \
      --tic2a-merged-json "${tic2a}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --source-manifest "${source_manifest}" \
      --out-dir "${out}" \
      --device cuda:0 \
      --folds "${fold}" \
      --pretraining-epochs 200 \
      --point-epochs 40 \
      --calibration-epochs 40 \
      --seed 0 \
      >> "${log}" 2>&1
}

tasks=()
for fold in {0..19}; do
  if [[ ! -f "${out}/puzzle_set_fold_result_fold${fold}_seed0.json" ]]; then
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
    if [[ -f "${out}/puzzle_set_fold_result_fold${fold}_seed0.json" ]]; then
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
    printf 'P1M3 fold %s with cuda_visible_devices_value=%s failed with status %s\n' \
      "${fold}" "${gpu}" "${task_status}" >&2
  elif ((failed == 0)); then
    if launch_next "${gpu}"; then
      :
    fi
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  printf 'one or more P1M3 prediction tasks failed; merge was not run\n' >&2
  exit 1
fi

merged="${out}/p1m3_complete_unscored_merge.json"
if [[ ! -f "${merged}" ]]; then
  "${python_bin}" -m scripts.reactflow_delta.merge_puzzle_set_meta_context_probe \
    --repo-root "${repo}" \
    --input-dir "${out}" \
    --phase P1M3 \
    --folds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --seeds 0 \
    --pretraining-epochs 200 \
    --point-epochs 40 \
    --calibration-epochs 40 \
    --parameter-count 6171697 \
    --trainable-parameter-count 1404417 \
    --out-json "${merged}"
fi
