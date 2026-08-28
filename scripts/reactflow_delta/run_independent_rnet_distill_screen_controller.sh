#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/independent_rnet_distill_20260828
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
artifact_root=/mnt/cunyuliu/reactflow_delta_independent_rnet_distill
pretrain_dir=${artifact_root}/rnd1_pretrain
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v8_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v8/v8m1_corrected_experts_seed0
v10_dir=/mnt/cunyuliu/reactflow_delta_model_rescue_v10/v10m2_screen_seed0
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5

if (($# < 1 || $# > 9)); then
  printf 'usage: %s RND2|RND3 [PHYSICAL_GPU ...]\n' "$0" >&2
  exit 2
fi
phase=$1
shift
gpus=("$@")

case "${phase}" in
  RND2)
    out=${artifact_root}/rnd2_smoke_seed0
    folds=(0 1)
    point_epochs=3
    calibration_epochs=3
    experiment_id=RND2_RNET_DISTILL_TWO_FOLD_GPU_ENGINEERING_SMOKE
    ;;
  RND3)
    out=${artifact_root}/rnd3_screen_seed0
    folds=({0..19})
    point_epochs=40
    calibration_epochs=40
    experiment_id=RND3_RNET_DISTILL_COMPLETE_SEED0_PREDICTION_ONLY
    ;;
  *)
    printf 'unsupported independent RNet downstream phase: %s\n' "${phase}" >&2
    exit 2
    ;;
esac

cd "${repo}"
"${python_bin}" -m scripts.reactflow_delta.validate_independent_rnet_distill_contract \
  --repo-root "${repo}" >/dev/null

preflight_gpu() {
  local gpu=$1
  local status
  if CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -c \
    "from scripts.reactflow_delta.gpu_runtime import require_cuda_device; require_cuda_device('cuda:0')"; then
    return 0
  else
    status=$?
    printf '%s %s CUDA preflight failed: physical_gpu=%s logical_device=cuda:0 status=%s\n' \
      "$(date --iso-8601=seconds)" "${phase}" "${gpu}" "${status}" >&2
    return "${status}"
  fi
}

result_path() {
  local fold=$1
  printf '%s/rnet_distill_fold_result_fold%s_seed0.json' "${out}" "${fold}"
}

task_is_complete() {
  local fold=$1
  [[ -f "$(result_path "${fold}")" ]]
}

run_task() {
  local gpu=$1
  local fold=$2
  local log=${out}/logs/fold${fold}_seed0_gpu${gpu}.log
  printf '%s phase=%s physical_gpu=%s logical_device=cuda:0 fold=%s seed=0 start\n' \
    "$(date --iso-8601=seconds)" "${phase}" "${gpu}" "${fold}" >> "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_independent_rnet_distill_downstream \
      --repo-root "${repo}" \
      --phase "${phase}" \
      --experiment-id "${experiment_id}" \
      --m2-csv "${m2}" \
      --pretrain-dir "${pretrain_dir}" \
      --v8-dir "${v8_dir}" \
      --v10-dir "${v10_dir}" \
      --tic2a-merged-json "${tic2a}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --out-dir "${out}" \
      --device cuda:0 \
      --folds "${fold}" \
      --point-epochs "${point_epochs}" \
      --calibration-epochs "${calibration_epochs}" \
      --seed 0 \
      >> "${log}" 2>&1
}

tasks=()
for fold in "${folds[@]}"; do
  if ! task_is_complete "${fold}"; then
    tasks+=("${fold}")
  fi
done

if ((${#tasks[@]} > 0)); then
  if ((${#gpus[@]} == 0)); then
    printf '%s has missing folds but no physical GPU was supplied; logical_device=cuda:0\n' \
      "${phase}" >&2
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
declare -A fold_by_pid=()
active_pids=()
next_task=0

launch_next() {
  local gpu=$1
  local fold pid
  while ((next_task < ${#tasks[@]})); do
    fold=${tasks[$next_task]}
    ((next_task += 1))
    if task_is_complete "${fold}"; then
      continue
    fi
    run_task "${gpu}" "${fold}" &
    pid=$!
    active_pids+=("${pid}")
    gpu_by_pid["${pid}"]=${gpu}
    fold_by_pid["${pid}"]=${fold}
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
  fold=${fold_by_pid["${finished_pid}"]}
  remove_active_pid "${finished_pid}"
  unset "gpu_by_pid[${finished_pid}]"
  unset "fold_by_pid[${finished_pid}]"

  if ((task_status != 0)); then
    failed=1
    printf '%s %s fold=%s physical_gpu=%s logical_device=cuda:0 failed status=%s; merge blocked\n' \
      "$(date --iso-8601=seconds)" "${phase}" "${fold}" "${gpu}" "${task_status}" >&2
  elif ((failed == 0)); then
    if launch_next "${gpu}"; then
      :
    fi
  fi
done
if ((failed != 0)); then
  exit 1
fi

merged=${out}/rnet_distill_complete_unscored_merge.json
if [[ ! -f "${merged}" ]]; then
  "${python_bin}" -m scripts.reactflow_delta.merge_independent_rnet_distill \
    --repo-root "${repo}" \
    --input-dir "${out}" \
    --phase "${phase}" \
    --out-json "${merged}"
fi
