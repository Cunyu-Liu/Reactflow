#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v12_20260825
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
unconstrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v5/v5m1_full/ensemble_delta_cache.h5
constrained=/mnt/cunyuliu/reactflow_delta_model_rescue_v6/v6m1_full/constrained_cache.h5
v11=/mnt/cunyuliu/reactflow_delta_model_rescue_v11/v11m3_screen_seed0/v11m3_complete_unscored_merge.json
out=/mnt/cunyuliu/reactflow_delta_model_rescue_v12/v12m2_real_smoke

if [[ "$#" -ne 2 ]]; then
  printf 'usage: %s PHYSICAL_GPU_FOR_FOLD0 PHYSICAL_GPU_FOR_FOLD1\n' "$0" >&2
  exit 2
fi

mkdir -p "${out}/logs"
cd "${repo}"

run_fold() {
  local fold=$1
  local gpu=$2
  if [[ -f "${out}/v12_fold_result_fold${fold}_seed0.json" ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m \
    scripts.reactflow_delta.run_model_rescue_v12 \
      --repo-root "${repo}" \
      --phase V12M2 \
      --m2-csv "${m2}" \
      --unconstrained-cache "${unconstrained}" \
      --constrained-cache "${constrained}" \
      --v11-merged-json "${v11}" \
      --out-dir "${out}" \
      --folds "${fold}" \
      --seed 0 \
      --inner-point-epochs 3 \
      --gate-steps 20 \
      --calibration-epochs 3 \
      --device cuda:0 \
      >> "${out}/logs/fold${fold}.log" 2>&1
}

run_fold 0 "$1" &
pid0=$!
run_fold 1 "$2" &
pid1=$!
failed=0
if ! wait "${pid0}"; then
  failed=1
fi
if ! wait "${pid1}"; then
  failed=1
fi
if [[ "${failed}" -ne 0 ]]; then
  printf 'one or more V12M2 smoke workers failed\n' >&2
  exit 1
fi

"${python_bin}" -m scripts.reactflow_delta.qualify_model_rescue_v12_smoke \
  --out-dir "${out}" \
  --out-json "${out}/v12m2_smoke_qualification.json"
