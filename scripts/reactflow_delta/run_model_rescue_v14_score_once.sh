#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v14_20260827
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
out=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0
merged=${out}/v14m3_complete_unscored_merge.json
score=${out}/v14m3_complete_score.json
qualification=${out}/v14m3_qualification.json
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json
v12_score=/mnt/cunyuliu/reactflow_delta_model_rescue_v12/v12m3_screen_seed0/v12m3_complete_score.json

cd "${repo}"
for required in "${merged}" "${m2}" "${tic2a}" "${v12_score}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'required frozen V14 score input is missing: %s\n' "${required}" >&2
    exit 1
  fi
done
if [[ -e "${score}" || -e "${qualification}" ]]; then
  printf 'V14 score-once output already exists; refusing to rerun\n' >&2
  exit 1
fi

"${python_bin}" -m scripts.reactflow_delta.score_model_rescue_v14 \
  --repo-root "${repo}" \
  --merged-json "${merged}" \
  --tic2a-merged-json "${tic2a}" \
  --v12-score-json "${v12_score}" \
  --m2-csv "${m2}" \
  --out-json "${score}"
"${python_bin}" -m scripts.reactflow_delta.qualify_model_rescue_v14 \
  --score-json "${score}" \
  --out-json "${qualification}"
