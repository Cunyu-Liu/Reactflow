#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/model_rescue_v14_20260827
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
out=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m4_formal_seeds0_4
merged=${out}/v14m4_complete_unscored_merge.json
assembly=${out}/v14m4_five_seed_prediction_only_assembly.json
score=${out}/v14m4_complete_formal_score.json
qualification=${out}/v14m4_formal_qualification.json
screen_qualification=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/v14m3_qualification.json
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
tic2a=/mnt/cunyuliu/reactflow_delta_target_identity_correction/tic2a_corrected_baselines/tic2a_corrected_merged_unscored.json

cd "${repo}"
for required in "${merged}" "${assembly}" "${screen_qualification}" "${m2}" "${tic2a}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'required frozen V14 formal score input is missing: %s\n' "${required}" >&2
    exit 1
  fi
done
if [[ -e "${score}" || -e "${qualification}" ]]; then
  printf 'V14 formal score-once output already exists; refusing to rerun\n' >&2
  exit 1
fi

"${python_bin}" -m scripts.reactflow_delta.score_model_rescue_v14_formal \
  --repo-root "${repo}" \
  --assembly-json "${assembly}" \
  --merged-json "${merged}" \
  --tic2a-merged-json "${tic2a}" \
  --m2-csv "${m2}" \
  --out-json "${score}"
"${python_bin}" -m scripts.reactflow_delta.qualify_model_rescue_v14_formal \
  --score-json "${score}" \
  --screen-qualification-json "${screen_qualification}" \
  --out-json "${qualification}"
