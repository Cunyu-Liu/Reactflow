#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/independent_rnet_distill_20260828
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
screen_dir=/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/rnd3_screen_seed0
merged=${screen_dir}/rnet_distill_complete_unscored_merge.json
score=${screen_dir}/rnet_distill_complete_score.json
qualification=${screen_dir}/rnet_distill_qualification.json
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v14_score=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/v14m3_complete_score.json

cd "${repo}"
for required in "${merged}" "${m2}" "${v14_score}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'required frozen RND4 score input is missing: %s\n' "${required}" >&2
    exit 2
  fi
done
if [[ -e "${score}" || -e "${qualification}" ]]; then
  printf 'RND4/RND5 canonical output already exists; refusing score rerun\n' >&2
  exit 2
fi

"${python_bin}" -m scripts.reactflow_delta.score_independent_rnet_distill \
  --repo-root "${repo}" \
  --merged-json "${merged}" \
  --m2-csv "${m2}" \
  --historical-v14-score-json "${v14_score}" \
  --out-json "${score}"
