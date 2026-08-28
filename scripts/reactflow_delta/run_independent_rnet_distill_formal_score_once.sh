#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/independent_rnet_distill_20260828
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
formal_dir=/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/rnd6_formal_seeds0_4
merged=${formal_dir}/rnet_distill_complete_unscored_merge.json
assembly=${formal_dir}/assembled/rnet_distill_five_seed_prediction_only_assembly.json
score=${formal_dir}/rnet_distill_complete_formal_score.json
qualification=${formal_dir}/rnet_distill_formal_qualification.json
m2=/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/openknot_m2/OK7a_M2_data.v4.5.2.csv
v14_score=/mnt/cunyuliu/reactflow_delta_model_rescue_v14/v14m3_screen_seed0/v14m3_complete_score.json

cd "${repo}"
for required in "${merged}" "${assembly}" "${m2}" "${v14_score}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'required frozen RND6S score input is missing: %s\n' "${required}" >&2
    exit 2
  fi
done
if [[ -e "${score}" || -e "${qualification}" ]]; then
  printf 'RND6S/RND6Q canonical output already exists; refusing score rerun\n' >&2
  exit 2
fi

"${python_bin}" -m scripts.reactflow_delta.score_independent_rnet_distill_formal \
  --repo-root "${repo}" \
  --merged-json "${merged}" \
  --assembly-json "${assembly}" \
  --m2-csv "${m2}" \
  --historical-v14-score-json "${v14_score}" \
  --out-json "${score}"
