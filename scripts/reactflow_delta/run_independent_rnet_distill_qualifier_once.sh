#!/usr/bin/env bash
set -euo pipefail

repo=/home/cunyuliu/reactflow_delta_worktrees/independent_rnet_distill_20260828
python_bin=/home/cunyuliu/miniconda3/envs/editflow/bin/python
screen_dir=/mnt/cunyuliu/reactflow_delta_independent_rnet_distill/rnd3_screen_seed0
score=${screen_dir}/rnet_distill_complete_score.json
qualification=${screen_dir}/rnet_distill_qualification.json

cd "${repo}"
if [[ ! -f "${score}" ]]; then
  printf 'required complete RND4 score is missing: %s\n' "${score}" >&2
  exit 2
fi
if [[ -e "${qualification}" ]]; then
  printf 'RND5 canonical qualification already exists; refusing rerun\n' >&2
  exit 2
fi

"${python_bin}" -m scripts.reactflow_delta.qualify_independent_rnet_distill \
  --repo-root "${repo}" \
  --score-json "${score}" \
  --out-json "${qualification}"
