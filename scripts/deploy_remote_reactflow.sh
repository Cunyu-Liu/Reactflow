#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

remote="${REMOTE:-cunyuliu@36.137.135.49}"
remote_port="${REMOTE_PORT:-22}"
remote_root="${REMOTE_ROOT:-/home/cunyuliu/reactflow}"
latest="${LATEST_RUN:-artifacts/full_runs/full_ablation_20260709_003012}"

ssh_cmd=(ssh -p "$remote_port" -o BatchMode=yes -o ConnectTimeout=15)
rsync_rsh="ssh -p $remote_port -o BatchMode=yes -o ConnectTimeout=15"
read -r -a rsync_flags <<< "${RSYNC_FLAGS:--a --partial}"

cd "$project_root"

"${ssh_cmd[@]}" "$remote" "mkdir -p '$remote_root/$latest/cache' '$remote_root/$latest/frozen/ribonanzanet2_sharded_full' '$remote_root/$latest/metadata' '$remote_root/$latest/splits'"

common_excludes=(
  --exclude ".coverage"
  --exclude ".pytest_cache/"
  --exclude "outputs/"
  --exclude "artifacts/full_runs/*/runs/"
  --exclude "artifacts/full_runs/*/logs/"
  --exclude "artifacts/full_runs/*/*.svg"
)

rsync "${rsync_flags[@]}" --delete -e "$rsync_rsh" \
  "${common_excludes[@]}" \
  README.md pyproject.toml src scripts tests docs \
  "$remote:$remote_root/"

rsync "${rsync_flags[@]}" -e "$rsync_rsh" \
  "$latest/cache/" \
  "$remote:$remote_root/$latest/cache/"

if [ -d "$latest/frozen/ribonanzanet2_sharded_full" ]; then
  rsync "${rsync_flags[@]}" -e "$rsync_rsh" \
    "$latest/frozen/ribonanzanet2_sharded_full/" \
    "$remote:$remote_root/$latest/frozen/ribonanzanet2_sharded_full/"
fi

for subdir in metadata splits; do
  if [ -d "$latest/$subdir" ]; then
    rsync "${rsync_flags[@]}" -e "$rsync_rsh" \
      "$latest/$subdir/" \
      "$remote:$remote_root/$latest/$subdir/"
  fi
done

rsync "${rsync_flags[@]}" -e "$rsync_rsh" \
  "$latest/"*.json "$latest/"*.md \
  "$remote:$remote_root/$latest/" 2>/dev/null || true

"${ssh_cmd[@]}" "$remote" "cd '$remote_root' && PYTHONPATH=src python3 -m pytest -q tests/test_build_sota_alignment_table.py tests/test_audit_queue_preflight.py"
