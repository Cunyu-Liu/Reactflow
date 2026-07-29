# ReactFlow-Δ EPRO R0 read-only preflight

- Task: `T-R0.1`
- Snapshot completed: `2026-07-29T13:10:30+08:00`
- Contract: `ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_EPRO_20260729.md`
- Contract SHA-256: `3efcc1504208d8089236dfe4e7d41553741441d3b86b6174c8b5af52d614ec10`
- Mode: read-only; no process signals, file edits, restarts, or learned training

## Legacy checkout

The protected legacy checkout is
`/home/cunyuliu/reactflow_c1_3_stage_20260722` on
`trae/c1-3-static-scale` at
`2cdf9faf02f075b6f9289e84411a1ae60ff8d45a`. Its remote is
`git@github.com:Cunyu-Liu/Reactflow.git`.

The worktree is deliberately preserved as dirty:

- 8 tracked modified files;
- 8 untracked files, including backups;
- no staged changes;
- 288 insertions and 112 deletions in the tracked diff.

Every dirty or untracked file is listed with its SHA-256 in
`manifests/reactflow_delta/r0/preflight_snapshot.json`. None was copied into a
new commit, cleaned, overwritten, or deleted.

The legacy `artifacts` symlink points to
`/home/cunyuliu/reactflow/artifacts`. It is historical shared storage and must
not receive ReactFlow-Δ artifacts.

## Protected workloads

Two user workloads were active in the snapshot:

1. Legacy C1-3 v4 training: `torchrun` PID `2353493`, ranks
   `2353828/2353829/2353830`, GPUs `0/2/5`, started
   `2026-07-29T08:49:01+08:00`.
2. Legacy full checkpoint evaluation: PID `482511`, workers
   `410295/410329`, GPU `4`, started
   `2026-07-29T05:52:24+08:00`.

The v4 process had one read-only observation at epoch 0, global step 4201 with
`skip_nan=false`. It must finish naturally. R0 must not kill, restart, alter,
or append a seed. Subsequent checks must follow the contract cadence.

No GPU was conflict-free and exclusively available. GPU 6 had substantial
free memory but also another user's workloads, so free memory was not treated
as ownership. R0 and D0 perform no learned training in any case.

## Storage, data, and artifacts

- `/home`: 7.0 TB total, 1.3 TB used, 5.4 TB available; 3% inode use.
- Shared historical artifacts: 51 GB.
- Legacy split: 307,641 static records in a 600 MB split directory.
- Legacy RibonanzaNet2 frozen features: 208,905 records, 409 shards, 47 GB.

These legacy records and features are static/proxy assets, not paired
experimental WT-mutant response truth. They cannot seed D0 pair counts.

## Historical evidence boundary

The old `gate_audit.json` is `FAIL` with 19/23 checks, zero completed seeds in
the multiseed artifact, and zero significance tests. It also preserves a
misleading approximately 0.80 quick-evaluation F1. The later full validation
F1 is approximately 0.4013, below the same-protocol EternaFold reference
approximately 0.7039.

Therefore:

- quick evaluation is historical negative evidence, not a scientific result;
- old static/proxy data are not intervention evidence;
- no old smoke, gate, or report is inherited as an EPRO PASS.

## Safety conclusion

T-R0.1 is recorded as a read-only snapshot. The legacy checkout, user files,
processes, and artifacts remain protected. This record does not complete
T-R0.2 or T-R0.3 and does not authorize D0 while the R0 task sequence remains
incomplete.
