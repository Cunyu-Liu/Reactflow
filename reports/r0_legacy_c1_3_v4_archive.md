# R0 legacy C1-3 v4 historical archive

This report records a protected legacy run as historical-only evidence. It is not an EPRO efficacy result and cannot be used to support a training, benchmark, or scientific-success claim.

## Scope

- Legacy checkout: `/home/cunyuliu/reactflow_c1_3_stage_20260722` at `2cdf9faf02f075b6f9289e84411a1ae60ff8d45a` on `trae/c1-3-static-scale`.
- Run: `pairformer_ribonanza_frozen_small_pair_fsdp_seed0_v4`.
- Artifact link resolves to `/home/cunyuliu/reactflow/artifacts`.
- The selected v4 run directory was present but contained no files or checkpoints at archive time.

## Authorized stop and preservation

The user explicitly authorized direct termination of this old task. `SIGTERM` was sent at `2026-07-30T11:56:17+08:00` to the torchrun leader and three verified workers. No `SIGKILL`, restart, or new seed was used. A post-grace audit found no target process or GPU binding.

The preserved launch log has 879 lines and SHA-256 `4e18b652b330a0e7370c19693daf3bb142a958ba229322d487d0b5e211edcf56`. Its final persisted `train_step` record is step 27000 (global step 27001); this is an observation, not a completion marker. The log records orderly worker shutdown after SIGTERM and a torch-elastic `SignalException` for signal 15.

## Boundaries

The V3 contract default natural-completion wait was superseded only for this legacy task by the explicit user authorization. This archive neither alters the legacy checkout nor authorizes resumption, extra seeds, or scientific interpretation. The machine-readable record is `manifests/reactflow_delta/r0/legacy_c1_3_v4_archive_manifest.json`; the external copy and raw log live under the R0 artifact root.
