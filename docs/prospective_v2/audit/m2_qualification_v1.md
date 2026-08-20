# ReactFlow-Delta M2 20-fold screen qualification

Status: `M2_NO_RESCUE_CANDIDATE`. Evidence: `DEVELOPMENT_CONSUMED_SCREEN_NOT_CONFIRMATION`.

| candidate | CRPS gain vs B1 | CRPS positive puzzles | signed-delta MAE gain vs B1 | delta positive puzzles | status |
|---|---:|---:|---:|---:|---|
| l2_aligned_rank2 | +0.00069942 | 15/20 | -0.00000524 | 11/20 | `M2_SCREEN_FAIL` |
| sparse_delta_mdn_h0 | +0.00906010 | 20/20 | -0.00958523 | 6/20 | `M2_SCREEN_FAIL` |
| sparse_delta_mdn_h01 | +0.00886563 | 19/20 | -0.01240502 | 5/20 | `M2_SCREEN_FAIL` |

M3 eligible families: `b1_rfd_direct_aligned`.

SparseDelta lambda policy: `SPARSE_DELTA_FAMILY_EXCLUDED`.

This consumed-development screen can remove failed families only. It does not establish prospective, external, mechanism, or SOTA evidence.
