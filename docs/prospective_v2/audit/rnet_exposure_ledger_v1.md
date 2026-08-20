# RNet published-comparator exposure ledger — Phase 4 (audit §9.1/§9.2)

> Status: FAIL_CLOSED (published comparator NOT fairly runnable on the current
> external data). 2026-08-18. This is the `ph4_rnet` deliverable: a fair run of
> the published comparator or, failing that, an exposure ledger recording why.

## 1. Comparator identity

- **RibonanzaNet2 (RNet2)**, Kaggle model `shujun717/ribonanzanet2/PyTorch/alpha/1`
  (alpha-v1 release). ~100M-parameter RNA chemical-probing model trained on
  DMS/2A3-MaP reactivity profiles (Ribonanza competition data).
- Source checkout present on A100: `/home/cunyuliu/ribonanzanet_src/`
  (Network.py, inference.py, make_submission.py, configs/).
- Frozen feature export present:
  `/home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012/frozen/ribonanzanet2_sharded_full/`
  (409 shards, 208,905 records) with `weights_sha256 = c94031719c8a1c70a9068d5de861f65083cdf0555a15570b3724a8d6d7750e35`.

## 2. Fail-closed findings

### 2.1 Checkpoint not present for a clean re-run
- The exported feature shards exist, but no `.pth` / `.safetensors` checkpoint file
  was found in the current workspace (`find` over the artifacts tree returned no
  weights file). A fresh, verifiable re-run would require re-downloading the Kaggle
  model and re-verifying `weights_sha256` — not available in this environment.

### 2.2 Training-distribution overlap makes the comparison unfair (primary blocker)
- The external components are **Das-lab M2-seq 2A3-MaP data**:
  M2SL5 (SL5), M3SARS (FSE), 15KLIB (diverse), plus M2RFOK/M2RFPK (BigLib2) —
  all part of the Ribonanza-era M2-seq system (2 studies, 3 NovaSeq batches).
- RibonanzaNet2 was trained on the Ribonanza Kaggle competition data, which is
  built from **the same Das-lab M2-seq 2A3/DMS chemistry family**. The external
  sequences are therefore inside (or near) the comparator's training distribution.
- A "fair" development-disconnected comparison is **impossible** on these
  components: RNet2 would be tested on data it was trained on
  (train-on-test exposure), making any RNet2-vs-LRSO delta uninterpretable.

### 2.3 Task mismatch (secondary)
- RNet2 is a sequence→reactivity model (predicts per-position 2A3/DMS reactivity
  for a single sequence). The M2 task here is mutant full-construct response given
  the WT profile + exact SNV. A fair adaptation (run RNet2 on WT and mutant
  sequences and difference the predictions) is possible in principle, but it does
  not remove the §2.2 exposure problem.

## 3. Ledger entries (fail-closed)

| item | evidence | status |
|---|---|---|
| RNet2 source | `/home/cunyuliu/ribonanzanet_src/` | PRESENT |
| RNet2 frozen feature export | sharded_full 409 shards; weights_sha256 recorded | PRESENT |
| RNet2 checkpoint (.pth/.safetensors) | not found in workspace | **MISSING** |
| external data train-overlap | external = Das-lab M2-seq 2A3 (same family as RNet2 training) | **OVERLAP → unfair** |
| fair run on external clusters | impossible without new non-Das-lab M2 data | **NOT_RUNNABLE** |
| verdict | `RNET_COMPARATOR_FAIR_RUN_NOT_POSSIBLE` (fail-closed) | **FAIL_CLOSED** |

## 4. Recommendation

- Do NOT report an RNet2-vs-LRSO comparison on these external components (would be
  train-on-test). If a published-comparator benchmark is required, it must use
  genuinely new, development-disconnected, non-Das-lab M2 (or M2-compatible
  full-spectrum) data, obtained with an outcome-blind pre-frozen protocol — none
  currently available (audit P0-4/P0-6).
- RNet2 may still be cited as related work / a strong sequence→reactivity prior,
  but not as a head-to-head external comparator in this paper.
