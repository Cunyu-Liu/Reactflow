---
slug: orchestrator-260828-independent-rnet-distill
date: 2026-08-28
status: RNET_DISTILL_TWO_FOLD_GPU_ENGINEERING_SMOKE_ONLY
parent: orchestrator-260827-v14-wt-profile
formal_chain_status: FROZEN_INACTIVE_PENDING_RND5_RESULT
formal_phase_order: [RND6P, RND6S, RND6Q, RND6T]
formal_activation_allowed: false
formal_score_accessed: false
formal_qualification_accessed: false
screen_result_status: NOT_FINALIZED
formal_result_status: NOT_FINALIZED
publication_ready: false
---

# Independent RNet2 single-feature distillation

## Objective

Test whether a lightweight student distilled from correctly aligned frozen
RibonanzaNet2 per-nucleotide chemical representations transfers useful context
to unseen-puzzle mutant-response prediction beyond an exactly matched student
distilled from the same teacher values after a fixed within-sequence shift.

## Scientific boundary

This is a new independent project branched from the terminal V14/B5 state; it
does not reopen or rerun any V14, P1, P2, B5 or P3 route. RibonanzaNet2 source
exposure cannot be fully excluded and OpenKnot M2 is a heavily consumed
development benchmark. Every result therefore remains exposure-disclosed
development evidence. Smoke, training loss, cache integrity and prediction
coverage are engineering evidence only.

## Frozen attribution

The candidate and null share the exact student architecture, initialization,
teacher record universe, data order, batches, optimizer, epoch count and
downstream schedule. The candidate learns teacher feature at coordinate `i`;
the null learns the same sequence's teacher feature at coordinate
`(i - min(17, L-1)) mod L` (the fixed positive `torch.roll`). Only coordinate
alignment differs. Pair features,
live RNet2 inference and OpenKnot mutant outcomes are unavailable to
distillation.

## Stages

- RND0: source binding and focused implementation only; all training and score
  access closed.
- RND1: one paired GPU-only distillation run after RND0 exact PASS.
- RND2: folds 0/1, seed 0, 3+3 epoch GPU engineering smoke; no score.
- RND3: fixed seed-0 twenty-fold prediction-only screen; complete before score.
- RND4: one canonical complete-score read.
- RND5: one frozen-Gate qualification.
- RND6P: fixed seeds 0-4 crossed with folds 0-19, 40+40 epoch CUDA-only
  prediction, target-free merge and equal-weight assembly; inactive unless RND5
  has exact canonical PASS.
- RND6S: one complete five-seed formal score with all training and qualification
  closed.
- RND6Q: one formal qualifier with training and scorer closed.
- RND6T: terminal PASS, FAIL or INDETERMINATE record with no runnable phase and
  every runtime right closed.

## Frozen inactive formal confirmation

The formal chain is frozen before the RND5 result is known and is not current
authority. Its only activation predecessor is exact canonical status
`RNET_DISTILL_TOP_JOURNAL_DEVELOPMENT_SCREEN_PASS`; RND5 FAIL or INDETERMINATE
does not authorize formal work. RND6 uses exactly 100 `(fold, seed)` pairs from
folds 0-19 and seeds 0-4, the RND3 40+40 schedule, and a fixed equal seed weight
of 0.2. Prediction, complete scoring and qualification are separate authorities
and never overlap.

The equal-seed mixture must pass the complete unchanged RND5 `screen_gates`
mapping. For each of signed delta, point absolute, task CRPS and distribution
absolute, at least four of five individual seeds must also have positive mean
gain over the frozen shift17 matched null. No best seed, extra seed, model or
threshold selection is allowed. Even an exact formal PASS remains
`EXPOSURE_DISCLOSED_DEVELOPMENT_ONLY`, not clean OOD, external replication,
SOTA or publication-ready evidence.

## Falsification

The mechanism is falsified if the complete candidate fails any frozen matched
null, feature41, historical-parent, coverage or influence Gate. Gates cannot be
lowered after score access, no favorable seed may be appended, and failure does
not authorize revival of any closed family.

## Progress log

- 2026-08-28: Three independent model/statistical/implementation reviews were
  completed before new score access. RNet2 aligned distillation was selected as
  the only week-feasible independent information route; same-universe selection
  bias and exposure limits were recorded.
- 2026-08-28: A clean independent worktree was created at terminal parent
  `4702d2c`; the 47 GB teacher cache began a non-destructive copy from the legacy
  `/home` artifact path to the project `/mnt/cunyuliu` source path.
- 2026-08-28: The legacy parent manifest was found to have stale content-hash
  declarations for 388 of 409 child shards, while record counts, weights,
  schema and physical files remained consistent. The final 9-record payload was
  verified exactly with the canonical shard reader. The project does not turn a
  47 GB rehash into a scientific Gate: it preserves the raw copy, discloses the
  stale metadata and binds child provenance plus every actual index/NPZ shape.
- 2026-08-28: The non-destructive `/mnt/cunyuliu` teacher-cache copy completed
  with 2,425 files, 49,798,728,468 bytes, zero deletions and exit 0. The focused
  RND0 implementation passed 44 tests, eight Python compilation checks, three
  shell syntax checks and clean authority validation; commit `d6a15a5` was
  pushed with upstream 0/0.
- 2026-08-28: The canonical target-free structural projection bound 208,905
  records, 409 shards and 384-wide single features with exact status
  `RNET2_TEACHER_STRUCTURAL_SOURCE_BINDING_EXACT_PASS`. It performed no live
  teacher inference, read no OpenKnot outcome and did not rehash the 47 GB
  cache. RND0 is closed and exactly one paired CUDA-only RND1 pretraining run
  is authorized; score, smoke and screen access remain closed.
- 2026-08-28: The inactive RND6 formal contract was frozen as four mutually
  exclusive phases `RND6P→RND6S→RND6Q→RND6T`, with canonical paths, 100
  fold-seed pairs, fixed 0.2 equal-seed mixture, unchanged screen Gates and the
  pre-result four-of-five matched-null seed-stability rule. This preparation did
  not change the active RND1 token, runnable phase, permissions or next action
  and did not authorize RND6.
- 2026-08-29: The single frozen RND1 paired CUDA pretraining run exited zero on
  physical GPU 6 mapped to logical `cuda:0`. The loss-blind terminal validator
  cross-bound the exact three artifacts, confirmed no CPU fallback or outcome,
  loss, or scientific-metric access, preserved identical residual heads, and
  found 84 changed encoder tensors between the aligned and shift17 students.
  Exact status is `RND1_PAIRED_PRETRAIN_EXACT_PASS`; this is engineering
  pretraining evidence only. RND1 is closed and exactly one seed-0 folds 0/1,
  3+3 epoch RND2 CUDA engineering smoke is authorized without score access.
