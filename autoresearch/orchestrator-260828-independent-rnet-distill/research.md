---
slug: orchestrator-260828-independent-rnet-distill
date: 2026-08-28
status: RND0_IMPLEMENTATION_SOURCE_BINDING_ONLY
parent: orchestrator-260827-v14-wt-profile
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
- RND6: fixed seeds 0-4 only after exact RND5 PASS.

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
