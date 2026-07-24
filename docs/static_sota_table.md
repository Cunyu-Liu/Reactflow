# Static RNA Secondary Structure Prediction — SOTA Comparison Table

**Protocol**: same_split_local (MMseqs-disjoint, C1-1 frozen splits)
**Date**: 2026-07-24
**Phase**: C1-3 Pilot

## Same-Split Results (MEA decoder)

| Model | Type | in_clan F1 | in_clan MCC | in_clan long F1 | novel_clan F1 | novel_clan MCC | novel_clan long F1 |
|-------|------|------------|-------------|-----------------|---------------|----------------|-------------------|
| ViennaRNA MFE | Thermodynamic | **0.682** | **0.694** | 0.575 | **0.682** | **0.694** | 0.580 |
| eFold/RNAndria | Deep learning | 0.220 | 0.224 | 0.192 | 0.212 | 0.217 | 0.190 |
| ReactFlow (ribonanza_frozen, 3 seeds) | PairFormer + frozen | 0.121 | — | — | 0.134 | — | — |
| ReactFlow (from_scratch, 3 seeds) | PairFormer | 0.058 | — | — | 0.063 | — | — |

## Gap Analysis

| Comparison | F1 Ratio | Gap |
|-----------|----------|-----|
| ribonanza_frozen / ViennaRNA | 0.18x | -0.561 |
| ribonanza_frozen / eFold | 0.55x | -0.099 |
| from_scratch / ViennaRNA | 0.09x | -0.624 |
| from_scratch / eFold | 0.26x | -0.162 |
| ribonanza_frozen / from_scratch | 2.09x | +0.063 |

## Notes

- All ReactFlow results are **pilot** (10K train / 1K eval / 5 epochs / batch_size=4 / max_len=400)
- ViennaRNA and eFold are **full** same-split baselines (evaluated on all 16,606 test + 46,147 novel sequences)
- ReactFlow pilot used only 4.4% of training data (10K / 228K)
- single_only and pair_feature configs are identical (fusion not integrated)
- Full-scale training required to close the gap

## Baselines Not Yet Run

| Model | Status | Reason |
|-------|--------|--------|
| EternaFold | Not run | Needs isolated venv |
| MXfold2 | Not run | Needs isolated venv |
| UFold | Not run | Needs isolated venv |
| RNAformer | Not run | Needs isolated venv |
| RiNALMo fine-tuned | Not run | Needs checkpoint download |
| RNAstructure | Not run | Needs isolated venv |
