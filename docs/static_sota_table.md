# Static RNA Secondary Structure Prediction - SOTA Comparison Table

**Protocol**: same_split_local (MMseqs-disjoint, C1-1 frozen splits)
**Date**: 2026-07-25 07:43
**Phase**: C1-3 (full-scale 19-epoch training in progress)

## Same-Split Baseline Results

| Model | Type | in_clan F1 | in_clan MCC | novel_clan F1 | novel_clan MCC | Status |
|-------|------|------------|-------------|---------------|----------------|--------|
| eternafold | DL (thermo+DL) | 0.7039 | 0.7128 | 0.7036 | 0.7127 | ok |
| mxfold2 | DL (thermo+DL) | 0.6872 | 0.6971 | 0.6865 | 0.6969 | ok |
| viennarna | Thermodynamic | 0.6819 | 0.6936 | 0.6822 | 0.6940 | ok |
| ufold | DL (U-Net) | 0.4007 | 0.4080 | 0.4009 | 0.4085 | ok |
| rnaformer | DL (Transformer) | 0.3939 | 0.4029 | 0.3907 | 0.3999 | ok |
| efold | DL (LSTM) | 0.2192 | 0.2240 | 0.2125 | 0.2173 | ok |

## ReactFlow Model (19-epoch training in progress)

| Config | Status |
|--------|--------|
| pairformer_ribonanza_frozen_small_pair_fsdp_seed0 | Epoch 0/19, step 200/19023, loss=305.96, ETA ~16h for epoch 0 |

## Notes

- All baselines evaluated on full same-split (16,606 in_clan + 46,147 novel_clan)
- ReactFlow: 228K samples, 19 epochs, FSDP on 3x A100, batch_size=4, max_len=512
- Curriculum: 7 stages (short_nested_ncRNA=3, mixed_rfam=5, pri_miRNA=2, pdb=3, viral=2, human_mRNA=2, lncRNA=2)
- Gate: model F1 > ViennaRNA (0.6819 in_clan, 0.6822 novel_clan)
- Multi-seed (10 seeds) pending main training completion; will use 50K samples for faster verification
- Monitor PID 3075119 auto-evaluates after each epoch checkpoint
