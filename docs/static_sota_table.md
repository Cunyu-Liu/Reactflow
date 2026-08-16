# Static RNA Secondary Structure Prediction - SOTA Comparison Table

**Protocol**: same_split_local (MMseqs-disjoint, C1-1 frozen splits)
**Date**: 2026-07-26 (updated)
**Phase**: C1-3 (full-scale 19-epoch FSDP training in progress)

## Same-Split Baseline Results

| Model | Type | in_clan F1 | in_clan MCC | novel_clan F1 | novel_clan MCC | Status |
|-------|------|------------|-------------|---------------|----------------|--------|
| eternafold | DL (thermo+DL) | 0.7039 | 0.7128 | 0.7036 | 0.7127 | ok |
| mxfold2 | DL (thermo+DL) | 0.6872 | 0.6971 | 0.6865 | 0.6969 | ok |
| viennarna | Thermodynamic | 0.6819 | 0.6936 | 0.6822 | 0.6940 | ok |
| ufold | DL (U-Net) | 0.4007 | 0.4080 | 0.4009 | 0.4085 | ok |
| rnaformer | DL (Transformer) | 0.3939 | 0.4029 | 0.3907 | 0.3999 | ok |
| efold | DL (LSTM) | 0.2192 | 0.2240 | 0.2125 | 0.2173 | ok |

## ReactFlow Model (FSDP training in progress)

| Config | Status |
|--------|--------|
| pairformer_ribonanza_frozen_small_pair_fsdp_seed0 | Epoch 0/19, step 25700, loss=6.78, GPU 0/2/5 100%, ETA ~13 days |

### Training Configuration

- **Architecture**: StaticPairFormer with PairFeatureAdapter (pair-aware fusion + OPM)
- **Data**: 228K samples, 7 curriculum stages, 20% replay ratio
- **Training**: FSDP on 3x A100 (40GB), batch_size=4, max_len=512, bf16, gradient checkpointing
- **Optimizer**: AdamW, lr=1e-4, 19 epochs
- **Checkpoint**: FULL_STATE_DICT (portable, rank0-only save)
- **Evaluation**: threshold + mea decoders (nussinov_dp skipped for speed)

### Automated Pipeline

| Component | PID | Status |
|-----------|-----|--------|
| FSDP training | 3018466 | Running (26h+) |
| 19-epoch monitor | 3290892 | Watching for checkpoint |
| Reports watcher | 3555886 | Waiting for eval completion |
| Multi-seed pipeline | 2707930 | Waiting for F1 > 0.682 |

### Gate Criteria

- Model F1 > ViennaRNA (0.6819 in_clan, 0.6822 novel_clan)
- 10-seed significance (paired t-test, p < 0.05)
- Calibration/legality/runtime reports (scripts ready)
- Gate audit: 12/19 passed (7 pending model training)

## Notes

- All baselines evaluated on full same-split (16,606 in_clan + 46,147 novel_clan)
- Multi-seed (10 seeds, 50K samples, 1 epoch each) will auto-launch when F1 > ViennaRNA
- Calibration/legality/runtime report scripts created (`scripts/generate_c1_3_reports.py`)
- Reports watcher auto-generates reports after evaluation completes
