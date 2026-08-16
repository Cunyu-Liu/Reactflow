# C1-3 Progress Update (2026-07-26)

## Current Status

**Phase**: C1-3 Foundation Encoder + Full-Scale Static SOTA Training
**Branch**: `trae/c1-3-static-scale`
**Training**: FSDP 3x A100, step 25700/epoch0, loss=6.78, ETA ~13 days

## Completed Work

### 1. Fusion Integration Fix
- **Problem**: `single_only` and `pair_feature` configs produced identical BPP outputs
- **Root cause**: Pair features were not being used in the pair initialization pathway
- **Fix**: Added `frozen_pair_fusion` flag and `frozen_opm` module in `StaticPairFormer`
- **Verification**: `tests/c1_3/test_fusion_fix.py` confirms different BPP outputs for different fusion strategies

### 2. Full-Scale Training Launch
- **Config**: `configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_ddp.yaml`
- **Architecture**: StaticPairFormer with PairFeatureAdapter (pair-aware fusion + OPM)
- **Data**: 228K samples, 7 curriculum stages, 20% replay ratio
- **Training**: FSDP on 3x A100 (GPU 0/2/5), batch_size=4, max_len=512, bf16
- **Checkpoint**: FULL_STATE_DICT (portable, rank0-only save)
- **Process**: PID 3018466, running 26+ hours, step 25700, loss converging (575 -> 6.78)

### 3. Evaluation Optimization
- **Decoders**: `threshold` + `mea` (skipped `nussinov_dp` for O(L^3) speedup)
- **Scripts**: `eval_checkpoint.py`, `eval_and_generate_grid.py`
- **FSDP handling**: Prefix stripping + non-strict load fallback
- **Auto-evaluation**: Monitor PID 3290892 auto-evaluates after each epoch checkpoint

### 4. Baseline Installation (6/6 Complete)

| Model | in_clan F1 | novel_clan F1 | Runtime |
|-------|-----------|--------------|---------|
| EternaFold | 0.7039 | 0.7036 | ~2h |
| MXfold2 | 0.6872 | 0.6865 | ~1h |
| ViennaRNA | 0.6819 | 0.6822 | ~0.5h |
| UFold | 0.4007 | 0.4009 | ~7h |
| RNAformer | 0.3939 | 0.3907 | ~10h |
| eFold | 0.2192 | 0.2125 | ~6h |

### 5. Calibration/Legality/Runtime Reports
- **Script**: `scripts/generate_c1_3_reports.py`
- **Metrics**: ECE, Brier score, reliability diagram, legal rate, inference time, peak memory
- **Watcher**: PID 3555886 auto-generates reports after evaluation completes

### 6. Multi-Seed Pipeline
- **Script**: `scripts/launch_multiseed.sh`
- **Config**: 10 seeds, 50K samples, 1 epoch each (~4h per seed)
- **Auto-launch**: PID 2707930 triggers when F1 > ViennaRNA (0.682)
- **Significance**: `scripts/generate_significance_report.py` (paired t-test, p < 0.05)

### 7. Gate Audit
- **Script**: `scripts/audit_c1_3_gate.py`
- **Status**: 12/19 checks passed, 7 pending (all due to model not trained)
- **Gate criteria**: F1 > ViennaRNA, 10-seed significance, calibration/legality/runtime reports

## Automated Pipeline

```
Training (PID 3018466)
  ↓ epoch checkpoint appears
Monitor (PID 3290892)
  ↓ runs eval_and_generate_grid.py
Reports Watcher (PID 3555886)
  ↓ runs generate_c1_3_reports.py + audit_c1_3_gate.py
Multi-seed Pipeline (PID 2707930)
  ↓ if F1 > 0.682, launches 10-seed verification
  ↓ runs generate_significance_report.py
```

## Files Added/Modified

### Source Code
- `src/reactflow/models/static_pairformer.py` (modified: fusion fix, gradient checkpointing)
- `src/reactflow/training_engine.py` (modified: FSDP support, FULL_STATE_DICT)

### Scripts
- `scripts/train_c1_3.py` (modified: FSDP, BatchPairFormer, curriculum)
- `scripts/eval_checkpoint.py` (new: checkpoint evaluation with FSDP handling)
- `scripts/eval_and_generate_grid.py` (new: grid evaluation + results aggregation)
- `scripts/generate_c1_3_reports.py` (new: calibration/legality/runtime reports)
- `scripts/audit_c1_3_gate.py` (new: 19-check gate audit)
- `scripts/launch_multiseed.sh` (new: 10-seed training launcher)
- `scripts/generate_multiseed_results.py` (new: multi-seed aggregation)
- `scripts/generate_significance_report.py` (new: paired t-test)
- `scripts/merge_baseline_results.py` (new: baseline results merger)
- `scripts/run_eternafold_baseline.py` (new: EternaFold baseline)
- `scripts/run_mxfold2_baseline_v2.py` (new: MXfold2 baseline)
- `scripts/run_ufold_baseline_v2.py` (new: UFold baseline with CPU multiprocessing)
- `scripts/run_rnaformer_baseline.py` (new: RNAformer baseline)
- `scripts/merge_efold_predictions.py` (new: eFold prediction merger)

### Configs
- `configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_ddp.yaml` (FSDP training)
- `configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_multiseed*.yaml` (multi-seed variants)

### Tests
- `tests/c1_3/test_fusion_fix.py` (fusion verification)

## Next Steps

1. Wait for epoch 0 checkpoint (~15h from 2026-07-26 09:43)
2. Auto-evaluate model F1 on in_clan + novel_clan
3. If F1 > ViennaRNA: launch 10-seed verification
4. Generate calibration/legality/runtime reports
5. Run gate audit (target: 19/19 PASS)
6. **Do NOT declare route complete** — C1-3 gate must pass before C1-4
