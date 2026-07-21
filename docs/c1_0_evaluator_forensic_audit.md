# C1-0: ReactFlow Evaluator Forensic Audit

**Phase:** C1-0
**Date:** 2026-07-21
**Auditor:** Trae AI agent (branch `trae/c1-0-evaluator-audit`)
**Repository:** `/home/cunyuliu/reactflow_c1_0_stage_20260721`
**Status:** COMPLETE — Gate verdict: **PASS** (evaluator exonerated; root cause is the model)

---

## Executive Summary

The C0 acceptance gate failed with calibrated_marginal mean F1 = 0.026, well below
the 0.10 threshold. This audit was commissioned to determine whether the low F1
originates from the **evaluator**, the **labels**, the **decoding**, or the
**evaluation protocol**.

**Conclusion:** The evaluator is exonerated. The F1 gap is caused by the
**ReactFlow model / inference wrapper**, not by any scoring, indexing, or
protocol issue. Specifically:

1. **All 13 gold fixtures pass** — the scorers compute TP/FP/FN/TN/F1/MCC correctly
   on every hand-verified edge case (Gate criterion 1: PASS).
2. **eFold and ReactFlow evaluators agree** on all 12 non-empty test cases and on
   all 200 real-data four-quadrant samples. The only difference is the
   empty-vs-empty convention (eFold=1.0, ReactFlow=0.0), which is documented
   and non-blocking (Gate criterion 2: PASS).
3. **eFold's official checkpoint produces F1=0.89** on the PDB tier, confirming
   the evaluation pipeline produces reasonable scores for a good model
   (Gate criterion 3: PASS).
4. **No unexplained differences** in index convention, diagonal handling, length
   mismatch, threshold, or pseudoknot scoring (Gate criterion 4: PASS).

**Root cause:** The ReactFlow `calibrated_marginal` decoder produces structures
that achieve F1=0.027 on the same data where eFold achieves F1=0.89. Since
swapping the evaluator does not change either model's score, the problem is in
the model, the CTMC ensemble, the calibration, or the checkpoint — not the
evaluator.

---

## 1. Data Flow Diagram

### ReactFlow inference → evaluation pipeline

```
PairwiseDenoiser (partner-class, K=L+1)
    │
    ▼
marginal_pair_matrix: P_ij = 0.5 * (pi_i[j+1] + pi_j[i+1])
    │
    ▼
CTMC sampling (num_steps=16, num_samples=8)
    │
    ▼
pair_frequency_matrix → pair_vs_unpaired_log_odds
    │  score_ij = log((pair + eps) / sqrt(p_unpaired_i * p_unpaired_j)) / T
    ▼
project_max_weight_nested (Nussinov DP, min_score=0.0, min_loop=3)
    │
    ▼
binary structure matrix (L×L, symmetric, {0,1})
    │
    ▼
c0_evaluate.structure_record_metrics
    ├── pair_confusion (upper-triangle, >0.5 threshold)
    ├── shifted_pair_counts (tolerance=1, greedy 1-to-1)
    ├── pair_confusion_by_distance (short/medium/long bins)
    └── aggregate_structure_records (micro/macro F1)
```

### eFold baseline → evaluation pipeline

```
eFold CLI (--basepair mode)
    │
    ▼
dot-bracket string OR pair list (1-based)
    │
    ▼
evaluate_external_baseline_predictions.py
    ├── _normalize_pair (1-based → 0-based, reject self-pairs)
    ├── _pair_confusion (set-based, frozenset operations)
    └── evaluate_tier (per-tier micro/macro F1)
```

### Key observation

Both pipelines converge on the **same scoring formula** (2*TP / (2*TP + FP + FN))
and the **same index convention** (0-based, upper-triangle). The matrix-based
scorer (`reactflow.metrics.pair_confusion`) and the set-based scorer
(`evaluate_external_baseline_predictions._pair_confusion`) produce identical
TP/FP/FN/TN on all test cases.

---

## 2. Gold Fixtures (Task 2)

**Artifact:** `artifacts/c1_0/evaluator_fixture_results.json`
**Test file:** `tests/test_evaluator_gold_fixtures.py`
**Result:** 13/13 fixtures PASS (14/14 tests including registration smoke test)

| # | Fixture | TP | FP | FN | TN | F1 | Notes |
|---|---------|----|----|----|----|----|-------|
| 01 | All-unpaired (empty) | 0 | 0 | 0 | 6 | 0.0 | ReactFlow convention (eFold=1.0) |
| 02 | Single hairpin perfect | 2 | 0 | 0 | 26 | 1.0 | |
| 03 | Two stems nested partial | 3 | 0 | 1 | 227 | 0.857 | |
| 04 | GU wobble pair | 2 | 0 | 0 | 26 | 1.0 | allow_wobble=True/False tested |
| 05 | Pseudoknot crossing | 2 | 0 | 0 | 229 | 1.0 | nested rejects, greedy accepts |
| 06 | Illegal diagonal pair | — | — | — | — | — | ValueError raised |
| 07 | 1-based vs 0-based | — | — | — | — | — | Normalization correct |
| 08 | Relaxed vs exact | 0/1 | 1/0 | 1/0 | — | 0.0/1.0 | Shifted F1 = 1.0 |
| 09 | Empty pred, nonempty target | 0 | 0 | 2 | 26 | 0.0 | |
| 10 | Nonempty pred, empty target | 0 | 2 | 0 | 26 | 0.0 | |
| 11 | Length mismatch | — | — | — | — | — | ValueError raised |
| 12 | Distance bins | 4 | 0 | 0 | — | 1.0 | short=2, medium=2, long=0 |
| 13 | Matrix vs set alignment | 2 | 1 | 1 | 41 | 0.667 | Both scorers agree |

---

## 3. Dual Evaluator Alignment (Task 3)

**Artifact:** `artifacts/c1_0/efold_dual_alignment.json`
**Script:** `scripts/c1_0_dual_evaluator_alignment.py`
**Commit:** `793e8fcb`

### eFold f1 function

```python
def f1(pred, true, threshold=0.5):
    pred = (pred > threshold).float()
    sum_pair = torch.sum(pred) + torch.sum(true)
    if sum_pair == 0:
        return 1.0                    # <-- empty-vs-empty → 1.0
    return (2 * torch.sum(pred * true) / sum_pair).item()
```

**Note:** The `@mask_and_flatten` decorator is commented out, so eFold's f1
operates on the full (L,L) matrix without UKN masking. This is algebraically
equivalent to ReactFlow's upper-triangle F1 for symmetric matrices with zero
diagonal.

### Results (15 test cases)

| Metric | Count |
|--------|-------|
| Total test cases | 15 |
| All 3 evaluators agree | 12 |
| Empty-vs-empty convention difference | 3 |
| Unexpected differences | **0** |

**Verdict:** The evaluators are aligned. The only difference is the documented
empty-vs-empty convention.

---

## 4. Four-Quadrant Localization (Task 4)

**Artifact:** `artifacts/c1_0/efold_four_quadrant_results.json`
**Script:** `scripts/c1_0_four_quadrant_analysis.py`
**Commit:** `eef07e60`

### Quadrant definitions

| Quadrant | Model | Evaluator |
|----------|-------|-----------|
| A | eFold official | eFold `f1` |
| B | eFold official | ReactFlow `f1_score` |
| C | ReactFlow `calibrated_marginal` | eFold `f1` |
| D | ReactFlow `calibrated_marginal` | ReactFlow `f1_score` |

### Results (100 samples per quadrant)

| Quadrant | Mean F1 | Mean F1 (non-empty) | Agree with counterpart |
|----------|---------|---------------------|------------------------|
| A | 0.890 | 0.889 | 99/100 (1 empty-vs-empty) |
| B | 0.880 | 0.889 | 99/100 (1 empty-vs-empty) |
| C | 0.027 | 0.027 | 100/100 |
| D | 0.027 | 0.027 | 100/100 |

### Key findings

- **A ≈ B** (0.889 vs 0.889 on non-empty): evaluators agree on eFold predictions
- **C ≈ D** (0.027 vs 0.027): evaluators agree on ReactFlow predictions
- **A vs C** (0.889 vs 0.027): the 33x gap is from the **MODEL**, not the evaluator
- **0 unexpected differences** across 200 real-data samples

**Conclusion:** Swapping the evaluator does not change any model's score. The
F1 gap is entirely attributable to the model/inference wrapper.

---

## 5. Data Leakage & Protocol Conflict (Task 5)

**Artifact:** `artifacts/c1_0/data_overlap_audit.json`
**Script:** `scripts/c1_0_data_leakage_audit.py`
**Commit:** `c216599`

### Findings

| Check | Verdict | Details |
|-------|---------|---------|
| Pair index protocol | **PASS** | 0-based, no self-pairs, no out-of-range, all i<j |
| T-to-U conversion | **PASS** | Zero T's found; 524,616 U's across all sequences |
| Truncation/padding | **WARNING** | 5,314 records at fixed lengths (intentional 256nt windowing); zero pair-overflow; zero window pair-drift |
| Sequence overlap (eFold train vs human_mRNA) | **FAIL** | 6,605/6,627 (99.7%) of human_mRNA is in eFold training set |
| Window parent overlap (exact split) | **FAIL** | 4,142 parent RNAs in both train and test |
| Window parent overlap (mmseqs split) | **PASS** | 14 parent RNAs in both train and test |

### Critical issues

1. **human_mRNA test set is contaminated**: 99.7% of human_mRNA sequences appear
   in eFold's training data. eFold's metrics on human_mRNA are train-set metrics,
   not generalization metrics. **Action:** Exclude human_mRNA from all
   eFold-vs-ReactFlow comparisons (locked in `static_v1.yaml`).

2. **exact split has severe parent leakage**: 4,142 parent RNAs contribute windows
   to both train and test. The `exact` clustering only deduplicates exact sequence
   strings, not parent sequences. **Action:** Use `mmseqs` split for all honest
   evaluation (locked in `static_v1.yaml`).

3. **efold_train is the union of ReactFlow splits**: This is by construction
   (ReactFlow reuses eFold's Dryad data). Cross-framework comparisons must account
   for this shared data origin.

---

## 6. Evaluator Contract v1 (Task 6)

**Artifact:** `configs/evaluation/static_v1.yaml`
**Test file:** `tests/test_evaluator_contract.py`
**Commit:** `701c2fd`

The frozen evaluator contract defines:

| Parameter | Value |
|-----------|-------|
| Pair types | Canonical (AU, UA, GC, CG) + Wobble (GU, UG) |
| allow_wobble | true |
| allow_pseudoknot | false (default decoder: nested DP) |
| min_loop | 3 |
| Index base | 0 |
| F1 formula | 2*TP / (2*TP + FP + FN) |
| F1 undefined | 0.0 (ReactFlow convention) |
| Shifted F1 tolerance | 1 |
| Threshold (matrix) | 0.5 |
| Threshold (decoder) | 0.0 (log-odds min_score) |
| Primary aggregation | micro_f1 |
| Secondary aggregation | macro_f1 |
| Distance bins | short (1-11), medium (12-23), long (24+) |
| Empty-vs-empty | F1=0.0 (ReactFlow; eFold=1.0 documented) |
| Default decoder | calibrated_marginal + nested_dp |
| Preferred split | mmseqs (14 parent overlaps vs 4142 for exact) |
| Excluded tiers | human_mRNA (99.7% train contamination) |

15 structural validation tests verify the YAML is well-formed and complete.

---

## 7. Gate Judgment

### Gate criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| 1. All fixtures 100% pass | **PASS** | 13/13 fixtures pass (evaluator_fixture_results.json) |
| 2. Evaluator differences eliminated or explained | **PASS** | Only empty-vs-empty convention difference, documented in static_v1.yaml |
| 3. eFold official checkpoint produces reasonable results | **PASS** | F1=0.89 on PDB tier (four_quadrant_results.json) |
| 4. No unexplained index/diagonal/length/threshold/pseudoknot differences | **PASS** | 0 unexpected differences across 200 real-data samples |

### Overall gate verdict: **PASS**

The evaluator is correct, aligned, and frozen. The C0 F1=0.026 is NOT caused by
an evaluator bug. The root cause is the ReactFlow model/inference wrapper.

### Root cause analysis

The four-quadrant analysis definitively localizes the F1 gap:

- **eFold model → any evaluator**: F1 ≈ 0.89 (good)
- **ReactFlow model → any evaluator**: F1 ≈ 0.027 (bad)
- **Evaluator swap effect**: ≈ 0 (no change)

The problem is in one or more of:
1. The `PairwiseDenoiser` model parameters (checkpoint quality)
2. The CTMC sampling and ensemble aggregation
3. The `pair_vs_unpaired_log_odds` calibration
4. The Nussinov DP decoder (`min_score=0.0` may be too permissive or too strict)
5. The frozen feature adapter (if features are uninformative)

### Before-after metrics

| Metric | Before (C0) | After (C1-0 audit) | Change |
|--------|-------------|---------------------|--------|
| Evaluator correctness | Unverified | 13/13 fixtures pass | Verified |
| eFold-ReactFlow alignment | Unknown | 0 unexpected diffs | Aligned |
| Root cause confidence | Unknown | Model (not evaluator) | Localized |
| Data leakage awareness | Unknown | 2 critical issues found | Documented |
| Protocol frozen | No | static_v1.yaml | Frozen |

### Unresolved differences

1. **Empty-vs-empty convention** (eFold=1.0, ReactFlow=0.0): Documented and
   non-blocking. The contract adopts the ReactFlow convention. This affects
   macro F1 on sequences with no pairs but does not explain the 0.027 vs 0.89 gap.

2. **Float32 vs float64 precision** (eFold uses torch float32, ReactFlow uses
   Python float64): Differences at the 1e-7 level, well within tolerance. Not
   a concern.

### Recommendations for next phase (C1-1+)

1. **Use the mmseqs split** for all future evaluation. The exact split has
   severe parent leakage (4,142 parents in both train and test).

2. **Exclude human_mRNA** from eFold-vs-ReactFlow comparisons. 99.7% of
   human_mRNA is in eFold's training set.

3. **Investigate the model, not the evaluator.** The four-quadrant analysis
   shows the evaluator is correct. Focus on:
   - Checkpoint quality (is the model trained to convergence?)
   - CTMC ensemble diversity (are the 8 samples diverse enough?)
   - Log-odds calibration (is `pair_vs_unpaired_log_odds` producing good scores?)
   - Nussinov decoder threshold (is `min_score=0.0` appropriate?)

4. **Run the symmetric pair trunk + structured decoder prototype** (the C0
   next-goal) using the frozen `static_v1.yaml` contract for evaluation.

5. **Re-run the C0 evaluation with the mmseqs split** to get an honest baseline
   F1 for the ReactFlow model.

---

## Artifacts Summary

| Artifact | Path | Status |
|----------|------|--------|
| Gold fixture results | `artifacts/c1_0/evaluator_fixture_results.json` | Generated |
| Dual alignment results | `artifacts/c1_0/efold_dual_alignment.json` | Generated |
| Four-quadrant results | `artifacts/c1_0/efold_four_quadrant_results.json` | Generated |
| Data overlap audit | `artifacts/c1_0/data_overlap_audit.json` | Generated |
| Evaluator contract | `configs/evaluation/static_v1.yaml` | Frozen |
| Gold fixture tests | `tests/test_evaluator_gold_fixtures.py` | 14/14 pass |
| Contract tests | `tests/test_evaluator_contract.py` | 15/15 pass |
| Audit document | `docs/c1_0_evaluator_forensic_audit.md` | This file |

## Git History

| Commit | Task | Description |
|--------|------|-------------|
| `47a6b4b` | 1 | Initial C1-0 baseline |
| `500a83f` | 2 | Gold fixtures (13 cases, all pass) + JSON artifact script |
| `793e8fc` | 3 | Dual evaluator alignment (eFold vs ReactFlow) |
| `eef07e6` | 4 | Four-quadrant localization analysis |
| `c216599` | 5 | Data leakage and protocol conflict audit |
| `701c2fd` | 6 | Freeze evaluator contract v1 + tests |
| (this commit) | 7-8 | Audit document + Gate judgment |
