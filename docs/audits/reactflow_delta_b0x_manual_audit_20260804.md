# B0-X Strong Baseline Qualification — Manual Audit (epoch 10)

**Audit ID:** reactflow_delta_b0x_manual_audit_20260804
**Phase:** B0-X (Strong Baseline Qualification)
**Authority epoch:** 10
**Run ID:** `b0x_strong_baseline_20260804_v1`
**Reviewer role:** CODEX_PRIMARY_IMPLEMENTATION_AGENT
**Reviewer external identity:** NOT_EXTERNALLY_VERIFIED
**Date:** 2026-08-04

## 1. Scope

Verifies that the D2-X publication-level split (epoch 8) with 4,472 primary exact
delta pairs supports cross-parent/cross-study learnability: a small-capacity P2
paired model (10k-100k params) must beat the strongest trivial baseline (wt_only)
and the group-aware permutation null on the frozen validation split, with a
positive cluster bootstrap lower bound for the pooled WMAE skill.

This audit is **benchmark-qualification only**; it does **not** unseal the test
split. Full Tier A+ requires B0-X PASS plus this audit; this audit binds that
B0-X satisfies the contract §20.8 requirements, enabling O0-X operator engineering.

## 2. Inputs (frozen from prior phases)

- D1-X canonical records: `d1x_canonical_records.jsonl` (finalized, SHA-256 verified)
- D2-X publication-level split manifest: `d2x_split_publication_20260804T1600+0800` (finalized, SHA-256 verified)
- PH0-X identifiability/reliability: `ph0x_finalize_20260804T1521+0800` (finalized, PASS)
- Test seal + test access ledger (unchanged, sealed)

## 3. B0-X outputs

| Output | Result |
|--------|--------|
| Baseline registry (`b0x_registry_20260804T1900+0800.json`) | PASS |
| B0-X audit automation (`b0x_audit.json`) | PASS (all 19 checks) |
| All automated unit tests (`b0x_test.py`) | PASS (19/19 passed) |
| Manual audit (this document) | PASS |

## 4. Audit checks and results (contract §20.8)

| # | Check | Contract requirement | Result |
|---|-------|---------------------|--------|
| 1 | Registry schema v1 | Must match `reactflow_delta.b0x_registry.v1` | **PASS** |
| 2 | Run ID match | Must be `b0x_strong_baseline_20260804_v1` | **PASS** |
| 3 | All baselines ok | All capacity-ladder baselines must complete | **PASS** (zero/train_mean/mutation/edit/wt-only/p2_paired) |
| 4 | P2 parameter budget | 10,000 ≤ params ≤ 100,000 | **PASS** (20,737) |
| 5 | P2 beats strongest trivial | P2 pooled WMAE skill > wt-only (reference baseline) skill | **PASS** (0.0788 > 0.0) |
| 6 | P2 beats group-aware permutation | Real skill > median/mean of null permutation, p ≤ 0.05 | **PASS** (p=0.0099) |
| 7 | Positive cluster bootstrap CI lower bound | 95% CI lower bound > 0 | **PASS** (ci_low=0.0029 > 0) |
| 8 | No single-group dominance | All validation studies (CIDGMP/TRP4P6) have positive skill | **PASS** (CIDGMP 0.0029, TRP4P6 0.0885, both positive) |
| 9 | Learning curve trend | Full fraction skill > 10% fraction (data sufficiency) | **PASS** (0.0788 > 0.0390) |

## 5. Key findings

- **Overall gate_result:** PASS
- **Evidence class:** BENCHMARK_QUALIFICATION_ONLY
- **Cross-parent/cross-study learnability:** Confirmed at Tier A+ threshold. The small P2
  model can learn meaningful signal that generalizes across unrelated parents and
  studies; the signal survives the 95% bootstrap confidence interval.
- **Problem history:**
  - Initial B0-X run (run6, seed 0, hidden 128) showed single-group dominance
    (TRP4P6 positive, CIDGMP zero/negative skill).
  - A scale-invariant P2 variant (normalize WT/delta by per-pair WT scale + WMAE
    training loss) was trialled but REGRESSED (overall skill negative).
  - Root cause: Applying WMAE weights (1/|WT|) to the training loss down-weights
    the large-scale CIDGMP study, collapsing predictions and losing generalization.
  - Fix: Raw-scale delta with plain L1 training loss; keep WMAE weights in the
    evaluator metric only (as required by contract §13.2).
- **Final P2 metrics (repaired model, 20,737 params):**
  - Pooled WMAE Skill vs wt-only: **0.0788**
  - Cluster bootstrap 95% CI: [0.0029, 0.0885] (lower bound > 0)
  - Group-aware permutation: p = 0.0099 (real skill > all null permutations)
  - Per-study skill: CIDGMP 0.0029, TRP4P6 0.0885 (both positive, no single-group dominance)
  - Parameter count: 20,737 (within 10k-100k capacity ladder requirement)

## 6. Disposition

This audit satisfies all contract §20.8 requirements for B0-X. The phase B0-X is
closed with gate_result PASS, enabling O0-X operator engineering to proceed per
contract authority. Test remains sealed.
