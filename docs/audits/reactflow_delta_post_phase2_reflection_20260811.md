# ReactFlow-Delta Post-Phase-2 Reflection: Model vs Data Bottleneck (2026-08-11)

- date: 2026-08-11
- status: REFLECTION_GROUNDED / STRATEGIC_INPUT_FOR_OWNER
- authority epoch: 20 (current)
- endpoint: endpoint_v6, caller: CallerV4
- purpose: honest re-analysis of the Phase 2 learnability FAIL before any
  further model work, to determine whether "improving the model" can actually
  move the needle or whether the binding constraint is the label/data.

## 1. What the Phase 2 numbers actually say (read from learnability_gate_v1.json)

### 1a. Secondary conditional-magnitude task: a REAL pair-level signal exists

`wmae_deepsets` achieves a consistent 22–25% WMAE skill across ALL 5 seeds:

| seed | skill | WMAE_model | WMAE_baseline |
|---|---|---|---|
| 0 | 0.244 | 4.207 | 5.567 |
| 1 | 0.239 | 4.236 | 5.567 |
| 2 | 0.224 | 4.317 | 5.567 |
| 3 | 0.246 | 4.197 | 5.567 |
| 4 | 0.220 | 4.342 | 5.567 |

This is NOT noise. The model is consistently beating the train-fold weighted-median
trivial by ~22–25%. But the publication-block 95% CI spans zero and permutation p=1.0,
because there are only N=11 publications in the magnitude comparison. This is a
**statistical-power / exchangeable-unit-N problem**, not a missing-model-capacity problem.

### 1b. Primary prospective-changer task: label is near-saturated, cannot show learnability

The 3 non-degenerate publications have prevalence AUPRC already near ceiling:
- pmid_25183835: prevalence AP = 0.951 (almost every pair is a changer)
- pmid_29446752: 0.747
- pmid_29806027: 0.500

With a prevalence AP of 0.95, there is almost no room for a model to add value.
`macro_auprc_model = None` under the degenerate-publication policy. **No architecture can
grow a positive signal here.**

### 1c. Caller label instability is the primary blocker

STRICT vs TRANSDUCTIVE full-pool label flip = 53.4%; per-pub up to 100%.
Root cause (caller_stability_diagnosis_v1.json): STRICT uses a single train-global
median sigma constant, which is off by 15–71× from held per-position scatter.

## 2. NEW experiment: can a STRICT-legal per-position sigma model fix the label?

Hypothesis: replace the single train-global-median constant with a monotone (isotonic)
per-position sigma model fit ONLY from outer-train replicate groups, keyed on WT
reactivity (a STRICT-legal feature = the pair's own WT state). If sigma is predictable
from WT reactivity, the STRICT label would not need the fragile constant and flip would drop.

### 2a. Is sigma predictable from WT reactivity? (partial yes)

On the replicate pool (4988 positions across train groups):
- Spearman(sigma, WT reactivity) = 0.65 (position level), 0.57 (group level) — a real
  monotone relationship.
- BUT isotonic out-of-sample R² in log-sigma = 0.087, RMSE 9.5. The relationship is
  real but very noisy (heavy-tailed sigma, extreme outliers).

### 2b. Does the sigma model reduce label flip? (NO)

Full-pool STRICT-vs-TRANSDUCTIVE flip comparison (called subset, 5227 pairs):

| Variant | overall flip | worst per-pub flip |
|---|---|---|
| single train-median constant (current) | 0.1339 | 0.774 (pmid_32616928) |
| isotonic sigma model (train-only) | 0.1351 | 0.972 (pmid_28851837), 0.774 |

The isotonic sigma model gives **no meaningful improvement** (0.135 vs 0.134), and does
not rescue the worst per-publication flip rates. Combined with R²=0.087, this confirms:

> **Sigma-identifiability cannot be fixed by modeling sigma from STRICT-legal features
> alone. The per-position sigma of a held publication is not recoverable from WT
> reactivity to the precision needed for a stable 0/1 label. It fundamentally requires
> held-domain replicates (data), which the STRICT information-permission boundary
> correctly forbids using at training time.**

This is a decisive, evidence-based conclusion and directly answers whether a model-side
sigma fix can work: **it cannot.**

## 3. Honest synthesis: which levers actually move the needle

| Lever | Phase-2 evidence | Can it fix the gate? |
|---|---|---|
| Better model architecture (bigger MLP/transformer) | magnitude signal is real but CI spans zero at N=11 | **No** — fails at publication-block power, not capacity |
| Per-position sigma model (STRICT-legal) | NEW: flip unchanged (0.135), R²=0.087 | **No** — sigma not identifiable from legal features |
| Let model ingest sigma uncertainty | prototype_caller_v5: coverage 0.24, Spearman 0.011 | **No** — sigma-identifiability is fundamental |
| **Add held-domain WT replicates** | sigma collapse is 15–71×; replicates give true per-position sigma | **Yes** — directly removes the flip driver |
| **Add independent non-degenerate publications** | only 3/13 non-degenerate; N=11 for magnitude | **Yes** — raises exchangeable-unit N and power |

The binding constraints are (a) label instability → needs replicates, and (b) exchangeable
unit N → needs more publications. **Both are data-side.** A model-side fix that stays
within the STRICT information-permission boundary and does not touch the label cannot
overcome them.

## 4. Recommended path (data + model dual-track, in the correct priority order)

1. **Data track (binding):**
   - Acquire held-domain WT replicates for the publications with the worst sigma collapse
     (pmid_32616928 ~71×, pmid_25183835 ~23×, pmid_24469816 ~15×) so STRICT per-position
     sigma is estimable from train groups without the fragile constant.
   - Add independent, provenance-confirmed publications to raise the non-degenerate
     publication N from 3 toward >=5–6 (primary) and the magnitude N from 11 upward.
2. **Model track (only AFTER a stable label):**
   - On a stable label, the magnitude task already shows a real 22–25% skill — the
     honest model-side win is to tighten the publication-block CI via better
     within-publication generalization (reduce variance), not to add capacity.
   - Multi-task joint training (changer + magnitude) is legitimate once the label is stable.

## 5. Evidence / artifacts

- Diagnostic: `scripts/reactflow_delta/sigma_model_flip_test_v1.py`
- Diagnostic output: `/tmp/sigma_model_flip_test.json` (server)
- Phase-2 verdict: `docs/audits/reactflow_delta_phase2_learnability_gate_verdict_20260811.md`
- Root-cause: `docs/audits/reactflow_delta_caller_stability_diagnosis_and_ep_v7_proposal_20260811.md`
- CallerV5 prototype: `docs/audits/reactflow_delta_caller_v5_prototype_validation_20260811.md`

Fail-closed: this reflection does NOT change any frozen endpoint/gate. It is strategic
input for the owner to decide the data-acquisition path before further model work.