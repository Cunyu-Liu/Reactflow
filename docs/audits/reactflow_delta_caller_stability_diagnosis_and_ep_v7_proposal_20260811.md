# ReactFlow-Delta Caller Stability Diagnosis + endpoint_v7 Governance Proposal

- date: 2026-08-11
- status: DIAGNOSIS_CONFIRMED / PROPOSAL_PENDING_OWNER_APPROVAL
- authority epoch: 20 (current) -> 21 (proposed, NEW)
- endpoint: endpoint_v6 (current) -> endpoint_v7 (proposed, NEW)
- caller: CallerV4 (current) -> CallerV5 (proposed, continuous measurement-error / abstention)

## 1. Purpose

Phase 2 learnability declared `METHOD_ROUTE_STOP_LEARNABILITY_NOT_ESTABLISHED` with the
primary blocker `CALLER_ENDPOINT_UNSTABLE` (STRICT vs TRANSDUCTIVE full-pool label flip 53.4%,
per-pub up to 100%). Per endpoint_v6 `caller_stability_gate.if_fail`, the remedy must be a
**continuous measurement-error / abstention model or added replicates** — never threshold
tuning. Owner chose to prioritize Caller-stability governance. This document records the
root-cause diagnosis (confirmed with evidence) and a pre-registered design for a new
authority epoch 21 / endpoint_v7 / CallerV5. No frozen endpoint is modified in place.

## 2. Root-cause diagnosis (CONFIRMED_FACT)

Reading `caller_v4.py`, the ONLY difference between STRICT and TRANSDUCTIVE is the sigma
source:

- STRICT  : held pairs (publication absent from train-only sigma map) fall back to the
  **single train-global median sigma** constant (`self._train_median_sigma`).
- TRANSDUCTIVE: held groups condition on their **own per-position empirical scatter**
  (`add_held_wt_replicates` -> `_empirical_scatter`).

Because `z_i = (mut_i - wt_i) / (sqrt(2) * sigma_i)`, sigma collapse changes z and flips labels.

Evidence (`scripts/reactflow_delta/caller_stability_diagnosis_v1.py`,
`.../phase2_learnability_20260810/caller_stability_diagnosis_v1.json`):

| held_pub | train_med σ | median σ ratio (train_med / held) | frac positions outside 2× band |
|---|---|---|---|
| pmid_24469816 | 0.0394 | 0.068 (held σ ~15× larger) | 0.920 |
| pmid_25183835 | 0.0394 | 0.043 (~23× larger) | 0.986 |
| pmid_32616928 | 0.0409 | 0.014 (~71× larger) | 0.995 |
| pmid_25883046 | 0.0482 | 2.251 (held σ ~2.3× smaller) | 0.706 |
| pmid_35982307 | 0.0440 | 5.2e14 (held σ ≈ 0) | 1.000 |
| pmid_29446752 | 0.0375 | 0.920 | 0.735 |
| pmid_22109276 | 0.0424 | 1.129 | 0.845 |
| pmid_29806027 | 0.0424 | 1.129 | 0.845 |

`frac_outside_2x` = fraction of eligible held positions where the train-median constant is
outside a 2× band of the held per-position sigma. It is 0.70–1.00 for every publication with
held groups. The single constant sigma (~0.02–0.05) is structurally unrepresentative of held
per-position scatter (off by 15–71× in either direction). This is the mechanistic driver of
the 53.4% label flip.

Conclusion: the binary CallerV4 label is not a stable target under the STRICT
information-permission boundary. The boundary itself is correct (it prevents held-out pooled
scatter leakage), but forcing a single constant sigma proxy makes the binary label brittle.

## 3. Proposed governance: endpoint_v7 + CallerV5 (continuous / abstention label)

Design intent: replace the hard binary changer label (which must pick a sigma and a z
threshold) with a **continuous, uncertainty-aware label plus explicit abstention**, so the
label is stable with respect to the sigma source and never forces a fragile 0/1.

### 3.1 CallerV5 output (new label schema)

For each pair, CallerV5 returns a **continuous reliability-weighted changer score** in
[0,1] and an **abstention flag**:

- `p_changer` = posterior probability that the pair is a changer under a measurement-error
  model that marginalizes over the sigma uncertainty (rather than a point z decision).
- `abstain` = True when the measurement error is too large to separate signal from noise
  (e.g. effective information / effective sample size below a pre-registered floor), i.e.
  the pair is not confidently 0 or 1.
- primary task uses `p_changer` on the **called (non-abstained)** subset for AUPRC; the
  abstained subset is reported separately (never zero-filled).

This removes the single-sigma brittleness: instead of a hard z>alpha threshold under one
sigma estimate, the score integrates over sigma and abstains when the decision is unstable.

### 3.2 New stability gate (pre-registered, not tunable post-hoc)

- score stability: publication-block rank correlation of `p_changer` under (a) train-only
  sigma (STRICT) vs (b) held-group sigma (TRANSDUCTIVE) must be >= 0.80 (vs the current
  binary-label flip which is meaningless for a continuous score).
- abstention coverage: called fraction per publication >= 0.50; overall >= 0.70.
- abstention must not be a silent exclusion: per-publication abstained counts reported.
- calibration: reliability curve of `p_changer` within ±0.10 on the called set.

### 3.3 Endpoint_v7 change-control (requires new epoch, in place of v6)

Per endpoint change-control rule, this unit/label/score change MUST be a new version. This
proposal does NOT modify endpoint_v6 or caller_v4 in place.

### 3.4 Discipline / fail-closed

- No threshold tuning after observing results; thresholds above are pre-registered.
- No candidtate-architecture training; baseline learnability only.
- If the continuous label is still unstable, the fallback is added replicates (data
  acquisition) — not relaxing the gate.

## 4. Required approval

Implementing endpoint_v7 / CallerV5 requires a **new authority epoch 21** amendment +
approval (owner signature), because it changes the label/score/schema. This document is the
pre-registered design for that approval.

## 5. Evidence / artifacts

- Diagnostic: `scripts/reactflow_delta/caller_stability_diagnosis_v1.py`
- Diagnostic output: `.../phase2_learnability_20260810/caller_stability_diagnosis_v1.json`
- This document: `docs/audits/reactflow_delta_caller_stability_diagnosis_and_ep_v7_proposal_20260811.md`