# ReactFlow-Delta CallerV5 Prototype Validation (epoch 21 proposal)

- date: 2026-08-11
- status: PROTOTYPE_VALIDATED / RECOMMENDATION_ADD_REPLICATES
- authority epoch: 20 (current) -> 21 (proposed)
- endpoint: endpoint_v6 (current) -> endpoint_v7 (proposed)
- caller: CallerV5 (continuous measurement-error / abstention, prototype only)
- script: `scripts/reactflow_delta/prototype_caller_v5_stability_v1.py`
- output: `.../benchmark_v3/phase2_learnability_20260810/prototype_caller_v5_stability_v1.json`

## 1. Purpose

Validate whether the proposed CallerV5 continuous + measurement-error-abstention label
resolves the `CALLER_ENDPOINT_UNSTABLE` blocker (epoch 20 learnability gate FAIL, STRICT vs
TRANSDUCTIVE full-pool binary label flip 53.4%). Prototype is read-only; no frozen endpoint
changed, no training, no confirmatory outcome.

## 2. Design (pre-registered, no tuning)

- `p_changer = 1 - null_exceedance` in [0,1] (higher = likely changer).
- Measurement-error abstention: for held pairs in STRICT mode, sigma is unknown; model it as
  `train_global_median * m`, where `m` is drawn (with replacement) from the distribution of
  `per-group-median-sigma / train_global_median` across TRAIN groups (STRICT-legal, train-only).
  Recompute `p_changer` under each of `n_draws=15` draws. If the 95% CI of `p_changer`
  straddles the decision boundary 0.5, abstain (label not robust to sigma uncertainty).
- This directly reproduces the sigma-induced flip mechanism behind the 53.4% binary flip.

## 3. Result (prototype on development pool, 6222 pairs / 13 pubs)

| Metric | Observed | Proposed gate |
|---|---|---|
| called coverage (overall) | **0.240** | >= 0.70 |
| publication-block Spearman of p_changer (STRICT vs TRANSDUCTIVE, called set) | **0.011** | >= 0.80 |
| binary flip rate on called set | 0.035 | n/a (low, but only 24% of pairs are callable) |
| per-pub call coverage | 0.000–0.803, most < 0.50 | >= 0.50 |

Breakdown:
- `n_pairs_total = 6222`, `n_called = 1496`, `n_abstained = 4726` (reliability 995, sigma 3731).
- 76% of pairs abstained due to sigma uncertainty; only 24% callable.
- Spearman ≈ 0 on the called set: even restricting to the called subset, STRICT and
  TRANSDUCTIVE scores are virtually uncorrelated.
- 8/13 publications degenerate (all-called pairs labeled changers under STRICT).

## 4. Conclusion

The continuous + abstention model does **not** resolve the sigma-identifiability problem:

1. Abstention cannot rescue the label: it abstains on 76% of pairs (the decision boundary is
   straddled for most pairs precisely because sigma is unidentifiable), while the remaining
   called set still shows near-zero STRICT/TRANSDUCTIVE score correlation (Spearman 0.011).
2. The root cause is a fundamental sigma-identifiability issue, not a threshold/binary vs
   continuous encoding issue. The single train-global median sigma is off by 15–71× from held
   per-position scatter (see `caller_stability_diagnosis_v1.json`), so no sigma-perturbation
   scheme built only on train groups can make the score stable for held publications.
3. Abstention is not a silent exclusion: abstained counts are reported per publication, but
   overall coverage falls to 0.24, far below the proposed >= 0.70 gate.

## 5. Recommendation (fallback, per endpoint_v6 diagnostics)

Per the diagnosis doc §3.4, if the continuous label is still unstable the fallback is **added
replicates (data acquisition)**, not relaxing the gate. This prototype confirms the fallback
must be triggered: we should **not** proceed to implement CallerV5 as the sole remedy.

Concrete next steps for the owner:
1. Do **not** implement CallerV5 as the primary fix — the prototype shows it cannot resolve
   sigma-identifiability with the current per-publication replicate structure.
2. **Acquire additional WT replicates** for held publications (or the publications named in
   `caller_stability_diagnosis_v1.json` with the worst sigma ratios) so that STRICT-mode
   per-position sigma can be estimated from train groups with adequate coverage, removing the
   dominant-source of label instability.
3. Re-run the Phase 2 learnability gate only after the caller label is stable OR after adding
   independent, provenance-confirmed publications to raise the non-degenerate publication N
   (currently 3/13, too few for a pre-registered inference).

## 6. Evidence / artifacts

- Script: `scripts/reactflow_delta/prototype_caller_v5_stability_v1.py`
  (sha256 `88334bc77a728793ebcac01d9de01e44b2896dcdb634b9d95761762723a979b8`)
- Prototype output: `.../phase2_learnability_20260810/prototype_caller_v5_stability_v1.json`
- Prototype log: `.../phase2_learnability_20260810/prototype_caller_v5_stability_v1.log`
- Root-cause diagnosis: `.../phase2_learnability_20260810/caller_stability_diagnosis_v1.json`
- Proposal doc: `docs/audits/reactflow_delta_caller_stability_diagnosis_and_ep_v7_proposal_20260811.md`

Fail-closed: epoch 21 / CallerV5 implementation is **not** recommended unconditionally; only
the added-replicates path is consistent with the evidence.