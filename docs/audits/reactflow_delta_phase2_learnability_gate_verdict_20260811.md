# ReactFlow-Delta Phase 2 (Learnability Gate) Verdict

- date: 2026-08-11
- authority epoch: 20 (PHASE2_AUTHORIZE_LEARNABILITY)
- endpoint: endpoint_v6 (RFD_ENDPOINT_V6)
- caller: CallerV4 (STRICT_INDUCTIVE_WT_ALLOWED primary / WT_REPLICATE_CONDITIONED_TRANSDUCTIVE sensitivity)
- run_id: phase2_baselines_v6_20260809
- gate verdict: **FAIL -> METHOD_ROUTE_STOP_LEARNABILITY_NOT_ESTABLISHED**
- caller stability gate: **FAIL -> CALLER_ENDPOINT_UNSTABLE** (primary blocker)
- confirmation status: DEVELOPMENT_ONLY (no confirmatory outcome opened; fail-closed)

## 0. Evidence

| Artifact | Path |
|---|---|
| Keyed predictions (69 MB, 136738 rows) | `/mnt/cunyuliu/.../benchmark_v3/phase2_baselines_v6_20260809/keyed_predictions.jsonl` |
| Baseline manifest | `.../benchmark_v3/phase2_baselines_v6_20260809/baselines_v6_manifest.json` |
| Learnability gate report | `.../benchmark_v3/phase2_learnability_20260810/learnability_gate_v1.json` |
| Baseline run log | `.../phase2_baselines_v6_20260809/run.log` |
| Learnability run log | `.../phase2_learnability_20260810/run.log` |

- `n_pair_recs = 6385`, `n_resolved_publications = 13`, `n_rows = 136738`.
- All 13 LOOCV folds completed (fold_progress.json), GPU A100 MIG (cuda_available=true, device 6).
- keyed_predictions_sha256 = `ace328369a85d5a7659e0de83ad822f983777012f31e5ec9bf737b7ecaaa5858`.

## 1. Caller stability gate (endpoint_v6 pre-registered) -> FAIL

Pre-registered thresholds (endpoint_v6 `caller_stability_gate`, on the **full pool**, 6222 pairs / 13 pubs):

| Gate | Threshold | Observed (full pool) | Verdict |
|---|---|---|---|
| overall_label_flip | <= 0.10 | **0.5344** | FAIL |
| per_publication_flip | <= 0.25 (non-tiny) | up to **1.0** (pmid_20190761/21239468/35982307), 0.78 (22109276), 0.95 (32616928) | FAIL |
| overall_callable_coverage | >= 0.70 | 0.8401 | PASS |
| per_publication_callable | >= 0.50 | (see coverage) | NOT-ESTABLISHED |

Note: the Phase-1 `caller_v4_sensitivity.json` fixture (191 outcome-blind rows) reported label_agreement=1.0 and the stability gate PASS. That was a **small diagnostic fixture subset**, not the full pool. The full-pool sensitivity in `learnability_gate_v1.json` is the authoritative evidence and shows the CallerV4 label is **not stable** between the two pre-registered information-permission modes: 53.4% of pair labels flip between STRICT and TRANSDUCTIVE on the real pool.

Per endpoint_v6 `caller_stability_gate.if_fail = CALLER_ENDPOINT_UNSTABLE; 禁止 Phase 2 learned baseline`. This is the **primary blocker** and means the Phase 2 learned-baseline results are only `DEVELOPMENT_ONLY` evidence.

## 2. Phase 2 PASS criteria (all 7 must hold) -> FAIL

| # | Criteria | Assessment | Verdict |
|---|---|---|---|
| 1 | Caller/endpoint/split/evaluator still PASS | Caller stability gate FAILS on full pool | FAIL |
| 2 | >=1 non-trivial simple/generic baseline, publication-level paired 95% CI lower>0 vs strongest trivial | Only `deepsets:0` has lower=0.0225, but over **n_pub=3** and it **vanishes** (UNIDENTIFIABLE) when dominant pub excluded; permutation p=1.0 | FAIL |
| 3 | Pre-registered alpha reachable / min p | All permutation p=1.0; only 3/13 pubs non-degenerate; dominant-excluded CI=UNIDENTIFIABLE | FAIL |
| 4 | Primary & conditional judged separately; each established | Primary not established; conditional not established (all CIs span zero) | FAIL |
| 5 | Not driven by single dominant publication | deepsets' only positive CI depends entirely on pmid_29446752 (2474 pairs, 38.8% of pool); excluding it -> UNIDENTIFIABLE | FAIL |
| 6 | >=70% dev publications same direction, no pub-specific shortcut | Only 3/13 pubs non-degenerate (rest per-pub AP=None); caller per-pub flip up to 100% indicates label shortcut | FAIL |
| 7 | Coverage & calibration acceptable | coverage 1.0 over emitted rows; caller callable 0.84; calibration not established under unstable labels | NOT-ESTABLISHED |

## 3. Primary task (prospective changer probability) -> NOT ESTABLISHED

Metric: publication-macro AUPRC (over NON-DEGENERATE pubs); paired = per-pub AP delta vs prevalence trivial. seed=0 shown; all seeds perm_p=1.0.

| Model | n_pairs | mean ΔAP | paired 95% CI (n_pub) | perm_p | dominant-LOO CI |
|---|---|---|---|---|---|
| p2_mlp | 5227 | 0.0677 | (−0.054, 0.175) n=3 | 1.0 | UNIDENTIFIABLE |
| deepsets | 5227 | 0.0844 | (0.0225, 0.144) n=3 | 1.0 | UNIDENTIFIABLE |
| gbm | 5227 | −0.0467 | (−0.112, 0.046) n=3 | 1.0 | UNIDENTIFIABLE |
| wlogit | 5227 | −0.0578 | (−0.120, 0.034) n=3 | 1.0 | UNIDENTIFIABLE |
| gam | 5227 | −0.0434 | (−0.102, 0.037) n=3 | 1.0 | UNIDENTIFIABLE |

Only 3 publications (pmid_25183835, pmid_29446752, pmid_29806027) have non-singleton labels; the other ~10 pubs are degenerate (all-changer or all-non-changer) -> per-pub AP = None. `macro_auprc_model = None` (unidentifiable under the degenerate-publication policy). No robust cross-publication learnability signal.

## 4. Secondary task (oracle-conditioned conditional magnitude) -> NOT ESTABLISHED

Metric: conditional WMAE skill (1 − WMAE_model/WMAE_trivial); paired = publication-block bootstrap CI. n_changers = 4540.

| Model | skill | paired 95% CI | perm_p |
|---|---|---|---|
| wmae_mlp | 0.0055 | (−0.0004, 0.0257) | 1.0 |
| wmae_deepsets | 0.2442 | (−0.167, 0.623) | 1.0 |
| wmse_gbm | 0.1784 | (−0.548, 0.461) | 1.0 |
| lad_lm | 0.0836 | (−0.667, 0.422) | 1.0 |
| wgam | 0.1576 | (−0.707, 0.423) | 1.0 |

All CIs span zero; all permutation p=1.0. No significant skill over the train-fold weighted-median trivial on a publication-block basis.

## 5. Null space / power

- `n_exchangeable_publications = 12` (for null-space accounting), `unique_sign_flip_assignments = 4096`, `min_2sided_p = 2^(1-12) = 0.00049`.
- However only **3** publications are non-degenerate for primary paired inference, so the effective null space for the observed skill is ~ 2^3 = 8 assignments and the minimum attainable exact p is ~ 0.25 — far too coarse to establish learnability (`UNIDENTIFIABLE_INSUFFICIENT_PUBLICATIONS`).

## 6. Determination

- **CALLER_ENDPOINT_UNSTABLE** (endpoint-level; primary blocker).
- **METHOD_ROUTE_STOP_LEARNABILITY_NOT_ESTABLISHED** (Phase 2 FAIL).
- All Phase-2 learned-baseline results are `DEVELOPMENT_ONLY`.
- Per plan: permanently stop this round's dedicated architecture development; do NOT reopen via more backbone/seed/position rows or post-hoc endpoint changes.

## 7. Recommended next route (requires owner token)

Transition to **Phase 5: benchmark / resource / measurement-identifiability** route, which is the only route currently authorized to proceed. Before any Phase 3 method modeling can be reconsidered, the open blockers are, in order of priority:

1. Caller label instability (STRICT vs TRANSDUCTIVE 53% flip) — resolve via a continuous measurement-error / abstention model or added replicates, not threshold tuning.
2. Insufficient non-degenerate publication N — add independent, provenance-confirmed publications before any confirmatory inference.
3. Establish calibration/coverage on a stable label before any learned baseline is re-authoritative.

No Phase 3 method token is requested. No confirmatory outcome was opened. Fail-closed.