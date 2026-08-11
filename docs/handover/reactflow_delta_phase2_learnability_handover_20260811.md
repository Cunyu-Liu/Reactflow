# ReactFlow-Delta Phase 2 (Learnability) Handover

- date: 2026-08-11
- authority epoch: 20 (PHASE2_AUTHORIZE_LEARNABILITY)
- run_id: phase2_baselines_v6_20260809
- gate verdict: **FAIL -> METHOD_ROUTE_STOP_LEARNABILITY_NOT_ESTABLISHED**
- caller stability gate: **FAIL -> CALLER_ENDPOINT_UNSTABLE**
- status: `STOPPED_AT_OWNER_REVIEW` (awaiting owner route decision / Phase 5 token)

## Verdict

Batch 2A (frozen strong baselines) and Batch 2B (learnability analysis) were executed on the
endpoint_v6 / CallerV4 / split_v3 development pool (6385 pairs, 13 resolved publications,
leave-one-publication-out CV, keyed predictions, GPU-confirmed). Cross-real-publication
learnability is **NOT established**:

- **Caller label unstable**: STRICT vs TRANSDUCTIVE full-pool label flip = 53.4%
  (pre-registered max 10%); per-publication flips up to 100%. -> `CALLER_ENDPOINT_UNSTABLE`.
- **Primary task**: only 3/13 publications non-degenerate; all permutation p = 1.0; the only
  positive CI (deepsets, lower=0.0225) collapses to UNIDENTIFIABLE when the dominant
  publication (pmid_29446752) is excluded.
- **Secondary conditional magnitude**: all models' publication-block CIs span zero; perm p = 1.0.

## Scope executed

- `run_baselines_v6.py` — 13-fold nested LOPO, keyed predictions (136738 rows), manifest, GPU.
- `run_learnability_gate_v1.py` — pooled/pub-macro metrics, per-pub effects, dominant-PUB LOO,
  caller-mode sensitivity, null space / min p, coverage, primary + magnitude gate report.
- Unit tests: `tests/reactflow_delta/test_baselines_v6.py`.
- Configs: `configs/reactflow_delta/baselines_v6.yaml`, `endpoint_v6.yaml`.

## Artifacts (read-only, under benchmark_v3)

- `phase2_baselines_v6_20260809/` — keyed_predictions.jsonl, baselines_v6_manifest.json,
  fold_progress.json, run.log.
- `phase2_learnability_20260810/` — learnability_gate_v1.json, run.log.
- `docs/audits/reactflow_delta_phase2_learnability_gate_verdict_20260811.md` — adjudication.
- `docs/handover/reactflow_delta_phase2_learnability_handover_20260811.md` — this document.
- `artifacts/benchmark_v3/phase2_learnability_20260810/phase2_learnability_gate_verdict_v1.json` — machine-readable verdict.

## Constraints respected

- No confirmatory outcome opened; development-only; fail-closed.
- No candidate/dedicated architecture trained.
- No CPU fallback for neural baselines (GPU confirmed, A100 MIG / cuda_available=true).
- No post-hoc endpoint change or threshold tuning to force a PASS.

## Next (requires owner token)

Recommended route: **Phase 5 benchmark / resource / measurement-identifiability**.
Blockers to resolve before any Phase 3 method modeling:
1. Caller label instability (STRICT vs TRANSDUCTIVE 53% flip) — continuous
   measurement-error / abstention model or added replicates, not threshold tuning.
2. Insufficient non-degenerate publication N — add independent, provenance-confirmed
   publications before confirmatory inference.
3. Calibration/coverage on a stable label before any learned baseline is re-authoritative.

Owner decision required: `AUTHORIZE_PHASE5_6_PUBLICATION_RELEASE` (or a directive to pursue
the Phase-5 negative-result / resource-identifiability paper route).

Status: STOPPED_AT_OWNER_REVIEW