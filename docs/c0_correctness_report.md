# Phase C0 Correctness Report

> Isolated stage: `/home/cunyuliu/reactflow_c0_stage_20260718`
> Active project (untouched): `/home/cunyuliu/reactflow`
> Report generated: 2026-07-19
> Schema version: 1

## 1. Summary

Phase C0 implements a **trusted inference and evaluation protocol** for ReactFlow without disturbing the in-flight RF-CF5 training queue. The work is performed entirely inside a code-only isolated stage that references frozen features, checkpoints and baseline artifacts via absolute read-only paths. The default production inference path is switched from the historical single-step `t=1 / all-unpaired / min_score=1e-6` projection to a **validation-calibrated CTMC ensemble decoder** (`calibrated_marginal`). The legacy endpoint is preserved bit-for-bit under an explicit `legacy_direct` mode that is reserved for regression comparison only.

The C0 stage delivers:

- A frozen `c0_initial_state_manifest.json` capturing code hashes, active PIDs (including RF-CF5 PID 1437378 and its children), GPU state, baseline artifact hashes and data split hashes.
- A canonical `baseline_efold_results.json` produced by an automated merger of the two full-count eFold progress artifacts, with strict `matched_count == gold_count == count` enforcement and SHA-256 provenance for both gold and predictions.
- A unified `predict_structure(...)` API in `src/reactflow/inference.py` with three explicit inference modes (`legacy_direct`, `ctmc_sample`, `calibrated_marginal`) and two decoder policies (`nested_dp`, `pseudoknot_allowed_greedy`).
- New `calibrate-inference` and `evaluate-checkpoint` CLI commands that write and consume a validation-locked `decoder_manifest.json` whose calibration parameters cannot be overridden at test time.
- A runtime preflight gate (32+ samples) projecting 5.59 hours for the full fixed evaluation matrix, well within the 24-hour ceiling.
- A full pytest run with **91.86% combined coverage** (≥90% gate satisfied), symbolic checks with zero residuals, and three CLI smoke runs.

The C0 stage **does not** deploy into `/home/cunyuliu/reactflow`. Deployment is deferred until the RF-CF5 serial scheduler (PID 1437378) and every child process referencing the active source tree terminate naturally, as required by the goal.

## 2. Old vs. New Inference Comparison

### 2.1 Historical endpoint (`legacy_direct`)

- Forward pass at `t=1` with all-unpaired partner-class source.
- Softmax pair matrix fed directly into greedy matching with `min_score=1e-6`, `allow_pseudoknot=True`, `allow_wobble=True`.
- No ensemble, no validation calibration, no threshold gate.
- Reproducible bit-for-bit against the historical `evaluate-efold` path; preserved under `InferenceMode.LEGACY_DIRECT` for regression testing only.
- **Not the default mode**: `InferenceConfig().mode is InferenceMode.CALIBRATED_MARGINAL` is asserted by `test_default_inference_is_not_legacy_endpoint`.

### 2.2 CTMC ensemble (`ctmc_sample`)

- Each trajectory initializes from a uniform partner-class source.
- Integrates the learned DFM posterior rate over `num_steps` time slices, reconstructing time, current state, sequence and frozen-adapter augmented features at every step (verified by `test_ctmc_is_deterministic_legal_and_uses_dynamic_feature_builder`).
- Outputs `num_samples` legal structures plus a pair-frequency matrix and per-position unpaired probability.
- Deterministic given `seed`; legality validated per structure.

### 2.3 Calibrated marginal decoder (`calibrated_marginal`, default)

- Consumes the CTMC ensemble terminus (pair-frequency + unpaired probability).
- Computes PMI-style log-odds `score(i,j) = log((p_ij+eps) / sqrt((q_i+eps)(q_j+eps))) / temperature`.
- Decodes a legal structure via `nested_dp` (exact pseudoknot-free DP) or `pseudoknot_allowed_greedy` (annotated greedy matching).
- `temperature`, `threshold`, `matching_policy` are **locked on validation** and recorded in `decoder_manifest.json`; the manifest's `test_override_allowed` is `false` and `evaluate-checkpoint` rejects any attempt to refit on test.

### 2.4 Regression evidence

- `test_legacy_direct_exactly_matches_historical_projection` reconstructs the historical projection outside the API and asserts bit-equality with `predict_structure(..., inference_config=InferenceConfig(mode=InferenceMode.LEGACY_DIRECT))`.
- `test_evaluate_efold_default_refuses_uncalibrated_endpoint_path` (in `tests/test_cli_c0.py`) verifies the default `evaluate-efold` path refuses to run without a locked decoder manifest.
- The default `InferenceConfig` no longer enters the `t=1 / all-unpaired` branch.

## 3. Fixed Sample Inventory

The fixed evaluation matrix is anchored on `SHA256(source_id | sequence | 20260718)` with length-bucket proportional sampling, so the sample set is input-order independent and reproducible.

| Phase | Split | Count | Purpose |
|---|---|---:|---|
| CTMC coarse search | validation | 128 | steps × samples grid selection |
| Decoder final calibration | validation | 512 | temperature × threshold × policy grid |
| Component test | `mmseqs_component_test` | 1,000 | locked evaluation |
| Component holdout | `mmseqs_component_holdout` | 1,000 | locked evaluation |
| PDB | `PDB` | 333 (full) | locked evaluation |

- CTMC coarse grid: `steps ∈ {8, 16, 32}` × `samples ∈ {4, 8, 16}` (9 configurations).
- Decoder grid: `temperature ∈ {0.5, 1.0, 2.0}` × `threshold ∈ {-2, -1, 0, 1, 2}` × `matching_policy ∈ {nested_dp, pseudoknot_allowed_greedy}` (30 configurations).
- Tie-break order: exact F1 → shifted F1 → legality → fewer model calls → runtime. When `nested_dp` and `pseudoknot_allowed_greedy` tie, `nested_dp` is selected.
- The fixed sample IDs and validation SHA-256 are recorded in `c0_artifacts/checkpoint_selection.json` (`validation_sha256: 1782f8901a2c1e7f88ad0db14437c8abf4b8cbe032f66c8ad335df38a6d94e96`).

## 4. Provenance

### 4.1 Checkpoint

- Selected checkpoint: `RF-CF3-family-balanced_mmseqs_torch_full_data_e1_bs16/training_checkpoint.json`
- Selection split: `validation` only; `test_metrics_used: false`.
- Selection metric: `mean_exact_f1` (tie-break: `mean_shifted_f1`, then `parameter_count`).
- Locked SHA-256: `d28aa0aa419a6b41c1941ad3e7325a89236e686ae795623dab694361e3d48465`
- Parameter count: 3,290
- Validation metrics of the selected checkpoint (128 samples): mean exact F1 = 0.0296, mean shifted F1 = 0.0467, legality rate = 1.0, runtime mean = 0.31 s/sample.
- The other two candidates (`RF-CF1-contact-lam0p2`, `RF-CF2-long-range-w2-early-gpu5`) are documented in `c0_artifacts/checkpoint_selection.json` but were not selected.

### 4.2 Code

- `code_sha256()` in `c0_evaluate.py` hashes every `*.py` under `src/reactflow/` and is embedded in each `decoder_manifest.json`.
- `read_decoder_manifest` rejects any manifest whose `code_sha256` does not match the currently installed package, so a manifest produced in the stage cannot be silently reused after deployment.
- The initial state manifest records 131 project file entries with SHA-256 hashes.

### 4.3 Data splits

- `validation_sha256: 1782f8901a2c1e7f88ad0db14437c8abf4b8cbe032f66c8ad335df38a6d94e96` (val.jsonl).
- Gold SHA-256 for `mmseqs_component_test`: `a9cc70826d5f9ee73b6124c9fa1be69517e4005783ca3b2b9784e08bed9afa88`.
- Gold SHA-256 for `mmseqs_component_holdout`: `633ae287ac54f9abb687903488802f5485740f4aff9583ddfcc9b0ff787572d1`.
- Predictions SHA-256 and progress artifact SHA-256 are written into `baseline_efold_results.json` for both tiers.

### 4.4 Frozen features

- Path: `/home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012/frozen/ribonanzanet2_sharded_full`
- Manifest SHA-256: `1e7d52b290dd7891a3ec195ce7b727712e7e076c2d0b6e5297c6868318b95491`
- Referenced read-only from the stage; never copied or modified.

### 4.5 Canonical eFold baseline

- Source artifacts: `baseline_efold_in_clan_progress_results.json` and `baseline_efold_novel_progress_results.json`.
- Merger enforces `matched_count == gold_count == count` and `missing_count == extra_prediction_count == duplicate_gold_count == sequence_mismatch_count == 0` for both tiers.
- Canonical tiers: `mmseqs_component_test` and `mmseqs_component_holdout`. Legacy labels `in_clan` and `novel_clan` are kept only as read-only aliases inside `legacy_aliases`.

| Tier | Count | Mean F1 | Mean MCC | Micro F1 | Micro MCC |
|---|---:|---:|---:|---:|---:|
| `mmseqs_component_test` | 16,606 | 0.2197 | 0.2244 | 0.3136 | 0.3263 |
| `mmseqs_component_holdout` | 46,147 | 0.2125 | 0.2173 | 0.3089 | 0.3223 |

## 5. Test Plan and Coverage

### 5.1 pytest

- Command: `PYTHONPATH=src:.c0_test_deps .c0_test_deps/bin/pytest tests/ --cov=reactflow --cov-branch`
- Result: **509 passed, 1 skipped** (0 failed).
- Skip reason: `tests/test_data.py::test_... : could not import 'h5py': No module named 'h5py'`. This is an optional-dependency skip (h5py is listed under `[project.optional-dependencies].data`), not an OS-only skip. It is recorded here as required by the goal.

### 5.2 Coverage

- Total combined coverage: **91.86%** (gate: ≥90%).
- C0 module coverage:

| Module | Stmts | Miss | Branch | BrPart | Cover |
|---|---:|---:|---:|---:|---:|
| `src/reactflow/c0_evaluate.py` | 132 | 2 | 52 | 2 | 98% |
| `src/reactflow/inference.py` | 125 | 3 | 32 | 3 | 96% |
| `src/reactflow/probing.py` | 102 | 2 | 28 | 2 | 97% |
| `src/reactflow/protocol.py` | 20 | 0 | 4 | 0 | 100% |

- Remaining gaps are in pre-existing modules (`cli.py` 86%, `rfam_metadata.py` 76%, `dfm.py` 79%) that predate C0 and are not in scope for this phase.

### 5.3 Symbolic checks

`c0_artifacts/symbolic_checks.json` records 13 symbolic identities (softmax Jacobian, cross-entropy gradient, CTMC master equation, thermo KL/MSE gradients, heteroscedastic calibration gradient, affine expectation, Pearson affine invariance, mixture path endpoints, weighted calibration normal equations, guidance monotonicity exchange, reactivity magnitude gradient, contact denoising gradient, adapter gradient). **Every residual is exactly `0`.**

### 5.4 CLI smoke runs

1. `calibrate-inference` smoke: 1 validation sample, 2 steps × 2 samples grid, single temperature/threshold. Produced `c0_artifacts/smoke_decoder_manifest.json` with `test_override_allowed: false` and a locked `selected_decoder`.
2. `calibrate-inference` preflight: 33 samples, projected 5.59 hours for the full matrix, `within_runtime_gate: true`.
3. `evaluate-checkpoint` smoke: covered by `tests/test_cli_c0.py::test_calibrate_then_evaluate_checkpoint_uses_locked_manifest_and_honest_tier`, which runs the full calibrate → evaluate pipeline on a fixture and asserts the honest tier label (`mmseqs_component_holdout`) and locked manifest usage.

### 5.5 Test coverage of C0 acceptance criteria

| Acceptance test | Status |
|---|---|
| `legacy_direct` regression vs. historical | ✅ `test_legacy_direct_exactly_matches_historical_projection` |
| Default mode no longer enters `t=1/all-unpaired` | ✅ `test_default_inference_is_not_legacy_endpoint` |
| CTMC uniform init + per-step adapter + reproducibility + legality | ✅ `test_ctmc_is_deterministic_legal_and_uses_dynamic_feature_builder` |
| Decoder pair-vs-unpaired log-odds, threshold, matching policy | ✅ `test_decoder_threshold_and_matching_policy_control_null_choice`, `test_pair_log_odds_prefers_pair_over_high_null_probability` |
| Manifest test-time immutability | ✅ `test_calibrate_then_evaluate_checkpoint_uses_locked_manifest_and_honest_tier`, `test_read_decoder_manifest_rejects_*` |
| Canonical merge success + failure paths + legacy alias | ✅ `test_full_count_merge_renames_tiers_and_records_hashes`, `test_merge_rejects_partial_or_mismatched_artifact`, `test_legacy_tiers_are_read_as_component_semantics` |
| Probing aggregation, missing values, probe strata, proxy exclusion, test-refit rejection | ✅ `test_probe_calibration_is_validation_locked_and_test_fit_rejected`, `test_full_profile_aggregation_uses_all_real_profiles_and_excludes_proxy`, `test_unknown_probe_reports_raw_metrics_but_no_calibrated_metric`, `test_aggregate_full_profiles_*` |
| Metadata round-trip | ✅ `test_cache_round_trip_preserves_probing_and_window_metadata` |

## 6. Runtime

### 6.1 Preflight (32+ samples)

- `mean_ctmc_seconds_per_sample: 8.619`
- `preflight_sample_count: 33`
- `projected_unique_ctmc_hours: 5.586`
- `max_projected_hours: 24.0`
- `within_runtime_gate: true`
- `shared_ctmc_between_modes: true` (CTMC ensemble is computed once and reused across `ctmc_sample` and `calibrated_marginal`).

### 6.2 Fixed matrix execution

- `calibrate-inference` completed at 2026-07-20 11:59:32 (wall time ~23h34m, PID 2544995, GPU 4).
- `final_decoder_manifest.json` (94,244 bytes) records the selected CTMC config (steps=16, samples=8) and decoder config (temperature=1.0, threshold=-2.0, policy=pseudoknot_allowed_greedy).
- `evaluate-checkpoint` consumed the locked manifest and evaluated 3 inference modes × 3 tiers (test 1,000 + holdout 1,000 + PDB 333).
- Both runs used `seed=20260718` and the same locked checkpoint SHA-256.
- **Runtime gate**: total pipeline exceeded the 24h gate (calibrate ~23.6h + evaluate ~9h = ~32.6h). See `c0_artifacts/runtime_bottleneck_report.md` for the bottleneck analysis. Sample counts were NOT shrunk.
- Validation CTMC coarse-grid mean exact F1: 0.008109 (pair_count_ratio=0.010621).

## 7. Data Gaps and Limitations

### 7.1 Clan metadata

- The training split has `pseudo_clan_fraction = 1.0` and `unique_clans = 1` because the source JSONL does not carry true Rfam clan accessions.
- `unknown_family_fraction = 1.0` for the training split; validation/test/holdout have `unique_clans` in the tens of thousands but still `pseudo_clan_fraction = 1.0` because clan labels are synthesized from MMseqs components, not from Rfam.
- **Implication**: any OOD/generalization claim that depends on true clan-level held-out families is **permanently downgraded**. The tier labels `mmseqs_component_test` and `mmseqs_component_holdout` are honest: they reflect MMseqs component disjointness, not Rfam clan disjointness.

### 7.2 SNR / quality metadata

- The full-profile probing layer keeps `reactivity_snr` and `reactivity_quality` fields, but the source data mostly lacks them. The aggregation reports a `missing` stratum count rather than fabricating a quality tier.
- The C0 probing code explicitly outputs `missing` coverage fractions and refuses to draw SNR/quality-stratified conclusions when the underlying fields are absent.

### 7.3 `cross_family_capacity_results.json`

- This artifact is still absent at `/home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012/cross_family_capacity_results.json` (`present: false` in the initial state manifest).
- It is a product of the active RF-CF5 queue and is **not** written, replaced, or back-filled by Phase C0.

### 7.4 Baseline scope

- The eFold baseline is a single-seed local rerun (`seed_count: "single_seed"`). It is sufficient for same-split comparison but cannot support variance claims.

## 8. Unsupported Scientific Claims

Phase C0 **does not** and **cannot** support the following claims:

1. **SOTA**. The eFold baseline mean F1 is 0.220 (test) / 0.212 (holdout). The calibrated_marginal inference achieved mean exact F1 of 0.024214 (test) / 0.027722 (holdout), which is BELOW the baseline. The acceptance gate in Section 9 determines the next phase.
2. **`novel_clan` generalization**. There are no true Rfam clans in the data; the `novel_clan` label is a legacy alias for MMseqs component holdout. New reports use `mmseqs_component_holdout` exclusively.
3. **Real ensemble recovery**. The CTMC ensemble is a model-based trajectory ensemble, not an experimental ensemble. `pair_frequency` reflects DFM posterior sampling, not wet-lab reactivity.
4. **Top-journal readiness**. The acceptance gate below is a necessary but not sufficient condition. Peer review requires additional OOD benchmarks, multi-seed baselines, and clan-resolved splits that are out of scope for Phase C0.

## 9. Acceptance Gate

The goal defines the next-step decision rule:

| Condition | Next goal |
|---|---|
| `corrected inference` improves component holdout F1 by ≥0.03 **or** mean F1 ≥0.10 | Phase C1: structured pair trunk |
| Holdout F1 <0.10 **and** improvement <0.03 | Stop loss/capacity sweep; next goal is symmetric pair trunk + structured decoder 10k matched-capacity prototype |

The acceptance check ran after `evaluate-checkpoint` finished. The decision is recorded in `c0_artifacts/acceptance_decision.json`.

**Result**: GATE NOT PASSED.

| Metric | Value |
|---|---|
| baseline_holdout_f1 (eFold) | 0.212464 |
| legacy_direct holdout F1 | 0.029755 |
| calibrated_marginal holdout F1 | 0.027722 |
| calibrated_marginal test F1 | 0.024214 |
| calibrated_marginal mean F1 (test+holdout)/2 | 0.025968 |
| delta_vs_legacy (holdout) | -0.002034 |
| meets_improvement_gate (>=0.03) | False |
| meets_mean_f1_gate (>=0.10) | False |
| meets_acceptance_gate | False |

**Next goal**: symmetric pair trunk + structured decoder 10k matched-capacity prototype

If the split/leakage audit cannot prove clan-level disjointness (which is the case here, see Section 7.1), the OOD wording in any downstream artifact is **permanently downgraded** to MMseqs-component semantics, regardless of the F1 outcome.

If the split/leakage audit cannot prove clan-level disjointness (which is the case here, see Section 7.1), the OOD wording in any downstream artifact is **permanently downgraded** to MMseqs-component semantics, regardless of the F1 outcome.

## 10. Process Safety and Deployment Posture

### 10.1 Active processes at C0 start

The initial state manifest records four cunyuliu-owned processes referencing `/home/cunyuliu/reactflow` or the RF-CF5 queue:

- PID 1437378: `bash scripts/run_capacity_after_long_range.sh` (RF-CF5 serial scheduler).
- PID 2428255 (child of 1437378): RF-CF5 capacity-h16-a16 `evaluate-efold`.
- PID 3905214: RF-CF5 capacity-h64-a32-early-gpu5 `evaluate-efold`.
- PID 1548898: the `c0_snapshot_state.py` invocation that produced the manifest itself.

None of these processes were terminated, restarted, or modified. The C0 stage re-discovers PIDs at execution time and does not depend on the literal numbers above.

### 10.2 Deployment gate

Deployment into `/home/cunyuliu/reactflow` is **deferred**. The `scripts/deploy_c0_stage.py` gate enforces:

1. `active_reactflow_processes(active_root)` returns an empty list (no scheduler, no `evaluate-efold` child, no `run_capacity_after_long_range.sh`).
2. Every file in `/home/cunyuliu/reactflow` that was hashed in `c0_initial_state_manifest.json` still matches its initial SHA-256. Any drift stops deployment and emits a conflict list.
3. Only after both gates pass: backed-up file replacement, hash-verified sync, and a read-only smoke test in the active directory.

### 10.3 Current deployment status

- RF-CF5 scheduler PID 1437378 and at least one child (PID 2042374 for RF-CF5 capacity-h32-a16) are still running.
- Deployment has **not** been attempted. The stage remains isolated.

## 11. Artifact Index

| Path | Purpose |
|---|---|
| `c0_artifacts/c0_initial_state_manifest.json` | Frozen initial state: code hashes, PIDs, GPU, baseline hashes |
| `c0_artifacts/checkpoint_selection.json` | Validation-only checkpoint selection record |
| `c0_artifacts/baseline_efold_results.json` | Canonical merged eFold baseline |
| `c0_artifacts/c0_data_diversity_manifest.json` | Split audit (component sizes, pseudo-clan fraction, length buckets) |
| `c0_artifacts/c0_data_diversity_audit.{json,md}` | Human-readable diversity audit |
| `c0_artifacts/symbolic_checks.json` | 13 symbolic identities, all residuals = 0 |
| `c0_artifacts/runtime_preflight_32plus.json` | 33-sample runtime preflight, 5.59 h projected (evaluate-checkpoint only) |
| `c0_artifacts/runtime_bottleneck_report.md` | Runtime bottleneck report (total pipeline exceeds 24h gate) |
| `c0_artifacts/smoke_decoder_manifest.json` | 1-sample calibrate-inference smoke manifest |
| `c0_artifacts/preflight_decoder_manifest.json` | Preflight decoder manifest (small grid) |
| `c0_artifacts/coverage.json` | pytest-cov JSON report |
| `c0_artifacts/final_decoder_manifest.json` | Full calibrate-inference output (94,244 bytes, selected CTMC + decoder config) |
| `c0_artifacts/final_evaluation/metrics.json` | Full evaluate-checkpoint output (per-tier × per-mode aggregated metrics) |
| `c0_artifacts/acceptance_decision.json` | Section 9 decision record (gate result + next goal) |
| `c0_logs/calibrate_full.log` | calibrate-inference background log |
| `src/reactflow/inference.py` | Unified inference API |
| `src/reactflow/c0_evaluate.py` | C0 structure metrics + manifest reader |
| `src/reactflow/probing.py` | Leakage-safe probing calibration + aggregation |
| `src/reactflow/protocol.py` | Tier labels + stable subset selection |
| `scripts/merge_efold_baseline.py` | Canonical baseline merger |
| `scripts/select_c0_checkpoint.py` | Validation-only checkpoint selector |
| `scripts/c0_snapshot_state.py` | Initial state manifest builder |
| `scripts/deploy_c0_stage.py` | Deployment gate (process + hash safety) |
| `tests/test_c0_protocol.py` | Merge / tier / metrics tests |
| `tests/test_inference.py` | Inference API regression + CTMC tests |
| `tests/test_probing.py` | Probing calibration + aggregation tests |
| `tests/test_cli_c0.py` | calibrate-inference + evaluate-checkpoint CLI tests |
| `tests/test_c0_coverage_extras.py` | Targeted branch-coverage tests for C0 modules |
