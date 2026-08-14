# ReactFlow-Delta: prospective full-spectrum RNA mutation-response benchmark — manuscript draft

> Status: DRAFT (P6 internal deliverable, 2026-08-14). Auto-filled from locked results;
> no placeholder/手工复制 headline. Every number below is traceable to the result artifacts
> in the replay report. Owner/human-author final approval required before submission.

## Title (draft)

**A prospective, outcome-blind benchmark of full-spectrum single-nucleotide mutation
response in 2A3-MaP mRNA structures: direct chemistry/distance models transport to
two development-disconnected external data sets, but an edit-site-concentrated
mechanism claim does not replicate.**

## Abstract

Given an unseen RNA puzzle's wild-type (WT) sequence, its WT 2A3-MaP reactivity profile,
and an unmeasured exact single-nucleotide variant (SNV), we asked whether a model can
predict the mutant full-construct reactivity profile. On a 20-puzzle leave-one-puzzle-out
(LOPO) development benchmark (OpenKnot M2 Round 3, 160 constructs, 13,976 exact SNVs),
a regularized direct model (reg_direct) beats the no-change WT-anchor baseline under
full-construct Gaussian CRPS at fixed scale 0.3 (20-puzzle paired D = +0.0127, 95% CI
[+0.0079, +0.0175]; sign-flip p = 1.9e-6). A purpose-built low-rank source–receiver
susceptibility model (RFD-LRSO) added no incremental skill (all ranks CI upper < 0),
so the simplest qualified direct model was adopted. On a first set of 24
development-disconnected external components (Ribonanza M2-style 2A3, 3,237 single-SNV
mutants, zero sequence overlap), the frozen direct model replicated its advantage
(component-macro D = +0.0410, 95% CI lower +0.0153 > 0; Holm-Bonferroni FWER over
{WT-anchor, train-median} passes; leave-dominant-out CI lower +0.0127; coverage/
calibration acceptable at the frozen scale). On a second, larger independent set
(M2RFOK/M2RFPK, DasLab BigLib2 OneMil2; 505 evaluable components / 106,904 single-SNV,
never outcome-accessed), the primary full-construct spatial-extension claim was strongly
confirmed (very-far-band D = +0.0907, 95% CI lower +0.0835, Holm pass p < 1e-89; edit-site
D CI lower +0.0790; 4/4 dataset groups positive), but the pre-registered feature-
dependence negative control did not cleanly pass on that set (permuted D CI upper
+0.0204 > 0, a shrinkage-to-mean artifact). We report external statistical
transportability as established and the edit-site-concentration mechanism claim as
**not established** (fail-closed per contract §12.7/§16.1).

## 1. Introduction

(背景：WT-profile-available mutation planning pain point; existing static/reactivity tools
do not directly model signed full-spectrum perturbation; direct chemistry/distance is a
strong competing explanation; directed low-rank susceptibility is a falsifiable
incremental capability — 见 contract §6 论文逻辑链。)

## 2. Results

### 2.1 Development direct learnability (P2)
- Benchmark: OpenKnot M2 Round 3, 20-puzzle LOPO, full-construct CRPS (scale 0.3).
- Primary: D_p = CRPS(T*) − CRPS(Direct*) = **+0.0127**, 95% CI **[+0.0079, +0.0175]**,
  20/20 puzzles positive, exhaustive sign-flip p = **1.9e-6**, leave-one-puzzle max shift 0.0011.
- Horizontal (held CRPS): reg_direct 0.2023 (+5.92% vs ZeroResponse 0.2150);
  nonlinear/flat_mlp/rfd_direct 0.2718 (−26.4%, overfitting).
- Mandatory secondary: signed-delta point MAE is negative (−6.5%) vs no-change anchor —
  the CRPS advantage is tail-driven; honest per-method calibration raises D_p to +0.0182.
- **Verdict: DIRECT_DEVELOPMENT_LEARNABILITY_PASS.**

### 2.2 LRSO adds no incremental skill (P3)
- RFD-LRSO (K_rank 2/4/8) vs fold-specific Direct*: all 20-puzzle CIs **upper < 0**
  (rank2 −0.0288..−0.0141; rank4 −0.0245..−0.0084; rank8 −0.0242..−0.0098).
- **Verdict: NO_INCREMENTAL_LRSO_SKILL** → adopt reg_direct (contract 17.2).

### 2.3 External confirmation (P4)
- External set: Ribonanza M2-style 2A3-MaP, development-disconnected (M2SL5/SL5,
  M3SARS/FSE, 15KLIB/diverse), **24 components, 3,237 single-SNV**, zero dev overlap.
- Frozen evaluator: reg_direct refit once on all development data; component-macro paired
  D = CRPS(WT-anchor) − CRPS(direct) over shared regions.
- Component-macro D vs WT-anchor = **+0.0410**, 95% CI **[+0.0153, +0.0667]**;
  vs train-median = +0.0446 (CI lower +0.0112); Holm-Bonferroni p = [0.0015, 0.0055] (FWER pass);
  leave-dominant-out CI lower +0.0127.
- Coverage/calibration at frozen scale 0.3: 68%→0.699, 95%→0.874 (acceptable; M3SARS noisier,
  emp. SD 1.37).
- **Verdict: P4_EXTERNAL_STATISTICAL_PASS** (PRACTICAL_IMPORTANCE_NOT_ESTABLISHED).

### 2.4 Mechanism (P5 + P5b, two independent external sets)
- **P5 (first set, 24 Ribonanza components)**: signed distance curve — D_vs_zero positive at
  every |dist| band (edit 0.031, 1–3 0.036, 4–10 0.038, 11–25 0.047, ≥26 0.040; all
  Holm-adjusted significant), but **spatially uniform** — edit-site − very-far heterogeneity
  = −0.0090, CI [−0.0199, +0.0019]. Negative control (permuted features, seed 20260813):
  D CI upper = −0.062 → **no skill** (feature-dependent). Region replication: M3SARS +0.083,
  15KLIB +0.031, M2SL5 −0.020 (2/3 positive). Failure cases: 8/24 components negative at
  the edit site.
- **P5b (second independent set, M2RFOK/M2RFPK, 505 evaluable components, 106,904 SNV)**:
  primary full-construct spatial-extension claim strongly confirmed — very-far-band
  D = +0.0907, 95% CI lower **+0.0835**, Holm pass (p < 1e-89); edit-site D CI lower +0.0790;
  4/4 dataset groups positive; leave-dominant-out very-far CI lower +0.0829. However, the
  pre-registered negative control did NOT cleanly pass on this set: permuted D mean +0.0066,
  one-sided CI upper **+0.0204** > 0 (a shrinkage-to-mean artifact driven by the readout-WT
  feature, coef ≈ 0.62). This is a real property of the BigLib2 data, not a bug.
- **Verdict: MECHANISM_NOT_ESTABLISHED** (fail-closed). The primary spatial-extension claim
  replicates on BOTH independent sets; the feature-dependence negative control passes on set 1
  but not set 2, and the edit-site-concentration claim is deleted per contract.

## 3. Discussion
- Transportability is established: the development direct-learnability signal replicates on
  two development-disconnected external 2A3-MaP component sets (Ribonanza 24 components and
  BigLib2 505 components), with FWER-controlled confidence and, on the first set, a passing
  feature-dependence negative control.
- The full-construct spatial-extension of direct skill is a robust, reproducible phenomenon:
  significant at every distance band on BOTH independent sets (including very-far positions
  |dist| ≥ 26). The direct model's advantage is uniform across the construct, not confined to
  the edit-site neighborhood.
- A mechanistic claim of feature-dependence is supported on set 1 (permuted features destroy
  skill) but not cleanly on set 2 (a shrinkage-to-mean artifact leaves a small positive
  permuted D). Per the pre-registered protocol the mechanism gate remains fail-closed
  (MECHANISM_NOT_ESTABLISHED); we do not over-claim.
- Limitations: PRACTICAL_IMPORTANCE_NOT_ESTABLISHED (no decision-utility evidence);
  single-seed model (five-seed ensemble is the deployment target, contract §9.1);
  fixed frozen scale 0.3 understates uncertainty in high-noise datasets (M3SARS);
  the second independent set shares the DasLab BigLib lineage with the development chemistry
  family and was not outcome-accessed before its single locked evaluation.

## 4. Methods (summary)
- Data: OpenKnot M2 Round 3 (dev) + two external 2A3-MaP sets via RMDB:
  (1) Ribonanza M2-style (M2SL5/M3SARS/15KLIB; 24 components/3,237 SNV);
  (2) DasLab BigLib2 OneMil2 M2RFOK/M2RFPK (694 components/106,904 SNV; 505 evaluable;
  never outcome-accessed before its single locked evaluation).
- Split: split_v4_lopo_puzzle (20-puzzle LOPO; inner 4-fold OOF selection).
- Model: reg_direct = ridge (λ=1e−2) over 12-dim chemistry/distance template
  [WT edit-site state, WT readout state, signed distance, tanh(distance), ref/alt one-hot].
- Evaluator: Gaussian CRPS fixed scale 0.3 over qualified full-construct positions.
- Statistics: paired one-sided 95% t-CI; exhaustive sign-flip (P2); Holm-Bonferroni FWER
  (P4/P5); pre-registered negative control; leave-dominant-component-out sensitivity.

## 5. Reproducibility
- Branch `codex/reactflow-delta-prospective-v2-20260813`; one-click replay via
  `scripts/reactflow_delta/run_replay_v1.py` (see REPRODUCE doc / supplement).
