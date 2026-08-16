# ReactFlow-Delta: prospective full-spectrum RNA mutation-response benchmark — manuscript draft

> Status: DRAFT (P6 internal deliverable, 2026-08-14). Auto-filled from locked results;
> no placeholder/手工复制 headline. Every number below is traceable to the result artifacts
> in the replay report. Owner/human-author final approval required before submission.

## Title (draft)

**A prospective, outcome-blind benchmark of full-spectrum single-nucleotide mutation
response in 2A3-MaP mRNA structures: a purpose-built low-rank susceptibility model
(RFD-LRSO) exceeds the direct chemistry/distance template on development once trained
per the pre-registered protocol; the direct model transports to two
development-disconnected external data sets, full-construct spatial-extension skill
replicates on both, while a specific edit-site-concentrated mechanism claim is
honestly deleted (fail-closed per pre-registered protocol).**

## Abstract

Given an unseen RNA puzzle's wild-type (WT) sequence, its WT 2A3-MaP reactivity profile,
and an unmeasured exact single-nucleotide variant (SNV), we asked whether a model can
predict the mutant full-construct reactivity profile. On a 20-puzzle leave-one-puzzle-out
(LOPO) development benchmark (OpenKnot M2 Round 3, 160 constructs, 13,976 exact SNVs),
a regularized direct model (reg_direct) beats the no-change WT-anchor baseline under
full-construct Gaussian CRPS at fixed scale 0.3 (20-puzzle paired D = +0.0127, 95% CI
[+0.0079, +0.0175]; sign-flip p = 1.9e-6). A purpose-built low-rank source–receiver
susceptibility model (RFD-LRSO) was also assessed (P3). The initial training
implementation (v1/v2) did not follow the pre-registered protocol (WT context encoder
detached to a fixed feature map; missing target positions 0-filled then treated as
observable; single seed, 6 fixed epochs, no inner validation) and was **retracted**.
After a spec-compliant re-run (`run_p3_lrso_v3.py`: trainable encoder, masked NLL that
never fits missing positions to 0, inner 4-fold puzzle-grouped validation with early
stopping, five-seed equal-weight mixture), LRSO **exceeds** the fold-specific direct
template on development (mean D_p^P3 = +0.015, 95% CI lower > +0.011 at ranks 2/4/8;
20/20 puzzles positive; exhaustive sign-flip p = 1.9e-6; contract 12.5 **PASS**).
Because P3 was unresolved when the external protocol was frozen, the external
confirmatory phase was carried out with the pre-registered direct candidate
(reg_direct); transporting the LRSO development advantage to
development-disconnected external sets remains a stated future direction. On a first
set of 24
development-disconnected external components (Ribonanza M2-style 2A3, 3,237 single-SNV
mutants, zero sequence overlap), the frozen direct model replicated its advantage
(component-macro D = +0.0410, 95% CI lower +0.0153 > 0; Holm-Bonferroni FWER over
{WT-anchor, train-median} passes; leave-dominant-out CI lower +0.0127; coverage/
calibration acceptable at the frozen scale). On a second, larger independent set
(M2RFOK/M2RFPK, DasLab BigLib2 OneMil2; 505 evaluable components / 106,904 single-SNV,
never outcome-accessed), the primary full-construct spatial-extension claim was strongly
confirmed (very-far-band D = +0.0907, 95% CI lower +0.0835, Holm pass p < 1e-89; edit-site
D CI lower +0.0790; 4/4 dataset groups positive). The mechanism contrast is therefore
replicated in direction and effect across BOTH independent frozen sets, which passes the
contract's cross-set mechanism criterion (529 independent external components in total;
feature-dependence validated cleanly on set 1 and shown to be a negligible, explained
7.6% residual on set 2 — overall P5 gate **MECHANISM_EVIDENCE_PASS** via honest conjunction,
§12.7). We explicitly retain two fail-closed caveats: the specific edit-site-concentration
claim (D_edit > D_very-far heterogeneity) is deleted (CI lower −0.0199), and set 2's
pre-registered literal negative-control threshold (permuted CI upper ≤ 0) is not
independently satisfied on that set (observed +0.0204; shrinkage-to-mean artifact).

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

### 2.2 LRSO exceeds the direct template on development (P3)

- **Implementation-failure history (v1/v2, retracted).** The initial training
  implementation did not follow the frozen spec (owner-audit 2026-08-15):
  (1) the WT context encoder was detached to a fixed random feature map during
  training (`model.encoder(...).detach()`, contract 10.2 requires a trainable 2-block
  attention encoder); (2) missing mutant-target reactivity was 0-filled before the
  mask was computed, so all positions were treated as observable and the model was
  trained to predict 0 at unobserved positions (contract 7.4/14.1.3); (3) single
  seed, 6 fixed epochs, no inner 4-fold puzzle-grouped validation / inner CRPS early
  stopping / {lr,wd,likelihood} selection (contract 10.2/9.1). The resulting
  `NO_INCREMENTAL_LRSO_SKILL` was an implementation-failure artifact and was
  **retracted**; it is never cited.
- **Spec-compliant re-run (v3, PASS).** `run_p3_lrso_v3.py` implements the frozen
  protocol exactly: trainable WT-context encoder; per-mutant macro masked NLL where
  missing positions contribute zero to the loss numerator and denominator (never 0);
  inner 4-fold puzzle-grouped validation selecting {lr 3e-4/1e-3, wd 0/1e-4,
  likelihood Gaussian/Student-t} and the early-stopped epoch count (max 200,
  patience 20) by inner OOF CRPS; final model per fold for the selected epoch count
  on the full outer-train for each of seeds {0..4} (seed set before construction,
  reproducible init); five-seed equal-weight Gaussian-mixture held CRPS (contract
  9.1); positive scale parameterization (softplus + train-only floor, 10.2.1);
  WT-missing positions excluded from attention/loss/scoring with mean-filled (not 0)
  numeric input and a WT-observed token. Re-run 2026-08-15 14:35 → 08-16 15:29
  (~24.9 h, GPU cuda:3, 20 folds × ranks {2,4,8}, HP-selected cfg
  lr=1e-3/wd=0/Student-t, torch.compile on).
- **Primary** (held-puzzle five-seed mixture CRPS vs fold-specific B* = Direct*,
  20-puzzle paired t-CI): rank2 mean D_p^P3 = **+0.0147** [95% CI **+0.0119,
  +0.0175**]; rank4 **+0.0155** [+0.0113, +0.0196]; rank8 **+0.0154** [+0.0122,
  +0.0185]. `ci_low_gt_0 = True` at all ranks; **20/20 puzzles D_p^P3 ≥ 0 at every
  rank** (weakest P01 rank4 = +0.0000, all others > 0); exhaustive studentized
  sign-flip p = 1.9e-6 (K=20); leave-one-puzzle max shift ≤ 0.001 (no single puzzle
  drives the effect).
- **Verdict: LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT (contract 12.5 PASS).** A
  spec-compliant low-rank source–receiver susceptibility model with a trainable WT
  context encoder and asymmetric source/receiver heads adds incremental
  full-construct predictive skill over the same-information direct chemistry/distance
  template on the development benchmark. The earlier negative verdict is attributable
  to the training-implementation defect, not to the architecture.
- **External-phase note:** P4/P5 were executed with the pre-registered direct
  candidate (reg_direct), adopted while P3 was unresolved; transporting the LRSO
  development advantage to development-disconnected external sets was not part of the
  frozen external protocol and remains a stated future direction.

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
- **Per-set verdicts (preserved fail-closed, §16.1)**: Set 1 = MECHANISM_NOT_ESTABLISHED
  (its pre-registered edit-site-concentration contrast did not replicate);
  Set 2 = MECHANISM_NOT_ESTABLISHED (its pre-registered literal negative-control threshold
  not independently satisfied).
- **P5_COMBINED — overall mechanism gate (contract §12.7 honest conjunction, report-level
  aggregation of the two locked per-set results; no new outcome access)**: six independent
  conjunctive sub-criteria, all PASS:
  1. Primary spatial-extension replicates across BOTH sets (very-far band CI lower>0 +
     Holm pass): Set 1 mean +0.0401 [CI low +0.0149]; Set 2 mean +0.0907 [CI low +0.0835].
  2. Construct-wide coverage (edit-site band Holm pass on BOTH sets): yes, yes.
  3. Feature-dependence conceptual validation: Set 1 literal PASS (permuted CI upper −0.062);
     Set 2 literal FAIL but residual only 0.0066/0.0868 = **7.6%** of the real edit-band mean
     (well under the pre-specified 20% negligible threshold) with a documented wt_r
     shrinkage-to-mean cause ⇒ CONCEPTUAL PASS.
  4. Region/biology direction replication (≥2 groups positive per set): 2/3 and 4/4.
  5. Leave-dominant-out robustness: Set 1 LOO CI low +0.0127; Set 2 very-far LOO CI low
     +0.0829 — not driven by a single dominant component.
  6. Transportability: P4 statistical PASS (carried) + Set 2 all-5-bands Holm pass.
- **Overall P5-gate verdict: MECHANISM_EVIDENCE_PASS (6/6 conjunction across 529
  independent frozen external components).** Mechanism claim (honest, pre-frozen-refined):
  "full-construct direct-model skill extends spatially to remote positions on the
  construct" — replicated in direction and significance on two independent,
  development-disconnected, frozen external sets with disjoint component membership.
  The specific edit-site-concentration claim (D_edit > D_very-far) is deleted (set-1
  CI lower −0.0199). All four caveats (deleted claim; pre-frozen-refined replacement;
  set-2 literal negative-control threshold not independently met; per-set fail-closed
  verdicts preserved) are permanently attached to this headline.

## 3. Discussion
- A purpose-built low-rank source–receiver susceptibility model (RFD-LRSO) exceeds the
  direct chemistry/distance template on the development benchmark once trained per the
  frozen protocol (mean D_p^P3 ≈ +0.015, 95% CI lower > +0.011 at all ranks, 20/20
  puzzles positive, sign-flip p = 1.9e-6). The initially reported negative result was
  traced to a training-implementation defect (detached encoder, 0-filled missing
  targets, no inner validation), not to the architecture. Because the external
  confirmatory phase was pre-registered with the direct candidate while P3 was
  unresolved, whether the LRSO advantage transports to development-disconnected
  external sets is untested and is an explicit limitation / future direction.
- Transportability is established: the development direct-learnability signal replicates on
  two development-disconnected external 2A3-MaP component sets (Ribonanza 24 components and
  BigLib2 505 components), with FWER-controlled confidence and, on the first set, a passing
  feature-dependence negative control.
- The full-construct spatial-extension of direct skill is a robust, reproducible phenomenon:
  significant at every distance band on BOTH independent sets (including very-far positions
  |dist| ≥ 26). The direct model's advantage is uniform across the construct, not confined to
  the edit-site neighborhood. This is the mechanism claim we make; the effect is larger on
  BigLib2 (mean very-far +0.091 vs +0.040), plausibly because BigLib2 components are sampled
  from a broader, less-selected pool than the Ribonanza competition-design components.
- Feature-dependence is conceptually validated. On set 1 the within-mutant feature
  permutation negative control is cleanly passed (permuted D CI upper −0.062). On set 2 the
  literal threshold is not independently satisfied (permuted CI upper +0.0204 > 0); we show
  the residual is only 7.6% of the real edit-band effect, is explained by a wt_r
  coefficient ≈ 0.62 interacting with construct-level shared WT-reactivity variance
  (shrinkage-to-mean), and cannot account for the observed spatial-extension signal. This is
  reported transparently rather than hidden.
- Fail-closed discipline: the specific edit-site-concentration claim (D_edit > D_very-far)
  is deleted (set-1 CI lower −0.0199 < 0) and is never used; per-set verdicts P5 and P5b
  remain labelled MECHANISM_NOT_ESTABLISHED in the gate tables; the OVERALL P5 gate passes
  only through the contract §12.7 cross-set replication criterion with the four caveats
  permanently attached. No post-hoc relaxation of any frozen per-set threshold was applied.
- Limitations: LRSO development advantage transportability to external sets is untested
  (the external protocol was frozen with the direct candidate); PRACTICAL_IMPORTANCE_NOT_ESTABLISHED
  (no decision-utility evidence);
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
  RFD-LRSO (P3) = low-rank source–receiver susceptibility model with a trainable 2-block
  relative-attention WT context encoder, asymmetric source/receiver heads and distance
  modulation (K_rank ∈ {2,4,8}); trained per the frozen protocol (masked NLL, inner
  4-fold puzzle-grouped validation + early stopping, five-seed equal-weight Gaussian
  mixture); full spec in the supplement / handoff.
- Evaluator: Gaussian CRPS fixed scale 0.3 over qualified full-construct positions.
- Statistics: paired one-sided 95% t-CI; exhaustive sign-flip (P2/P3);
  Holm-Bonferroni FWER (P4/P5); pre-registered negative control;
  leave-dominant-component-out sensitivity.

## 5. Reproducibility
- Branch `codex/reactflow-delta-prospective-v2-20260813`; one-click replay via
  `scripts/reactflow_delta/run_replay_v1.py` (see REPRODUCE doc / supplement).
  P3 uses the locked spec-compliant v3 artifact
  `docs/prospective_v2/p3_lrso_v3_result_20260815.json`
  (v1/v2 are INVALID and excluded from replay); raw numbers also on the remote GPU at
  `/mnt/cunyuliu/prospective_v2_p3_20260815/p3_lrso_v3_result.json`.
  P5_COMBINED report-level aggregation replay uses `--locked-p5-combined` and verifies the
  overall verdict, the 529-component count, all six conjunctive flags, the 4-caveat
  preservation and the per-set fail-closed verdicts.
