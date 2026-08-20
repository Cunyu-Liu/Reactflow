# ReactFlow-Delta: prospective full-spectrum RNA mutation-response benchmark — manuscript draft

> Status: DRAFT (P6 internal deliverable, 2026-08-14). Auto-filled from locked results;
> no placeholder/手工复制 headline. Every number below is traceable to the result artifacts
> in the replay report. Owner/human-author final approval required before submission.

## Title (draft — downgraded per audit §9.3, 2026-08-18)

**A prospective, outcome-blind benchmark of full-spectrum single-nucleotide mutation
response in 2A3-MaP RNA structures: the direct chemistry/distance model establishes
development learnability, and a low-rank source–receiver increment contributes a
small but significant incremental signal over the same-architecture rank-0 model
(paired 20-puzzle D_p ≈ +0.0018, CI lower > 0); external replication is exploratory
only (K_joint = 2 study clusters, underpowered), and the external direct result is
retracted as invalid due to a seqpos-alignment defect (audit P4-M1).**

> 降级依据（audit §9.3）：不得写"low-rank susceptibility exceeds direct / 主导"；
> 只能按 P2-v3 主 null（selected-rank vs rank-0，D_p=+0.0018，约占总增益 11%）
> 措辞为 "small but significant incremental low-rank signal"。外部必须是最终 LRSO
> （非 ridge）、seqpos-correct 对齐、cluster 级 exploratory。

## Abstract (downgraded per audit §9.3, 2026-08-18)

Given an unseen RNA puzzle's wild-type (WT) sequence, its WT 2A3-MaP reactivity profile,
and an unmeasured exact single-nucleotide variant (SNV), we asked whether a model can
predict the mutant full-construct reactivity profile. On a 20-puzzle leave-one-puzzle-out
(LOPO) development benchmark (OpenKnot M2 Round 3, 160 constructs, 13,976 exact SNVs),
a regularized direct model (reg_direct) beats the no-change WT-anchor baseline under
full-construct Gaussian CRPS at fixed scale 0.3 (paired 20-puzzle D = +0.0268, 95% CI
[+0.0237, +0.0298]; audit P2-v3 unified evaluator). The full neural candidate
(RFD-Direct, K_rank=0) adds a further +0.0153 over ridge, and a low-rank
source–receiver term (selected rank vs same-architecture rank-0, matched seeds/epochs)
adds a **small but significant incremental signal**: paired 20-puzzle
D_p = **+0.0018**, 95% CI [+0.0003, +0.0033], exhaustive sign-flip p = 0.016, 15/20
puzzles positive — roughly 11% of the total network-vs-ridge gain (audit P2-v3).
The earlier "LRSO exceeds direct" phrasing is downgraded to this magnitude: the
low-rank term is real but small, and the earlier external-phase direct result
(`run_p4_external_v1.py`, component-macro D = +0.0410 "PASS") is **retracted as
invalid**: it indexed external RDAT reactivity by raw sequence index while the true
(seqpos) alignment is offset by ~26 (audit finding P4-M1), so its features and
scoring positions were misaligned. External replication is therefore **exploratory
only**: the external candidate must be the final LRSO (not ridge) with seqpos-correct
alignment, evaluated at the highest independent study/batch cluster level
(K_joint = 2: SL5 PNAS 2024; Ribonanza 2024), which is underpowered for a
confirmatory claim (2 < K_required = 9). The proposed mechanism claims (edit-site
concentration, spatial-extension replication, cross-set "MECHANISM_EVIDENCE_PASS",
"529 independent external components") are all downgraded: the edit-site-concentration
claim is deleted, mechanism remains `MECHANISM_NOT_ESTABLISHED`, and the external
component count is restated as "718 WT anchors across 7 Das-lab M2-seq dataset files;
2 study clusters; K_joint = 2" — never "independent".

## 1. Introduction

(背景：WT-profile-available mutation planning pain point; existing static/reactivity tools
do not directly model signed full-spectrum perturbation; direct chemistry/distance is a
strong competing explanation; directed low-rank susceptibility is a falsifiable
incremental capability — 见 contract §6 论文逻辑链。)

## 2. Results

### 2.1 Development direct learnability (P2, audit P2-v3 unified evaluator)
- Benchmark: OpenKnot M2 Round 3, 20-puzzle LOPO, full-construct CRPS (scale 0.3),
  method-balanced estimand (position→mutant→cell→method→puzzle).
- Primary (correct estimand): Direct(ridge) vs WT-anchor = **+0.0268**, 95% CI
  **[+0.0237, +0.0298]**; vs train-median = +0.0691 [CI +0.0603, +0.0779].
- Independent nonlinear direct (independent MLP): vs WT-anchor +0.0269, vs
  train-median +0.0692 — same signal (audit §7.1 closed positively).
- RFD-Direct (K_rank=0) vs ridge = +0.0153 [CI +0.0098, +0.0208] (full
  network/uncertainty stack).
- Mandatory secondary: signed-delta point MAE is negative vs no-change anchor —
  the CRPS advantage is tail-driven; honest per-method calibration raises D_p.
- **Verdict: DIRECT_DEVELOPMENT_LEARNABILITY_PASS.**

### 2.2 Low-rank increment over the rank-0 network on development (P2-v3 + P3, downgraded)

- **Correct reframing (audit §7.1/§7.2, P2-v3 20-fold unified protocol).** Under a
  unified evaluator with matched seeds/epochs, the full neural stack (RFD-Direct,
  K_rank=0) beats ridge by +0.0153 [CI +0.0098, +0.0208]; the **low-rank term
  (selected rank vs same-architecture rank-0)** adds a **small but significant**
  +0.0018 [CI +0.0003, +0.0033] (sign-flip p=0.0164; 15/20 puzzles positive;
  selected rank = 2 on 18/20 folds). This is the honest headline magnitude:
  the low-rank term is real but constitutes ~11% of the network-vs-ridge gain, not
  a dominant effect.
- **P3 v3 spec-compliant re-run** (`run_p3_lrso_v3.py`): trainable WT-context encoder,
  masked NLL (missing never 0), inner 4-fold puzzle-grouped validation + early
  stopping, five-seed equal-weight mixture; per-rank held CRPS vs fold-specific
  ridge B* gives rank2 +0.0147 / rank4 +0.0155 / rank8 +0.0154 (CI lower > +0.011;
  exhaustive sign-flip p=1.9e-6). This "LRSO vs ridge" effect is dominated (~89%)
  by the network/uncertainty stack (rank0 vs ridge, +0.0153), NOT by the low-rank
  term.
- **Verdict (downgraded):** the incremental low-rank signal is
  `SMALL_BUT_SIGNIFICANT_ON_DEVELOPMENT` (NOT "LRSO_EXCEEDS_DIRECT"; not
  "low-rank dominates"). The earlier `NO_INCREMENTAL_LRSO_SKILL` (v1/v2) remains
  retracted as an implementation-failure artifact.

### 2.3 External (final-LRSO, seqpos-correct, EXPLORATORY — replaces the invalid ridge run)

- External set: Ribonanza M2-style 2A3-MaP (M2SL5/SL5, M3SARS/FSE, 15KLIB/diverse),
  24 components / 3,237 single-SNV, zero dev sequence overlap; **K_joint = 2 study
  clusters** (SL5 PNAS 2024; Ribonanza 2024) — underpowered for confirmatory claims.
- **The legacy ridge external result is RETRACTED (audit P4-M1):**
  `run_p4_external_v1.py` indexed RDAT reactivity by raw sequence index while the
  true (seqpos) alignment is offset by ~26, so its features and scoring positions
  were misaligned; its `P4_EXTERNAL_STATISTICAL_PASS` (component-macro D=+0.0410)
  is not usable as any evidence. The old P5/P5b absolute CRPS/D numbers are likewise
  not citable (same alignment defect).
- **Current external evidence = final-LRSO exploratory run**
  (`run_p4_external_lrso_v1.py`, frozen protocol
  `p4_external_lrso_frozen_protocol_20260818.md`): candidate = final LRSO
  (rank=2, 5-seed mixture, frozen cfg, dev-inner-selected epoch=50), baseline =
  ZeroResponse (fixed scale 0.3), seqpos-correct alignment. **Verdict:**
  `DEVELOPMENT_REPLICATION_EXPLORATORY` (confirmatory eligibility NOT_ESTABLISHED,
  practical importance NOT_ESTABLISHED).
- **Result (K_eff=24, no attrition).** Component-level D_vs_zero = +0.0307
  (CI [+0.0152, +0.0461]; transparency only — counts 21 Ribonanza components as
  independent). Cluster-level (K_joint=2, df=1): **study_sl5 = +0.0008 (≈0, 3
  components) vs study_ribonanza = +0.0350 (21 components)**; K=2 cluster-macro
  mean +0.0179, 95% CI [−0.199, +0.235] (uninformative). LOSO: leave-out SL5 →
  +0.035, leave-out Ribonanza → +0.0008.
- **Interpretation (fail-closed):** the component-level positive signal is driven
  almost entirely by the Ribonanza cluster (shared batches); the independent SL5
  study shows no LRSO increment. The final-LRSO low-rank increment therefore does
  **not replicate across the two independent studies** — neither confirmatory nor
  a consistent replication signal. This supports the audit §9.1 decision-tree "no"
  branch: a development method paper / benchmark / direct route, not a broad
  generalization claim (unless new, non-Das-lab, development-disconnected
  independent study/batch data is obtained).

### 2.4 Mechanism (P5 + P5b — downgraded: MECHANISM_NOT_ESTABLISHED)

> audit §9.3：删除 "MECHANISM_EVIDENCE_PASS" 与 "529 independent external components"；
> combined 仅 exploratory synthesis。且旧 P5/P5b 的绝对 CRPS/D 数字受 P4-M1 错位影响，
> 不可引用。

- **P5 (first set, 24 Ribonanza components, alignment-invalidated absolute numbers)**:
  signed distance curve and edit-site-concentration contrast were computed on the
  misaligned scoring grid (P4-M1); only the **direction** of the spatial-extension
  pattern is retained as an exploratory observation, not its magnitude. The
  pre-registered edit-site-concentration claim (D_edit > D_very-far) is **deleted**
  (set-1 CI lower −0.0199).
- **P5b (second set, M2RFOK/M2RFPK, 505 evaluable components, 106,904 SNV)**:
  same alignment caveat applies to absolute D numbers; the literal negative-control
  threshold was not independently satisfied on this set (permuted CI upper +0.0204 > 0).
- **Per-set verdicts (preserved fail-closed):** Set 1 = `MECHANISM_NOT_ESTABLISHED`;
  Set 2 = `MECHANISM_NOT_ESTABLISHED`.
- **Combined:** no cross-set "MECHANISM_EVIDENCE_PASS"; the P5_COMBINED synthesis is
  `EXPLORATORY_ONLY` and carries the P4-M1 alignment caveat plus the four original
  caveats (deleted claim; pre-frozen-refined replacement; set-2 literal negative
  control not met; per-set fail-closed verdicts preserved). External unit count is
  restated as "718 WT anchors across 7 Das-lab M2-seq dataset files; 2 study
  clusters; K_joint=2" — never "independent".

## 3. Discussion (downgraded per audit §9.3)

- The low-rank source–receiver term (selected rank vs same-architecture rank-0,
  matched seeds/epochs) provides a **small but significant incremental signal** on the
  development benchmark (paired 20-puzzle D_p = +0.0018, 95% CI [+0.0003, +0.0033],
  sign-flip p = 0.016). Decomposing the original "LRSO vs ridge" +0.015, the full
  network/uncertainty stack accounts for ~89% (+0.0153, rank0 vs ridge) and the
  low-rank term for ~11% (+0.0018). The headline is therefore a **small but
  significant low-rank increment**, not a dominant low-rank effect, and not
  "LRSO exceeds direct".
- External transportability is **not established**. The external direct result
  (`P4_EXTERNAL_STATISTICAL_PASS`, component D = +0.0410) is retracted as invalid:
  the RDAT reactivity arrays were indexed by raw sequence index while the true
  (seqpos) alignment is offset by ~26 (audit finding P4-M1), corrupting features
  and scoring positions. The old P5/P5b absolute numbers share this defect.
- Current external evidence is the final-LRSO exploratory run on K_joint = 2 study
  clusters with seqpos-correct alignment
  (`run_p4_external_lrso_v1.py` → `DEVELOPMENT_REPLICATION_EXPLORATORY`); it is
  underpowered (2 < K_required = 9) and supports at most a "replication signal
  direction" observation, never a confirmatory claim.
- The full-construct spatial-extension mechanism claim is downgraded to
  `MECHANISM_NOT_ESTABLISHED`; the edit-site-concentration claim is deleted;
  "MECHANISM_EVIDENCE_PASS" and "529 independent external components" are removed.
- Published comparators (e.g. RibonanzaNet2): cannot be fairly run on these external
  components because (a) the checkpoint is absent from the current workspace
  (only a frozen feature export for a different task exists), and (b) — more
  fundamentally — the external datasets are Das-lab M2-seq 2A3 data of the same
  family as the comparator's training distribution, so a comparison would be
  train-on-test. This is recorded in the exposure ledger.
- Limitations: external confirmatory eligibility NOT_ESTABLISHED (K_joint=2;
  no new independent study/batch data); PRACTICAL_IMPORTANCE_NOT_ESTABLISHED;
  fixed frozen scale 0.3; the low-rank increment is small and must be described at
  that magnitude.

## 4. Methods (summary)
- Data: OpenKnot M2 Round 3 (dev) + external Das-lab M2-seq 2A3-MaP via RMDB
  (M2SL5/M3SARS/15KLIB + M2RFOK/M2RFPK; 718 WT anchors across 7 dataset files;
  **K_joint = 2 study clusters** — SL5 PNAS 2024, Ribonanza 2024; 3 NovaSeq batches).
- Split: split_v4_lopo_puzzle (20-puzzle LOPO; inner 4-fold OOF selection).
- Model: reg_direct = ridge (λ=1e−2) over 12-dim chemistry/distance template
  [WT edit-site state, WT readout state, signed distance, tanh(distance), ref/alt one-hot].
  RFD-LRSO = low-rank source–receiver susceptibility model with a trainable 2-block
  relative-attention WT context encoder, asymmetric source/receiver heads and distance
  modulation; trained per the frozen protocol (masked NLL, inner 4-fold puzzle-grouped
  validation + early stopping, five-seed equal-weight Gaussian mixture); full spec in
  the supplement / handoff.
- Evaluator: Gaussian CRPS fixed scale 0.3 over qualified full-construct positions.
- **External scoring alignment (P4-M1 fix):** RDAT reactivity arrays cover the window
  [seqpos[0]-1, seqpos[0]-1+n_seqpos) of the full construct sequence (offset 26 for all
  three datasets), not index 0; the final-LRSO external run (`run_p4_external_lrso_v1.py`)
  aligns via seqpos and scores the shared-region ∩ observed window at the study-cluster
  level (K_joint=2, exploratory).
- Statistics: paired one-sided 95% t-CI; exhaustive sign-flip (P2/P3);
  Holm-Bonferroni FWER (P4/P5); pre-registered negative control;
  leave-dominant-component-out sensitivity; cluster-macro CI + LOSO (external exploratory).

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
