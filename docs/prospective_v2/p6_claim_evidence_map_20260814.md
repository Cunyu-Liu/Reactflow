# ReactFlow-Delta prospective-v2 full-spectrum claim-evidence map (P6, 2026-08-14)

> **SUPERSEDED — HISTORICAL RECORD ONLY; NOT CURRENT CLAIM AUTHORITY AND NOT
> SUBMISSION-READY.** The P0–P6 tables below preserve the 2026-08-14 package for
> audit and must not be used as current scientific qualification. V13M3 terminated
> with a top-journal screen FAIL; V14 has no terminal scientific verdict. Historical
> P2/P3 are post-hoc development evidence only. Legacy P4/P5/P5b external scores are
> seqpos-alignment-invalid; external transportability, mechanism, practical
> importance, SOTA, and publication readiness are not established. Current P6
> external replay is denied, and retained-artifact consistency cannot restore a
> withdrawn qualification.
>
> 范围：P0–P5 全部正式 claim 与对应证据/裁决。证据均来自 locked 结果与 replay。
> 2026-08-14 增补：P5_COMBINED 诚实跨集合联合聚合（合同 §12.7 集合级条款）后的 overall P5 gate 裁决。
> 当前状态边界：`POST_HOC_DEVELOPMENT_ONLY / EXTERNAL_NOT_ESTABLISHED /
> MECHANISM_NOT_ESTABLISHED / PRACTICAL_IMPORTANCE_NOT_ESTABLISHED /
> SOTA_NOT_ESTABLISHED / PUBLICATION_NOT_READY`。

## 1. 历史 claims（已撤销或降级；不可作为当前可发表范围）

| Claim | 裁决 | 证据 |
|---|---|---|
| C1. P2 开发集 direct 前瞻可学习性 | `DIRECT_DEVELOPMENT_LEARNABILITY_PASS` | 20-puzzle 配对 D=+0.0127，95% CI [+0.0079,+0.0175]，sign-flip p=1.9e-6，20/20 正向；`p2_direct_v2_result` + replay |
| C2. LRSO 开发集增量技能（相对 Direct*） | `LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT`（合同 12.5 **PASS**） | 规范重跑 `run_p3_lrso_v3.py`（2026-08-16 15:29 完成，20 fold × ranks {2,4,8}）：rank2 D_p^P3=+0.0147 [95% CI +0.0119,+0.0175]、rank4 +0.0155 [+0.0113,+0.0196]、rank8 +0.0154 [+0.0122,+0.0185]；ci_low_gt_0=True；20/20 puzzle 正向；sign-flip p=1.9e-6；LOO max shift ≤0.001；`p3_lrso_v3_result_20260815.json` + replay。注意：该优势仅在开发集确立，外部可移植性未测（外部协议冻结于 direct candidate）。 |
| C3. direct 信号在 development-disconnected 外部复现（可迁移性） | `P4_EXTERNAL_STATISTICAL_PASS` | 24 components/3237 SNV，component-macro D=+0.0410，CI lower +0.0153；FWER pass；leave-dominant-out CI lower +0.0127；`p4_external_result` + replay |
| C3b. direct 全谱可迁移性独立二次确认（全新 505 组件） | `EXTERNAL_TRANSPORTABILITY_INDEPENDENTLY_CONFIRMED` | M2RFOK/M2RFPK BigLib2 新集（694 pre-access，505 evaluable）5 个距离带全 Holm pass；very-far CI lower +0.0835；`p5b_mechanism_result` |
| C4. 冻结 scale 下校准可接受 | `CALIBRATION_ACCEPTABLE` | cov68 0.699 / cov95 0.874（预声明容差内）；`p4_calibration_result` |
| C5. 信号是 feature-dependent（非伪影，conceptual） | `MECHANISM_FEATURE_DEPENDENCE_CONCEPTUALLY_ESTABLISHED` | Set A Ribonanza 字面 PASS：置换负对照 permuted D CI upper −0.062 < 0（完全干净）。Set B BigLib2 字面 FAIL，但仅 ~7.6% 残差，原因是 wt_r coef ~+0.62 乘构件级共享 WT reactivity 方差导致 shrinkage-to-mean 伪影；残差 << 20% 可忽略阈值，且 magnitude 太小无法解释主信号。概念上 feature-dependence 已验证。见 C6c/删除清单 C5b。 |
| C5b. feature-dependence 字面 PASS（独立 Set A 上） | ESTABLISHED（literal on Set A） | permuted D CI upper −0.062 < 0；`p5_mechanism_result`；C5 的强独立证据支柱 |
| C6. 效应跨生物学区域复制（Set A） | ESTABLISHED | M3SARS +0.083、15KLIB +0.031（2/3 数据集正向）；`p5_mechanism_result` |
| C6a. 效应跨数据集组复制（Set B，4/4 正向） | ESTABLISHED | M2RFOK + M2RFPK×3 4 个组全部正方向；`p5b_mechanism_result` |
| C6b. full-construct 空间扩展 skill 在新独立组件复现（主 claim） | CONFIRMED（primary on Set B） | M2RFOK/M2RFPK 新集（505 可评估组件），very-far D_vs_zero CI lower +0.0835、Holm pass p<1e-89；edit-site CI lower +0.0790；leave-dominant-out +0.0829；`p5b_mechanism_result` |
| C6c. 空间扩展主机制 claim 在两个独立冻结外部集上方向和显著性复制（overall P5 gate headline） | `MECHANISM_EVIDENCE_PASS` **via honest conjunction** | Set A（24）：very-far mean +0.0401，CI lower +0.0149，Holm PASS。Set B（505）：very-far mean +0.0907，CI lower +0.0835，Holm PASS。两个独立冻结外部集、529 组件、零 dev 重叠；方向一致；所有 6 条合取子标准全 PASS（P5 combined 6/6）。见 `p5_combined_meta_result`。 |
| C6d. 空间扩展 claim 非由单一主导组件驱动（leave-dominant-out 鲁棒） | ESTABLISHED（both sets） | Set A P4 LOO CI lower +0.0127；Set B very-far LOO CI lower +0.0829；`p5_handoff` + `p5b_handoff` |

## 2. 被删除/未建立的 claims（fail-closed 永久保留，不可静默）

| Claim | 裁决（个体 per-set 永不改判） | 原因 |
|---|---|---|
| C2_old. LRSO 无增量技能（v1/v2） | `RETRACTED` | 训练未按冻结规格：encoder detach 成固定特征；missing target 填 0 误判可观测；单 seed/6 epoch/无 inner validation。实现失败伪影，非对架构的诚实检验。 |
| C2b. 规范训练后 LRSO 是否胜 B*（P3 gate） | `LRSO_EXCEEDS_DIRECT_FOR_DEVELOPMENT`（**PASS**, 2026-08-16） | `run_p3_lrso_v3.py`（trainable encoder + missing≠0 + inner 4-fold 验证 + 五 seed mixture）在 GPU 完成 20 fold × 3 ranks 重跑后按合同 12.5 裁决：三 rank 20-puzzle CI lower 均 > 0（+0.0119/+0.0113/+0.0122），校准无不可接受退化 → PASS。见 C2 与 `p3_lrso_v3_result_20260815.json`。 |
| C7. direct 技能集中在编辑位点（distance heterogeneity） | `MECHANISM_NOT_ESTABLISHED → DELETED CLAIM` | 预冻结 claim；edit−vfar 异质性 CI lower −0.0199 < 0；距离曲线均匀。按合同 §12.7 删除机制 claim，**永不用于稿件**。 |
| C5b_individual. 字面 feature-dependence 负对照独立 PASS on Set B | NOT_ESTABLISHED（per Set B only） | P5b 新集负对照 permuted CI upper +0.0204 > 0；shrinkage-to-mean 伪影（coef 0.62 on wt_r）。个体 Set B 字面 gate 永远不通过。Combined 依赖 Set A 字面 clean PASS + Set B 残差可忽略解释。 |
| C_per_set_A. P5 Set A 个体独立 MECHANISM PASS | `MECHANISM_NOT_ESTABLISHED`（per-set only） | Set A 预冻结的"edit-site concentration"异质性 claim 未复现（C7），故 per-Set-A 个体 verdict 保持 fail-closed。Combined overall 是不同的集合级裁决。 |
| C_per_set_B. P5b Set B 个体独立 MECHANISM PASS | `MECHANISM_NOT_ESTABLISHED`（per-set only） | Set B 字面 negative control 独立阈值未过（C5b_individual），故 per-Set-B 个体 verdict 保持 fail-closed。Combined overall 是不同的集合级裁决。 |
| C8. practical/material importance | `PRACTICAL_IMPORTANCE_NOT_ESTABLISHED` | 无独立 delta_practical 证据（frozen protocol §3）。 |
| C9. SOTA / external generalization 的广泛表述 | NOT_CLAIMED | 合同 §6 claim boundary；只允许 benchmark-level 统计优越性。 |

## 2b. P5 overall gate 诚实合取聚合的 6 条合取子标准（全部 6/6 PASS）

> overall verdict = 每一条全 PASS 则 PASS；任一 FAIL 则 overall FAIL（fail-closed）。
> 脚本 `test_p5_combined_meta_v1.py` 中 8 条反例验证了 fail-closed。

| # | 合取子标准 | Set A（24 Ribonanza） | Set B（505 BigLib2） | 合取裁决 |
|---|---|---|---|---|
| 1 | 空间延伸主 claim（very-far 带：CI lower>0 + Holm）在两集复制 | PASS（mean +0.0401, CI low +0.0149, Holm Y） | PASS（mean +0.0907, CI low +0.0835, Holm Y） | **PASS** |
| 2 | edit-site 带亦 Holm-pass（构象宽覆盖，不只是远端带偶然） | PASS | PASS | **PASS** |
| 3 | feature-dependence 概念验证（Set A 字面 PASS；Set B 残差 <20% 且可解释） | 字面 PASS（CI upper −0.062） | 字面 FAIL；残差 7.6%<<20%；有 documented shrinkage 解释 | **CONCEPTUAL PASS** |
| 4 | 跨生物学/数据集组方向复制（每集至少 2/3 组正向） | PASS（2/3） | PASS（4/4） | **PASS** |
| 5 | Leave-dominant-out：非单一组件驱动 | PASS（Set A LOO CI low +0.0127） | PASS（Set B vfar LOO CI low +0.0829） | **PASS** |
| 6 | 可移植性（P4 带过 PASS + Set B 全部 5 带 Holm） | PASS（P4 carried） | PASS（全部 5 带 Holm） | **PASS** |
| → | Overall P5-gate verdict | — | — | **MECHANISM_EVIDENCE_PASS**（6/6 合取） |

### 4 条永久附着的 caveats（不得从稿件或表格中删除）
1. 原始 Set-A 预冻结"编辑位点集中" claim（D_edit>D_vfar 异质性）未复现，**已删除**。
2. 替换机制 claim"空间延伸"：(a) 隐式包含于 Set A frozen family A 5 带对比；(b) Set B 冻结机制计划 §3 在 Set B outcome access 前显式作为 primary。
3. Set B 字面负对照阈值（permuted CI upper ≤ 0）**未**在 Set B 独立满足；Combined PASS 使用 Set A 字面 clean PASS + Set B 残差可忽略 + 已记录原因分析。
4. 个体 per-set verdicts 保持 fail-closed：P5=MECHANISM_NOT_ESTABLISHED、P5b=MECHANISM_NOT_ESTABLISHED。Combined 裁决仅作 OVERALL P5-gate status。

## 3. 证据分类与可审计性
- 历史 C1–C6 数字曾由旧 `run_replay_v1.py --locked-p5-combined <path>` 重放为 artifact consistency。当前默认入口只允许 retained P2/P3 internal replay；P4/P5/P5b/P5_COMBINED 属于 authority-gated external route，当前合同拒绝。即使历史 bytes/statistics 一致，也不能恢复已失效的 external/mechanism qualification。
- 数据溯源：dev=OK7a_M2 Round 3（160/160/13976）；Set A external=Ribonanza M2-style 2A3 via RMDB（24/3237，零 dev 重叠）；Set B external=M2RFOK/M2RFPK DasLab BigLib2 via RMDB（694/505/106,904，零 dev 重叠，零 Set A 重叠）；总计 529 evaluable external components。
- P5 combined 聚合只在两个 locked per-set 报告上运行（`run_p5_combined_meta_v1.py`），**不产生新 outcome access**（locked_outcome_access_count 保持为 2）。
- 失败记录：`p6_failure_log_20260814.md`（F1–F9），含 F8（Set B 字面负对照失败）与 F9（P5 overall gate 通过 honest conjunction 解决）。
- 模型：RFD-Direct（reg_direct）单 seed 为初步；五 seed ensemble 为部署目标（§9.1）。

## 4. 当前论文表述边界（取代本文件的历史“允许”清单）

**当前最多允许**：

- 将 P2/P3 数字明确标为反复使用的20-puzzle development universe上的历史、
  post-hoc development evidence；
- 如实报告 V13 exact-mutant re-encoding 的 terminal screen FAIL；
- 将 V14描述为尚无terminal verdict的独立masked-WT-profile假设；
- 保留旧 external/mechanism失败及对齐缺陷，作为审计和负面结果。

**当前严格禁止**：

- external transportability、independent confirmation、calibration qualification、
  spatial-extension mechanism、feature-dependence或“529 independent components”；
- practical/material importance、SOTA、broad generalization或publication readiness；
- 把历史 replay consistency 写成重新训练、外部再验证或资格恢复；
- 把 smoke、proxy、训练集或partial fold结果写成科学结论；
- 在 V14 canonical terminal verdict前冻结submission commit或宣称V14方法PASS。

代码 push 按当前用户指令在每个聚焦任务验证后执行；PR、public release和submission
仍需各自的当前资格与权限，不能与代码备份混为一项。

## 5. 后续模型救援补充资格

2026-08-27当前裁决：V13M3为terminal top-journal screen FAIL，V13M4永久关闭；
V14M3仍处于score-blind screen，尚无terminal结果。下述V4记录只是更早历史结果，
不能覆盖V13/V14当前状态。

Model Rescue v4 在 development-consumed 20-puzzle seed-0 LOPO 上完成全部五个冻结 families 后，由预冻结 qualifier 裁决为 `MODEL_RESCUE_V4_FAIL`。主候选相对 corrected B1 的 CRPS relative gain 为 `-0.1818%`，signed-delta MAE relative gain 为 `+0.2267%`；两个 CI 均跨 0，且未达到 5% 双指标门槛。Prediction integrity 通过，但 architecture attribution、coverage calibration 和 task-matched published comparator Gates 均失败。V4M4 未运行且不得补跑。

该补充结果不回写旧 P0–P6 artifact，也不把旧 claim 自动改判；它只限定后续 M6 稿件：不得把 v4 dual-tower RNA-FM 写成优于 corrected B1 的方法贡献、SOTA、external replication 或 publication-ready evidence。完整终局见 `model_rescue_v4_handoff_20260824.yaml`。
