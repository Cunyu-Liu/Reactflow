# ReactFlow-Delta prospective-v2 full-spectrum claim-evidence map (P6, 2026-08-14)

> 范围：P0–P5 全部正式 claim 与对应证据/裁决。证据均来自 locked 结果与 replay。
> 状态边界：`DEVELOPMENT_REPLICATED / EXTERNAL_TRANSPORTABILITY_ESTABLISHED /
> PRACTICAL_IMPORTANCE_NOT_ESTABLISHED / MECHANISM_CONCENTRATION_NOT_ESTABLISHED /
> PUBLICATION_NOT_RELEASED`

## 1. 正式 claims（可发表范围）

| Claim | 裁决 | 证据 |
|---|---|---|
| C1. P2 开发集 direct 前瞻可学习性 | `DIRECT_DEVELOPMENT_LEARNABILITY_PASS` | 20-puzzle 配对 D=+0.0127，95% CI [+0.0079,+0.0175]，sign-flip p=1.9e-6，20/20 正向；`p2_direct_v2_result` + replay |
| C2. LRSO 无增量技能 | `NO_INCREMENTAL_LRSO_SKILL` | rank2/4/8 CI upper 全 <0（−0.0141/−0.0084/−0.0098）；`p3_lrso_v2_result` + replay |
| C3. direct 信号在 development-disconnected 外部复现（可迁移性） | `P4_EXTERNAL_STATISTICAL_PASS` | 24 components/3237 SNV，component-macro D=+0.0410，CI lower +0.0153；FWER pass；leave-dominant-out CI lower +0.0127；`p4_external_result` + replay |
| C4. 冻结 scale 下校准可接受 | `CALIBRATION_ACCEPTABLE` | cov68 0.699 / cov95 0.874（预声明容差内）；`p4_calibration_result` |
| C5. 信号是 feature-dependent（非伪影） | ESTABLISHED | 置换负对照 permuted D CI upper −0.062 <0；`p5_mechanism_result` |
| C6. 效应跨生物学区域复制 | ESTABLISHED | M3SARS +0.083、15KLIB +0.031（2/3 数据集正向）；`p5_mechanism_result` |

## 2. 被删除/未建立的 claims

| Claim | 裁决 | 原因 |
|---|---|---|
| C7. direct 技能集中在编辑位点（distance heterogeneity） | `MECHANISM_NOT_ESTABLISHED` | 预冻结 claim；edit−vfar 异质性 CI lower −0.0199 <0；距离曲线均匀。按合同 §12.7 删除机制 claim。 |
| C8. practical/material importance | `PRACTICAL_IMPORTANCE_NOT_ESTABLISHED` | 无独立 delta_practical 证据（frozen protocol §3）。 |
| C9. SOTA / external generalization 的广泛表述 | NOT_CLAIMED | 合同 §6 claim boundary；只允许 benchmark-level 统计优越性。 |

## 3. 证据分类与可审计性
- 所有 C1–C6 数字均可由 `run_replay_v1.py` 从 clean checkout + artifacts 重放复现（`REPLAY_CONSISTENT`）。
- 数据溯源：dev=OK7a_M2 Round 3（160/160/13976）；external=Ribonanza M2-style 2A3 via RMDB（24/3237，零 dev 重叠）。
- 失败记录：`p6_failure_log_20260814.md`（F1–F7）。
- 模型：RFD-Direct（reg_direct）单 seed 为初步；五 seed ensemble 为部署目标（§9.1）。

## 4. 论文允许/禁止表述
- 允许：development direct-learnability PASS；LRSO NO_INCREMENTAL；外部统计可迁移 PASS（benchmark-level）；负对照与区域复制；机制集中 claim NOT_ESTABLISHED。
- 禁止：practical/material importance；SOTA；"模型整体失败"；编辑位点集中机制；未复现的"LRSO 优势"。
