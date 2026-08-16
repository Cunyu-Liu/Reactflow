# endpoint_v3 / caller_v3 修正案 — 提案（PROPOSED，未激活）

> 状态：**PROPOSED_NOT_ACTIVE**。本文件仅记录经 R5 诊断后建议的科学修复路径，**不构成任何授权**。激活必须经用户显式批准 + 新 authority epoch，并按 `PRIMARY_ENDPOINT_NEVER_SILENT_CHANGE` 产生 `endpoint_v3`、`caller_v3`、新 bundle/sentinel。

## 1. 背景与证据（R5 P2 learnability gate）

- R5 在 GPU（A100-40GB, CUDA 确认）跑完 18 folds 嵌套 leave-one-publication-out。
- 冻结的 `caller_v2` + 冻结 `d1x_v2` 数据产生**近恒定 binary 标签**：3 changers / 3178 non / 3204 NO_CALL。
- 按 `endpoint_v2.degenerate_policies.constant_label`，主估参 publication-macro AUPRC 判为 **UNIDENTIFIABLE**，未伪造数字。
- R6 独立裁决：7/8 Phase 1 gates PASS，`P2_LEARNABILITY_GO` = FAIL → **STOP_METHOD_ROUTE**，Phase 3 架构迭代被阻断。

## 2. 根因诊断（来自 R5 verdict）

| 证据 | 数值 | 解读 |
|---|---|---|
| 跨 study 反应性尺度异质 | 中位数 0.0005–4.08（~4000×），max 达 47222 (TRP4P6) | 不同 study 位于不可比尺度 |
| 报告误差失准 | 中位 2.5× / 均值 8.4×（最多 642×）小于经验复现散差；存在负误差 | caller z-score 与 null 系统性被抬高 |
| null 膨胀 | pooled null median cluster stat ≈ 44；WT-WT |z| p99=44.3, max=1521 | 只有 3 个 changer 是校准伪影 |
| 位置级信号存在 | 458,532 eligible positions 中 22.7% 满足 \|Δreact\|>0.3 | 突变确实改变反应性；退化是 caller/null 校准问题，非生物学负结果 |

## 3. 建议修复（需新授权后才能执行）

1. **per-study 反应性归一化**到公共尺度（如中位数/MAD 或 Z-scaling）。
2. **误差重校准**：用 train-fold 内经验复现散差（而非报告误差）作为噪声源。
3. 重校准后重跑 `caller_v3` → 重导出 binary 标签。
4. 用未变化的 `evaluate_v2` 与 `split_v2` 重跑嵌套 leave-one-publication-out → 重判 `P2_LEARNABILITY_GO/STOP`。

## 4. 不变量（激活后仍须保持）

- test（SL5 family）仍封存；16SFWJ 仍 DEVELOPMENT_CONSUMED。
- 不改变核心科学问题；不降低 gate 阈值；不做 outcome fitting / seed gaming。
- 任何 estimand 变更必须显式版本化为 `endpoint_v3`。

## 5. 审批要求

- 需用户显式批准 + 新 authority epoch（epoch 15）+ 新 amendment + 新 bundle/sentinel。
- 批准前本文件保持 PROPOSED_NOT_ACTIVE，不产生任何训练或 evaluator 结果。
