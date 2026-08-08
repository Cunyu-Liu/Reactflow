# ReactFlowDelta M0-X — Phase 3 模型架构迭代授权提案 (epoch 18)

- **状态**: PROPOSED_NOT_ACTIVE（待用户显式授权激活）
- **提案日期**: 2026-08-08
- **治理合同**: `docs/contracts/ReactFlowDelta_M0X_strict_scientific_engineering_audit_20260807.md`
- **前序**: Phase 2 (REBUILD-P2) → endpoint_v5 conditional magnitude = **GO**
  (run `p2_v5_magnitude_20260808`, deepsets skill 0.677, bootstrap CI low 0.323)
- **§13.4 P0 完成判据**: 7/8 PASS；`P2_LEARNABILITY_GO` 已由 conditional-magnitude GO 满足

## 1. 提案背景

P2 learnability gate 已在 **conditional magnitude estimand**（endpoint_v5）下判定
**GO**（fail-closed）：对 caller_v3 真实 changer，其 profile-level |Δr| magnitude
可由 allowed inputs 跨 publication 学习（deepsets mean skill 0.677，paired
publication-block bootstrap CI low 0.323 > 0，10/10 held-out pubs 方向一致）。

合同 Phase 3（模型架构迭代）的前置依赖 "Phase 2 GO" 已满足，输入/输出/endpoint
（endpoint_v5）冻结。本提案为 Phase 3 申请 authority epoch 18，并预先写入每方案
预算、最大迭代轮数与停止规则（合同要求写入 authority）。

## 2. 范围（严格限定，只检验合同规定的三种能力）

| 方案 | 能力 | 说明 |
|------|------|------|
| 方案一 (pair/conditional head) | pair-level 对齐 + conditional | 主方案；conditional magnitude head（优先，直接承接 GO 的 deepsets 能力） |
| 方案二 (exact_alt_v1) | exact-alt × WT-state generic interaction | generic interaction 基线 |
| 方案三 (epro_v2) | repaired EPRO（可控 nonlocal 传播） | 若方案一/二不足以区分归纳偏置 |

- 每次只改变一个能力（合同 ⑤）；5 seeds 配对消融（合同 ⑥）
- 容量/训练时间与 generic 匹配（合同 ④）；梯度/residual/NaN/长度/内存测试（合同 ⑦）
- development winner 按预注册规则冻结（合同 ⑧）

## 3. 预算与迭代规则（预先写入 authority）

- **算力**: 全部训练/验证使用 GPU（CUDA 不可用或 CPU 静默降级则 STOP 并留证据）
- **seed**: 5 个确定性 seed (0,1,2,3,4)，固定预算、固定 budget
- **模型规模**: 每方案与 capacity-matched generic 对齐（参数量/FLOPs/训练时间匹配）
- **最大迭代轮数**: 每方案最多 **2 轮** 核心迭代（合同 Phase 3 失败处理）
- **停止规则**:
  - 单方案在 development outer folds 相对 capacity-matched generic 的 CI 下界 ≤ 0
    → 该方案立即退役
  - 全方案不胜 generic → 使用最简单 generic 并转 benchmark/resource 路线，
    **不再开 dev13+ 自由搜索**
  - 不靠 post-hoc aggregation；不事后调 test/held-out

## 4. 验收标准（合同 Phase 3）

- candidate 相对 capacity-matched generic 在 development outer folds CI 下界 > 0
- exact-alt/nonlocal 消融符合预注册方向
- 无梯度/收敛失败；增益大于 seed variance
- 不靠 post-hoc aggregation 翻转

## 5. 涉及新增模块

`reactflow_delta/models/pair_v1.py`, `exact_alt_v1.py`, `epro_v2.py`,
`train_v2.py`, `samplers.py`, `tests/reactflow_delta/test_model_invariants_v2.py`
（旧 dev01–12 不原地改写）

## 6. 明确不授权 / 边界

- 不降低主任务 endpoint_v4 STOP；不覆盖任何 frozen verdict
- 不访问/解开 confirmatory test（`OLD_TEST_RETIRED_NEW_TEST_UNTOUCHED` 保持）
- 无 certified untouched confirmatory publications（ge3_untouched=False）→ Phase 4
  confirmatory 需另行解决 publication 身份或 prospective cohort，**不在本 epoch 授权内**
- `scientific_claim_authorized = false`：Phase 3 产出为 development/architecture
  证据，不构成 SOTA 或跨 publication confirmatory 主张

## 7. 申请授权

请求用户显式授权 **authority epoch 18, Phase 3 模型架构迭代**（以 conditional
magnitude pair head 为主方案），并批准上述预算/最大 2 轮/停止规则。
