# ReactFlow-Delta V4 Recovery 验收记录

记录时间：`2026-08-03T22:40:15+08:00`

## 结论

`RECOVERY_CONTRACT_REWRITE` 可在且仅在本 authority child commit 的运行时 Git head policy 验证通过后登记为 `TERMINAL / PASS`。该 PASS 只表示治理与 pilot 收口完整，不是数据 Gate、模型 Gate 或科学结果。

## 已定位证据

| 项目 | 结果 | 证据 |
|---|---|---|
| V4 raw bytes | PASS | `docs/contracts/ReactFlowDelta科研合同_v4_data_first.md`；SHA-256 `631962f88790103aa3383c9ed22de2943f6874455b4fcb587e18eb2a7d277c15` |
| 原 V3 保持不变 | PASS | SHA-256 `3efcc1504208d8089236dfe4e7d41553741441d3b86b6174c8b5af52d614ec10` |
| 用户批准上下文 | PASS_WITH_IDENTITY_LIMITATION | `docs/approvals/reactflow_delta_v4_approval_20260803.yaml`；平台事件 ID 和密码学签名不可取得，未作虚假断言 |
| Pilot closure | PASS | 外部 closure manifest SHA-256 `04b2749642e534fe7007905e1e43eaa204829980506aac11b456913165babdb4` |
| Pilot closure ledger | PASS | 外部 `SHA256SUMS` SHA-256 `6c9ddf9e5cd187c1154ca7d1553464e99d74bcdd4280b90502adb8c2334b3c72`，重新执行逐项校验全部通过 |
| Pilot terminal marker | PASS | 外部 `DEVELOPMENT_CLOSED` SHA-256 `dd531569f4ff90f2e72f72d89ddb0f05b60bdf59cf105aa776fbe1d5ceca1dde` |
| Pilot claim boundary | PASS | 1,509/7,660 均为 `DEVELOPMENT_ONLY / CONTRACT_NONCONFORMING / NO_CONFIRMATORY_CLAIM` |
| mRNA additive interface | PASS | SHA-256 `65f847e7d799789d50652f6ccb8558cd62e6743ec36551ea51bf5720ce8cec52`；oracle 仍未授权 |
| 新数据召回 | NOT_RUN | Recovery 阶段未下载或解析新数据 |
| 数据集与 split | NOT_RUN | 未生成 canonical dataset，未建立或修改 split |
| 模型与 baseline | NOT_RUN | 未训练、未评测、未解封 test |
| push / PR | NOT_RUN | 本轮默认禁止 |

## Git 生效条件

本记录不嵌入包含自身的未来 commit hash。Recovery authority 只有在运行时同时满足以下条件时生效：

1. authority commit 为单亲、非 merge commit；
2. 第一父提交精确为 `10e52412a1612667993209e56117b1608a084297`；
3. diff 仅包含 approval、Recovery manifest、active manifest、detached ledger、sentinel 和本验收记录；
4. tracked worktree clean；
5. 预存的 6 个用户 untracked 文件路径与 SHA-256 均未改变；
6. detached ledger 与 sentinel 的所有 hash 验证通过。

若任一条件失败，Recovery 状态必须解释为 `FAIL_CLOSED_NOT_EFFECTIVE`，不得以 Markdown 中的 PASS 文本替代机器证据。

