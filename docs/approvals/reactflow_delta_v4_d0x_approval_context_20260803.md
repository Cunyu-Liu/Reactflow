# ReactFlow-Delta V4 批准上下文记录

- 记录时间：`2026-08-03T22:40:15+08:00`
- 会话角色：当前 Codex 任务中的用户
- 用户消息原文：`批准执行`
- 原文字节编码：UTF-8，无尾随换行
- 原文字节 SHA-256：`2769e7199035a5e0f5da59ab8dc837c7e7d3c78add9f839d996c6ca7e00714a2`
- 被批准的合同：`reactflow_delta_v4_data_first`
- 合同 raw-byte SHA-256：`631962f88790103aa3383c9ed22de2943f6874455b4fcb587e18eb2a7d277c15`

## 上下文解释

本消息紧接用户给出的《ReactFlow-Delta V4 数据优先救援与合同重写计划》。该计划明确规定：用户审阅并批准 V4 后，后续从 `D0-X Exact Mutation Recovery` 数据 Gate 开始；本轮不得训练、不得生成新训练数据集、不得重建 split、不得运行 baseline/P2/EPRO，也不得解封确认性 test。

因此，本记录把“批准执行”解释为两个严格受限且顺序执行的授权：

1. 批准精确 hash 的 V4，允许在全部机器验收通过后将 `RECOVERY_CONTRACT_REWRITE` 收口为 `TERMINAL/PASS`；
2. 允许创建只覆盖 `D0-X` 的 authority amendment，并仅在 source universe、许可、fixture、Git 和运行边界全部冻结后执行 D0-X。

## 明确不包含的授权

本批准不授权 `D1-X`、canonical dataset、split、normalization/threshold fitting、任何模型或基线训练、sealed test 访问、跨项目导出、湿实验、push 或 PR。

## 身份与签名限制

本记录只证明当前任务中由平台标记为用户角色的消息及其上下文。平台事件 ID、账户实名、原始事件导出、密码学签名均不可从当前执行环境取得，故不得声称已验证；这些字段在机器记录中使用 `NOT_AVAILABLE_NOT_ASSERTED` 或 `NOT_PROVIDED_NOT_ASSERTED`。

