# ReactFlow-Delta V4 Recovery 人工审计与前向修复记录

审阅者角色：`CODEX_PRIMARY_IMPLEMENTATION_AGENT`

外部身份保证：`NOT_EXTERNALLY_VERIFIED`。本记录仅陈述当前任务中执行者角色，不声称独立审稿人或密码学签名。

## 抽查与全量核验范围

本审计逐项检查了冻结 V4 的 Recovery 条款、批准记录、V3–V3.5 保存状态、V4 与 mRNA interface hash、pilot closure 外部清单、Git parent/diff、detached ledger、sentinel、active 权限、预存 untracked 文件以及当前进程边界。

## 发现

### F-RECOVERY-001：原 authority receipt 完整，但 formal phase manifest 缺项

提交 `940697de985dbd7425112babd2d429e81ddde4cb` 的 Git/hash/权限边界全部可重放；但是其 `recovery_v4_terminal_manifest_20260803.yaml` 未显式包含 V4 第 20.1 节要求的 `automated_tests` 与 `manual_audit` 对象，也没有 Recovery finalizer artifact。因此它只能作为 `RECOVERY_HASH_AND_GIT_INTEGRITY_EVIDENCE`，不能单独支撑 formal PASS。

处置：`REPAIR_FORWARD_REQUIRED`。不得修改或删除该历史提交；使用新 source commit 加入只读 validator、自动测试与 finalizer，再在 Git 外部不可覆盖的 Recovery root 生成正式 terminal manifest、ledger 和 sentinel，最后由新的 focused authority child 绑定这些字节。

### F-RECOVERY-002：raw document status 与 effective authority status 需要分轴

冻结 V4 raw bytes 仍写 `DRAFT_AWAITING_USER_SIGNOFF`，而用户随后在当前任务中发送“批准执行”。不能回写冻结合同来改变原文。

处置：新 active manifest 必须同时保留：

- `raw_document_declared_status: DRAFT_AWAITING_USER_SIGNOFF`；
- `effective_authority_status: V4_ACTIVE_AUTHORIZED_GOVERNANCE_ONLY`；
- approval 的平台角色、上下文解释和无密码学签名限制。

### F-RECOVERY-003：历史 epoch head policy 不能直接套用于未来 HEAD

epoch 1 的 exact-one-child 条件只描述提交 `940697d...` 相对其父 `10e5241...` 的历史事实。未来 source/authority commit 会自然使当前 HEAD 改变。

处置：新 finalizer 使用 `git show <receipt_commit>:<path>` 重放历史 blob、ledger 和 sentinel；后续 D0 authority 绑定新的 Recovery repair receipt，而不是要求 epoch 1 始终为当前 HEAD。

## 通过项

- V4 raw SHA-256：`631962f88790103aa3383c9ed22de2943f6874455b4fcb587e18eb2a7d277c15`。
- V3 raw SHA-256 未改变：`3efcc1504208d8089236dfe4e7d41553741441d3b86b6174c8b5af52d614ec10`。
- 原 receipt 为单亲、非 merge，父提交 `10e52412a1612667993209e56117b1608a084297`，diff 精确 7 个治理文件。
- epoch 1 ledger 八个成员和 sentinel cross-hash 可从 Git blob 重放。
- pilot closure 最终 ledger 全部通过，scientific status 仍为 development-only。
- approval 未伪造用户实名、平台 event ID 或密码学签名。
- Recovery 时没有 full recall、dataset、split、training、test 解封、跨项目导出或湿实验。
- 预存的 6 个用户 untracked 文件路径和 bytes 未改变。

## 人工审计结论

`PASS_AFTER_FORWARD_REPAIR_ONLY`。只有新的 committed finalizer source 实际运行成功、外部 terminal manifest 显式包含 `automated_tests`/`manual_audit`/`finalizer`、checksum ledger 和 sentinel 全部闭合，并由新的 authority child 绑定后，Recovery 才可解释为 formal `TERMINAL / PASS`。

