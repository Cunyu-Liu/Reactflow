# ReactFlow-Delta 两批非正式 pilot 收口报告

状态：`DEVELOPMENT_CLOSED`  
`evidence_class=DEVELOPMENT_ONLY`  
`contract_conformance=CONTRACT_NONCONFORMING`  
`claim_eligibility=NO_CONFIRMATORY_CLAIM`  
机器状态轴：`lifecycle_status=TERMINAL`；`closure_gate_result=PASS`；`scientific_gate_result=NOT_RUN`；`terminal_marker=DEVELOPMENT_CLOSED`  
Gate 边界：仅 closure Gate=`PASS`；scientific Gate=`NOT_RUN`。本报告只证明安全收口与证据保全，不证明模型或科学结果通过。

## 1. 收口对象与结果

| Pilot | 精确 PID | 启动时间 | GPU UUID | 最佳 checkpoint | 最后日志轮次 | 收口结果 |
|---|---:|---|---|---|---:|---|
| 1509 | 3881872 | 2026-08-03 15:29:36 +08:00 | `GPU-1bbaf049-a78e-a985-19d4-02448974ccf8` | epoch 79, val_skill -0.015123 | 370 | SIGTERM 后 `/proc` 与 GPU 进程表均无该 PID |
| 7660 | 4172301 | 2026-08-03 15:53:47 +08:00 | `GPU-a590f174-04eb-13f7-58a4-1067bc8bd765` | epoch 19, val_skill 0.007754 | 90 | SIGTERM 后 `/proc` 与 GPU 进程表均无该 PID |

温和终止前，对每个进程连续做了两次完全相同的身份校验：所有者、PID/PPID、`/proc` 启动时钟、8 个命令参数、工作目录、解释器、stdout/stderr 句柄、`CUDA_VISIBLE_DEVICES` 和 GPU UUID 必须同时匹配。两轮均 PASS 后，信号只发送给两个 Python 子 PID；没有使用进程名匹配、进程组终止、`SIGKILL`，也没有向父进程或其他 GPU 作业发送信号。`kill` 返回 0 不是进程退出码；没有取得 wait status，因此只能声称“发送 SIGTERM 后观察到目标进程消失”，不能声称正常或 checkpoint-aware graceful completion。

## 2. 已保全证据

成功收口根目录：

`/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/recovery/pilot_closure_20260803T200500+0800_r2`

其中包括终止前进程/GPU/Git/reflog/源码快照、当前配置快照、完整最终日志、两个最佳 checkpoint、checkpoint CPU 可读性结果、终止守卫、终止后进程/GPU 复核、机器可读 closure manifest、terminal sentinel 与最终 SHA-256 ledger。最终 ledger 对其列出的每个文件均验证通过；封存方式是“checksummed + 只读权限”，不是物理 WORM。

关键 SHA-256：

| 文件 | SHA-256 |
|---|---|
| 1509 最佳 checkpoint | `15ccb6418efe886e1934eb43d9f8f6b7655af9433f1f2221d485488ad43c5574` |
| 7660 最佳 checkpoint | `26a01f16cb43e11a1656358e65fabe273497d8b577c54509a0e96657b5c0b34e` |
| 1509 最终日志 | `e2b1e24342b0c168416958de0300b152decf9cc3ce02f39b53507cc3148057cb` |
| 7660 最终日志 | `85ff27688642a2b409ec051817b42627f5b9934cbe3c1aafe07e2b73f069b01f` |

第一次 snapshot 目录因命令中的 `find -printf` 条件错误而提前失败；在失败前未发送任何信号，该目录保留为 `FAILED_INCOMPLETE_PRETERM_EVIDENCE`，不能作为 canonical closure。成功目录中的第一次守卫执行又因脚本控制流提前退出而未发送信号；复核两个进程仍在后，修正脚本并重新做双重身份校验。这些失败没有被删除或改写。

终止前 ledger `SHA256SUMS.preterm` 作为历史证据原样保留；它的 `termination_guard.txt` 条目会失败，因为 ledger 生成后同一 guard 文件又追加了即时二次校验和信号结果。最终 `SHA256SUMS` 在所有写入结束后重新生成并逐项 PASS，不能把旧 preterm ledger 误报为全 PASS。

## 3. Provenance 限制

进程启动窗口对应 reflog 中可定位的提交是 `b9a52c6cd5fcc4f7025c75dfe6454eae84887e32`，收口预检时仓库 HEAD 为 `6515667020184fa6e5f8dc70acd199b2c3a8fbcb`。但启动时工作树、配置和源码字节没有预先冻结；因此本报告只写：

`runtime_source_commit_binding = NOT_AVAILABLE_NOT_ASSERTED`

收口时保存的源码与配置 hash 只能证明“当前路径快照”，不能反推训练启动时精确执行字节。runner 没有 checkpoint-aware 信号处理、`finally` finalizer 或 natural-run terminal sentinel；温和终止不会保存 best checkpoint 之后的内存状态。`cuda_fallback_count` 也没有被 runner 记录。

证据位于 NFS；观测到其存储 mtime 比 compute host 写入文件内容的 ISO 时间慢约 104–107 秒。事件顺序以文件内嵌 ISO 时间为准，NFS mtime 只保留为存储元数据，不能把该时钟偏差解释为事件倒置。

## 4. 科学与合同边界

- 1509 的日志最后打印 epoch 370（loss 约 0.0113；最后一次 evaluation 在 epoch 360，val skill 为 -0.1183）；7660 最后打印 epoch 90（loss 约 0.5204；最后一次 evaluation 在 epoch 80，val skill 为 0.0063）。两者都没有 runner 的 `Done` 行。它们与用户观察到的稀疏目标退化一致，但不是正式统计检验。
- 两个目录都只有“验证指标改善时保存的最佳 checkpoint”，没有自然结束产物、正式 run manifest、结构化 metrics、finalizer 或 checksum closure；checkpoint 不能等同于正式 PASS。
- 1509/7660、M0/M0-R/M0-R2/R3 一律登记为 `DEVELOPMENT_ONLY / CONTRACT_NONCONFORMING / NO_CONFIRMATORY_CLAIM`。
- 两批数据和 checkpoint 禁止进入 V4 Gate lineage，禁止计入 Tier B+/A+ exact pair 数，禁止作为 untouched test，禁止支持 EPRO 成功或失败的确认性主张。
- Csde1/Tetrahymena 相关 test 已开发消费，机器状态为 `DEVELOPMENT_CONSUMED`，不得改名后再次解封。

## 5. 工作树保护

收口前后 tracked worktree 均无新增修改；原有 6 个 untracked 文件保持原状。终止后 GPU 表仍有无关作业存在，但目标 PID 已消失；本次动作没有向这些作业发送信号，不扩张为“服务器所有无关状态均未变化”的主张。没有把 raw data、checkpoint、权重、cache 或 `/tmp` 日志加入 Git。成功 closure 位于项目专属 `/mnt` artifact root；远端 Git 只会保存本小型报告及其外部证据指针。

本次收口没有执行 RMDB 全库下载、构建新 dataset、重建 split、运行 baseline/P2/EPRO、push 或创建 PR。
