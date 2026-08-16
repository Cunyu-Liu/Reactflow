# ReactFlow-Δ M0-X Controlled Development 执行计划（草案）

> 中文名称：ReactFlow-Δ EPRO-Small 受控开发执行计划（草案）
> 计划版本：V0.1（草案，待确认）
> 计划日期：2026-08-04
> 适用阶段：M0-X（Controlled Development）
> 主合同：`提示词/ReactFlowDelta科研合同_v4_data_first.md`（§10、§12、§13、§14、§16、§17、§18、§19、§20.10）
> 证据类别：`DEVELOPMENT_ONLY`
> 前置阶段：O0-X PASS（terminal manifest SHA `815168cc`）
> 对应仓库工作树：`/home/cunyuliu/reactflow_delta_goal_20260729`

---

## 1. 目标与依据

### 1.1 目标

在固定 development seed 与冻结的开发数据/split/caller/evaluator/baselines 下，比较
**EPRO-Small、matched generic、from-scratch/pretraining 双臂**（§20.10），证明受约束
EPRO 在相同容量/预算下提供超越 strongest simple baseline 与 generic 的稳定增量。
产出唯一 frozen final candidate 或诚实的 negative close。

### 1.2 主合同回顾（M0-X 关键约束）

| 条款 | 要求 | 本计划覆盖 |
|---|---|---|
| §14 容量阶梯 | 必须 low→high 执行；50k-250k 为 generic/EPRO-Small rung | Step 3 |
| §12/§13 统计 | 双 estimand、cluster CI、group-aware permutation、唯一 selection rule | Step 5 |
| §16 预训练双臂 | Arm A from-scratch / Arm B 静态预训练，共享全部协议 | Step 4 |
| §17 开发窗口 | 28 天 / 6 迭代，每轮一假设一 seed，run_id 唯一，早停 | Step 6 |
| §18/§19 | 真实 CUDA fallback=0、manifest/finalizer/ledger、Git 安全 | Step 7 |
| §20.10 PASS/FAIL | 预注册 CI/permutation/calibration 达标 + 唯一候选冻结 | Step 8 |

---

## 2. 前置条件（全部须 PASS）

- [ ] **O0-X PASS**：terminal manifest SHA `815168cc`，`O0X_CLOSED.yaml` sentinel 存在。
- [ ] **D2-X / PH0-X**：publication split frozen（`d2x_split_publication_20260804T1600+0800`），
      test 密封未访问，`test_access_ledger` 为空。
- [ ] **B0-X PASS**：strongest simple baseline 冻结（`wt_only`），P2 paired baseline =
      `b0x_strong_baseline_20260804_v1`（20,737 参数，WMAE skill 0.0788，cluster CI low 0.0029，
      permutation p=0.0099）。
- [ ] **M0-X 权威修正案（epoch 12）**：用户签署，active contract 更新为 `M0-XAUTHORIZED`。
- [ ] 服务器：GPU 可用（用 `CUDA_VISIBLE_DEVICES=1`，**禁止占用 GPU 4**），磁盘配额充足。
- [ ] 评估器：`static_v1.yaml`（13 个 gold fixtures + 15 个契约测试）已就绪。

---

## 3. 数据与输入（frozen）

| 项 | 值 |
|---|---|
| UI 数据可用性 | `TIER_B_PLUS_DATA_CANDIDATE`（tier_a_plus_data_ready=false） |
| primary pairs | 4,472（train 3,516 / validation 548 / test 408 密封） |
| studies / parents / publications | 9 / 42 / 8 |
| 层级 | study → parent → pair（nucleotide position 非独立样本） |
| P2 输入 | WT-anchored（可读 WT profile，不读 mutant profile/test） |
| forbidden inputs | mutant profile、test target、caller post-hoc label、held-out normalization stats |

所有输入 hash 锁定：data/split/caller/evaluator/baselines/seed 的 SHA-256 写入每轮 run manifest。

---

## 4. 模型、容量阶梯与双臂

### 4.1 容量阶梯（§14，从低到高执行）

| 级 | 模型 | 状态 |
|---|---|---|
| 1 | zero / train mean / mutation-type / edit-only / WT-only | B0-X 已冻结（`wt_only` 为 strongest trivial） |
| 2 | thermo-only linear/ridge + 简单 tree | 需在 M0-X 评估并报告 |
| 3 | 10k-100k 参数 P2 paired baseline | B0-X 已冻结（20,737 参数） |
| 4 | **50k-250k generic paired + EPRO-Small** | **M0-X 目标 rung** |
| 5 | ≤1M EPRO-Lite | 本次不授权（需 Tier A+，4472<5000 未达） |

### 4.2 对比臂（§20.10）

- **EPRO-Small**：受约束 EPRO 算子，50k-250k 参数。
- **matched generic**：与 EPRO-Small 共享 exact records/split/mask/input/heads/optimizer/
  budget/early stop/selection/evaluator/参数±5%。

### 4.3 预训练双臂（§16）

- **Arm A**：from-scratch。
- **Arm B**：Ribonanza / 旧 ReactFlow-Structure 静态 structure/reactivity 预训练。
- 预训练 profile 不计入 Delta pair 数；两臂共享同一数据/split/mask/容量/预算/evaluator/
  selection/test；Arm B 必须完成 exposure ledger 审计（§16.2）。

### 4.4 分头建模（§15.3）

- changer head：概率、Brier/log loss/calibration；
- conditional signed magnitude head：Student-t location/scale/df（仅 caller-defined changer 参与）；
- all-position continuous head：共同次级 raw-scale response。
- 三个 head 的 loss/metric/梯度范数/mask count 分开记录；不得用真实 effect 大小制造 active signal。

---

## 5. 评估与统计（§12、§13）

### 5.1 双 estimand

- **Estimand A**：可重复 changer detection/ranking（主 comparand）。
- **Estimand B**：changer 条件下 signed magnitude（点估计 ≥5% 相对改善才可保留 headline）。
- **共同次级**：全位置连续 Delta。

### 5.2 统计协议

- 主比较：cluster-level paired loss difference `d_g = L_g(main) - L_g(comparator)`；
- 报告跨 study/parent clusters 的点估计、95% CI、cluster 数、pair 数、changer 数；
- CI 方法/重采样层级/次数/seed 在 test 解封前冻结；
- ratio Skill 只用 pooled ratio-of-sums（§13.2）；禁止对 near-zero pair 的 ratio 宏平均。

### 5.3 唯一 selection rule（§13.5，先冻结）

1. 剔除 run closure/invariant/reliability FAIL、Brier 或 log loss 劣于 strongest baseline、
   不满足预注册 risk-coverage 约束的候选；
2. 第一选择量 = **study-macro AUPRC gain over strongest baseline**（study 等权，study 内
   parent/pair 计算，CI 用 study→parent cluster bootstrap）；
3. 在 one-standard-error 内依次：参数更少 → parent-macro K=10 top-k recall gain 更高 →
   Brier 更低 → log loss 更低 → 相同则按 model_id lexical order（**禁止**看 magnitude/
   continuous/test 决定）。

### 5.4 calibration（对准 E0-X §20.11.1）

- 在 PH0-X 预隔离的 calibration fold（development groups 内）nested 拟合，按 study/parent 分组；
- 报告 Brier、log loss、calibration slope/intercept、fixed-bin ECE、90%/95% coverage/width、cluster CI；
- 非 E0-X 硬 Gate，但为 M0-X PASS 的辅助证据。

---

## 6. 开发窗口（§17）

- **window_id**：`m0x_dev_window_20260804`
- **start**：首个已授权 prediction-changing run 的 UTC 时间；**deadline** = 其后 28 calendar days。
- **最长迭代**：`EPRO_DEV_01` 至 `EPRO_DEV_06`（最多 6 次科学迭代）。
- 时间或次数任一先到即关闭；finalizer 取两者先到者，人工不得重置。
- **每轮**：一个预注册科学假设、固定同一 development seed、唯一 run_id + parent_run_id、
  失败 artifact 不覆盖。
- **计轮条件**（§17.6）：任何改变 prediction 的 data eligibility/caller/target/loss/feature/
  architecture/capacity/initialization/optimizer/calibration/checkpoint selection 都计一轮。
- **infrastructure-only retry**（§17.7）：scientific inputs byte-identical + 故障 evidence 完整
  + 新 run ID，可不计科学迭代。
- **早停**（§17.8）：reliability/permutation/provenance Gate 失败，或连续 3 轮不能优于
  strongest simple baseline，立即早停。
- **window registry**（机器可读）：window_id、authorization_sha256、window_started_at_utc、
  window_deadline_utc、maximum_iterations=6、consumed/remaining_iterations、每轮
  run_id/parent_run_id/hypothesis_id/change_category/prediction_changing/counts_as_iteration/
  evidence_sha256/status。

### 6.1 计划迭代框架（预注册）

| 迭代 | hypothesis_id | 变更类别 | 内容 |
|---|---|---|---|
| EPRO_DEV_01 | m0x_h01_frozen_fromscratch | 基线匹配对拍 | 从scratch训练 EPRO-Small + matched generic + P2 paired baseline，validation 评估，冻结 selection metric |
| EPRO_DEV_02 | m0x_h02_pretrain_arm | 预训练双臂 | Arm B 静态预训练 + 微调，与 Arm A 对比，exposure ledger 审计 |
| EPRO_DEV_03 | m0x_h03_refinement | loss/calibration | 首个预注册 refinement（如 loss 加权/calibration 拟合），仅 validation |
| EPRO_DEV_04-06 | 待预注册 | 受控 refinement | 基于前轮 evidence 的预注册 refinement 或 early-stop |

> 注：以上为框架，每轮具体假设在开始时以 `m0x_preregistration.json` 单独冻结，先于该轮 prediction 生成。

---

## 7. 执行步骤

### Step 1：M0-X 权威预检（fail-closed validator）

新建 `scripts/reactflow_delta/m0x_validate_authority.py`，校验：
- current_phase == M0-X，runnable_phases == [M0-X]；
- O0-X terminal PASS（SHA `815168cc`）；
- training_allowed == True；
- 评估器/static_v1 契约测试通过。

任一失败 → 生成 preflight failure artifact，停止。

### Step 2：冻结窗口 registry 与预注册

- 写 `m0x_dev_window_20260804` registry（§6 字段）；
- 写 `EPRO_DEV_01` 的 `m0x_preregistration.json`（假设、变更类别、selection rule、指标、seed）；
- 锁定 data/split/baseline/evaluator/seed 的 SHA-256。

### Step 3：容量阶梯评估（只读，不训练）

- 评估并 report 阶梯 2（thermo-only linear/ridge + tree）与阶梯 3（P2 paired baseline，复用 B0-X）
  在 validation 上的指标；确定 "strongest simple baseline" 的最终对照集合（§14）。

### Step 4：EPRO_DEV_01 —— from-scratch 匹配对拍

- 训练 EPRO-Small + matched generic（同一 budget/seed/early stop/selection）；
- 真实 CUDA、fallback=0、GPU 1；
- 全 run manifest/finalizer/ledger；
- 计算 study-macro AUPRC gain over strongest baseline + cluster CI + permutation。

### Step 5：EPRO_DEV_02 —— 预训练双臂

- Arm B：静态预训练（Ribonanza/旧 ReactFlow-Structure）+ 微调；
- exposure ledger 审计（§16.2）；
- 对照 Arm A from-scratch；评估 pretraining 是否带来稳定增量。

### Step 6：EPRO_DEV_03-06 —— 受控 refinement（每轮预注册）

- 每轮一个预注册假设；只改一项；只评估 validation；
- 早停判断（连续 3 轮不胜 / reliability/permutation drift）；
- 每轮独立 manifest/finalizer/ledger。

### Step 7：唯一 final candidate 选择

- 按 §5.3 selection rule 在 validation 上冻结唯一 candidate + 全部协议（model/threshold/
  pretraining/split/metrics/baseline/tie-break）；
- 冻结前 test 仍不可见。

### Step 8：审计、finalizer、sentinel 与提交

- 完整审计；finalizer 写 `m0x_terminal_manifest.yaml`、SHA256SUMS、`M0X_CLOSED.yaml`（或 FAIL evidence）；
- 更新 active contract 至 M0-X terminal（PASS→route E0-X；FAIL→route REPORT-X）；
- focused commit、push 到 `origin/codex/reactflow-delta-d0r`。

---

## 8. 交付物

- M0-X 权威修正案（epoch 12）+ 批准记录
- `m0x_dev_window` registry（机器可读）
- 每轮 `m0x_preregistration.json` + run manifest + finalizer + ledger
- 容量阶梯评估报告（阶梯 2/3/4）
- EPRO-Small vs matched generic vs strongest baseline 横向对比表（study-macro AUPRC、cluster CI、permutation、Brier/log loss、calibration）
- 预训练双臂对比（Arm A vs Arm B）+ exposure ledger
- 唯一 final candidate + 协议冻结（或 negative close）
- `M0X_CLOSED.yaml` sentinel + SHA256SUMS ledger
- 单元测试（所有新代码配套）

---

## 9. 验收标准（§20.10 PASS）

- development 上预注册的**主 CI/置换/calibration criteria** 全部满足；
- 相对 strongest simple baseline 的 cluster CI 下界 > 0（§13.4.2）；
- 相对 matched generic 的 cluster CI 下界 > 0（§13.4.3）；
- 真实标签优于 group-aware permutation null（§13.4.1）；
- 无单 study/parent 驱动（leave-one-study/parent sensitivity 方向稳定，§13.4.5）；
- 唯一 final candidate 与全部协议冻结，test 未访问。
- 任一 FAIL → 路由 REPORT-X（诚实 negative benchmark/P2/resource 证据）。

---

## 10. 停止规则（stop rule）

- 连续 3 轮不能优于 strongest simple baseline → 早停；
- 28 天或 6 迭代任一先到 → 关闭窗口；
- 禁止换 seed、扩大模型、增加第 7 轮、换 test 或改名重启窗口；
- window 关闭后若 final Gate 不通过 → 自动转 data/provenance resource、negative benchmark 或 P2 result，不再扩大 EPRO。

---

## 11. 未决问题（需用户确认）

1. **M0-X 权威修正案（epoch 12）**：授权训练 EPRO-Small/generic（50k-250k）与预训练双臂，且明确禁止 EPRO-Lite（本次 Tier B+ 未达 Tier A+）。是否签署？
2. **预训练 Arm B 数据源**：Ribonanza 是否可用？还是只做 from-scratch（Arm A）单臂？这决定是否引入外部预训练依赖。
3. **迭代预算**：是否按 6 次迭代上限推进，还是先执行 EPRO_DEV_01-02 后由 evidence 决定是否继续？
4. **GPU / 计算预算**：确认用 GPU 1（`CUDA_VISIBLE_DEVICES=1`，避开 GPU 4），每轮训练预算上限（如时间/epoch）待定。