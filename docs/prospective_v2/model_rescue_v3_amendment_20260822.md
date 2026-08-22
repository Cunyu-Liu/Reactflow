# ReactFlow-Delta Model Rescue v3 阶段性合同

日期：2026-08-22  
当前执行状态：`R3M2_ENGINEERING_SMOKE_ONLY`
父合同：Model Rescue v2 终局 `TERMINAL_R2M3_MEAN_GATE_FAIL_CALIBRATION_BASELINE_ONLY`  
父合同终局 commit：`97ce496c4dda944d0554b49342ce388d8f9d97c1`

## 1. 合同地位与授权边界

本文件是用户在 v2 终局后明确要求继续提升模型并达到原 R2M4 门槛所建立的独立
amendment。它只覆盖一个由完整 OOF 失败图谱支持的专家分歧 gate；不得修改或重新解释
Model Rescue v1 的 `M2_NO_RESCUE_CANDIDATE`、v2 的 `MODEL_RESCUE_V2_FAIL`，也不得把
v2 的 0.8833% 改写为 PASS。

v2 对“第三次 rescue”的禁止在 v2 合同内部仍保持原文和终局效力。本 amendment 的新
执行权限直接来自 2026-08-22 用户指令；这是一项向前生效的 owner override，不是对旧
合同的追溯篡改。20 个 OpenKnot puzzles 继续标为 `DEVELOPMENT_CONSUMED`。本合同不读取
任何新 external outcome，不建立 external、SOTA、mechanism 或 publication PASS。

R3M1 只允许代码与不变量测试；在实现 commit、validator 与真实数据 smoke 全部通过前，
`training_allowed=false`。任何 partial fold score 都不得用于修改候选、门槛或阈值。

## 2. 已确认问题与排除方向

完整 seed-0 20-fold OOF 诊断确认：

- v2 MeanAligned 相对 B1 的 signed-delta 改善为 0.883309%，16/20 puzzles 正向；
- calibrated residual 的 CRPS gain 为 +0.00547832，20/20 puzzles 正向；
- `|target delta|>0.20` 区域的 signed-delta MAE 相对恶化 2.5066%，而 near-zero 区域
  改善 34.7801%；
- 训练与 evaluator 的位置/突变聚合层级虽在实现上不一致，但在当前真实数据上数值完全
  相同，不能解释性能瓶颈；
- method-conditioned blend 的诊断结果较强，但 endpoint v7 不允许 design-method label，
  因此该方向不具候选资格；
- 四段 magnitude/disagreement gate 不优于两段 disagreement gate，因此不保留。

唯一保留的诊断方向使用合法输入 `|mu_B1-mu_MeanAligned|`：完整但非资格性 OOF probe
得到 signed-delta 相对改善 1.239336%、19/20 puzzles 正向、CRPS gain +0.00657513、
20/20 puzzles 正向。该 probe 仅用于冻结方向，因为不同 OOF 专家之间可能间接见过 probe
held puzzle；它不得成为 R3M3 或 R3M4 的证据。

## 3. 核心问题与可证伪假设

问题：在不改变 endpoint-v7 输入权限、不增加 backbone 宽度/深度和不读取 held outcome
的条件下，B1 的尾部能力与 MeanAligned 的近零/中等效应能力能否通过 fold-legal 的低容量
disagreement gate 组合，并在零均值 residual calibration 后同时越过原 R2M4 CRPS 和
signed-delta MAE 门槛？

假设：当两个专家高度分歧时，MeanAligned 的 L1 shrinkage 更可能损害大效应预测；当分歧
较小时，MeanAligned 的 conditional-median 优势占主导。由 outer-train inner-OOF 数据拟合
两个 convex weights 可保留两种能力：

\[
d_i=|\mu_{B1,i}-\mu_{MA,i}|,
\quad
\alpha_i=\begin{cases}\alpha_{lo},&d_i\le q_{0.95}\\
\alpha_{hi},&d_i>q_{0.95}\end{cases},
\]

\[
\mu_{blend,i}=\mu_{B1,i}+\alpha_i(\mu_{MA,i}-\mu_{B1,i}),
\qquad 0\le\alpha_{lo},\alpha_{hi}\le1.
\]

反证包括：合法 inner-crossfit gate 的 seed-0 完整 screen 未通过双 Gate；或正式五 seed
mixture 未通过任一原 R2M4 条件；或 target-invariance/coverage/zero-mean invariant 失败。

## 4. 唯一候选与公平性披露

候选 ID：`b1_meanaligned_disagreement_gate_calibrated_residual`。

### 4.1 两个冻结专家

- Expert B1：`b1_rfd_direct_aligned`，沿用 v1 的 fold-legal normalization、delta
  parameterization、train/eval distribution alignment、method-balanced training 和
  train-only calibration；`d=96`、4 heads、hidden=64、K_rank=0、40 epochs。
- Expert MA：v2 `b1_mean_aligned`，相同 B1 encoder 容量，exact method-balanced
  signed-delta L1、Adam lr 1e-3、weight decay 0、40 epochs、clip 5、无 early stop。
- 两个专家每 fold/seed 独立从头训练；不得使用 held target/error/mask。
- 不扩大单个 backbone，但候选同时运行两个 backbone，mean-path 参数量和推理计算约为
  单 B1 的 2 倍。论文、表格与 SOTA 比较必须披露这一点，不得称为等计算量提升。

### 4.2 Gate 的唯一合法输入与拟合

gate 应用阶段只读取两个冻结专家的 per-position signed-delta predictions。禁止读取：

- design method、puzzle ID、dataset ID、publication ID；
- held target、target error、target mask 或任何 target-derived changer/tail label；
- external outcome；
- structure、teacher、foundation embedding 或新增实验条件。

每个 outer fold 内，必须在 19 个 outer-train puzzles 上执行固定四折 puzzle-grouped
cross-fitting：每个 inner-held puzzle 的 B1/MA prediction 均由未见该 puzzle outcome 的
inner model产生。所有 inner-OOF predictions 完整后，按
position → mutant → method cell → puzzle 的精确层级权重计算：

1. `d=|mu_B1-mu_MA|` 的 weighted 95th percentile；分位点固定为 0.95，不搜索；
2. low/high 两组分别用 weighted-L1 breakpoint 的加权中位数求 alpha；
3. alpha clip 到 [0,1]；bin 数固定为 2；
4. 随后在完整 outer-train 19 puzzles 上训练最终 B1 与 MA，gate 参数保持冻结；
5. held puzzle 只能使用冻结专家和冻结 gate 预测。

不允许用 outer-train in-sample expert predictions 代替 inner-OOF gate fit；不允许选择
quantile、feature、bin 数、alpha grid、epoch 或 seed 子集。

### 4.3 零均值 residual calibration

最终 blended mean 完成后冻结两个 expert 与 gate。以 v2 hidden-64 conditional
two-Gaussian scale mixture 为 residual family，在 outer-train 上直接优化 exact mixture
CRPS 40 epochs：

\[
p(\Delta|x)=\pi(x)N(\mu_{blend},\sigma_n^2(x))+
[1-\pi(x)]N(\mu_{blend},\sigma_w^2(x)).
\]

两个 component location 必须等于同一个 `mu_blend`，`sigma_w>=sigma_n>0`；calibration
梯度不得进入 expert 或 gate，且不得改变 point mean。Candidate point prediction 始终是
`mu_blend`。

## 5. 阶段状态机

```text
R3M0_CONTRACT_AND_DIAGNOSTIC_FREEZE
  → R3M1_IMPLEMENTATION_AND_INVARIANTS_PASS
  → R3M2_REAL_DATA_ENGINEERING_SMOKE_PASS
  → R3M3_SEED0_TWENTY_FOLD_SCREEN_PASS / R3_CANDIDATE_FAIL
  → R3M4_ORIGINAL_R2M4_FIVE_SEED_PASS / R3_CANDIDATE_FAIL
  → R3M5_ARTIFACT_FREEZE_AND_MAIN_CONTRACT_HANDOFF
```

R3M0 的完成只冻结 amendment 与诊断证据。R3M1 完成前禁止 GPU 训练。R3M2 只允许
P01/P02、seed0、每网络/阶段最多 3 epochs 的工程 smoke；smoke score 不得用于修改配置。

## 6. R3M3 seed-0 Gate

固定 20 outer LOPO folds、seed0、两个 expert 40 epochs、inner crossfit 4 folds、residual
40 epochs。候选相对同一 B1 comparator 必须同时满足：

- signed-delta MAE mean gain >0；相对改善 >=1%；至少 12/20 puzzles 正向；
- full-construct CRPS mean gain >0；至少 12/20 puzzles 正向；
- registered prediction coverage=100%；failure=0；unexpected keys=0；
- held target/error/mask 改变不得影响 experts、gate threshold、alpha、scale、weight 或
  prediction row；
- inner gate ledger 必须覆盖每个 outer-train puzzle 恰好一次，inner-held outcome 不得进入
  对应 expert training；
- residual calibration 前后 point mean 在 `atol=1e-7, rtol=0` 下相同。

只有完整 20 folds 合并后由预冻结 qualifier 同时判定 Mean 与 Calibration Gate PASS，
才可开放 R3M4。partial fold 结果不能触发恢复、调参或候选替换。

## 7. R3M4：复用原 R2M4 正式门槛

固定 seeds 0--4、20 folds。每 seed 独立训练两个 experts、inner gate 和 residual；不得删除
失败 seed、选择最佳 seed 或调参。B1 形成 5-seed 等权 Gaussian mixture；候选形成 5-seed
等总质量、每 seed 内按 residual weights 分配的 10-component mixture。point prediction 为
五 seed blended mean 的算术平均。

正式 PASS 必须全部满足：

- puzzle-paired CRPS 95% Student-t CI lower >0；
- CRPS gain >= `max(0.003, 2% × B1 CRPS)`；至少 14/20 puzzles 正向；
- puzzle-paired signed-delta MAE 95% Student-t CI lower >0；
- signed-delta MAE relative gain >=1%；至少 12/20 puzzles 正向；
- leave-one-puzzle-out 后两指标 effect 都保持正；
- 任一 puzzle 的原冻结 combined influence fraction <=25%；
- coverage=100%、failure=0、unexpected keys=0；
- 68% 和 95% coverage absolute error 相对 B1 各自不恶化超过 2 个百分点；
- headline 只能来自唯一五-seed mixture。

qualifier 的正式通过状态必须精确为 `R2M4_POST_HOC_DEVELOPMENT_PASS`，以满足原成功
predicate；阶段记录同时写 `R3M4_ORIGINAL_R2M4_GATE_PASS`。

## 8. Prediction artifact 与必备 ledger

prediction-only artifact 至少包含 v2 schema 字段并新增：

- `b1_delta_mean`、`meanaligned_delta_mean`、`expert_disagreement`；
- `gate_threshold`、`gate_alpha_low`、`gate_alpha_high`、`gate_alpha_applied`；
- `inner_crossfit_ledger_path`、两个 expert checkpoint path；
- `delta_mean/point_mean/locations/scales/weights/status`。

artifact 禁止包含 target、target error、qualified target mask 或 score。scorer 独立 join target。
inner ledger 必须记录 outer fold、inner fold、train/held puzzle IDs、seed、checkpoint 与 key
coverage，但不复制 outcome。不得增加无消费者的 hash、checksum、compat layer 或 feature flag。

## 9. 计算与执行边界

- 不设 GPU 日或日历上限，但仅允许唯一两段 gate 候选和固定 seeds/folds/epochs；
- 使用物理 GPU0--5，不抢占、不终止、不干预无关进程；无安全卡时等待；
- 允许 fold/inner-fold 分片，必须拒绝 duplicate/missing shard；
- 不允许 method gate、更多 bins、连续 gate、threshold search、rank、structure、teacher、
  foundation ensemble、更大 backbone 或额外 loss search；
- 不访问新 external outcome；不使用已有 external outcome 选择 gate；
- v3 失败不会被人工升级为 PASS，也不得改写 v1/v2；若失败，冻结为独立 negative result，
  后续是否另立不同架构 amendment 由用户目标与新证据决定。

## 10. 证据资格与主合同衔接

R3M4 PASS 只产生 `POST_HOC_DEVELOPMENT_PASS`，主合同回到 M6，route 可写为
`BENCHMARK_WITH_DISAGREEMENT_GATED_DEVELOPMENT_MODEL`。允许主张：在 consumed OpenKnot
LOPO 上，fold-legal disagreement-gated ensemble 同时改善点预测与概率预测。禁止主张：
external replication、task-matched SOTA、机制、广泛泛化或 publication readiness。

R3M3/R3M4 FAIL 时关闭本候选训练并返回 M6；所有 artifact 与负结果保留。无论 PASS/FAIL，
v1/v2 终局、external lock 和 benchmark 主线均保持可追溯。

## 11. R3M1 实现记录（2026-08-22）

focused commit `91e904a2888ca490fa06b6911042e1199cf7b9ca` 实现生产 gate、inner-crossfit
runner、zero-mean residual 和 R3M2/R3M3/R3M4 qualifier。v2/v3 聚焦回归共 26 tests
通过，覆盖 exact hierarchy weights、inner-held 排除与完整覆盖、blend 算术重放、禁止
method/target 字段、calibration gradient 隔离及原 Gate parity。该结果只授权真实 P01/P02
三 epoch engineering smoke；R3M3 与 R3M4 继续关闭。
