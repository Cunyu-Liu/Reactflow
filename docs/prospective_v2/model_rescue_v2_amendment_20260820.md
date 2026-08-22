# ReactFlow-Delta Model Rescue v2 阶段性合同

日期：2026-08-20
状态：`TERMINAL_R2M3_MEAN_GATE_FAIL_CALIBRATION_BASELINE_ONLY`
父合同终局：`M2_NO_RESCUE_CANDIDATE`，commit `00c0cf3a804effb89ff99a8e9ea009963dc650d0`

## 1. 合同地位

本文件是 Model Rescue v1 终止后的独立 amendment。v1 的结果、资格和历史 artifact
保持不变；本合同只为一个新的、严格受限的均值—校准解耦假设临时重新开放内部
development training。任何 v2 结果均不得改写 v1，也不得自动产生 external、SOTA、
mechanism 或 publication PASS。

20 个 OpenKnot puzzles 已永久标记为 `DEVELOPMENT_CONSUMED`。v2 的 LOPO 结果只能
估计开发集泛化，不能写为 untouched prospective confirmation。新 external outcome
在整个 v2 中保持锁定。

## 2. 科学问题与可证伪假设

问题：在不扩大 B1 backbone、不加入结构、低秩、teacher 或 foundation model 的条件下，
能否通过先对齐 signed-delta point objective、再拟合严格零均值残差，同时改善
method-balanced full-construct CRPS 与 signed-delta MAE？

假设：v1 SparseDelta 的 CRPS 增益来自 gate/scale，但联合 NLL 梯度使 point mean 向
no-change 收缩。若先以 method-balanced signed-delta L1 训练均值并冻结，再以 exact
CRPS 拟合零均值 residual distribution，则 calibration 不能改变 point mean。

反证：MeanAligned 相对 B1 的 signed-delta MAE 改善不足 1%，或零均值 calibration
无法在逐 key point mean 不变时改善 CRPS。

## 3. 唯一模型集合

### 3.1 B1-MeanAligned

保持 B1 的 `d=96`、4 heads、hidden 64、两层 encoder、K_rank=0 和实际输入。
每 outer fold/seed 从头训练，不加载旧 NLL checkpoint。输出显式 `delta_mean`，损失为：

```text
puzzle → method → mutant → qualified position 的等权 signed-delta L1
```

正式训练固定 Adam、lr 1e-3、weight decay 0、40 epochs、clip 5；不 early-stop。

### 3.2 MeanAligned global residual

冻结完整 mean model，以 exact Gaussian CRPS 拟合单一 train-only global positive scale：

\[
p_A(\Delta|x)=N(\mu_\Delta(x),\sigma_g^2).
\]

它是 mean ablation 的最小概率输出，不决定 calibration Gate。

### 3.3 MeanAligned-CalibratedResidual

与 3.2 共享同一个 mean checkpoint。calibration 只读取 detach 后的 B1 hidden、mutation
identity 和 signed distance，用固定 hidden-64 的一层 MLP 输出 mixture weight、narrow
scale 和 positive wide-scale gap：

\[
p_B(\Delta|x)=\pi N(\mu_\Delta,\sigma_n^2)+(1-\pi)N(\mu_\Delta,\sigma_w^2).
\]

两个 component location 必须完全相同，因此 predictive mean 恒等于
`mu_delta`。Stage 2 只以闭式 Gaussian-mixture CRPS 训练 calibration head 40 epochs；
mean 参数不得产生 gradient 或变化。

## 4. 明确排除

- SparseDelta change gate 与 lambda=0.1；
- LRSO、任何 rank 或 rank search；
- StructDelta、RNA structure module；
- 更大 encoder、更多 attention layer；
- teacher、foundation ensemble、预训练特征；
- Student-t、NLL、额外 Huber、sign loss；
- hidden size、component count、loss weight、epoch 或 seed 搜索；
- 旧五-seed nested M3；
- existing/new external outcome 驱动的任何选择。

GPU 日与日历时间不设上限，但候选空间固定。只使用 GPU0–5，且不得干预无关任务。

## 5. 阶段与 Gate

### R2M0：合同与 authority

冻结 human/machine contract、ledger、implementation plan 和 active pointer。focused commit
前 training 为 false。父合同文件不得修改。

### R2M1：实现与不变量

实现 mean-only forward、method-balanced L1、global/conditional zero-mean calibration、
Torch exact mixture CRPS、prediction-only schema、runner 和 qualifier。mean freeze、
target-invariance、full-output 或 CRPS parity 任一测试失败时禁止训练。

### R2M2：真实数据工程 smoke

只在真实 OpenKnot P01/P02 上运行每阶段 3 epochs；验证 finite、coverage、failure、
target-invariance 与 Candidate A/B point-mean identity。score 不用于选择或调参。

### R2M3：seed-0 20-fold screen

固定 40+40 epochs。Mean Gate 要求 signed-delta MAE gain >0、相对改善至少 1%、
至少 12/20 puzzles 正向。Calibration Gate 要求逐 key point mean 与 Candidate A 在
1e-7 内一致、CRPS gain >0、至少 12/20 puzzles 正向。coverage 必须 100%，failure 与
unexpected keys 必须为 0。两个 Gate 都通过才进入 R2M4。

#### 2026-08-22 可恢复中断登记

原始 R2M3 进程在完成 fold 0 和 fold 1 后停止，停止原因当前为
`UNKNOWN_NOT_ASSERTED`。这不是科学 FAIL 或 PASS；R2M3 Gate 保持 `IN_PROGRESS`，执行
子状态登记为 `R2M3_INTERRUPTED_RECOVERABLE`。

fold 0/1 的 result、两类 prediction、mean/global/mixture checkpoints 已完成结构审计：
checkpoint 可加载且 tensor 有限，prediction schema、key identity、共享 mean、严格同中心
mixture、正 scale、权重归一、target 字段隔离和 target-invariance 全部通过。该审计没有读取
CRPS、signed-delta MAE 或任何部分 Gate 结果。

恢复只允许在持久化会话中运行缺失 folds 2--19，并保存完整日志。fold 0/1 必须保留且不得
覆盖；20 folds 完成前禁止检查部分分数、修改模型、阈值、epoch 或 Gate。低频监控只检查
进程存活、资源冲突、日志是否继续和新增 fold artifact 是否结构完整，默认间隔 4 小时，
不得输出或解释部分指标。只有 20 folds 完整合并并由预冻结 qualifier 裁决后，才能决定
R2M3 PASS/FAIL；Mean Gate 与 Calibration Gate 必须同时通过才可开放 R2M4。

恢复已于 2026-08-22 00:29:57 +08:00 从 commit `935e32e` 启动，固定使用物理 GPU2，
tmux session 为 `reactflow_delta_r2m3_recovery_20260822`。恢复进程只接收 folds 2--19，
日志写入合同登记路径。合并工具固定为 `scripts/reactflow_delta/merge_model_rescue_v2.py`
（commit `42f3d3d`）；它在 fold universe 不完整、重复、seed 非 0、候选不符或被引用
artifact 缺失时拒绝合并，并且不会自行授权 R2M4。

### R2M4：固定五 seed 正式确认

只比较 B1 与 frozen CalibratedResidual，seeds 0–4，20 folds，无 nested selection。
CRPS 与 signed-delta puzzle-paired CI lower 均须 >0；CRPS 至少改善
`max(0.003, 2%)`，signed-delta 至少改善 1%；positive puzzles 至少 14/20 与 12/20；
LOO 均正、单 puzzle influence 不超过 25%；coverage 100%、failure 0；68%/95%
coverage error 不恶化超过 2 个百分点。

CI 固定为 20 个 puzzle-level paired effects 上的双侧 Student-t 95% CI。单 puzzle
influence 沿用 v1 正式定义：先计算每个 puzzle 的
`0.5 × CRPS_gain / mean_B1_CRPS + 0.5 × delta_gain / mean_B1_delta_MAE`，再以最大
绝对值除以 20 个绝对值之和。68% 与 95% coverage absolute-error guardrail 分别判断，
不得先平均两个 coverage level 后掩盖其中一个失败。

### R2M5：主合同衔接

PASS 时模型仅为 `POST_HOC_DEVELOPMENT_PASS`，主合同回到 M6，route 为
`BENCHMARK_WITH_MEAN_FIRST_DEVELOPMENT_BASELINE`。FAIL 时 route 为
`BENCHMARK_ROUTE_LOCKED`；CRPS-only 时为 `CALIBRATION_BASELINE_ONLY`。两种情况均
关闭训练并保持 external outcome 锁定。

## 6. Prediction 与证据边界

prediction artifact 至少包含 keys、delta_mean、point_mean、locations、scales、weights、
candidate、fold 和 seed；不得包含 target/error/qualified target mask。所有 registered
mutant×position 先生成 prediction/status，再由 evaluator join target。

任何 PASS 只能由预冻结 qualifier 从完整 artifacts 机械产生。工程 PASS、测试绿灯、
CRPS-only 改善、最佳 seed、部分 puzzles 或事后阈值均不得升级主张。

本合同完成不等于模型成功；它要求对唯一假设给出可复现 PASS 或明确 FAIL，并在失败时
停止，而不是扩大模型空间。

## 7. R2M3 终局与 M6 handoff（2026-08-22）

恢复后的 folds 0--19 已完整合并，预冻结 qualifier 给出
`MODEL_RESCUE_V2_FAIL`。MeanAligned 的 signed-delta MAE gain 为 `+0.00214289`，
20 个 puzzles 中 16 个为正，但相对改善为 `0.8833%`，低于预注册的 `1%` 初筛门槛，
因此 Mean Gate 为 `MEAN_GATE_FAIL`。零均值 residual calibration 的 CRPS gain 为
`+0.00547832`，20/20 puzzles 为正，Candidate A/B point mean 和 signed-delta MAE
逐 key 完全相同，因此 Calibration Gate 为 `CALIBRATION_GATE_PASS`。

合同要求两个 Gate 同时通过，故 R2M4 不授权、不运行。v2 的终局模型资格为
`CALIBRATION_BASELINE_ONLY`，不是 mutation-effect predictor improvement。训练关闭，
禁止第三次 model rescue，项目返回主合同 M6 的 `BENCHMARK_ROUTE_LOCKED`。该结果只能
作为 consumed-development negative/ablation evidence，不建立 external replication、
SOTA、mechanism 或 publication readiness。
