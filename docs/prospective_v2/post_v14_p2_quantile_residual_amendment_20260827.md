# Post-V14 P2 单调分位数残差合同（冻结、未激活）

日期：2026-08-27  
状态：`DRAFT_FROZEN_INACTIVE`  
机器合同：`configs/reactflow_delta/post_v14_p2_quantile_residual_amendment.yaml`

## 1. 当前权限边界

这是一份新的、聚焦的、在任何 P2 评分之前作出的工程与科学设计判断；它不是旧合同事实，也不是实验结果。当前唯一作用是把 branch 6 diagnostic PASS 后可能采用的 P2 vertical slice 冻结到可审计状态。

本合同目前不签发任何权限。激活、源投影、训练、预测、smoke、screen、held-score 读取、partial-fold score 读取、评分、qualification、formal confirmation 和新外部 outcome 访问全部为 `false`。`configs/reactflow_delta/active_contract.yaml` 不由本合同修改，P2 没有 runnable phase，也没有训练 token 或通用训练 token。

所有实际 V14 终态路径、router/diagnostic 路径、源与输出路径，以及从终态合同逐字复制的 V14 Gates，均保持 `PENDING_TERMINAL_BINDING`。在这些绑定仍 pending 时，任何 active/runnable 解释都必须由静态 validator 拒绝。

## 2. 唯一入口

未来激活必须同时满足以下精确父状态：

- first-matching router 选择 branch `6`；
- classification 为 `DISTRIBUTION_ONLY_FAILURE`；
- diagnostic schema 为 `reactflow_delta.post_v14_branch6_tail_diagnostic.v1`；
- diagnostic status 为 `POST_V14_BRANCH6_TAIL_DIAGNOSTIC_PASS`；
- primary statistic 为 `LOWER_MINUS_UPPER_TAIL_MISS90`；
- puzzle-level 95% interval 完全位于零的一侧，20 个 puzzle 中至少 14 个方向一致；
- next action 为 `OPEN_FOCUSED_P2_AMENDMENT_AUTHORITY`。

这个 diagnostic PASS 只说明路线可进入 P2 合同绑定，不是 P2 科学结果，也不是训练权限。缺失、矛盾、过期或不同 branch 的父状态必须 fail closed。

## 3. 数据、estimand 与输入

P2 保留 OpenKnot M2 v4.5.2、split-v4 二十折 leave-one-puzzle-out 和 `EXACT_PUZZLE_METHOD_MUTATION` 身份。科学聚合严格按 position → equal-mutant mean → equal-method-cell mean → equal-puzzle mean；20 个 held puzzles 是独立推断单位。

候选与 matched V10 replay 使用完全相同、按如下顺序构造的 244 维 target-free 输入：

```text
feature41 basis                              41
frozen V14 point                             1
absolute frozen V14 point                    1
frozen trained V8 direct features          201
                                           ---
                                           244
```

V8 direct features 固定为 source hidden96、receiver hidden96、signed distance1 和 mutation one-hot8。复用 V10 的 `calibration_input` 顺序和 `TrainOnlyStandardizer`；每个 outer fold 仅用 outer-train 行拟合 244 个均值与 population standard deviations，scale 小于 `1e-6` 时替换为 `1.0`。method/puzzle/dataset ID、held target/error/mask、external outcome 和 score-derived field 均禁止进入输入。

V14 point 绑定同 outer-fold、seed-0 candidate。outer-train checkpoint 必须 eval、不可训练、`no_grad`，并在分布训练前后保持完整 state bitwise 不变且无 point gradient；held point 由绑定的 V14 prediction 按 biological key 读取，GPU 重算不是 authority。raw point 单独传入两个 distribution head，候选 `tau=0.50` 直接赋 detached float64 point。point replay 独立使用 `atol=1e-7, rtol=0`，并保留 candidate median 的 exact array equality；初始化 grid 的 `1e-6` 容差不得扩展到这里。

## 4. 候选预测分布与科学评分

从 P2M3 screen 开始，以下 13 个节点和固定质量本身定义候选预测分布：

```text
taus = [.025, .05, .10, .20, .30, .40, .50,
        .60,  .70, .80, .90, .95, .975]

weights = [.0375, .0375, .075, .1, .1, .1, .1,
           .1,    .1,    .1,   .075, .0375, .0375]
```

总质量为 `1.0`，median 下方/节点本身/上方质量分别为 `0.45/0.10/0.45`。候选训练 surrogate 固定为 `2 * weighted pinball`，只用于训练。候选 screen 科学 CRPS 必须是 13-atom finite distribution 的 exact CRPS：

```text
sum_i w_i |y-q_i| - 0.5 sum_i sum_j w_i w_j |q_i-q_j|
```

候选 distribution-derived absolute value 是 `sum_i w_i |q_i|`。matched V10 replay 继续把其 two-Gaussian mixture 当作声明的完整预测分布，并使用 exact Gaussian-mixture CRPS。两臂评估的是同一个 proper-scoring-rule estimand（CRPS），但各自对自身声明分布精确求值；weighted pinball 绝不是 scientific CRPS，不能写入 score/Gate 字段。

## 5. 模型与初始化

候选固定为：

```text
Linear(244, 248) -> ReLU -> Linear(248, 12)
(244 + 1) * 248 + (248 + 1) * 12 = 63,748
```

12 个输出定义 `gap_j = 1e-4 + softplus(raw_j)`。以 `q_6` 为 frozen V14 point，向两侧累计 gaps，因此单调性与 median 是结构性质。learned layers 为 float32，gap、quantile、quadrature 与持久化分布为 float64。

matched comparator 是现有 V10 `MedianAsymmetricResidual`：

```text
Linear(244, 256) -> ReLU -> Linear(256, 4)
(244 + 1) * 256 + (256 + 1) * 4 = 63,748
```

P2-specific 初始化先构造现有 comparator，再把整个 `output_layer.weight` 清零；四个 bias 精确设为 mixture-weight logit `0`、`inverse_softplus(0.08)`、`inverse_softplus(0.20)`、allocation `0`。其初始 mixture 因此对输入无关。使用 fixed bounded float64 bisection 在 13 个 taus 上求该 mixture 的 inverse CDF；每个相邻 target gap 必须严格大于 `1e-4`。

候选整个 output weight 同样为零，bias `j` 为 `inverse_softplus(target_gap_j - 1e-4)`。唯一可执行初始化 grid replay 判据是：

```text
INITIAL_GRID_REPLAY_ATOL_1E_6_RTOL_0
np.allclose(candidate_initial_grid_float32,
            registered_comparator_initial_grid_float64,
            atol=1.0e-6, rtol=0.0)
```

该容差仅覆盖 fixed float64 bisection 到 float32 bias/forward 的 round trip。它不适用于 median point replay 或任何科学 score。只允许称两个初始 quantile grids 在注册容差内匹配；13-atom candidate 与 continuous two-Gaussian comparator 的完整初始分布并不相同。

## 6. 训练日程与阶段

两臂接收相同 input、point、rows、standardizer、seed、epochs 和 puzzle order。优化器为 Adam，learning rate `1e-3`，weight decay `0`，gradient clip `5.0`；禁止 early stopping 和 best-epoch selection；puzzle order 使用 `seed * 100003 + epoch`。

- `P2M0`：仅 inactive design/amendment/ledger/validator/实现和 synthetic focused tests；不读写 artifact，不改 active pointer，不训练、不评分。
- `P2M1`：未来的 source projection only；当前未授权。
- `P2M2`：未来 engineering smoke，folds `[0,1]`、seed `[0]`、3 epochs；scientific scorer 禁止。
- `P2M3`：未来二十折 screen，folds `[0..19]`、seed `[0]`、40 epochs；完整 prediction-only merge 后才允许一次 score。
- `P2M4`：仅 exact P2M3 PASS 后的 fixed formal，folds `[0..19]`、seeds `[0..4]`、40 epochs；seed0 重训，共 100 runs，不复用 screen prediction，不删 seed、不选 best seed。
- `P2M5`：terminal。完整有效 score 的 Gate failure 是 scientific FAIL；完整性或 provenance 缺陷是 INDETERMINATE。

未来真实训练必须使用现有 CUDA fail-fast；无 CUDA 或 CPU fallback 立即停止并保留证据。不设置显存 Gate。当前 inactive Batch 不运行 GPU validation 或训练。

## 7. Screen 与 formal Gates

P2M3 完整性要求 coverage `1.0`、failure rate `0.0`、unexpected keys `0`、完整 20 folds、point state 不变、无 point gradients、finite strict quantiles、median exact、point replay `1e-7/0`、无 held-target prediction input、无 partial score、无 external outcome。

激活时必须从 canonical V14 terminal contract 逐字复制所有 feature41/terminal CRPS、distribution-absolute、coverage/calibration Gates，并原样重复 signed/point-absolute replay Gates；当前数值仍是 `PENDING_TERMINAL_BINDING`，不得凭记忆填写或放宽。

新增 matched V10 replay Gates 固定为：

- candidate-vs-V10 exact CRPS relative gain 至少 `0.015`；
- distribution-absolute MAE relative gain 至少 `0.01`；
- 两项 paired 95% t-interval lower bound 都大于 `0`，`t(0.975,19)=2.093024054408263`；
- 每项至少 `14/20` puzzles 为正；
- 所有 headline leave-one-puzzle-out effects 为正；
- maximum single-puzzle effect fraction 不超过 `0.20`。

P2M4 candidate formal distribution 是五个 seed 的 equal mixture：每个 seed 的 13 个 atoms 使用 `weight_i/5`，合计 65 atoms。formal 使用同一个 exact finite-distribution CRPS estimand，不平均 quantile curves，也不平均 per-seed CRPS。V10 formal 是五个 Gaussian mixtures 的 equal mixture并用 exact mixture CRPS。全部 screen Gates 重复，且 CRPS 与 distribution-absolute 两项都至少 `4/5` seeds 为正。

## 8. Artifact 与结论边界

本合同只冻结 source manifest、fold、prediction、merged、score、qualification 与 formal-assembly schema 名称，不实现 runtime。prediction-only 文件不得含 held target/error/mask、score、per-puzzle effect、Gate 或 external outcome。未来 merge 必须在任何 score 前确认完整 folds/seeds、canonical provenance、两臂 63,748 参数、相同 taus/weights、point replay、strict monotonicity、finite arrays 和 prediction-only schema。

即使 formal exact PASS，最大声明仍为 `POST_HOC_DEVELOPMENT_FORMAL_PASS`。它不构成 external replication、SOTA、mechanism、practical utility、publication readiness，也不能恢复任何历史 qualification。smoke、proxy 或 training loss 永远不是最终科学结论。

当前 Batch 1 的 validator PASS 只表示这份 inactive 静态声明内部一致；它不签发 authority，不是科学 PASS，也不允许进入 P2M1。
