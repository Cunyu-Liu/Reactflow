> [!IMPORTANT]
> **已由 ReactFlow-Δ V2 合同取代（2026-07-29；仅保留为历史工程资产与负结果证据）。**
> 本文件不再是可执行 Goal，也不得用于启动训练、追加 seed、降低 Gate 或支撑新的科学主张。当前唯一有效合同为
> `docs/contracts/ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v2_20260729.md`
>（SHA-256：`5d2dc9e2ac0e6b8c6355791f4ff95958b2e9ab5722d2d2eba49c6578a3e87c13`）。

# ReactFlow SOTA 追赶目标与 Todo List

本文档定义 ReactFlow 下一阶段的明确目标：**在同一公开数据、同一 split、同一评估脚本、同一统计协议下，尽快追赶并尽可能超过 RNA 二级结构 SOTA 模型**。任何没有同协议重算的外部数字，只能放在 `cited` 栏，不能作为论文主结论。

## 1. 总目标

ReactFlow 要从当前 weak engineering baseline 升级为论文级模型。核心目标是：

> 学习一个带物理约束和 probing forward operator 的 RNA 二级结构分布 `p_theta(S | x)`，在 MMseqs-disjoint / Rfam-disjoint novel-family 上达到或超过同协议 SOTA，并同时解释 chemical probing reactivity、保持结构合法性和可校准 ensemble uncertainty。

最终论文主张必须同时满足：

| Gate | 必须满足的条件 | 证明 artifact |
|---|---|---|
| Engineering gate | 三份基础 final result 与 RF-CF 系列结果可解析；所有 audit healthy | `goal_readiness_audit.json`, `final_queue_audit.json`, `reproducibility_manifest.json` |
| Cross-family claim gate | `novel_clan_mean_f1 >= 0.15` 且 `gap_mean_f1 <= 0.10`，并优于 RF-M0/RF-M1 | `cross_family_*_metric_audit.json` |
| Same-split baseline gate | 至少复现 2 个外部 baseline，并在同一 MMseqs split 上比较 | `baseline_rerun_results.json`, `sota_alignment_table.md` |
| Statistical gate | best config 至少 3-seed pilot；论文主结果 10 seeds；bootstrap CI 与 permutation test 显著 | `multiseed_summary.json`, `significance_report.md` |
| Paper gate | 所有 README/SOTA 表格只使用同协议 local rerun 或明确 cited/local 分栏 | README, `docs/sota_gap_report.md` |

## 2. 当前状态与差距

当前项目工程闭环已经打通，但性能仍远低于 SOTA。这里需要区分三个口径：readiness audit 当前 overall best 是 RF-A1 warm exact 路径的 `novel` F1；MMseqs 当前最好是 RF-M1 warm；最新 RF-CF3 family-balanced 是更接近最终 cross-family claim 的攻坚线但未改善。它们都没有达到 early gate。

| 项目 | 当前最好结果 | 参考目标 | 当前差距 |
|---|---:|---:|---:|
| Readiness audit overall `best_novel_mean_f1` | `0.0624` from RF-A1 warm exact | early gate `0.15` | `-0.0876` |
| MMseqs RF-M1-warm `novel_clan_mean_f1` | `0.0447` | early gate `0.15` | `-0.1053` |
| 最新 RF-CF3 family-balanced `novel_clan_mean_f1` | `0.0286` | early gate `0.15` | `-0.1214` |
| 最新 RF-CF3 family-balanced `novel_clan_mean_mcc` | `0.0271` | 必须同向提升 | 低于 RF-M1-warm |
| 最新 RF-CF3 family-balanced `gap_mean_f1` | `0.0003` | `<=0.10` | gap 本身可接受，但 F1 太低 |
| ArchiveII mean F1 | `0.0295` | cited RNADiffFold `~0.880` | `~ -0.8505` |
| viral mean F1 | `0.0156` | cited eFold `~0.730` | `~ -0.7144` |
| lncRNA mean F1 | `0.0113` | cited eFold `~0.440` | `~ -0.4287` |
| human_mRNA mean F1 | `0.0287` | 需要 same-split baseline | 未对齐 |

注意：ArchiveII / viral / lncRNA 的 SOTA 数字来自文献 cited protocol，不是同 split rerun。它们用于量化警戒线；论文主表必须以同协议 local rerun 为准。

## 3. 必须对齐的下游任务和指标

### 3.1 Structure prediction tiers

所有结构预测任务必须统一输出以下指标：

| Tier | 数据来源 | 必报指标 | 论文用途 |
|---|---|---|---|
| `in_clan` | MMseqs test | mean/micro F1, mean/micro MCC, pair count, runtime | 训练分布内 sanity check |
| `novel_clan` | MMseqs novel | mean/micro F1, mean/micro MCC, family_macro_f1, retention | 主 OOD 指标 |
| `archiveII` | eFold/RNAndria cache | mean/micro F1, MCC | 公共 benchmark 对齐 |
| `PDB` | eFold/RNAndria cache | mean/micro F1, MCC | 结构可靠 tier |
| `viral` | eFold/RNAndria cache | mean/micro F1, MCC | eFold cited 对齐 |
| `lncRNA` | eFold/RNAndria cache | mean/micro F1, MCC | 长 RNA 泛化 |
| `human_mRNA` | eFold/RNAndria cache | mean/micro F1, MCC | 长序列/真实转录本泛化 |

### 3.2 Long-range pair recovery

RF-CF2 必须单独报告 distance-bin 指标：

| Bin | 定义 | 必报指标 | 目标 |
|---|---|---|---|
| short | `1 <= |i-j| <= 11` | precision/recall/F1/MCC | 不因 long-range reweighting 明显下降 |
| medium | `12 <= |i-j| <= 23` | precision/recall/F1/MCC | 稳定提升 |
| long | `|i-j| >= 24` | precision/recall/F1/MCC | RF-CF2 主目标，novel_clan long recall 必须提升 |

### 3.3 Chemical probing consistency

ReactFlow 的论文特色不能退化成普通 contact predictor。所有 best configs 必须报告：

| 指标 | 定义 | 目标 |
|---|---|---|
| Pearson / Spearman | predicted vs observed reactivity shape | 与结构 F1 同向改善 |
| calibrated MAE | affine-calibrated reactivity error | 低于 base/warm |
| ECE / MCE | ensemble calibration error | 启用 `lambda_calib` 后下降 |
| population diversity | sampled structure ensemble diversity | 不能 collapse 到单一结构 |

### 3.4 Runtime and scale metrics

| 指标 | 目标 |
|---|---|
| samples/s | 每个配置必须记录，不能只报 F1 |
| GPU memory | 记录峰值与 batch ladder |
| wall-clock | 每个 run 必须有 profile summary |
| params | RF-CF5/RF-CF6 必须报告参数量 |
| failure mode | OOM/NaN/Inf 必须进入 retry log |

### 3.5 Same-split 对齐表契约

所有 README 和论文表格必须由同一个表格生成脚本产出，字段固定如下，避免手工混用 protocol：

| 字段 | 要求 |
|---|---|
| `model` | ReactFlow config 或 baseline 名称 |
| `protocol` | 只能是 `same_split_local`、`local_closest_protocol`、`cited_only` 三类之一 |
| `split` | MMseqs/Rfam/public tier 名称，必须可追溯到 manifest |
| `seed_count` | single seed、3 seeds、10 seeds 必须明确区分 |
| `mean_f1`, `mean_mcc` | 所有结构 tier 必填 |
| `long_f1`, `long_recall` | 有 contact 距离分箱时必填 |
| `reactivity_corr`, `calibration_ece` | probing/calibration 路线必填 |
| `runtime_s_per_sample` | 论文效率表必填 |
| `artifact` | 结果 JSON 或 markdown report 路径 |

## 4. SOTA 对齐目标

### 4.1 必须重跑的 baseline

| Baseline | 优先级 | 对齐方式 | 输出 |
|---|---|---|---|
| eFold/RNAndria | P0 | 用当前 eFold/RNAndria cache 和 MMseqs split rerun 或复现 closest protocol | `baseline_efold_results.json` |
| RNADiffFold | P0 | 复现 public model / script；若无法同 split，标注 cited-only | `baseline_rnadifffold_results.json` |
| RibonanzaNet2-derived | P0 | frozen encoder + simple decode / adapter baseline | `baseline_ribonanzanet2_decode_results.json` |
| TVAE-RNA | P1 | 若代码可用，统一 7-tier evaluation | `baseline_tvae_results.json` |
| MERGE-RNA | P1 | probing/ensemble task 对齐，不作为直接 2D F1 baseline | `baseline_merge_rna_report.md` |

### 4.2 目标阈值

| 阶段 | `novel_clan_mean_f1` | 公共 tier F1 | 统计要求 |
|---|---:|---:|---|
| 当前 | `0.0447` | `0.011-0.175` | single seed |
| Engineering claim | `>=0.15` | 至少所有 tier 优于 RF-M1 | single seed + audit |
| Strong internal | `>=0.30` | ArchiveII/PDB/viral/lncRNA 全部显著提升 | 3 seeds |
| Paper candidate | 同 split 超过 best baseline | 同 split 超过 best baseline | 10 seeds + CI |
| SOTA claim | 同 split 第一，或 cited protocol 可比第一 | 同 split 第一，且 public tiers 不弱 | 10 seeds + significance |

### 4.3 单项指标追赶目标

| 指标 | 近期最低目标 | 论文候选目标 | 达不到时的升级动作 |
|---|---:|---:|---|
| `novel_clan_mean_f1` | `>=0.15` | 超过 same-split best baseline | 从 loss sweep 升级到 pair-aware adapter / distillation |
| `novel_clan_mean_mcc` | 与 F1 同向提升 | 超过 same-split best baseline | 检查空结构偏置和 threshold/decoder |
| `long_range_recall` | 高于 RF-M1/RF-CF3 | 显著高于 baseline | 启用 RF-CF2、window stitching、global pair head |
| public tier F1 | 全部高于 RF-M1 | ArchiveII/PDB/viral/lncRNA 不弱于 best baseline | 检查 split、decode、长序列窗口拼接 |
| reactivity correlation | 高于 structure-only | 与 F1 同向提升 | 接入真实 probing subset，避免 pseudo probing |
| calibration ECE | 低于未校准模型 | 10-seed 显著降低 | 调整 heteroscedastic variance 和 ensemble diversity |
| runtime | 不因 scale-up 失控 | 报告可复现吞吐 | batch ladder、bucket/window、profile-driven 优化 |

## 5. 下一阶段执行里程碑

| 时间窗口 | 必须完成的目标 | 成功定义 | 未达标时立即动作 |
|---|---|---|---|
| 0-24 小时 | 完成 RF-CF1 contact sweep 首轮读取；确认 RF-CF2/RF-CF5 watcher 状态；补齐 SOTA 对齐文档入口 | `cross_family_contact_sweep_results.json` 或 watcher audit 可读；本文档和 README/状态报告均链接 | 若 RF-CF1 无提升，直接进入 RF-CF2 long-range；不再继续只调 `lambda_contact` |
| 1-3 天 | 完成 RF-CF2 long-range sweep 与 RF-CF5 capacity 第一轮；实现 same-split 表格生成脚本；启动至少 1 个 baseline rerun | `cross_family_long_range_results.json`、capacity result、`sota_alignment_table.md` 落盘 | 若 F1 仍接近 0，优先排查 decoder empty-bias 和 pair-score 表达力 |
| 3-7 天 | 上 pair-aware adapter 与真实 probing subset；完成 eFold/RNAndria + RibonanzaNet2-derived baseline | 至少一个配置超过 MMseqs RF-CF3 的 `0.0447`，并继续冲击 overall best `0.0624` 和 early gate `0.15`，且 public tiers 不退化 | 若只提升 in-clan，不提升 novel-clan，停止该路线并改做 family-invariant pair prior |
| 1-2 周 | best configs 进入 3-seed pilot；RNADiffFold 复现或明确 cited-only；生成第一版论文主表 | `multiseed_summary.json`、baseline rerun report、CI 草表可读 | 若 3-seed 方差过大，扩大 validation 和 family-macro selection |
| 投稿前 | 10-seed 主实验、配对显著性检验、最终 README/SOTA 表和 artifact manifest | `significance_report.md`、`reproducibility_manifest.json`、论文表全部同协议 | 未完成 same-split baseline 前禁止写 SOTA claim |

## 6. 当前自动实验队列

当前已接入自动接力，远端实际启动由 `scripts/run_cross_family_chain_after_remote_ready.sh`
统一 gate。最新核查显示 frozen gate 已通过，RF-CF chain 已启动；
`cross_family_balanced_results.json` 已生成并完成 audit，但 RF-CF3 未达到
claim gate；RF-CF1 contact sweep 已自动接力运行。`cross_family_contact_sweep_results.json`、
`cross_family_long_range_results.json` 和 `cross_family_capacity_results.json`
仍未生成，其中 RF-CF2/RF-CF5 watcher 正在等待上游结果。

1. `RF-CF3-family-balanced`
   - 状态：已完成；`novel_clan_mean_f1=0.0286`，`novel_clan_mean_mcc=0.0271`，`gap_mean_f1=0.0003`，`cross_family_claim_ready=false`。
   - 目的：验证 family-balanced sampler 是否改善 novel-family 泛化。
2. `RF-CF1-contact-strong`
   - 状态：已自动接力；当前 `lambda_contact=0.1` run 正在 `novel_clan` post-training eval。
   - 目的：增强 pair consistency。
3. `RF-CF2-long-range`
   - 状态：等待 RF-CF1 结果。
   - 目的：提升 `|i-j| >= 24` pair recall。
4. `RF-CF5-capacity`
   - 状态：等待 RF-CF2 结果。
   - 目的：将 `hidden_size=8` 的弱 baseline 扩到 `16/32`，并扩大 adapter。

### 6.1 最新执行状态记录（截至 2026-07-14 09:49 CST）

- 已新增 `scripts/build_sota_alignment_table.py`，并生成 `docs/sota_alignment_table.md/json`；README public benchmark 行已改为读取 local artifact，不再用 `pending` 手工占位。
- 已恢复 `artifacts/full_runs/full_ablation_20260709_003012/metadata/rfam_current_exact_metadata.*` 和 `splits/rfam_current_exact_seed0/*`。该 split 使用 `cluster_method="exact"` 和 RF family fallback，只能作为 `local_closest_protocol` / 工程恢复证据，不能替代 MMseqs same-split 主结论。
- 服务器已完成 full MMseqs metadata 与 `splits/rfam_current_mmseqs_seed0/*`，且远端 `splits/rfam_current_exact_seed0/{train,val,test,novel,split_manifest}` 已补齐并与本地恢复文件 sha256 对齐。2026-07-13 05:32 CST，完整 frozen features gate 已通过，RF-CF 自动链路不再受 frozen shard 实体缺失阻塞。
- 已新增 `scripts/deploy_remote_reactflow.sh`，用于同步代码、cache、frozen features、metadata/splits 和结果表到服务器 `/home/cunyuliu/reactflow`。服务器已安装 MMseqs2 到 `/home/cunyuliu/tools/mmseqs2/bin/mmseqs`，版本 hash `cb12a2d75a9808ee61721029d064a7bd80af6fec`。
- 已确认原 `ribonanzanet2_sharded_full` 只有部分 `features.npz` 实体，不能仅靠本地同步恢复到 manifest 声称的 `409` shards / `208,905` records。服务器已下载并校验 RibonanzaNet2 Kaggle alpha checkpoint（`weights_sha256=c94031719c8a...`），真实 torch exporter smoke、batch smoke、single-vs-batch 一致性检查和 length-256 batch16 压测均通过。
- 远端 frozen 恢复已完成：4 个共享队列 pool worker 产出 `features/index/provenance=409/409`，`record_count=208,905`，finalizer 已验证 `missing=0` 并重建 parent `sharded_manifest.json`。RF-CF chain launcher 于 2026-07-13 05:30 CST 观察到 `complete_shards=409` 和 `cross_family_inputs_ready=true`，并启动 RF-CF3/RF-CF1/RF-CF2/RF-CF5 watchers。
- RF-CF targeted inputs 除 frozen 外已在服务器确认 ready：`mmseqs_final_results.json`、MMseqs split、public eval caches 均存在且非空；RF-CF3/RF-CF1/RF-CF2/RF-CF5 四个 watcher 脚本 `bash -n` 均通过。
- 已新增并部署 `scripts/evaluate_external_baseline_predictions.py`，用于把 eFold/RNAndria、RNADiffFold 等外部模型导出的 prediction JSONL 与 ReactFlow gold split JSONL 做同协议评测，输出可被 `scripts/build_sota_alignment_table.py` 读取的 `baseline_*_results.json`。当前远端 `baseline_efold_results.json` 只是 pending artifact：MMseqs `in_clan` gold count `16,606`、`novel_clan` gold count `46,147` 均记录为 `missing_predictions`，`rows=0`，不会向 SOTA 表释放任何 same-split baseline 分数。
- 首次 RF-CF3 run 因 watcher 默认使用无 PyTorch 的 `python3` 失败（`torch backend requires optional dependency PyTorch`）。已修复四个 RF-CF watcher 的解释器选择：优先 `TORCH_PYTHON`、其次 `PYTHON_BIN`、再自动使用 `/home/cunyuliu/miniconda3/envs/editflow/bin/python`，最后才回退 `python3`；远端验证 `torch_ok 2.5.1+cu121 True`。
- RF-CF chain 已于 2026-07-13 10:07 CST 重新启动并继承 `CUDA_VISIBLE_DEVICES=0`。RF-CF3 已完成并写出 `training_checkpoint.json`、`stdout.json`、`cross_family_balanced_results.json` 和 `cross_family_balanced_metric_audit.json`；`stderr.log` 为空。
- RF-CF3 metric audit 显示 `cross_family_healthy=true` 但 `cross_family_claim_ready=false`：`in_clan_mean_f1=0.0289`，`novel_clan_mean_f1=0.0286`，`novel_clan_mean_mcc=0.0271`，`gap_mean_f1=0.0003`，`retention=0.9896`。该结果低于 RF-M1-warm 的 `novel_clan_mean_f1=0.0447`，也远低于 early gate `0.15`，因此不能进入 claim。
- RF-CF1 contact sweep 已自动接力：`contact_sweep_after_cross_family_balanced.log` 显示 2026-07-13 22:22 CST 因 RF-CF3 未达标而启动 `lambda_contact=0.1`。当前 RF-CF1 子进程 PID `2576763` 正在运行，`stderr.log` 为空；`active_eval_progress_audit.json/md` 显示 `eval_progress_healthy=true`，`novel_clan processed=19,931 / 46,147`，`progress_fraction≈0.4319`，平均 `0.2214s/sample`，ETA 约 `5,803s`（约 1.61 小时）。
- 当前 RF-CF runtime/resource 审计健康：`runtime_health_rf_cf_active.json/md` 为 `healthy=true`、`pass=13`、`warn=0`、`fail=0`；`system_resource_audit_rf_cf_active.json/md` 为 `resource_healthy=true`、`pass=8`、`fail=0`，RF-CF1/RF-CF2/RF-CF5 watcher pidfile 均 alive，RF-CF1 Python 子进程 CPU 约 `99.7%`、RSS 约 `84.0 GiB`。
- 已为后续 RF-CF1/RF-CF2/RF-CF5 的新进程部署 `evaluate-efold` pre-training load heartbeat：`--profile-path` 会在训练前写入 `load_frozen_*`、`load_train_*`、`load_eval_*` 事件。当前 RF-CF3 进程启动早于该 patch，因此不会 retroactively 写入这些加载阶段事件，避免为监控心跳而重启浪费已完成的加载进度。
- 已为后续 RF-CF1/RF-CF2/RF-CF5 的新进程部署 JSONL cache 流式读取优化：`load_efold_samples` 通过 `iter_sample_cache` 流式读取 cache row，不再先构造临时全量 tuple 后再复制到 accepted sample list。当前 RF-CF3 进程启动早于该 patch，因此不重启；后续接力或失败重跑会使用更低峰值内存路径。

## 7. Scale-up 计划

### 7.1 数据 scale-up

| 阶段 | 数据 | 目标 |
|---|---|---|
| S0 | 当前 MMseqs split | 保持主 benchmark 稳定 |
| S1 | eFold/RNAndria full train `307,641` | 多 epoch 收敛 |
| S2 | 长序列 cache + window stitching | 改善 lncRNA/human_mRNA |
| S3 | Ribonanza2 probing subset | 启用真实 `lambda_react` / `lambda_calib` |
| S4 | multi-source public RNA profiles | 强化 ensemble/probing claim |

### 7.2 模型 scale-up

| 阶段 | 配置 | 目的 |
|---|---|---|
| M0 | `hidden_size=8`, `adapter_dim=8` | 当前 weak baseline |
| M1 | `hidden_size=16`, `adapter_dim=16` | RF-CF5 第一档 |
| M2 | `hidden_size=32`, `adapter_dim=16/32` | 主力追赶配置 |
| M3 | pair-aware adapter | 将 frozen encoder 转成 pair prior |
| M4 | multi-encoder adapter | RiNALMo/RNA-FM/HydraRNA features |
| M5 | contact distillation + DFM | 融合 RNADiffFold/eFold soft contacts |

### 7.3 训练 scale-up

| 阶段 | 设置 | 验收 |
|---|---|---|
| T0 | 1 epoch, 1 seed | 工程闭环 |
| T1 | 3 epochs, validation selection | F1 不再极低 |
| T2 | 3 seeds | 方向稳定 |
| T3 | 10 seeds | 论文统计 |
| T4 | best config + baseline rerun | SOTA 对齐 |

## 8. Todo List

### P0：立即执行

- [ ] 等 RF-CF1 contact sweep 完成，读取 `cross_family_contact_sweep_results.json`。
- [ ] 若 RF-CF1 不达标，确认 RF-CF2 自动启动并读取 long-range bin 指标。
- [ ] 若 RF-CF2 不达标，确认 RF-CF5 capacity 自动启动。
- [x] 新建 `scripts/build_sota_alignment_table.py`，统一生成 cited/local/same-split 表。
- [x] 在 README 中将所有 public benchmark 行从 `pending` 改为自动读取 local rerun artifact。
- [ ] 启动 eFold/RNAndria same-split baseline rerun。
- [ ] 启动 RNADiffFold 可复现性调研与同协议 wrapper。

### P1：模型性能主攻

- [ ] RF-CF5 扩大到 `hidden_size in {16,32,64}`。
- [ ] Adapter 扩大到 `adapter_dim in {16,32,64}`。
- [ ] 实现 pair-aware frozen adapter：`h_i, h_j, h_i*h_j, |h_i-h_j| -> pair prior`。
- [ ] 实现 long-window stitching consistency loss。
- [ ] 启用 `lambda_calib` sweep：`{0.05,0.1,0.2}`。
- [ ] 在真实 probing subset 上启用 `lambda_react`，禁止 structure-only pseudo probing。
- [ ] 做 thermo-guided decode `eta` sweep，并报告 energy/F1 tradeoff。

### P2：指标和论文闭环

- [ ] 每个 RF-CF run 输出 7-tier result + distance-bin result + family-macro result。
- [ ] 所有 best configs 进入 3-seed pilot。
- [ ] 进入论文前扩展到 10 seeds。
- [ ] 对 best config vs baseline 做 bootstrap CI。
- [ ] 对 paired predictions 做 permutation test。
- [ ] 生成 final SOTA table：`docs/sota_alignment_table.md`。
- [ ] 生成 final ablation table：`docs/ablation_experiment_filled.md`。

### P3：风险控制

- [ ] 如果 RF-CF5 仍低于 `0.15`，不要继续只调小 loss；必须升级 pair-aware architecture。
- [ ] 如果 public tiers 仍接近 0，检查解码是否偏向 empty/low-pair structures。
- [ ] 如果 long-range recall 接近 0，优先做 global pair head / window stitching。
- [ ] 如果 `in_clan` 提升但 `novel_clan` 不动，说明记忆 family，不进入主模型路线。
- [ ] 如果 same-split baseline 无法复现，README 必须保持 cited/local 分栏，不允许写 SOTA claim。

## 9. Definition of Done

只有同时满足以下条件，才能说“接近论文发表”：

1. ReactFlow best config 在同一 MMseqs split 上超过至少一个强 baseline。
2. `novel_clan_mean_f1`、`novel_clan_mean_mcc`、long-range recall 同向提升。
3. public tiers 不出现明显退化。
4. probing consistency / calibration 指标证明 ReactFlow 的 ensemble 特色有效。
5. 10-seed 统计显著，CI 不跨 0。
6. README、data governance、manifest、audit、checkpoint、baseline table 全部可复现。
7. 没有任何 SOTA 数字来自混用 split 或混用 cited/local protocol。
