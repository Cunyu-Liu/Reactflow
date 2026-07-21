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

### 2.1 eFold/RNAndria 对 cross-family 泛化的直接启发

eFold/RNAndria 论文明确把 RNA 二级结构泛化问题归因到数据分布：常见训练集偏向少数短 ncRNA family，模型在 PDB/ArchiveII 等近域集合上表现好，但在 viral mRNA、long ncRNA、human mRNA 等更长、更复杂、family 更分散的 RNA 上明显掉点。论文的核心结论不是“模型越大越好”，而是 **仅扩大数据库规模不足以跨 family 泛化；必须增加结构类型的多样性与复杂度**。参考来源：Science Advances `10.1126/sciadv.adz4967` 与 Dryad `10.5061/dryad.79cnp5j95`。

这对 ReactFlow 的约束如下：

| eFold 经验 | 对 ReactFlow 的要求 | 不能替代的证据 |
|---|---|---|
| 训练数据不能只堆数量；要覆盖不同 RNA 类型、长度和结构复杂度 | 训练/采样必须按 source/family/length/structure-complexity 做 balanced 或 curriculum，而不是只扩大 `hidden_size`/epoch | `data_diversity_audit.json/md`, `source_family_length_manifest.json` |
| 新增 pri-miRNA、human mRNA、viral fragment、lncRNA 等复杂域可以缩小 generalization gap | ReactFlow 必须把 eFold/RNAndria public tiers 从“评估缓存”升级为多源训练/验证 curriculum 的显式阶段 | `efold_rnandria_diversity_curriculum_results.json` |
| 长病毒/lncRNA 需要保留结构域，不能随意切窗 | windowing/stitching 要记录 parent coordinates、domain agreement、跨窗 pair loss | `window_stitching_audit.json`, distance-bin long recall |
| cross-family 评估必须暴露 domain drop | 主表继续保留 MMseqs `novel_clan`，并附 PDB/ArchiveII/viral/lncRNA/human_mRNA 分层结果 | `sota_alignment_table.md/json` |

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

### 3.5 Data diversity and complexity metrics

借鉴 eFold/RNAndria，ReactFlow 后续不能只报告训练样本数，还必须报告训练数据的多样性与结构复杂度：

| 指标 | 定义 | 目标 |
|---|---|---|
| source mix | bpRNA/RNAstralign/Ribonanza/eFold pri-miRNA/human_mRNA/viral/lncRNA/Rfam synthetic 等来源占比 | 不让短 ncRNA 或单一来源支配 batch |
| clan/family coverage | Rfam clan/family/MMseqs cluster 覆盖与 long-tail 分布 | family-balanced sampler 后 long-tail family 有足够 exposure |
| length coverage | `<=64`, `65-128`, `129-256`, `257-512`, `513-1024`, `>1024/domain` 分桶 | long RNA 不被训练过滤掉，只能经过 domain/window 策略处理 |
| structure complexity | pair count、long-range pair ratio、stem/loop/domain count、pseudoknot/noncanonical 标记（如有） | 训练 curriculum 覆盖从局部 stem 到复杂长程结构 |
| domain segmentation quality | 对 viral/lncRNA/human_mRNA window/domain 的 parent-coordinate、closed-loop/domain consistency 记录 | 防止随机切窗破坏真实结构域 |

### 3.6 Same-split 对齐表契约

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
claim gate；RF-CF1 contact sweep 已完成并生成 `cross_family_contact_sweep_results.json`，
但仍未达到 claim gate。RF-CF2 官方 long-range watcher 已自动接力运行；
`cross_family_long_range_results.json` 和 `cross_family_capacity_results.json`
仍未生成，其中 RF-CF5 watcher 正在等待 RF-CF2 上游结果。

1. `RF-CF3-family-balanced`
   - 状态：已完成；`novel_clan_mean_f1=0.0286`，`novel_clan_mean_mcc=0.0271`，`gap_mean_f1=0.0003`，`cross_family_claim_ready=false`。
   - 目的：验证 family-balanced sampler 是否改善 novel-family 泛化。
2. `RF-CF1-contact-strong`
   - 状态：已完成；`lambda_contact=0.1/0.2/0.4/0.8` 均已落盘，best 为 `lambda=0.8`，`novel_clan_mean_f1=0.0435`，仍低于 RF-M1-warm `0.0447` 和 early gate `0.15`。
   - 目的：增强 pair consistency。
3. `RF-CF2-long-range`
   - 状态：官方 watcher 已接力启动 `w=2`；early `w=2/4/8` 已完成，best early 为 `w=8`，`novel_clan_mean_f1=0.0404`。
   - 目的：提升 `|i-j| >= 24` pair recall。
4. `RF-CF5-capacity`
   - 状态：等待 RF-CF2 结果。
   - 目的：将 `hidden_size=8` 的弱 baseline 扩到 `16/32`，并扩大 adapter。

### 6.1 最新执行状态记录（截至 2026-07-17 15:00 CST）

- 已新增 `scripts/build_sota_alignment_table.py`，并生成 `docs/sota_alignment_table.md/json`；README public benchmark 行已改为读取 local artifact，不再用 `pending` 手工占位。
- 已恢复 `artifacts/full_runs/full_ablation_20260709_003012/metadata/rfam_current_exact_metadata.*` 和 `splits/rfam_current_exact_seed0/*`。该 split 使用 `cluster_method="exact"` 和 RF family fallback，只能作为 `local_closest_protocol` / 工程恢复证据，不能替代 MMseqs same-split 主结论。
- 服务器已完成 full MMseqs metadata 与 `splits/rfam_current_mmseqs_seed0/*`，且远端 `splits/rfam_current_exact_seed0/{train,val,test,novel,split_manifest}` 已补齐并与本地恢复文件 sha256 对齐。2026-07-13 05:32 CST，完整 frozen features gate 已通过，RF-CF 自动链路不再受 frozen shard 实体缺失阻塞。
- 已新增 `scripts/deploy_remote_reactflow.sh`，用于同步代码、cache、frozen features、metadata/splits 和结果表到服务器 `/home/cunyuliu/reactflow`。服务器已安装 MMseqs2 到 `/home/cunyuliu/tools/mmseqs2/bin/mmseqs`，版本 hash `cb12a2d75a9808ee61721029d064a7bd80af6fec`。
- 已确认原 `ribonanzanet2_sharded_full` 只有部分 `features.npz` 实体，不能仅靠本地同步恢复到 manifest 声称的 `409` shards / `208,905` records。服务器已下载并校验 RibonanzaNet2 Kaggle alpha checkpoint（`weights_sha256=c94031719c8a...`），真实 torch exporter smoke、batch smoke、single-vs-batch 一致性检查和 length-256 batch16 压测均通过。
- 远端 frozen 恢复已完成：4 个共享队列 pool worker 产出 `features/index/provenance=409/409`，`record_count=208,905`，finalizer 已验证 `missing=0` 并重建 parent `sharded_manifest.json`。RF-CF chain launcher 于 2026-07-13 05:30 CST 观察到 `complete_shards=409` 和 `cross_family_inputs_ready=true`，并启动 RF-CF3/RF-CF1/RF-CF2/RF-CF5 watchers。
- RF-CF targeted inputs 除 frozen 外已在服务器确认 ready：`mmseqs_final_results.json`、MMseqs split、public eval caches 均存在且非空；RF-CF3/RF-CF1/RF-CF2/RF-CF5 四个 watcher 脚本 `bash -n` 均通过。
- 已新增并部署 `scripts/evaluate_external_baseline_predictions.py`，用于把 eFold/RNAndria、RNADiffFold 等外部模型导出的 prediction JSONL 与 ReactFlow gold split JSONL 做同协议评测，输出可被 `scripts/build_sota_alignment_table.py` 读取的 `baseline_*_results.json`。`scripts/run_efold_same_split_baseline.py` 已补齐 eFold same-split wrapper，可从 ReactFlow gold JSONL 导出 `<tier>.efold.predictions.jsonl` 并复用 scorer；本地 targeted tests 已更新为 `20 passed`，远端 `py_compile` 通过。远端隔离 venv `/home/cunyuliu/reactflow_external_envs/efold_py310` 已恢复 eFold 依赖：`efold-0.1.2-py3-none-any.whl` 经本地下载并验证 `sha256=f632e6fb3b1b5e7e9016372264961b3c26c6c7e8b9472d431b15b349729fd71b` 后上传安装，`pip check` 无 broken requirements，`efold` CLI 与 `from efold import inference` 均可用；真实 eFold smoke、archiveII limit-35 回归 smoke 和 GPU CUDA module smoke 已通过。完整 eFold/RNAndria same-split baseline 初次 full run 无 traceback 退出在 `PDB 333/333`、`archiveII 968/2052`；第二次 same-split run 暴露 eFold diagonal self-pair artifact，wrapper 已修复为丢弃 diagonal pair 并支持 `--resume-existing`；CPU resume 过慢后 wrapper 新增 `--device cuda`。当前 MMseqs same-split rerun 已拆成双 GPU：PID `986449` 在 GPU6 追加 `in_clan`，PID `1124346` 在 GPU7 追加 `novel_clan`；截至 2026-07-17 15:00 CST 已写出 `PDB 333/333`、`archiveII 2052/2052`、`human_mRNA 33/6627`、`in_clan 1632/16606` 和 `novel_clan 227/46147`。主 `baseline_efold_results.json` 在 full rerun 完成前仍不能向 SOTA 表释放 eFold 分数。
- 首次 RF-CF3 run 因 watcher 默认使用无 PyTorch 的 `python3` 失败（`torch backend requires optional dependency PyTorch`）。已修复四个 RF-CF watcher 的解释器选择：优先 `TORCH_PYTHON`、其次 `PYTHON_BIN`、再自动使用 `/home/cunyuliu/miniconda3/envs/editflow/bin/python`，最后才回退 `python3`；远端验证 `torch_ok 2.5.1+cu121 True`。
- RF-CF chain 已于 2026-07-13 10:07 CST 重新启动并继承 `CUDA_VISIBLE_DEVICES=0`。RF-CF3 已完成并写出 `training_checkpoint.json`、`stdout.json`、`cross_family_balanced_results.json` 和 `cross_family_balanced_metric_audit.json`；`stderr.log` 为空。
- RF-CF3 metric audit 显示 `cross_family_healthy=true` 但 `cross_family_claim_ready=false`：`in_clan_mean_f1=0.0289`，`novel_clan_mean_f1=0.0286`，`novel_clan_mean_mcc=0.0271`，`gap_mean_f1=0.0003`，`retention=0.9896`。该结果低于 RF-M1-warm 的 `novel_clan_mean_f1=0.0447`，也远低于 early gate `0.15`，因此不能进入 claim。
- RF-CF1 contact sweep 已完成并生成 `cross_family_contact_sweep_results.json`（35 rows）和 `cross_family_contact_sweep_metric_audit.json`。`lambda_contact=0.1/0.2/0.4/0.8` 均未达到 cross-family gate：`lambda=0.1/0.2/0.4/0.8` 的 `novel_clan_mean_f1` 分别为 `0.0188/0.0245/0.0401/0.0435`，best `lambda=0.8` 的 `novel_clan_mean_mcc=0.0430`、`gap_mean_f1=0.0008`，仍低于 RF-M1-warm 的 `0.0447` 和 early gate `0.15`。
- RF-CF2 官方 long-range watcher 已于 2026-07-16 09:59 CST 自动接力启动 `w=2`；`cross_family_long_range_results.json` 尚未生成。RF-CF2 early `w=2/4/8` 已完成，其中 `w=8 early` 为 `novel_clan_mean_f1=0.0404`、`novel_clan_mean_mcc=0.0397`，低于 contact `lambda=0.8` 与 RF-M1-warm。
- 当前 RF-CF best 已更新为 `RF-CF1-contact-lam0p8-early-gpu5_mmseqs_torch_full_data_e1_bs16`，`best_novel_mean_f1=0.0435`、`best_generalization_gap=0.0008`、`cross_family_claim_ready=false`。`docs/sota_alignment_table.md/json` 已重建为 `115` 行，并以 `same_split_local` 记录完整 RF-CF1 contact sweep 结果。
- 已新增并运行 `scripts/audit_data_diversity.py`，输出 `data_diversity_audit.json/md` 和 `source_family_length_manifest.json`。远端 artifact 覆盖 `efold_train`、PDB、ArchiveII、viral、lncRNA、human_mRNA、exact train/val/test/novel 以及 MMseqs train/test/novel 共 `915,715` records，overall long-range pair fraction 为 `0.5376`。审计同时暴露当前数据 gate 仍未通过：public/eFold train tiers 缺少 clan/family metadata，MMseqs train/test/novel 的 fallback pseudo-clan fraction 均为 `1.0000`，因此后续 source/family-balanced curriculum 必须先做 metadata join 和 pseudo-clan 清洗，不能把“已有缓存”直接等同于“多样性达标”。
- 为落实“空闲 GPU 不等待”的加速策略，独立 `RF-CF*-early-gpu*` 任务仍不覆盖官方 watcher 产物：RF-CF2 `w=2/4/8 early` 已完成，RF-CF5 capacity `h16:a16` 与 `h32:a16` 仍在 post-training eval，RF-CF5 `h64:a32 early` 于 GPU5 继续训练（batch 8，PID `3905214`，`progress_fraction≈0.1601`、`samples/s≈28.6530`）。
- 当前 RF-CF runtime/resource 审计健康：`runtime_health_audit.json/md` 为 `healthy=true`、`audited_run_count=13`、`pass=68`、`warn=1`、`fail=0`；`queue_progress_audit.json/md` 为 `progress_healthy=true`、`pass=77`、`warn=4`、`fail=0`；`system_resource_audit.json/md` 为 `resource_healthy=true`、`pass=8`、`warn=2`、`fail=0`，8 张 GPU 可见、`process_count=6`。warn 主要来自 early/post-training eval run 暂无 progress 或 manual pidfile 无子进程，不是 OOM 或 stderr failure。
- 已为后续 RF-CF1/RF-CF2/RF-CF5 的新进程部署 `evaluate-efold` pre-training load heartbeat：`--profile-path` 会在训练前写入 `load_frozen_*`、`load_train_*`、`load_eval_*` 事件。当前 RF-CF3 进程启动早于该 patch，因此不会 retroactively 写入这些加载阶段事件，避免为监控心跳而重启浪费已完成的加载进度。
- 已为后续 RF-CF1/RF-CF2/RF-CF5 的新进程部署 JSONL cache 流式读取优化：`load_efold_samples` 通过 `iter_sample_cache` 流式读取 cache row，不再先构造临时全量 tuple 后再复制到 accepted sample list。当前 RF-CF3 进程启动早于该 patch，因此不重启；后续接力或失败重跑会使用更低峰值内存路径。

## 7. Scale-up 计划

### 7.1 数据 scale-up

| 阶段 | 数据 | 目标 |
|---|---|---|
| S0 | 当前 MMseqs split | 保持主 benchmark 稳定 |
| S1 | eFold/RNAndria full train `306,557/307k` + source/family/length manifest | 多 epoch 收敛，并证明不是只增加样本数 |
| S2 | eFold-inspired diversity curriculum：pri-miRNA、human_mRNA、viral fragments、lncRNA、PDB/ArchiveII 分源分桶 | 改善跨 family 与 public tier 泛化 |
| S3 | 长序列 cache + domain/window stitching | 改善 lncRNA/human_mRNA，保留 parent coordinates 和结构域一致性 |
| S4 | Ribonanza2 probing subset + eFold DMS/SHAPE records | 启用真实 `lambda_react` / `lambda_calib` |
| S5 | multi-source public RNA profiles + synthetic clan-balanced RNACentral/Rfam domains | 强化 ensemble/probing claim 与 long-tail family coverage |

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

- [x] 等 RF-CF1 contact sweep 完成，读取 `cross_family_contact_sweep_results.json`（35 rows；best `lambda=0.8` 仍未过 gate）。
- [x] 若 RF-CF1 不达标，确认 RF-CF2 自动启动并读取 long-range bin 指标（官方 `w=2` 已于 2026-07-16 09:59 CST 启动）。
- [ ] 若 RF-CF2 不达标，确认 RF-CF5 capacity 自动启动。
- [x] 新建 `scripts/build_sota_alignment_table.py`，统一生成 cited/local/same-split 表。
- [x] 在 README 中将所有 public benchmark 行从 `pending` 改为自动读取 local rerun artifact。
- [x] 启动 eFold/RNAndria same-split baseline rerun（隔离 venv 依赖已恢复，当前 GPU6 PID `986449` 正在写 `in_clan`，GPU7 PID `1124346` 正在写 `novel_clan`；截至 2026-07-17 15:00 CST `in_clan=1632/16606`、`novel_clan=227/46147`，full rerun 完成前 `baseline_efold_results.json` 仍不释放 same-split baseline 分数）。
- [ ] 启动 RNADiffFold 可复现性调研与同协议 wrapper。
- [x] 新增 eFold/RNAndria-inspired 数据多样性审计：`scripts/audit_data_diversity.py` 已生成 `data_diversity_audit.json/md` 和 `source_family_length_manifest.json`；审计显示 current cache/split 仍有 family metadata 缺失和 fallback pseudo-clan 占比过高问题，后续 balanced sampling 前必须先补 metadata join。

### P1：模型性能主攻

- [ ] RF-CF5 扩大到 `hidden_size in {16,32,64}`。
- [ ] Adapter 扩大到 `adapter_dim in {16,32,64}`。
- [ ] 实现 pair-aware frozen adapter：`h_i, h_j, h_i*h_j, |h_i-h_j| -> pair prior`。
- [ ] 实现 long-window stitching consistency loss。
- [ ] 借鉴 StructRFM 的 curriculum 思路，把 CDMC/CTMC 编辑流改成“局部编辑 -> 全局结构约束”的课程学习：第一阶段只训练短 span / 局部 pair 修补和近邻 stem consistency，第二阶段加入 long-range pair、互选对称性和 window stitching，第三阶段再加入 nested/global legality、thermo-guided decode 与 full-structure projection；验收指标必须分开报告 local edit F1、long-range recall、novel_clan mean F1/MCC 和 global legality rate，确认课程学习不是只提升局部而牺牲全局结构。
- [ ] 借鉴 eFold 的 data-first 泛化路线，建立多源复杂度 curriculum：先用当前 MMseqs split 保持 benchmark 稳定，再分阶段加入 pri-miRNA/human_mRNA/viral/lncRNA/domain windows，并按 source/family/length/complexity 做 balanced sampling；每个阶段必须报告 `novel_clan`、viral、lncRNA、human_mRNA 和 long-range bins 的同向变化。
- [ ] 将热力学物理先验 `L_thermo` 接入当前 full-scale torch 主训练：当前 `lambda_thermo` 已在 stdlib/pilot 路径实现，但 torch backend 仍限制 `lambda_thermo=0`，且 Turner-style unpaired prior 的 `O(N L^3)` 预计算尚未做 GPU/缓存化工程验证；因此 RF-CF 主线目前只启用 legality/canonical mask、contact/long-range consistency 和 capacity/family-balanced 路线。进入主训练前需完成 torch 版 thermo gradient/缓存策略、小规模 ablation、再加入 RF-CF grid。
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
- [ ] 如果扩大 `hidden_size/adapter_dim` 后 `novel_clan` 仍不提升，停止单纯 scale-up，优先补数据多样性/复杂度 curriculum；这条来自 eFold/RNAndria 的 cross-family 泛化结论。
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
