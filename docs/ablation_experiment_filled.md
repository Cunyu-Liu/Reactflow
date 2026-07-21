# ReactFlow 消融实验表格（已回填结果）

本文件是 `docs/ablation_experiment_template.md` 的当前结果回填版。所有 full-data 数字来自服务器 artifact:

- 结果表: `artifacts/full_runs/full_ablation_20260709_003012/full_data_ablation_results.md`
- 图表: `artifacts/full_runs/full_ablation_20260709_003012/full_data_mean_f1.svg`
- JSON: `artifacts/full_runs/full_ablation_20260709_003012/runs/RF-A8-torch_full_data_e1_bs16/eval_summary.recovered.json`
- Profile summary: `artifacts/full_runs/full_ablation_20260709_003012/runs/RF-A8-torch_full_data_e1_bs16/profile.summary.json`

## 1. 当前 full-data 结果

| Run ID | 分组 | 变量 | 实验设置 | Train split | Eval tier | Count | Mean F1 | Micro F1 | Mean MCC | Micro MCC | 状态 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| RF-A8-torch_full_data_e1_bs16 | 加速后端 | torch full-data | `backend=torch`, `batch_size=16`, `epochs=1`, `lambda_react=0` | fallback `train.jsonl` 209,195 windows | PDB | 333 | 0.1931 | 0.1310 | 0.1722 | 0.1225 | completed |
| RF-A8-torch_full_data_e1_bs16 | 加速后端 | torch full-data | 同上 | 同上 | archiveII | 2,052 | 0.0320 | 0.0296 | 0.0263 | 0.0246 | completed |
| RF-A8-torch_full_data_e1_bs16 | 加速后端 | torch full-data | 同上 | 同上 | human_mRNA | 6,627 | 0.0118 | 0.0120 | 0.0107 | 0.0108 | completed |
| RF-A8-torch_full_data_e1_bs16 | 加速后端 | torch full-data | 同上 | 同上 | in_clan | 26,149 | 0.0317 | 0.0259 | 0.0280 | 0.0235 | completed |
| RF-A8-torch_full_data_e1_bs16 | 加速后端 | torch full-data | 同上 | 同上 | lncRNA | 289 | 0.0119 | 0.0119 | 0.0109 | 0.0106 | completed |
| RF-A8-torch_full_data_e1_bs16 | 加速后端 | torch full-data | 同上 | 同上 | novel_clan | 46,147 | 0.0360 | 0.0280 | 0.0320 | 0.0252 | completed |
| RF-A8-torch_full_data_e1_bs16 | 加速后端 | torch full-data | 同上 | 同上 | viral | 97 | 0.0170 | 0.0161 | 0.0152 | 0.0144 | completed |

## 2. Profile 结论

| Backend | Train windows | Epochs | Batch size | Epoch total | Slowest step | Step total | Mean per sample | 状态 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| torch | 209,195 | 1 | 16 | 13,516.25s | `model_forward` | 4,319.34s | 0.02065s | completed |
| torch | 209,195 | 1 | 16 | 13,516.25s | `projection_f1` | 3,919.57s | 0.01874s | completed |
| torch | 209,195 | 1 | 16 | 13,516.25s | `reactivity_loss_grad` | 3,023.17s | 0.01445s | completed |
| stdlib | partial | 1 attempted | full batch | stopped | `model_backward` | partial profile only | L=204 sample ~0.22s | stopped_for_performance |

## 3. Blockers / 自动调整记录

| Item | 状态 | 处理动作 | 下一步 |
|---|---|---|---|
| stdlib full-data run | stopped_for_performance | 运行超过 7 小时仍未完成 1 epoch，已停止并切换 torch backend | 保留 partial profile，后续不再用 stdlib 做 full-data 主路径 |
| full warm-start frozen export | completed | `ribonanzanet2_sharded_full/sharded_manifest.json` 已生成；full export 共 `409` 个 child shards、`208,905` 条 unique windows；checkpoint hash `c94031719c8a...` | 已被 RF-A1/RF-A2/RF-M1 warm-start 队列消费；后续 RF-CF/容量扩展实验继续复用该 frozen artifact |
| sharded frozen-feature loading | optimized | `FrozenFeatureLookup` 已从 whole-shard materialization 升级为 targeted NPZ-member read，并新增 LRU-capacity aware mini-batch multi-member prefetch：batch 边界按 shard 分组，只预取 batch 顺序中前 `K=max_loaded_shards` 个 missing shard，避免跨 shard batch 发生自驱逐；每个 selected shard 用一次 ZIP session 读取多个 `single` members；每个 shard 至少做一次 content hash 校验，后续重复访问复用 verified set；CLI `--frozen-cache-shards` 默认 `4` 控制 bounded LRU row cache | 已同步到服务器；本地与远端相关测试通过，远端 `tests/test_train.py` 覆盖 torch backend 的 `frozen_batch_prefetch` 路径；algorithm doc audit strict-ready，后续新增容量/adapter 实验继续继承该路径 |
| RF-A1-warm full-data | completed | full-run 队列状态：warm/contact/MMseqs final result 已完成；`RF-CF3-family-balanced_mmseqs_torch_full_data_e1_bs16` 7 tier ok，但 `novel_clan_mean_f1=0.0286` 未达标；`RF-CF1-contact-lam0p1/lam0p2/lam0p4/lam0p8` 已完成并生成 `cross_family_contact_sweep_results.json`，best `lambda=0.8` 为 `novel_clan_mean_f1=0.0435` 但仍低于 RF-M1-warm `0.0447`；RF-CF2 official `w=2` 已自动接力运行，RF-CF2 `w8 early` 为 `novel_clan_mean_f1=0.0404`；另有 RF-CF5 early 并行任务 | 已生成的 final result 由 `audit_final_queue.py` 内容契约证明；当前未完成 active 行继续进入 queue/progress/cross-family audits |
| RF-A3-contact watcher | completed | `scripts/run_contact_after_warm_rfam_current_exact.sh` 已完成默认 `lambda_contact=0.2`, `contact_negative_weight=0.25` 的 exact-split contact run，并生成 `contact_rfam_current_exact_results.json` | 默认 contact 配置未解决 novel_clan 指标；下一步需 RF-CF1 contact-strong sweep |
| run monitor snapshot | active | `scripts/monitor_reactflow_run.py` 已同步到服务器并生成当前 active run 的 `monitor_snapshot.json` 与 `.md`；当前 `RF-CF1-contact-lam0p1_mmseqs_torch_full_data_e1_bs16` snapshot 训练 epoch 已闭合，post-training eval 进度以 `active_eval_progress_audit.json` 为准，stderr `0 bytes` | 后续巡检统一使用该脚本，避免临时解析日志 |
| ablation summarizer | upgraded | `scripts/summarize_ablation_results.py` 现在读取 `eval_summary*.json`/`stdout*.json`、`profile.summary.json`、`monitor_snapshot.json`、`training_checkpoint.json` 和 `stderr.log`，输出 F1/MCC + seconds + samples/s + progress + checkpoint 状态 | 当前已用于 exact/MMseqs/RF-CF3 completed rows 和 RF-CF1 active row；后续 RF-CF 系列会进入同一张状态表 |
| current queue status | active | `current_queue_status.json/md/svg` 由统一刷新脚本自动生成；刷新脚本支持多 `--glob`，会同时覆盖 exact、MMseqs 和 RF-CF 队列；当前 `RF-CF1-contact-lam0p1/lam0p2/lam0p4/lam0p8` 为 `status=ok`，`novel_clan_mean_f1=0.0188/0.0245/0.0401/0.0435`；RF-CF2 official `w=2` 正在运行；RF-CF2 `w2/w4/w8 early` 已完成，`novel_clan_mean_f1=0.0213/0.0203/0.0404`；其余 RF-CF5 early run 继续运行/评估 | 后续 RF-CF、capacity 和 multi-seed run 目录出现后会自动进入同一张状态表 |
| paper artifact audit | active | 新增 `scripts/audit_paper_artifacts.py`，检查 public cache、MMseqs metadata、leakage-safe split、run stderr/profile/checkpoint/metrics，并输出 `paper_artifact_audit.json/md` | 当前 `ok_for_paper_table=true`, `pass=23`, `warn=0`, `fail=0`；artifact 完整性通过，但论文主结果仍受 cross-family 指标门槛约束 |
| paper artifact audit snapshot | active | `paper_artifact_audit.json/md` 由统一刷新脚本生成，检查 public cache、MMseqs metadata、leakage-safe split、run stderr/profile/checkpoint/metrics | 当前 `ok_for_paper_table=true`, `pass=23`, `warn=0`, `fail=0`；paper artifact 完整性通过后，最终论文 claim 仍需 cross-family claim gate 和多 seed 统计共同支撑 |
| runtime health audit | active | `runtime_health_audit.json/md` 已刷新；completed run 使用 `stdout.json + training_checkpoint.json + tiers` 作为完成证据，active run 仍要求 profile heartbeat；加载/post-training eval 阶段 `progress=None` 记 warn 而非 fail | 当前 `healthy=true`, `audited_run_count=13`, `pass=68`, `warn=1`, `fail=0`；只有未完成阶段的 watcher/manual pidfile 纳入 liveness gate；已完成阶段改由 final-result 内容契约证明 |
| system resource audit | active | 新增 `scripts/audit_system_resources.py`，固化 `nvidia-smi` GPU 利用率/显存和 watcher pidfile 的递归子进程 CPU/RSS 快照，输出 `system_resource_audit.json/md` | 当前 RF-CF active resource audit 为 `resource_healthy=true`, `pass=8`, `warn=2`, `fail=0`, `process_count=6`；服务器有 8 张 A100 可见，official RF-CF2 `w=2`、RF-CF5 `h32/h64` 等仍在运行/评估；warn 来自 manual pidfile 无子进程，不是 OOM/进程死亡 |
| profile bottleneck audit | active | 新增 `scripts/audit_profile_bottlenecks.py`，读取 active run 的 `monitor_snapshot.json` 并计算 phase 占比 `rho_p=T_p/sum_qT_q`，同时检查 `frozen_batch_prefetch` 是否出现在 profile 中 | 当前 `profile_bottleneck_audit.json/md` 显示 `bottleneck_healthy=true`, `pass=35`, `warn=0`, `fail=0`；`RF-CF1-contact-lam0p1_mmseqs_torch_full_data_e1_bs16` 的 `path_sample_features` 占比 `0.8097`；该 audit 同时记录 `frozen_batch_prefetch` 是否出现，用于量化预取收益 |
| queue progress audit | active | 新增 `scripts/audit_queue_progress.py`，把 `current_queue_status.json` 追加到 `logs/current_queue_status_history.jsonl`，并审计运行中任务的 progress delta、throughput floor 和趋势 ETA | 远程 `queue_progress_audit.json/md` 已生成；当前 `progress_healthy=true`, `pass=77`, `warn=4`, `fail=0`；official RF-CF2 `w=2` 正在运行；early/post-training eval run 暂无 progress 时记 warn，不是运行失败 |
| active eval progress audit | active | 新增 `scripts/audit_active_eval_progress.py`，从 active run 的 `profile.jsonl` tail 读取最新 `eval_sample_total` 事件，并按 tier JSONL 行数把全局 eval `sample_index` 转成 tier-local progress 与 ETA | 当前 `eval_progress_healthy=true`, `pass=2`, `warn=1`, `fail=0`；RF-CF2/RF-CF5 active eval 继续推进 |
| cross-family metric audit | active | 新增 `scripts/audit_cross_family_metrics.py`，从 `current_queue_status.json` 或 final result rows 中按 run 匹配 `in_clan` / `novel_clan`，计算 `novel_clan_mean_f1`、`novel_clan_mean_mcc`、`gap_mean_f1=F1(in_clan)-F1(novel_clan)` 与 `retention`；低分先作为 improvement warning，不误伤运行队列 | 当前 MMseqs best 为 RF-CF1 contact `lambda=0.8`：`novel_clan_mean_f1=0.0435`, `novel_clan_mean_mcc=0.0430`, `gap_mean_f1=0.0008`；仍低于 RF-M1-warm `0.0447` 和 gate `0.15`；`cross_family_claim_ready=false` |
| ablation ledger status updater | active | `scripts/update_ablation_ledger_status.py` 从 `current_queue_status.json`、`queue_progress_audit.json`、`profile_bottleneck_audit.json` 和 `system_resource_audit.json` 自动回填台账监控行；`refresh_full_run_status.sh` 已在 manifest/readiness 之前调用该脚本 | 已补 RF-CF3 pending row 的 `progress_fraction=None` / `samples_per_second=None` 稳健处理；本地与远端 targeted tests 通过，后续巡检不再手动 patch 这些监控数字 |
| queue preflight audit | completed | `scripts/audit_queue_preflight.py` 对 full-run 队列做静态预检：检查 warm、warm recovery、contact、MMseqs final、RF-CF3 cross-family、RF-CF1 contact-sweep、RF-CF2 long-range、RF-CF5 capacity 和 final-readiness watcher 的 `bash -n` 语法和关键 marker，同时验证 exact/MMseqs split、evaluation cache、frozen manifest、batch retry ladder、OOM/instability regex 与预期 final output 文件约定 | 当前 `preflight_healthy=true`, `pass=151`, `warn=0`, `fail=0`；RF-CF3 的 `--family-balanced-batches`、RF-CF1 的 `--lambda-contact`、RF-CF2 的 `--contact-long-range-weight`、RF-CF5 的 `--hidden-size`/`--adapter-dim`，以及 `cross_family_balanced_results.json` / `cross_family_contact_sweep_results.json` / `cross_family_long_range_results.json` / `cross_family_capacity_results.json` 均已纳入预检，降低后续 cross-family 接力实验遗漏关键开关的风险 |
| non-finite retry guard | upgraded | `train_pilot`/`train_pilot_torch` 新增 `_assert_finite_training_scalar`，对 sample loss、batch loss 和 epoch total 的 NaN/Inf 抛出 `FloatingPointError: non-finite training value`；torch backend 还会在 backward/update 阶段检查 `gradient:*` 和 `parameter:*` 张量，非有限值会抛出 `FloatingPointError: non-finite training tensor`；contact/MMseqs/RF-CF watcher 的 `instability_pattern` 会匹配 OOM、`FloatingPointError`、`non-finite`、`nan/inf` 和发散关键词 | 本地与远端 targeted tests 通过；queue preflight 当前 `pass=151`, `fail=0`；后续队列会按 `16 -> 8 -> 4 -> 2 -> 1` batch ladder 对 OOM/数值异常自动降 batch 重试 |
| warm tail recovery watcher | completed | 新增 `scripts/run_warm_tail_recovery_after_watcher_exit.sh`：当主 warm watcher 仍 alive 时只 sleep；若主 watcher 退出但 `warm_rfam_current_exact_results.json` 仍缺失，则检测 RF-A1/RF-A2-adapter4/RF-A2-adapter16 哪些已有可解析 metrics，只补跑缺失项，并用 `--frozen-cache-shards 4`、`instability_pattern` 和 `16 -> 8 -> 4 -> 2 -> 1` batch ladder 兜底 | warm final result 已生成并通过内容契约；该 watcher 作为后续长队列恢复模板保留，不再作为当前 liveness blocker |
| final queue audit | completed | `scripts/audit_final_queue.py` 已升级为内容级接力审计，并与 `scripts/audit_goal_readiness.py` 共享 `reactflow.final_results` 结果契约：结果缺失时 watcher 必须仍 alive；结果文件一旦出现，必须是覆盖七个 eval tiers 的 `status=ok` metric-row 列表，非空但仅含 `running_or_pending_json`、缺少 F1/MCC/count 或出现 NaN/Inf metric 的文件会直接 fail | 远程 `final_queue_audit.json/md` 已生成；当前 `final_queue_healthy=true`, `final_results_ready=true`, `pass=4`, `warn=0`, `fail=0`；warm/contact/MMseqs 三份 final result 均已通过内容契约；本地与远端 `tests/test_final_results.py`/`tests/test_audit_final_queue.py`/`tests/test_audit_goal_readiness.py` 通过 |
| algorithm doc audit | completed | `scripts/audit_algorithm_docs.py` 已接入统一刷新，AST 扫描公开函数/方法/类的 docstring、`Complexity` 标记、关键算法公式标记和占位实现；当前报告 `public_nodes=277`, `passing_doc_rows=277`, `placeholder_bodies=0`, `text_markers=0`, `missing_complexity=0`, `missing_docstrings=0`, `missing_math_markers=0`, `strict_ready=true` | 算法文档审计门槛已闭环；后续新增公开 API 必须保持该脚本通过 |
| coverage gate audit | completed | 新增 `scripts/audit_coverage_gate.py`，读取 `coverage.json` 并输出 `coverage_audit.json/md`；当前远程 `percent_covered=93.07996832937451`, `threshold=90.0`, `passed=true` | 覆盖率证据已从终端输出固化为 artifact，并纳入 goal readiness |
| reproducibility manifest | active | `scripts/build_reproducibility_manifest.py` 记录环境、包版本、工具路径、代码/文档/脚本/测试和关键 artifact 的 SHA256、大小、mtime 以及审计摘要；默认避免主动重哈希大体积 frozen shard | 远程已生成 `reproducibility_manifest.json/md`，当前 `file_count=295`；已覆盖 active eval progress、cross-family metric audit、RF-CF3 watcher、RF-CF1 contact-sweep watcher、RF-CF2 long-range watcher、RF-CF5 capacity watcher、queue preflight、runtime/resource/profile/final/readiness audits、progress history、MMseqs logs 与关键脚本/测试；记录 `torch=2.10.0`, `numpy=2.2.6`, `mmseqs=/home/liucunyu/tools/mmseqs2-avx2/bin/mmseqs` |
| goal readiness audit | active | `scripts/audit_goal_readiness.py` 聚合 algorithm doc、runtime health、system resource、queue progress、queue preflight、profile bottleneck、final queue、paper artifact、reproducibility manifest、coverage audit、README/data governance、cross-family claim gate 和 final result 文件；final result 通过 `reactflow.final_results` 统一契约校验，必须是非空 `status=ok` metric-row 列表，覆盖七个 eval tiers 且 F1/MCC/count 字段为有限值 | 当前远程报告 `pass=23`, `fail=1`, `ready_for_goal_completion=false`；唯一失败项为 `cross_family_metrics`，因为 `cross_family_claim_ready=false`；即使三份 final result 已齐全，也不得标记总目标完成 |
| final readiness watcher | completed | 新增 `scripts/run_goal_readiness_after_final_results.sh`，等待 warm/contact/MMseqs 三个 final result 文件齐全后自动运行 `refresh_full_run_status.sh` 和 `audit_goal_readiness.py --fail-if-not-ready` | 三份 final result 已齐全，该一次性 watcher 已完成使命；`refresh_full_run_status.sh` 现仅在 final result 未齐时检查该 pidfile，避免 dead one-shot pid 误伤 runtime/resource 审计 |
| unified status refresh | upgraded | `scripts/refresh_full_run_status.sh` 现在会为所有匹配 `QUEUE_GLOBS` 的 active run 生成 `monitor_snapshot.json/md`，并写 `logs/refresh_full_run_status.monitor_runs.jsonl` 记录本次覆盖的 run 和 total-sample denominator；同时刷新 `current_queue_status.*`、`paper_artifact_audit.*`、`runtime_health_audit.*`、`cross_family_metric_audit.*` 和 `goal_readiness_audit.*` | exact/MMseqs/RF-CF 队列统一进入 monitor 与队列表；完成态 export/MMseqs/split 由 artifact audit 验证 |
| final SOTA split | completed_mmseqs | 已新增 `scripts/build_rfam_metadata.py` 和 `reactflow.rfam_metadata`；服务器已安装用户级 MMseqs2 `/home/liucunyu/tools/mmseqs2-avx2/bin/mmseqs`；`rfam_current_mmseqs_metadata.tsv` 和 `splits/rfam_current_mmseqs_seed0` 已生成并通过 `validate_split_leakage` | MMseqs split 已用于 RF-M0/RF-M1 final 和 RF-CF3 active run；exact split 只保留为工程诊断，不进入最终主表 claim |
| MMseqs final run queue | completed | `scripts/run_mmseqs_final_after_exact_queue.sh` 已完成，exact RF-A3-contact 后运行了 `RF-M0-base` 和 `RF-M1-warm` | 已输出 `mmseqs_final_results.json/md/svg`；RF-M0 novel_clan mean F1 `0.0267`，RF-M1-warm novel_clan mean F1 `0.0447`，仍未达 cross-family claim gate |
| RF-CF3 family-balanced watcher | completed_below_gate | 新增 `scripts/run_cross_family_after_mmseqs_final.sh`，等待 `mmseqs_final_results.json` 后先审计 MMseqs cross-family claim gate；若 `cross_family_claim_ready=false`，自动启动 `RF-CF3-family-balanced`，使用 `--family-balanced-batches` 按 `cluster -> family -> source_id` 在 length bucket 内轮转采样 | RF-CF3 已生成 `cross_family_balanced_results.json` 与 metric audit；`novel_clan_mean_f1=0.0286`，低于 RF-M1-warm `0.0447` 和 early gate `0.15`，因此不进入 claim |
| RF-CF1 contact-strong watcher | completed_below_gate | 新增 `scripts/run_contact_sweep_after_cross_family_balanced.sh`，等待 `cross_family_balanced_results.json` 后先审计 RF-CF3 claim gate；若 `cross_family_claim_ready=false`，自动按 `CONTACT_SWEEP_LAMBDAS="0.1 0.2 0.4 0.8"` 在 MMseqs split 上运行 `RF-CF1-contact-strong` sweep，使用 `--lambda-contact` 和 frozen adapter | RF-CF1 已完成并输出 `cross_family_contact_sweep_results.json` 与 `cross_family_contact_sweep_metric_audit.json`；`lambda_contact=0.1/0.2/0.4/0.8` 的 `novel_clan_mean_f1=0.0188/0.0245/0.0401/0.0435`，best `lambda=0.8` 仍低于 RF-M1-warm `0.0447` 与 early gate `0.15` |
| RF-CF2 long-range watcher | active_plus_early_complete | 新增 `scripts/run_long_range_after_contact_sweep.sh`，等待 `cross_family_contact_sweep_results.json` 后先审计 RF-CF1 claim gate；若 `cross_family_claim_ready=false`，自动按 `LONG_RANGE_WEIGHTS="2 4 8"` 在 MMseqs split 上运行 `RF-CF2-long-range` sweep，使用 `--contact-long-range-min-distance 24` 和 `--contact-long-range-weight` | 官方 watcher 已于 2026-07-16 09:59 CST 启动 `RF-CF2-long-range-w2_mmseqs_torch_full_data_e1_bs16`；early `w2/w4/w8` 已完成，`novel_clan_mean_f1=0.0213/0.0203/0.0404`，独立 run_id 不覆盖官方结果 |
| RF-CF5 capacity watcher | waiting_official_plus_early | 新增 `scripts/run_capacity_after_long_range.sh`，等待 `cross_family_long_range_results.json` 后先审计 RF-CF2 claim gate；若 `cross_family_claim_ready=false`，自动按 `CAPACITY_GRID="16:16 32:16"` 在 MMseqs split 上运行 `RF-CF5-capacity` sweep，组合更大 hidden size、adapter dim、family-balanced、contact auxiliary 和 long-range reweighting | 官方 watcher 仍等待 RF-CF2 long-range 结果；为抢占空闲 GPU，已并行启动 early run：`h16:a16`、`h32:a16` 在 post-training eval，`h64:a32` 继续训练，独立 run_id 不覆盖官方结果 |
| stdout tee | warning | `tee` 目录创建早于 CLI output-dir，stdout 文件未保存 | 已从 captured stdout 恢复 `eval_summary.recovered.json`，后续脚本先建 output-dir |

### Rfam metadata split 账本

| Artifact | 值 |
|---|---|
| Metadata TSV | `artifacts/full_runs/full_ablation_20260709_003012/metadata/rfam_current_exact_metadata.tsv` |
| Metadata manifest | `artifacts/full_runs/full_ablation_20260709_003012/metadata/rfam_current_exact_metadata.manifest.json` |
| Split dir | `artifacts/full_runs/full_ablation_20260709_003012/splits/rfam_current_exact_seed0` |
| Input / metadata records | 307,641 / 307,641 |
| Rfam accessions | 29 unique; 225,699 records parsed |
| Rfam clan mapped / family fallback / no accession | 106,401 / 119,298 / 81,942 |
| Cluster method | `exact` fallback because server lacks `mmseqs`, `cd-hit-est`, `seqkit` |
| Split counts | train 206,443; val 25,805; test 25,805; novel 49,588 |

### MMseqs2 full split 账本

| Artifact | 值 |
|---|---|
| MMseqs binary | `/home/liucunyu/tools/mmseqs2-avx2/bin/mmseqs` |
| MMseqs version hash | `771ee2387007d040819436e26062ff69f320144e` |
| Metadata output | `artifacts/full_runs/full_ablation_20260709_003012/metadata/rfam_current_mmseqs_metadata.tsv` |
| Metadata manifest | `artifacts/full_runs/full_ablation_20260709_003012/metadata/rfam_current_mmseqs_metadata.manifest.json` |
| Metadata PID/log | `logs/build_rfam_metadata_mmseqs.pid` / `logs/build_rfam_metadata_mmseqs.log` |
| Split watcher | `scripts/run_mmseqs_split_after_metadata.sh` -> `splits/rfam_current_mmseqs_seed0` |
| MMseqs cluster count | 180,671 |
| MMseqs split group count | 70,144 |
| MMseqs split counts | train 228,282; val 16,606; test 16,606; novel 46,147 |
| Current status | completed and leakage-validated; `--threads 2` triggered MMseqs2 nucleotide `align2clust` segfault (`getDbKey: local id >= db size`), final metadata used `--threads 1` |

### Sequence-identity cluster 方法账本

| Method | 用途 | 数学定义 / 命令 | 是否可进论文主表 |
|---|---|---|---|
| `mmseqs` | full-scale final split | `mmseqs easy-cluster --min-seq-id 0.9 -c 0.8 --cov-mode 1` | 是，最终主表要求 |
| `python-identity` | 小规模 sensitivity / CI | ungapped global identity `matches/min_len >= 0.9` 且 coverage `min_len/max_len >= 0.8`；复杂度 `O(N^2 L)`，有 `--python-identity-max-records` 保护 | 否，只能作为敏感性分析 |
| `exact` | 最低成本 fallback | SHA1 exact-sequence cluster | 否，只能标注 fallback |

## 4. 当前不能作为最终 SOTA 的原因

- MMseqs2 90% identity split 已完成并用于 RF-M0/RF-M1 final；但当前 MMseqs `novel_clan` F1 仍极低。
- 三份 final result 已闭环；当前不能作为最终 SOTA 的原因不再是缺文件，而是 `cross_family_claim_ready=false`。
- 当前 full-data run 仍主要是 1 epoch weak engineering baseline；论文主表需要更大容量、多 epoch、多 seed。
- cross-family 主指标尚未达标：当前 best `novel_clan` mean F1 仍为 `0.0624`，MMseqs RF-M1-warm 为 `0.0447`；任何后续主张都必须来自 MMseqs split，并报告 `novel_clan_mean_f1`、`novel_clan_mean_mcc`、`gap_mean_f1` 和 `retention`。

## 5. 下一批必须补跑

- `RF-M0/RF-M1`: 优先完成 MMseqs final split base/warm queue，把 `novel_clan` 作为主排序指标。
- `RF-CF1-contact-strong`: 在 `lambda_contact in {0.1,0.2,0.4,0.8}` 上补 sweep，检查 novel F1/MCC 与 reactivity consistency 是否同向改善。
- `RF-CF2-long-range-head`: 增加 long-range pair diagnostic / reweighting，单独报告 `|i-j| >= 24` recall。
- `RF-CF3-family-balanced`: 引入 family/MMseqs-cluster balanced sampler，验证 family_macro_f1 是否改善。
- `RF-CF4-thermo-decode`: 固定 best checkpoint 做 `eta` guidance scan，只接受 novel F1/MCC 与能量合法性同时改善的设置。
- `RF-A1/RF-A2`: RF-A1 与 RF-A2-adapter4 已回填 tier 指标，继续等待 RF-A2-adapter16 完成并生成 `warm_rfam_current_exact_results.json`；必要时按 active profile 调整 batch/cache。
- baseline rerun: eFold/RNAndria、RibonanzaNet2-derived。
