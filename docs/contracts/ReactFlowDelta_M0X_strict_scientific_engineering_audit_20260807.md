# ReactFlow-Delta M0-X 严格、证据驱动的科研与工程审查

审查日期：2026-08-07  
远端仓库：`/home/cunyuliu/reactflow_delta_goal_20260729`  
冻结快照：`codex/reactflow-delta-d0r@9fe8ad8c6966a8d6640f58ba2a05a01d2d91d070`  
审查方式：远端只读、validation development artifact 的确定性无拟合重放、外部主来源定向检索  
证据词汇：`CONFIRMED_FACT`、`REASONED_INFERENCE`、`UNKNOWN_NOT_ASSERTED`、`NEW_EXPERIMENT_REQUIRED`

## 1. 执行摘要

### 总体裁决

**当前项目处于 A：工程/科研原型，尚未形成论文级方法贡献。方法论文评分为 20/50；现阶段不具备 SOTA 主张，也不具备 confirmatory 跨 publication 结论。最现实的下一阶段不是继续堆叠模型模块，而是重建 benchmark/evaluator，并用简单模型重新检验任务是否可识别。**

这一裁决不是因为“神经网络分数低”这一件事，而是四条相互独立的 P0 证据链同时不成立：

1. `[CONFIRMED_FACT | FAIL]` **authority 未闭合。** `configs/reactflow_delta/active_contract.yaml` 自报 epoch 12，却仍绑定 epoch 5；旧 bundle 对当前 active manifest 校验失败。epoch 13 amendment 已被 registry 使用，但未被 active authority/bundle 绑定。正式 M0-X 已 `FAIL`，active manifest 仍写 `RUNNING / NOT_RUN / training_allowed=true`。在其自身 `FAIL_CLOSED` 规则下，当前训练授权不能被信任。
2. `[CONFIRMED_FACT | INVALIDATED_BY_EXPOSURE]` **现有 test 已被 PH0 消费。** `scripts/reactflow_delta/ph0x_caller.py:374-377` 把全部 canonical records 传给 `NoiseModel` 和 `frozen_call`；`frozen_call:287-330` 遍历全部 split roles，既有 aggregate 还记录了 193 个 test changers。D0-era ledger 的 `SEALED / events=[]` 已过时。16SFWJ 不能再充当一次性 confirmatory test。
3. `[CONFIRMED_FACT | FAIL]` **没有统一 estimand。** PH0 caller 在 validation 给出 49/548 changers；dev10 训练的是逐 position 的 `abs(delta)>0.05*scale`；M0 gate 以“至少一半 position 为正”得到 102/548，却用 `max(position probability)` 评分。caller、loss、checkpoint selection、pair label、score aggregation 和最终统计检验没有围绕同一对象。
4. `[CONFIRMED_FACT | FAIL]` **validation 不是跨 publication 验证。** CIDGMP 与 TRP4P6 是两个 study ID、548 个 pair，但同属 PMID 25303992，因此最高层有效验证样本数是 1 个 publication。任何所谓“跨研究 CI”都只能是 development diagnostic。

### 当前 M0-X 到底说明了什么

正式 artifact `results/m0x_gate_20260807/gate_report.json`（SHA-256 `068ee4d7…`）记录：

| 项目 | 正式值 | 本审查裁决 |
|---|---:|---|
| candidate | `epro_dev10_best` | 实际为 42 维 position features 上的 flat MLP，不是 EPRO propagation operator |
| validation pairs | 548 | 只有一个 publication domain，且长期用于开发 |
| majority positives | 102，18.61% | 与 PH0 的 49/548 不同；不是冻结 caller endpoint |
| candidate study-macro AUPRC | 0.265203 | legacy 实现可精确重放，但 AP 对 ties 的行顺序敏感 |
| tree study-macro AUPRC | 0.298319 | learned HistGradientBoosting，却被元数据写成 `is_learned=false, param_count=0` |
| candidate − tree | −0.033117 | 正式方向不利 |
| 95% CI | [−0.135609, 0.066899] | publication N=1，不能作跨 publication confirmatory CI |
| permutation p | 1.0，100 次 | 只能说明当前口径没有候选优越性证据，不能说明生物任务不可学习 |
| formal gate | `FAIL` | 保留；不解锁 P0-X，不支持 SOTA |

本轮用冻结的 validation-only pair scores 做了无拟合重放：legacy candidate study-macro AP 精确复现为 `0.2652025367`。标准 tie-aware AP 得到 candidate-only `0.3008413277`；100 个确定性行重排使 legacy pooled AP 在 `0.3460–0.4578` 之间变化，而标准 AP 固定为 `0.3829001574`。但冻结 bundle 没有 tree pair scores，因此不能公平重算 tree，也不能把 0.3008 与旧 tree 0.2983 直接比较。正确状态是：

> `FORMAL_M0_FAIL` + `EVALUATOR_REPRODUCIBILITY_FAIL` + `STANDARDIZED_RANK_UNRESOLVED_NOT_REPLAYABLE`，而不是“模型其实赢了”。

### 距离 SOTA 的瓶颈排序

| 优先级 | 瓶颈 | 证据等级 | 为什么先于架构 |
|---|---|---|---|
| 1 | endpoint、mask、score、selection 与统计单位不一致 | `CONFIRMED_FACT` | 不知道模型究竟在优化和评价什么，任何增益都不可解释 |
| 2 | test 暴露、validation publication N=1 | `CONFIRMED_FACT` | 没有独立证据域，无法区分泛化与开发集选择 |
| 3 | 1024 assets 仅解析 414；独立数据有效 N 很小 | `CONFIRMED_FACT` | 4472 pair/约47万 position rows 不能替代 publication-level N |
| 4 | B0/PH0/M0 evaluator 和训练实现 bug | `CONFIRMED_FACT` | 既有 PASS/FAIL 的科学含义被削弱，模型概念尚未被公平检验 |
| 5 | actual-alt、pair-aware 与传播归纳偏置不足 | `REASONED_INFERENCE` | 是可能的模型上限问题，但只有前四项修复后才能识别其增量价值 |
| 6 | 领域强基线没有运行 | `CONFIRMED_FACT` | 不能建立 SOTA 排名或现有方法局限 |

### 四个决策问题的直接答案

| 决策问题 | 答案 |
|---|---|
| 仓库实现了什么？ | 数据层实现了 matched WT–single-mutant probing resource；最终候选实现的是逐 position 手工特征分类器，再以 ad hoc max 聚合到 pair。EPRO operator 代码存在，但不是最终候选的实际主体。 |
| 低性能主要来自哪里？ | 当前无法单因果归因。已确认的首要原因是 estimand/评价协议、有效独立 N、test 暴露与训练/evaluator bug；架构能力不足是次级且尚未被干净识别的假设。 |
| 能否支持 SOTA/投稿？ | 不能支持 SOTA 或跨 publication 方法论文。重建后有 benchmark/resource/negative-result 潜力；领域方法论文需新增独立 publications 和 confirmatory gain；顶级交叉领域当前没有证据基础。 |
| 下一步是什么？ | 停止架构自由搜索，先完成 authority/exposure 终态、1024-asset disposition、新 caller/mask/publication split 和统一 evaluator；再用简单模型做 P2 learnability gate。 |

### 审查边界与完整性声明

- 本轮没有读取 test 样本、test 标签、test predictions，也没有重算任何 test metric。
- 本轮没有训练或拟合模型、tree、caller、calibrator 或新 baseline。
- 本轮只对已存在的 validation development pair scores 做确定性无拟合重算。
- 本轮没有写远端文件、修改合同/authority/ledger/Git 状态或干预远端进程。
- 历史信息只用于定位风险；所有当前结论均以本轮远端快照或主来源检索为准。
- 详细 claim–evidence 映射见 `ReactFlowDelta_M0X_evidence_ledger_20260807.tsv`；重放规范与结果见 `ReactFlowDelta_M0X_readonly_replay_20260807.json`。

## 2. 项目真实状态

### 2.1 Authority、阶段与暴露的真实状态

| 层级 | 仓库声明 | 本轮实际核验 | 裁决 |
|---|---|---|---|
| 合同文本 | V4 为项目合同 | 远端合同 SHA-256 `631962f8…`；本地附件 `51b508e0…`，字节不一致 | 远端文本作为本轮权威；本地只做差异参考 |
| active authority | epoch 12，M0X_AUTHORIZED | integrity 仍指向 epoch 5；epoch 5 bundle 对 active manifest `FAILED` | `FAIL_CLOSED_AUTHORITY_INTEGRITY_FAILURE` |
| epoch 13 scope | unlimited iterations/capacity extension | amendment 存在且 registry 消费，但 active authority 未 bind；无 epoch 13 bundle/sentinel | `LOCATED_NOT_INTEGRITY_BOUND` |
| M0 phase | RUNNING / NOT_RUN / training allowed | final gate 已 `FAIL / DEVELOPMENT_ONLY / no P0 unlock` | active 状态过期；正式科学终态是 `FAIL` |
| test ledger | sealed/no events | PH0 代码与 aggregate 已消费 test outcome | `INVALIDATED_FOR_CONFIRMATORY_USE` |
| Git/evidence | tracked clean | 大量 untracked dev11/dev12/SOTA/eval recovery scripts/results | tracked clean，不是 evidence clean |

关键证据：

- `configs/reactflow_delta/active_contract.yaml:7-12,25-26,69-98,313-329,403-430`
- `configs/reactflow_delta/authority_epoch_5.bundle.sha256`
- `docs/contracts/amendments/reactflow_delta_v4_m0x_epro_scope_20260805.yaml`
- `docs/governance/m0x_window_registry_20260804.json`
- `results/m0x_gate_20260807/gate_report.json`
- `data_registry/d0x/{initial_exposure_ledger,test_access_ledger}.yaml`

本审查不修改 authority。若以后获新授权，治理上必须先：将 M0-X terminalize 为 FAIL、关闭 training、把 current phase 移到报告/修复阶段、绑定最终 exposure、生成新的 bundle/sentinel。否则不能把下一次训练称为合同内连续执行。

### 2.2 从数据到论文的成熟度矩阵

以下严格区分“有代码”“可运行”“有可靠结果”“足以支撑论文”。

| 环节 | 有代码 | 可运行证据 | 可靠结果 | 足以支撑论文主张 | 审查结论与核心证据 |
|---|---|---|---|---|---|
| 数据获取 | 是 | 是 | 部分 | 否 | 1024 frozen assets，但只解析 414；610 未完成 controlled disposition。`D0 terminal bb4b3cf8…` |
| 清洗/预处理 | 是 | 是 | 部分 | 否 | 4472 primary pairs 已构建；primary mask 未排除 edited/alignment/probe eligibility changes。`d1x_canonicalize.py::_build_reactivity_layers` |
| 数据划分 | 是 | 是 | 开发级 | 否 | train/val/test 已分；parent 定义漂移，family annotation 缺失，validation publication N=1，test 已消费。`d2x_split.py` |
| noise/caller | 是 | 是 | 不可靠 | 否 | ICC 异质；null 打散空间相关；伪随机不可复现；sliding-window 实现偏差；caller 与 gate 标签不一致。核心实现见 `ph0x_caller.py::_max_cluster:78-98,_null_distribution:217-271`；独立 permutation 路径另有随机性问题。 |
| 模型实现 | 是 | 是 | 部分 | 否 | flat MLP 能训练；EPRO operator 也有代码，但 exact-alt、mask、scheduler、梯度和结构 summary 均有问题。 |
| 训练流程 | 是 | 是 | 开发级 | 否 | dev10 有 checkpoint；逐 position sampling/selection 与 pair endpoint 不一致；dev05 scheduler 严重缩小 LR。 |
| 推理流程 | 是 | 是 | 开发级 | 否 | 可产 position scores；没有语义一致的 pair probability/magnitude output；max 后处理替代模型输出。 |
| 评估流程 | 是 | 是 | 失败 | 否 | formal gate 执行并诚实 FAIL，但 AP ties、signed/absolute、bootstrap、permutation、score type 有实现问题。 |
| 内部基线 | 是 | 是 | 部分 | 否 | trivial/P2/tree 已跑；tree 元数据错误；现有最强正式 tree 胜 candidate。 |
| 公开强基线 | 部分适配代码/计划 | 多数否 | 否 | 否 | RNAsnp/SNPfold/remuRNA/Riprap/VariantFoldRNA/RibonanzaNet 未在当前 frozen protocol 上执行。 |
| 消融 | 零散 | 是 | 否 | 否 | structure 增量约0.0009，仅 position metric、无配对 seeds；exact-alt、WT anchor、nonlocal propagation 未做最终 endpoint 消融。 |
| 泛化 | split 有 | 是 | 否 | 否 | 两个 validation study 同一 PMID；无有效 publication-held-out confirmatory result。 |
| 可解释性/机制 | 少量 proxy | 部分 | 否 | 否 | dev12 z-max 属事后 proxy/sign 搜索；没有跨 publication 机制证据。 |
| 复现性 | manifest/hash 较多 | 部分 | 否 | 否 | authority/bundle 断裂，关键 prediction/source untracked，candidate freeze 未形成完整 code–data–split–prediction closure。 |
| 工程质量 | 中等偏强 | 是 | 部分 | 否 | 有 gated phases、terminal artifacts、测试与日志；但状态传播、实现语义和 metadata QA 不足。 |
| 论文材料 | 合同、报告、结果均有 | 不适用 | 否 | 否 | 科学问题有价值，但方法→能力、实验→主张、结果→广泛价值三处断裂。 |

### 2.3 数据规模的正确口径

| 口径 | 数量 | 可否视为独立 N |
|---|---:|---|
| frozen source assets | 1024 | 否；其中 610 未解析 |
| parsed assets | 414 | 否；一个 asset 可含多 profile/pair |
| primary exact pairs | 4472 | 否；共享 WT、sequence、study、publication |
| position training rows | 约 471344 | 否；由 pair 展平而来，强相关 |
| code-defined source parents | 42 | 否；定义为 source_accession，跨阶段不一致 |
| unique sequences | 25 | 更接近序列域，但仍不等于独立 publication |
| studies | 9 | 不保证 publication 独立 |
| publications | 8 | 全数据域上限；当前 validation 只有 1 |
| validation studies | 2 | 同一 PMID，不是 2 个 publication replicates |
| validation publications | 1 | 不能支持跨 publication inference |

### 2.4 D0–M0 阶段状态重裁决

| 阶段 | 工程状态 | 科学状态 | 论文状态 |
|---|---|---|---|
| D0-X | materialization 完成 | `INCOMPLETE_RECALL` | 不能声称公开数据完整 |
| D1-X | canonical pairs 生成 | `PRIMARY_MASK_CONTRACT_GAP` | endpoint 尚不合格 |
| D2-X | split/manifest 生成 | `LOW_EFFECTIVE_N / FAMILY_UNKNOWN` | 不能声称 robust OOD split |
| PH0-X | caller/terminal 可运行 | `INVALIDATED_BY_EXPOSURE_AND_NULL_IMPLEMENTATION` | 不能作为 frozen caller/test 证据 |
| B0-X | baseline/evaluator 可运行 | `SCIENTIFIC_PASS_INVALIDATED` | 不能证明 learnability/data sufficiency |
| M0-X | gate artifact 完成 | `FORMAL_FAIL + METRIC_IMPLEMENTATION_GAPS` | `NO_SOTA / NO_CONFIRMATORY_CLAIM` |

### 2.5 七类科研完整性失败模式

| 失败模式 | 当前判定 | 仓库现象 | 处理边界 |
|---|---|---|---|
| Implementation bug | `CONFIRMED` | AP ties、B0 permutation、learning curve、LR scheduler、BPP diagonal、tree metadata | 所有受影响结果降级；修复需新版本、新哈希 |
| Citation hallucination | `NOT_OBSERVED_IN_THIS_AUDIT` | 本报告关键外部方法已用论文/官方来源核验 | bounded absence 不写成“first” |
| Hallucinated result | `RISK_PRESENT` | 历史 0.7435 与 formal 0.2652属于不同单位/endpoint；untracked结果可被误当正式 | 不说结果不存在，但标明 `SUPERSEDED/DEVELOPMENT_ONLY` |
| Shortcut | `CONFIRMED_RISK` | position flattening、目标mask进入physics、max聚合、missing→0、study主导 | 新 protocol 必须预注册信息权限和 mask |
| Bug-as-insight | `CONFIRMED_RISK` | dev12 sign flip/z-max 可能把失败回归包装成“机制” | proxy 仅探索，不升格为主张 |
| Methodology fabrication | `CONFIRMED_IMPLEMENTATION_MISMATCH` | 文档写 train+validation-only/sliding/absolute max，代码行为不同 | 论文方法必须按实际代码写，并先修复差异 |
| Frame lock | `CONFIRMED_RISK` | 在 endpoint 尚未成立时持续把问题定义为“模型不够复杂” | 先做 benchmark/P2 falsification gate |

## 3. 核心科学问题审查

### 3.1 一句话科学问题

**在不读取 mutant 实验 profile 的条件下，能否由 WT sequence、exact single-nucleotide mutation、允许的 WT experimental state 与实验条件，跨 publication 预测 mutation-induced experimental reactivity response？**

这是有价值、可证伪、可以形成论文的核心问题。但当前代码、实验和文档**没有围绕同一个形式化任务收敛**：合同强调 train-only replicate-aware pair caller；dev10 优化逐 position threshold classification；M0 gate评价 majority label；历史报告又以 position AUPRC 或连续回归 proxy 为主。因此，“项目问题清晰”与“当前执行一致”必须分开判断。

### 3.2 目标、输入、输出和假设

| 要素 | 应有定义 | 当前实现 | 裁决 |
|---|---|---|---|
| 预测单位 | matched WT–exact single-mutant pair | 训练单位为 position row，最终才聚合 pair | 不一致 |
| 输入 | WT sequence；ref/alt/position；条件；允许的 WT reactivity；不含 mutant profile | 手工position features、WT/local features、平均三alt thermo、contact/BPP；mask可能带目标缺失模式 | actual-alt与信息权限不干净 |
| 主输出 | pair-level `P(C_i=1)`，其中C由train-only frozen caller定义 | `max(position probability)` | 不是直接pair probability |
| 条件输出 | changer 条件下 profile-level Δreactivity magnitude | dev12 regression/proxies | 当前为负或事后proxy，未通过 |
| 次输出 | eligible positions 上的连续 noise-weighted signed Δ | 有回归尝试 | 评价与主任务混用；尚无可信skill |
| 核心假设 | exact-alt perturbation × WT state + 可验证非局部传播可超过简单模型/static proxies | exact-alt physics被three-alt平均；最终候选无传播算子 | 假设未被干净实现和检验 |

### 3.3 项目想超越的现有方法

应该超越三类方法，而不是笼统“超过所有 RNA 模型”：

1. **同信息条件直接基线：** trivial、linear/GAM、tree、P2、容量匹配 generic paired model。
2. **相邻 mutation-effect/riboSNitch 方法：** SNPfold、remuRNA、RNAsnp、Riprap/VariantFoldRNA。这些输出结构扰动或 pair score，不直接输出实验 Δreactivity。
3. **transfer proxies：** RibonanzaNet 的 `F(mutant)-F(WT)` static reactivity 差分，以及 ViennaRNA/EternaFold/RNAformer/eFold/UFold/MoEFold2D 的 static structure差分。

读取 observed mutant trace 的 classSNitch、dStruct、deltaSHAPE 等只能作为 caller/oracle，不是 prospective prediction baseline。

### 3.4 当前“创新”到底是什么

| 候选创新 | 类型 | 当前证据 | 严格判断 |
|---|---|---|---|
| matched WT–mutant probing benchmark及provenance | 数据/资源创新 | 已有4472 pairs和阶段化manifest，但recall/mask/split/exposure未闭合 | 最有现实潜力，但仍是development resource |
| WT experimental anchor | 建模/信息条件创新 | feature存在，无最终endpoint独立消融 | 合理但未证实 |
| exact-alt conditioning | 建模创新 | alt one-hot存在；thermo却平均三个alt且cache缺alt | 概念存在，实现不完整 |
| nonlocal EPRO propagation | 建模创新 | operator代码存在；最终dev10不用；EPRO训练有scheduler/gradient/forcing问题 | 尚未被公平检验，不能称有效贡献 |
| structure/contact features | 工程整合/建模 | position AUPRC约+0.0009，无seeds/pair endpoint | 未证实，不能称机制创新 |
| pair-level caller + conditional magnitude两阶段建模 | 训练/任务设计 | 合同叙事存在，当前代码未闭环 | 是推荐的新核心，但不是当前已完成贡献 |

所以当前项目主要是：**有潜力的数据/benchmark工程 + 多轮建模探索**，不是已经被实验证实的概念或机制创新。

### 3.5 Endpoint crosswalk

| 环节 | 标签/目标 | 单位 | mask | score/metric | 状态 |
|---|---|---|---|---|---|
| 合同主任务 | train-only replicate-aware caller `C_i` | pair | 排除edited/alignment/probe changes | pair probability；publication-macro AUPRC | 目标定义 |
| PH0 caller | max-cluster/noise/FDR caller | pair | 继承不完整primary mask，并可包含mutation site | caller statistic/label | validation 49/548；test已消费 |
| B0 P2 | continuous Δ/WMAE skill | position/profile/pair混合 | 继承D1 mask | skill、错误permutation/curve | scientific PASS失效 |
| dev10 training | `abs(delta)>0.05*scale` | position row | 继承D1 mask | focal loss | 与caller不同 |
| dev10 selection | 同上 | position | 同上 | position AUPRC | 与最终pair metric不同 |
| M0 pair label | eligible positions中>=50%越阈 | pair | 同上 | majority positive 102/548 | 与PH0不同 |
| M0 candidate score | `max(position probability)` | pair | 同上 | legacy AP | 对应pair-any，不对应majority |
| M0 regression score | `max(signed prediction)` | pair | 同上 | 与probability混用 | 文档称max absolute，代码不符 |
| 历史0.7435 | position endpoint | position | 开发口径 | position AUPRC | 不能和0.2652横比 |

### 3.6 重定义后的可投稿研究问题

在保留现有合同主任务的前提下，准确、可验证的论文问题应改为：

> 在 provenance-complete、正确mask、publication-disjoint 且 test untouched 的 matched WT–single-mutant probing benchmark 上，train-only frozen caller定义的 pair response 是否可被跨 publication 预测；若可，exact-alt × WT-state 的 pair-aware模型及受控非局部传播是否带来超过最强同信息条件基线的稳定增益？

这一定义把两个问题拆开：

1. **任务是否可识别/可学？** 由新 benchmark、caller reliability、简单基线和 publication-level P2 gate回答。
2. **EPRO 是否有增量价值？** 只有问题1通过后，由容量匹配、配对seeds和locked confirmatory evaluation回答。

如果团队希望把连续 Δreactivity 改成唯一 primary endpoint，这是合理的资源论文方向，但属于合同/estimand amendment，必须显式新版本化，不能在本轮或下轮静默切换。

## 4. SOTA 距离分析

### 4.1 SOTA 必须绑定到具体任务

截至 2026-08-07 的主来源定向检索，定位到了 mutation-induced structure、riboSNitch、static reactivity 与 observed-profile comparison 方法，但**没有定位到**同时满足本项目输入权限、完整experimental Δ输出、publication-disjoint split、相同mask/aggregation/metric和预训练exposure处理的统一公开 leaderboard。该结论是有边界检索结果，不是“同任务文献不存在”的证明。

因此当前状态为：`SOTA_NOT_ESTABLISHED / UNKNOWN_NOT_ASSERTED`；禁止“first”“世界首个”“领域SOTA”。

### 4.2 三个任务的 SOTA 定义表

| 项目 | 主任务：pair changer | 条件任务：changer magnitude | 次任务：continuous position Δ |
|---|---|---|---|
| 目标任务 | 预测train-only caller定义的`P(C_i=1)` | 对真实changer预测profile-level absolute burden/magnitude | 对所有eligible positions预测signed Δreactivity |
| 主要数据集 | 完成1024 assets disposition后的全部合格exact pairs | 同上，仅在预注册changer条件内 | 同上 |
| 数据划分 | publication-disjoint；内层parent/lineage；confirmatory publications untouched | 同主任务 | 同主任务 |
| 当前划分 | val 548 pairs但publication N=1；test已消费 | 同左 | 同左 |
| 主要指标 | publication-macro AUPRC、Brier/calibration、coverage、ΔAUPRC CI | conditional WMAE skill及CI、rank/correlation sensitivity | noise-weighted MAE/skill、signed correlation、profile calibration |
| 强直接基线 | trivial、logistic/GAM、tree、P2、capacity-matched DeepSets/paired model | mean/linear/tree/P2/generic paired regression | mean/linear/tree/P2/generic paired regression |
| 相邻/transfer | RNAsnp/SNPfold/remuRNA/Riprap；RibonanzaNet/static structure proxies | RibonanzaNet Δ、thermo/structure burden | RibonanzaNet exact-alt Δ最接近；static structure仅proxy |
| 当前结果 | formal candidate 0.2652 < tree 0.2983；标准化排名无法重放 | dev12 WMAE skill为负/最终约负，z-max为事后proxy | 历史开发结果与当前主endpoint不一致 |
| 已知 SOTA | `UNKNOWN_NOT_ASSERTED` | `UNKNOWN_NOT_ASSERTED` | `UNKNOWN_NOT_ASSERTED` |
| 主要差距 | endpoint/独立性/强基线/实际pair output | 目标可靠性和回归skill均未成立 | 数据噪声、mask、signed target和评价未闭合 |
| 泄漏风险 | test PH0消费；validation反复选模；family/homology未知 | 同左 | 同左；预训练RMDB overlap未知 |
| 公平性问题 | candidate probability与baseline signed score混用；tree metadata错误 | 模型输出/尺度/condition需对齐 | probe head、normalization、coverage、预训练数据需对齐 |

### 4.3 当前可确认的 baseline 排名

在仓库 formal pair-majority / legacy study-macro AUPRC 口径下：

| 方法 | 值 | 层级 | 状态 |
|---|---:|---|---|
| zero/train mean/edit-only | 0.1928 | direct trivial | 已运行 |
| mutation-type mean | 0.2066 | direct simple | 已运行 |
| WT-only ridge | 0.2110 | direct learned | 已运行 |
| P2 paired MLP | 0.2592 | direct generic | 已运行 |
| `epro_dev10_best` | 0.2652 | candidate | 已运行；formal FAIL |
| HistGradientBoosting tree | 0.2983 | direct learned | 已运行；当前formal最强 |

由于 legacy AP tie bug，以上具体排序需要统一重算；只有 candidate scores 被本轮安全重放，tree pair scores缺失，所以不能据此给出修正版冠军。

探索性 static proxies 为 RNAformer 0.2263、eFold 0.2106、MoEFold2D 0.2092、UFold 0.2036、Vienna 0.1999、EternaFold 0.1994。它们均为 `DEVELOPMENT_ONLY`，原因包括：任务不相同、实际双序列成功coverage不完整、部分失败可能使用fallback、预训练exposure未知、同一开发publication反复使用、关键结果有untracked证据。它们不能与direct paired主表混为一谈。

### 4.4 必须比较的公开方法与公平边界

| 层级 | 方法 | 原始能力 | 当前状态 | 进入论文的方式 |
|---|---|---|---|---|
| 相邻mutation-effect | [SNPfold](https://doi.org/10.1371/journal.pgen.1001074) | ensemble/global结构变化 | `NOT_RUN` | exact-alt pair score；单列相邻baseline |
| 相邻mutation-effect | [remuRNA](https://doi.org/10.1093/nar/gks1009) | Boltzmann ensemble relative entropy | `NOT_RUN` | 固定温度/序列窗口；不得事后调阈 |
| 相邻mutation-effect | [RNAsnp](https://doi.org/10.1002/humu.22273) | local BPP change/p-value | `NOT_RUN` | 固定mode/window；记录长度和失败 |
| 相邻riboSNitch | [Riprap](https://pmc.ncbi.nlm.nih.gov/articles/PMC7671322/) | WT/mutant BPP的局部扰动score/region | `NOT_RUN` | 固定folding backend；pair-level相邻表 |
| baseline执行器 | [VariantFoldRNA](https://pubmed.ncbi.nlm.nih.gov/40443739/) | 容器化运行上述多种方法 | `NOT_RUN` | pipeline不是独立第五个模型；保留子方法原始输出 |
| static reactivity proxy | [Ribonanza/RibonanzaNet](https://pubmed.ncbi.nlm.nih.gov/38464325/) | sequence→static chemical reactivity | `NOT_RUN` | exact-alt `F(mut)-F(WT)`；probe head与exposure审计 |
| static structure proxy | ViennaRNA、[EternaFold](https://doi.org/10.1038/s41592-022-01605-0)、RNAformer、UFold、eFold、MoEFold2D | sequence→structure/contact/BPP | 部分development运行 | common/full coverage双报告；不能称direct SOTA |
| analysis oracle | [classSNitch](https://pmc.ncbi.nlm.nih.gov/articles/PMC5447233/)、dStruct、deltaSHAPE | 读取WT和mutant observed traces | 不进入预测主表 | 仅caller sensitivity/experimental upper-bound表 |

Riprap论文明确使用WT/mutant序列的预测BPP并输出局部结构扰动；其SHAPE benchmark包含五种RNA、462个单突变序列，但任务与本项目完整experimental Δ/profile不同。VariantFoldRNA则是集成SNPfold、remuRNA、RNAsnp和Riprap的可扩展pipeline。Ribonanza包含约两百万序列的化学mapping数据，RibonanzaNet适合作为强static-transfer proxy，但它不是现成的paired response predictor；其训练暴露必须单独审计。

### 4.5 差距归因

| 候选原因 | 证据 | 当前权重 |
|---|---|---|
| 表征能力不足 | flat MLP只看42个position特征，无显式pair pooling/传播 | 可能，但未能独立识别 |
| 归纳偏置不合理 | exact-alt thermo被平均；position独立；max后处理；mask耦合physics | 已确认实现问题 |
| 损失与目标不一致 | position focal/position AUPRC vs pair-majority/pair AP | 首要已确认问题 |
| 训练策略不足 | position过采样、高量study主导；dev05 LR bug；seed证据不足 | 高优先级已确认问题 |
| 数据规模/质量不足 | 610/1024 parse failures；25 sequences；val publication N=1；ICC异质 | 首要已确认问题 |
| 数据划分简单 | publication伪独立，family/homology未知，test消费 | 首要已确认问题 |
| 推理/解码不足 | 没有pair head；max不匹配majority；signed/absolute混乱 | 高优先级已确认问题 |
| 评估不完整 | AP ties、错误bootstrap/permutation、coverage/exposure缺失 | 首要已确认问题 |
| baseline不公平 | tree metadata、score type混用、公开强基线未跑 | 高优先级已确认问题 |
| 科学问题定义有误 | 问题本身有价值，但仓库执行在多个endpoint间漂移 | 是执行定义错误，不是领域问题无价值 |

结论：**没有证据允许把当前低性能主要归因于“模型容量不够”。** 最合理的因果顺序是先修复 estimand/数据/评价/训练实现，再用预注册、容量匹配实验识别架构上限。

## 5. 发表潜力评分

### 5.1 十项严格评分

| 维度 | 分数 | 评分依据 |
|---|---:|---|
| 1. 科学问题的重要性 | 4/5 | 从static state转向mutation-conditioned experimental response，对变体解释和实验设计重要 |
| 2. 现有方法的明确局限 | 3/5 | 文献上确有static structure、riboSNitch、static reactivity和observed-profile oracle之间的任务缺口；仓库尚未统一实证 |
| 3. 核心方法的新颖性 | 2/5 | WT anchor/exact-alt/nonlocal传播有合理概念，但最终主体仍是flat MLP，exact-alt physics不完整，传播未被公平评估 |
| 4. 跨任务或跨领域意义 | 2/5 | perturbation-response框架可推广，但无跨family/probe/platform或其他perturbation证据 |
| 5. 实验设计完整性 | 1/5 | 单publication validation反复选模；test消费；直接公开强基线、locked test、多seed、主endpoint消融均缺 |
| 6. 结果可信度 | 2/5 | formal FAIL报告诚实且legacy candidate可重放；但endpoint漂移与evaluator bug阻断正/负科学解释 |
| 7. 机制解释或科学发现 | 1/5 | 没有稳定方法增益，也没有证明传播、结构或WT-state机制 |
| 8. 泛化能力 | 1/5 | validation publication N=1，family/homology独立未知，test失效 |
| 9. 可复现性 | 2/5 | 有大量manifest/hash/terminal；但authority断链、关键prediction/source未完全tracked、active终态过期 |
| 10. 论文故事完整性 | 2/5 | 痛点和资源切入点存在；方法→能力、实验→主张、结果→广泛价值断裂 |
| **总分** | **20/50** | **阶段 A** |

### 5.2 三条投稿路线

| 路线 | 当前阶段 | 进入下一阶段的必要条件 | 当前可用措辞 |
|---|---|---|---|
| 顶级交叉领域 | A，暂无基础 | 方法confirmatory gain + prospective/wet-lab + 跨family/probe/platform + 新机制/实际效用 | 不应准备此级别主张 |
| 强领域方法 | A | 新untouched publications；强基线全覆盖；5 seeds；ΔAUPRC CI下界>0；valid p<0.05；conditional WMAE skill>0；机制消融 | “当前gate未证明优越性” |
| benchmark/resource/negative-result | A，最接近早期B潜力 | 1024 asset闭环；正确mask/caller/split；直接/相邻/proxy/oracle分层；跨publication稳定发现；clean replay | “已建立development benchmark，尚待publication-grade重建” |

### 5.3 特别判断

- **当前是否只是“加模块并提高指标”？** 是，至少最终方法证据层面是。dev04/dev06/dev10主要是在手工features与flat MLP上迭代；结构增量仅约0.0009 position AUPRC，未形成最终endpoint机制证据。
- **是否有清晰核心发现？** 没有已确认的正发现。可以发展成的负发现是“static proxies和常规supervised模型跨publication失效”，但现在只有一个validation publication，尚不能成立。
- **能否回答为什么有效？** 不能。正式候选还未胜tree；EPRO operator也没有被有效训练和公平消融。
- **能否产生单一benchmark之外的新认识？** 当前不能。需要证明WT state、exact-alt与非局部传播的可迁移机制，或给出跨实验域的可识别性边界。
- **即使未来达到某个benchmark SOTA，是否仍缺广泛价值？** 是。单一小样本benchmark的小幅AUPRC提升不足以支撑广泛生命科学价值；仍需独立实验域、机制/失败规律或实验选择效用。

## 6. 模型架构问题

### 6.1 最终候选的实际数据流

最终 candidate 不是合同叙述的完整 EPRO operator。实际数据流为：

`canonical pair`  
→ D1/B0 delta与finite mask  
→ 将每个pair展平为多个position rows  
→ B0基础features 31维 + thermo 5维 + contact/BPP 6维 = 42维  
→ flat MLP（约142,849参数）  
→ 每个position的changer probability  
→ position focal loss训练；position validation AUPRC选checkpoint  
→ validation上对每个pair取最大position probability  
→ 与pair-majority label计算legacy study-macro AP。

核心路径：

- `scripts/reactflow_delta/m0x_epro_dev06.py:94-207,225-253`
- `scripts/reactflow_delta/b0x_baselines.py:45-92`
- `scripts/reactflow_delta/m0x_epro_hpsearch.py:126-166`
- dev10 run manifest：约471,344 position training rows，candidate manifest SHA-256 `d116b629…`

这意味着：

1. 每个position被当作近似独立训练样本；长序列、共享WT block和高量study被过度加权。
2. 模型没有直接输出pair-level probability，也没有学习与majority定义一致的pooling。
3. “约47万样本”是相关position rows，不是47万个独立生物样本。
4. formal candidate性能不能归因于EPRO传播机制。

### 6.2 EPRO operator代码中的实际数据流

真正EPRO代码存在于 `reactflow_delta/model.py`，大致为：

input/thermo encoding  
→ `ThermoEncoder`  
→ `Forcing` 将mutation/WT信息形成position forcing  
→ susceptibility/contact operator  
→ dense Neumann-style nonlocal propagation  
→ switch/observation heads  
→ position response。

路径包括：`ThermoEncoder:117-167`、`Forcing:175-307`、susceptibility `315-451`、switch `459-510`、observation `518-579`、forward `588-724`。概念上它试图引入“局部扰动经结构联系传播”的归纳偏置，但当前有以下直接实现障碍。

### 6.3 输入表示、交互、输出和约束审查

| 环节 | 当前实现 | 必要性/问题 | 裁决 |
|---|---|---|---|
| WT sequence与local context | B0手工features/编码 | 必要；是最基本信息 | 必要，但需与WT-reactivity消融 |
| exact ref/alt | 有one-hot/identity特征 | 必要 | 实现部分成立 |
| thermo delta | 对三个非ref alt平均；cache key缺alt | 与核心exact-alt假设冲突 | 明确实现偏差 |
| WT reactivity | 作为anchor/features | 可能是核心增益来源 | 合理但未证实；需sequence-only对照 |
| condition/probe/platform | 条件表达有限 | 跨实验域必需 | 当前能力不足或证据缺失 |
| contact/BPP | 6维summary或dense operator | 可支持nonlocal传播 | dev06 indexing bug，增量未证实 |
| position flattening | 一pair拆多行 | 工程简单，但违背pair统计单位 | 限制上限并造成权重偏差 |
| cross-position interaction | 最终dev10无；EPRO有dense propagation | 是核心假设能力 | 最终候选缺失；EPRO未被有效评估 |
| Forcing | 每position标量后广播latent channels | rank-1通道瓶颈 | 可能限制表达能力 |
| susceptibility init | edge MLP全零线性权重/bias | 可能造成早期层零梯度/恒定修正 | 明确优化风险 |
| mask | target eligibility同时作physics forcing mask | 可能泄漏目标缺失模式 | 信息条件不干净 |
| missing | `nan_to_num`静默变0 | missing≠zero | 明确治理冲突 |
| pair output head | 无，靠max后处理 | 无法对齐majority/caller probability | 明确缺失 |
| loss | 显式sigmoid+clamp+log focal | 数值稳定性弱于logits实现 | 合理目标未必必要；需稳定实现 |
| constraints | EPRO operator/switch | 可能提供物理归纳偏置 | identity/antisymmetry、residual和exact-alt未完整验证 |

### 6.4 训练与推理差异及优化问题

1. `[CONFIRMED_FACT]` **dev05 scheduler bug。** `m0x_epro_dev05.py:213-228` 的 `lr_at(step)`返回绝对LR，却作为 `LambdaLR` multiplier；base LR约`1e-3`时，实际可变成约`1e-6`，warmup可低至约`5e-9`。因此dev05接近随机的结果不能证伪EPRO概念。
2. `[CONFIRMED_FACT]` **梯度检查不充分。** susceptibility edge MLP全部线性层零初始化；`o0x_run.py:161-192`只检查有限顶层梯度，没有证明每层梯度/参数更新和operator residual正常。
3. `[CONFIRMED_FACT]` **结构summary indexing bug。** `m0x_epro_dev06.py:166` 的 `row[row != i]`比较BPP数值与整数索引，而非排除对角元素。
4. `[CONFIRMED_FACT]` **selection mismatch。** 训练/早停看position AUPRC，正式看pair-majority AP；训练期没有优化最终metric的稳定surrogate。
5. `[REASONED_INFERENCE]` **study/length imbalance。** position-row sampling会让ADD140和长序列贡献更多梯度；若没有pair/publication-balanced sampler，模型容易学域特征。
6. `[CONFIRMED_FACT]` **dev12负结果。** regression WMAE skill为负，raw burden Spearman为负；后续z-max正相关来自事后proxy/sign选择，不能作为magnitude能力。

### 6.5 时间、空间和长序列扩展性

- Flat MLP部分约为 `O(N_positions × d × hidden)`，计算便宜，但position独立，缺非局部能力。
- BPP/contact预计算和dense EPRO operator涉及 `O(L²)`存储/计算；Neumann迭代近似 `O(T·L²·d)`。
- 现有smoke主要覆盖约`L<=200`的开发设置，没有长序列长度扫描、GPU/CPU峰值内存、每样本residual convergence和数值失败率。
- dense operator若直接扩大L，可能导致内存平方增长、迭代收敛不稳定和梯度爆炸/消失。
- 推荐先用稀疏top-k contact或低秩operator做最小验证，并把residual、谱半径/收敛诊断、NaN/Inf和梯度范数写入日志；在这些通过前不能声称长序列可扩展。

### 6.6 必要、冗余、未证实与上限限制

| 分类 | 设计 |
|---|---|
| 真正必要 | exact ref/alt/position；WT sequence；明确条件；合法WT reactivity；pair identity；mask与missingness；direct pair output；publication-aware sampling/evaluation |
| 可能冗余 | 在数据/endpoint未成立前叠加多个结构模型summary；多个高度相关thermo/contact proxy；事后z-max/sign组合 |
| 合理但未证实 | WT anchor；nonlocal propagation；structure/contact；noise weighting；switch mechanism；conditional two-stage head |
| 明确限制上限 | three-alt thermo平均；cache缺actual alt；position flattening；max aggregation；rank-1 forcing；零初始化edge MLP；condition interaction不足；mask耦合physics |

### 6.7 是否继续局部优化

**不应在当前 dev10/M0 evaluator 上继续局部调参或增加模块。** 这并不等于立刻废弃全部EPRO代码；正确顺序是：

1. 新版本benchmark/evaluator闭合；
2. direct pair-aware generic model建立可信地板；
3. 修复后的EPRO作为一个预注册、容量匹配的增量假设被检验；
4. 若两轮仍不胜generic/tree，则退役EPRO方法主张。

最高优先级的三个架构方案在第9节给出完整最小实现、对照、消融、成功标准和失败处理。

## 7. 最不确定的三个问题

### 7.1 不确定性一：修正 protocol 后，实验 Δreactivity 是否具有可跨 publication 学习的信号？

**这是最可能推翻整个原方法项目的不确定性。**

| 项目 | 回答 |
|---|---|
| 为什么仓库无法确认 | B0 permutation和learning curve实现无效；primary mask不合格；PH0 caller与M0标签不同；validation publication N=1；test已消费。因此既有P2 PASS、M0 FAIL都不能回答“跨publication可学性”。 |
| 对SOTA/论文的影响 | 若信号不存在或只在publication内成立，任何更复杂架构只能拟合study-specific偏差；EPRO SOTA主张失去科学基础。若信号存在，才值得比较归纳偏置。 |
| 需要什么证据 | 新版parser/mask；train-only caller及reliability；至少多个publication outer folds；简单模型学习曲线；label-permutation和negative controls；publication-balanced uncertainty。 |
| 最小验证实验 | 在Phase 1闭合数据后，运行nested leave-one-publication-out：只用trivial、linear/GAM、tree、P2和一个capacity-matched generic paired model；每个outer fold内冻结caller/特征/阈值；不接触confirmatory set。 |
| 正面结果 | 至少多个held-out publications上，相对强简单基线的方向一致；publication-level CI/有效permutation支持skill；学习曲线随独立publication增加而改善。 |
| 负面结果 | skill不超过permutation/simple baseline；方向由单publication主导；caller reliability低于预设阈值；加入publication后性能不升反降。 |
| 负面后如何调整 | 终止EPRO SOTA架构路线；转向benchmark/resource、测量可识别性、domain-shift或严格负结果论文；如可能，引入新实验而不是继续position-level调参。 |

最低成功门槛不应是“pooled AUPRC比prevalence高”，而应是：在合法outer folds中，相对最强同信息条件简单模型有可重复的正skill，且结果不由某个大study或单publication决定。

### 7.2 不确定性二：610 个失败 assets 能否恢复足够 exact pairs 和独立 publications？

| 项目 | 回答 |
|---|---|
| 为什么仓库无法确认 | D0只记录失败类别，没有逐asset最终裁决为可恢复/不可恢复/非目标数据/重复/许可受限；也没有说明恢复后会增加多少publication和unique lineage。 |
| 对SOTA/论文的影响 | 新独立publication是confirmatory test和跨域泛化的必要条件。只增加共享parent的pair/position不会提高最高层有效N。 |
| 需要什么证据 | 1024-row controlled-disposition ledger；parser rescue规则；每个恢复asset的source/profile/pair/publication/condition映射；人工抽样QA；许可和provenance closure。 |
| 最小验证实验 | 不直接全量重跑训练。先按失败类别分层抽样，每类开发最小strict parser/rescue fixture；估计可恢复率及新增publication yield，再决定是否全量实现。 |
| 正面结果 | 恢复多个此前未使用的独立publications，且exact matching、replicate、condition和probe metadata足以建立outer folds/untouched test。 |
| 负面结果 | 大部分失败是不可重建、非matched或仍集中在现有publication；有效publication N没有实质增加。 |
| 负面后如何调整 | 不再以公开数据支撑强监督SOTA；转向resource quality/coverage paper、与实验组建立prospective cohort，或缩窄到特定family/probe但明确外推边界。 |

这里的关键评价量不是“多恢复了多少行”，而是：**增加了多少合法 exact pairs、unique sequence/lineage、尤其是独立 publications。**

### 7.3 不确定性三：公平 exact-alt 条件下，pair-aware 归纳偏置是否有超过 tree/generic model 的增量？

| 项目 | 回答 |
|---|---|
| 为什么仓库无法确认 | final candidate不是EPRO；thermo平均三个alt；dev05 LR错误；edge MLP初始化/梯度有风险；structure增量无seeds；当前评价目标不一致。EPRO既没有被支持，也没有被有效证伪。 |
| 对SOTA/论文的影响 | 如果修复后仍不胜generic/tree，方法新颖性只剩工程整合；即使benchmark有价值，也不能把EPRO作为论文主贡献。 |
| 需要什么证据 | exact-alt cache；pair-balanced sampler；direct pair head；generic DeepSets/attention对照；固定容量和算力；至少5 seeds；component paired ablation；publication outer folds。 |
| 最小验证实验 | 只在Phase 2 learnability PASS后，比较：tree、P2、DeepSets pair model、修复后EPRO；相同输入、预算、selection rule；一次只增加一个能力。 |
| 正面结果 | EPRO相对capacity-matched generic在多个outer publications方向一致，paired CI下界>0；exact-alt/nonlocal ablation按预注册方向下降。 |
| 负面结果 | 只在一个study赢、seed variance大于增益、去掉propagation无影响、或generic model同样好/更好。 |
| 负面后如何调整 | 退役EPRO主张，保留更简单pair-aware模型；论文转benchmark/resource或“简单模型已足够”的科学发现。 |

### 7.4 哪一个最可能推翻整个项目

**不确定性一最可能推翻整个原方法项目：在合法 mask、train-only caller 和 publication-disjoint 协议下，目标信号是否可跨 publication 学习。**

原因是它位于因果链最上游。610 assets能否恢复属于数据可行性；EPRO能否胜generic属于方法选择。唯有“任务本身在允许输入下是否具备跨publication可预测信息”决定是否存在监督方法论文的对象。若答案为负，模型复杂度、传播operator和更多调参都无法挽救原主张。

## 8. 最大盲区

### 8.1 盲区定义

**当前最大盲区是：项目实际上没有在一个冻结、独立、语义一致的科学 endpoint 上完成开发和评价。**

它不是单个 metric bug，而是以下链条同时漂移：

`eligible position mask`  
→ `caller/label definition`  
→ `training unit and loss`  
→ `checkpoint selection`  
→ `pair aggregation`  
→ `statistical unit`  
→ `test exposure`。

### 8.2 哪些仓库现象表明它存在

1. PH0 caller给validation 49/548 changers，M0 majority给102/548。
2. D1 mask未排除edited site、alignment或probe-eligibility change；mutation site还能进入PH0 winning cluster。
3. dev10训练/早停使用position labels和position AUPRC；M0最终评价pair-majority。
4. pair-majority score却取单一position maximum；synthetic counterexample显示单热点会压过真正多数响应。
5. candidate是probability，tree/regression是signed Δ，gate用相同AP比较，却没有共同pair probability semantics。
6. `_average_precision`对ties行顺序敏感；同一冻结candidate仅换行序就产生大幅AP变化。
7. CIDGMP/TRP4P6同一publication，却被study-macro/bootstrap包装成多个验证域。
8. PH0读取test outcomes，但ledger仍写sealed/no events。
9. 历史0.7435、formal 0.2652、dev12 z-max属于不同单位/endpoint/proxy，却容易被放进同一“性能轨迹”。

### 8.3 为什么团队容易忽视

- **pair和position数量很大。** 4472 pairs与约47万position rows给人“大数据”错觉，掩盖publication N极小。
- **每个阶段都有manifest/PASS。** 工程closure、hash和terminal容易被误读为科学estimand也已闭合。
- **多个endpoint看起来都叫mutation effect。** `any`、`majority`、caller、absolute Δ、signed Δ、z-max都有关联，但不是可互换统计对象。
- **指标看起来熟悉。** AUPRC、bootstrap、permutation的名称正确，不意味着实现、重采样单位和交换性成立。
- **模型迭代提供即时反馈。** position AUPRC可快速升降，促使团队继续加features，而不是停下来验证最终pair question。

### 8.4 如何造成虚假提升或错误论文结论

| 机制 | 可能的虚假结论 |
|---|---|
| 在同一publication反复选择模型/聚合 | 把开发集适配写成跨研究泛化 |
| 从many/any/majority/max/z-max中事后选择 | 把aggregation自由度变成“模型能力” |
| ties依赖行序 | 相同预测仅因记录排序产生不同AUPRC |
| position flattening | 把共享WT/study的相关行当独立样本，低估不确定性 |
| edited-site/missingness进入目标/特征 | 模型学到实验或标签生成shortcut而非非局部response |
| test先被caller消费 | 将已参与标签统计的域误当一次性确认集 |
| candidate probability与signed baseline混比 | 排名来自score semantics而非预测质量 |
| publication N=1却做cluster CI | 给出看似精确、实际无跨域含义的置信区间 |

### 8.5 最小成本验证

在不训练任何新模型的最小验证层面，本轮已经完成三项 falsification：

1. **endpoint crosswalk：** 明确49/548与102/548来自不同定义。
2. **AP ties：** 冻结candidate的100次行重排令legacy pooled AP变动，标准AP不变。
3. **aggregation语义：** 单热点和广泛响应synthetic example证明max与majority不一致；signed example证明max signed与absolute task不一致。

要真正消除盲区，下一版本最小实施为：

- 冻结一份pair-level endpoint specification和primary mask；
- caller只在训练fold拟合/冻结；
- 模型直接输出pair probability；
- selection metric与final metric一致；
- publication为outer unit；
- 新confirmatory publications只在final lock后访问一次；
- evaluator用标准ties实现和预注册resampling；
- prediction artifact完整绑定data/split/code/model/hash。

### 8.6 四个层面的解决方案

| 层面 | 必须动作 |
|---|---|
| 模型 | direct pair head；pair-balanced sampling；actual-alt条件；target mask与physics/input mask分离；不以max后处理伪造pair output |
| 数据 | 1024资产全 disposition；新mask；统一parent/lineage/publication定义；退役现test；获得新untouched publications |
| 实验 | nested publication split；frozen caller/evaluator；5 seeds；common/full coverage；有效permutation；confirmatory一次访问 |
| 论文 | 主张绑定evidence ID；明确development vs confirmatory；direct/adjacent/proxy/oracle分表；禁用历史过期SOTA和“first” |

### 8.7 什么结果证明盲区已消除

必须同时满足：

1. 一个版本化 endpoint ID 对应唯一 mask、caller、unit、score、metric和resampling spec；
2. 对每个outer publication，caller只由允许的训练域生成，不读取outer outcomes；
3. pair label prevalence、模型pair output和评价score语义完全一致；
4. 标准AP对行顺序不变；constant-label/mixed-block边界返回显式`UNIDENTIFIABLE`；
5. parent、study、publication、lineage定义跨dataset、sampler、bootstrap一致；
6. 至少3个从未参与开发的confirmatory publications或等价prospective cohort；
7. test access ledger与实际I/O日志一致，final access仅一次；
8. clean checkout可由hash-bound predictions重放所有主表数字。

### 8.8 次要盲区（最多三个）

1. **数据recall盲区：** 610个失败assets尚未定性，不能把当前负结果写成公开数据或任务上限。
2. **“EPRO已被测试”盲区：** final candidate不是EPRO，真实EPRO又受LR、初始化、forcing和mask问题影响；当前既不能支持也不能否定EPRO。
3. **baseline/exposure盲区：** 强公开方法未运行，static proxy覆盖和预训练污染未知，无法建立公平SOTA表。

## 9. 推荐解决方案

### 9.1 总体优先级

| 优先级 | 动作 | 原因 | 是否本轮执行 |
|---|---|---|---|
| P0 | 关闭authority/exposure终态，建立新版本benchmark/evaluator | 使科学对象、权限和独立性可识别 | 否；需要新授权 |
| P0 | 用简单模型做publication-level learnability gate | 判断是否值得进行架构研究 | 否；需要新授权与新数据版本 |
| P1 | 最多三个架构方向，容量匹配、一次只变一种能力 | 识别真正的归纳偏置增益 | 否；仅在P2 gate通过后 |
| P1 | 完整公开baseline和locked confirmatory evaluation | 建立SOTA或负结果证据链 | 否；development winner冻结后 |
| P2 | 机制/wet-lab/prospective验证 | 支撑高水平广泛价值 | 否；需要前述可信结果 |

### 9.2 架构方案一：直接 pair-level changer head + conditional profile head

| 字段 | 方案 |
|---|---|
| 要解决的瓶颈 | 当前逐position训练、position早停和max聚合不对应pair caller/majority endpoint；长序列和大study过度加权。 |
| 架构修改 | 对每个pair先编码position features，再用固定简单pooling或DeepSets得到pair representation；直接输出`P(C_i=1)`；第二个head只在训练fold定义的changer条件下输出profile magnitude；continuous Δ作为预注册辅助head。 |
| 理论/经验依据 | 统计单位与预测单位一致；permutation-invariant pooling适配可变长profile；两阶段模型把发生概率与幅度分开，避免零膨胀/噪声混合。 |
| 新增能力 | 学习pair evidence aggregation；校准pair probability；避免用事后max代替模型；可做pair/publication-balanced sampling。 |
| 预计收益 | 首先提升可解释性和评价对齐；数值收益未知，不能预先承诺。即使不涨分，也能给出可信负结果。 |
| 主要风险 | pooling过强可能丢局部热点；changer标签可靠性不足；conditional head样本更少。 |
| 最小实现 | B0/42维features不变；2层position encoder + sum/mean/max三种预注册pooling（不选最优作主结果，默认mean+sum统计）+ logistic pair head；magnitude用masked weighted regression head。 |
| 对照实验 | tree、P2、当前max后处理、mean/fraction surrogates、capacity-matched DeepSets；相同input、split、budget。 |
| 消融 | 无WT anchor；无exact-alt；position-balanced vs pair-balanced；单head vs two-head；mean vs learned pooling。 |
| 成功标准 | nested held-out publications中，相对最强direct baseline方向一致；5 seeds paired CI下界>0；Brier/calibration改善；不依赖单study。 |
| 失败处理 | 若不胜tree/generic，停止复杂化，保留benchmark/resource路线；分析caller reliability和domain shift，而非再搜索pooling。 |

### 9.3 架构方案二：exact-alt WT/mutant interaction 的简洁 pair-aware generic model

| 字段 | 方案 |
|---|---|
| 要解决的瓶颈 | 当前thermo对三alt平均、cache缺alt；WT state与exact mutation交互弱；condition表达不足。 |
| 架构修改 | 为WT与actual mutant分别构造sequence/thermo/static representations；使用`[WT, Mut, Mut-WT, WT×(Mut-WT), condition]`显式交互；cache key至少含`(parent,pos,ref,alt,condition,feature_version)`；target mask和input/physics mask完全分离。 |
| 理论/经验依据 | 突变效应是条件反应而非单一状态；difference与interaction显式编码降低模型自行推导exact perturbation的样本复杂度。 |
| 新增能力 | 区分同一ref/position的三个actual alt；建模WT experimental state调制；明确probe/platform/condition effect。 |
| 预计收益 | 若核心假设正确，应在exact-alt ablation和跨publication上产生稳定增量；收益大小未知。 |
| 主要风险 | static feature provider有预训练exposure/coverage；interaction维度增加；small publication N下过拟合。 |
| 最小实现 | 不用Transformer；共享小MLP/1D encoder + DeepSets pool；仅使用版本化允许features；所有外部feature失败保持missing indicator，不填零。 |
| 对照实验 | ref-only、mutation-type mean、WT-only ridge、tree、P2、同参数量generic concat model、RibonanzaNet exact-alt proxy。 |
| 消融 | actual-alt vs three-alt average；WT experimental anchor vs sequence-only；difference vs concat；condition on/off；thermo/contact on/off。 |
| 成功标准 | exact-alt组件在至少2/3 confirmatory publications方向有利；paired publication-level CI支持增量；无coverage/fallback驱动。 |
| 失败处理 | 若exact-alt/WT interaction无增益，删除这些主张，采用最简单稳定model；重点转向数据reliability或资源论文。 |

### 9.4 架构方案三：修复并严格证伪 EPRO propagation operator

| 字段 | 方案 |
|---|---|
| 要解决的瓶颈 | 非局部传播是最有潜在新颖性的组件，但当前没有被有效训练或作为final candidate公平评估。 |
| 架构修改 | 修复LambdaLR；non-degenerate edge initialization；逐层gradient/update测试；vector-valued forcing；actual-alt thermo/cache；独立physics/input/target masks；稀疏top-k或低秩contact operator；每样本residual/收敛日志；switch单独开关。 |
| 理论/经验依据 | 如果mutation response沿base-pair/contact网络传播，受控operator可提供generic pooling没有的结构归纳偏置；但必须由最终endpoint消融证明。 |
| 新增能力 | 可解释传播路径、长程作用与结构context；更接近项目原核心方法叙事。 |
| 预计收益 | 完全未知；当前证据不能提供可信数值预期。价值在于得到一次公平、可证伪的检验。 |
| 主要风险 | `O(TL²d)`计算/内存；operator不稳定；小数据下高方差；结构proxy噪声；修复后仍不胜generic。 |
| 最小实现 | 只在短/中长度development folds；单一sparse contact operator、vector forcing、无复杂switch；与DeepSets匹配参数/训练时间；先过synthetic identity/antisymmetry/gradient/residual tests。 |
| 对照实验 | repaired EPRO vs方案二generic model；local-only operator；random/permuted contacts；no-propagation identity；同参数/同算力。 |
| 消融 | exact-alt、forcing rank、contact、propagation depth、switch、WT anchor、noise weighting逐项配对。 |
| 成功标准 | 相对capacity-matched generic在5 seeds及多个held-out publications有CI下界>0；random contacts无同等增益；传播指标与预注册机制一致；无数值失败。 |
| 失败处理 | 两轮预注册核心迭代仍不胜generic，则退役EPRO方法主张和operator复杂度；不得继续无限dev版本搜索。 |

### 9.5 哪些当前设计应保留或退役

- **保留但重新版本化：** exact pair provenance、WT anchor候选、noise/reliability元数据、结构feature provider接口、phase manifests和hash习惯。
- **立即退役为legacy：** 当前M0 evaluator、pair-majority+max组合、row-order-sensitive AP、现有16SFWJ confirmatory角色、D0-era sealed ledger、历史0.7435 SOTA叙事。
- **仅作为探索性：** dev12 z-max、static proxy开发排名、未闭合epoch13结果、untracked dev11/dev12/SOTA artifacts。
- **待公平证伪：** EPRO propagation、structure/contact增量、two-stage response head。

## 10. 重构后的论文故事

### 10.1 当前故事在哪里断裂

当前隐含故事是：

> mutation response重要 → static方法不足 → EPRO传播可建模 → 模块增加提高position指标 → 模型具备SOTA潜力。

断裂点有三处：

1. **现有方法局限 → EPRO必要性：** 公开强mutation-effect和RibonanzaNet基线未运行，尚未证明现有方法在同协议下的具体失败。
2. **EPRO设计 → 新能力：** final candidate不是EPRO；exact-alt被平均；传播组件无最终endpoint多seed消融。
3. **开发结果 → SOTA/广泛价值：** formal candidate未胜tree；validation publication N=1；test消费；无机制或prospective utility。

### 10.2 可成立的benchmark-first逻辑链

领域痛点  
→ experimental mutation-response数据稀少、跨study异质，static structure/reactivity并不等于paired response  
→ 现有研究缺少provenance-complete、publication-disjoint、noise-aware的统一直接benchmark  
→ 本项目先构建matched WT–single-mutant probing benchmark，并审计mask、caller、exposure与effective N  
→ 用direct/simple、adjacent mutation-effect、static reactivity、static structure和oracle五层方法进行公平评估  
→ 先回答任务在哪些publication/family/probe条件下可学或不可学  
→ 若可学，再验证exact-alt × WT-state与nonlocal propagation是否提供稳定增量  
→ 获得新能力：跨publication预测，或新认识：static-state proxies为何及何时无法转移到perturbation response  
→ 为RNA变体解释、实验优先级选择和perturbation-response ML提供可复现基准与机制边界。

这条故事允许两种诚实结局：

- **正方法结局：** pair-aware exact-alt模型在独立publications稳定胜最强同信息条件基线，并由消融支持机制。
- **资源/负结果结局：** 多类强模型在严谨split下均失效，失效与caller reliability、publication shift或static-response mismatch有可重复关系。

### 10.3 建议的论文核心主张

**核心主张（当前仅为目标，不是已获证据）：**

> 我们建立了一个provenance-complete、noise-aware、publication-disjoint的matched WT–single-mutant experimental reactivity benchmark，并用预注册协议定量界定了RNA mutation response的跨publication可预测性。

| 所需证据 | 当前已有 | 当前缺失 |
|---|---|---|
| 1024 assets controlled disposition；exact pair/provenance/condition闭包 | frozen资产和4472 development pairs | 610失败项裁决、正确mask、family/lineage/许可QA |
| publication-disjoint development/confirmatory split | 8 publications aggregate存在 | 新untouched confirmatory publications；现test退役 |
| train-only caller/reliability | PH0 caller代码与ICC信息 | 合法mask、fold-local caller、null/seed/window修复 |
| 五层baseline矩阵 | internal baseline和部分static proxy | 相邻强方法、RibonanzaNet、公平coverage/exposure |
| 预注册evaluator和clean replay | manifests/hash习惯、本轮bug复现 | 新版标准AP、hierarchical stats、完整prediction closure |

### 10.4 最多三个次级主张

| 次级主张 | 需要的证据 | 当前证据 | 缺失证据 |
|---|---|---|---|
| 1. static structure/reactivity差分不能充分替代paired experimental response | 多个held-out publications；公平exact-alt proxies；common/full coverage；exposure分层；失败regime分析 | development static proxies较弱 | 目前单publication、多覆盖问题、RibonanzaNet未跑 |
| 2. WT experimental state × exact-alt interaction带来可迁移增量 | sequence-only/WT-only/three-alt/exact-alt配对消融；5 seeds；publication CI | WT/alt features存在 | thermo非exact-alt；无最终endpoint消融 |
| 3. 受控nonlocal propagation解释长程response | repaired EPRO；random/local/no-contact对照；传播距离/结构边机制；外部复现 | operator代码和概念 | final candidate不用；训练bug；无可信增益/机制 |

### 10.5 必须避免的过度表述

- 不能把 formal `p=1.0` 写成“RNA mutation response不可预测”。
- 不能把610个D0 parse failures写成“公开数据中没有更多样本”。
- 不能把position-level 0.7435与pair-level 0.2652写成同一性能演进。
- 不能把candidate-only tie-aware 0.3008与旧tree 0.2983比较后宣布胜出。
- 不能把static proxy的开发集排名写成“超过已发表SOTA”。
- 不能把classSNitch/dStruct等读取mutant trace的方法放入prospective prediction主表。
- 不能把dev12 z-max/sign结果写成magnitude任务成功或机制发现。
- 不能声称structure/contact有效，除非最终endpoint、matched budget、paired seeds消融通过。
- 不能声称跨study/跨publication泛化；当前validation最高层只有一个publication。
- 不能使用“first/世界首个/领域SOTA”，除非以后完成系统检索和同任务locked比较。

## 11. Final Goal

### Final Goal

- **核心科学问题：** 在不读取 mutant 实验 profile 的条件下，能否由 WT sequence、exact single-nucleotide mutation、实验条件和允许的 WT reactivity，跨 publication 预测 mutation-induced experimental reactivity response？
- **核心假设：** exact-alt perturbation 与 WT experimental state 的交互，加上可验证的非局部传播归纳偏置，能提供超出简单统计模型、static structure差分和static reactivity差分的增量信息。
- **核心方法贡献：** 首先建立 publication-disjoint、provenance-complete、noise-aware benchmark；只有 publication-level P2 learnability 通过后，才贡献 pair-aware、exact-alt、两阶段 response model，并把 EPRO propagation 作为可独立证伪的增量组件。
- **目标任务：** ① train-only frozen replicate-aware caller定义的 pair-level changer probability；② changer条件下的profile-level Δreactivity magnitude；③全部eligible positions的连续noise-weighted signed Δreactivity，作为预注册次级任务。
- **主要数据集：** 完成1024 assets controlled disposition后获得的全部合格 exact WT–single-mutant pairs；当前 CIDGMP/TRP4P6/16SFWJ 降为 development/exposed evidence；另建真正 untouched 的 publication-level confirmatory set。
- **关键评价指标：** publication-macro AUPRC；相对最强同信息条件baseline的ΔAUPRC及publication-cluster 95% CI；有效group-aware permutation；conditional WMAE skill及CI；Brier/calibration；coverage；跨publication方向一致性。
- **必须超过的基线：** strongest trivial、linear/logistic/GAM、tree、P2、容量匹配generic paired model，以及可公平适配的SNPfold、remuRNA、RNAsnp、Riprap/VariantFoldRNA、RibonanzaNet exact-alt static proxy和static-structure proxies。
- **必须完成的泛化验证：** publication、parent/lineage、RNA family、probe/platform和condition分层；至少3个从未参与开发的独立confirmatory publications，或一个等价的prospective external cohort；confirmatory set只在模型与协议冻结后访问一次。
- **必须完成的机制或解释性分析：** exact-alt对three-alt marginalization、WT anchor对sequence-only、local对nonlocal propagation、真实对随机contacts、structure/contact features、noise/reliability weighting的paired ablation；结果必须跨publication方向一致。
- **论文级最终交付物：** 版本化benchmark；data/split/caller/exposure manifests；全baseline matrix；预注册evaluator；模型、消融和机制结果；common/full coverage；clean-checkout replay；主表/图；代码/权重/artifact checksums；claim–evidence map；Data Availability与限制。
- **项目成功标准：** 数据、authority、exposure和独立性gate全部通过；至少5 seeds、固定预算和预注册selection下，candidate相对最强同信息条件baseline的publication-level 95% CI下界>0；有效permutation `p<0.05`；conditional magnitude WMAE skill CI下界>0；pooled、publication-macro和至少2/3 confirmatory publications方向一致；核心消融支持预先声明机制，而不是靠事后proxy/后处理翻转。
- **项目终止或转向条件：** 无法获得新的untouched publications；train-only caller/reliability/mask修复后主endpoint仍不可识别；publication-level P2 learnability不能胜permutation与简单baseline；两轮预注册、容量匹配的架构迭代仍不能超过最强tree/generic baseline；外部强baseline或预训练exposure无法公平裁决。触发后终止“EPRO SOTA方法”主张，转向benchmark/resource、测量可识别性、domain-shift或严格负结果路线。

## 12. 分阶段 TODO

以下 Phase 0 是本轮只读审查；Phase 1–6 的任何代码、数据、合同、训练或test动作都**尚未授权**，必须建立新authority epoch、版本号、哈希与停止边界后执行。不得在旧artifact上原地覆盖。

### Phase 0：仓库、数据和实验真实性审计

| 字段 | 内容 |
|---|---|
| 阶段目标 | 回答当前仓库真正实现、运行和证明了什么；裁决authority、exposure、endpoint、数据、模型、统计与SOTA证据链。 |
| 任务 | 冻结HEAD/Git/hash；建立authority lineage；静态追踪test访问；对齐caller/train/gate endpoints；审计1024 assets、split/effective N、mask/noise；追踪actual model data flow；无拟合重放validation scores；分层检索公开baseline；生成报告/ledger/replay。 |
| 涉及模块 | `configs/reactflow_delta/active_contract.yaml`；`docs/contracts/**`；`data_registry/**`；`scripts/reactflow_delta/{d1x_canonicalize,d2x_split,ph0x_*,b0x_*,m0x_*}.py`；`reactflow_delta/model.py`；`results/**`；`/mnt/.../d0x..m0x` aggregate manifests。 |
| 前置依赖 | 用户批准的只读边界；远端SSH可访问；禁止test样本级读取、训练和远端写。 |
| 输出物 | 本报告；evidence ledger TSV；readonly replay JSON。 |
| 验收标准 | 14节齐全；formal/engineering/scientific/publication状态分开；关键事实有路径/符号/hash；legacy与standard replay分开；test本轮无新增访问；远端状态不变。 |
| 风险 | untracked artifacts缺role/hash；关键baseline scores缺失；单publication限制统计重放。 |
| 失败处理 | 缺artifact标`NOT_REPLAYABLE`；不补训、不重新拟合、不打开test；结论降级而不是填值。 |
| 优先级 | P0；本轮完成。 |

### Phase 1：可靠 benchmark 与 evaluator 重建

| 字段 | 内容 |
|---|---|
| 阶段目标 | 建立唯一、冻结、无test污染、publication-disjoint且可从clean checkout重放的benchmark/evaluator。 |
| 任务 | ①新authority epoch绑定M0 FAIL与exposure终态；②为1024 assets建逐项controlled-disposition；③版本化strict parser/rescue；④重建canonical pair/profile manifest；⑤实现edited/alignment/probe-eligibility exclusions；⑥统一parent/study/publication/lineage/family；⑦train-fold-only noise/caller；⑧退役16SFWJ test并新建untouched publication split；⑨实现标准AP、Brier、coverage、WMAE skill、hierarchical resampling和`UNIDENTIFIABLE`边界；⑩append-only exposure/claim ledger。 |
| 涉及模块 | 建议新增 `data_registry/d1x_v2/`、`configs/reactflow_delta/endpoint_v2.yaml`、`configs/reactflow_delta/split_v2.yaml`、`scripts/reactflow_delta/d1x_v2_*`、`caller_v2.py`、`evaluate_v2.py`、`tests/reactflow_delta/test_*_v2.py`；旧文件只读保留。 |
| 前置依赖 | 新授权；明确数据许可；M0 terminal/authority闭合；确认可用外部publications；定义primary endpoint不再静默变更。 |
| 输出物 | 1024-row disposition TSV/JSONL；dataset/caller/split/exposure manifests及hash；endpoint spec；validation-only fixtures；evaluator package；dataset capability matrix；new untouched test designation。 |
| 验收标准 | 1024/1024有非missing disposition；所有primary positions满足mask；caller在outer outcome不可见条件下生成；ID/hash闭合；至少3个独立confirmatory publications或prospective替代；标准AP行序不变；constant-label/permutation退化显式失败；旧test标`DEVELOPMENT_CONSUMED`。 |
| 风险 | 资产不可恢复；publication N仍不足；caller在低ICC studies不可靠；family/condition metadata缺失。 |
| 失败处理 | 若无法形成独立数据域，停止方法路线；选择狭窄resource/negative paper，或与实验组建立prospective cohort。 |
| 优先级 | P0；需要新授权。 |

### Phase 2：验证核心科学假设

| 字段 | 内容 |
|---|---|
| 阶段目标 | 在不使用复杂EPRO的情况下回答“允许输入下的response是否跨publication可学”以及哪些信息提供增量。 |
| 任务 | ①预注册hypothesis matrix；②nested leave-one-publication-out；③trivial/linear/logistic/GAM/tree/P2/DeepSets generic；④sequence-only、WT anchor、actual-alt、local/nonlocal proxy、noise weighting、publication balancing消融；⑤真正train-fraction学习曲线，每个fraction重训；⑥negative controls和有效label permutation；⑦caller reliability与performance关联。 |
| 涉及模块 | 建议新增 `configs/reactflow_delta/p2_hypotheses_v1.yaml`、`scripts/reactflow_delta/run_p2_v1.py`、`evaluate_v2.py`、`results/p2_v1/<run_id>/`；不复用旧B0 PASS。 |
| 前置依赖 | Phase 1全部PASS；confirmatory test仍封存；算力/seed/selection预算冻结。 |
| 输出物 | hypothesis matrix；outer-fold predictions；学习曲线；baseline/ablation表；publication-level effect/CI/permutation；STOP/GO decision manifest。 |
| 验收标准 | 至少5 seeds；模型和数据规模匹配；多个held-out publications方向一致；最强simple/generic相对null有预注册skill；无single-study domination；所有predictions/hash可重放。 |
| 风险 | label reliability不足；domain shift主导；有效publication太少；简单模型没有稳定skill。 |
| 失败处理 | 若P2 gate失败，立即停止架构Phase 3；转resource/measurement/negative路线，不通过增加隐藏层重新开gate。 |
| 优先级 | P0；仅在Phase 1 PASS后。 |

### Phase 3：模型架构迭代

| 字段 | 内容 |
|---|---|
| 阶段目标 | 只检验三种明确能力：pair-level对齐、exact-alt × WT-state、可控nonlocal传播；识别其独立增量。 |
| 任务 | ①方案一pair/conditional heads；②方案二exact-alt generic interaction；③方案三repaired EPRO；④容量/训练时间匹配；⑤每次只改变一个能力；⑥5 seeds配对消融；⑦梯度、residual、NaN、长度/内存测试；⑧development winner按预注册规则冻结。 |
| 涉及模块 | 建议新增 `reactflow_delta/models/pair_v1.py`、`exact_alt_v1.py`、`epro_v2.py`、`train_v2.py`、`samplers.py`、`tests/reactflow_delta/test_model_invariants_v2.py`；旧dev01–12不原地改写。 |
| 前置依赖 | Phase 2 GO；输入/输出/endpoint不再变；每方案预算、最大两轮和停止规则写入authority。 |
| 输出物 | model cards；configs；5-seed predictions；配对消融；复杂度/内存/残差表；winner freeze；退役清单。 |
| 验收标准 | candidate相对capacity-matched generic在development outer folds CI下界>0；exact-alt/nonlocal消融符合预注册方向；无梯度/收敛失败；增益大于seed variance；不靠post-hoc aggregation。 |
| 风险 | 小样本高方差；EPRO计算平方增长；结构feature噪声；过多自由度。 |
| 失败处理 | 单方案未达阈值即退役；最多两轮核心迭代；全部不胜则使用最简单generic并转benchmark/resource，不再开dev13+自由搜索。 |
| 优先级 | P1。 |

### Phase 4：SOTA 对比、消融与泛化

| 字段 | 内容 |
|---|---|
| 阶段目标 | 在完整且公平的baseline矩阵上判定“proposed frozen benchmark内是否有SOTA”，并只访问一次confirmatory test。 |
| 任务 | ①执行direct、adjacent、static-reactivity、static-structure、oracle五层矩阵；②冻结版本、input权限、参数/预算、pretraining exposure；③common/full coverage；④publication/family/probe/platform/condition分层；⑤final candidate与selection rule锁定；⑥一次性confirmatory run；⑦统计、校准和coverage联合gate。 |
| 涉及模块 | `baselines/manifest_v1.yaml`、每个官方工具adapter/container、`exposure_matrix.tsv`、`evaluate_v2.py`、`results/confirmatory_<id>/`、append-only access ledger。 |
| 前置依赖 | Phase 3 winner冻结；至少3个untouched publications或prospective cohort；独立审查确认test未暴露；环境和checkpoint hash冻结。 |
| 输出物 | 全baseline矩阵；common/full coverage表；final predictions；publication-level effects；calibration plots；confirmatory terminal manifest。 |
| 验收标准 | 相对最强同信息条件baseline的95% CI下界>0；有效permutation p<0.05；conditional WMAE skill CI下界>0；至少2/3 publications方向一致；no fallback-as-zero；all headline values可追溯。 |
| 风险 | 工具覆盖低、长度限制、pretraining污染、confirmatory方向反转、外部baseline适配不公平。 |
| 失败处理 | 不称SOTA；若benchmark结论仍稳定，转resource/negative；否则停止投稿并回到数据资格审查，不得重新看test调模型。 |
| 优先级 | P1。 |

### Phase 5：机制分析、科学发现与论文叙事

| 字段 | 内容 |
|---|---|
| 阶段目标 | 解释增益或稳定负结果，获得超过单benchmark排名的新认识。 |
| 任务 | ①response传播距离与contact/base-pair network关系；②WT-state dependence；③static ΔBPP/Δreactivity与experimental Δ失配；④caller reliability/domain shift/failure regimes；⑤实验选择utility；⑥必要时prospective/wet-lab验证；⑦claim–evidence map与预先定义分析。 |
| 涉及模块 | `analysis/mechanism_v1/`、figure specs、locked predictions、prospective protocol、claim ledger；不从test反向选择新proxy。 |
| 前置依赖 | Phase 4可信正增益，或跨多个publications稳定的benchmark负结果。 |
| 输出物 | 机制主图/补图；failure regime表；utility分析；wet-lab/prospective结果；核心/次级claim证据矩阵。 |
| 验收标准 | 机制方向跨publication一致；random/contact/local controls支持特异性；分析在看confirmatory outcome前冻结；新结论有独立证据。 |
| 风险 | 相关不等于机制；事后分组；multiple testing；无外部复现。 |
| 失败处理 | 删除机制强主张；保留方法或resource层级结论，完整报告negative analysis。 |
| 优先级 | P1。 |

### Phase 6：复现、代码整理与投稿准备

| 字段 | 内容 |
|---|---|
| 阶段目标 | 使论文每个headline数字、图和claim从clean checkout独立重放，并确保限制/数据可用性诚实。 |
| 任务 | ①clean environment one-command replay；②版本/权重/容器/checksum；③主表/图自动生成；④Data/Code Availability；⑤历史claim退役；⑥独立统计、代码、文献和exposure审查；⑦supplement、model/data cards；⑧投稿目标按证据等级选择。 |
| 涉及模块 | `environment.lock`/container；`reproduce.sh`或任务runner；`artifacts/manifest.json`；paper figures/tables；README；Data Availability；claim ledger。 |
| 前置依赖 | 所有主张与frozen artifacts对齐；test终态封存；许可和隐私审查通过。 |
| 输出物 | clean-checkout replay log；release tag；checksums；paper/supplement；editable figures/tables；independent audit signoff。 |
| 验收标准 | 独立机器重放全部主表/主图；所有prediction/model/data/code hashes闭合；headline claim无`UNKNOWN/NOT_RUN`依赖；限制和负结果完整；无旧SOTA/PASS残留。 |
| 风险 | 环境漂移、外部工具许可、untracked依赖、表图与结果不一致。 |
| 失败处理 | 任一主表不可重放即阻止投稿；修复release而不是手工改论文数字。 |
| 优先级 | P1/P2。 |

## 13. P0 立即执行清单

### 13.1 本轮只读 P0 执行结果

| 顺序 | 任务 | 输入 | 输出 | 状态 | 可信性判据 | 串并行 |
|---|---|---|---|---|---|---|
| P0-01 | 冻结HEAD/Git/合同/manifest/bundle/amendment/registry/gate/ledger/process快照 | 远端repo与aggregate artifacts | repo snapshot、hash表 | `COMPLETED` | HEAD 9fe8ad8c；远端合同631962f8；记录status hash | 必须最先串行 |
| P0-02 | authority lineage和test exposure crosswalk | active manifest、bundles、PH0代码/aggregate、ledgers | authority/exposure审计裁决 | `COMPLETED_AUDIT_ONLY` | checksum失败可重放；不访问test样本；本报告将16SFWJ裁决为`INVALIDATED_FOR_CONFIRMATORY_USE`，但repository/ledger的正式退役仍为`NOT_DONE_PENDING_R0_AND_NEW_AUTHORITY` | 依赖01，串行 |
| P0-03 | 对齐合同/PH0/B0/dev/M0 endpoint | 代码、manifest、aggregate | endpoint crosswalk | `COMPLETED` | 49/548与102/548来源定位；mask、majority/max、signed/abs均定位 | 依赖02，串行 |
| P0-04 | validation prediction identity/hash与无拟合重放 | validation pairlevel NPZ/JSON | replay JSON | `COMPLETED_WITH_MISSING_ARTIFACTS` | legacy candidate精确重放；标准AP行序不变；tree标准排名标NOT_REPLAYABLE | 依赖03，串行 |
| P0-05A | 数据/provenance/effective N/noise审计 | D0–D2/PH0 aggregate与代码 | data capability结论 | `COMPLETED` | missing不填零；pair/position/publication N分开 | 03后并行 |
| P0-05B | actual model flow/thermo/cache/sampling/complexity审计 | dev01–12、model.py、run manifests | model capability/delta表 | `COMPLETED` | 每项结论有路径/符号；README不作实现证据 | 03后并行 |
| P0-05C | 主来源文献、baseline层级和公平性审计 | 论文/官方代码/仓库运行状态 | SOTA/baseline矩阵 | `COMPLETED_BOUNDED_SEARCH` | direct/adjacent/proxy/oracle分开；未定位不写不存在 | 03后并行 |
| P0-06 | 综合评分、盲区、Goal、TODO和交付物 | 01–05结果 | MD/TSV/JSON | `COMPLETED` | 14节齐全；关键数字可回溯；JSON/TSV结构通过；最终远端HEAD、Git status hash和ledger/gate hashes与初始快照一致 | 最后串行 |

### 13.2 下一次新授权后的 P0 修复顺序

这些任务可以直接交给代码智能体，但必须先获新authority；“建议文件名”不是对现有仓库的自动修改授权。

| 顺序 | 直接执行任务 | 输入 | 输出 | 新增/修改文件建议 | 必须运行的测试/实验 | 完成后如何判断可信 | 串并行 |
|---|---|---|---|---|---|---|---|
| R0 | 终结旧authority/exposure | 本报告、M0 gate、现有ledgers | 新epoch terminal/bundle/sentinel；M0 FAIL；test consumed | 新版本authority与append-only exposure record；不覆盖旧文件 | checksum closure；conflicting epoch fixture应fail closed | bundle全成员hash通过；training=false；旧test不能被标sealed | 必须最先串行 |
| R1 | 冻结endpoint v2与信息权限 | 远端V4合同、endpoint crosswalk | `endpoint_v2.yaml`、mask/caller/score/stat spec | `configs/reactflow_delta/endpoint_v2.yaml` | synthetic hotspot/majority、signed/abs、pair-any-degenerate、missing-info tests | 每个task只有唯一unit/label/score/metric；变更需新version | 依赖R0，串行 |
| R2A | 1024 assets controlled disposition | frozen asset manifest、610失败日志 | 1024-row ledger、rescue yield by publication | `data_registry/d0x_v2/asset_disposition.*`、parser fixtures | 每个失败类别synthetic fixture；分层人工抽样；row-count/hash test | 1024/1024无空disposition；missing不为zero；新增publication yield可审计 | R1后可并行 |
| R2B | 修复parser/canonical mask | raw assets、endpoint v2 | canonical dataset v2、position eligibility reason codes | `scripts/reactflow_delta/d1x_v2_*`、`data_registry/d1x_v2/` | edited-site、alignment-change、probe-eligibility、NaN、length mismatch tests | 所有primary position有明确eligibility；旧/新差异有crosswalk | 依赖R1；与R2A协作并行 |
| R2C | 统一group atoms与split | canonical v2、publication/lineage metadata | split v2、overlap report、新test designation | `scripts/reactflow_delta/d2x_v2_split.py`、`configs/.../split_v2.yaml` | same PMID=one publication；parent definition一致；homology/lineage overlap tests | publication/family/lineage overlap=0或明确豁免；新test未暴露 | 依赖R2A/R2B关键metadata，后段串行 |
| R3 | 实现fold-local caller v2 | train-fold replicates/controls、mask v2 | caller params/labels/reliability per outer fold | `caller_v2.py`、`caller_manifest_v2.json` | deterministic seed；sliding cluster；spatial-block null；no outer-row I/O mock；low-ICC failure | 同输入hash同输出；outer outcome访问硬失败；reliability过低返回NO_CALL | 依赖R2B/R2C，串行 |
| R4 | 实现evaluator v2 | endpoint/split/caller specs | 标准metric/resampling package | `evaluate_v2.py`、tests | tied AP row-order；constant label；no mixed blocks；plus-one permutation；publication<3；signed/abs；coverage/missing | 与sklearn/R reference交叉一致；退化场景返回UNIDENTIFIABLE而非数字 | R1后可并行开发，R3后集成 |
| R5 | 重建直接基线/P2 gate | dataset/caller/evaluator v2 | nested outer predictions、learnability manifest | `run_p2_v1.py`、configs/results新目录 | 5 seeds；真正train-fraction重训；label permutation；single-study dominance；clean replay | multiple outer publications方向一致；effect/CI/permutation达到预注册阈值 | R2–R4全部PASS后串行 |
| R6 | GO/STOP科学裁决 | R5 results | `P2_GO`或`STOP_METHOD_ROUTE` terminal | append-only decision manifest | 独立审查脚本检查全部gates | 任何gate缺失即STOP/UNKNOWN，不允许人工override为GO | 最后串行 |

### 13.3 P0 必须覆盖的测试场景

1. **Authority fail-closed：** active hash不匹配、amendment未bind、terminal状态落后时拒绝训练。
2. **Test exposure：** caller读取test rows、计算统计或汇总labels均写入消费事件；aggregate不恢复sealed。
3. **Endpoint：** 单热点不得代表majority；负/正大幅变化在absolute task同rank；pair-any全阳返回degenerate。
4. **Metric：** tied AP与行序无关；legacy与standard分开；candidate/baseline必须是同类型score。
5. **统计：** 两study同PMID只算一publication；publication<3禁止confirmatory CI；无mixed blocks返回`UNIDENTIFIABLE`；permutation用`(b+1)/(B+1)`。
6. **数据/mask：** edited/alignment/probe changes排除；parse failure/missing不填零；group atom跨阶段一致。
7. **模型条件：** 同parent/position、不同alt不能共享thermo result；target mask不能作为prospective input；逐层梯度和参数更新非零。
8. **baseline公平：** learned tree元数据正确；工具失败不是零预测；teacher/feature provider/direct/oracle分表；unknown exposure不进入clean OOD claim。
9. **复现闭包：** code commit、data/split/caller/model/prediction/evaluator hash组成一条可重放链；缺环只报`REPORTED_NOT_REPLAYED`。

### 13.4 P0 完成判据

进入任何新模型训练前，必须同时看到以下机器可验证状态：

- `AUTHORITY_CLOSED_PASS`
- `ASSET_DISPOSITION_1024_OF_1024`
- `PRIMARY_MASK_V2_PASS`
- `GROUP_ATOMS_AND_PUBLICATION_SPLIT_PASS`
- `OLD_TEST_RETIRED_NEW_TEST_UNTOUCHED`
- `CALLER_V2_FOLD_LOCAL_AND_RELIABLE`
- `EVALUATOR_V2_REFERENCE_TESTS_PASS`
- `P2_LEARNABILITY_GO`

任一项为`FAIL/UNKNOWN/NOT_RUN`，Phase 3不得开始。

## 14. 项目转向或终止条件

### 14.1 立即停止继续增加模型模块的条件

当前已经满足：

- authority integrity断链；
- test被PH0消费；
- endpoint/selection/aggregation不一致；
- validation publication N=1；
- M0 formal gate FAIL；
- evaluator存在AP、resampling和score semantics问题。

因此在Phase 1–2完成前，应停止dev13/dev14式架构搜索。这里的“停止”是停止原protocol上的科学迭代，不是删除代码或否定问题价值。

### 14.2 转向 benchmark/resource/negative-result 路线

满足任一条件即转向：

1. 610 assets救援后仍无法获得至少3个untouched confirmatory publications或prospective替代；
2. corrected caller在多个studies可靠性不足，无法定义稳定binary response；
3. publication-level simple/generic P2没有稳定skill，但跨publication失效规律可重复；
4. static/mutation-effect/paired模型均在公平协议下失败，并可将失败定位到domain shift、noise或state-response mismatch；
5. EPRO不胜generic，但benchmark/provenance/评估协议本身构成可复用资源。

该路线的成功标准不是“候选最好”，而是数据资格、评估协议、强baseline和负结果在多个publication上可重放、可解释、可推广。

### 14.3 终止“EPRO SOTA方法”主张

满足任一条件即永久退役该主张：

- repaired EPRO在两轮预注册、容量/算力匹配迭代后仍不胜最强tree/generic paired model；
- propagation/contact消融不降低最终endpoint，或random contacts产生同等增益；
- 增益小于seed variance、只在单publication出现或confirmatory方向反转；
- exact-alt、WT anchor或nonlocal组件的机制方向不能跨publication复现；
- 任何正结果依赖事后threshold、z-max/sign、coverage fallback或test-informed选择。

退役EPRO不等于终止整个项目：可保留最简单稳定模型，并把方法论文改为benchmark/resource/negative finding。

### 14.4 暂停或终止整个监督预测项目

满足以下核心条件时，应暂停或终止监督prediction主线：

1. 合法caller/mask下label reliability不足，目标本身不可重复；
2. 允许输入中没有跨publication skill，且增加合法数据后学习曲线仍不改善；
3. 无法获得新数据域，现有所有publication均已参与开发或暴露；
4. 数据许可/provenance无法支持公开benchmark或独立复现；
5. 所需confirmatory/wet-lab资源长期不可获得，且resource/negative路线也没有足够跨域证据。

此时更合理的研究方向是：实验测量可靠性、跨平台normalization、domain adaptation、active data acquisition，或明确范围的single-family/probe study，而不是继续宣称general RNA mutation-response SOTA。

### 14.5 允许恢复方法路线的唯一条件

只有Phase 1数据/独立性全部PASS且Phase 2 simple/generic P2 gate提供跨publication正skill，才允许恢复架构研究。恢复后最多执行第9节三个方案、每个最多预注册轮数；confirmatory test不能因development不理想而解封、重分或重复使用。

当前项目最应该先做的是冻结并重建一个没有 endpoint 漂移、test 污染和 publication 伪独立性的 benchmark/evaluator，因为在评价对象本身尚未成立时继续增加模型模块，只会扩大选择偏差，无法建立 SOTA 或投稿所需的可信证据链。
