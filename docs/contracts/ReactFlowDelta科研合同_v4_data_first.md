---
contract_schema: reactflow_delta.contract.v4
contract_id: reactflow_delta_v4_data_first
project_task_id: reactflow_delta
version: v4_data_first
document_status: DRAFT
authorization_status: DRAFT_AWAITING_USER_SIGNOFF
current_phase: RECOVERY_CONTRACT_REWRITE
canonical_repo_path: docs/contracts/ReactFlowDelta科研合同_v4_data_first.md
active_authority_manifest: configs/reactflow_delta/active_contract.yaml
training_allowed: false
full_data_recall_allowed: false
new_split_allowed: false
confirmatory_test_access_allowed: false
cross_project_export_allowed: false
data_scope: PUBLIC_RETROSPECTIVE_ONLY
new_wet_lab_allowed: false
allowed_phases:
  - PILOT_CLOSURE
runnable_phases: []
default_test_status: DEVELOPMENT_CONSUMED
cross_project_default: REACTFLOW_DELTA_ORACLE_NOT_AUTHORIZED
---

# ReactFlow-Delta V4 Data-First 科研与执行合同

版本：V4 Data-First  
日期：2026-08-03（Asia/Shanghai）  
仓库：/home/cunyuliu/reactflow_delta_goal_20260729  
当前阶段：RECOVERY_CONTRACT_REWRITE  
当前授权：DRAFT_AWAITING_USER_SIGNOFF  
训练授权：false

本文是一份新的、自包含的治理与科研合同。只有本文 exact raw-byte hash 经用户签署、active manifest 切换为 ACTIVE_AUTHORIZED 后，它才取代 V3 系列用于未来执行；当前 draft 只在已获批准的 recovery/closure 范围内收窄权限，不能追认或扩张旧 authority。本文不删除、不补签、不改写旧文档，也不授权 D0-X 全库召回、dataset/split 构建、baseline、P2、EPRO 或 confirmatory test 访问。

---

## 0. 一页执行结论

1. ReactFlow-Structure 与 ReactFlow-Delta 是两个不同任务。旧结构生成状态、静态 reactivity 数据和 F1 只能作为历史背景或去污染预训练候选，不能计作 Delta 数据或 Delta 科学证据。
2. 现有 1,509/7,660、M0/M0-R/M0-R2/R3 统一登记为：

   ~~~text
   DEVELOPMENT_ONLY
   CONTRACT_NONCONFORMING
   NO_CONFIRMATORY_CLAIM
   ~~~

3. 旧 Csde1/Tetrahymena scientific test 已用于开发，唯一机器状态固定为 DEVELOPMENT_CONSUMED；改名、重切或移动文件不能恢复 untouched。
4. 正式主监督只接受可回到原始 profile 的 exact ref/alt 单碱基 substitution、匹配 WT-mutant 条件和完整 parent lineage。alt=X 只能是 AUXILIARY_LATENT_ALT。
5. 数据与 benchmark 优先。Tier B+ 只授权 P2 WT-anchored 与小模型；只有 Tier A+ 且 P2 已显示跨 parent/study 可学习信号时，P1 才可成为共同主任务。
6. 主估计量改为可重复 changer 的 detection/ranking；changer 条件下 signed magnitude 为第二个估计量。全位置连续 Delta 是共同次级 endpoint。
7. EPRO 只在数据、可靠性、强基线、置换和工程 Gate 全部通过后获得最长 28 天、最多 6 个科学迭代；任一上限先到即停。
8. 本轮只完成 pilot closure、V4、active manifest、mRNA additive interface、同步和 focused commit；到用户签署点停止。

---

## 1. 项目身份、科学问题与任务边界

### 1.1 两个不可混同的项目任务

| 任务 | 输入/目标 | 本合同中的用途 | 禁止混写 |
|---|---|---|---|
| ReactFlow-Structure | 从静态序列预测/生成结构或单 profile reactivity | historical context；STATIC_PRETRAINING 候选 | 结构 F1、静态样本数不得成为 Delta pair 数、Tier 或 E0-X |
| ReactFlow-Delta | 给定 WT endpoint、具体 mutant endpoint、edit、condition，预测突变诱导的实验响应差 | 本合同唯一正式任务 | 不得把静态结构成功写成突变响应成功 |

ReactFlow-Delta 的基本对象是：

\[
(x_w, x_m, e, c, r_w) \longrightarrow \Delta r = r_m-r_w .
\]

P2 可使用 WT profile r_w，但任何预测任务都不得把 mutant profile r_m 作为输入。任务输出必须服务 single-mutant 结构响应筛选与排序，并同时给出 changer 概率、方向、幅度、位置响应图、不确定性和拒绝状态。

### 1.2 P2 与 P1 的授权顺序

- P2_WT_ANCHORED 是 Tier B+ 后的首要正式任务：输入具体 WT/mutant 序列、exact edit、condition，允许读取 WT profile；禁止读取 mutant profile。
- P1_SEQUENCE_ONLY 不读取 WT 或 mutant 实验响应；只有 Tier A+、P2 跨 parent/study 可学习 Gate PASS、且 active manifest 另行授权后，才可升级为共同主任务。
- P1/P2 具有不同信息条件，必须分开 protocol、selection、结果表和 claim；是否共享冻结 encoder/物理主干或使用预注册 multi-task training，只能在 Tier A+ 后由新授权决定。无论是否共享参数，P2 WT anchor 不得泄漏到 P1，且不得合并指标来隐藏任一路线失败。

### 1.4 数据与湿实验 scope

V4 只允许公开、可审计的 retrospective data。本合同不包含任何新增湿实验、样本采集或 prospective validation；未来湿实验只能作为另立合同的外部升级路线，不能被预记为当前数据、Gate 或 claim evidence。

### 1.3 当前最重要的未决经验问题

1. 固定全库 source universe 后，究竟能恢复多少 exact-alt、condition-matched、独立 study/parent 的 primary pair？
2. 最外 1% Delta 尾部是可重复生物学响应，还是 mutation 解析、坐标、对齐、归一化、probe eligibility、batch 或尺度问题？
3. 在 study/parent/replicate-block 作为有效单位时，public data 是否足以定义可靠 changer 并支持新 untouched study？

在回答这些问题前，不得用更大模型替代数据识别性。

---

## 2. Authority、优先级与旧合同处置

### 2.1 单一 authority

唯一机器 authority 路径为：

~~~text
configs/reactflow_delta/active_contract.yaml
~~~

runner 必须先验证该 manifest，再读取本文。文件名、mtime、Markdown 空白签名、checkpoint 路径存在或“最新版本”文字都不是 authority。

优先级：

~~~text
active authority manifest
→ 本 V4 raw bytes
→ 已绑定的 run manifest
→ 历史文档与历史 artifacts
~~~

如果找到多份声称 SINGLE_ACTIVE_AUTHORITY 的 ReactFlow-Delta manifest，必须在任何数据或模型加载前 fail closed。

### 2.2 旧合同状态

| 文档 | SHA-256 / 状态 | V4 处置 |
|---|---|---|
| V3 | 3efcc1504208d8089236dfe4e7d41553741441d3b86b6174c8b5af52d614ec10 | V4 激活后 SUPERSEDED_HISTORICAL_ONLY；当前 recovery freeze；原字节必须不变 |
| V3.1 | 09d646aef8b0dddf789ee860e0e14499fb56aab654afecf9140f2d2448d1c091 | V4 激活后 SUPERSEDED_HISTORICAL_ONLY；保留 |
| V3.2 | ad9d439d1e9bf3b85bb19088b2d9a2f7df520bf214e3fe1d93300cc6ccabdc39 | V4 激活后 SUPERSEDED_HISTORICAL_ONLY；保留 |
| V3.3 | 9a38b220ecb6f324cf8a33b19bd30bfa98ada1112df0db440d9c1bc67ecf4f45；当前观察为 untracked，签署为空 | NEVER_VALIDLY_AUTHORIZED / PRESERVE_AS_FOUND |
| V3.4 | b1669ed553cf868f2a725e9315a707a5961af173c83746e8f25db00c8a122d35；签署为空 | NEVER_VALIDLY_AUTHORIZED / PRESERVE_AS_FOUND |
| V3.5 | 12cf38c256712d5f619090d16b3d1b9d44da729f951dfdb9d456c7a3d5d7d39a；签署为空 | NEVER_VALIDLY_AUTHORIZED / PRESERVE_AS_FOUND |

在 V4 签署前，当前用户批准的 recovery order 使所有旧训练 authority 保持冻结；V3.3-V3.5 因签署为空本来就不是有效 authority。V4 激活后的 supersession 只改变未来 authority，不改变旧文件、失败结果、时间戳或历史证据类别。不得补签、删除或把旧开发活动追认成 V4 conforming。

active manifest 只能收窄本文权限，不能放宽本文的数据、test、模型、claim 或安全禁令。manifest 与本文冲突时一律采用更严格边界并 fail closed；任何扩权必须形成新的 raw-hash-bound 合同 amendment 和用户批准记录。

### 2.3 当前 draft 的授权状态

active manifest 必须保持：

~~~yaml
authorization:
  status: DRAFT_AWAITING_USER_SIGNOFF
  approval_status: NOT_PROVIDED
  signed_by: null
  signed_at: null
  allowed_phases:
    - PILOT_CLOSURE
  runnable_phases: []
  training_allowed: false
  full_data_recall_allowed: false
  new_split_allowed: false
  confirmatory_test_access_allowed: false
~~~

其公共接口还必须包含 schema_version=reactflow_delta.active_contract.v1、project_task_id、SINGLE_ACTIVE_AUTHORITY、canonical V4 path/raw hash、current_phase、preflight Git binding、data/split/config/exposure bindings、test status 和 supersession list。未来签署记录必须绑定 approved_contract_sha256、approval scope、signer、RFC3339 timestamp、approval artifact hash，以及 supersession/revocation 规则。active manifest 与 detached ledger 的自引用限制按第 2.4 节执行。

用户审阅 V4 不是自动训练授权。即使用户签署本文，D0-X 仍需一个刷新后的 authority amendment，绑定执行 Git 与 frozen source-universe manifest；任何训练 phase 还必须绑定非空 data/split/config/exposure hashes 和对应 phase 权限。

allowed_phases 记录本轮用户批准过的最大 scope；PILOT_CLOSURE 已 TERMINAL 后，runnable_phases 必须为空且 phase.execution_authorized=false。runner 只接受同时位于 allowed_phases 与 runnable_phases、lifecycle 非 TERMINAL、rerun_allowed=true 的 phase；因此旧 PID closure 不能重入。

### 2.4 自引用与 Git 绑定

- 本文不嵌入自身 SHA-256；active manifest 绑定本文 raw bytes。
- active manifest 不嵌入自身 SHA-256；Git commit 或外部 checksum 可绑定它。
- draft 中的 Git 6515667020184fa6e5f8dc70acd199b2c3a8fbcb 仅表示本轮 preflight base commit，不是包含 V4 的提交，也不是训练 source authorization。
- 本轮可创建只含四类治理文件的 descendant governance commit；未来正式执行必须更新为 clean、完整、可复现的 authorized execution source commit。

---

## 3. 状态语义、证据类别与 claim 语法

### 3.1 三条独立证据轴

~~~yaml
evidence_class:
  - ENGINEERING_ONLY
  - DEVELOPMENT_ONLY
  - DATA_QUALIFICATION_ONLY
  - BENCHMARK_QUALIFICATION_ONLY
  - CONFIRMATORY_ELIGIBLE
  - HISTORICAL_ONLY
contract_conformance:
  - CONFORMING
  - CONTRACT_NONCONFORMING
  - UNKNOWN_NOT_ASSERTED
claim_eligibility:
  - NO_CONFIRMATORY_CLAIM
  - QUALIFICATION_CLAIM_ONLY
  - CONFIRMATORY_CLAIM_ELIGIBLE
  - INVALIDATED
~~~

checksum PASS、工程 PASS、数据 Gate PASS、benchmark PASS 和科学主比较 PASS 互不继承。

### 3.2 生命周期与 Gate 结果

- lifecycle：PLANNED、RUNNING、TERMINAL、SUPERSEDED；
- gate result 只有四态：NOT_RUN、PASS、FAIL、UNKNOWN_NOT_ASSERTED。运行等待、外部依赖或资源阻塞写入 lifecycle/stop_reason，不新增第五种 Gate result。

PASS 只允许在 phase terminal、所有 conjunctive 条件有机器可定位 evidence、finalizer、terminal marker、manifest 和 checksum ledger 全部闭环时写入。缺证据项只能是 FAIL、NOT_RUN 或 UNKNOWN_NOT_ASSERTED。

复合 Gate 与 required sub-Gates 必须按同一确定性优先级汇总：任一已执行且已知不满足阈值的 required 项为 FAIL；若无 FAIL 但任一 required 项的值或证据不可核实，则为 UNKNOWN_NOT_ASSERTED；只有所有 required 项均为 PASS 才为 PASS；required prerequisite 尚未执行时为 NOT_RUN。已知阈值失败不得写 UNKNOWN_NOT_ASSERTED，未知或 artifact 缺失也不得伪写 FAIL。UNKNOWN_NOT_ASSERTED 与 NOT_RUN 均 fail closed、不能解锁 downstream，也不满足 REPORT-X 的 terminal FAIL entry condition。

正式 run 的 terminal marker 使用 phase 专用 DONE 仅在合同定义的全部 formal closure 条件满足后。当前 pilot 只能使用 DEVELOPMENT_CLOSED，不得使用 formal DONE 或 VERIFIED。

### 3.3 事实语言

- LOCATED_EVIDENCE：当前可定位、hash 或原路径可审计；
- INFERENCE：由 evidence 推得，必须说明推理链与替代解释；
- HYPOTHESIS_NOT_TESTED：待验证假设；
- FUTURE_PROPOSAL_NOT_RUN：未来设计，禁止写成已完成；
- NOT_AVAILABLE_NOT_ASSERTED：证据不可得，禁止填零或当 PASS。

---

## 4. 当前历史结果和 pilot closure

### 4.1 统一科学分类

1509 pilot、7660 pilot、M0/M0-R/M0-R2/R3-like work、历史 B0/PH0/O0 标签全部是历史 development，不进入 V4 formal lineage。

统一状态：

~~~text
DEVELOPMENT_ONLY / CONTRACT_NONCONFORMING / NO_CONFIRMATORY_CLAIM
~~~

原因至少包括：签署 authority 链无效；1,509 的 concrete alt/profile sequence 证据不足且使用 alt=X 边际化；7,660 增量包含 annotation-only/non-verifiable candidates；旧 test 已开发消费；launch-time source/config bytes 未冻结；formal finalizer/manifest/checksum closure 缺失。

1,509 和 7,660 只能称为历史 candidate/development record count，不能称为 exact endpoint pair，也不能计入 Tier。

一项历史 development diagnostic 在 500 个 candidate pairs、118,750 个位置上观察到：Delta reactivity normalized 的均值约 0.0033、标准差约 0.0640、中位数约 -0.0003；约 91.4% 的位置绝对值小于 0.01，约 40.9% 小于 0.001。历史 thermo delta 的结构概率/熵类特征也大量为零，能量类特征相对更有变化。该结果只解释“全位置 MSE 容易被近零位置支配”的风险，不能证明近零是生物学不变、数据正确或任务不可学；D1-X/PH0-X 必须按 study、parent、probe、raw layer 和 controls 重新审计。

### 4.2 已完成的安全收口

canonical closure：

~~~text
/mnt/cunyuliu/reactflow_delta_artifacts_20260729/reactflow_delta/recovery/
pilot_closure_20260803T200500+0800_r2
~~~

小型报告：

~~~text
docs/audits/reactflow_delta_pilot_closure_20260803.md
~~~

两批进程在 user、PID/PPID、/proc start tick、cwd、exe、8 个 argv token、日志 fd、CUDA logical device 和 GPU UUID 连续两轮完全匹配后，仅向两个 Python child PID 发送 SIGTERM。随后观察到目标 PID 从 /proc 和 GPU process table 消失。没有 wait status/exit code，因此这是“安全终止并观察消失”，不是 normal completion，也不是 checkpoint-aware graceful completion。

runner 没有 signal-aware latest checkpoint、finalizer 或 natural-run sentinel；保留的是 best-validation development checkpoint，best 之后的内存状态未保存。cuda_fallback_count=NOT_RECORDED。runtime_source_commit_binding=NOT_AVAILABLE_NOT_ASSERTED。

旧 SHA256SUMS.preterm 因 guard 文件在生成后继续追加而保留一项历史失败；final SHA256SUMS 在所有证据和 v2 manifest 写入后重新生成并逐项 PASS。该 integrity PASS 只证明包完整，不证明模型有效。

### 4.3 旧 test

Csde1/Tetrahymena 的标签与域已参与历史 baseline/model selection 或 development inspection，唯一机器枚举统一登记 DEVELOPMENT_CONSUMED。不得通过改名、重切、换目录、换 seed 或只看部分指标恢复 untouched。未来 E0-X 必须来自 D0-X 后找到并在任何开发消费前冻结的新 study。

---

## 5. 决策日志

| ID | 旧定义/问题 | 新定义 | 证据类型 | 影响 Gate | 当前允许主张 |
|---|---|---|---|---|---|
| D-01 | Structure 状态与 Delta 混杂 | 两任务显式分离 | located docs | 全部 | 旧数据仅历史/预训练 |
| D-02 | V3.3 用 2 study/6 parent 宣布 Tier B | Tier B+ 为不可降的全合取 Gate | located contract contradiction | D2-X | 当前 NOT_RUN |
| D-03 | alt=X 三 alt 均值近似 concrete mutant | primary 必须 exact ref/alt；X 为 auxiliary | located schema/config | D0-X/D1-X | 不能称 exact endpoint |
| D-04 | P1 primary、P2 enhancement | Tier B+ 先 P2；Tier A+ 后 P1 co-primary | identifiability inference | D2-X/B0-X | P1 未授权 |
| D-05 | 单一全位置 MSE/Skill | changer ranking + conditional magnitude；continuous secondary | target sparsity evidence | PH0-X/E0-X | 当前未证明可学习 |
| D-06 | observed outcome percentile 作 noise ceiling | 只允许 controls/replicates/upstream error model | circularity audit | PH0-X | 历史 PH0 不继承 |
| D-07 | pair-level ratio Skill 宏平均 | pooled ratio-of-sums + UNSCORABLE_RATIO | metric audit | B0-X/E0-X | 保留绝对损失差 |
| D-08 | 2-6M EPRO-Lite 作为 Tier B 默认 | 小容量 ladder；历史约 4.46M 不继承 | effective sample-size risk | B0-X/M0-X | 大模型未授权 |
| D-09 | matched generic 未证实匹配 | 50k-250k、参数与主计算 ±5%，协议完全匹配 | audit | B0-X/E0-X | 旧比较仅 development |
| D-10 | 静态数据可混入 Delta 数量 | from-scratch/去污染预训练双臂，样本数分离 | exposure risk | D2-X/M0-X | static 不计 Delta pair |
| D-11 | 旧 test 可继续使用 | DEVELOPMENT_CONSUMED，新 test 一次解封 | located exposure | D2-X/E0-X | 旧 test 不确认 |
| D-12 | 无限 repair 或改名继续 | 28 天/6 科学迭代，连续 3 次不胜强基线早停 | historical loop | M0-X | terminal FAIL 路由 REPORT-X |
| D-13 | Markdown 空签名授权 | 单一 active manifest 与 raw-byte hash | located unsigned chain | authority | 当前 training=false |
| D-14 | checkpoint/exit 代表 DONE | finalizer + sentinel + checksum + evidence closure | closure audit | 全部 | pilot 仅 DEVELOPMENT_CLOSED |
| D-15 | 数据库大计数等于可用监督 | source universe → profile proof → exact pair | source semantics | D0-X | 大计数仅 context |
| D-16 | EPRO 可自动成为 mRNA oracle | producer Gate + consumer amendment 双授权 | cross-project risk | P0-X | 当前 NOT_AUTHORIZED |

所有新 decision 必须追加，不得静默覆盖本表。新条目要记录旧/新规则、evidence、受影响 Gate、rerun 要求与 claim effect。

---

## 6. Source universe 与“穷尽”定义

### 6.1 D0-X 的未来召回范围

D0-X 本轮 NOT_RUN，执行权限为 false。用户签署并另行授权后，source universe 至少包含：

1. 固定 Git commit/release/full manifest 的完整 RMDB snapshot，而不是 filename filter；
2. construct-level 与 profile-level ANNOTATION_DATA、mutation、sequence、name、experiment type、condition 和 error 字段；
3. mutate-and-map、M2-seq、mutate-map-rescue；
4. Eterna/OpenKnot、公开 riboswitch variant profiling；
5. primary paper supplement、GEO/SRA 与 accession crosswalk；
6. classSNitch 关联的 17 个 RNA、2,019 mutant traces，作为交叉核验/外部 evaluation candidate；
7. Ribonanza 静态 chemical-mapping profiles，主要作为去污染预训练候选；只有独立重建 exact matched intervention 的子集才可重新走 primary eligibility；
8. 其他在 D0-X 开始前经 signed source-universe amendment 加入的公开、许可明确来源。

2026-08-03 访问的 RMDB 页面显示 1,024 entries、4,556,825 constructs、520,709,190 data points；该页面也说明 RMDB 包含 SHAPE、DMS、CMCT、1M7、mutate-and-map 等并以 RDAT/Git 版本化。数字会变化，D0-X 必须重新冻结；construct 不是 pair。

Ribonanza primary record 描述了约 200 万条多样 RNA 序列的 chemical-mapping measurement。该规模是静态 profile/pretraining context，不是 200 万 Delta pairs；只有能够独立重建 exact matched intervention、并通过本合同全部 primary cleaning/split Gate 的子集才可另行计数。

RDAT 当前规范说明 data-level ANNOTATION_DATA 可覆盖 construct-level annotation；解析器必须保留两层原文、resolution trace 和最终值，不能只读构造级字段。

### 6.2 “穷尽”的可审计定义

“穷尽”只在以下全部存在时成立：

- frozen source-universe manifest、版本/commit/release、许可与 raw SHA-256；
- 每个 accession 的状态：SEARCHED、NOT_SEARCHED、UNAVAILABLE_WITH_EVIDENCE、LICENSE_RESTRICTED、DOWNLOAD_FAILED、PARSE_FAILED、PARSED；
- 下载、镜像、重复文件、publication/accession crosswalk；
- parser coverage、silent-drop test、字段保持率、profile count 对账；
- source → canonical 的可逆映射与 exclusion flow；
- 按 source/study 抽样的人工审计；
- 所有失败 stderr、exit code 与 partial artifact。

未检查来源必须写 NOT_SEARCHED，不可写成 0。一个 publication 被多个 repository 镜像不得增加独立 study 数。

### 6.3 数据许可与原始层

- raw file 只读，先记录 license/version/source URI/accession/size/hash，再解析；
- 许可不清或不可再分发的数据不得放入普通 Git，也不得标为 public reusable training；
- raw、upstream-processed、train-frozen 三层分开保存、各自 hash；
- raw 不覆盖；parser 修复产生新 parent-linked dataset build；
- large data、checkpoint、cache、日志留在专属 artifact root，Git 只跟踪小型 manifest/report。

---

## 7. Canonical 数据接口

### 7.1 强制字段

每个候选 record 至少包含：

~~~yaml
source_accession: string
source_profile_index: string_or_integer
raw_mutation_token: string_or_null
ref_allele: A_C_G_U_or_null
alt_allele: A_C_G_U_or_null
mutation_coordinate_system: controlled_object_or_null
exact_mutation_evidence_status: controlled_enum
source_to_canonical_retention_status: controlled_enum
parent_lineage_evidence: controlled_object
condition_match_evidence: controlled_object
noise_source: controlled_object_or_null
replicate_block_id: string_or_null
measurement_variance: number_or_null
data_role: closed_enum
exclusion_reason: controlled_enum_or_null
~~~

同时必须保留 source/canonical sequence、source/canonical coordinate、profile pointer、study/publication、parent/design lineage、probe、temperature、ligand、buffer、batch、in vivo/in vitro、WT reuse group、reactivity-layer refs、position mask 和 missing reason。

null 必须伴随 missing reason；无测量、无效、解析失败和零是四种不同状态。

canonical schema 固定为 reactflow_delta.data_record.v4.0；执行者不得自行扩展以下 machine enums：

- exact_mutation_evidence_status：VERIFIED_EXACT_SINGLE_SUBSTITUTION、LATENT_ALT、MULTI_EDIT、CONFLICTING_EVIDENCE、INVALID_COORDINATE、MISSING_EVIDENCE；
- source_to_canonical_retention_status：LOSSLESS_REVERSIBLE、LOSSY_EXCLUDED、NOT_AVAILABLE_EXCLUDED；
- condition_match_evidence.status：MATCHED_ALL_REQUIRED、MISMATCH_PROBE、MISMATCH_TEMPERATURE、MISMATCH_LIGAND、MISMATCH_BUFFER、MISMATCH_BATCH、MISMATCH_ENVIRONMENT、MISSING_REQUIRED_FIELD；
- noise_source.type：MATCHED_REPLICATE、NO_EDIT_CONTROL、WT_WT_CONTROL、CONTROL_MUTATION、UPSTREAM_ERROR_MODEL、NO_IDENTIFIABLE_NOISE_MODEL；
- missing_status：MEASURED、UNMEASURED、INVALID_VALUE、PARSER_FAILURE、ALIGNMENT_EXCLUDED、PROBE_INELIGIBLE；
- exclusion_reason：LATENT_ALT、MULTI_EDIT、REF_MISMATCH、ALT_MISMATCH、COORDINATE_AMBIGUOUS、SEQUENCE_MISMATCH、LINEAGE_UNKNOWN、CONDITION_MISMATCH、CONDITION_MISSING、PROFILE_POINTER_MISSING、DUPLICATE_RECORD、MASK_EMPTY、LICENSE_RESTRICTED、EXTERNAL_ONLY、STATIC_ONLY、OTHER_PREAPPROVED_CODE。

OTHER_PREAPPROVED_CODE 只有在 schema amendment 于 D0-X 前签署后才能出现，且必须带 explanation；自由文本不能替代 enum。

默认 strict condition policy 为：probe/modifier token 完全相同；temperature 经单位归一后差值不超过 0.5 摄氏度；ligand identity 与浓度、buffer composition、batch 和 in vivo/in vitro environment 必须相同。若 source protocol 可证明某字段在整个 experiment 恒定，可用 hash-bound protocol pointer 代替 profile 值；否则任何 required field 缺失都排除 primary。

稳定 dedup key 为 SHA-256(source study identity、parent lineage、WT canonical sequence、mutant canonical sequence、exact edit、condition tuple、source profile pointer) 的 canonical serialization。pair_id 由 dedup key 派生；镜像/重复 processing 不产生新 pair。

### 7.2 data_role 闭集

| data_role | 允许内容 | 科学用途 |
|---|---|---|
| PRIMARY_EXACT_DELTA | exact ref/alt、单 substitution、source-profile pointer、同条件 WT-mutant、完整 lineage | P2/P1 正式监督与 Tier 计数 |
| AUXILIARY_LATENT_ALT | alt=X、null alt、三 alt 边际化或具体 endpoint 无法核验 | 候选召回或单独命名的 latent-alt 辅助任务 |
| RESCUE_MULTI_EDIT | 双突变、补偿突变、rescue series | 探索性 rescue ranking |
| STATIC_PRETRAINING | Ribonanza、旧 ReactFlow Structure/static reactivity | 去污染预训练，不计 Delta pair |
| EXTERNAL_EVAL_ONLY | classSNitch、PARS/RiboSNitch 等冻结外部 stress/sanity | 禁止训练、选择、阈值和 calibration |

alt=X、unknown/null alt、ref/alt 冲突、多 edit、lossy coordinate 或 profile pointer 缺失永远不能通过改名成为 PRIMARY_EXACT_DELTA。

### 7.3 Primary exact eligibility

PRIMARY_EXACT_DELTA 必须同时满足：

1. ref 与 alt 都是 A/C/G/U 且不同；
2. WT 与 mutant 同长、恰好一个 substitution；
3. mutation token、source coordinate、source WT base、source mutant sequence 四者一致；
4. source → canonical coordinate 可逆，并明确 0/1-based、offset、orientation；
5. WT 与 mutant probe/temperature/ligand/buffer/batch/environment 符合冻结的 match policy；
6. parent、design lineage、barcode/scaffold 与 functional region 可分离且有 evidence；
7. source profile pointer 可直接定位 raw reactivity/error；
8. position mask 排除 edited site、对齐改变和 probe eligibility 改变位置；
9. eligibility 与 exclusion reason 由版本化规则产生并可重放。

所有非 primary candidate 都必须有 controlled exclusion_reason，而不是从数据流中静默消失。

---

## 8. 清洗闭环

冻结顺序，不得跳步：

1. raw file 只读、license/version/hash/source universe 冻结；
2. profile-level mutation、sequence、condition、error 解析；data-level override 保留；
3. exact WT→mutant substitution 与坐标核验；
4. parent、design lineage、barcode/scaffold、functional region、shared-WT group 分离；
5. probe、温度、配体、buffer、batch、in vivo/in vitro 匹配；
6. raw/upstream/train-frozen reactivity 三层保留；
7. missing/invalid/unmeasured 使用 null + mask + reason，绝不填零；
8. edited site、alignment change、probe eligibility change 从 primary endpoint 排除；
9. replicate/no-edit/control 建模，估计 study/probe/batch noise、ICC/reliability、shared-WT covariance；
10. 按 study、parent、probe、processing layer 重审 near-zero spike 与最外 1% 尾部；每个尾部点回到 raw profile；
11. 完成 outcome-blind eligibility cleaning 后，只依据 study/parent/design-lineage/source metadata 生成 provisional group-role assignment 并立刻封存 test；不得依据 Delta 或 changer prevalence 调整角色；
12. 只用 provisional training role 的 controls/replicates 拟合 normalization、scale 和 caller，再由 blind curator 生成 test aggregate viability certificate；最终 split/exposure manifest 必须与 provisional assignment 身份相同，禁止 outcome-driven reassignment。

禁止：

- 按观察到的绝对 Delta 加权来制造 active signal；
- 删除近零、裁剪不利尾部或将 missing 填零；
- 依据 test/validation outcome 选择 source、pair、mask 或 normalization；
- 把 shared WT 下的 mutants 当作独立测量重复；
- 用 outcome percentile 自己定义 noise ceiling；
- active oversampling 后忘记原总体权重与概率校准。

若重采样用于优化效率，必须保留每条样本的原 inclusion probability、population weight，并在原分布上校准/评估。

---

## 9. 噪声、可靠性与 changer caller

### 9.1 noise source 优先级

1. 同 study/probe/batch 的 biological/technical replicates；
2. no-edit、WT-WT、control mutations；
3. 有 provenance 的 upstream measurement-error model；
4. 若以上不足：NO_IDENTIFIABLE_NOISE_MODEL。

观察到的全体 Delta 分布、模型 residual 或 test labels 不得用于生成正式 changer threshold。

### 9.2 Replicate-aware caller

- caller 只在 training folds 拟合；
- caller 版本、输入 controls、variance model、shared-WT covariance、阈值、FDR/decision rule 与 seed 全部冻结；
- validation/test label 由冻结 caller 和各自独立测量输入生成，模型不能参与；
- 没有足够重复性证据时，changer label 状态为 UNRELIABLE_NOT_PRIMARY；Estimand A changer detection/ranking 与 Estimand B conditional magnitude 都停止作为 primary，最多保留全位置 continuous descriptive/benchmark endpoint，并进入 data/resource 路由；
- caller uncertainty 必须进入 calibration/敏感性分析，而不是伪装成无误差标签。

### 9.3 PH0-X identifiability

PH0-X 的问题是“数据与 controls 是否能区分可重复响应和测量噪声”，不是“某模型 loss 是否下降”。若 reliability、changer count、matched noise coverage 或 group-aware permutation Gate 失败，停止模型路线并转入只做证据包装的 REPORT-X；不得继续模型优化。REPORT-X 可以诚实形成 data/resource/benchmark 或 negative report，但其 PASS 只证明报告闭环，不反向升级失败的科学 Gate。

---

## 10. Split、exposure 与 sealed test

split 在 eligibility cleaning 第 1-10 步后按第 11-12 步一次确定：先做 outcome-blind group assignment，再 fit training-only transforms，最终 manifest 不得改变 group roles。推断层级为：

~~~text
study → parent → design_lineage → pair → position
~~~

必须满足：

- train/validation/test 的 study、parent、design-lineage overlap 全部为 0；
- exact sequence、near-duplicate、family、structure、source mirror 与 pretraining exposure 单独审计；
- normalization、caller、feature fitting、pretraining choice、calibration 与 early stop 只读 permitted folds；
- test study 在模型、阈值、pretraining、split、metrics、baseline、capacity、selection tie-break 全冻结后只解封一次；
- test access 产生 append-only ledger；失败或进程中断不能再次解封；
- external-eval sources 不参与模型/阈值/calibration 选择；
- 任何 overlap unknown 都 fail closed；不得写 assumed clean。

同一 publication、同一 lab/platform、共享 parent/WT/scaffold 或同一原始 deposit 的多个 accession 不自动算独立 study。有效样本量按 study/parent/replicate block 报告，不能用 nucleotide positions 代替。

### 10.1 Blind test-viability certificate

Tier 要求 test study 具有足够 exact pairs/changers，但 development team 不得因此读取样本级 test labels。D2-X 必须采用独立 blind curator/evaluator：在任何 test outcome inspection 前冻结候选 study 的确定性优先序、training-only caller、eligibility rule 和最小 aggregate schema；curator 只返回 exact-pair count、changer count、control/replicate count 及 PASS/FAIL certificate，不返回 pair identity、position label、profile、prediction 或 per-pair statistic。所有被筛查 study 和 aggregate exposure 进入 append-only ledger；失败 study 不得回流训练或改名为 untouched。没有独立 curator、冻结 caller 或 aggregate-only enforcement 时，test changer count 为 UNKNOWN_NOT_ASSERTED，Tier 不 PASS。

---

## 11. Tier B+ 与 Tier A+

### 11.1 Tier B+：全部合取

Tier B+ 同时要求：

1. 至少 3 个独立 study/publication；
2. 至少 10 个独立 parent；
3. 至少 1,000 个唯一 exact-alt、primary-eligible、condition-matched single-mutant pair；
4. 至少 1 个从未消费的完整 test study，含至少 100 exact pair；
5. training pool 至少 100 个可重复 changer；
6. validation 与 test 各至少 20 个可重复 changer；
7. 至少 3 个独立 study/parent control 或 replicate block，总计至少 100 control/replicate observations；
8. 至少 80% primary pair 绑定 study/probe/batch-matched noise estimate；
9. primary pair 的 exact mutation、condition、parent lineage、source-profile pointer 完整率 100%；
10. parent、study、design-lineage overlap 为 0；
11. 单一 parent 不超过 primary domain pair 的 40%；
12. 至少一个 probe/condition domain 可独立成集。

每个条件必须记录 observed value、comparator、evidence path/hash、status。未知或缺失即整个 Gate 不 PASS。不得用加权分数、豁免、改名、降低阈值或混入 auxiliary/static/external 数据恢复 PASS。

Tier B+ 只允许 P2 WT-anchored；只允许 trivial/linear/tree、10k-100k paired baseline、50k-250k matched generic/EPRO-Small；任何 development 训练仍需 active manifest 明确授权。

Tier B+ 不允许 P1 headline、2-6M EPRO、当前约 4.46M 模型、sealed test 解封或 confirmatory claim。

D2-X 只能产生 TIER_B_PLUS_DATA_CANDIDATE；PH0-X 在 training-only caller、blind test certificate 和 reliability/permutation evidence 完成后才可把 TIER_B_PLUS 写为 PASS。D2-X 自身不得提前宣告完整 Tier B+。

### 11.2 Tier A+：全部合取

Tier A+ 同时要求：

1. 至少 5 个独立 study；
2. 至少 20 个独立 parent；
3. 至少 5,000 exact primary pair；
4. 2 个未消费 test study，每个至少 100 exact pair、至少 20 changer；
5. training 至少 200 可重复 changer；
6. Tier B+ 所有适用 integrity/reliability 条件继续满足；
7. family/structure/exposure audit PASS；
8. from-scratch 和 decontaminated-pretraining arms 可在同一 benchmark 执行；
9. P2 已在冻结 development protocol 下显示跨 parent 和跨 study 可学习信号。

D2-X/PH0-X 只能产生 TIER_A_PLUS_DATA_READY。完整 Tier A+ 还要求 B0-X 的 frozen P2 cross-parent/cross-study learnability PASS；因此 Tier A+ 只能在 B0-X terminal 时最终判定，不能成为 B0-X 的前置依赖。完整 Tier A+ 后才允许申请 P1 co-primary 和不超过 1M 参数的 EPRO-Lite；仍需 learning curve 与新的 active authorization。Tier A+ 本身不授权训练或科学 claim。

### 11.3 失败路由

- 低于 Tier B+：data/provenance resource + leakage-resistant benchmark；
- Tier B+ 但低于 Tier A+：P2 小模型路线；P1 和大于 250k 方法 headline 禁止；
- Tier A+：P2 + 可申请 P1/不超过 1M，但仍受 PH0-X/B0-X/O0-X；
- 任一 exact/noise/test/exposure 条件 FAIL：不通过改名或降阈值回到模型路线。

---

## 12. 双 estimand、endpoint 与输出

### 12.1 Estimand A：可重复 changer detection/ranking

目标是对 exact single-mutant pair 估计：

\[
P(C_i=1\mid x_w,x_m,e,c,r_w),
\]

其中 C_i 只能由冻结的 replicate/control-aware caller 定义。主要用途是每个 parent 的固定筛选预算下排序。

caller 的统计对象固定为 pair。对每个 pair，先在 primary eligible position mask 上计算 control-standardized Delta，并以预定义 contiguous/contact regions 的 max-cluster statistic 形成 T_i；cluster null 只来自 training controls/replicates。pair 内用 frozen max-cluster correction，study 内对 pair-level p values 使用 Benjamini-Hochberg FDR q=0.05。C_i=1 当且仅当 corrected pair decision 为 changer。caller 同时输出独立于模型的 significant-position mask Q_i，供 conditional magnitude 评分；模型不能修改 C_i 或 Q_i。

主要指标：

- AUPRC，同时报告 held-out prevalence、random prevalence baseline 和 strongest baseline；
- 每个 parent 固定 k 或固定预算下的 top-k recall、precision、enrichment；
- Brier score、log loss、calibration slope/intercept、reliability curve；
- risk-coverage、selective error、OOD abstention；
- local/contact/distal 与 probe/condition/study strata。

不得只报告 AUROC；高度稀疏时必须把 AUPRC 与 prevalence 并列。

固定筛选主预算为每 parent K=10。只有 candidate 数 n>=20 且 caller-defined changer 数至少 3 的 parent 可计算 primary top-k；n<K 或 zero/insufficient-positive 均标为 UNSCORABLE_TOPK，不填零、不删除。top-k recall/enrichment 先按 parent 计算，再 parent-macro、study-macro 报告；至少 80% held-out parents 且至少 80% held-out pairs 位于 scoreable parents，否则该指标 Gate FAIL。K=5 可作为预注册 sensitivity，不得取代 K=10 主指标。

### 12.2 Estimand B：changer 条件下 signed magnitude

仅在独立 caller 定义、且与模型预测无关的 changer 上评估：

- raw-scale WMAE/MAE；
- sign accuracy；
- Spearman；
- Student-t NLL 与区间覆盖；
- local/contact/distal 位置图；
- study/parent/domain 分层。

conditional magnitude 的正式张量是 Q_i 上的 masked position vector；position-wise WMAE/Student-t NLL、sign accuracy 和 Spearman 只在 Q_i 上计算。pair scalar 仅作筛选摘要：A_i 为 Q_i 上 inverse-variance weighted mean absolute Delta，S_i 为同权重 signed mean，direction dominance D_i=绝对加权和除以加权绝对和。D_i>=0.5 时 direction=sign(S_i)，D_i<0.5 时 direction=MIXED_SIGN，scalar sign accuracy 标为 UNSCORABLE_MIXED_SIGN。signed_magnitude_mean 等于 S_i；position_delta_mean 是向量，不得用 scalar 替代位置图。全位置 continuous endpoint 使用更大的 primary eligible mask，而不是 Q_i。

若 changer caller 本身不可靠，Estimand A 与 B 都不能继续作 primary；若 caller 可靠但 magnitude reliability 单独不足，则只有本 estimand 自动降为 secondary。不得用模型挑出的“高响应”子集恢复 primary。

### 12.3 共同次级 endpoint：全位置连续 Delta

保留全位置、原始尺度的 MSE、MAE、WMAE、Student-t NLL、位置图和相对 reference 的绝对 loss difference。它不再独占模型选择。

primary mask 必须排除 edited site、alignment-changed、probe-eligibility-changed、missing/invalid/unmeasured 位置。empty mask pair 必须有 exclusion reason，不能产生 0 loss。

### 12.4 输出 schema

每个 candidate 至少输出：

~~~yaml
pair_id: string
changer_probability: number
changer_decision: CHANGER_NONCHANGER_ABSTAIN
signed_magnitude_mean: number_or_null
signed_magnitude_interval: [low, high] or null
position_delta_mean: array_with_mask
position_delta_interval: array_with_mask
uncertainty_status: CALIBRATED_UNCALIBRATED_NOT_AVAILABLE
ood_score: number_or_null
abstention_reason: controlled_enum_or_null
support_domain_status: IN_DOMAIN_OOD_UNKNOWN
~~~

uncertainty 未校准时不得用 confidence、可信区间或风险保证语言。

---

## 13. 统计与 evaluator 合同

### 13.1 推断单位与重采样

主推断层级是 study → parent → pair。nucleotide position 不是独立统计样本。bootstrap、permutation 与 paired comparison 必须保持 cluster 和 shared-WT block。

主比较使用 cluster-level paired loss difference：

\[
d_g=L_g(\text{main})-L_g(\text{comparator}),
\]

并报告跨 study/parent clusters 的点估计、95% CI、cluster 数、pair 数和 changer 数。CI 方法、重采样层级、次数与 seed 在 test 解封前冻结。

### 13.2 Skill

ratio Skill 只能用 pooled ratio-of-sums：

\[
\mathrm{Skill}_{pool}=1-
\frac{\sum_i w_i L_i(\hat y_i,y_i)}
     {\sum_i w_i L_i(y_{ref,i},y_i)} .
\]

当 reference loss 低于 training controls 推导的 noise floor 时，该 block 标为 UNSCORABLE_RATIO；仍必须报告绝对 loss difference。禁止对 near-zero pair 的 ratio 直接宏平均。

### 13.3 group-aware permutation

label 或 edit 的 permutation 必须在冻结的合法 exchangeability block 内进行，保留 study/parent/shared-WT/mask 结构。真实标签结果至少要优于 group-aware permutation null；否则停止 method claim。

### 13.4 confirmatory 主张条件

方法主张至少同时要求：

1. 主任务真实标签表现优于 group-aware permutation null；
2. 相对 strongest simple baseline 的 cluster CI 下界大于 0（按“baseline loss - main loss”方向定义）；
3. 相对 capacity/compute matched generic baseline 的 cluster CI 下界大于 0；
4. signed magnitude 点估计至少 5% 相对改善；若该 estimand 因 reliability 降级，不得保留 magnitude headline；
5. 结果不由一个 study 或 parent 驱动，leave-one-study/parent sensitivity 方向稳定；
6. calibration、risk-coverage 与 OOD failure 均如实报告；
7. model、threshold、pretraining、split、metrics、baseline、tie-break 在新 test 前全部冻结。

不满足上述条件可以形成 resource/benchmark/P2 negative result，但不是 EPRO 方法成功。

### 13.5 唯一 development selection rule

在任何 validation prediction 前冻结以下顺序。首先剔除 run closure/invariant/reliability FAIL、Brier 或 log loss 劣于 strongest baseline、或预注册 risk-coverage 约束不满足的 candidate。其余候选的第一选择量是 study-macro AUPRC gain over strongest baseline：每个 scoreable study 等权，study 内由 parent/pair predictions 计算，CI 用 study→parent cluster bootstrap。处于最佳值 one-standard-error 范围内时依次选择：参数更少者、parent-macro K=10 top-k recall gain 更高者、Brier 更低者、log loss 更低者；仍相同时按预先登记的 model_id lexical order，禁止看 magnitude/continuous/test 决定。conditional magnitude 与 all-position continuous 只作共同报告/科学 Gate，不得反向推翻 changer-primary selection。

---

## 14. Baseline 与容量阶梯

必须按容量从低到高执行，前一级未冻结/未报告，不得直接跳到后一级：

1. zero、train mean、mutation-type、edit-only、WT-only；
2. thermo-only linear/ridge 与简单 tree；
3. 10k-100k 参数 P2 paired baseline；
4. 50k-250k 参数 generic paired 与 EPRO-Small；
5. 只有 Tier A+、P2 可学习、learning curve 支持时，才允许不超过 1M 参数 EPRO-Lite；
6. 历史约 4.46M 参数版本不能继承为 formal candidate。

strongest simple baseline 由 validation protocol 预先定义的比较集合和 tie-break 选出，不能在 test 后重选。

matched generic 与 EPRO-Small 必须共享：

- exact records、split、mask、input information、heads；
- optimizer family/budget、batch/steps、early stop、selection metric；
- from-scratch/pretraining arm；
- evaluator 与 bootstrap/permutation；
- 参数量和主要计算预算均在 ±5%。

无法证明匹配时比较状态是 UNMATCHED_DEVELOPMENT_DIAGNOSTIC。

---

## 15. EPRO exact endpoint-response 算子

### 15.1 输入与禁止输入

正式算子输入具体 x_w、x_m、edit、condition。P2 可额外读取 WT profile；P1 不读取任何实验 response。mutant profile、test target、caller post-hoc label、held-out normalization statistics 都是 forbidden inputs。

alt=X 三 alt 平均不是 exact endpoint-response EPRO。

### 15.2 必须由参数化保证的不变量

1. Identity：x_m=x_w 且 no-edit 时输出严格为 0；FP32 max_abs error 小于 1e-7；
2. P1 full endpoint-swap antisymmetry：交换 WT/mutant endpoint 与 edit 方向时输出符号翻转，FP32 max_abs error 小于 1e-6；
3. P2 conditional sequence/edit antisymmetry：因 P2 只允许 r_w、没有合法 r_m，不能声称 full physical endpoint swap。固定同一 r_w，sequence/edit component 必须采用 H(r_w,x_w,x_m,e,c)-H(r_w,x_m,x_w,e_inverse,c) 的构造，交换 sequence/edit 后符号翻转，max_abs error 小于 1e-6；任何需要 r_m 的 swap test 禁止执行或用于 claim；
4. Forcing support：edit forcing 只从合法 mutation support 注入，mask 外直接 forcing 的 max_abs 小于 1e-7；
5. deterministic eval：同一 checkpoint/input/device 重复 eval 必须 bitwise equal；若底层确定性 kernel 只保证数值容差，则 max_abs 小于 1e-8 且原因/版本进入 manifest；
6. stability：传播算子使用确定性 bound，rho_max 不得超过 0.98；
7. solver：每个样本记录 residual、iteration、convergence status，relative residual 小于 1e-5；
8. probe observation：在冻结 probe/domain 内，observation map 对 latent accessibility 单调不减，容差 1e-7，probe-specific parameters/version 必须绑定；
9. no permanent zero-gradient：初始化不得使早期必需层永久零梯度。

仅用 soft loss 鼓励 identity/swap 不足以通过 O0-X；构造与测试都必须成立。

### 15.3 分头建模

- changer head：概率、Brier/log loss/calibration；
- conditional signed magnitude head：Student-t location/scale/df，只有 caller-defined changer 参与对应条件 likelihood；
- all-position continuous head：共同次级 raw-scale response。

三个 head 的 loss、metric、gradient norm、mask count 分开记录。不得用真实 effect 大小权重制造 active signal。

### 15.4 O0-X 工程必测

- identity、swap、forcing support property tests；
- deterministic eval，包括禁用随机 power iteration 排名；
- deterministic spectral bound 与 solver residual；
- 8-32 pair tiny-subset overfit：训练误差低于 constant baseline 的 1%；
- 每个科学参数 block 梯度有限且非永久零；
- effective batch size、mask、edited-site exclusion；
- model/input/target/forward/backward device；
- P2 mutant-profile access audit 的 read count 必须为 0；
- probe observation monotonicity 与第 15.2 节全部数值阈值；
- evaluator 与独立 reference implementation 对拍；
- NaN/Inf、empty mask、long sequence、all-nonchanger edge cases。

tiny-subset overfit 失败先判工程 FAIL，不得通过超参搜索把它写成科学困难。

---

## 16. 静态预训练双臂

### 16.1 Arm 定义

- Arm A：from-scratch；
- Arm B：Ribonanza、旧 ReactFlow-Structure 或其他静态 structure/reactivity 预训练。

预训练 profile 永远不计入 Delta pair 数。两臂共享同一 exact Delta data、split、mask、downstream capacity、optimizer budget、evaluator、model selection 与 test。

### 16.2 exposure ledger

Arm B 必须审计：

- exact sequence；
- parent/design lineage；
- RNA family/clan；
- structural similarity；
- study/source deposit；
- test/validation 的 label-derived transform 或 pseudolabel；
- checkpoint/representation 的训练来源与 license。

任一 validation/test overlap 或 contamination unknown 时，Arm B 只能 secondary，不得选择 headline candidate；Arm A 保持主对照。test 不得决定采用哪一臂。

预训练收益只能表述为“在冻结 Delta benchmark 上，相对 from-scratch 的增量”；不得把直接/近重复 exposure 当作 representation learning。

---

## 17. 受控 EPRO development window

window 只有 D2-X、PH0-X、B0-X、O0-X 所有必需 Gate PASS，active manifest 明确授权 M0-X，且 sealed test 仍未访问时才能开启。

硬规则：

1. 最长 28 个 calendar days；
2. 最多 6 个科学迭代 EPRO_DEV_01 至 EPRO_DEV_06；
3. 时间或次数任一先到即关闭；
4. 每轮一个预注册科学假设、固定同一个 development seed；
5. 每轮唯一 run_id 和 parent_run_id；失败 artifact 不覆盖；
6. 任何改变 prediction 的 data eligibility、caller/target、loss、feature、architecture、capacity、initialization、optimizer、calibration、checkpoint selection 都计一轮；
7. infrastructure-only retry 只有在 scientific inputs byte-identical、故障 evidence 完整且新 run ID 时可不计科学迭代；
8. reliability/permutation/provenance Gate 失败，或连续 3 轮不能优于 strongest simple baseline，立即早停；
9. sealed test 在第 6 轮结束并冻结唯一 final candidate 前不可见；
10. 关闭后禁止换 seed、扩大模型、增加第 7 轮、换 test 或改名重启窗口。

若最终 Gate 不通过，自动转为 data/provenance resource、leakage-resistant/negative benchmark 或 P2 result；不再扩大 EPRO。

window registry 必须机器可读，至少包含 window_id、authorization_sha256、window_started_at_utc、window_deadline_utc、maximum_iterations=6、consumed_iterations、remaining_iterations，以及每轮 run_id/parent_run_id/hypothesis_id/change_category/prediction_changing/counts_as_iteration/evidence_sha256/status。window_started_at 是首个已授权 prediction-changing run 的 UTC 时间，deadline 精确等于其后 28 个 calendar days；finalizer 取时间与计数两个 stop 条件的先到者，不允许人工重置。

---

## 18. GPU、进程、监控与 Git 安全

### 18.1 正式 learned run

- 必须真实 CUDA；记录 GPU index 与 UUID；
- 启动后验证 model、input、target、forward/backward 全在预期 CUDA；
- 记录 runtime CUDA forward/backward count 与 CPU fallback count；
- formal learned run 要求 fallback count=0；仅 nvidia-smi 不足以证明；
- CUDA/OOM/driver failure fail closed，不自动 CPU；
- 禁止抢占、杀死、暂停或修改无关用户/项目进程。

### 18.2 启动前 read-only preflight

记录 HEAD/branch/upstream、tracked/untracked/dirty state、active process 与 /proc identity、GPU、disk/quota、artifact root、contract/data/split/config hashes。工作树脏或 source origin 不可绑定的训练只能 DEVELOPMENT_ONLY。

### 18.3 长任务监控与 safe stop

- metrics/log/checkpoint 路径必须在启动前记录；
- 低频、只读、事件驱动监控；
- NaN/Inf、资源异常、无进展或连续五次 validation 不变触发暂停诊断；
- runner 必须实现 checkpoint-aware SIGTERM、finally finalizer 与 stop reason；
- 终止前再次核对 uid/PID/start tick/cwd/exe/argv/log fd/GPU UUID；
- 禁止 pkill、killall、进程组广泛终止和 SIGKILL，除非另有用户明确紧急授权。

### 18.4 Git

- 原始数据、checkpoint、模型权重、cache、secret、完整日志不进 Git；
- 每个实质阶段经验证后 focused commit，只含本阶段文件；
- 保留用户原有 dirty/untracked 文件；
- 默认不 push、不 PR；
- 失败 run 不删除、不覆盖、不改标签制造 PASS。

---

## 19. 未来 run manifest 与 terminal closure

每个新 run manifest 强制包含：

~~~yaml
run_id: unique
parent_run_id: string_or_null
phase_id: canonical
hypothesis_id: string
evidence_class: enum
contract_conformance: enum
claim_eligibility: enum
active_contract:
  path: repo_relative
  sha256: sha256
  authorization_status: enum
git:
  commit: sha1
  source_origin_inventory: object
  worktree_status: enum
config:
  requested_path: path
  resolved_sha256: sha256
data_manifest_sha256: sha256
split_manifest_sha256: sha256
exposure_ledger_sha256: sha256
test_access_ledger_sha256: sha256
pretrained_checkpoint_sha256: sha256_or_null
software:
  driver_version: string
  cuda_runtime_version: string
  framework_version: string
  library_lock_sha256: sha256
runtime:
  argv: array
  cwd: absolute_path
  seed: integer
  pid: integer
  process_start_ticks: integer
  started_at: rfc3339
  ended_at: rfc3339_or_null
  target_device: cuda
  requested_batch_size: integer
  effective_batch_size: integer
  gradient_accumulation_steps: integer
  optimizer_update_count: integer
  precision: fp32_bf16_or_other_frozen
gpu:
  uuid: string
  model_device: cuda
  input_device: cuda
  forward_calls: integer
  backward_calls: integer
  fallback_count: integer
  max_memory_allocated_bytes: integer
  preflight_snapshot_sha256: sha256
artifacts:
  logs: refs
  metrics: refs
  checkpoints: refs
  evaluator: ref
  invariant_audit: ref
  finalizer: ref
  checksum_ledger: ref
  terminal_sentinel: ref
stop_reason: controlled_enum
exit_code: integer_or_null
signal: string_or_null
~~~

环境只记录 allowlist；token、key、cookie、credential 不得写 manifest/log。

B0-X learned baselines、O0-X tiny-overfit、M0-X 与任何正式 learned run 都适用相同的真实 CUDA、fallback=0、software/batch/precision、manifest、finalizer、checksum 与 sentinel 规则；不得把 baseline 或 engineering test 作为例外。

### 19.1 runner fail-closed 绑定

在模型或 protected data 加载前验证：

1. active contract path/hash/status；
2. requested phase 在 allowed_phases；
3. training_allowed 与 phase training flag；
4. Git/source origin；
5. data/split/config/exposure/pretraining hashes；
6. test access ledger；
7. phase lifecycle 不是 TERMINAL、rerun_allowed=true、execution_authorized=true；terminal phase 即使仍出现在陈旧 allowed list 也必须拒绝；
8. artifact root 与 run ID 唯一。

任一 null、missing、drift、duplicate authority 或权限不符直接退出，并生成 preflight failure artifact。

### 19.2 formal PASS 的闭环

exit code 0、checkpoint、GPU execution 或 unit tests 不能单独成为 PASS。只有 finalizer 写出：

- terminal manifest；
- complete metrics/evaluator/invariant audit；
- artifact inventory；
- checksum ledger 自检 PASS；
- phase-specific terminal sentinel；
- acceptance report 每项 evidence ref；

才可把 lifecycle 写 TERMINAL，并按合同判断 Gate。finalizer 失败时保留所有 partial evidence，Gate 不是 PASS。

---

## 20. Phase graph 与 phase contracts

### 20.1 Canonical graph

~~~text
PILOT_CLOSURE ─┐
               ├→ D0-X → D1-X → D2-X → PH0-X → B0-X → O0-X
RECOVERY_CONTRACT_REWRITE ┘                                  ↓
                                                          M0-X
                                                            ↓
                                                          E0-X
                                                            ↓
                                                          P0-X

D0-X / D1-X / D2-X / PH0-X / B0-X / O0-X / M0-X / E0-X
                         └── terminal FAIL ──→ REPORT-X ──→ STOP
~~~

所有正式 phase 还隐含依赖 active manifest 对该 phase 的明确授权。FAIL、UNKNOWN_NOT_ASSERTED、NOT_RUN 都不能解锁 downstream。上图中的 REPORT-X 不是失败后自动运行：它必须看见一个列明的 upstream phase 已 `TERMINAL / FAIL`、绑定该 failure manifest，并获得单独 authority amendment；它不得下载数据、训练、访问新 test 或修改失败结果。

| Phase | 当前 lifecycle / result | evidence class | training |
|---|---|---|---|
| RECOVERY_CONTRACT_REWRITE | RUNNING / NOT_RUN（等待用户签署） | ENGINEERING_ONLY | no |
| PILOT_CLOSURE | TERMINAL / PASS | DEVELOPMENT_ONLY | no |
| D0-X | PLANNED / NOT_RUN | DATA_QUALIFICATION_ONLY | no |
| D1-X | PLANNED / NOT_RUN | DATA_QUALIFICATION_ONLY | no |
| D2-X | PLANNED / NOT_RUN | DATA_QUALIFICATION_ONLY | no |
| PH0-X | PLANNED / NOT_RUN | DATA_QUALIFICATION_ONLY | no |
| B0-X | PLANNED / NOT_RUN | BENCHMARK_QUALIFICATION_ONLY | yes when separately authorized |
| O0-X | PLANNED / NOT_RUN | ENGINEERING_ONLY | yes, tiny engineering tests when separately authorized |
| M0-X | PLANNED / NOT_RUN | DEVELOPMENT_ONLY | yes |
| E0-X | PLANNED / NOT_RUN | CONFIRMATORY_ELIGIBLE | evaluation only |
| REPORT-X | PLANNED / NOT_RUN | ENGINEERING_ONLY | no |
| P0-X | PLANNED / NOT_RUN | CONFIRMATORY_ELIGIBLE | no |

每个 phase manifest 还必须显式列出 automated_tests 与 manual_audit 两个对象：命令/版本/exit code/报告 hash、审阅者角色、抽查范围、发现与 disposition。任一对象缺失或未闭合时不得 PASS。下列 phase card 中的 required outputs 均隐含该 acceptance checklist、finalizer、checksum ledger 与 phase-specific sentinel，不得用一段 prose 代替。

### 20.2 RECOVERY_CONTRACT_REWRITE

- 目标：闭合旧 authority 冲突，收口 pilot，交付 V4/manifest/interface。
- 假设：无需运行数据或模型即可建立单一、可审计、fail-closed authority。
- 依赖：无；PILOT_CLOSURE 可并行但必须在本 phase 完成前闭合。
- 唯一输入：冻结的 V3 系列、mRNA base contract、只读 repo/pilot snapshot、用户批准的 V4 决策。
- 允许：read-only audit、精确 pilot closure、文档/manifest/interface、focused governance commit。
- 禁止：full recall、新 dataset/split、baseline/P2/EPRO、confirmatory access、push/PR。
- 输出：本文、active manifest、mRNA interface、pilot report/closure pointer、hash/validation report。
- PASS：用户对本文 exact raw-byte SHA-256 明确签署，且本轮全部 delivery acceptance tests、finalizer、terminal manifest、checksum ledger 与 closure sentinel 均 PASS。签署是必要但不充分条件；此前保持 RUNNING/NOT_RUN。
- FAIL：V3 hash 改变、local/remote V4 不同、manifest 多 authority/不可解析、training=true 或 Git diff 越界。
- NOT_RUN：交付物尚未形成；UNKNOWN_NOT_ASSERTED：某来源/旧状态不可核实。
- stop rule：在 DRAFT_AWAITING_USER_SIGNOFF 停止；不得自动进入 D0-X。
- evidence class：ENGINEERING_ONLY。
- PASS 后终态：STOP_AWAIT_D0X_AUTHORITY_AMENDMENT。唯一可申请下一 phase 为 D0-X，且需新 active amendment；不得把本 phase PASS 写成 D0-X 已授权。

### 20.3 PILOT_CLOSURE

- 目标：只对核验后的 1509/7660 child PID 做安全收口并保全 evidence。
- 假设：现有 best checkpoint/log 可在不干扰其他作业的情况下保存。
- 依赖：用户明确批准本 closure；不依赖旧 M0 Gate。
- 输入：/proc/GPU/Git/config/log/checkpoint read-only snapshot。
- 允许：copy/hash、双重 exact identity guard、SIGTERM-only child PID、post-check、closure manifest/ledger。
- 禁止：SIGKILL、pkill/killall、parent/process-group signal、续训、把结果变成 V4 lineage。
- 输出：canonical closure root、DEVELOPMENT_CLOSED、final SHA256SUMS、tracked small report。
- PASS：目标 PID 双重 guard PASS、SIGTERM 后 /proc/GPU 均无目标、checkpoint/log hash、manifest/sentinel/final ledger 全部闭合。
- FAIL：identity drift、signal scope 不明、copy/hash 失败或可能影响无关作业；此时保留运行或 preserved failure。
- NOT_RUN：未尝试；UNKNOWN_NOT_ASSERTED：exit code/runtime source 无法取得，必须明确 NOT_AVAILABLE_NOT_ASSERTED。
- stop rule：closure 后不重启、不继续训练。
- evidence class：DEVELOPMENT_ONLY；PASS 只表示 closure Gate。
- 当前 evidence：docs/audits/reactflow_delta_pilot_closure_20260803.md。
- 下一状态：STOP_AWAIT_USER_SIGNOFF。

### 20.4 D0-X Exact Mutation Recovery

- 目标：对冻结 source universe 做 accession/profile-level exhaustive recall。
- 假设：public archives 中可能存在足以重建 exact single-mutant endpoints 的未充分利用记录。
- 依赖：RECOVERY_CONTRACT_REWRITE PASS、PILOT_CLOSURE PASS、新 D0-X authority amendment。
- 输入：source-universe manifest、license policy、raw artifact root、parser fixture。
- 允许：下载/镜像/parse/crosswalk/人工抽查；不生成训练 split。
- 禁止：filename-only filter、未查写零、construct count 充 pair、模型训练。
- 输出：raw inventory、per-accession disposition、download ledger、parser coverage、candidate profile table。
- PASS：source universe 100% 有 disposition；raw license/version/hash；parser fixture 与 silent-drop tests PASS；field retention/audit 完整。
- FAIL：source universe 漏项、silent parse loss、许可/版本/hash 不可审计、未检查项被写零。
- NOT_RUN：本轮固定状态。
- stop rule：FAIL 只可申请 REPORT-X 证据包装或新的 data-repair authority；不得进入 D1-X。
- evidence class：DATA_QUALIFICATION_ONLY。
- PASS 解锁：D1-X。

### 20.5 D1-X Exact Canonicalization and Cleaning

- 目标：把 candidate profiles 转为可逆、角色明确、噪声可追踪的 canonical records。
- 假设：经过 profile-level mutation/sequence/condition 解析后，可分离 exact primary 与 auxiliary/rescue/static/external。
- 依赖：D0-X PASS。
- 输入：D0-X frozen raw/candidate manifests、cleaning rule version、condition policy。
- 允许：exact mutation reconstruction、lineage/condition match、mask、dedup、noise/control linkage、tail audit。
- 禁止：missing→0、alt=X→primary、effect-based filtering、test-aware normalization、模型训练。
- 输出：canonical table、role/exclusion manifests、source-to-canonical maps、tail audit、noise/control inventory。
- PASS：primary exact/condition/lineage/profile pointer 100%；每个 non-primary 有 reason；三 reactivity layers/hash；outer 1% 可逐点回 raw；all parser/retention tests PASS。
- FAIL：不可逆坐标、静默丢 profile、错误角色、tail 不可追溯或清洗依赖 held-out outcome。
- NOT_RUN：无 D0-X PASS 时。
- stop rule：terminal FAIL 路由 REPORT-X；任何 parser/cleaning 新 build 都需新 authority/run lineage，旧失败 immutable。
- evidence class：DATA_QUALIFICATION_ONLY。
- PASS 解锁：D2-X。

### 20.6 D2-X Split, Exposure and Data-Tier Candidate Freeze

- 目标：建立 leak-resistant split，冻结 test/exposure，并生成不提前消费 test label 的 data-tier candidate。
- 假设：D1-X 数据可能满足 Tier 的数据规模/完整性前提，且存在真正未消费 test。
- 依赖：D1-X PASS。
- 输入：D1-X canonical/noise manifests、study/parent/design/family/structure crosswalk。
- 允许：hierarchical split、exposure audit、Tier calculation、test seal。
- 禁止：按 response/model score 分 split、重用旧 test、混入非-primary Tier count。
- 输出：split manifest、test seal/access ledger、exposure ledger、blind viability certificate、TIER_B_PLUS_DATA_CANDIDATE/TIER_A_PLUS_DATA_READY checklist、data card。
- PASS：数据规模、exact/lineage/condition/noise pointer、overlap、independent-test aggregate 等 D2 可判项有 value/evidence；overlap=0；test 样本级标签 untouched；hash 全冻结。此 PASS 不是完整 Tier B+/A+。
- FAIL：低于 data candidate 门槛、新 test 不存在、blind certificate 不可实现、overlap/contamination unknown 或任一 D2 合取条件不满足。
- NOT_RUN：本轮状态。
- stop rule：FAIL 路由 REPORT-X；不得通过阈值变更恢复，也不得解锁 PH0-X。
- evidence class：DATA_QUALIFICATION_ONLY。
- PASS 解锁：PH0-X；完整 Tier B+ 由 PH0-X 判定，完整 Tier A+ 由 B0-X 后判定。

### 20.7 PH0-X Identifiability and Reliability

- 目标：确认 changer 与 magnitude 在 controls/replicates 下可识别。
- 假设：真实响应超过 matched measurement noise，并可在 train-only caller 下复现。
- 依赖：D2-X PASS。
- 输入：frozen split、controls/replicates、shared-WT blocks、caller preregistration。
- 允许：noise/ICC/reliability、caller freeze、group-aware permutation、measurement-error sensitivity。
- 禁止：outcome percentile noise、test threshold、模型生成 label。
- 输出：noise manifest、caller manifest、reliability report、changer counts、permutation report。
- PASS：matched noise coverage 与 Tier changer counts 满足；caller reliability 冻结；真实 signal 优于合法 permutation null；无单 study 驱动；由此将 TIER_B_PLUS 从 candidate 判为 PASS，并可维持 TIER_A_PLUS_DATA_READY。
- FAIL：noise 不可识别、reliability/permutation 失败或 changer 数不足。
- NOT_RUN：当前状态。
- stop rule：FAIL 立即路由 REPORT-X；magnitude 与 changer primary 同时停止，只能保留 all-position continuous descriptive endpoint，不得伪称 primary PASS。
- evidence class：DATA_QUALIFICATION_ONLY。
- PASS 解锁：B0-X。

### 20.8 B0-X Strong Baseline Qualification

- 目标：建立 zero 到 small paired model 的最强简单 benchmark 与 frozen evaluator。
- 假设：若 exact Delta 有可学习信号，小容量 P2 至少能超越 trivial/permutation baseline。
- 依赖：PH0-X PASS、至少 Tier B+。
- 输入：frozen train/validation split、caller、metrics、capacity ladder。
- 允许：zero/mean/mutation/edit/WT/thermo/ridge/tree 与 10k-100k P2；若预授权可做 50k-250k generic qualification。
- 禁止：EPRO method optimization、sealed test、换 seed 重试。
- 输出：baseline registry、matched budgets、predictions/metrics、cluster CI、learning curve、strongest baseline freeze。
- PASS：所有 baseline 运行闭合；至少一个 nontrivial P2 在冻结 validation 上优于 group-aware permutation 与 strongest trivial baseline，CI/敏感性满足预注册门槛。若 TIER_A_PLUS_DATA_READY 同时成立，此 PASS 才能把完整 TIER_A_PLUS 判为 PASS。
- FAIL：无可学习信号、结果由单一 group 驱动、evaluator/matching 不合格。
- NOT_RUN：当前状态。
- stop rule：FAIL 路由 REPORT-X 形成 benchmark/negative result；不以大 EPRO 越过。
- evidence class：BENCHMARK_QUALIFICATION_ONLY。
- PASS 解锁：O0-X。

### 20.9 O0-X Operator Engineering

- 目标：证明 exact endpoint-response 实现满足数学与训练工程不变量。
- 假设：EPRO-Small 可在不使用 test/mutant profile 下稳定优化。
- 依赖：B0-X PASS。
- 输入：frozen operator spec、8-32 pair train-only fixture、reference evaluator。
- 允许：property/unit/tiny-overfit tests；不做科学模型选择。
- 禁止：sealed validation/test consumer、把 unit PASS 写方法成功。
- 输出：identity/swap/forcing/stability/determinism/gradient/batch/mask/device/evaluator reports。
- PASS：第 15.4 节每项测试 PASS，真实 CUDA forward/backward 和 fallback=0，finalizer/ledger 闭合。
- FAIL：任一 invariant、tiny-overfit、gradient、device 或 determinism 失败。
- NOT_RUN：当前状态。
- stop rule：同一授权 O0-X 内的非终态工程修复使用新 parent-linked run；一旦 finalizer 写出 terminal FAIL，则路由 REPORT-X。O0-X PASS 前不得 M0-X。
- evidence class：ENGINEERING_ONLY。
- PASS 解锁：M0-X。

### 20.10 M0-X Controlled Development

- 目标：在 28 天/6 轮内比较 EPRO-Small、matched generic、from-scratch/pretraining arms。
- 假设：受约束 EPRO 在相同容量/预算下提供超越 strongest baseline 和 generic 的稳定增量。
- 依赖：O0-X PASS；Tier/capacity/pretraining permissions；active M0-X authorization。
- 输入：frozen development data/split/caller/evaluator/baselines/seed 与 sealed test pointer。
- 允许：每轮一个 preregistered hypothesis；第 17 节的有限更改。
- 禁止：test access、换 seed、覆盖失败、超 6 轮/28 天、超 capacity。
- 输出：iteration lineage、每轮 manifest/finalizer/ledger、frozen final candidate 或 negative close。
- PASS：在 development 上预注册主要 CI/置换/calibration criteria 满足，唯一 final candidate 与所有协议冻结。
- FAIL：reliability/permutation drift、连续 3 轮不胜、窗口耗尽或最终 criteria 不满足。
- NOT_RUN：当前状态。
- stop rule：FAIL 路由 REPORT-X，诚实保留 resource/benchmark/P2 证据；PASS 才可申请 E0-X 一次解封。
- evidence class：DEVELOPMENT_ONLY。
- PASS 解锁：E0-X。

### 20.11 E0-X One-Time Sealed Scientific Evaluation

- 目标：对新 untouched study/studies 做一次性 frozen evaluation。
- 假设：final candidate 跨 study/parent 泛化并在 screening、magnitude 和风险控制上优于强对照。
- 依赖：M0-X PASS、test seal 完整、用户显式 E0-X unseal authorization。
- 输入：唯一 frozen checkpoint、threshold、pretraining arm、baseline、split、metrics、CI/permutation、test access ledger。
- 允许：一次性 inference/evaluation/finalization；不允许训练。
- 禁止：看到任何 test 结果后改模型、阈值、calibration、reporting subset 或重跑选择。
- 输出：sealed predictions、cluster statistics、all strata、calibration/risk-coverage、failure analyses、final manifest/ledger。
- PASS：第 13.4 节全部适用主张条件通过；UNCERTAINTY_CALIBRATION 与 OOD_ABSTENTION 子 Gate 也有独立 PASS evidence。
- FAIL：任一主比较 CI、permutation、calibration/OOD、single-group sensitivity 或 closure Gate 失败。
- UNKNOWN_NOT_ASSERTED：无已知 FAIL，但任一 required result、子 Gate、artifact 或 exposure 状态不可核实；立即 fail closed，不解锁 P0-X，也不能进入仅接收 terminal FAIL 的 REPORT-X。
- NOT_RUN：当前状态。
- stop rule：test 只消费一次；FAIL 不重开，只可路由 REPORT-X 包装 negative benchmark/P2/resource 证据。
- evidence class：CONFIRMATORY_ELIGIBLE；只有结果 PASS 后才可能成为 confirmatory claim evidence。
- PASS 解锁：P0-X；不自动授权 mRNA。

#### 20.11.1 UNCERTAINTY_CALIBRATION Gate

- 目标：证明 changer probability 与 magnitude interval 具有可复查的概率语义；不是任意 confidence score。
- 输入：在 D2-X 时预先从 development groups 隔离、从未用于 architecture/checkpoint selection 的 calibration fold；frozen final candidate；sealed E0-X test 只用于一次评估。
- fit 规则：temperature/isotonic/variance scaling 的候选与 tie-break 在 calibration fold 内 nested、按 study/parent 分组拟合；不得在 sealed test 拟合或重选。
- 指标：Brier、log loss、calibration slope/intercept、fixed-bin ECE、90%/95% interval coverage/width、cluster bootstrap CI。
- PASS：sealed test 的 Brier 与 log loss 均不劣于 strongest calibrated baseline；slope 在 0.8-1.2、absolute intercept 不超过 0.1、ECE 不超过 0.05；90% interval coverage 位于 0.85-0.95，95% coverage 位于 0.90-0.98；所有阈值按 study/parent cluster 报告且无单组驱动。
- FAIL/UNKNOWN_NOT_ASSERTED：任一阈值失败、calibration fold 与 selection/test 暴露重叠、interval 通过无限加宽获得或 artifact/hash 不全。
- artifacts：calibration split/fit manifest、frozen transform、predictions、metrics/CI、finalizer、checksum、Gate report。
- stop：FAIL 不允许 cross-project producer；不能在 test 后重新校准。
- evidence class：CONFIRMATORY_ELIGIBLE，且不自动升级 E0-X 其他主张。

#### 20.11.2 OOD_ABSTENTION Gate

- 目标：在预定义外域中以冻结规则拒绝不可靠预测，并量化 coverage-risk trade-off。
- 输入：至少 2 个彼此独立的 OOD groups；每组至少 50 exact pairs、5 caller-defined changers，总计至少 100 pairs。group 可由 unseen study/probe-condition/family-structure 定义，但定义和优先级在任何目标 exposure 前冻结。
- fit 规则：OOD score、threshold 与目标 coverage 只在 training/calibration groups 拟合；不得读取 mRNA target、E0 sealed sample labels 或 external-eval outcome 调整。
- 指标：OOD AUROC/AUPRC、每组 accepted coverage、common-coverage selective risk、risk-coverage monotonicity、abstained/accepted calibration。
- PASS：OOD AUROC 至少 0.80；每组 accepted coverage 至少 50%；在共同 coverage 上主 loss 不高于 strongest abstaining baseline；coverage 降低时 risk 不得系统上升；所有 group 均满足且有 cluster CI。
- FAIL/UNKNOWN_NOT_ASSERTED：group/样本/changer 数不足、threshold post-hoc、任一组 coverage/risk 失败、只汇总后隐藏 failure。
- artifacts：OOD group manifest、exposure ledger、frozen scorer/threshold、per-group metrics、risk-coverage、finalizer/checksum/Gate report。
- stop：FAIL 保持 REACTFLOW_DELTA_ORACLE_NOT_AUTHORIZED；不得以总平均覆盖失败 group。
- evidence class：CONFIRMATORY_ELIGIBLE。

### 20.12 REPORT-X Fail-forward Evidence Packaging

- 目标：在 D0-X、D1-X、D2-X、PH0-X、B0-X、O0-X、M0-X 或 E0-X 出现 terminal FAIL 后，完整包装失败证据、可复现边界与诚实的 data/resource/benchmark/negative conclusion。
- 假设：科学 Gate 失败不妨碍形成可审计的资源、benchmark 或负结果，但报告完整性不能反向改变科学状态。
- 依赖：依赖图保持空以避免多入口环；运行时 entry condition 必须绑定上述至少一个 phase 的 `TERMINAL / FAIL` manifest、其 finalizer/checksum/sentinel，以及单独的 REPORT-X authority amendment。UNKNOWN_NOT_ASSERTED 或 NOT_RUN 不能冒充 entry condition。
- 唯一输入：失败 phase 的 immutable manifests、predictions/metrics（如有）、data card、exposure/test-access ledger、adverse-study matrix、limitations 与许可记录。
- 允许：只读聚合、claim-evidence matrix、失败复现说明、resource/benchmark/negative manuscript/data-card packaging。
- 禁止：下载新数据、重新清洗/分 split、训练或微调、重开 test、改阈值/模型/排除规则、隐藏失败 group，或把 REPORT-X PASS 写成 upstream science PASS。
- 输出：fail-forward report、完整 adverse/failure matrix、reproducibility inventory、claim ceiling、Data/Code Availability 草案、terminal manifest/finalizer/checksum/sentinel。
- PASS：entry condition 与 authority 均有效；每条结论绑定证据；失败与不确定项无遗漏；所有 artifact/hash/许可闭合。PASS 的语义仅为 `REPORTING_INTEGRITY_PASS`。
- FAIL：缺失 upstream terminal FAIL、选择性报告、claim 超证据、test/reanalysis 越权、复现或许可链不闭合。
- NOT_RUN：当前状态；UNKNOWN_NOT_ASSERTED：任一关键 failure/source/exposure 状态无法核实。
- stop rule：PASS 或 FAIL 后均停止并等待明确的新合同；不解锁 P0-X、任何模型 phase 或 mRNA producer authority。
- evidence class：ENGINEERING_ONLY。

### 20.13 P0-X Publication and Cross-Project Packaging

- 目标：按真实证据层级形成 data/resource、benchmark、P2 或 conditional EPRO 论文包。
- 假设：即使方法 Gate 不通过，完整 provenance 与 negative evidence 仍可形成可信产出。
- 依赖：仅 E0-X PASS。若 E0-X FAIL，只能进入 REPORT-X 形成诚实的 negative/resource report，不得把 REPORT-X 改名为 P0-X 或把其 PASS 传播为 E0-X PASS。
- 输入：所有 manifests、data card、exposure、stats、failure matrix、literature audit。
- 允许：report/manuscript/figure/data-card packaging；申请 cross-project producer 资格。
- 禁止：改数据/模型、补跑 test、选择性隐藏 study、无界 first/SOTA。
- 输出：claim-evidence matrix、reproducibility package、limitations、release checklist。
- PASS：每个 claim 绑定 evidence；所有 negative/adverse study 保留；Data/Code Availability 与许可正确；无泄漏或状态升级。
- FAIL：claim 超证据、复现 hash 不闭合、选择性报告或隐私/许可问题。
- NOT_RUN：当前状态。
- stop rule：等待明确 release/push/consumer authorization。
- evidence class：与 E0-X 结果一致，但 packaging PASS 本身不升级科学结果。

---

## 21. mRNA-EditFlow 跨合同 authority

canonical interface：

~~~text
docs/contracts/interfaces/ReactFlowDelta_mRNAEditFlow_authority_interface_v1.md
~~~

当前固定 marker：

~~~text
REACTFLOW_DELTA_ORACLE_NOT_AUTHORIZED
~~~

规则：

1. ReactFlow-Delta 的 data/test/training Gate 与论文 evidence 不自动继承给 mRNA-EditFlow，反向亦然；
2. 两项目不得共享 confirmatory test labels 或把对方 held-out exposure 写成未暴露；
3. 静态预训练跨用只有共同 exposure ledger 完整且无污染时才可申请；
4. producer 至少要求 E0-X、UNCERTAINTY_CALIBRATION、OOD_ABSTENTION 三个 canonical Gate 各自 PASS，具备 finalizer/manifest/checksum/export schema；
5. producer PASS 后仍需 mRNA consumer 的新用户签署 amendment；不得自动激活；
6. consumer 必须冻结唯一角色：GUIDANCE_CRITIC、SELECTOR 或 EVALUATION_ONLY_ORACLE；
7. 参与 guidance/selection 的组件不能同时是 independent final evaluator；
8. 任一方工程 PASS 不升级另一方科学状态；prediction 不是 measurement。

跨项目 run 启动前必须同时验证 producer active contract/export manifest 与 consumer active contract/import authorization。任一 hash、role、exposure 或 test boundary 不清立即 fail closed。

---

## 22. 可发表性路线、盲区与 claim 边界

### 22.1 固定优先级

~~~text
data/provenance resource
→ leakage-resistant benchmark
→ P2 variant ranking
→ conditional EPRO method claim
→ P1 sequence-only upgrade
~~~

路线由 Gate 决定，不由希望发表哪类论文倒推。

### 22.2 最大可发表性盲区

- 独立性：accession 数不等于独立 study；共享 lab/platform/WT/preprocessing 可显著降低有效样本量；
- shared-WT covariance：大量 mutant 共享一个 WT 时，pair 数会夸大 precision；
- annotation selection bias：有 exact-alt 元数据的 studies 可能系统性不同于缺注释 studies；
- label reliability：near-zero spike 与 changer prevalence 可能主要由 probe/batch/normalization 决定；
- condition heterogeneity：不同 probe/ligand/in vivo 状态可能不是一个可合并任务；
- external test scarcity：反复使用少量经典 RNA 会把 external eval 变成 development；
- calibration/OOD group 数不足：任意置信分数不能替代可校准 uncertainty；
- 无 wet lab：最终只能是 retrospective public-data claim，不能声称 prospective utility 或实验验证；
- novelty：bounded literature audit 只能限定 scope，不能证明无条件 first。

### 22.3 允许的负结果

若 exact data、noise reliability、新 untouched test 或模型主比较 Gate 不满足，允许的结论是：

> 在冻结的 source universe、清洗、split 和统计 Gate 下，当前已审计的公开证据不足以支持该指定主张。

禁止扩张为“RNA mutation response 原理上不可预测”。

### 22.4 明确禁止的语言

- “1,509/7,660 exact pairs”；
- 历史 Tier B 或 V3.3 training authorization 有效；
- 当前 pilot 证明 EPRO 收敛、泛化、优越或任务不可学；
- RMDB constructs/Ribonanza profiles 等于百万级 Delta supervision；
- Csde1/Tetrahymena 是 untouched test；
- outcome percentile 是 measurement noise ceiling；
- alt=X 是 exact endpoint；
- 未验证预算匹配却称 matched generic；
- nucleotide-level pseudo-replication 支持 cross-study generalization；
- retrospective prediction 是 structural truth、causal mechanism、wet-lab validation 或 prospective utility；
- 无 bounded evidence 的 first、SOTA 或发表保证；
- checksum/CUDA/unit test/checkpoint 是 scientific PASS。

---

## 23. 本轮必须通过的 delivery tests

1. 原 V3 raw SHA-256 仍为 3efcc1504208d8089236dfe4e7d41553741441d3b86b6174c8b5af52d614ec10；
2. 本文 local contract-directory copy 与 remote repo copy raw bytes/hash 完全相同；
3. active manifest 严格 YAML 可解析、拒绝 duplicate key，本文 hash 非空且匹配；
4. repository 只有一个 ReactFlow-Delta SINGLE_ACTIVE_AUTHORITY；
5. phase graph 必需节点完整、依赖无环、next route 合法；
6. formal phases 全是 PLANNED/NOT_RUN/unauthorized；
7. training/full recall/new split/confirmatory access/cross-project export/new wet lab 六项均为 false；
8. PILOT_CLOSURE 的 PASS 绑定 machine-readable manifest、DEVELOPMENT_CLOSED 与 final checksum；不得称 formal DONE；
9. interface 与本文/manifest 的 E0-X、UNCERTAINTY_CALIBRATION、OOD_ABSTENTION 和 unauthorized marker 一致；
10. mRNA base scientific内容仅 additive amendment，无其他改写；
11. focused diff 只有 V4、active manifest、interface、小型 pilot report；
12. Git 不含 raw data、checkpoint、weights、cache、secret、完整临时日志或 6 个预存 untracked 文件；
13. 只创建 local commit；不 push、不 PR；
14. 不执行 full-data download、dataset/split、baseline、P2 或 EPRO。

任何测试无 evidence 不得写 PASS。

---

## 24. V4 预先规定但本轮 NOT_RUN 的未来测试

### 24.1 数据/parser

- 以含 exact mutation:C13G、profile override、invalid token、multi-edit 的 RDAT fixture 验证 ref/alt 保真；
- source-universe accession coverage、silent parse failure、重复 raw/profile、mirror crosswalk；
- source-to-canonical 每字段 retention rate 与 controlled exclusion reason；
- outer 1% tail 每点 raw pointer；
- shared-WT、replicate/control 与 measurement variance；
- train-only normalization/caller，held-out target 不参与；
- study/parent/design-lineage overlap=0；
- family/structure/exposure contamination。

### 24.2 模型/evaluator

- 8-32 pair tiny overfit 小于 constant-baseline error 的 1%；
- identity、swap、forcing、deterministic eval、spectral stability、solver residual；
- early-layer gradient、effective batch、mask、edited-site exclusion；
- model/input/forward/backward CUDA 与 fallback=0；
- group-aware permutation、cluster bootstrap、pooled Skill、UNSCORABLE_RATIO；
- from-scratch/pretraining ledger 与同协议比较；
- calibration、risk-coverage、OOD abstention；
- finalizer、manifest、checksum ledger、terminal sentinel。

上述测试未执行前都为 NOT_RUN，不得在本文写成已通过。

---

## 25. 用户签署与后续 authority transition

当前 active status 固定为 DRAFT_AWAITING_USER_SIGNOFF，signed_by/signed_at/approval_record 均为 null。

用户审阅后如同意 V4，下一次动作不是直接训练，而是：

1. 创建 user approval record，记录本文 raw hash、批准范围与时间；
2. 重新只读 preflight Git/process/GPU/disk/source availability；
3. 创建 D0-X 专用 active amendment，allowed_phases 仅含 D0-X；
4. 绑定 authorized governance/source commit 与 source-universe manifest；
5. 仍保持 training_allowed=false；
6. 只执行 D0-X，完成后在新的用户审阅点停止。

D0-X 不能自动解锁 D1-X；后续每个阶段都依赖 machine Gate 与 active phase authorization。任何 binding 漂移都 fail closed。

---

## 26. 参考与 source-boundary

- RMDB（执行时须重新冻结版本）：https://rmdb.stanford.edu/
- RDAT format specification：https://rmdb.stanford.edu/deposit/specs/
- classSNitch primary paper：https://academic.oup.com/bioinformatics/article/33/11/1647/2953246
- Ribonanza primary record：https://pubmed.ncbi.nlm.nih.gov/38464325/

这些来源支持数据资源与格式的范围性描述，不自动证明某条记录满足 PRIMARY_EXACT_DELTA、独立 study、noise reliability 或 publication Gate。任何 novelty 结论需要在 P0-X 前做 bounded、dated、可复查的 primary-literature audit；本文不主张 first。

---

## 27. 当前终止声明

本文交付后必须停在：

~~~text
RECOVERY_CONTRACT_REWRITE
DRAFT_AWAITING_USER_SIGNOFF
training_allowed=false
REACTFLOW_DELTA_ORACLE_NOT_AUTHORIZED
~~~

不得执行 RMDB 全库下载、生成新训练 dataset、重建 split、运行 baseline/P2/EPRO、解封 test、push 或 PR。下一步只能由用户显式批准 V4 后，另行从 D0-X 数据 Gate 开始。
