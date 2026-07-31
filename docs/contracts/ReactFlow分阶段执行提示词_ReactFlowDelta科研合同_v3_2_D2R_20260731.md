# ReactFlow-Δ EPRO v3.2 D2-R 证据补齐审计增量科研合同

> 中文名称：ReactFlow-Δ 平衡态扰动响应算子科研合同 v3.2（D2-R 受限证据补齐审计授权增量）
> 合同版本：V3.2
> 基线合同：V3.0（`ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_EPRO_20260729.md`）+ V3.1（`..._v3_1_D1Cleanup_20260730.md`）
> 增量日期：2026-07-31
> 授权阶段：Phase D2-R（受限证据补齐审计，RSIB-v1 之后、任何 learned training 之前）
> 适用仓库工作树：`/home/cunyuliu/reactflow_delta_goal_20260729`
> 授权证据：D2 RSIB-v1 已执行（commit `fdbae41`）；`artifacts/reactflow_delta/d2/d2_tier_judgment.json` 记录 `true_pairs=0`、`outcome=below_tier_b_data_audit`、`binding_blocker=annotation_only_alt_not_verifiable (7761/7761)`

> **For Codex/Claude:** 本合同为 v3.0+v3.1 的增量，必须与基线合同一起阅读；v3.0/v3.1 未被本合同取代的部分继续有效。本合同**只授权一项受限审计操作**（§2），不授权训练、不授权 Tier 降阈、不授权任何从名称或邻近样本推断 alt 的行为。

---

## 0. 文档权威性、继承关系与生效条件（v3.2 增补）

### 0.1 v3.2 性质

本合同是 v3.0+v3.1 的**增量合同**，不取代基线的：

- 架构（v3 §3–§4）、物理与数学假设（v3 §5）；
- 数据优先级与 Tier 定义（v3 §6.1、§8）；
- 模型等级与阶段（v3 §4.9、§11）；
- Split、污染与 benchmark（v3 §9）；
- 强制基线、消融与负控制（v3 §10）；
- 测评与统计（v3 §12）；
- Fail-forward 合同（v3 §18）；
- 最终执行原则（v3 §20）；
- v3.1 全部 true_pair 升级规则（v3.1 §3）与 D1 Gate（v3.1 §4）。

v3.2 只授权**一个新阶段**：Phase D2-R 受限证据补齐审计。D2-R **禁止任何 learned training**、**禁止降低 Tier 阈值**、**禁止从构造名或邻近样本推断 alt**。

### 0.2 继承关系

v3.2 完整继承 v3.0 §0.2 全部冻结决策（含第 8 条"完成 D0–D2 数据 Gate 前禁止启动任何 learned training"）、§0.1 权威性条款、§6 数据与清洗合同、§8 数据可行性 Gate、§9 Split 与污染、§18 Fail-forward、§20 最终执行原则；并完整继承 v3.1 §2.2 全部禁止操作、§3 true_pair 升级规则、§4 D1 Gate、§6 Fail-forward 边界。

### 0.3 生效条件

- 本合同由**项目决策者审核发布后生效**；在此之前 `d2r_allowed = False`。
- 生效不等于授权训练；`training_allowed` 在本合同下**保持 `False`**，无论 D2-R 审计结果如何（见 §5）。
- 训练启动仍需 D2 重判达到 Tier B（v3 §0.2 第 8 条、v3.1 §7），且需独立授权。

### 0.4 决策者路径选择的记录

D2 Gate 完成后（commit `fdbae41`），项目决策者面临二选一：

| 路径 | 结论 | 边界 |
|---|---|---|
| A. 关闭模型路线 | 生成 D2 acceptance，状态 `complete_gate_not_passed_below_tier_b_data_audit`；归档数据审计/负结果 | 不证明科学假设为假；只证明当前公开证据无法支撑所需 true-pair 训练集 |
| B. 新建受限 D2-R 证据补齐审计 | 只查公开、既有 source/release/supplement 中是否有 per-profile 序列或同 parent 独立 replicate 佐证 | 不训练、不改 Tier、不从名称或邻近样本推断 alt；找不到即回到路径 A |

决策者**推荐路径 B**，但路径 B 的执行**必须以本合同生效为前置**。本合同预先固定 D2-R 的目标、成功条件、停止条件与 training 边界（§2–§5）。若 D2-R 审计在停止条件（§4）下未取得合格证据，则冻结 EPRO learned-model 路线，按路径 A 输出负结果。

---

## 1. D2 证据基础与结论定性（v3.2 增补）

### 1.1 D2 RSIB-v1 执行结果

来源：commit `fdbae41`；`artifacts/reactflow_delta/d2/` 下四个 artifact。

- **T-D2.1 lineage graph + parent lineage 验证**：7,761/7,761 候选完成"同一 RDAT 文件 + header reference"层面的**文件内 parent/reference 一致性核对**（`parent_lineage_verified = 同 rdat_sha256 AND ref_verified_against=="header_SEQUENCE"`）。
- **T-D2.2 overlap audit**：split group overlap 门控机制已就位；true_pair=0 无可分区对象，T-D2.3-5 split 冻结 deferred。
- **T-D2.10 Tier 判定**：`true_pair_count = 0/7761`；Tier A/B/C 全部 FAIL；`outcome = below_tier_b_data_audit`。

### 1.2 "parent lineage verified" 的解释边界（关键）

D2 产出的 `parent_lineage_verified = True` **必须**被解释为：

> **文件内 parent/reference 一致性已核对**（WT 与 mutant 同出一 RDAT 文件，且 annotated ref base 与 RDAT header SEQUENCE 字段一致）。

**不得**误读为：

> "已获得可升级 true_pair 的独立实验佐证"。

后者需要 per-profile 序列证据或同 parent 的独立 replicate（v3.1 §3.2），D2 的文件内一致性核对**不构成**该佐证。

### 1.3 D2 §3.2 修复的正确性

D2 修复了 D1 executor 的一个 §3.2 合规缺口：原先 285 条"有具体 alt 注释"的 annotation-only 记录在 D2 清除 `parent_lineage_unverified` 后会被误升格为 true_pair。修复后 annotation-only 候选一律需 per-profile 序列证据才能 `substitution_verified=True`，285 条回归 `annotation_only_alt_not_verifiable`。该修复是正确的，**不可回退**。

### 1.4 当前科学结论（须作为负结果保留）

> **瓶颈不再是 parser、清洗或 lineage 图，而是原始公开记录没有提供足以验证 annotation-only alt 的独立观测。**

全部 7,761 条候选均为 `audit_method=annotation_only_mutation_ref_verified_against_header`、`encoding_source=annotation`，既无 per-profile 序列，也无同 parent 的独立 replicate 佐证。这正是应当被保留的负结果，**不得**用推断补齐的空白。D2-R 的任务是**查证**是否有被遗漏的公开独立观测，**不是**制造或推断观测。

---

## 2. D2-R 授权范围（v3.2 增补）

### 2.1 唯一目标

D2-R 的**唯一目标**是：在公开、既有的数据源范围内，逐条查证是否存在可独立验证 D0-R v2 候选 annotation-only alt 的证据，具体为以下两类之一：

1. **per-profile 序列证据**：能把 annotation-only 候选的 alt（含 `alt=X` 的 M2-seq 类）解析为具体碱基，并在 per-profile 序列 vs WT anchor 上验证 pos/ref/alt 三者全部通过（v3.1 §3.1）；**或**
2. **同 parent 的独立 replicate 佐证**：同一 parent 的 replicate / no-edit / control（v3.1 §3.1 §3.2 意义下的独立佐证，仅有序列级 self-consistency **不构成**佐证）。

D2-R **不**重跑 D1，**不**重判 Tier，**不**训练，**不**修改任何 raw RDAT。

### 2.2 允许的操作（machine-readable）

D2-R 仅允许以下只读、forward-only 操作：

- 读取 D0-R v2 的 48 个 RDAT 文件（`rdat_tierA_20260730/`，checksum-verified，**只读**）；
- 读取 D0-R v2 relations 与 D2 lineage verification artifact；
- 查证 RDAT 文件本身是否携带此前未被解析的 per-profile 序列字段（如 `SEQUENCE`、`SEQPOS`、per-profile sequence annotation）；
- 查证 RMDB raw / source release / publication supplementary material（仅限 D0-R v2 已收录的 8 个 DOI / 6 个 owner / 31 个 rmdb_id 范围）是否公开提供 per-profile 序列或独立 replicate；
- 对每条候选记录查证结果（命中/未命中）及来源、checksum、provenance；
- 输出 D2-R 证据清单 manifest（§6）。

### 2.3 禁止的操作（machine-readable）

D2-R 期间**禁止**以下任何操作，违反即触发 fail-forward（v3 §18、v3.1 §6）：

- 任何 learned training（含预训练、自监督、distillation）；
- 任何模型 forward / backward（EPRO 或任何 baseline）；
- 任何超参搜索 / model selection；
- 任何 test set peeking；
- **从构造名（construct name）、邻近样本、同 family 样本推断 alt 碱基**；
- **从 ref base、位置、probe、condition 推断 alt**；
- 修改 raw RDAT 文件（raw 只读，checksum-verified，v3 §6.2）；
- 删除 D0-R / D1 / D2 的候选、失败记录或 artifact（历史证据保留，v3.1 §6.1）；
- 降低 v3 §8 Tier A/B/C 阈值；
- 把 annotation-only 候选当作序列验证 pair（v3.1 §3.2、§4）；
- 仅凭序列级 self-consistency 升级 sequence-based 候选（v3.1 §3.1）；
- 在 D2-R 末尾自动启动训练；
- 跨 parent/library 合并证据以凑齐佐证；
- 把"文件内 parent/reference 一致性"（D2 已核对）冒充"独立实验佐证"。

### 2.4 数据源范围（预先固定）

D2-R 查证的数据源**严格限定**为 D0-R v2 已收录范围的公开既有记录：

- 48 个 RDAT 文件（`/mnt/cunyuliu/reactflow_delta_raw/rmdb/rdat_tierA_20260730/`）；
- RMDB 对应 release（与上述 RDAT 同 source/version）；
- 8 个 citation_doi 对应的 publication supplementary material（公开版本）；
- 6 个 owner 对应的公开 laboratory data release。

**不**纳入：新抓取的网络数据、未公开发表的私有数据、D0-R v2 范围之外的新研究。范围用尽即触发停止条件（§4）。

---

## 3. 成功条件（v3.2 增补）

D2-R 的"成功"不以数量为目标，而以**每条升级记录的证据完整性**为标准。任一候选若要由 D2-R 证据补齐而升级为 `true_pair`，**必须**同时满足：

1. **来源（source）**：per-profile 序列或独立 replicate 来自 §2.4 范围内的公开既有记录，记录其 source entry id / DOI / owner / release version；
2. **checksum**：对所用原始文件记录 SHA256；与 D0-R v2 raw RDAT checksum 一致性可核对（若同一文件）；
3. **parent / condition / probe 对齐**：证据所属的 parent、condition（modifier）、probe 与 D0-R v2 relation 的 `parent_prefix`、`modifier`、probe 字段严格对齐；
4. **v3.1 §3.2 证据**：
   - per-profile 序列路径：pos/ref/alt 三者在 per-profile 序列 vs WT anchor 上全部验证通过（DNA→RNA T→U 规范后），且 `functional_edit_count==1`、substitution；
   - replicate 佐证路径：同 parent 的 replicate/control（v3.1 §3.1 §3.2），且仅有序列级 self-consistency 不算；
5. **逐条匹配**：证据与候选一一对应，**不得**用一条证据批量覆盖多条 alt 不同的候选；**不得**用同 study 内不同 parent 的 replicate 冒充同 parent replicate。

任一条件不满足的候选**保持** `candidate_only` 并附 `annotation_only_alt_not_verifiable`（或适用时 `sequence_based_no_independent_corroboration`），不得升级。

### 3.1 不允许的"凑证据"模式

- 不允许把"同 RDAT 文件内另一个 profile"自动当作"独立 replicate"——除非该 profile 在 v3.1 §3.1 §3.2 意义下确实是同 parent 的 replicate/control（需 RDAT 注释或 publication 明示）；
- 不允许把"同 study 不同 construct"当作"同 parent replicate"；
- 不允许用 D2 已核对的"文件内 parent/reference 一致性"作为 §3.2 的独立佐证。

---

## 4. 停止条件与 Fail-forward（v3.2 增补，引用 v3 §18、v3.1 §6）

### 4.1 正常停止

D2-R 在以下任一条件成立时停止：

1. **范围用尽**：§2.4 全部预设数据源/研究范围已逐条查证完毕，无新的合格证据；或
2. **合格证据已逐条核对完毕**：所有命中 §3 的候选均已记录来源/checksum/对齐/§3.2 证据，并完成 re-judgment。

### 4.2 负结果路径（回到路径 A）

若 §4.1.1 成立（范围用尽仍无合格证据），则：

- 冻结 EPRO learned-model 路线；
- 输出 D2-R 负结果 manifest（§6），记录查证范围、命中/未命中、未命中原因；
- 生成 D2 acceptance，状态 `complete_gate_not_passed_below_tier_b_data_audit`；
- **不**证明科学假设为假；只证明当前公开证据无法支撑所需 true-pair 训练集；
- 历史证据（D0/D0-R/D1/D2/D2-R 全部候选、失败记录、artifact）保留，不删除、不覆盖、不回缩（forward-only）。

### 4.3 正结果路径（受限 re-judgment）

若 §4.1.2 成立且有候选满足 §3 全部条件：

- 对满足条件的候选重新运行 D1 executor 的 true_pair 升级判定（仅这些候选，不降阈、不放宽 §3）；
- 重判 Tier A/B/C（T-D2.10 规则不变）；
- **即便达到 Tier B**，`training_allowed` 仍为 `False`（§5）；训练需独立授权；
- 未满足 §3 的候选保持 `candidate_only` + reason，不得因部分候选升级而整体放宽。

### 4.4 Fail-forward 边界

- D2-R 不得降低 v3 §8 Tier 阈值；
- D2-R 不得删除或覆盖 D0-R v1/v2、D1、D2 的任何历史证据；
- §2.3 任一禁止操作被触发即 fail-forward，记录违规并停止；
- D2-R 的全部查证记录（命中与未命中）作为历史证据保留。

---

## 5. training_allowed 边界（v3.2 增补，对应 v3 §0.2 第 8 条、v3.1 §7）

- 在 v3.2 下，**无论 D2-R 审计结果如何**，`training_allowed = False`；
- 训练启动的**必要条件**（非充分）：D2 重判达到 Tier B（`true_pair_count >= 1000` 且满足 v3 §8.2）；
- 达到 Tier B 后，训练仍需**独立授权**（新合同），不在本合同范围内；
- 本合同**不**授权任何 learned training、**不**授权任何 model forward/backward、**不**授权任何超参搜索。

---

## 6. D2-R 输出（v3.2 增补）

### 6.1 证据清单 manifest

D2-R 须输出 `artifacts/reactflow_delta/d2r/d2r_evidence_manifest.json`，逐候选记录：

- 候选 key（`rdat_sha256`、`wt_profile_index`、`mutant_profile_index`、`parent_prefix`、`rmdb_id`、`citation_doi`）；
- 查证状态（`evidence_found` / `no_evidence_in_scope`）；
- 证据类型（`per_profile_sequence` / `same_parent_replicate` / `none`）；
- 来源（source entry id / DOI / owner / release version / 文件 SHA256）；
- parent / condition / probe 对齐核对结果；
- §3.2 证据核对结果（pos/ref/alt 验证 或 replicate 佐证）；
- 未命中原因（machine-readable，如 `no_per_profile_sequence_in_rdat`、`no_supplementary_sequence_published`、`replicate_belongs_to_different_parent`）。

### 6.2 Re-judgment artifact

若 §4.3 成立，输出 `artifacts/reactflow_delta/d2r/d2r_tier_judgment.json`（结构与 `d2_tier_judgment.json` 一致，basis 注明 `D2-R evidence-supplemented re-judgment`）。

### 6.3 D2 acceptance（负结果或正结果）

输出 `artifacts/reactflow_delta/d2r/d2r_acceptance.json`：

- 负结果：`status = complete_gate_not_passed_below_tier_b_data_audit`；
- 正结果：`status = d2r_evidence_supplemented_tier_<X>`，并附 Tier 判定引用。

### 6.4 Tests、commit、push

D2-R 须附 tests（`tests/reactflow_delta/test_d2r_*.py`）覆盖：证据匹配规则、§3.2 证据完整性校验、§3.1 反"凑证据"规则、停止条件判定。完成后 commit + push。

---

## 7. 决策者签署栏（v3.2 增补）

本合同在决策者签署前为**草案**，`d2r_allowed = False`。签署后 `d2r_allowed = True`，但 `training_allowed` 仍为 `False`（§5）。

- 决策者：________________  日期：________________
- 路径选择（A / B）：________________
- 若选 B，本合同生效；D2-R 执行完毕后按 §4 停止条件闭环。

---

## 8. 一句话总结（v3.2 增补）

> D2-R 只做一件事：在公开既有记录里**查证**是否有被遗漏的 per-profile 序列或同 parent 独立 replicate，以解锁 annotation-only alt 的 v3.1 §3.2 升级。找到就重判，找不到就诚实归档负结果。**无论结果如何，都不训练、不降阈、不推断 alt。**
