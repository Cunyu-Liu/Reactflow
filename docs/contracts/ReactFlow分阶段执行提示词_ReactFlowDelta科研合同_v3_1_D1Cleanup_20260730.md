# ReactFlow-Δ EPRO v3.1 D1 Cleanup-Only 增量科研合同

> 中文名称：ReactFlow-Δ 平衡态扰动响应算子科研合同 v3.1（D1 cleanup-only 授权增量）
> 合同版本：V3.1
> 基线合同：V3.0（`ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_EPRO_20260729.md`）
> 增量日期：2026-07-30
> 授权阶段：Phase D1（清洗、配对、噪声与标签）cleanup-only
> 适用仓库工作树：`/home/cunyuliu/reactflow_delta_goal_20260729`
> 授权证据：D0-R v2 重审计（`reports/d0r_data_feasibility_audit.md` 末尾 "D0-R v2 Re-Audit" 章节 + `manifests/reactflow_delta/d0r/d0r_reaudit_tierA_manifest.json` + `manifests/reactflow_delta/d0r/d0r_acceptance.json`）

> **For Codex/Claude:** 本合同为 v3.0 的增量，必须与 v3.0 合同一起阅读；v3.0 未被本合同取代的部分继续有效。

---

## 0. 文档权威性、继承关系与生效条件（v3.1 增补）

### 0.1 v3.1 性质

本合同是 v3.0（V3_EPRO）的**增量合同**，不取代 v3.0 的：

- 架构（§3–§4）、物理与数学假设（§5）；
- 数据优先级与 Tier 定义（§6.1、§8）；
- 模型等级与阶段（§4.9、§11）；
- Split、污染与 benchmark（§9）；
- 强制基线、消融与负控制（§10）；
- 测评与统计（§12）；
- Fail-forward 合同（§18）；
- 最终执行原则（§20）。

v3.1 只授权**一个新阶段**：Phase D1 cleanup-only（清洗、配对、substitution verification、标签），**禁止任何 learned training**。

### 0.2 继承关系

v3.1 完整继承 v3.0 的：

- §0.2 全部冻结决策（含第 8 条"完成 D0–D2 数据 Gate 前禁止启动任何 learned training"）；
- §0.1 权威性条款与 fail-forward 原则；
- §6 数据与清洗合同（§6.2 原始层只读、§6.5 第一版范围、§6.6 清洗顺序、§6.7 禁止操作）；
- §8 数据可行性 Gate；
- §15 Phase D1 的 Todo（T-D1.1 至 T-D1.12）与 Gate；
- §18 Fail-forward 合同；
- §20 最终执行原则。

### 0.3 生效条件

- 本合同由**项目决策者审核发布后生效**；在此之前 `re_d1_allowed = False`（见 D0-R v2 `Stage Permissions`）。
- 生效后，D0-R v2 的 `re_d1_allowed` 由 `False` 翻转为 `True`，但仅限 cleanup-only 范围（见本合同 §2）。
- 生效不等于授权训练；训练仍需 D2 Tier B 批准（见 v3 §0.2 第 8 条、本合同 §7）。

---

## 1. D0-R 证据基础（v3.1 增补）

v3.1 的全部授权建立在 D0-R v2 重审计的 forward-only 证据之上。证据只读、不覆盖、不回缩。

### 1.1 D0-R v2 重审计结果

来源：`reports/d0r_data_feasibility_audit.md` 末尾 "D0-R v2 Re-Audit" 章节；slim manifest `manifests/reactflow_delta/d0r/d0r_reaudit_tierA_manifest.json`。

| 指标 | 值 |
|---|---|
| Tier A 候选 RDAT 文件 | 101 |
| 下载成功 | 101 |
| 解析成功 | 72 |
| 解析错误（诚实记录） | 29 |
| 零候选文件（解析 OK 但无候选） | 24 |
| **候选单突变对（candidate_only）** | **7,761** |
| 独立 study（owner, doi） | 8 |
| 独立 parent（rmdb_id prefix） | 31 |
| 独立 owner | 6 |
| `re_tier_judgment` | **Tier A** |
| `re_d1_allowed`（v3.1 发布前） | **False** |
| `re_triage_decision` | `reaudit_qualified_to_propose_v3_1_non_learning_d1_cleanup_only` |

**Tier A Gate 核验**（对照 v3 §8.1：≥5 study / ≥20 parent / ≥5,000 pair）：实测 8 / 31 / 7,761，**三项全部满足**。此为数据可用性 Gate，不是模型质量声明。

### 1.2 D0-R v1 严格证据子集

D0-R v1（M2SL5 单 study）产出的 **744 个候选**（372 2A3 + 372 DMS，functional Hamming==1，pos/ref/alt 序列级全部验证）是 v2 证据的**严格子集**：序列级编辑可验证。v1 结果作为历史证据保留，未被 v2 覆盖或回缩。

### 1.3 诚实记录的失败

v3.1 明确记录 D0-R v2 的以下失败，不掩盖、不静默丢弃：

- **29 个解析错误**：
  - 27 个为 `RDAT_VERSION != 0.34`（D0 fail-closed 解析器不接受）：`BSUGLY_DMS_0003..0014`（12 个，v0.4）、`TRP4P6_DMS_0002..0014`（13 个，v0.22/v0.24/`VERSION`-key）、`CBAG4P_DMS_0003..0004`（2 个）；
  - 2 个为 `invalid indexed annotation key`：`GLYCFN_KNK_0001`、`GLYCFN_KNK_0002`。
- **24 个零候选文件**（解析 OK 但无可配对候选）：
  - 8 个 `HIV3PR_*`：`annotation_ref_mismatch`——annotation 使用 HIV genome numbering（offset），非 construct-local，WT 找到但 ref 不匹配；
  - 12 个 `SL5CV2/SL5HKU/SL5MER_*`：`no_wt_anchor`；
  - 2 个 `TODEX_1M7/DMS_0000`：`no_encoded_mutation`；
  - 2 个 `RNASEP_RSQ_0000`、`TODS7_MUT_0001`：`no_wt_anchor`。
- **annotation-only 的 alt=X 不可验证**：M2-seq 文件编码 alt 为 `X`（variable pool），序列级 edit 在无 per-profile 序列时**不可验证**，证据强度弱于 sequence-based 路径。

上述 29 个解析错误与 24 个零候选文件**应在 D1 中尽力修复**（见本合同 §3.3 HIV3PR offset、§5 解析器扩展），修复属于 forward-only：不回缩失败记录，只新增可解析证据。

### 1.4 候选状态声明

D0-R v2 的全部 7,761 个候选均为：

- `candidate_only_pending_parent_lineage_and_functional_region_validation`；
- `true_pair = False`。

**D1 的任务**：在不训练的前提下，把这 7,761 个候选中**符合条件的**升级为 `true_pair = True`，并给每个不升级的候选附 machine-readable exclusion reason。D1 **不得**自动把候选当 pair。

---

## 2. D1 cleanup-only 授权范围（v3.1 增补，对应 v3 §15 Phase D1）

### 2.1 允许的操作

D1 仅允许 v3 §15 Phase D1 的 T-D1.1 至 T-D1.12，且全部为非学习操作：

- **T-D1.1** 冻结 construct/pair schema（见 v3 §6.3、§6.4）；
- **T-D1.2** condition exact matching；
- **T-D1.3** substitution verification（含 HIV3PR genome-numbering offset 修复，见本合同 §3.3）；
- **T-D1.4** alignment 与 unchanged mask；
- **T-D1.5** probe eligibility；
- **T-D1.6** 识别 replicate/no-edit/control；
- **T-D1.7** raw/upstream/project-normalized 三层；
- **T-D1.8** 估计 study/probe measurement noise（仅用 train/validation，见 §4）；
- **T-D1.9** 有重复时运行 frozen differential caller；
- **T-D1.10** 生成 quality weight 和 exclusion reasons；
- **T-D1.11** 建手算 fixtures；
- **T-D1.12** tests、commit、push。

允许的 RDAT 解析器扩展见本合同 §5（forward-only 修复 D0-R 失败）。

### 2.2 禁止的操作（machine-readable）

D1 期间**禁止**以下任何操作，违反即触发 fail-forward（见 §6）：

- 任何 learned training（含预训练、自监督预训练、teacher distillation）；
- 任何模型 forward / backward（EPRO 或任何 baseline）；
- 任何超参搜索 / model selection；
- 任何 test set peeking（test labels 隔离见 v3 §9.3）；
- 用 test 估计 noise 或 normalization；
- 降低 v3 §8 Tier A/B/C 阈值；
- 把 construct 数冒充 pair 数（v3 §6.7）；
- 把 annotation-only 候选当作序列验证 pair（见 §3.2）；
- 仅凭序列级 self-consistency 升级 sequence-based 候选（见 §3.1）；
- 修改 raw RDAT 文件（raw 只读，checksum-verified，见 v3 §6.2）；
- 删除 D0-R 候选或失败记录（历史证据保留，见 §6）；
- 在 D1 末尾自动启动训练（见 §7）。

### 2.3 引用 v3 §6.5 第一版范围

D1 升级 true_pair 时**只纳入**满足 v3 §6.5 的候选：

- substitution；
- `edit_count = 1`；
- WT/mutant 等长；
- probe 完全一致；
- condition 完全匹配；
- 至少 60% 未编辑位置可比；
- primary physics domain 优先 in vitro。

延后项（insertion/deletion、多编辑主训练、跨 probe 数值回归、in vivo/in vitro 混合、parent 不明的近邻、未知 normalization pair）不在 D1 升级范围。双突变只进入 rescue 子集。

### 2.4 引用 v3 §6.6 清洗顺序与 §6.7 禁止操作

所有 true_pair 升级**必须**通过 v3 §6.6 的 14 步清洗顺序（解析 raw → 校验长度 → T/U 规范 → 验证编辑关系 → 核对 condition → alignment → probe eligibility → replicate/no-edit → missingness/SNR/coverage → measurement noise → 冻结 normalization → 生成 Δreactivity → physical features → split）。

v3 §6.7 全部禁止操作在 D1 继续生效（test 参与 normalization、pair 自选最小差异缩放、missing 当 0、模型预测填主标签、删除 no-change/负结果、按 observed effect 选 test、跨 parent/library split、无条件合并 probe、teacher 冒充 experimental、construct 冒充 pair、latent energy 冒充 kcal/mol）。

---

## 3. true_pair 升级规则（v3.1 增补，关键章节）

本节定义哪些 `candidate_only` 候选可在 D1 升级为 `true_pair = True`。升级是 D1 的核心交付物，必须逐候选判定并留 machine-readable 证据。

**核心原则（覆盖 §3.1–§3.4）**：所有序列的候选（无论 sequence-based 还是 annotation-only）**必须经过独立佐证之后**才能升级为 `true_pair`。任何路径均不允许"直接升级"。

### 3.1 sequence-based 候选升级

**执行前置要求**：本节升级规则的执行必须以 v3.0 主合同（`ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_EPRO_20260729.md`）的规范与要求为前置依据。执行前必须对照并满足主合同 §6（数据与清洗合同）、§8（数据可行性 Gate）、§15 Phase D1 的全部规范与要求；本节规则是主合同规范的**增量约束**，不取代主合同的任何前置要求。主合同与本节冲突时，以主合同为准（本节只能更严，不能放宽）。

sequence-based 候选（pos/ref/alt 序列级全验证，对应 D0-R v1 的 744 严格子集类型）**不能仅凭序列级 self-consistency 直接升级**为 `true_pair`。升级须满足以下**全部**条件，并附加**独立佐证**：

- `functional_edit_count == 1`，且为 substitution（非 indel）；
- pos/ref/alt 三者在 per-profile 序列 vs WT anchor 上**全部验证通过**（DNA→RNA T→U 规范后）；
- condition exact matching（v3 §6.5、T-D1.2）；
- probe eligibility 不变（T-D1.5）；
- ≥60% 未编辑位置可比（v3 §6.5）；
- 通过 v3 §6.6 全部 14 步清洗顺序；
- 通过本合同 §4 的 D1 Gate；
- **独立佐证**：同一 parent 的 replicate（T-D1.6 识别的 replicate/control）或同 study 内独立 profile / condition 的二次观测等独立证据。仅有序列级 self-consistency **不构成**佐证。

未取得独立佐证的 sequence-based 候选保持 `candidate_only`，并附 exclusion reason = `sequence_based_no_independent_corroboration`。

### 3.2 annotation-only 候选升级

annotation-only 候选（alt=X 不可验证，M2-seq 类）**不能直接升级**为 `true_pair`。升级需额外证据：

- 有 per-profile 序列证据，能把 alt=X 解析为具体碱基并验证；**或**
- 有同一 parent 的 replicate 佐证（T-D1.6 识别的 replicate/control）。

否则保持 `candidate_only`，并附 exclusion reason = `annotation_only_alt_not_verifiable`。

**禁止**把 annotation-only 候选当作序列验证 pair 计入任何 Tier gate 的 pair 数。

### 3.3 HIV3PR genome-numbering offset

HIV3PR 系列 8 个文件的 `annotation_ref_mismatch` 属于 substitution verification（T-D1.3）问题：

- annotation 使用 HIV genome numbering（offset），非 construct-local 1-indexed；
- D1 **必须**用正确的 genome-numbering offset 重新验证 ref；
- offset 修复后 ref 验证通过，且 alt 可验证（sequence-based）或符合 §3.2 规则（annotation-only），方可升级；
- offset 修复**不**自动升级任何候选；仍须通过 §3.1 或 §3.2 全部条件。

### 3.4 无法验证候选的处理

- 任何无法验证 `edit_count = 1` substitution 的候选：保持 `candidate_only` **或**排除，并附 machine-readable reason（取值集合见 §4）；
- 任何 ref/alt 验证失败的候选：exclusion reason = `substitution_not_verifiable`；
- 任何 condition 不匹配 / probe 不一致 / 可比位置 <60% 的候选：按 v3 §6.5 排除并附 reason；
- sequence-based 候选未取得独立佐证的：exclusion reason = `sequence_based_no_independent_corroboration`；
- 所有 exclusion reason 写入 pair schema 的 `exclusion_reasons` 字段（v3 §6.4）。

---

## 4. D1 Gate（v3.1 增补，对应 v3 §15 Phase D1 Gate）

D1 完成须通过以下 Gate（全部 bullet 必须为 PASS）：

- **fixtures 100% 通过**（T-D1.11 手算 fixtures）；
- **missing 不作 0**（v3 §6.7、§6.6 第 9 步）；
- **noise 不用 test 估计**（仅 train/validation，见 §2.2）；
- **normalization 不最小化 pair difference**（v3 §6.7）；
- **每个 exclusion 有 machine-readable reason**，取值至少包含：
  - `annotation_only_alt_not_verifiable`
  - `sequence_based_no_independent_corroboration`
  - `substitution_not_verifiable`
  - `annotation_ref_mismatch`
  - `condition_mismatch`
  - `probe_mismatch`
  - `comparable_positions_below_60pct`
  - `edit_count_not_one`
  - `indel_not_substitution`
  - `no_wt_anchor`
  - `normalization_domain_unknown`
  - `parent_lineage_unverified`
  - `in_vivo_in_vitro_mixed`
- **不自动进入训练**：D1 通过后 `training_allowed` 仍为 `False`；训练需 D2 Tier B 批准（v3 §0.2 第 8 条、本合同 §7）；
- **Tier gate 不被降低**：D1 输出的 `true_pair` 数量必须诚实报告，不得用 construct 数或 annotation-only 候选冒充序列验证 pair。

---

## 5. D1 应当尽力修复的解析器扩展（v3.1 增补，forward-only 修复）

针对 D0-R v2 的 29 个解析错误（§1.3）与 24 个零候选文件中的可修复子集，D1 **应当尽力**进行以下 forward-only 解析器扩展与修复。这些扩展只修复解析兼容性，**不**改变 Tier 定义、**不**自动产出 pair——产出仍须通过本合同 §3 的 true_pair 升级规则。"尽力"意味着：尝试所有合理技术路径，对仍无法修复的文件诚实记录失败原因，不静默丢弃。

### 5.1 VERSION 作为 RDAT_VERSION 别名

- 解决 `TRP4P6_DMS_0002..0014`（13 个文件）的 `VERSION`-key 问题（文件用 `VERSION` 头，解析器期望 `RDAT_VERSION`）；
- 实现方式：解析器接受 `VERSION` 作为 `RDAT_VERSION` 的别名。

### 5.2 接受 RDAT_VERSION 0.4 / 0.22 / 0.24

- 解决 `BSUGLY_DMS_0003..0014`（12 个，v0.4）与 `CBAG4P_DMS_0003..0004`（2 个）的版本拒绝问题；
- 接受后，版本门控的 26→27 个文件可被解析（含 §5.1 的 TRP4P6 系列）；
- 解析器仍 fail-closed：未明确接受的版本继续报错并诚实记录。

### 5.3 GLYCFN 索引 annotation 格式

- 解决 `GLYCFN_KNK_0001`、`GLYCFN_KNK_0002`（2 个）的 `invalid indexed annotation key`；
- 处理 `ANNOTATION_DATA:1 modifier:DMS` 索引格式。

### 5.4 产出约束

- §5.1–§5.3 扩展产出的候选**仍**须通过 §3 升级规则，**不能**因解析成功而自动成为 `true_pair`；
- 扩展属于 forward-only：不回缩 D0-R 的失败记录，只新增可解析证据；
- HIV3PR 的 offset 修复**不**在本节范围，属于 substitution verification（T-D1.3，见 §3.3）；
- 对尽力尝试后仍无法修复的文件，必须诚实记录失败原因（machine-readable），保留为历史失败证据，不静默丢弃。

---

## 6. Fail-forward 边界（v3.1 增补，引用 v3 §18）

### 6.1 历史证据保留

- D0-R v1（744 候选）和 v2（7,761 候选）的全部候选、失败记录、parse errors、零候选文件作为历史证据保留，**不得删除、覆盖或回缩**；
- D0 原始 NO_GO acceptance 与 D0-R v1 acceptance 均保留为历史证据；
- §5 解析器扩展与 §3.3 HIV3PR offset 修复属于 forward-only：不回缩 D0-R 的失败记录，只新增可解析证据；尽力尝试后仍失败的文件保留为历史失败证据。

### 6.2 D1 不得降低的边界

- D1 **不得**降低 v3 §8 的 Tier A/B/C gate 阈值；
- D1 **不得**把 construct 数冒充 pair 数（v3 §6.7）；
- D1 **不得**把 annotation-only 候选当作序列验证 pair（§3.2）；
- D1 **不得**仅凭序列级 self-consistency 升级 sequence-based 候选（§3.1）；
- D1 **不得**因解析器扩展（§5）而自动升级任何候选。

### 6.3 Tier B 不达标的处理

D1 清洗完成后，根据**实际** `true_pair` 数量评估是否达到 v3 §8.2 Tier B gate（≥1,000 primary-eligible pair）。**本合同不预先承诺**具体的 fail-forward 处理路径（如转数据/benchmark/negative result 路线、是否继续 ReactFlow-Δ、是否返回 PCCNG 等），相关决策留待 D1 Gate 报告产出后由项目决策者基于实际证据重新审核决定。

无论后续如何决策，以下边界仍强制生效：

- **不降低** v3 §8 Tier A/B/C gate 阈值；
- **不删除、不覆盖、不回缩** D0-R v1/v2 的全部清洗证据与失败记录；
- **不在 Gate 未通过时训练**（v3 §0.2 第 8 条）；
- **不隐藏失败**，Tier 判定与 true_pair 数量必须诚实报告。

### 6.4 引用 v3 §18 其余条款

v3 §18 的冻结流程（run/config/git/data/split/feature hashes、logs、metrics、invariants、failure evidence）与七层定位（A 数据 / B 观测 / C 泄漏 / D 物理算子 / E 基线 / F 优化 / G 科学假设）在 D1 适用。D1 期间的失败定位主要落在 **A 数据** 与 **B 观测** 层。

---

## 7. 下一阶段执行 Goal（v3.1 增补，对应 v3 §16）

```text
你现在只执行 ReactFlow-Δ EPRO 的 Phase D1 cleanup-only。

服务器：
ssh -p 22 cunyuliu@36.137.135.49

工作树：
/home/cunyuliu/reactflow_delta_goal_20260729

权威合同：
- 基线：docs/contracts/ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_EPRO_20260729.md
- 增量：docs/contracts/ReactFlow分阶段执行提示词_ReactFlowDelta科研合同_v3_1_D1Cleanup_20260730.md

最高目标：
在不启动任何 learned training 的前提下，把 D0-R v2 的 7,761 个 candidate_only
候选中符合条件的升级为 true_pair，完成清洗、配对、substitution verification、
标签、noise 估计与 exclusion reasons。

强制边界：
1. 不训练任何 learned model（含预训练）。
2. 不做任何模型 forward / backward。
3. 不做超参搜索 / model selection。
4. 不 peek test set。
5. 不把 construct 数当 pair 数。
6. 不把 annotation-only 候选当序列验证 pair。
7. 不降低 v3 §8 Tier gate。
8. 不修改 raw RDAT（只读、checksum-verified）。
9. 不删除 D0-R 候选或失败记录（forward-only）。
10. 所有 true_pair 升级走本合同 §3 规则，执行前必须先对照并满足 v3.0 主合同
    （§6 数据与清洗合同、§8 数据可行性 Gate、§15 Phase D1）的全部规范与要求。
    - sequence-based 候选必须有独立佐证（replicate 或同 study 独立 profile/condition），不能仅凭序列级 self-consistency 升级（§3.1）。
    - annotation-only 候选必须有 per-profile 序列证据或 replicate 佐证（§3.2）。
11. 应当尽力修复 D0-R v2 的 29 个解析错误文件和 24 个零候选文件中的可修复子集
    （见 §5 解析器扩展、§3.3 HIV3PR genome-numbering offset）；修复属 forward-only，
    仍无法修复的文件诚实记录失败原因，不静默丢弃。
12. 每个 T-D1 完成后 targeted tests、focused commit、push。
13. GitHub 不提交 raw data、weights、checkpoints、cache、secret。

D1 解析器扩展与修复（forward-only，见 §5、§3.3）：
- VERSION 作为 RDAT_VERSION 别名（TRP4P6 13 文件）
- 接受 RDAT_VERSION 0.4/0.22/0.24（BSUGLY/CBAG4P）
- GLYCFN 索引 annotation 格式（2 文件）
- HIV3PR genome-numbering offset 属 T-D1.3 substitution verification（见 §3.3）

依次完成 T-D1.1 至 T-D1.12。

必须输出：
- 升级后的 true_pair registry（含 exclusion_reasons）
- raw/upstream/normalized 三层
- measurement noise 估计（train/validation only）
- quality weight
- 手算 fixtures 全通过
- D1 Gate 报告（逐 bullet PASS/FAIL）

报告必须包含：
- 候选总数（7,761）
- 升级为 true_pair 的数量（分 sequence-based-with-corroboration / annotation-only-with-evidence）
- 保持 candidate_only 的数量与 reason 分布
- 排除的数量与 reason 分布
- §5/§3.3 修复尝试结果：成功修复的文件数、仍失败的文件数与失败原因
- study / parent / owner 分布（升级后）
- probe / condition / in-vitro-in-vivo 分布
- Tier A/B/C 重判（用 true_pair 数，不用候选数）
- 是否达到 Tier B（≥1,000 true_pair）
- commit SHA / branch / push status

Gate 未通过：
- 不训练
- 不降低阈值
- 保留审计
- fail-forward 路径由项目决策者基于实际证据重新审核决定（见本合同 §6.3）
```

### 7.1 D1 完成后不自动进入 M0

- D1 Gate 通过**不**自动进入 M0（EPRO-Lite 可学习性）；
- 训练前置条件：D0–D2 全部 Gate 通过 + Tier B 以上（v3 §0.2 第 8 条）；
- D1 完成后须先完成 D2（RSIB-v1 与数据 Gate，v3 §15 Phase D2），D2 批准 Tier B 以上方可启动 M0。

---

本合同由项目决策者审核发布后生效。生效前 `re_d1_allowed = False`，D1 任何工作均不被授权。
