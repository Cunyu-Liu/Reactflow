# ReactFlow-Delta Target Profile Identity Correction Amendment

**冻结日期：** 2026-08-24  
**状态：** `IDENTITY_FIX_VERIFIED_CORRECTED_SCIENTIFIC_REBUILD_NOT_YET_AUTHORIZED`  
**作用域：** OpenKnot M2 full-mutant target/error profile 的 construct identity；不改变 split、metric、Gate、external lock 或任何既有 artifact 内容。

## 1. 为什么必须另立 amendment

在对已经终局的 v5/v6 prediction 做分层误差复盘时，真实数据路径暴露出一个可重复的 construct identity 错误。旧 `M2Universe._index_full_profiles()` 以 raw row-id prefix 建立 full-profile index；`mutant_full_profile()` 却使用 `puzzle + method` 生成访问 key。OpenKnot 中两者并不总相同：

- method=`Starting sequence` 的 raw prefix 是 `{puzzle}_WT`；
- method=`gRNAde` 或 `gRNAde-no3d` 的 raw prefix 是 `{puzzle}_gRNAde2`。

旧 key 查找失败后使用 mutation-only suffix fallback，并从整个数据集中选择首个相同 `design_pos/ref/alt` 的 row。该 fallback 不约束 puzzle、method 或 construct，因此会读取其他 construct 的 target/error profile。

真实 M2 v4.5.2 的 outcome identity audit 确认：

- registered mutants：`13,976`；
- 错误 fallback：`3,494` mutants；
- 受影响 cells：`40/160`；
- `Starting sequence`：`1,747` mutants；
- `gRNAde-no3d`：`1,305` mutants；
- `gRNAde`：`442` mutants；
- 所有 `3,494` 次 fallback 都解析到不同 construct；
- 修复后逐 mutant 对原始 puzzle/method/mutation row 比较，target mismatch=`0`、error mismatch=`0`、missing=`0`。

这会同时影响 outer-train target 与 held evaluation target，不能通过只重算汇总表修复。

## 2. 历史证据处理

所有旧文件永久保留，不覆盖、不删除、不把原有 FAIL 改成 PASS：

- v1/v2/v4/v5/v6 的原始终局字符串保持不变；
- v3 登记为 `R3C3_INTERRUPTED_INVALIDATED_TARGET_IDENTITY_RECOVERABLE`，不是 PASS 或 FAIL；
- 已完成的 v3 10 个 fold artifacts 保留，但不得进入 merge、qualifier、checkpoint reuse 或论文；
- 使用旧 `mutant_full_profile()` 训练或评分得到的 performance effect、CI、positive-puzzle count、calibration 与机制解释统一降级为 `SCIENTIFICALLY_INVALIDATED_TARGET_IDENTITY`；
- 旧 artifact 只能用于证明历史执行过程和定位 bug，不能用于比较模型能力。

不受该 target identity bug 影响、但仍需各自原资格验证的 outcome-blind artifact：

- split_v4；
- 注册 mutation/sequence/coordinate metadata；
- v4 RNA-FM cache；
- v5 unconstrained thermodynamic feature cache；
- v6 WT-2A3-constrained thermodynamic feature cache；
- v7 RiNALMo dependency cache。

这些 cache 不包含 mutant target，但在重用前仍须通过原 schema、key universe 与 corrected coordinate/identity alignment。

## 3. 修复定义

唯一合法 full-profile identity：

```text
(puzzle, method, design_pos, canonical_ref, canonical_alt)
```

内部 canonical key：

```text
{puzzle}_{method}_mm_{design_pos}_{ref}_{alt}
```

要求：

- raw row-id prefix 不再作为 accessor identity；
- T/U 在 ref 与 alt 两侧均规范化为 U；
- 删除 mutation-only suffix fallback；
- canonical key 重复直接报错；
- build 阶段要求所有 registered mutants 都存在 exact canonical full profile；
- 真实数据资格要求 target/error identity `13,976/13,976` exact match。

## 4. 与 v7 的衔接

V7M1 只读取 `id, sequence, puzzle, method, sub_start, mutA` 并生成 outcome-blind dependency，因此可以在 correction 期间继续；V7M1 cache PASS 不是科学模型 PASS。

旧 V7M2 中“replay v6 candidate predictions at `1e-12`”的要求失效，因为旧 v6 predictions 使用了错误 target。V7M2 在任何 score access 前改为：

- 从修复后的 accessor 重新拟合 direct18、v5-feature30、v6-feature41；
- corrected baseline41 与 dependency47 使用相同 outer-train target、权重、standardization、alpha 和 key universe；
- 旧 v6 predictions 只用于证明不可重用，不进入数值 replay；
- corrected baseline41 的 algorithm replay 由独立重新执行的同算法 fold artifact 验证；
- 保持原 1% signed-delta eligibility Gate、absolute-delta guardrail、20-fold complete-before-score 与所有 coverage Gate不变。

v3/corrected B1 必须使用修复后的 accessor从 fold 0 开始完整重建；不得保留前 10 个旧 checkpoint，也不得只补跑 folds 10–19。

## 5. 状态机

```text
TIC0_BUG_CONFIRMED_AND_INVALID_RESULTS_QUARANTINED
  -> TIC1_ACCESSOR_FIX_AND_REAL_DATA_IDENTITY_PASS
  -> V7M1_OUTCOME_BLIND_CACHE_PASS
  -> TIC2_CORRECTED_LINEAR_BASELINES_AND_V7M2_COMPLETE_PROBE
  -> only exact V7M2 eligibility PASS: TIC3_CORRECTED_B1_AND_DEPENDENCY_OPERATOR
  -> original V7M4/V7M5 high-effect Gates
```

当前只允许 TIC1、V7M1 outcome-blind cache与不会读取新 score 的实现/测试。corrected model training、held score 与 partial score继续关闭，直到 focused commit、real-data identity PASS、V7M1 exact PASS和新的 authority更新全部完成。

## 6. 不可协商边界

- 不访问 external outcome；
- 不覆盖任何旧 artifact；
- 不把旧 FAIL 美化为 PASS，也不把 invalidation误写为模型 FAIL；
- 不降低 V7 的 1% eligibility 或 5% dual-metric top-journal Gate；
- 不因 identity fix 搜索模型、alpha、epoch、seed、feature subset或threshold；
- corrected run必须使用新目录并从 fold 0 开始；
- 完整 universe 前仍禁止读取 partial score。

