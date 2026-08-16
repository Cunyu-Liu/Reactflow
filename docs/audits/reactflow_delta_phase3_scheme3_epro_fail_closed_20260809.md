# ReactFlow-Delta Phase 3 Scheme-3 (Repaired EPRO Propagation) — FAIL-CLOSED

审查日期：2026-08-09
Authority：epoch 18（Phase 3 模型架构迭代，合同 §9.4）
Endpoint：endpoint_v5（conditional WMAE skill vs trivial）
Run：`results/phase3_scheme3_20260809/`（`phase3_scheme3_verdict.json`）

## 1. 目的

对方案三（修复 EPRO propagation operator：稀疏 top-k base-pair contact 上的非局部传播）
做 nested leave-one-publication-out 验收，与同容量 scheme-2 generic concat 基线
（[WT, Mut, cond]）比较。验收标准：paired publication-block bootstrap 的
epro − generic skill 差异 CI 下界 > 0；random-contacts 不产生同等增益；
propagation 消融（epro_local）按预注册方向退化。

## 2. 实现

- `scripts/reactflow_delta/models/epro_v1.py`：修复 EPRO。反称向量 forcing（swap → −h）、
  softplus 对称非负权重、bias-free magnitude readout（swap-invariant，identity=0）、
  稀疏 Neumann 传播（显式 rho<1 归一）。`GLOB_DIM=37` 与真实 condition 特征对齐。
- `scripts/reactflow_delta/build_epro_contacts.py`：ViennaRNA BPP → 每个位置 top-4 稀疏 contact 图。
- `scripts/reactflow_delta/run_phase3_scheme3.py`：LOOCV 编排；epro / epro_local / epro_random
  / generic / trivial，5 seeds，30 epochs，CUDA（GPU 3，A100 40GB）。
- `tests/reactflow_delta/test_epro_v1.py`：5 项合成不变量单测全 PASS
  （identity / antisymmetry / residual+rho<1 / gradient non-vanishing / capacity-matched）。

## 3. 结果（每 seed 的 conditional-WMAE skill）

| variant | seed0 | seed1 | seed2 | seed3 | seed4 | mean |
|---|---|---|---|---|---|---|
| epro | 0.6666 | 0.6184 | 0.6594 | 0.6834 | 0.6548 | **0.6565** |
| epro_local | 0.6647 | 0.6262 | 0.6522 | 0.6657 | 0.6391 | 0.6496 |
| epro_random | 0.6683 | 0.6261 | 0.6580 | 0.6840 | 0.6579 | 0.6589 |
| generic | 0.6995 | 0.6772 | 0.6772 | 0.6843 | 0.6541 | **0.6785** |
| trivial | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

epro − generic 差异 CI（1000 bootstrap，10 publications）：
`ci_low_min = −0.2263`（各 seed ci_low ∈ [−0.2263, −0.1795]）。

## 4. 验收判定

| 标准 | 值 | 判定 |
|---|---|---|
| epro CI 下界 > 0 vs generic | −0.2263 | **FAIL** |
| epro_mean > generic_mean | 0.6565 < 0.6785 | FAIL |
| random-contacts 不产生同等增益 | epro_random 0.6589 ≥ epro 0.6565 | **FAIL** |
| propagation 消融方向（epro_local 更低） | 0.6496 < 0.6565 | PASS（弱） |
| estimand 可识别（全 seed） | True | PASS |

**最终裁决：`PHASE3_SCHEME3_REPAIRED_EPRO_FAIL_CLOSED_RETIRED`**

修复后的非局部传播 EPRO 未能在跨 publication 的 conditional-magnitude 任务上胜过
同容量 generic concat 基线（diff CI 下界为负）；且随机 contact 图给出与真实 contact
相同的 skill，说明真实 base-pair contact 传播未提供可测得的增量信息，非局部传播假设
在该 endpoint 上未被支持。

## 5. 与方案一、二的关系

Phase 3 三个授权方案（PairHeadV1 DeepSets、exact-alt 显式交互、修复 EPRO 传播）均已
fail-closed 退役——无一在 CI 下界 > 0 的标准下胜过同容量 generic 基线。conditional-magnitude
endpoint 上最强的仍是 P2-v5 定案中的 deepsets（Route B, GO, epoch 17）。

## 6. 证据与复现

- verdict：`results/phase3_scheme3_20260809/phase3_scheme3_verdict.json`（SHA 见 SHA256SUMS 若生成）
- heldout .npz：`results/phase3_scheme3_20260809/heldout_{variant}_seed{s}.npz`
- 单测：`python -m pytest tests/reactflow_delta/test_epro_v1.py -q` → 5 passed
- 复现命令（GPU 3）见 `run_phase3_scheme3.py --help`。
