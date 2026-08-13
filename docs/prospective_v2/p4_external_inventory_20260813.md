# P4 外部候选盘点 / outcome-blind 资格预检 (2026-08-13)

## 结论（outcome-blind，尚未打开任何 locked external outcome）
当前未识别出可构成"development-disconnected、task-compatible、统计充分"的外部 confirmatory 组件。
按合同 §7.6，若公开数据无法形成合格 external set，则 Phase 4 停止为
`PUBLIC_EXTERNAL_NOT_QUALIFIED` / `UNDERPOWERED_NOT_CONFIRMATORY_PREACCESS`。

## 盘点证据（read-only）
- OpenKnot M2 官方 release：`PRIMARY_PUBLIC_DEVELOPMENT`（合同角色，不能作为 confirmatory）。
- RMDB 下载：`/mnt/cunyuliu/reactflow_delta_raw/rmdb`（2245 文件，含 rdat）——**单条件 SHAPE 反应性谱**，
  非 M2 式"全谱突变响应 + WT 2A3 profile + exact mutation"结构 → **task-incompatible**（不能用于 primary estimand）。
- SHAPE `.rdat`（GSE173083 等）：单条件 SHAPE，非 2A3-MaP M2 mutant-response → task-incompatible。
- 未发现其他 development-disconnected 且任务相容的 public M2 full-spectrum 突变响应数据集。

## joint_dependency / K_preaccess 判定
- `joint_dependency_component_v1` 机制已就绪（outcome-blind）。
- 当前 candidate 池：仅 RMDB/单条件 SHAPE（task-incompatible），无合格 disconnected 组件。
- **`K_preaccess`：无可计划评价的合格组件**（预计为 0 或远低于 `K_required_planned`）。
- `K_eff_realized`：未打开（outcome-blind），保持 null。

## 影响与后续
- P3 若 PASS，进入 P4 前必须先建立独立正 `delta_practical` 与 `delta_power` 计算 `K_required_planned`；
  若 `K_preaccess < K_required_planned` → **不打开 locked outcomes**，输出
  `PUBLIC_EXTERNAL_NOT_QUALIFIED` / `UNDERPOWERED_NOT_CONFIRMATORY_PREACCESS`，转 owner resource review。
- 本合同不授权新湿实验；不拼接不可交换 probe/platform 制造 N。

## 处置
- 不强行把 SHAPE/单条件数据当作 M2 confirmatory。
- 保留 development 结论；P4 资格判定需 owner 提供或指定合格外部数据源后再进行。
