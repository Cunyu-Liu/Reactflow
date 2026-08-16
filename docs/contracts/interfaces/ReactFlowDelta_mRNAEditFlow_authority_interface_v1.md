---
schema_version: reactflow_delta.cross_contract_interface.v1
interface_id: reactflow_delta__mrna_editflow_authority_v1
change_type: ADDITIVE_ONLY
status: REACTFLOW_DELTA_ORACLE_NOT_AUTHORIZED
created_at: "2026-08-03T12:18:59Z"
producer:
  project_task_id: reactflow_delta
  contract_id: reactflow_delta_v4_data_first
  contract_path: docs/contracts/ReactFlowDelta科研合同_v4_data_first.md
  contract_sha256: 631962f88790103aa3383c9ed22de2943f6874455b4fcb587e18eb2a7d277c15
  active_manifest_path: configs/reactflow_delta/active_contract.yaml
  authorization_status: DRAFT_AWAITING_USER_SIGNOFF
  training_allowed: false
  required_gate_ids:
    - E0-X
    - UNCERTAINTY_CALIBRATION
    - OOD_ABSTENTION
  export_manifest: null
consumer:
  project_task_id: mrna_editflow
  goal_contract_id: utr_editflow_goal_v2
  base_goal_path: /Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/提示词/mrna 最新构建合同-先做.md
  base_goal_sha256_before_interface: 3a3a654ca5c10a988eca897bff40be2e0b45c841f744f7423fdfd60b298b5791
  authority_marker: REACTFLOW_DELTA_ORACLE_NOT_AUTHORIZED
  approval_status: NOT_PROVIDED
  new_consumer_amendment_required: true
  permitted_roles: []
data_governance:
  test_label_sharing_allowed: false
  heldout_exposure_sharing_allowed: false
  static_pretraining_cross_use_allowed: false
  exposure_ledger_required: true
  exposure_ledger_sha256: null
claim_boundary:
  producer_engineering_pass_upgrades_consumer_science: false
  consumer_engineering_pass_upgrades_producer_science: false
  predictions_may_be_called_measurements: false
  automatic_activation_after_e0: false
  guidance_or_selection_may_also_be_independent_final_evaluator: false
---

# ReactFlow-Delta → mRNA-EditFlow authority interface v1

这是对 mRNA-EditFlow 合同的最小加法接口，不修改其核心科学问题、Edit Flow 主方法地位、UTR 范围或既有证据状态。

当前固定状态为 `REACTFLOW_DELTA_ORACLE_NOT_AUTHORIZED`。ReactFlow-Delta 的数据、测试、训练 Gate、工程结果和论文证据不会自动继承给 mRNA-EditFlow；两个项目不得共享确认性 test labels 或把对方的 held-out exposure 写成未暴露。

只有 `E0-X`、`UNCERTAINTY_CALIBRATION` 和 `OOD_ABSTENTION` 三项 Gate 均以完整 finalizer、manifest 和 checksum evidence 得到 `PASS`，并完成去污染 exposure ledger 后，ReactFlow-Delta 才能成为候选 producer。`E0-X` 在这里专指一次性 sealed scientific evaluation，不是 operator unit test 或 GPU engineering Gate。即便如此，mRNA-EditFlow 仍须通过新的、用户明确批准的 consumer amendment 冻结用途，状态才可改变；不得自动激活。

未激活前，mRNA-EditFlow 必须显示 `REACTFLOW_DELTA_ORACLE_NOT_AUTHORIZED`。静态预训练数据也不能在当前状态跨项目复用；未来只有 exact/parent/family/structure/source exposure ledger 完整且双方 amendment 明确许可时才能申请。激活后必须在 `GUIDANCE_CRITIC`、`SELECTOR` 或 `EVALUATION_ONLY_ORACLE` 中只选择一个角色；用于 guidance/selection 的组件不得同时作为 independent final evaluator。任何一方的工程 PASS 都不得升级另一方的科学状态，模型预测不得写成实验测量。

任何未来跨项目 run 必须同时验证 producer contract/export manifest 与 consumer contract/import amendment 的 raw-byte hash、Git/source origin、角色和 exposure ledger；任一缺失或漂移都在模型/数据加载前 fail closed。当前 V4 的数据、split、config、exposure hashes 为空，所有跨项目输出均未授权。
