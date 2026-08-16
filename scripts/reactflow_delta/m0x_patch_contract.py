#!/usr/bin/env python3
"""Patch active_contract.yaml to authorize M0-X (epoch 12)."""
from pathlib import Path

p = Path("configs/reactflow_delta/active_contract.yaml")
s = p.read_text(encoding="utf-8")

M0X_AMEND = "add97a4863ec4680fb8fd811bf7511ad58448b9ff8c4ab69bc1f79a1a8fca6c6"
M0X_APPROVAL = "dce0809b6dd3f7e813d294602a5bed27b5255efe23bfc17714cb303df042c820"

def rep(old, new, label):
    global s
    assert old in s, f"not found: {label}"
    s = s.replace(old, new, 1)

# 1. authorization status
rep("  status: ACTIVE_O0X", "  status: ACTIVE_M0X", "auth status")
# 2. approval record path
rep("  approval_record: docs/approvals/reactflow_delta_v4_o0x_approval_20260804.yaml",
    "  approval_record: docs/approvals/reactflow_delta_v4_m0x_approval_20260804.yaml",
    "approval record path")
# 3. approval record sha
rep("  approval_record_sha256: beea3847f98e16f5d5fc35176922d1866f0f4627e7d57ece7db2f0add99e2983",
    f"  approval_record_sha256: {M0X_APPROVAL}", "approval record sha")
# 4. approval_scope
rep("  approval_scope:\n  - V4_SIGNOFF\n  - CONDITIONAL_PH0X_ONLY_AUTHORITY",
    "  approval_scope:\n  - V4_SIGNOFF\n  - M0X_CONTROLLED_DEVELOPMENT\n  - window_id: m0x_dev_window_20260804",
    "approval scope")
# 5. allowed_phases add M0-X
rep("  - B0-X\n  - O0-X\n  runnable_phases:",
    "  - B0-X\n  - O0-X\n  - M0-X\n  runnable_phases:", "allowed phases")
# 6. runnable_phases
rep("  runnable_phases:\n  - O0-X", "  runnable_phases:\n  - M0-X", "runnable phases")

# 7. authority_epoch
rep("  authority_epoch: 11", "  authority_epoch: 12", "authority epoch")
# 8. current_phase
rep("  current_phase: O0-X", "  current_phase: M0-X", "current phase")
# 9. authority state
rep("  current_authority_state: O0X_CLOSED_AWAIT_M0X",
    "  current_authority_state: M0X_AUTHORIZED", "authority state")
# 10. current_runnable_phase
rep("  current_runnable_phase: O0-X", "  current_runnable_phase: M0-X", "runnable phase")

# 11. bindings: add m0x after o0x_plan_doc_sha256
rep("  o0x_plan_doc_sha256: e26c2f8e4127de06a5e9c3471c9e7c6eb0ed94e7cd727e2eea880801d110219c",
    "  o0x_plan_doc_sha256: e26c2f8e4127de06a5e9c3471c9e7c6eb0ed94e7cd727e2eea880801d110219c\n"
    "  m0x_amendment_path: docs/contracts/amendments/reactflow_delta_v4_m0x_20260804.yaml\n"
    f"  m0x_amendment_sha256: {M0X_AMEND}\n"
    "  m0x_approval_record_path: docs/approvals/reactflow_delta_v4_m0x_approval_20260804.yaml\n"
    f"  m0x_approval_record_sha256: {M0X_APPROVAL}\n"
    "  m0x_window_id: m0x_dev_window_20260804",
    "m0x bindings")

# 12. M0-X phase execution_authorized true + lifecycle RUNNING
rep("""- phase_id: M0-X
  dependencies:
  - O0-X
  required_gate: TIER_B_PLUS
  lifecycle_status: PLANNED
  gate_result: NOT_RUN
  evidence_class: DEVELOPMENT_ONLY
  execution_authorized: false""",
    """- phase_id: M0-X
  dependencies:
  - O0-X
  required_gate: TIER_B_PLUS
  lifecycle_status: RUNNING
  gate_result: NOT_RUN
  evidence_class: DEVELOPMENT_ONLY
  execution_authorized: true""",
    "M0-X phase")

p.write_text(s, encoding="utf-8")
print("active_contract patched for M0-X")