#!/usr/bin/env python3
"""O0-X Operator Engineering audit (contract §20.9, §15.4).

Verifies the O0-X run manifest produced by o0x_run.py: schema and run identity,
all §15.4 engineering checks PASS (invariants, deterministic eval, CUDA
forward/backward with fallback=0, sanity gradient, tiny-subset overfit, edge
cases, evaluator vs independent reference), and an overall gate_result of PASS.
Engineering-only; no scientific claim, no test unseal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REGISTRY_SCHEMA = "reactflow_delta.o0x_registry.v1"
RUN_ID = "o0x_operator_engineering_20260804_v1"


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(registry_path: Path) -> dict:
    reg = json.loads(registry_path.read_text(encoding="utf-8"))

    checks: dict[str, bool] = {}
    checks["registry_schema_v1"] = reg.get("schema_version") == REGISTRY_SCHEMA
    checks["run_id"] = reg.get("run_id") == RUN_ID
    checks["gate_result_pass"] = reg.get("gate_result") == "PASS"

    c = reg.get("checks", {})

    # A. invariants (45/45, all_pass)
    inv = c.get("invariant_suite", {})
    checks["invariant_suite_all_pass"] = inv.get("all_pass") is True
    checks["invariant_suite_45_45"] = inv.get("n_checks") == 45 and inv.get("n_passed") == 45

    # B. deterministic eval (bitwise equal or max_abs < 1e-8)
    det = c.get("deterministic_eval", {})
    checks["deterministic_eval_pass"] = det.get("pass") is True
    checks["deterministic_bitwise_equal"] = det.get("bitwise_equal") is True
    checks["deterministic_max_abs_lt_1e8"] = (
        _finite(det.get("max_abs_diff")) and det["max_abs_diff"] < 1e-8
    )

    # C. CUDA forward/backward, fallback=0
    cuda = c.get("cuda_forward_backward", {})
    checks["cuda_status_pass"] = cuda.get("status") == "PASS"
    checks["cuda_fallback_zero"] = cuda.get("fallback_count") == 0
    checks["cuda_model_on_cuda"] = cuda.get("model_cuda") is True
    checks["cuda_input_on_cuda"] = cuda.get("input_cuda") is True
    checks["cuda_grads_finite"] = cuda.get("grads_finite") is True
    checks["cuda_pass"] = cuda.get("pass") is True

    # D. sanity gradient (no permanent zero-gradient)
    grad = c.get("sanity_gradient", {})
    checks["gradient_all_finite"] = grad.get("all_finite") is True
    checks["gradient_no_permanent_zero"] = grad.get("no_permanent_zero_grad") is True
    checks["gradient_block_count_ge_1"] = grad.get("non_zero_blocks", 0) >= 1
    checks["gradient_pass"] = grad.get("pass") is True

    # E. tiny-subset overfit (train error < 1% of constant baseline)
    tiny = c.get("tiny_overfit", {})
    checks["tiny_overfit_pass"] = tiny.get("pass") is True
    checks["tiny_n_pairs_8_32"] = _finite(tiny.get("n_pairs")) and 8 <= tiny["n_pairs"] <= 32
    checks["tiny_error_lt_1pct_baseline"] = (
        _finite(tiny.get("final_train_error")) and _finite(tiny.get("overfit_target"))
        and tiny["final_train_error"] < tiny["overfit_target"]
    )

    # F. edge cases
    edge = c.get("edge_cases", {})
    checks["edge_nan_input_handled"] = edge.get("nan_input_handled") is True
    checks["edge_empty_mask_handled"] = edge.get("empty_mask_handled") is True
    checks["edge_long_sequence_handled"] = edge.get("long_sequence_handled") is True
    checks["edge_pass"] = edge.get("pass") is True

    # G. evaluator vs independent reference
    ref = c.get("eval_reference", {})
    checks["eval_reference_pass"] = ref.get("pass") is True
    checks["eval_reference_skill_defined"] = ref.get("skill_defined") is True

    all_pass = all(checks.values())
    return {
        "schema_version": "reactflow_delta.o0x_audit.v1",
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "checks": checks,
        "n_checks": len(checks),
        "n_passed": sum(1 for v in checks.values() if v),
        "all_pass": all_pass,
        "gate_result": "PASS" if all_pass else "FAIL",
        "evidence_class": "ENGINEERING_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.registry.resolve())
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"all_pass": result["all_pass"], "gate_result": result["gate_result"]}))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())