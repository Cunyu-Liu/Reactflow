"""O0-X engineering checks (T-O0.1~T-O0.14, §15.4, §24.2).

Unit-level tests backing the O0-X runner: deterministic eval, CUDA forward/
backward, sanity gradient, tiny-overfit, edge cases, and the independent
reference evaluator.  These are engineering checks only; no scientific model
selection, no test access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.model import EPROConfig, EPROModel  # noqa: E402

from scripts.reactflow_delta.o0x_run import (  # noqa: E402
    _make_batch,
    _make_model,
    check_cuda_forward_backward,
    check_deterministic_eval,
    check_edge_cases,
    check_eval_reference,
    check_sanity_gradient,
    check_tiny_overfit,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestDeterministicEval:
    def test_bitwise_equal(self):
        r = check_deterministic_eval(DEVICE)
        assert r["pass"]


class TestCudaForwardBackward:
    def test_forward_backward(self):
        r = check_cuda_forward_backward(DEVICE)
        assert r["status"] == "PASS"
        assert r["fallback_count"] == 0
        assert r["forward_calls"] >= 1
        assert r["backward_calls"] >= 1
        assert r["grads_finite"]


class TestSanityGradient:
    def test_no_permanent_zero_grad(self):
        r = check_sanity_gradient(DEVICE)
        assert r["no_permanent_zero_grad"]
        assert r["all_finite"]


class TestTinyOverfit:
    def test_tiny_overfit(self):
        r = check_tiny_overfit(DEVICE, epochs=800, lr=1e-4, n_pairs=8)
        # contract 15.4: train error < 1% of constant baseline
        assert r["final_train_error"] is not None
        assert r["final_train_error"] < r["overfit_target"]


class TestEdgeCases:
    def test_edge_cases(self):
        r = check_edge_cases(DEVICE)
        assert r["pass"]


class TestEvalReference:
    def test_reference_skill_defined(self):
        r = check_eval_reference(DEVICE)
        assert r["skill_defined"]
