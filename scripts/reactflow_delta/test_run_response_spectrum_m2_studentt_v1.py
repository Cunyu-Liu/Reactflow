#!/usr/bin/env python3
"""test_run_response_spectrum_m2_studentt_v1 — unit tests for the Student-t runner.

Imports the model module (residual_spectrum_v6) directly and checks the runner's
constants via AST so we don't pull in server-only imports (run_baselines_v6 etc).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_spectrum_v6 as rsv6  # noqa: E402

RUNNER = Path(__file__).resolve().parent / "run_response_spectrum_m2_studentt_v1.py"


def _runner_constants():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out


def test_runner_constants():
    c = _runner_constants()
    assert c.get("MODEL_VARIANT") == "wmae_resid_studentt_spectrum"
    assert c.get("BASELINE") == "wmed_spectrum"
    assert c.get("POS_DIM") == 7
    assert c.get("MODEL_ID") == "response_spectrum_m2_studentt_v1"


def test_model_functions_exist():
    assert callable(rsv6.train_posaware_student_t)
    assert callable(rsv6.predict_posaware_student_t)
    assert callable(rsv6.split_pos_glob)
    assert callable(rsv6._student_t_nll)


def test_defaults_sane():
    assert rsv6.DEFAULT_NU == 4.0
    assert rsv6.DEFAULT_EPOCHS == 30
    assert rsv6.DEFAULT_NLAYERS == 1
