# ReactFlow-Delta prospective-v2: Submission Materials Package

> Auto-generated 2026-08-14 from locked artifacts (branch
> codex/reactflow-delta-prospective-v2-20260813 @ 13d34ac).
> P6_REPRODUCIBILITY_DELIVERED; release decision is owner-controlled.

## Contents

| File | Purpose |
|------|---------|
| pr_readiness_checklist.md | PR automated check list + all gate states (A-E) |
| supplementary_data_manifest.md | Data availability + artifacts + results + attrition |
| declaration_statements.md | Data/code availability, competing interests, claims, limitations |
| out/main_tables.md / .tex | Main manuscript tables |
| out/figures/fig1..fig4.png | Main figures |
| out/cards.md | Model/Data/Code cards |
| environment.yml | Conda environment spec |

## Gate summary

P0 PASS | P1 FAIL_CLOSED_OPEN | P2 PROSPECTIVE_SIGNAL | P3 NO_INCREMENTAL_LRSO |
P4 EXTERNAL_STATISTICAL_PASS + CALIBRATION | P5 MECHANISM_NOT_ESTABLISHED | P6 REPLAY_CONSISTENT

## How to verify

CI detected...
2 channel Terms of Service accepted
Retrieving notices: - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / - \ | / done

==================================== ERRORS ====================================
__________ ERROR collecting tests/reactflow_delta/test_evaluate_v4.py __________
ImportError while importing test module '/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/tests/reactflow_delta/test_evaluate_v4.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../../anaconda3/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/reactflow_delta/test_evaluate_v4.py:15: in <module>
    import evaluate_v4 as v4
scripts/reactflow_delta/evaluate_v4.py:31: in <module>
    from evaluate_v2 import (
E   ModuleNotFoundError: No module named 'evaluate_v2'
__________ ERROR collecting tests/reactflow_delta/test_evaluate_v5.py __________
ImportError while importing test module '/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/tests/reactflow_delta/test_evaluate_v5.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../../anaconda3/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/reactflow_delta/test_evaluate_v5.py:14: in <module>
    import evaluate_v5 as v5
scripts/reactflow_delta/evaluate_v5.py:36: in <module>
    from evaluate_v2 import (
E   ModuleNotFoundError: No module named 'evaluate_v2'
_____ ERROR collecting tests/reactflow_delta/test_m0x_dev12_regression.py ______
ImportError while importing test module '/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/tests/reactflow_delta/test_m0x_dev12_regression.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../../anaconda3/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/reactflow_delta/test_m0x_dev12_regression.py:14: in <module>
    from scripts.reactflow_delta.m0x_epro_dev12_regression import (  # noqa: E402
scripts/reactflow_delta/m0x_epro_dev12_regression.py:53: in <module>
    import m0x_epro_dev06 as dev06  # noqa: E402  (feature pipeline + pair records)
scripts/reactflow_delta/m0x_epro_dev06.py:66: in <module>
    from b0x_baselines import run_baseline, _pair_scale, _build_features as p2_features  # noqa: E402
E   ModuleNotFoundError: No module named 'b0x_baselines'
___ ERROR collecting tests/reactflow_delta/test_m0x_magnitude_calibration.py ___
ImportError while importing test module '/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/tests/reactflow_delta/test_m0x_magnitude_calibration.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../../anaconda3/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/reactflow_delta/test_m0x_magnitude_calibration.py:19: in <module>
    from scripts.reactflow_delta.m0x_dev12_magnitude_calibration import (  # noqa: E402
scripts/reactflow_delta/m0x_dev12_magnitude_calibration.py:41: in <module>
    from b0x_baselines import _pair_scale  # noqa: E402
E   ModuleNotFoundError: No module named 'b0x_baselines'
____________ ERROR collecting tests/reactflow_delta/test_pair_v2.py ____________
ImportError while importing test module '/Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/tests/reactflow_delta/test_pair_v2.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../../anaconda3/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/reactflow_delta/test_pair_v2.py:15: in <module>
    from models.pair_v2 import build_scheme2_features, POS_DIM, _condition_feat
scripts/reactflow_delta/models/pair_v2.py:25: in <module>
    from run_p2_v3 import (  # noqa: E402
run_p2_v3.py:47: in <module>
    from caller_v3 import CallerV3
caller_v3.py:43: in <module>
    from caller_v2 import (
E   ModuleNotFoundError: No module named 'caller_v2'
=========================== short test summary info ============================
ERROR tests/reactflow_delta/test_evaluate_v4.py
ERROR tests/reactflow_delta/test_evaluate_v5.py
ERROR tests/reactflow_delta/test_m0x_dev12_regression.py
ERROR tests/reactflow_delta/test_m0x_magnitude_calibration.py
ERROR tests/reactflow_delta/test_pair_v2.py
!!!!!!!!!!!!!!!!!!! Interrupted: 5 errors during collection !!!!!!!!!!!!!!!!!!!!
5 errors in 0.92s
