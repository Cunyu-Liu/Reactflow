#!/usr/bin/env python3
"""Run ReactFlow symbolic checks from a source checkout."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reactflow.symbolic import run_all_symbolic_checks  # noqa: E402


def main() -> int:
    """Print symbolic verification results and return non-zero on failure."""

    results = run_all_symbolic_checks()
    print(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True))
    residuals = []
    for check in results.values():
        residuals.extend(value for key, value in check.items() if key.startswith("residual"))
    return 0 if all(value == "0" for value in residuals) else 1


if __name__ == "__main__":
    raise SystemExit(main())
