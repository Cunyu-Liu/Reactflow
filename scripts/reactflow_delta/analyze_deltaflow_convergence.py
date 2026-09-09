#!/usr/bin/env python3
"""CFM convergence analysis for freezing the Stage-2 final epoch (Task 6.7/7).

Reads the flow_audit json artifacts written by run_deltaflow_fold.py,
summarizes the candidate/null loss curves across folds, and proposes the
final epoch by the frozen decision rule: the smallest epoch E such that
(1) the mean relative loss reduction over the trailing 10-epoch window
drops below 1% (per arm, pooled across folds), and (2) E is at least 40.
The proposal is reported only -- the freeze itself is a Task 7 commit
decision, never automatic.

Usage:
  python analyze_deltaflow_convergence.py --audit-dir DIR [--min-epochs 40]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TRAILING_WINDOW = 10
RELATIVE_REDUCTION_THRESHOLD = 0.01
DEFAULT_MIN_EPOCHS = 40


def load_histories(audit_dir: Path) -> list[dict]:
    audits = sorted(audit_dir.glob("deltaflow_flow_audit_fold*_seed*.json"))
    if not audits:
        raise FileNotFoundError(f"no flow audit artifacts under {audit_dir}")
    rows = []
    for path in audits:
        data = json.loads(path.read_text(encoding="utf-8"))
        candidate = np.asarray(data["candidate_history"], dtype=np.float64)
        null = np.asarray(data["null_history"], dtype=np.float64)
        if candidate.shape != null.shape or candidate.size == 0:
            raise ValueError(f"history shape mismatch in {path}")
        if not (np.isfinite(candidate).all() and np.isfinite(null).all()):
            raise ValueError(f"nonfinite history in {path}")
        rows.append(
            {
                "path": str(path),
                "fold": int(data["outer_fold"]),
                "seed": int(data["seed"]),
                "epochs": int(data["epochs"]),
                "candidate": candidate,
                "null": null,
                "n_train_mutants": int(data["n_flow_train_mutants"]),
            }
        )
    epochs = {row["epochs"] for row in rows}
    if len(epochs) != 1:
        raise ValueError(f"audit epoch schedules differ across folds: {sorted(epochs)}")
    return rows


def propose_epoch(
    curves: np.ndarray, *, min_epochs: int, window: int, threshold: float
) -> tuple[int, list[float]]:
    """curves: [n_folds, n_epochs] mean loss per epoch."""
    pooled = curves.mean(axis=0)
    n_epochs = pooled.shape[0]
    reductions: list[float] = []
    for epoch in range(1, n_epochs):
        start = max(0, epoch - window)
        base = float(pooled[start:epoch].mean())
        current = float(pooled[epoch])
        reductions.append((base - current) / base if base > 0 else float("nan"))
    proposal = n_epochs
    for index, reduction in enumerate(reductions, start=1):
        epoch = index + 1
        if epoch >= min_epochs and np.isfinite(reduction) and reduction < threshold:
            proposal = epoch
            break
    return proposal, reductions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS)
    parser.add_argument("--window", type=int, default=TRAILING_WINDOW)
    parser.add_argument("--threshold", type=float, default=RELATIVE_REDUCTION_THRESHOLD)
    args = parser.parse_args()

    rows = load_histories(args.audit_dir)
    n_epochs = rows[0]["epochs"]
    candidate_matrix = np.stack([row["candidate"] for row in rows])
    null_matrix = np.stack([row["null"] for row in rows])
    print(f"folds={len(rows)} epochs={n_epochs} (seeds: {sorted({r['seed'] for r in rows})})")
    for label, matrix in (("candidate", candidate_matrix), ("null", null_matrix)):
        pooled = matrix.mean(axis=0)
        print(
            f"{label}: pooled first={pooled[0]:.6f} epoch{args.min_epochs}="
            f"{pooled[args.min_epochs - 1]:.6f} last={pooled[-1]:.6f}"
        )
    for label, matrix in (("candidate", candidate_matrix), ("null", null_matrix)):
        proposal, reductions = propose_epoch(
            matrix,
            min_epochs=args.min_epochs,
            window=args.window,
            threshold=args.threshold,
        )
        window_at = reductions[proposal - 2] if proposal >= 2 else float("nan")
        print(
            f"{label}: proposed final epoch = {proposal} "
            f"(trailing-window relative reduction {window_at:.4f} < "
            f"{args.threshold}; rule: smallest epoch >= {args.min_epochs})"
        )
    print(
        "NOTE: this is a proposal only; the Stage-2 final-epoch freeze is a "
        "Task 7 pre-registration commit decision."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
