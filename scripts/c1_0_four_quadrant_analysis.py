#!/usr/bin/env python3
"""C1-0 Task 4: Four-quadrant localization analysis.

This script isolates the source of the F1 gap between the eFold baseline
(F1 ~ 0.21) and the ReactFlow model (F1 ~ 0.026) by crossing two model
sources with two evaluators on REAL RNA structure data:

    Quadrant A = eFold model predictions + eFold official evaluator (efold.core.metrics.f1)
    Quadrant B = eFold model predictions + ReactFlow evaluator (reactflow.metrics.f1_score)
    Quadrant C = ReactFlow model predictions + eFold official evaluator
    Quadrant D = ReactFlow model predictions + ReactFlow evaluator

If the evaluators agree on real data (A ~= B and C ~= D), the F1 gap is
attributable to the MODEL (or its wrapper), NOT the evaluator.  This
complements Task 3 (dual evaluator alignment on synthetic fixtures) by
running the same comparison on real RNA secondary-structure predictions.

Data sources
------------
eFold predictions (Quadrants A, B):
    /home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012/
        baselines/efold_same_split/predictions/<tier>.efold.predictions.jsonl
    Format: JSONL with ``predicted_pairs``, ``sequence``, ``source_id``.
    Ground truth is matched from the cache JSONL by ``source_id``.

ReactFlow predictions (Quadrants C, D):
    /home/cunyuliu/reactflow_c0_stage_20260718/c0_artifacts/final_evaluation/
        predictions.jsonl
    Format: JSONL with ``predicted_pairs``, ``target_pairs``, ``mode``,
    ``source_id``, ``sequence_length``.  The ``calibrated_marginal`` mode
    (C0 default decoder) is used.  Each record already carries its target.

Equivalence rationale
---------------------
For a symmetric binary pair matrix with zero diagonal, the eFold
full-matrix sum doubles every quantity consistently, so
``efold_f1 = 2*TP/(2*TP+FP+FN)`` which is identical to ReactFlow's
upper-triangle F1.  The only documented disagreement is the
empty-vs-empty convention: eFold returns 1.0 (``sum_pair == 0`` branch)
while ReactFlow returns 0.0 (``denominator == 0``).  This is recorded
per-sample and excluded from the "unexpected difference" count.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import List, Sequence, Tuple

import torch

# Make efold (isolated venv), reactflow src, and helper scripts importable.
_EFOLD_SITE_PACKAGES = (
    "/home/cunyuliu/reactflow_external_envs/efold_py310/lib/python3.10/site-packages"
)
_STAGE_ROOT = "/home/cunyuliu/reactflow_c1_0_stage_20260721"
sys.path.insert(0, _EFOLD_SITE_PACKAGES)
sys.path.insert(0, os.path.join(_STAGE_ROOT, "src"))
sys.path.insert(0, os.path.join(_STAGE_ROOT, "scripts"))

from efold.core.metrics import f1 as efold_f1  # noqa: E402
from reactflow.constraints import pairs_to_matrix  # noqa: E402
from reactflow.metrics import f1_score as reactflow_matrix_f1  # noqa: E402


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Tolerance for eFold-vs-ReactFlow agreement.  eFold's f1 returns a Python
# float via ``.item()`` on a torch.float32 scalar, so its ULP error for
# values near 1.0 is ~1.2e-7 (float32 epsilon).  1e-6 leaves comfortable
# headroom while still catching genuine formula disagreements.
EFOLD_TOL = 1e-6

# Sample sizes (kept small for speed; task asks for 50-100 per source).
EFOLD_SAMPLE_SIZE = 100
REACTFLOW_SAMPLE_SIZE = 100

# Data paths.
EFOLD_PRED_DIR = (
    "/home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012"
    "/baselines/efold_same_split/predictions"
)
CACHE_DIR = (
    "/home/cunyuliu/reactflow/artifacts/full_runs/full_ablation_20260709_003012/cache"
)
REACTFLOW_PRED_PATH = (
    "/home/cunyuliu/reactflow_c0_stage_20260718/c0_artifacts/final_evaluation"
    "/predictions.jsonl"
)

# eFold prediction tiers to draw Quadrant A/B samples from.  PDB is fully
# matched to its cache (333/333) and covers a range of lengths, so it is the
# cleanest real-data source.  archiveII is available as a fallback/extra.
EFOLD_TIERS = ["PDB", "archiveII"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _build_matrix(pairs: Sequence[Tuple[int, int]], size: int) -> List[List[float]]:
    """Build a symmetric binary matrix from a list of (i, j) pairs."""

    return [list(row) for row in pairs_to_matrix(list(pairs), size)]


def _matrix_to_torch(matrix: Sequence[Sequence[float]]) -> torch.Tensor:
    """Convert a list-of-lists matrix to a torch.float32 tensor."""

    return torch.tensor(matrix, dtype=torch.float32)


def _efold_f1_from_pairs(
    pred_pairs: Sequence[Tuple[int, int]],
    target_pairs: Sequence[Tuple[int, int]],
    size: int,
) -> float:
    """Run eFold's official f1 on pair lists."""

    if size <= 0:
        # eFold's sum_pair == 0 branch fires on an empty tensor -> 1.0.
        pred_tensor = torch.zeros((0, 0), dtype=torch.float32)
        target_tensor = torch.zeros((0, 0), dtype=torch.float32)
    else:
        pred_tensor = _matrix_to_torch(_build_matrix(pred_pairs, size))
        target_tensor = _matrix_to_torch(_build_matrix(target_pairs, size))
    return float(efold_f1(pred_tensor, target_tensor))


def _reactflow_f1_from_pairs(
    pred_pairs: Sequence[Tuple[int, int]],
    target_pairs: Sequence[Tuple[int, int]],
    size: int,
) -> float:
    """Run ReactFlow's matrix f1_score on pair lists."""

    if size <= 0:
        # ReactFlow convention: empty-vs-empty -> 0.0.
        return 0.0
    pred_matrix = _build_matrix(pred_pairs, size)
    target_matrix = _build_matrix(target_pairs, size)
    return float(reactflow_matrix_f1(pred_matrix, target_matrix))


def _classify_difference(
    pred_pairs: Sequence[Tuple[int, int]],
    target_pairs: Sequence[Tuple[int, int]],
    efold_val: float,
    reactflow_val: float,
) -> str:
    """Return a human-readable reason for any evaluator disagreement."""

    if abs(efold_val - reactflow_val) < EFOLD_TOL:
        return "none"

    # Empty-vs-empty convention: eFold returns 1.0, ReactFlow returns 0.0.
    if (
        len(pred_pairs) == 0
        and len(target_pairs) == 0
        and abs(efold_val - 1.0) < EFOLD_TOL
        and abs(reactflow_val - 0.0) < EFOLD_TOL
    ):
        return "empty_vs_empty_convention"

    # One side empty (e.g. all-unpaired target with non-empty prediction, or
    # vice versa).  Both evaluators should still agree because the
    # denominator is non-zero.  If they disagree here it is unexpected.
    return (
        "unexpected_difference(efold=%s, reactflow=%s, pred_pairs=%d, "
        "target_pairs=%d)"
        % (efold_val, reactflow_val, len(pred_pairs), len(target_pairs))
    )


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def _load_cache_index() -> dict:
    """Build a source_id -> {pairs, sequence} index from cache JSONL files."""

    cache_idx = {}
    for fn in ["PDB.jsonl", "archiveII.jsonl", "human_mRNA.jsonl",
               "viral.jsonl", "lncRNA.jsonl"]:
        path = os.path.join(CACHE_DIR, fn)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                cache_idx[d["source_id"]] = {
                    "pairs": d["pairs"],
                    "sequence": d["sequence"],
                    "cache_file": fn,
                }
    return cache_idx


def _load_efold_predictions(
    cache_idx: dict,
    sample_size: int,
) -> List[dict]:
    """Load eFold predictions matched to cache ground truth.

    Returns a list of dicts with keys: source_id, sequence, length,
    predicted_pairs, target_pairs, tier.
    """

    records = []
    for tier in EFOLD_TIERS:
        path = os.path.join(EFOLD_PRED_DIR, "%s.efold.predictions.jsonl" % tier)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                sid = d["source_id"]
                if sid not in cache_idx:
                    continue
                cache = cache_idx[sid]
                seq = d.get("sequence") or cache["sequence"]
                # Sanity: the eFold prediction sequence must match the cache
                # sequence for the target pairs to be valid.
                if cache["sequence"] and seq and seq != cache["sequence"]:
                    continue
                records.append({
                    "source_id": sid,
                    "sequence": seq,
                    "length": len(seq),
                    "predicted_pairs": [tuple(p) for p in d["predicted_pairs"]],
                    "target_pairs": [tuple(p) for p in cache["pairs"]],
                    "tier": tier,
                })
        if len(records) >= sample_size:
            break
    return records[:sample_size]


def _load_reactflow_predictions(sample_size: int) -> List[dict]:
    """Load ReactFlow calibrated_marginal predictions (target_pairs inline).

    Returns a list of dicts with keys: source_id, length, predicted_pairs,
    target_pairs, tier, mode.
    """

    records = []
    with open(REACTFLOW_PRED_PATH, "rb") as f:
        for raw in f:
            try:
                d = json.loads(raw)
            except Exception:
                # One malformed line is known; skip defensively.
                continue
            if d.get("mode") != "calibrated_marginal":
                continue
            length = d.get("sequence_length")
            if not isinstance(length, int) or length <= 0:
                continue
            records.append({
                "source_id": d.get("source_id"),
                "length": length,
                "predicted_pairs": [tuple(p) for p in d.get("predicted_pairs", [])],
                "target_pairs": [tuple(p) for p in d.get("target_pairs", [])],
                "tier": d.get("tier"),
                "mode": d.get("mode"),
            })
            if len(records) >= sample_size:
                break
    return records[:sample_size]


# ----------------------------------------------------------------------------
# Quadrant evaluation
# ----------------------------------------------------------------------------

def _evaluate_one(record: dict) -> dict:
    """Run both evaluators on one (prediction, target) pair."""

    pred_pairs = record["predicted_pairs"]
    target_pairs = record["target_pairs"]
    size = record["length"]

    efold_val = _efold_f1_from_pairs(pred_pairs, target_pairs, size)
    reactflow_val = _reactflow_f1_from_pairs(pred_pairs, target_pairs, size)
    reason = _classify_difference(
        pred_pairs, target_pairs, efold_val, reactflow_val
    )

    return {
        "source_id": record.get("source_id"),
        "tier": record.get("tier"),
        "length": size,
        "predicted_pair_count": len(pred_pairs),
        "target_pair_count": len(target_pairs),
        "efold_f1": efold_val,
        "reactflow_f1": reactflow_val,
        "agree": reason == "none",
        "difference_reason": reason,
    }


def _quadrant_summary(name: str, results: List[dict]) -> dict:
    """Aggregate mean F1 and agreement stats for one quadrant.

    Note: a "quadrant" here is (model source, evaluator).  We compute the
    evaluator-specific mean F1 across the same set of samples, then check
    whether the two evaluators agree per-sample.
    """

    if not results:
        return {
            "quadrant": name,
            "sample_count": 0,
            "mean_f1": None,
            "mean_f1_non_empty": None,
            "agree_count": 0,
            "empty_vs_empty_count": 0,
            "unexpected_difference_count": 0,
            "all_agree_non_empty": None,
        }

    efold_vals = [r["efold_f1"] for r in results]
    reactflow_vals = [r["reactflow_f1"] for r in results]

    # Non-empty samples (exclude the empty-vs-empty convention case where
    # both prediction and target are empty).  These are the samples where
    # evaluator agreement is mathematically required.
    non_empty = [
        r for r in results
        if r["difference_reason"] != "empty_vs_empty_convention"
    ]
    non_empty_agree = sum(1 for r in non_empty if r["agree"])
    unexpected = [
        r for r in results
        if r["difference_reason"] not in ("none", "empty_vs_empty_convention")
    ]
    empty_vs_empty = sum(
        1 for r in results if r["difference_reason"] == "empty_vs_empty_convention"
    )

    # The "quadrant F1" is the evaluator-specific mean.  For quadrants A and
    # C we report efold_f1; for B and D we report reactflow_f1.
    if name in ("A", "C"):
        mean_f1 = mean(efold_vals) if efold_vals else None
        mean_f1_non_empty = (
            mean([r["efold_f1"] for r in non_empty]) if non_empty else None
        )
    else:
        mean_f1 = mean(reactflow_vals) if reactflow_vals else None
        mean_f1_non_empty = (
            mean([r["reactflow_f1"] for r in non_empty]) if non_empty else None
        )

    return {
        "quadrant": name,
        "sample_count": len(results),
        "mean_f1": mean_f1,
        "mean_f1_non_empty": mean_f1_non_empty if non_empty else None,
        "non_empty_sample_count": len(non_empty),
        "agree_count": sum(1 for r in results if r["agree"]),
        "empty_vs_empty_count": empty_vs_empty,
        "unexpected_difference_count": len(unexpected),
        "all_agree_non_empty": (
            non_empty_agree == len(non_empty) if non_empty else None
        ),
        "unexpected_differences": unexpected,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    cache_idx = _load_cache_index()
    print("cache entries loaded: %d" % len(cache_idx))

    efold_records = _load_efold_predictions(cache_idx, EFOLD_SAMPLE_SIZE)
    print("efold predictions loaded: %d" % len(efold_records))

    reactflow_records = _load_reactflow_predictions(REACTFLOW_SAMPLE_SIZE)
    print("reactflow predictions loaded: %d" % len(reactflow_records))

    # Quadrants A & B: eFold model predictions, both evaluators.
    efold_results = [_evaluate_one(r) for r in efold_records]
    quadrant_A = _quadrant_summary("A", efold_results)
    quadrant_B = _quadrant_summary("B", efold_results)

    # Quadrants C & D: ReactFlow model predictions, both evaluators.
    reactflow_results = [_evaluate_one(r) for r in reactflow_records]
    quadrant_C = _quadrant_summary("C", reactflow_results)
    quadrant_D = _quadrant_summary("D", reactflow_results)

    # Cross-quadrant evaluator alignment: A vs B and C vs D must agree on
    # non-empty samples.  The model gap is A/B vs C/D.
    ab_align = (
        quadrant_A["all_agree_non_empty"] and quadrant_B["all_agree_non_empty"]
    )
    cd_align = (
        quadrant_C["all_agree_non_empty"] and quadrant_D["all_agree_non_empty"]
    )

    # Conclusion: if A~=B and C~=D, the evaluator is NOT the source of the
    # gap.  The gap is then attributable to the model (or its wrapper).
    evaluator_exonerated = bool(ab_align and cd_align)
    if evaluator_exonerated:
        conclusion = (
            "Evaluators agree on real RNA data (A~=B and C~=D on non-empty "
            "samples).  The F1 gap is NOT attributable to the evaluator.  "
            "It is attributable to the ReactFlow MODEL (or its inference "
            "wrapper): eFold predictions score ~0.21 while ReactFlow "
            "predictions score ~0.026 under the SAME evaluator."
        )
    else:
        conclusion = (
            "Evaluators disagree on some real-data samples; the evaluator "
            "cannot be fully exonerated.  Inspect unexpected_differences."
        )

    summary = {
        "schema_version": 1,
        "description": (
            "Four-quadrant localization: cross two model sources (eFold, "
            "ReactFlow) with two evaluators (efold.core.metrics.f1, "
            "reactflow.metrics.f1_score) on REAL RNA secondary-structure "
            "data.  A~=B and C~=D implies the evaluator is NOT the source "
            "of the F1 gap; the gap is then attributable to the model or "
            "its wrapper."
        ),
        "quadrants_definition": {
            "A": "eFold model predictions + eFold official evaluator (efold.core.metrics.f1)",
            "B": "eFold model predictions + ReactFlow evaluator (reactflow.metrics.f1_score)",
            "C": "ReactFlow model predictions + eFold official evaluator",
            "D": "ReactFlow model predictions + ReactFlow evaluator",
        },
        "tolerance_efold_vs_reactflow": EFOLD_TOL,
        "data_sources": {
            "efold_predictions_dir": EFOLD_PRED_DIR,
            "efold_tiers_used": EFOLD_TIERS,
            "cache_dir": CACHE_DIR,
            "reactflow_predictions_path": REACTFLOW_PRED_PATH,
            "reactflow_mode_used": "calibrated_marginal",
        },
        "sample_sizes": {
            "efold": len(efold_records),
            "reactflow": len(reactflow_records),
        },
        "quadrants": {
            "A": quadrant_A,
            "B": quadrant_B,
            "C": quadrant_C,
            "D": quadrant_D,
        },
        "evaluator_alignment": {
            "A_approx_B_non_empty": ab_align,
            "C_approx_D_non_empty": cd_align,
            "evaluator_exonerated": evaluator_exonerated,
        },
        "model_gap": {
            "efold_mean_f1_evaluator_A": quadrant_A["mean_f1"],
            "efold_mean_f1_evaluator_B": quadrant_B["mean_f1"],
            "reactflow_mean_f1_evaluator_C": quadrant_C["mean_f1"],
            "reactflow_mean_f1_evaluator_D": quadrant_D["mean_f1"],
        },
        "conclusion": conclusion,
        "samples": {
            "efold": efold_results,
            "reactflow": reactflow_results,
        },
    }

    output_path = Path("artifacts/c1_0/efold_four_quadrant_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "quadrant_A_mean_f1": quadrant_A["mean_f1"],
                "quadrant_B_mean_f1": quadrant_B["mean_f1"],
                "quadrant_C_mean_f1": quadrant_C["mean_f1"],
                "quadrant_D_mean_f1": quadrant_D["mean_f1"],
                "A_approx_B": ab_align,
                "C_approx_D": cd_align,
                "evaluator_exonerated": evaluator_exonerated,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
