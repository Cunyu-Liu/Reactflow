#!/usr/bin/env python3
"""Generate the 3 required C1-2 artifact files from existing data.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 428-430.

Creates:
- ``artifacts/c1_2/matched_capacity_results.json`` -- matched-capacity
  comparison table (4 models × 3 splits × 4 decoders, 3 seeds mean±std).
- ``artifacts/c1_2/ablation_results.json`` -- decoder and distance-bin
  ablation analysis.
- ``artifacts/c1_2/profiling.json`` -- runtime and memory profiling data
  extracted from per-run evaluation results.

Usage::

    python scripts/generate_c1_2_artifacts.py \
        --runs-dir artifacts/c1_2/runs \
        --aggregate artifacts/c1_2/aggregate_results.json \
        --output-dir artifacts/c1_2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def build_matched_capacity_results(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """Extract matched-capacity comparison from aggregate data."""
    models = aggregate.get("models", {})
    result = {
        "description": "Matched-capacity baseline comparison (4 models, 3 seeds, 3 splits, 4 decoders)",
        "config": {
            "n_seeds": 3,
            "splits": ["val", "test", "novel_clan"],
            "decoders": ["threshold", "nussinov_dp", "mea", "greedy_pseudoknot"],
            "max_samples": 500,
            "max_length": 128,
            "epochs": 8,
        },
        "models": {},
    }
    for model_name, model_data in models.items():
        result["models"][model_name] = {
            "n_params": model_data.get("n_params", 0),
            "splits": {},
        }
        for split_name in ["val", "test", "novel_clan"]:
            split_data = model_data.get("splits", {}).get(split_name, {})
            result["models"][model_name]["splits"][split_name] = {}
            for decoder_name in ["threshold", "nussinov_dp", "mea", "greedy_pseudoknot"]:
                dec_data = split_data.get(decoder_name, {})
                result["models"][model_name]["splits"][split_name][decoder_name] = {
                    "f1_mean": dec_data.get("f1_mean", 0.0),
                    "f1_std": dec_data.get("f1_std", 0.0),
                    "mcc_mean": dec_data.get("mcc_mean", 0.0),
                    "mcc_std": dec_data.get("mcc_std", 0.0),
                    "auprc_mean": dec_data.get("auprc_mean", 0.0),
                    "pair_ece_mean": dec_data.get("pair_ece_mean", 0.0),
                    "empty_rate_mean": dec_data.get("empty_rate_mean", 0.0),
                    "illegal_rate_mean": dec_data.get("illegal_rate_mean", 0.0),
                }
    return result


def build_ablation_results(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """Build decoder and distance-bin ablation analysis."""
    models = aggregate.get("models", {})
    result = {
        "description": "Decoder ablation and distance-bin analysis",
        "decoder_ablation": {},
        "distance_bin_ablation": {},
    }

    # Decoder ablation: for each model, compare F1 across decoders on val
    for model_name, model_data in models.items():
        val_split = model_data.get("splits", {}).get("val", {})
        result["decoder_ablation"][model_name] = {}
        for dec_name in ["threshold", "nussinov_dp", "mea", "greedy_pseudoknot"]:
            dec_data = val_split.get(dec_name, {})
            result["decoder_ablation"][model_name][dec_name] = {
                "f1_mean": dec_data.get("f1_mean", 0.0),
                "f1_std": dec_data.get("f1_std", 0.0),
                "empty_rate_mean": dec_data.get("empty_rate_mean", 0.0),
                "illegal_rate_mean": dec_data.get("illegal_rate_mean", 0.0),
            }

        # Distance bin ablation (MEA decoder on val)
        mea_data = val_split.get("mea", {})
        result["distance_bin_ablation"][model_name] = {
            "short": mea_data.get("distance_bins", {}).get("short", {}),
            "medium": mea_data.get("distance_bins", {}).get("medium", {}),
            "long": mea_data.get("distance_bins", {}).get("long", {}),
        }

    return result


def build_profiling_results(runs_dir: Path) -> Dict[str, Any]:
    """Extract runtime/memory profiling from per-run evaluation results."""
    result = {
        "description": "Runtime and memory profiling (per-run, per-split, per-decoder)",
        "runs": {},
        "summary": {},
    }

    all_runtimes: Dict[str, List[float]] = {}

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        eval_file = run_dir / "evaluation_results.json"
        if not eval_file.exists():
            continue
        with open(eval_file) as f:
            eval_data = json.load(f)

        run_name = run_dir.name
        run_info = {
            "n_params": eval_data.get("n_params", 0),
            "model_type": eval_data.get("model_type", "unknown"),
            "splits": {},
        }

        splits = eval_data.get("splits", {})
        for split_name, split_data in splits.items():
            split_info = {}
            for dec_name, dec_data in split_data.items():
                # Skip non-dict entries like _eval_time_sec
                if not isinstance(dec_data, dict):
                    continue
                inference_time = dec_data.get("inference_time_sec", 0.0)
                n_samples = dec_data.get("total_samples", 1)
                per_sample_time = inference_time / max(n_samples, 1)
                split_info[dec_name] = {
                    "total_inference_time_sec": inference_time,
                    "n_samples": n_samples,
                    "per_sample_time_ms": per_sample_time * 1000,
                }
                key = f"{run_name.split('_seed')[0]}/{dec_name}"
                if key not in all_runtimes:
                    all_runtimes[key] = []
                all_runtimes[key].append(per_sample_time)
            run_info["splits"][split_name] = split_info

        result["runs"][run_name] = run_info

    # Summary: mean per-sample time per model/decoder
    for key, times in all_runtimes.items():
        result["summary"][key] = {
            "mean_per_sample_time_ms": sum(times) / len(times) * 1000,
            "n_runs": len(times),
        }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/c1_2/runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/c1_2/aggregate_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/c1_2"))
    args = parser.parse_args()

    with open(args.aggregate) as f:
        aggregate = json.load(f)

    # 1. matched_capacity_results.json
    mc = build_matched_capacity_results(aggregate)
    mc_path = args.output_dir / "matched_capacity_results.json"
    with open(mc_path, "w") as f:
        json.dump(mc, f, indent=2)
    print(f"[ok] {mc_path}")

    # 2. ablation_results.json
    ab = build_ablation_results(aggregate)
    ab_path = args.output_dir / "ablation_results.json"
    with open(ab_path, "w") as f:
        json.dump(ab, f, indent=2)
    print(f"[ok] {ab_path}")

    # 3. profiling.json
    prof = build_profiling_results(args.runs_dir)
    prof_path = args.output_dir / "profiling.json"
    with open(prof_path, "w") as f:
        json.dump(prof, f, indent=2)
    print(f"[ok] {prof_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
