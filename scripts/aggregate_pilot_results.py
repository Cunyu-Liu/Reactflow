#!/usr/bin/env python3
"""Aggregate C1-3 pilot results across configs and seeds.

Scans pilot output directories for evaluation_results.json files,
computes mean/std across seeds, and compares with baseline results.

Usage::

    python scripts/aggregate_pilot_results.py \
        --results-root /tmp/c1_3_results \
        --baseline-json artifacts/c1_3/baseline_efold_results.json \
        --baseline-json artifacts/c1_3/baseline_viennarna_results.json \
        --output artifacts/c1_3/model_grid_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# Metrics to extract from evaluation_results.json
METRIC_KEYS = [
    "f1", "precision", "recall", "mcc",
    "f1_shifted", "empty_rate", "legal_rate",
    "f1_short", "f1_medium", "f1_long",
]
SPLITS = ["val", "test", "novel"]
DECODER = "mea"  # Primary decoder for comparison


def load_eval_results(path: Path) -> Dict[str, Any]:
    """Load evaluation_results.json from a run directory."""
    eval_path = path / "evaluation_results.json"
    if not eval_path.exists():
        return {}
    with open(eval_path, encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(eval_results: Dict[str, Any], decoder: str = DECODER) -> Dict[str, Dict[str, float]]:
    """Extract metrics for each split from eval results.

    Returns {split: {metric: value}}.
    """
    out: Dict[str, Dict[str, float]] = {}
    for split in SPLITS:
        if split not in eval_results:
            continue
        split_data = eval_results[split]
        if decoder not in split_data:
            continue
        d = split_data[decoder]
        metrics: Dict[str, float] = {}
        for k in METRIC_KEYS:
            if k in d:
                metrics[k] = float(d[k])
        # Also extract runtime/memory
        if "_runtime_sec" in split_data:
            metrics["runtime_sec"] = float(split_data["_runtime_sec"])
        if "_peak_memory_bytes" in split_data:
            metrics["peak_memory_gb"] = float(split_data["_peak_memory_bytes"]) / 1e9
        out[split] = metrics
    return out


def aggregate_seeds(seed_results: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """Aggregate metrics across seeds.

    Returns {split: {metric: {mean, std, min, max, values}}}.
    """
    if not seed_results:
        return {}
    all_splits = set()
    for sr in seed_results:
        all_splits.update(sr.keys())

    aggregated: Dict[str, Any] = {}
    for split in all_splits:
        split_metrics: Dict[str, Any] = {}
        # Collect all metric keys across seeds
        all_keys = set()
        for sr in seed_results:
            if split in sr:
                all_keys.update(sr[split].keys())
        for key in all_keys:
            values = []
            for sr in seed_results:
                if split in sr and key in sr[split]:
                    values.append(sr[split][key])
            if not values:
                continue
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            split_metrics[key] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "values": [round(v, 6) for v in values],
            }
        aggregated[split] = split_metrics
    return aggregated


def load_baselines(baseline_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Load baseline result JSONs."""
    baselines = []
    for p in baseline_paths:
        if not p.exists():
            print(f"[WARN] Baseline file not found: {p}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for row in data.get("rows", []):
            baselines.append(row)
    return baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate C1-3 pilot results")
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Root directory containing per-run output dirs")
    parser.add_argument("--baseline-json", type=Path, action="append", default=[],
                        help="Baseline results JSON (can repeat)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output model_grid_results.json")
    parser.add_argument("--decoder", type=str, default=DECODER,
                        help="Decoder to use for comparison")
    args = parser.parse_args()

    # Find all run directories (pattern: <config_name>_seed<N>)
    config_seeds: Dict[str, List[Path]] = {}
    for d in sorted(args.results_root.iterdir()):
        if not d.is_dir():
            continue
        # Parse config name and seed from directory name
        name = d.name
        if "_seed" not in name:
            continue
        parts = name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        config_name, seed_str = parts
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        config_seeds.setdefault(config_name, []).append(d)

    print(f"[INFO] Found {len(config_seeds)} configs:", file=sys.stderr)
    for cfg, dirs in sorted(config_seeds.items()):
        print(f"  {cfg}: {len(dirs)} seeds", file=sys.stderr)

    # Aggregate each config
    grid_results: List[Dict[str, Any]] = []
    for config_name, dirs in sorted(config_seeds.items()):
        seed_evals = []
        for d in dirs:
            eval_results = load_eval_results(d)
            if eval_results:
                metrics = extract_metrics(eval_results, args.decoder)
                seed_evals.append(metrics)
                print(f"  {config_name} seed={d.name}: {metrics.get('test', {}).get('f1', 'N/A')}", file=sys.stderr)
        if not seed_evals:
            print(f"[WARN] No eval results for {config_name}", file=sys.stderr)
            continue
        aggregated = aggregate_seeds(seed_evals)
        grid_results.append({
            "config": config_name,
            "decoder": args.decoder,
            "num_seeds": len(seed_evals),
            "metrics": aggregated,
        })

    # Load baselines
    baselines = load_baselines(args.baseline_json)
    print(f"[INFO] Loaded {len(baselines)} baseline rows", file=sys.stderr)

    # Build output
    output = {
        "schema_version": 1,
        "generated_at": str(Path(__file__).stat().st_mtime),
        "decoder": args.decoder,
        "configs": grid_results,
        "baselines": baselines,
        "baseline_summary": {
            row.get("model", "unknown"): {
                "tier": row.get("tier"),
                "mean_f1": row.get("mean_f1"),
                "mean_mcc": row.get("mean_mcc"),
                "long_f1": row.get("long_f1"),
            }
            for row in baselines
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Written to {args.output}", file=sys.stderr)

    # Print summary table
    print("\n=== Pilot Results Summary (MEA decoder) ===", file=sys.stderr)
    print(f"{'Config':<45} {'test_f1':>10} {'novel_f1':>10} {'val_f1':>10}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    for cfg in grid_results:
        name = cfg["config"]
        test_f1 = cfg["metrics"].get("test", {}).get("f1", {}).get("mean", "N/A")
        novel_f1 = cfg["metrics"].get("novel", {}).get("f1", {}).get("mean", "N/A")
        val_f1 = cfg["metrics"].get("val", {}).get("f1", {}).get("mean", "N/A")
        print(f"{name:<45} {test_f1:>10} {novel_f1:>10} {val_f1:>10}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    for row in baselines:
        model = row.get("model", "unknown")
        tier = row.get("tier", "")
        f1 = row.get("mean_f1", "N/A")
        print(f"{model + ' (' + tier + ')':<45} {f1:>10}", file=sys.stderr)


if __name__ == "__main__":
    main()
