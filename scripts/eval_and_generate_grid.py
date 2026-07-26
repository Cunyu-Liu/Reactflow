#!/usr/bin/env python3
"""Evaluate trained checkpoint and generate model_grid_results.json for gate audit.

This script:
1. Runs eval_checkpoint.py on the FSDP checkpoint
2. Converts evaluation results to model_grid_results.json format
3. Optionally generates significance_report.json from multi-seed results

Usage::

    PYTHONPATH=src python scripts/eval_and_generate_grid.py \
        --config configs/models/c1_3/pairformer_ribonanza_frozen_small_pair_ddp.yaml \
        --checkpoint artifacts/c1_3/runs/.../best.pt \
        --device cuda:0 \
        --output-dir artifacts/c1_3 \
        --config-name pairformer_ribonanza_frozen_small_pair_fsdp_seed0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_eval_checkpoint(
    config: Path,
    checkpoint: Path,
    device: str,
    eval_decoders: str,
    output_dir: Path,
    batch_size: int = 4,
) -> Path:
    """Run eval_checkpoint.py and return path to results."""
    eval_output = output_dir / "eval_results"
    cmd = [
        sys.executable, "scripts/eval_checkpoint.py",
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--device", device,
        "--eval-decoders", eval_decoders,
        "--output-dir", str(eval_output),
        "--batch-size", str(batch_size),
    ]
    print(f"[INFO] Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] eval_checkpoint.py failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stderr, file=sys.stderr)

    results_path = eval_output / "evaluation_results.json"
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)
    return results_path


def convert_to_grid_format(
    eval_results_path: Path,
    config_name: str,
    output_path: Path,
    decoder: str = "mea",
) -> None:
    """Convert eval_checkpoint.py output to model_grid_results.json format.

    The gate audit expects:
    {
        "configs": [
            {
                "config": "...",
                "tiers": {
                    "in_clan": {"mean_f1": ..., "mean_mcc": ..., "long_f1": ...},
                    "novel_clan": {"mean_f1": ..., "mean_mcc": ..., "long_f1": ...}
                }
            }
        ]
    }
    """
    with open(eval_results_path) as f:
        eval_results = json.load(f)

    # Map eval split names to tier names
    # test -> in_clan, novel -> novel_clan
    tier_map = {"test": "in_clan", "novel": "novel_clan", "val": "val"}

    config_entry: Dict[str, Any] = {
        "config": config_name,
        "decoder": decoder,
        "num_seeds": 1,
        "tiers": {},
    }

    for split_name, split_data in eval_results.items():
        tier_name = tier_map.get(split_name, split_name)
        if decoder not in split_data:
            print(f"[WARN] Decoder {decoder} not found in {split_name}, available: {list(split_data.keys())}", file=sys.stderr)
            continue
        metrics = split_data[decoder]

        # Extract metrics
        tier_entry = {
            "mean_f1": metrics.get("f1", 0.0),
            "mean_mcc": metrics.get("mcc", 0.0),
            "f1_shifted": metrics.get("f1_shifted", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "long_f1": metrics.get("f1_long", metrics.get("long_f1", 0.0)),
            "legal_rate": metrics.get("legal_rate", 0.0),
            "runtime_sec": metrics.get("_runtime_sec", 0.0),
        }
        config_entry["tiers"][tier_name] = tier_entry

    # Load existing grid results or create new
    if output_path.exists():
        with open(output_path) as f:
            grid_data = json.load(f)
        # Replace existing config with same name
        grid_data["configs"] = [
            c for c in grid_data.get("configs", []) if c.get("config") != config_name
        ]
        grid_data["configs"].append(config_entry)
    else:
        grid_data = {
            "schema_version": 1,
            "decoder": decoder,
            "configs": [config_entry],
        }

    with open(output_path, "w") as f:
        json.dump(grid_data, f, indent=2)
    print(f"[INFO] Written model_grid_results.json to {output_path}", file=sys.stderr)
    print(f"[INFO] Config: {config_name}", file=sys.stderr)
    for tier_name, tier_data in config_entry["tiers"].items():
        print(f"  {tier_name}: F1={tier_data['mean_f1']:.4f} MCC={tier_data['mean_mcc']:.4f}", file=sys.stderr)


def generate_significance_report(
    grid_path: Path,
    baseline_path: Path,
    output_path: Path,
    decoder: str = "mea",
) -> None:
    """Generate significance report comparing model vs baselines.

    For now, this is a placeholder that generates the required file structure.
    A proper significance test requires multi-seed results.
    """
    with open(grid_path) as f:
        grid_data = json.load(f)
    with open(baseline_path) as f:
        baseline_data = json.load(f)

    # Find best model config
    best_model_f1 = 0.0
    best_config = ""
    for cfg in grid_data.get("configs", []):
        for tier_name, tier_data in cfg.get("tiers", {}).items():
            if tier_name in ("in_clan", "novel_clan"):
                f1 = tier_data.get("mean_f1", 0.0)
                if f1 > best_model_f1:
                    best_model_f1 = f1
                    best_config = cfg.get("config", "?")

    # Find best baseline
    best_baseline_f1 = 0.0
    best_baseline_name = ""
    for b in baseline_data.get("baselines", []):
        for row in b.get("rows", []):
            if row.get("tier") in ("in_clan", "novel_clan"):
                f1 = row.get("mean_f1", 0.0)
                if f1 > best_baseline_f1:
                    best_baseline_f1 = f1
                    best_baseline_name = b.get("model", "?")

    # Generate significance report
    report = {
        "schema_version": 1,
        "decoder": decoder,
        "tests": [
            {
                "name": "model_vs_best_baseline",
                "model": best_config,
                "baseline": best_baseline_name,
                "model_f1": best_model_f1,
                "baseline_f1": best_baseline_f1,
                "delta": best_model_f1 - best_baseline_f1,
                "p_value": None,  # Requires multi-seed for proper test
                "significant": False,  # Cannot determine without multi-seed
                "note": "Single-seed result; p-value requires multi-seed (>=3 seeds)",
            },
        ],
        "multiseed_available": False,
        "note": "Significance test requires multi-seed results. Run multi-seed training first.",
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Written significance_report.json to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint and generate grid results")
    parser.add_argument("--config", type=Path, required=True, help="Model config YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-decoders", type=str, default="threshold,mea")
    parser.add_argument("--decoder", type=str, default="mea", help="Decoder for grid results")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/c1_3"))
    parser.add_argument("--config-name", type=str, default="model_seed0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, just convert existing results")
    args = parser.parse_args()

    if args.skip_eval:
        eval_results_path = args.output_dir / "eval_results" / "evaluation_results.json"
        if not eval_results_path.exists():
            print(f"[ERROR] {eval_results_path} not found", file=sys.stderr)
            sys.exit(1)
    else:
        eval_results_path = run_eval_checkpoint(
            args.config, args.checkpoint, args.device,
            args.eval_decoders, args.output_dir, args.batch_size,
        )

    # Convert to grid format
    grid_path = args.output_dir / "model_grid_results.json"
    convert_to_grid_format(eval_results_path, args.config_name, grid_path, args.decoder)

    # Generate significance report
    baseline_path = args.output_dir / "baseline_same_split_results.json"
    sig_path = args.output_dir / "significance_report.json"
    if baseline_path.exists():
        generate_significance_report(grid_path, baseline_path, sig_path, args.decoder)
    else:
        print(f"[WARN] Baseline results not found at {baseline_path}", file=sys.stderr)

    print("\n[INFO] Done! Gate audit can now be run:", file=sys.stderr)
    print(f"  python scripts/audit_c1_3_gate.py --artifacts-dir {args.output_dir} --docs-dir docs", file=sys.stderr)


if __name__ == "__main__":
    main()
