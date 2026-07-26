#!/usr/bin/env python3
"""Generate multiseed_results.json from individual seed evaluation results.

Reads model_grid_results.json and collects results from all seeds
into a unified multiseed_results.json file.

Fixed version: parses seed number from config names like
'pairformer_ribonanza_frozen_small_pair_fsdp_seed{N}'.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev


def parse_seed_from_config(config_name: str) -> int:
    """Parse seed number from config name.

    Looks for patterns like 'seed0', 'seed1', '_seed_2', etc.
    Returns 0 if no seed pattern found.
    """
    match = re.search(r"seed[_-]?(\d+)", config_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def get_base_config(config_name: str) -> str:
    """Get base config name by removing seed suffix.

    e.g. 'pairformer_ribonanza_frozen_small_pair_fsdp_seed3' ->
         'pairformer_ribonanza_frozen_small_pair_fsdp'
    """
    return re.sub(r"[_-]?seed[_-]?\d+", "", config_name, flags=re.IGNORECASE)


def collect_seed_results(artifacts_dir: Path, num_seeds: int) -> list[dict]:
    """Collect results from all seeds."""
    seeds = []
    grid_path = artifacts_dir / "model_grid_results.json"

    if grid_path.exists():
        with open(grid_path) as f:
            grid_data = json.load(f)

        for cfg in grid_data.get("configs", []):
            config_name = cfg.get("config", "")
            seed = parse_seed_from_config(config_name)
            base_config = get_base_config(config_name)
            for tier_name, tier_data in cfg.get("tiers", {}).items():
                if isinstance(tier_data, dict):
                    seeds.append({
                        "seed": seed,
                        "config": base_config,
                        "original_config": config_name,
                        "tier": tier_name,
                        "mean_f1": tier_data.get("mean_f1", 0),
                        "mean_mcc": tier_data.get("mean_mcc", 0),
                        "long_f1": tier_data.get("long_f1", 0),
                    })

    # Also try individual seed result files (legacy fallback)
    for seed in range(num_seeds):
        seed_grid_path = artifacts_dir / f"model_grid_results_seed{seed}.json"
        if seed_grid_path.exists():
            with open(seed_grid_path) as f:
                seed_data = json.load(f)
            for cfg in seed_data.get("configs", []):
                config_name = cfg.get("config", "")
                base_config = get_base_config(config_name)
                for tier_name, tier_data in cfg.get("tiers", {}).items():
                    if isinstance(tier_data, dict):
                        seeds.append({
                            "seed": seed,
                            "config": base_config,
                            "original_config": config_name,
                            "tier": tier_name,
                            "mean_f1": tier_data.get("mean_f1", 0),
                            "mean_mcc": tier_data.get("mean_mcc", 0),
                            "long_f1": tier_data.get("long_f1", 0),
                        })

    return seeds


def compute_stats(values: list[float]) -> dict:
    """Compute mean, std, min, max, and confidence interval."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "ci_low": 0, "ci_high": 0, "n": 0}
    n = len(values)
    m = mean(values)
    s = stdev(values) if n > 1 else 0
    if n > 1:
        ci_margin = 1.96 * s / (n ** 0.5)
    else:
        ci_margin = 0
    return {
        "mean": round(m, 6),
        "std": round(s, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "ci_low": round(m - ci_margin, 6),
        "ci_high": round(m + ci_margin, 6),
        "n": n,
    }


def generate_multiseed(artifacts_dir: Path, num_seeds: int) -> dict:
    """Generate multiseed_results.json."""
    all_results = collect_seed_results(artifacts_dir, num_seeds)

    # Group by config + tier
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in all_results:
        key = (r["config"], r["tier"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    configs = []
    for (config_name, tier_name), results in sorted(grouped.items()):
        f1_values = [r["mean_f1"] for r in results]
        mcc_values = [r["mean_mcc"] for r in results]
        long_f1_values = [r["long_f1"] for r in results]

        config_entry = {
            "config": config_name,
            "tier": tier_name,
            "num_seeds": len(results),
            "seeds": sorted([r["seed"] for r in results]),
            "f1_stats": compute_stats(f1_values),
            "mcc_stats": compute_stats(mcc_values),
            "long_f1_stats": compute_stats(long_f1_values),
            "individual_results": results,
        }
        configs.append(config_entry)

    # Determine unique seeds
    all_seeds = sorted(set(r["seed"] for r in all_results))

    return {
        "schema_version": 1,
        "seeds": all_seeds,
        "num_seeds": len(all_seeds),
        "configs": configs,
        "note": f"Multi-seed results from {len(all_seeds)} seeds" if all_seeds else "No seed results found",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multiseed_results.json")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/c1_3"))
    parser.add_argument("--num-seeds", type=int, default=10)
    args = parser.parse_args()

    multiseed = generate_multiseed(args.artifacts_dir, args.num_seeds)

    output_path = args.artifacts_dir / "multiseed_results.json"
    with open(output_path, "w") as f:
        json.dump(multiseed, f, indent=2)

    print(f"Written {output_path}")
    print(f"Seeds: {multiseed['num_seeds']}")
    for cfg in multiseed["configs"]:
        print(f"  {cfg['config']} {cfg['tier']}: F1={cfg['f1_stats']['mean']:.4f} +/- {cfg['f1_stats']['std']:.4f} (n={cfg['num_seeds']})")


if __name__ == "__main__":
    main()
