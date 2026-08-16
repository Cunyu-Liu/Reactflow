#!/usr/bin/env python3
"""Decide optimal multi-seed dataset size based on seed 0's F1.

After seed 0 evaluation completes, this script reads the F1 and recommends:
- F1 > 0.72: use 50K dataset (fastest, ~4.6h/seed, 10 seeds = ~46h)
- F1 0.69-0.72: use 100K dataset (medium, ~9.3h/seed, 10 seeds = ~93h)
- F1 0.68-0.69: use full dataset (safest, ~16h/seed, 10 seeds = ~160h)
- F1 < 0.68: WARNING - model doesn't beat ViennaRNA, investigate

Usage: python3 /tmp/smart_multiseed_decision.py
"""
import json
import sys
from pathlib import Path

VIENNARNA_F1 = 0.6819
ARTIFACTS = Path("/home/cunyuliu/reactflow_c1_3_stage_20260722/artifacts/c1_3")


def get_seed0_f1() -> float:
    """Get seed 0's best F1 from model_grid_results.json."""
    grid_path = ARTIFACTS / "model_grid_results.json"
    if not grid_path.exists():
        return 0.0
    with open(grid_path) as f:
        data = json.load(f)
    best_f1 = 0.0
    for cfg in data.get("configs", []):
        config_name = cfg.get("config", "")
        if "seed0" not in config_name and "fsdp_seed0" not in config_name:
            continue
        for tier_name, tier_data in cfg.get("tiers", {}).items():
            if isinstance(tier_data, dict):
                f1 = tier_data.get("mean_f1", 0)
                if f1 > best_f1:
                    best_f1 = f1
    return best_f1


def main() -> None:
    f1 = get_seed0_f1()
    print(f"Seed 0 best F1: {f1:.4f}")
    print(f"ViennaRNA F1:   {VIENNARNA_F1:.4f}")
    print(f"Margin:         {f1 - VIENNARNA_F1:+.4f}")
    print()

    if f1 < 0.01:
        print("STATUS: Seed 0 not evaluated yet.")
        print("Run this script after seed 0 evaluation completes.")
        sys.exit(1)

    if f1 < VIENNARNA_F1:
        print("WARNING: Model does NOT beat ViennaRNA!")
        print("Consider: more epochs, unfreezing embeddings, or architectural changes.")
        print("Do NOT start multi-seed training until model beats ViennaRNA.")
        sys.exit(2)

    if f1 > 0.72:
        dataset = 50000
        config = "multiseed_50k"
        eta_h = 46
        print(f"RECOMMENDATION: Use 50K dataset (fastest)")
        print(f"  Config: pairformer_ribonanza_frozen_small_pair_{config}.yaml")
        print(f"  ETA: ~{eta_h}h for 10 seeds")
        print(f"  Command: bash scripts/launch_multiseed_after_seed0.sh {dataset}")
    elif f1 > 0.69:
        dataset = 100000
        config = "multiseed_100k"
        eta_h = 93
        print(f"RECOMMENDATION: Use 100K dataset (medium)")
        print(f"  Config: pairformer_ribonanza_frozen_small_pair_{config}.yaml")
        print(f"  ETA: ~{eta_h}h for 10 seeds")
        print(f"  Command: bash scripts/launch_multiseed_after_seed0.sh {dataset}")
    else:
        dataset = 0
        config = "multiseed_full"
        eta_h = 160
        print(f"RECOMMENDATION: Use full dataset (safest)")
        print(f"  Config: pairformer_ribonanza_frozen_small_pair_{config}.yaml")
        print(f"  ETA: ~{eta_h}h for 10 seeds")
        print(f"  Command: bash scripts/launch_multiseed_after_seed0.sh {dataset}")

    print()
    print(f"To switch pipeline:")
    print(f"  1. kill <pipeline_pid>  # kill current pipeline")
    print(f"  2. bash scripts/launch_multiseed_after_seed0.sh {dataset}")


if __name__ == "__main__":
    main()
