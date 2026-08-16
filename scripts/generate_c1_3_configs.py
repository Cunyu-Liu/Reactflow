#!/usr/bin/env python3
"""Generate C1-3 model grid configuration YAML files.

Spec reference: ReactFlow分阶段执行提示词.md lines 501-510 (full-scale model grid).

Generates ``configs/models/c1_3/pairformer_*.yaml`` configs for the 3-seed
pilot grid and the 10-seed expansion grid.

Reduced pilot grid (3 seeds each):
  - 2 backbones: from_scratch, ribonanza_frozen
  - 2 sizes: small (12 blocks, 384/96), medium (24 blocks, 512/128)
  - 2 fusions: single_only, pair_feature
  = 8 configs

Expansion grid (10 seeds for top 2-3):
  - ribonanza_frozen + pair_feature + large (36 blocks, 768/192)

Usage::

    python scripts/generate_c1_3_configs.py --output-dir configs/models/c1_3
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

FROZEN_FEATURES_PATH = (
    "/home/cunyuliu/reactflow/artifacts/full_runs/"
    "full_ablation_20260709_003012/frozen/ribonanzanet2_sharded_full"
)

SPLITS_PATH = (
    "/home/cunyuliu/reactflow/artifacts/full_runs/"
    "full_ablation_20260709_003012/splits/rfam_current_mmseqs_seed0"
)


def _base_config() -> Dict[str, Any]:
    """Return the shared base config dict."""
    return {
        "schema_version": "2.0",
        "loss": {
            "bce_weight": 1.0,
            "focal_weight": 0.5,
            "soft_f1_weight": 0.5,
            "dice_weight": 0.0,
            "pair_count_weight": 0.1,
            "symmetry_weight": 0.01,
            "calibration_weight": 0.1,
            "unpaired_weight": 0.5,
            "long_range_weight": 0.1,
            "long_range_threshold": 24,
            "long_range_upweight_factor": 2.0,
            "focal_gamma": 2.0,
            "bce_pos_weight": None,
            "bce_pos_weight_auto": True,
            "ece_num_bins": 15,
        },
        "training": {
            "device": "cuda:6",
            "world_size": 1,
            "use_ddp": False,
            "use_fsdp": False,
            "use_bf16": True,
            "use_grad_checkpoint": True,
            "use_flash_attention": False,
            "max_grad_norm": 1.0,
            "oom_retry_max": 3,
            "nan_inf_guard": True,
            "log_interval": 100,
            "optimizer": "adamw",
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
            "warmup_steps": 1000,
            "cosine_decay_steps": 0,
            "resume_from": None,
            "save_every": 1,
            "eval_every": 1,
            "early_stop_patience": 5,
            "early_stop_min_delta": 0.001,
            "batch_size": 16,
            "eval_batch_size": 8,
            "num_workers": 4,
            "pin_memory": True,
        },
        "curriculum": {
            "replay_ratio": 0.2,
            "balance_keys": [
                "source", "family", "clan",
                "length_bin", "pair_distance_bin", "structure_complexity",
            ],
        },
        "data": {
            "splits_path": SPLITS_PATH,
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "novel_split": "novel",
            "max_length": 512,
            "min_length": 4,
        },
        "decoder": {
            "default_mode": "mea",
            "threshold": 0.5,
            "min_score": 0.0,
            "min_loop": 3,
            "allow_wobble": True,
            "seq_pad_index": 5,
        },
    }


def _backbone_entry(name: str, mode: str, *, frozen_dim: int = 0,
                    lora_rank: int = 0, frozen_path: str = "") -> Dict[str, Any]:
    """Build a backbone config section."""
    entry: Dict[str, Any] = {"name": name, "mode": mode, "lora_rank": lora_rank}
    if frozen_dim:
        entry["frozen_feature_dim"] = frozen_dim
    if frozen_path:
        entry["frozen_features_path"] = frozen_path
    return entry


def _fusion_entry(fusion_type: str, single_dim: int, pair_dim: int,
                  num_backbones: int = 1) -> Dict[str, Any]:
    """Build a fusion config section."""
    return {
        "fusion_type": fusion_type,
        "single_dim": single_dim,
        "pair_dim": pair_dim,
        "num_backbones": num_backbones,
        "dropout": 0.0,
    }


def _model_entry(single_dim: int, pair_dim: int, num_blocks: int,
                 frozen_feature_dim: int, *, max_len: int = 512) -> Dict[str, Any]:
    """Build a StaticPairFormer model config section."""
    return {
        "model_type": "static_pairformer",
        "single_dim": single_dim,
        "pair_dim": pair_dim,
        "max_len": max_len,
        "num_blocks": num_blocks,
        "num_heads_pair": max(4, single_dim // 64),
        "num_heads_single": max(4, single_dim // 64),
        "triangle_hidden_dim": None,
        "ffn_expansion": 4,
        "outer_product_mean_dim": 16,
        "use_onehot": False,
        "learnable_pos": True,
        "frozen_feature_dim": frozen_feature_dim,
        "dropout": 0.0,
        "block_dropout": 0.0,
        "num_pair_types": 0,
        "use_calibration": True,
        "init_temperature": 1.0,
        "share_pair_init_projection": True,
        "num_region_labels": 0,
    }


def _curriculum_stages() -> Dict[str, Any]:
    """Build the curriculum stage_epochs section."""
    return {
        "stages": [
            "short_nested_ncRNA", "mixed_rfam", "pri_miRNA",
            "pdb", "viral", "human_mRNA", "lncRNA",
        ],
        "stage_epochs": {
            "short_nested_ncRNA": 3,
            "mixed_rfam": 5,
            "pri_miRNA": 2,
            "pdb": 3,
            "viral": 2,
            "human_mRNA": 2,
            "lncRNA": 2,
        },
    }


# ---------------------------------------------------------------------------
# Grid points
# ---------------------------------------------------------------------------

def _build_grid() -> List[Dict[str, Any]]:
    """Build the full pilot + expansion grid.

    Returns a list of config dicts, each ready to be dumped as YAML.
    """
    grid: List[Dict[str, Any]] = []

    # Size presets: (label, num_blocks, single_dim, pair_dim)
    sizes = [
        ("small", 12, 384, 96),
        ("medium", 24, 512, 128),
        ("large", 36, 768, 192),
    ]

    # Backbone presets
    backbones = [
        ("from_scratch", "from_scratch", "full_fine_tune", 0, 0, ""),
        ("ribonanza_frozen", "ribonanzanet2", "frozen", 384, 0, FROZEN_FEATURES_PATH),
    ]

    # Fusion presets
    fusions = ["single_only", "pair_feature"]

    # Pilot grid: 2 backbones x 2 sizes (small, medium) x 2 fusions = 8
    for bb_label, bb_name, bb_mode, bb_fdim, bb_lora, bb_path in backbones:
        for size_label, n_blocks, s_dim, p_dim in sizes[:2]:  # small, medium only
            for fusion in fusions:
                cfg = _base_config()
                config_name = f"pairformer_{bb_label}_{size_label}_{fusion.split('_')[0]}"
                cfg["config_name"] = config_name

                cfg["backbone"] = _backbone_entry(
                    bb_name, bb_mode, frozen_dim=bb_fdim,
                    lora_rank=bb_lora, frozen_path=bb_path,
                )

                # When using frozen features, single_dim must exceed
                # frozen_feature_dim to leave room for nucleotide+positional.
                # We add frozen_feature_dim on top of the base single_dim.
                model_fdim = bb_fdim if bb_mode == "frozen" else 0
                total_s_dim = s_dim + model_fdim
                cfg["model"] = _model_entry(
                    total_s_dim, p_dim, n_blocks, model_fdim,
                )
                cfg["fusion"] = _fusion_entry(fusion, total_s_dim, p_dim)

                cur = _curriculum_stages()
                cfg["curriculum"].update(cur)
                total_epochs = sum(cur["stage_epochs"].values())
                cfg["training"]["epochs"] = total_epochs
                cfg["training"]["checkpoint_dir"] = (
                    f"artifacts/c1_3/runs/{config_name}_seed{{seed}}"
                )
                cfg["seed"] = 0
                grid.append(cfg)

    # Expansion grid: ribonanza_frozen + pair_feature + large
    bb_label, bb_name, bb_mode, bb_fdim, bb_lora, bb_path = backbones[1]
    size_label, n_blocks, s_dim, p_dim = sizes[2]  # large
    fusion = "pair_feature"
    cfg = _base_config()
    config_name = "pairformer_ribonanza_frozen_large_pair"
    cfg["config_name"] = config_name
    cfg["backbone"] = _backbone_entry(
        bb_name, bb_mode, frozen_dim=bb_fdim,
        lora_rank=bb_lora, frozen_path=bb_path,
    )
    total_s_dim = s_dim + bb_fdim  # ensure room for nucleotide+positional
    cfg["model"] = _model_entry(total_s_dim, p_dim, n_blocks, bb_fdim)
    cfg["fusion"] = _fusion_entry(fusion, total_s_dim, p_dim)
    cur = _curriculum_stages()
    cfg["curriculum"].update(cur)
    total_epochs = sum(cur["stage_epochs"].values())
    cfg["training"]["epochs"] = total_epochs
    cfg["training"]["checkpoint_dir"] = (
        f"artifacts/c1_3/runs/{config_name}_seed{{seed}}"
    )
    cfg["training"]["batch_size"] = 8  # smaller for large model
    cfg["training"]["eval_batch_size"] = 4
    cfg["seed"] = 0
    grid.append(cfg)

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("configs/models/c1_3"),
        help="Output directory for YAML configs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = _build_grid()
    for cfg in grid:
        name = cfg["config_name"]
        path = args.output_dir / f"{name}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)
        print(f"  wrote {path}")

    print(f"\nGenerated {len(grid)} configs in {args.output_dir}/")


if __name__ == "__main__":
    main()
