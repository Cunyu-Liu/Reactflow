#!/usr/bin/env python3
"""EPRO-Lite training script (v3 EPRO §4.9, Phase M0 T-M0.4~T-M0.6).

Modes:
  * ``overfit``: single-pair GPU overfit (T-M0.4).
  * ``small_batch``: small-batch GPU overfit (T-M0.5).
  * ``pilot``: fixed seed/budget train-validation pilot (T-M0.6).

Data flow:
  1. Load M0 pair manifest (seq_positions, parent linkage, edit_arr_idx).
  2. Load parent thermo .npz (per-position features + sparse contacts).
  3. Load registry for delta_true + endpoint_mask (via evaluate.load_split_pairs).
  4. Build per-pair feature tensors aligned to the delta array.
  5. Train with Student-t NLL + measurement variance.

GPU is required (CUDA_VISIBLE_DEVICES set externally). CPU silent fallback
triggers an immediate stop with evidence preserved.

Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python scripts/reactflow_delta/train.py \
        --config configs/reactflow_delta/epro_lite.yaml \
        --mode pilot \
        --output-dir /mnt/cunyuliu/reactflow_delta_artifacts_20260729/m0/pilot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.evaluate import load_split_pairs, make_rdat_loader, compute_pair_metrics, aggregate_metrics  # noqa: E402
from reactflow.delta.model import EPROModel, EPROConfig, make_epro0, make_epro_lite  # noqa: E402
from reactflow.delta.losses import StudentTNLL, HeteroscedasticNLL, WeightedMAELoss, WeightedMSELoss, compute_skill  # noqa: E402

TRAIN_SCHEMA_VERSION = "reactflow-delta-m0-train-v1"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class PairData:
    """All tensors needed to run one pair through the model."""

    pair_id: str
    parent: str
    split: str
    features: torch.Tensor  # (n, 5)
    edges: torch.Tensor  # (2, n_edges)
    edge_features: torch.Tensor  # (n_edges, 3)
    edit_pos: int  # 0-indexed
    delta_true: torch.Tensor  # (n,)
    endpoint_mask: torch.Tensor  # (n,) bool
    pair_quality_weight: float
    measurement_variance: float | None
    n: int


def load_parent_thermo(npz_path: str) -> dict[str, np.ndarray]:
    """Load a parent thermo .npz file."""

    return dict(np.load(npz_path))


def build_pair_data(pair_meta: dict, parent_thermo: dict[str, np.ndarray],
                    pair_record) -> PairData:
    """Build per-pair tensors from manifest metadata + parent thermo + PairRecord.

    ``pair_record`` is a PairRecord from evaluate.load_split_pairs (provides
    delta_true, endpoint_mask, edit_arr_idx, pair_quality_weight).
    """

    n = pair_meta["aligned_length"]
    seq_positions = pair_meta["seq_positions"]  # list of 1-indexed seq pos or None
    edit_arr_idx = pair_meta["edit_arr_idx"]
    seq_length = int(parent_thermo["seq_length"])

    # Per-position features (n, 5).
    unpaired = parent_thermo["unpaired_prob"]  # (seq_length,)
    entropy = parent_thermo["positional_entropy_bits"]
    bpp_paired = parent_thermo["bpp_paired_prob"]

    features = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        sp = seq_positions[i]
        if sp is not None and 1 <= sp <= seq_length:
            idx = sp - 1  # 0-indexed
            features[i, 0] = float(unpaired[idx])
            features[i, 1] = float(entropy[idx])
            features[i, 2] = float(bpp_paired[idx])
            features[i, 3] = float(sp) / float(seq_length)  # normalized pos
            features[i, 4] = abs(sp - pair_meta["edit_pos_1indexed"]) / float(seq_length)
        else:
            # Missing position: zero features.
            features[i, 4] = 1.0  # max distance

    # Build edges in array coordinates.
    # 1. Sequence-adjacent edges.
    edges_list: list[tuple[int, int]] = []
    edge_feats_list: list[list[float]] = []
    for i in range(n - 1):
        edges_list.append((i, i + 1))
        edges_list.append((i + 1, i))
        edge_feats_list.append([0.0, 1.0, 0.0])  # (bpp, seq_dist, contact_weight)
        edge_feats_list.append([0.0, 1.0, 0.0])

    # 2. Contact edges (from parent thermo, mapped to array coords).
    contact_edges = parent_thermo["contact_edges"]  # (2, n_contacts) in seq coords (0-indexed)
    contact_weights = parent_thermo["contact_weights"]  # (n_contacts,)

    # Build seq_pos -> array_idx lookup.
    seq_to_arr: dict[int, int] = {}
    for i in range(n):
        sp = seq_positions[i]
        if sp is not None:
            seq_to_arr[sp] = i  # sp is 1-indexed; seq coords in contact_edges are 0-indexed

    if contact_edges.shape[1] > 0:
        for k in range(contact_edges.shape[1]):
            seq_i = int(contact_edges[0, k])  # 0-indexed seq coord
            seq_j = int(contact_edges[1, k])
            # Convert to 1-indexed for lookup.
            arr_i = seq_to_arr.get(seq_i + 1)
            arr_j = seq_to_arr.get(seq_j + 1)
            if arr_i is not None and arr_j is not None and arr_i != arr_j:
                bpp_val = float(contact_weights[k])
                seq_dist = abs(seq_positions[arr_i] - seq_positions[arr_j])
                edges_list.append((arr_i, arr_j))
                edges_list.append((arr_j, arr_i))
                edge_feats_list.append([bpp_val, float(seq_dist), float(contact_weights[k])])
                edge_feats_list.append([bpp_val, float(seq_dist), float(contact_weights[k])])

    edges = torch.tensor(edges_list, dtype=torch.long).T  # (2, n_edges)
    edge_features = torch.tensor(edge_feats_list, dtype=torch.float32)  # (n_edges, 3)

    # Delta true and mask from PairRecord.
    delta_true = torch.tensor(pair_record.delta_true, dtype=torch.float32)
    endpoint_mask = torch.tensor(pair_record.endpoint_mask, dtype=torch.bool)

    # Measurement variance.
    meas_var = pair_meta.get("measurement_variance")
    if meas_var is not None:
        meas_var = float(meas_var)

    return PairData(
        pair_id=pair_meta["pair_id"],
        parent=pair_meta["parent"],
        split=pair_meta["split"],
        features=torch.tensor(features, dtype=torch.float32),
        edges=edges,
        edge_features=edge_features,
        edit_pos=edit_arr_idx if edit_arr_idx is not None else 0,
        delta_true=delta_true,
        endpoint_mask=endpoint_mask,
        pair_quality_weight=pair_record.pair_quality_weight,
        measurement_variance=meas_var,
        n=n,
    )


def load_dataset(config: dict, split: str = "train") -> list[PairData]:
    """Load all pairs for a split as PairData objects."""

    manifest_path = config["data"]["manifest_path"]
    parent_thermo_dir = config["data"]["parent_thermo_dir"]
    registry_path = config["data"]["registry_path"]
    split_path = config["data"]["split_path"]

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load parent thermo npz files (cached).
    parent_cache: dict[str, dict] = {}
    for parent, info in manifest["per_parent"].items():
        npz_path = os.path.join(parent_thermo_dir, os.path.basename(info["npz_path"]))
        parent_cache[parent] = load_parent_thermo(npz_path)

    # Load PairRecords via evaluate.py (for delta_true, endpoint_mask).
    # We need rdat_loader for seq_positions, but we already have them in the
    # manifest. However, load_split_pairs uses rdat_loader to fill seq_positions.
    # We'll use load_split_pairs without rdat_loader (seq_positions will be NaN,
    # but we don't need them since we have them from the manifest).
    records = load_split_pairs(
        split,
        registry_path=registry_path,
        split_members_path=split_path,
    )
    record_by_pid = {r.pair_id: r for r in records}

    # Build PairData for each pair in the split.
    dataset: list[PairData] = []
    for pm in manifest["per_pair"]:
        if pm["split"] != split:
            continue
        pid = pm["pair_id"]
        if pid not in record_by_pid:
            continue
        parent = pm["parent"]
        thermo = parent_cache[parent]
        pd = build_pair_data(pm, thermo, record_by_pid[pid])
        dataset.append(pd)

    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(model: EPROModel, dataset: list[PairData], loss_fn,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    config: dict, grad_clip: float = 1.0) -> dict[str, float]:
    """Train for one epoch over the dataset."""

    model.train()
    total_loss = 0.0
    total_wmae = 0.0
    n_pairs = 0

    # Shuffle order.
    indices = np.random.permutation(len(dataset))

    for idx in indices:
        pd = dataset[idx]
        optimizer.zero_grad()

        batch = {
            "features": pd.features.to(device),
            "edit_pos": pd.edit_pos,
            "edges": pd.edges.to(device),
            "edge_features": pd.edge_features.to(device),
            "mask": pd.endpoint_mask.to(device),
        }

        out = model(batch)
        mu = out["delta_r_hat"]

        # Mask for loss: endpoint_mask AND non-NaN delta_true.
        valid_mask = pd.endpoint_mask.to(device) & ~torch.isnan(pd.delta_true.to(device))
        target = torch.where(torch.isnan(pd.delta_true.to(device)),
                             torch.zeros_like(pd.delta_true.to(device)),
                             pd.delta_true.to(device))

        meas_var = pd.measurement_variance if config["loss"].get("use_measurement_variance", True) else None

        # Handle different loss signatures based on actual loss_fn type.
        if isinstance(loss_fn, WeightedMSELoss):
            loss = loss_fn(mu, target, valid_mask, weight=pd.pair_quality_weight)
        else:
            loss = loss_fn(mu, target, valid_mask, measurement_variance=meas_var)

        if torch.isfinite(loss) and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_wmae += (mu.detach() - target).abs()[valid_mask].mean().item() if valid_mask.sum() > 0 else 0.0
            n_pairs += 1

    return {
        "loss": total_loss / max(n_pairs, 1),
        "wmae": total_wmae / max(n_pairs, 1),
        "n_pairs": n_pairs,
    }


@torch.no_grad()
def evaluate_model(model: EPROModel, dataset: list[PairData],
                   device: torch.device) -> dict[str, Any]:
    """Evaluate model on a dataset, computing per-pair Skill."""

    model.eval()
    pair_metrics_list = []
    skills = []
    predictions: dict[str, np.ndarray] = {}

    for pd in dataset:
        batch = {
            "features": pd.features.to(device),
            "edit_pos": pd.edit_pos,
            "edges": pd.edges.to(device),
            "edge_features": pd.edge_features.to(device),
            "mask": pd.endpoint_mask.to(device),
        }
        out = model(batch)
        mu = out["delta_r_hat"].detach().cpu().numpy()
        predictions[pd.pair_id] = mu

        # Compute per-pair Skill (matching evaluator).
        from reactflow.delta.evaluate import PairRecord
        delta_true = pd.delta_true.numpy()
        mask = pd.endpoint_mask.numpy()

        if mask.sum() == 0:
            continue

        true = delta_true[mask]
        pred = mu[mask]
        wmae_pred = np.mean(np.abs(pred - true))
        wmae_zero = np.mean(np.abs(true))
        skill = 1.0 - wmae_pred / wmae_zero if wmae_zero > 0 else float("nan")
        if not np.isnan(skill):
            skills.append(skill)

    mean_skill = float(np.nanmean(skills)) if skills else float("nan")
    return {
        "mean_skill": mean_skill,
        "n_pairs": len(skills),
        "n_total": len(dataset),
        "predictions": predictions,
    }


def run_training(config: dict, mode: str, output_dir: str) -> dict[str, Any]:
    """Main training entry point."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed.
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device check: GPU required.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        # CPU silent fallback: STOP and preserve evidence.
        error_msg = (
            "FATAL: CUDA not available. GPU training required (contract §user requirement 3). "
            "CPU silent fallback is not permitted. Stopping with evidence."
        )
        print(error_msg, file=sys.stderr, flush=True)
        evidence = {
            "error": error_msg,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (output_dir / "gpu_failure_evidence.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}", flush=True)

    # Load data.
    print("Loading train data...", flush=True)
    train_dataset = load_dataset(config, split="train")
    print(f"  train: {len(train_dataset)} pairs", flush=True)

    print("Loading validation data...", flush=True)
    val_dataset = load_dataset(config, split="validation")
    print(f"  validation: {len(val_dataset)} pairs", flush=True)

    # Mode-specific subset.
    if mode == "overfit":
        n_pairs = config["training"].get("overfit_n_pairs", 1)
        train_dataset = train_dataset[:n_pairs]
        print(f"  overfit mode: using {len(train_dataset)} train pairs", flush=True)
        max_epochs = 5000
    elif mode == "small_batch":
        n_pairs = config["training"].get("small_batch_n_pairs", 8)
        train_dataset = train_dataset[:n_pairs]
        print(f"  small_batch mode: using {len(train_dataset)} train pairs", flush=True)
        max_epochs = config["training"].get("small_batch_max_epochs", 3000)
    elif mode == "pilot":
        max_epochs = config["training"].get("pilot_max_epochs", 100)
    else:
        max_epochs = config["training"]["max_epochs"]

    # Build model.
    model_config = EPROConfig(
        model_type=config["model"]["model_type"],
        latent_dim=config["model"]["latent_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        n_encoder_layers=config["model"]["n_encoder_layers"],
        local_window=config["model"]["local_window"],
        rho_max=config["model"]["rho_max"],
        neumann_iter=config["model"]["neumann_iter"],
        switch_enabled=config["model"]["switch_enabled"],
    )
    model = EPROModel(model_config).to(device)
    param_count = model.param_count()
    print(f"Model: {model_config.model_type}, params: {param_count:,}", flush=True)

    # Loss. Allow mode-specific loss type override (e.g. MSE for overfit,
    # Student-t NLL for pilot).
    loss_config = dict(config["loss"])
    mode_loss_key = f"{mode}_loss_type"
    if mode_loss_key in loss_config:
        loss_config["type"] = loss_config[mode_loss_key]
        print(f"  {mode} mode: using loss={loss_config['type']} (override)", flush=True)
    if loss_config["type"] == "student_t":
        loss_fn = StudentTNLL(
            learned_scale=loss_config["learned_scale"],
            init_df=loss_config["init_df"],
            min_df=loss_config["min_df"],
            max_df=loss_config["max_df"],
            min_sigma=loss_config["min_sigma"],
            fixed_sigma=loss_config.get("fixed_sigma", 1.0),
        ).to(device)
    elif loss_config["type"] == "heteroscedastic":
        loss_fn = HeteroscedasticNLL().to(device)
    elif loss_config["type"] == "mse":
        loss_fn = WeightedMSELoss().to(device)  # true MSE for overfit (gradient ∝ error)
    else:
        raise ValueError(f"unknown loss type: {loss_config['type']}")

    # Optimizer. Allow mode-specific lr override for stability.
    opt_config = config["optimizer"]
    lr = opt_config["lr"]
    mode_lr_key = f"{mode}_lr"
    if mode_lr_key in opt_config:
        lr = opt_config[mode_lr_key]
        print(f"  {mode} mode: using lr={lr} (override)", flush=True)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr,
        weight_decay=opt_config.get("weight_decay", 0.0),
    )

    # Scheduler. T_max is tied to the actual max_epochs (not config) so cosine
    # decay matches the training duration for each mode.
    sched_config = config.get("scheduler", {})
    if sched_config.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs,
            eta_min=sched_config.get("eta_min", 1e-6),
        )
    else:
        scheduler = None

    # Training loop.
    log = {
        "schema_version": TRAIN_SCHEMA_VERSION,
        "mode": mode,
        "model_type": model_config.model_type,
        "param_count": param_count,
        "gpu_name": gpu_name,
        "seed": seed,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "max_epochs": max_epochs,
        "epochs": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    best_val_skill = float("-inf")
    best_epoch = -1
    log_every = config["training"].get("log_every", 10)
    eval_every = config["training"].get("eval_every", 20)
    grad_clip = opt_config.get("grad_clip", 1.0)

    t_start = time.time()
    for epoch in range(max_epochs):
        train_stats = train_one_epoch(
            model, train_dataset, loss_fn, optimizer, device, config, grad_clip
        )
        if scheduler is not None:
            scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_wmae": train_stats["wmae"],
        }

        # Evaluate.
        if (epoch + 1) % eval_every == 0 or epoch == max_epochs - 1:
            train_eval = evaluate_model(model, train_dataset, device)
            val_eval = evaluate_model(model, val_dataset, device)
            epoch_log["train_skill"] = train_eval["mean_skill"]
            epoch_log["val_skill"] = val_eval["mean_skill"]

            if val_eval["mean_skill"] > best_val_skill:
                best_val_skill = val_eval["mean_skill"]
                best_epoch = epoch
                # Save best checkpoint.
                ckpt_path = output_dir / "best_checkpoint.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss_state_dict": loss_fn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_skill": val_eval["mean_skill"],
                    "param_count": param_count,
                }, str(ckpt_path))

        epoch_log["elapsed_s"] = time.time() - t_start
        log["epochs"].append(epoch_log)

        if (epoch + 1) % log_every == 0 or epoch == max_epochs - 1:
            ts = f"loss={train_stats['loss']:.4f} wmae={train_stats['wmae']:.4f}"
            if "val_skill" in epoch_log:
                ts += f" train_skill={epoch_log.get('train_skill', 'NA'):.4f} val_skill={epoch_log['val_skill']:.4f}"
            print(f"  epoch {epoch+1}/{max_epochs}: {ts} ({epoch_log['elapsed_s']:.0f}s)", flush=True)

    # Final evaluation.
    final_train = evaluate_model(model, train_dataset, device)
    final_val = evaluate_model(model, val_dataset, device)
    log["final"] = {
        "train_skill": final_train["mean_skill"],
        "val_skill": final_val["mean_skill"],
        "best_val_skill": best_val_skill,
        "best_epoch": best_epoch,
        "total_elapsed_s": time.time() - t_start,
    }

    # Save final predictions.
    pred_path = output_dir / "predictions_val.json"
    pred_serializable = {k: [float(x) for x in v] for k, v in final_val["predictions"].items()}
    pred_path.write_text(json.dumps(pred_serializable, indent=2), encoding="utf-8")

    # Save log.
    log_path = output_dir / "train_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\nDone. best_val_skill={best_val_skill:.4f} (epoch {best_epoch})", flush=True)
    print(f"  train_skill={final_train['mean_skill']:.4f} val_skill={final_val['mean_skill']:.4f}", flush=True)
    print(f"  log: {log_path}", flush=True)

    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", required=True, choices=["overfit", "small_batch", "pilot"],
                        help="Training mode")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_training(config, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
