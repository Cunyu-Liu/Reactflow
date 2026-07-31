#!/usr/bin/env python3
"""Diagnose pilot_v3 val_skill=-0.71 failure.

Checks:
1. delta_true distribution per parent (mean/std/abs_mean/max)
2. Zero-change baseline (abs_mean) per parent per split
3. Model val predictions vs targets (bias, scale, correlation)
4. Feature distribution shift train vs val
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
import torch  # noqa: E402

from reactflow.delta.evaluate import load_split_pairs  # noqa: E402
from train import load_dataset, load_parent_thermo  # noqa: E402
from reactflow.delta.model import EPROModel, EPROConfig  # noqa: E402

CONFIG_PATH = "configs/reactflow_delta/epro_lite.yaml"
PILOT_DIR = "/mnt/cunyuliu/reactflow_delta_artifacts_20260729/m0/pilot_v3"


def main():
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("1. delta_true distribution per parent per split")
    print("=" * 70)

    # Load manifest
    manifest = json.load(open(config["data"]["manifest_path"]))

    # Load all pairs' delta_true via load_split_pairs
    by_parent_split = defaultdict(lambda: defaultdict(list))
    for split in ("train", "validation", "test"):
        records = load_split_pairs(
            split,
            registry_path=config["data"]["registry_path"],
            split_members_path=config["data"]["split_path"],
        )
        # Map by parent
        pid_to_parent = {p["pair_id"]: p["parent"] for p in manifest["per_pair"]}
        for r in records:
            parent = pid_to_parent.get(r.pair_id, "?")
            dt = r.delta_true
            mask = r.endpoint_mask
            valid = dt[mask]
            by_parent_split[parent][split].append(valid)

    print(f"{'parent':<45} {'split':<10} {'n_pair':>7} {'n_pos':>8} {'mean':>9} {'std':>8} {'abs_mean':>9} {'max':>7}")
    for parent in sorted(by_parent_split.keys()):
        for split in ("train", "validation", "test"):
            arrs = by_parent_split[parent][split]
            if not arrs:
                continue
            all_d = np.concatenate(arrs)
            print(f"{parent[:45]:<45} {split:<10} {len(arrs):>7} {len(all_d):>8} "
                  f"{all_d.mean():>+9.4f} {all_d.std():>8.4f} {np.abs(all_d).mean():>9.4f} {np.abs(all_d).max():>7.3f}")

    print()
    print("=" * 70)
    print("2. Model val predictions vs targets")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = load_dataset(config, split="validation")
    print(f"val pairs: {len(val_dataset)}")

    # Load checkpoint
    ckpt = torch.load(f"{PILOT_DIR}/best_checkpoint.pt", map_location=device, weights_only=False)
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']} val_skill={ckpt['val_skill']:.4f}")

    all_pred = []
    all_true = []
    all_parent = []
    for pd in val_dataset:
        batch = {
            "features": pd.features.to(device),
            "edit_pos": pd.edit_pos,
            "edges": pd.edges.to(device),
            "edge_features": pd.edge_features.to(device),
            "mask": pd.endpoint_mask.to(device),
        }
        with torch.no_grad():
            out = model(batch)
        mu = out["delta_r_hat"].detach().cpu().numpy()
        mask = pd.endpoint_mask.numpy()
        dt = pd.delta_true.numpy()
        valid = mask & ~np.isnan(dt)
        all_pred.append(mu[valid])
        all_true.append(dt[valid])
        all_parent.append([pd.parent] * int(valid.sum()))

    preds = np.concatenate(all_pred)
    trues = np.concatenate(all_true)
    parents = np.concatenate(all_parent)

    print(f"\nAll val (n={len(preds)}):")
    print(f"  pred:  mean={preds.mean():+.4f} std={preds.std():.4f} min={preds.min():+.4f} max={preds.max():+.4f} abs_mean={np.abs(preds).mean():.4f}")
    print(f"  true:  mean={trues.mean():+.4f} std={trues.std():.4f} min={trues.min():+.4f} max={trues.max():+.4f} abs_mean={np.abs(trues).mean():.4f}")
    print(f"  zero-change WMAE (abs_mean(true)): {np.abs(trues).mean():.4f}")
    print(f"  model WMAE: {np.mean(np.abs(preds - trues)):.4f}")
    skill = 1.0 - np.mean(np.abs(preds - trues)) / np.mean(np.abs(trues))
    print(f"  Skill: {skill:+.4f}")
    print(f"  pred bias (pred-true mean): {(preds - trues).mean():+.4f}")
    print(f"  Pearson r: {np.corrcoef(preds, trues)[0,1]:+.4f}")

    print()
    print("=" * 70)
    print("3. Feature distribution: train vs val")
    print("=" * 70)

    train_dataset = load_dataset(config, split="train")

    def feat_stats(ds, key):
        arrs = [pd.features.numpy() for pd in ds]
        mask = np.concatenate([pd.endpoint_mask.numpy() for pd in ds])
        feats = np.concatenate(arrs, axis=0)
        # feats may not align with mask length per-row, so just use all rows
        return feats

    tf = feat_stats(train_dataset, None)
    vf = feat_stats(val_dataset, None)
    feat_names = ["unpaired", "entropy", "bpp_paired", "norm_pos", "edit_dist"]
    print(f"{'feat':<14} {'train_mean':>10} {'train_std':>9} {'val_mean':>10} {'val_std':>9} {'shift_sigma':>12}")
    for i, name in enumerate(feat_names):
        tm, ts = tf[:, i].mean(), tf[:, i].std()
        vm, vs = vf[:, i].mean(), vf[:, i].std()
        shift = abs(tm - vm) / (ts + 1e-8)
        print(f"{name:<14} {tm:>10.4f} {ts:>9.4f} {vm:>10.4f} {vs:>9.4f} {shift:>12.2f}")

    print()
    print("=" * 70)
    print("4. Per-parent train skill (does model fit each train parent?)")
    print("=" * 70)
    train_subset = train_dataset
    by_parent = defaultdict(list)
    for pd in train_subset:
        by_parent[pd.parent].append(pd)
    print(f"{'parent':<45} {'n_pair':>7} {'pred_abs':>10} {'true_abs':>10} {'WMAE':>9} {'skill':>8}")
    for parent, pds in by_parent.items():
        ps, ts = [], []
        for pd in pds:
            batch = {
                "features": pd.features.to(device),
                "edit_pos": pd.edit_pos,
                "edges": pd.edges.to(device),
                "edge_features": pd.edge_features.to(device),
                "mask": pd.endpoint_mask.to(device),
            }
            with torch.no_grad():
                out = model(batch)
            mu = out["delta_r_hat"].detach().cpu().numpy()
            mask = pd.endpoint_mask.numpy()
            dt = pd.delta_true.numpy()
            valid = mask & ~np.isnan(dt)
            ps.append(mu[valid])
            ts.append(dt[valid])
        p = np.concatenate(ps)
        t = np.concatenate(ts)
        wmae = np.mean(np.abs(p - t))
        zero = np.mean(np.abs(t))
        sk = 1.0 - wmae / zero if zero > 0 else float("nan")
        print(f"{parent[:45]:<45} {len(pds):>7} {np.abs(p).mean():>10.4f} {zero:>10.4f} {wmae:>9.4f} {sk:>+8.4f}")

    print("\nDIAGNOSIS SUMMARY:")
    print(f"  val_skill = {skill:+.4f}")
    print(f"  val parent: {parents[0] if len(parents) else '?'}")
    print(f"  val |true| mean = {np.abs(trues).mean():.4f}, |pred| mean = {np.abs(preds).mean():.4f}")
    print(f"  pred bias = {(preds - trues).mean():+.4f}")
    print(f"  Pearson r = {np.corrcoef(preds, trues)[0,1]:+.4f}")


if __name__ == "__main__":
    main()
