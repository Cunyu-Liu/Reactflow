#!/usr/bin/env python3
"""Diagnostic: check model output magnitudes and gradient flow."""

from __future__ import annotations
import os
import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reactflow.delta.model import EPROModel, EPROConfig
from reactflow.delta.losses import WeightedMAELoss, WeightedMSELoss
from reactflow.delta.evaluate import load_split_pairs


@dataclass
class PairData:
    pair_id: str
    parent: str
    features: torch.Tensor
    edges: torch.Tensor
    edge_features: torch.Tensor
    edit_pos: int
    delta_true: torch.Tensor
    endpoint_mask: torch.Tensor
    pair_quality_weight: float
    n: int


def load_parent_thermo(npz_path):
    return dict(np.load(npz_path))


def build_pair_data(pair_meta, parent_thermo, pair_record):
    n = pair_meta["aligned_length"]
    seq_positions = pair_meta["seq_positions"]
    edit_arr_idx = pair_meta["edit_arr_idx"]
    seq_length = int(parent_thermo["seq_length"])

    unpaired = parent_thermo["unpaired_prob"]
    entropy = parent_thermo["positional_entropy_bits"]
    bpp_paired = parent_thermo["bpp_paired_prob"]

    features = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        sp = seq_positions[i]
        if sp is not None and 1 <= sp <= seq_length:
            idx = sp - 1
            features[i, 0] = float(unpaired[idx])
            features[i, 1] = float(entropy[idx])
            features[i, 2] = float(bpp_paired[idx])
            features[i, 3] = float(sp) / float(seq_length)
            features[i, 4] = abs(sp - pair_meta["edit_pos_1indexed"]) / float(seq_length)
        else:
            features[i, 4] = 1.0

    edges_list = []
    edge_feats_list = []
    for i in range(n - 1):
        edges_list.append((i, i + 1))
        edges_list.append((i + 1, i))
        edge_feats_list.append([0.0, 1.0, 0.0])
        edge_feats_list.append([0.0, 1.0, 0.0])

    contact_edges = parent_thermo["contact_edges"]
    contact_weights = parent_thermo["contact_weights"]

    seq_to_arr = {}
    for i in range(n):
        sp = seq_positions[i]
        if sp is not None:
            seq_to_arr[sp] = i

    if contact_edges.shape[1] > 0:
        for k in range(contact_edges.shape[1]):
            seq_i = int(contact_edges[0, k])
            seq_j = int(contact_edges[1, k])
            arr_i = seq_to_arr.get(seq_i + 1)
            arr_j = seq_to_arr.get(seq_j + 1)
            if arr_i is not None and arr_j is not None and arr_i != arr_j:
                bpp_val = float(contact_weights[k])
                seq_dist = abs(seq_positions[arr_i] - seq_positions[arr_j])
                edges_list.append((arr_i, arr_j))
                edges_list.append((arr_j, arr_i))
                edge_feats_list.append([bpp_val, float(seq_dist), float(contact_weights[k])])
                edge_feats_list.append([bpp_val, float(seq_dist), float(contact_weights[k])])

    edges = torch.tensor(edges_list, dtype=torch.long).T
    edge_features = torch.tensor(edge_feats_list, dtype=torch.float32)

    delta_true = torch.tensor(pair_record.delta_true, dtype=torch.float32)
    endpoint_mask = torch.tensor(pair_record.endpoint_mask, dtype=torch.bool)

    return PairData(
        pair_id=pair_meta["pair_id"],
        parent=pair_meta["parent"],
        features=torch.tensor(features, dtype=torch.float32),
        edges=edges,
        edge_features=edge_features,
        edit_pos=edit_arr_idx if edit_arr_idx is not None else 0,
        delta_true=delta_true,
        endpoint_mask=endpoint_mask,
        pair_quality_weight=pair_record.pair_quality_weight,
        n=n,
    )


def main():
    config_path = "configs/reactflow_delta/epro_lite.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")

    manifest_path = config["data"]["manifest_path"]
    parent_thermo_dir = config["data"]["parent_thermo_dir"]
    registry_path = config["data"]["registry_path"]
    split_path = config["data"]["split_path"]

    print("Loading manifest...", flush=True)
    with open(manifest_path) as f:
        manifest = json.load(f)

    parent_cache = {}
    for parent, info in manifest["per_parent"].items():
        npz_path = os.path.join(parent_thermo_dir, os.path.basename(info["npz_path"]))
        parent_cache[parent] = load_parent_thermo(npz_path)

    print("Loading split pairs...", flush=True)
    records = load_split_pairs("train", registry_path=registry_path, split_members_path=split_path)
    record_by_pid = {r.pair_id: r for r in records}

    # Build first pair
    pd = None
    for pm in manifest["per_pair"]:
        if pm["split"] != "train":
            continue
        pid = pm["pair_id"]
        if pid not in record_by_pid:
            continue
        parent = pm["parent"]
        thermo = parent_cache[parent]
        pd = build_pair_data(pm, thermo, record_by_pid[pid])
        break

    print(f"Pair: {pd.pair_id}, n={pd.n}, edit_pos={pd.edit_pos}", flush=True)
    print(f"  delta_true range: [{pd.delta_true.min().item():.6f}, {pd.delta_true.max().item():.6f}]", flush=True)
    print(f"  delta_true abs mean (WMAE zero): {pd.delta_true.abs().mean().item():.6f}", flush=True)
    valid_mask = pd.endpoint_mask & ~torch.isnan(pd.delta_true)
    print(f"  valid positions: {valid_mask.sum().item()}", flush=True)
    print(f"  pair_quality_weight: {pd.pair_quality_weight}", flush=True)
    print(f"  features range: [{pd.features.min().item():.4f}, {pd.features.max().item():.4f}]", flush=True)
    print(f"  edges: {pd.edges.shape}", flush=True)

    # Build model
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
    print(f"\nModel params: {model.param_count():,}", flush=True)
    print(f"output_scale: {model.output_scale.item():.6f}", flush=True)

    # Forward pass
    batch = {
        "features": pd.features.to(device),
        "edit_pos": pd.edit_pos,
        "edges": pd.edges.to(device),
        "edge_features": pd.edge_features.to(device),
        "mask": pd.endpoint_mask.to(device),
    }

    model.eval()
    with torch.no_grad():
        out = model(batch)

    print(f"\n=== Forward pass magnitudes ===", flush=True)
    for key in ["z_w", "delta", "b", "h_lin", "h", "delta_r_hat"]:
        t = out[key]
        print(f"  {key}: shape={t.shape}, mean={t.mean().item():.6f}, std={t.std().item():.6f}, abs_max={t.abs().max().item():.6f}", flush=True)

    K = out["K"]
    print(f"  K: shape={K.shape}, abs_max={K.abs().max().item():.6f}", flush=True)
    # Check spectral radius
    eigvals = torch.linalg.eigvals(K)
    rho = eigvals.abs().max().item()
    print(f"  K spectral radius: {rho:.6f} (rho_max={model_config.rho_max})", flush=True)

    delta_r_hat = out["delta_r_hat"]
    print(f"\n  delta_r_hat abs mean: {delta_r_hat.abs().mean().item():.6f}", flush=True)
    print(f"  delta_true abs mean: {pd.delta_true.abs().mean().item():.6f}", flush=True)

    # Gradient check
    print(f"\n=== Gradient flow check ===", flush=True)
    model.train()
    model.zero_grad()

    out = model(batch)
    mu = out["delta_r_hat"]
    target = torch.where(torch.isnan(pd.delta_true), torch.zeros_like(pd.delta_true), pd.delta_true)

    loss_fn = WeightedMSELoss()
    loss = loss_fn(mu, target, valid_mask, weight=pd.pair_quality_weight)
    print(f"Loss (MSE): {loss.item():.6f}, requires_grad: {loss.requires_grad}", flush=True)

    loss.backward()

    print(f"\nPer-parameter gradient magnitudes:", flush=True)
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.data.norm().item()
            print(f"  {name}: param={param_norm:.6f}, grad={grad_norm:.6f}, ratio={grad_norm/(param_norm+1e-12):.6f}", flush=True)
        else:
            print(f"  {name}: NO GRADIENT", flush=True)

    # One optimizer step
    print(f"\n=== One optimizer step (MSE loss, lr=5e-3) ===", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    optimizer.zero_grad()
    out = model(batch)
    mu = out["delta_r_hat"]
    loss = loss_fn(mu, target, valid_mask, weight=pd.pair_quality_weight)
    print(f"  MSE loss: {loss.item():.6f}", flush=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out2 = model(batch)
    print(f"  Before: delta_r_hat abs mean = {delta_r_hat.abs().mean().item():.6f}", flush=True)
    print(f"  After:  delta_r_hat abs mean = {out2['delta_r_hat'].abs().mean().item():.6f}", flush=True)
    print(f"  output_scale: {model.output_scale.item():.6f}", flush=True)

    wmae_before = (mu.detach() - target).abs()[valid_mask].mean().item()
    wmae_after = (out2["delta_r_hat"].detach() - target).abs()[valid_mask].mean().item()
    wmae_zero = target.abs()[valid_mask].mean().item()
    print(f"\n  WMAE before: {wmae_before:.6f}", flush=True)
    print(f"  WMAE after:  {wmae_after:.6f}", flush=True)
    print(f"  WMAE zero:   {wmae_zero:.6f}", flush=True)

    # Multiple steps with lower lr (reinitialize model first)
    print(f"\n=== 200 steps with lr=1e-4 (MSE loss, fresh model) ===", flush=True)
    torch.manual_seed(42)
    model2 = EPROModel(model_config).to(device)
    model2.train()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    for step in range(200):
        optimizer2.zero_grad()
        out = model2(batch)
        mu = out["delta_r_hat"]
        loss = loss_fn(mu, target, valid_mask, weight=pd.pair_quality_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        optimizer2.step()
        if (step + 1) % 20 == 0:
            model2.eval()
            with torch.no_grad():
                out_e = model2(batch)
            wmae = (out_e["delta_r_hat"].detach() - target).abs()[valid_mask].mean().item()
            print(f"  step {step+1}: loss={loss.item():.6f}, wmae={wmae:.6f}, output_abs_mean={out_e['delta_r_hat'].abs().mean().item():.6f}", flush=True)
            model2.train()


if __name__ == "__main__":
    main()
