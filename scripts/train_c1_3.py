#!/usr/bin/env python3
"""Training script for Phase C1-3 (StaticPairFormer, full-scale SOTA training).

Usage::

    PYTHONPATH=src python scripts/train_c1_3.py --config configs/models/c1_3/foo.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from reactflow.backbones.embeddings import PAD_INDEX, encode_sequence
from reactflow.curriculum import CurriculumConfig, CurriculumSampler, CurriculumStage
from reactflow.decoders import DecoderConfig, decode
from reactflow.losses import LossConfig, pairformer_loss
from reactflow.models.static_pairformer import PairFormerConfig, StaticPairFormer
from reactflow.training_engine import TrainingConfig, TrainingEngine

try:
    from reactflow.constraints import validate_pair_matrix
    _HAS_CONSTRAINTS = True
except ImportError:
    _HAS_CONSTRAINTS = False
    validate_pair_matrix = None  # type: ignore[assignment]

_NPZ_CACHE: Dict[str, Any] = {}
DIST_BINS: Dict[str, Tuple[int, int]] = {"short": (1, 11), "medium": (12, 23), "long": (24, 10_000)}
_VOCAB_IDX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "U", 4: "N", 5: "N"}


@dataclass
class DataRecord:
    """Duck-typed record consumed by CurriculumSampler and C1_3Dataset."""
    source: str
    family: str
    clan: str
    sequence: str
    pairs: List[Tuple[int, int]]
    record_id: str
    length_bucket: str
    dataset_index: int = 0


class FrozenFeatureStore:
    """Lazy index over RibonanzaNet2 sharded frozen features.

    Builds ``record_id -> (npz_path, row)`` by scanning ``index.jsonl`` files.
    NPZ handles are cached per-worker via the module-level ``_NPZ_CACHE``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._index: Dict[str, Tuple[str, int]] = {}
        root = self.root
        if not root.exists():
            return
        if (root / "provenance.json").exists():
            shard_dirs: List[Path] = [root]
        else:
            shard_dirs = sorted(
                p for p in root.iterdir() if p.is_dir() and (p / "index.jsonl").exists()
            )
        for sd in shard_dirs:
            idx_path = sd / "index.jsonl"
            if not idx_path.exists():
                continue
            npz_path = str(sd / "features.npz")
            for line in idx_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    self._index[str(e["record_id"])] = (npz_path, int(e["row"]))

    def get(self, record_id: str, length: int, dim: int = 384) -> Optional[torch.Tensor]:
        if record_id not in self._index:
            return None
        npz_path, row = self._index[record_id]
        if npz_path not in _NPZ_CACHE:
            _NPZ_CACHE[npz_path] = np.load(npz_path, allow_pickle=False)
        arr = _NPZ_CACHE[npz_path][f"{row:06d}.single"]
        out = np.zeros((length, dim), dtype=np.float32)
        l = min(arr.shape[0], length)
        out[:l] = arr[:l].astype(np.float32)
        return torch.from_numpy(out)

    def __len__(self) -> int:
        return len(self._index)


class C1_3Dataset(Dataset):
    """Yields dicts with indices, target_matrix, mask, record_id, length, frozen_features."""

    def __init__(self, records: Sequence[DataRecord],
                 feature_store: Optional[FrozenFeatureStore] = None,
                 frozen_dim: int = 0) -> None:
        self.records = list(records)
        self.feature_store = feature_store
        self.frozen_dim = frozen_dim

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        seq, L = rec.sequence, len(rec.sequence)
        target = torch.zeros(L, L, dtype=torch.float32)
        for i, j in rec.pairs:
            if 0 <= i < L and 0 <= j < L:
                target[i, j] = target[j, i] = 1.0
        item: Dict[str, Any] = {
            "indices": encode_sequence(seq), "target_matrix": target,
            "mask": torch.ones(L, dtype=torch.bool), "record_id": rec.record_id, "length": L,
        }
        if self.frozen_dim > 0 and self.feature_store is not None:
            feat = self.feature_store.get(rec.record_id, L, self.frozen_dim)
            item["frozen_features"] = feat if feat is not None else torch.zeros(L, self.frozen_dim)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length samples to max length in batch."""
    B, max_len = len(batch), max(item["length"] for item in batch)
    indices = torch.full((B, max_len), PAD_INDEX, dtype=torch.long)
    targets = torch.zeros(B, max_len, max_len, dtype=torch.float32)
    masks = torch.zeros(B, max_len, dtype=torch.bool)
    has_frozen = batch[0].get("frozen_features") is not None
    frozen = torch.zeros(B, max_len, batch[0]["frozen_features"].shape[-1],
                         dtype=torch.float32) if has_frozen else None
    record_ids, lengths = [], []
    for i, item in enumerate(batch):
        L = item["length"]
        indices[i, :L] = item["indices"]
        targets[i, :L, :L] = item["target_matrix"]
        masks[i, :L] = item["mask"]
        if frozen is not None:
            frozen[i, :L] = item["frozen_features"]
        record_ids.append(item["record_id"])
        lengths.append(L)
    out: Dict[str, Any] = {"indices": indices, "targets": targets, "mask": masks,
                           "record_ids": record_ids, "lengths": lengths}
    if frozen is not None:
        out["frozen_features"] = frozen
    return out


class BatchPairFormer(nn.Module):
    """Wraps StaticPairFormer to accept a batch dict (required by TrainingEngine)."""
    def __init__(self, model: StaticPairFormer) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: Dict[str, Any]) -> Any:
        kwargs: Dict[str, Any] = {"mask": batch["mask"]}
        if batch.get("frozen_features") is not None:
            kwargs["frozen_features"] = batch["frozen_features"]
        return self.model(batch["indices"], **kwargs)

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing_disable()


def _indices_to_seq(indices: torch.Tensor, length: int) -> str:
    return "".join(_VOCAB_IDX_TO_BASE.get(int(i), "N") for i in indices[:length].tolist())


def _dilate(mat: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Dilate a binary (L, L) matrix by k positions (8-connectivity)."""
    L = mat.shape[-1]
    padded = torch.nn.functional.pad(mat, (k, k, k, k))
    out = torch.zeros_like(mat)
    for di in range(2 * k + 1):
        for dj in range(2 * k + 1):
            out = torch.maximum(out, padded[..., di:di + L, dj:dj + L])
    return out


def _accumulate(pred: torch.Tensor, target: torch.Tensor, counts: Dict[str, int]) -> None:
    """Accumulate TP/FP/FN/TN (exact + shifted) and per-distance-bin counts."""
    L = pred.shape[-1]
    upper = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    p, t = (pred > 0.5) & upper, (target > 0.5) & upper
    counts["tp"] += int((p & t).sum())
    counts["fp"] += int((p & ~t).sum())
    counts["fn"] += int((~p & t).sum())
    counts["tn"] += int((~p & ~t & upper).sum())
    t_s = (_dilate(target.float()) > 0.5) & upper
    counts["tp_shifted"] += int((p & t_s).sum())
    counts["fp_shifted"] += int((p & ~t_s).sum())
    counts["fn_shifted"] += int((~p & t).sum())
    idx = torch.arange(L)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    for bn, (lo, hi) in DIST_BINS.items():
        bm = (dist >= lo) & (dist <= hi) & upper
        pb, tb = p & bm, t & bm
        counts[f"{bn}_tp"] += int((pb & tb).sum())
        counts[f"{bn}_fp"] += int((pb & ~tb).sum())
        counts[f"{bn}_fn"] += int((~pb & tb).sum())
    if int(p.sum()) == 0:
        counts["empty_pred"] += 1
    counts["total_samples"] += 1


def _finalize(counts: Dict[str, int]) -> Dict[str, float]:
    """Compute final metrics from accumulated counts."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    eps = 1e-8
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    denom = max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)
    mcc = (tp * tn - fp * fn) / (denom ** 0.5 + eps)
    ts, fs = counts["tp_shifted"], counts["fp_shifted"]
    f1s = 2 * ts / max(2 * ts + fs + counts["fn_shifted"], 1)
    res: Dict[str, float] = {
        "f1": float(f1), "precision": float(prec), "recall": float(rec), "mcc": float(mcc),
        "f1_shifted": float(f1s), "empty_rate": counts["empty_pred"] / max(counts["total_samples"], 1),
        "total_samples": float(counts["total_samples"]),
    }
    for bn in DIST_BINS:
        bt, bf, bnn = counts[f"{bn}_tp"], counts[f"{bn}_fp"], counts[f"{bn}_fn"]
        res[f"f1_{bn}"] = float(2 * bt / max(2 * bt + bf + bnn, 1))
    return res


def evaluate_split(model: nn.Module, loader: DataLoader, decoder_cfg: DecoderConfig,
                   device: torch.device, max_samples: int = 0) -> Dict[str, Any]:
    """Evaluate with threshold, nussinov_dp, and mea decoders."""
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    cache: List[Tuple] = []
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            bd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model(bd)
            cache.append((out.logits.cpu(), out.bpp.cpu(), out.unpaired_prob.cpu(),
                          out.temperature.cpu(), batch["indices"].cpu(),
                          out.mask.cpu() if out.mask is not None else None,
                          batch["targets"].cpu(), batch["lengths"]))
            n_seen += len(batch["lengths"])
            if max_samples > 0 and n_seen >= max_samples:
                break
    results: Dict[str, Any] = {}
    for mode in ("threshold", "nussinov_dp", "mea"):
        counts: Dict[str, int] = {
            "tp": 0, "fp": 0, "fn": 0, "tn": 0, "tp_shifted": 0, "fp_shifted": 0,
            "fn_shifted": 0, "empty_pred": 0, "total_samples": 0,
        }
        for bn in DIST_BINS:
            counts[f"{bn}_tp"] = counts[f"{bn}_fp"] = counts[f"{bn}_fn"] = 0
        legal = 0
        for (logits, bpp, unp, temp, indices, mask, targets, lengths) in cache:
            decoded = decode(logits, indices=indices, mask=mask, bpp=bpp,
                             unpaired_prob=unp, temperature=temp, config=decoder_cfg, mode=mode)
            for i in range(len(lengths)):
                L = lengths[i]
                _accumulate(decoded[i, :L, :L], targets[i, :L, :L], counts)
                if _HAS_CONSTRAINTS and validate_pair_matrix is not None:
                    seq = _indices_to_seq(indices[i], L)
                    if validate_pair_matrix(seq, decoded[i, :L, :L].tolist(),
                                            min_loop=decoder_cfg.min_loop,
                                            allow_wobble=decoder_cfg.allow_wobble).valid:
                        legal += 1
        metrics = _finalize(counts)
        if _HAS_CONSTRAINTS:
            metrics["legal_rate"] = legal / max(counts["total_samples"], 1)
        results[mode] = metrics
    results["_runtime_sec"] = time.time() - t0
    results["_peak_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return results


def _filter_fields(d: Dict[str, Any], cls: type) -> Dict[str, Any]:
    """Filter dict to only dataclass fields of *cls*."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _source_from_id(sid: str) -> str:
    """Map source_id prefix to curriculum source name (pilot simplification)."""
    return "Rfam" if sid.startswith("RF") else sid


def _build_records(raw: List[Dict[str, Any]], min_len: int, max_len: int) -> List[DataRecord]:
    """Build filtered DataRecord list from raw JSONL dicts."""
    records: List[DataRecord] = []
    for idx, r in enumerate(raw):
        seq = r["sequence"]
        if not (min_len <= len(seq) <= max_len):
            continue
        records.append(DataRecord(
            source=_source_from_id(r.get("source_id", "")),
            family=r.get("family") or "none", clan=r.get("clan") or "none",
            sequence=seq, pairs=[(int(i), int(j)) for i, j in r.get("pairs", [])],
            record_id=r.get("source_id", str(idx)),
            length_bucket=r.get("length_bucket", ""), dataset_index=idx,
        ))
    return records


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="C1-3 StaticPairFormer training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/c1_3/results"))
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--max-len", type=int, default=None, help="Override config max_length filter")
    args = parser.parse_args()

    # --- Config & seeds ---
    with open(args.config, encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_str = args.device or cfg["training"].get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU", file=sys.stderr)
        device_str = "cpu"
    device = torch.device(device_str)
    ckpt_dir = (args.checkpoint_dir or cfg["training"].get("checkpoint_dir", "checkpoints")).format(seed=seed)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"]
    eval_bs = min(cfg["training"].get("eval_batch_size", batch_size), batch_size)
    n_workers = cfg["training"].get("num_workers", 0)
    pin_mem = cfg["training"].get("pin_memory", False)

    # --- Data ---
    sp = Path(cfg["data"]["splits_path"])
    min_l = cfg["data"]["min_length"]
    max_l = args.max_len if args.max_len is not None else cfg["data"]["max_length"]
    print(f"[INFO] Loading data from {sp}", file=sys.stderr)
    train_recs = _build_records(_load_jsonl(sp / f"{cfg['data']['train_split']}.jsonl"), min_l, max_l)
    val_recs = _build_records(_load_jsonl(sp / f"{cfg['data']['val_split']}.jsonl"), min_l, max_l)
    test_recs = _build_records(_load_jsonl(sp / f"{cfg['data']['test_split']}.jsonl"), min_l, max_l)
    novel_path = sp / f"{cfg['data']['novel_split']}.jsonl"
    novel_recs = _build_records(_load_jsonl(novel_path), min_l, max_l) if novel_path.exists() else []
    if args.max_train_samples > 0:
        train_recs = train_recs[:args.max_train_samples]
    if args.max_eval_samples > 0:
        for lst in (val_recs, test_recs, novel_recs):
            del lst[args.max_eval_samples:]
    print(f"[INFO] train={len(train_recs)} val={len(val_recs)} test={len(test_recs)} novel={len(novel_recs)}",
          file=sys.stderr)

    # --- Frozen features ---
    frozen_dim = cfg["model"].get("frozen_feature_dim", 0)
    fstore: Optional[FrozenFeatureStore] = None
    if frozen_dim > 0:
        fp = cfg["backbone"].get("frozen_features_path", "")
        if fp:
            print(f"[INFO] Building frozen feature index from {fp}", file=sys.stderr)
            fstore = FrozenFeatureStore(fp)
            print(f"[INFO] Frozen feature index: {len(fstore)} records", file=sys.stderr)
    train_ds = C1_3Dataset(train_recs, fstore, frozen_dim)
    val_ds = C1_3Dataset(val_recs, fstore, frozen_dim)
    test_ds = C1_3Dataset(test_recs, fstore, frozen_dim)
    novel_ds = C1_3Dataset(novel_recs, fstore, frozen_dim) if novel_recs else None

    # --- Curriculum ---
    cd = cfg.get("curriculum", {})
    stages = [CurriculumStage(s) for s in cd.get("stages", [])]
    se = {CurriculumStage(k): v for k, v in cd.get("stage_epochs", {}).items()}
    cur_cfg = CurriculumConfig(stages=stages, stage_epochs=se,
                               replay_ratio=cd.get("replay_ratio", 0.2),
                               balance_keys=cd.get("balance_keys", ["source"]), seed=seed)
    curriculum = CurriculumSampler(train_recs, cur_cfg)

    # --- Model & loss ---
    model_cfg = PairFormerConfig(**_filter_fields(cfg["model"], PairFormerConfig))
    base_model = StaticPairFormer(model_cfg)
    model = BatchPairFormer(base_model)
    n_params = base_model.num_parameters()
    print(f"[INFO] Model parameters: {n_params:,}", file=sys.stderr)
    loss_cfg = LossConfig(**_filter_fields(cfg["loss"], LossConfig))
    loss_fn = lambda output, batch: pairformer_loss(output, batch["targets"], config=loss_cfg)

    # --- Engine ---
    tc = _filter_fields(cfg["training"], TrainingConfig)
    tc["device"], tc["seed"], tc["checkpoint_dir"] = device_str, seed, ckpt_dir
    train_cfg = TrainingConfig(**tc)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_mem, collate_fn=collate_fn)
    placeholder = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=n_workers, pin_memory=pin_mem, collate_fn=collate_fn)
    engine = TrainingEngine(model, placeholder, val_loader, train_cfg, loss_fn, device=device)
    engine.setup_distributed()
    engine.setup_optimizer()

    # --- Training loop (manual, per-epoch curriculum sampling) ---
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val, val_hist = float("inf"), []
    ckpt_path = Path(ckpt_dir)
    print(f"[INFO] Training {epochs} epochs on {device}", file=sys.stderr)
    for epoch in range(epochs):
        ep_records = curriculum.get_epoch_records(epoch)
        stage = curriculum.get_stage(epoch)
        if ep_records:
            indices = [r.dataset_index for r in ep_records]
            sampler = curriculum.get_sampler(epoch)
            if sampler is None or len(indices) == 0:
                indices = list(range(len(train_ds)))
                sampler = None
        else:
            # Stage has no matching records (e.g. all data is Rfam but stage
            # expects PDB/viral); fall back to all train records with shuffle.
            indices = list(range(len(train_ds)))
            sampler = None
        subset = Subset(train_ds, indices)
        engine.train_loader = DataLoader(subset, batch_size=batch_size,
                                         sampler=sampler, shuffle=(sampler is None),
                                         num_workers=n_workers, pin_memory=pin_mem,
                                         collate_fn=collate_fn)
        print(f"[INFO] Epoch {epoch}/{epochs} stage={stage.value} recs={len(indices)}", file=sys.stderr)
        tm = engine.train_epoch(epoch)
        history["train_loss"].append(tm["loss"])
        if val_loader is not None and epoch % max(1, train_cfg.eval_every) == 0:
            vm = engine.evaluate()
            vl = vm["loss"]
            history["val_loss"].append(vl)
            val_hist.append(vl)
            print(f"[INFO] Epoch {epoch} train_loss={tm['loss']:.4f} val_loss={vl:.4f}", file=sys.stderr)
            if vl < best_val - train_cfg.early_stop_min_delta:
                best_val = vl
                engine.save_checkpoint(ckpt_path / "best.pt")
            if engine._should_stop_early(val_hist):
                print(f"[INFO] Early stopping at epoch {epoch}", file=sys.stderr)
                break
        if epoch % max(1, train_cfg.save_every) == 0:
            engine.save_checkpoint(ckpt_path / "latest.pt")
    engine.save_checkpoint(ckpt_path / "latest.pt")

    # --- Save training results ---
    training_results = {
        "config": cfg, "seed": seed, "device": device_str, "model_params": n_params,
        "training_history": history, "best_val_loss": best_val, "git_commit": _get_git_commit(),
        "epochs_run": len(history["train_loss"]),
    }
    with open(args.output_dir / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(training_results, f, indent=2, default=str)

    # --- Post-training evaluation ---
    print("[INFO] Loading best checkpoint for evaluation", file=sys.stderr)
    best = ckpt_path / "best.pt"
    if best.exists():
        try:
            engine.load_checkpoint(best)
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}; using current model", file=sys.stderr)
    else:
        print("[WARN] best.pt not found, using latest model", file=sys.stderr)
    decoder_cfg = DecoderConfig(**_filter_fields(cfg.get("decoder", {}), DecoderConfig))
    eval_splits = {"val": val_ds, "test": test_ds, "novel": novel_ds}
    eval_results: Dict[str, Any] = {}
    for name, ds in eval_splits.items():
        if ds is None or len(ds) == 0:
            print(f"[INFO] Skipping {name} (empty)", file=sys.stderr)
            continue
        print(f"[INFO] Evaluating on {name} ({len(ds)} samples)", file=sys.stderr)
        loader = DataLoader(ds, batch_size=eval_bs, shuffle=False,
                            num_workers=n_workers, pin_memory=pin_mem, collate_fn=collate_fn)
        eval_results[name] = evaluate_split(model, loader, decoder_cfg, device, args.max_eval_samples)
    with open(args.output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"[INFO] Done. Results saved to {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
