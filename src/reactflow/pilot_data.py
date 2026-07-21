"""Pilot dataset loader for the C1-2 static PairFormer experiment.

Spec reference: ``ReactFlow分阶段执行提示词.md`` lines 396-401 (pilot protocol).

This module builds PyTorch ``Dataset`` / ``DataLoader`` instances from the
frozen C1-1 benchmark splits, filtered to ``L <= max_length`` for the pilot.

Data flow
---------
1. Load the frozen benchmark manifest (``artifacts/c1_1/frozen_benchmark_manifest.json``)
   to get the record_id -> split assignment.
2. For each split, load the corresponding records from the cache JSONL files
   (under ``data/cache/<source>/<source_version>/<shard>.jsonl``).
3. Filter records to ``min_length <= L <= max_length``.
4. Build a :class:`PairFormerDataset` that yields:
   - ``indices``: LongTensor (L,) of nucleotide vocab indices
   - ``target_matrix``: FloatTensor (L, L) symmetric binary pair matrix
   - ``mask``: BoolTensor (L,) all True (no padding within a sequence)
5. Pad collated batches to the max length in the batch.

Complexity
----------
- Loading: ``O(N)`` over records.
- Per-sample: ``O(L^2)`` to build the target matrix.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# Allow imports from src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reactflow.backbones.embeddings import encode_sequence, PAD_INDEX  # noqa: E402


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PilotRecord:
    """A single record ready for the PairFormer pilot.

    Attributes:
        record_id: unique identifier.
        sequence: uppercase RNA sequence (ACGU + N).
        pairs: tuple of (i, j) with i < j.
        split: split label (train / val / test_mmseqs / novel_clan).
        length: sequence length L.
    """

    record_id: str
    sequence: str
    pairs: Tuple[Tuple[int, int], ...]
    split: str
    length: int


def load_split_assignments(manifest_path: Path) -> Dict[str, str]:
    """Load the frozen benchmark manifest and return ``record_id -> split``.

    Args:
        manifest_path: path to ``frozen_benchmark_manifest.json``.

    Returns:
        Dict mapping record_id to split name.

    Complexity: ``O(N)`` over assignments.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assignments = manifest.get("assignments", [])
    return {a["record_id"]: a["split"] for a in assignments}


def iter_cache_records(
    cache_root: Path,
    *,
    wanted_ids: Optional[set] = None,
) -> Iterator[Dict]:
    """Iterate over records in the cache JSONL files.

    Args:
        cache_root: root directory containing ``<source>/<version>/<shard>.jsonl``.
        wanted_ids: optional set of record_ids to keep (others skipped).

    Yields:
        Raw record dicts from the JSONL files.
    """
    for jsonl_path in sorted(cache_root.rglob("*.jsonl")):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = row.get("record_id")
                if rid is None:
                    continue
                if wanted_ids is not None and rid not in wanted_ids:
                    continue
                yield row


# Mapping from C1-2 split names to the split-file basenames produced by C1-1.
_SPLIT_FILE_MAP = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test_mmseqs": "test.jsonl",
    "novel_clan": "novel.jsonl",
    "novel_family": "novel.jsonl",  # C1-1 v2 merged novel_family into novel_clan
}


def _resolve_split_dir(manifest_path: Path) -> Optional[Path]:
    """Return the split directory referenced by the registry manifest, if any.

    The C1-1 ``global_registry_manifest.json`` stores a ``split_dir`` field
    pointing to the directory containing ``train.jsonl``, ``val.jsonl``,
    ``test.jsonl``, ``novel.jsonl`` and a ``split_manifest.json``.  The
    ``frozen_benchmark_manifest.json`` does not have an ``assignments`` field;
    instead, records are stored directly in these per-split JSONL files.
    """
    registry_path = manifest_path.parent / "global_registry_manifest.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        split_dir = reg.get("split_dir")
        if split_dir:
            p = Path(split_dir)
            if not p.is_absolute():
                p = (manifest_path.parent.parent.parent / p).resolve()
            if p.exists():
                return p
    return None


def _iter_split_file(jsonl_path: Path) -> Iterator[Dict]:
    """Iterate records from a single per-split JSONL file."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_pilot_records(
    manifest_path: Path,
    cache_root: Path,
    *,
    splits: Sequence[str] = ("train", "val", "test_mmseqs", "novel_clan"),
    max_length: int = 128,
    min_length: int = 4,
    max_per_split: Optional[int] = None,
) -> List[PilotRecord]:
    """Load pilot records from the frozen C1-1 splits, filtered by length.

    Two loading modes are supported:

    1. **Split-file mode** (default for the C1-1 v2 manifest): the
       ``frozen_benchmark_manifest.json`` does not contain an ``assignments``
       field; instead, records live in per-split JSONL files under the
       ``split_dir`` referenced by ``global_registry_manifest.json``.
       Each split file already contains ``sequence`` and ``pairs``.
    2. **Cache mode** (legacy): the manifest has an ``assignments`` list of
       ``{record_id, split}`` dicts, and records are loaded from
       ``cache_root/<source>/<version>/<shard>.jsonl``.

    Args:
        manifest_path: path to ``frozen_benchmark_manifest.json``.
        cache_root: root directory containing cache JSONL files (cache mode).
        splits: which splits to load.
        max_length: keep records with ``L <= max_length``.
        min_length: keep records with ``L >= min_length``.
        max_per_split: optional cap on records per split (for very fast pilot).

    Returns:
        List of :class:`PilotRecord` instances.

    Complexity: ``O(N)`` over all records in the selected splits.
    """
    # Detect mode: try split-file first, fall back to cache mode.
    split_dir = _resolve_split_dir(manifest_path)
    use_split_files = split_dir is not None and (
        manifest_path.parent / "global_registry_manifest.json"
    ).exists()

    # In cache mode, verify assignments exist.
    if not use_split_files:
        split_map = load_split_assignments(manifest_path)
        if not split_map:
            raise RuntimeError(
                f"manifest {manifest_path} has no 'assignments' field and no "
                f"global_registry_manifest.json with a 'split_dir' was found; "
                f"cannot resolve split assignments."
            )

    wanted_splits = set(splits)
    records: List[PilotRecord] = []
    per_split_counts: Dict[str, int] = {sp: 0 for sp in splits}

    def _record_from_row(row: Dict, split: str, fallback_idx: int) -> Optional[PilotRecord]:
        seq = str(row.get("sequence", "")).upper()
        # Normalize T -> U and strip gaps
        seq = seq.replace("T", "U").replace("-", "")
        L = len(seq)
        if L < min_length or L > max_length:
            return None
        rid = row.get("record_id") or row.get("source_id") or f"{split}_{fallback_idx}"
        raw_pairs = row.get("pairs") or row.get("structure") or []
        pairs = []
        for p in raw_pairs:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                i, j = int(p[0]), int(p[1])
                if i == j:
                    continue
                if i < 0 or j < 0 or i >= L or j >= L:
                    continue
                if i > j:
                    i, j = j, i
                pairs.append((i, j))
        return PilotRecord(
            record_id=str(rid),
            sequence=seq,
            pairs=tuple(pairs),
            split=split,
            length=L,
        )

    if use_split_files:
        # Split-file mode: read each split file directly.
        for split in splits:
            fname = _SPLIT_FILE_MAP.get(split)
            if fname is None:
                print(f"WARNING: no split file mapping for {split!r}", file=sys.stderr)
                continue
            jsonl_path = split_dir / fname
            if not jsonl_path.exists():
                print(f"WARNING: split file {jsonl_path} does not exist", file=sys.stderr)
                continue
            for idx, row in enumerate(_iter_split_file(jsonl_path)):
                if max_per_split is not None and per_split_counts[split] >= max_per_split:
                    break
                rec = _record_from_row(row, split, idx)
                if rec is None:
                    continue
                records.append(rec)
                per_split_counts[split] += 1
    else:
        # Cache mode: filter cache records by the split assignment.
        wanted_ids = {rid for rid, sp in split_map.items() if sp in wanted_splits}
        if not wanted_ids:
            raise RuntimeError(
                f"no records found for splits {splits} in manifest {manifest_path}"
            )
        for row in iter_cache_records(cache_root, wanted_ids=wanted_ids):
            rid = row.get("record_id")
            if rid is None:
                continue
            split = split_map.get(rid)
            if split is None or split not in wanted_splits:
                continue
            if max_per_split is not None and per_split_counts[split] >= max_per_split:
                continue
            rec = _record_from_row(row, split, 0)
            if rec is None:
                continue
            records.append(rec)
            per_split_counts[split] += 1

    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _pairs_to_matrix(pairs: Sequence[Tuple[int, int]], length: int) -> torch.Tensor:
    """Build a symmetric binary pair matrix from a list of pairs."""
    mat = torch.zeros(length, length, dtype=torch.float32)
    for i, j in pairs:
        if 0 <= i < length and 0 <= j < length and i != j:
            mat[i, j] = 1.0
            mat[j, i] = 1.0
    return mat


class PairFormerDataset(Dataset):
    """PyTorch Dataset for the static PairFormer pilot.

    Each sample is a (indices, target_matrix, mask) tuple where:
    - ``indices``: LongTensor (L,)
    - ``target_matrix``: FloatTensor (L, L) symmetric binary
    - ``mask``: BoolTensor (L,) all True

    Args:
        records: list of :class:`PilotRecord`.
    """

    def __init__(self, records: Sequence[PilotRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        indices = encode_sequence(rec.sequence)
        target = _pairs_to_matrix(rec.pairs, rec.length)
        mask = torch.ones(rec.length, dtype=torch.bool)
        return indices, target, mask


def collate_padded(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length samples into a padded batch.

    Pads ``indices`` with ``PAD_INDEX`` (5), ``target_matrix`` with 0, and
    ``mask`` with False.

    Returns:
        (indices, targets, masks) where:
        - indices: LongTensor (B, L_max)
        - targets: FloatTensor (B, L_max, L_max)
        - masks: BoolTensor (B, L_max)
    """
    B = len(batch)
    L_max = max(len(item[0]) for item in batch)
    indices = torch.full((B, L_max), PAD_INDEX, dtype=torch.long)
    targets = torch.zeros(B, L_max, L_max, dtype=torch.float32)
    masks = torch.zeros(B, L_max, dtype=torch.bool)
    for b, (idx, tgt, msk) in enumerate(batch):
        L = len(idx)
        indices[b, :L] = idx
        targets[b, :L, :L] = tgt
        masks[b, :L] = msk
    return indices, targets, masks


def build_pilot_dataloaders(
    manifest_path: Path,
    cache_root: Path,
    *,
    max_length: int = 128,
    min_length: int = 4,
    batch_size: int = 32,
    eval_batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_per_split: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, DataLoader]:
    """Build train/val/test/novel DataLoaders for the pilot.

    Args:
        manifest_path: path to frozen benchmark manifest.
        cache_root: root of cache JSONL files.
        max_length: pilot length filter.
        min_length: minimum sequence length.
        batch_size: training batch size.
        eval_batch_size: evaluation batch size.
        num_workers: DataLoader workers.
        pin_memory: DataLoader pin_memory.
        max_per_split: cap records per split.
        seed: for shuffling.

    Returns:
        Dict with keys ``train``, ``val``, ``test``, ``novel`` mapping to DataLoader.
    """
    splits = ("train", "val", "test_mmseqs", "novel_clan")
    records = load_pilot_records(
        manifest_path, cache_root,
        splits=splits, max_length=max_length, min_length=min_length,
        max_per_split=max_per_split,
    )

    by_split: Dict[str, List[PilotRecord]] = {sp: [] for sp in splits}
    for rec in records:
        by_split[rec.split].append(rec)

    # Map split names to dataloader keys
    split_key_map = {
        "train": "train",
        "val": "val",
        "test_mmseqs": "test",
        "novel_clan": "novel",
    }

    loaders: Dict[str, DataLoader] = {}
    for split, key in split_key_map.items():
        split_records = by_split[split]
        if not split_records:
            print(f"WARNING: no records for split {split!r}", file=sys.stderr)
            continue
        ds = PairFormerDataset(split_records)
        is_train = split == "train"
        loaders[key] = DataLoader(
            ds,
            batch_size=batch_size if is_train else eval_batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_padded,
            drop_last=is_train,
            generator=torch.Generator().manual_seed(seed) if is_train else None,
        )
        print(f"  {key}: {len(ds)} samples", file=sys.stderr)

    return loaders
