#!/usr/bin/env python3
"""Build the outcome-blind paired RNA-FM sequence-embedding cache for v4."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yaml


SCHEMA = "reactflow_delta.model_rescue_v4_rnafm_cache.v1"
OFFICIAL_REPOSITORY = "https://github.com/ml4bio/RNA-FM"
OFFICIAL_REPOSITORY_COMMIT = "348951516e0963d22bbb33b3c9fc18c89081d38e"
OFFICIAL_CHECKPOINT_SOURCE = (
    "https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth"
)
OUTCOME_BLIND_COLUMNS = ("id", "puzzle", "method", "sequence")
REPRESENTATION_LAYER = 12
REPRESENTATION_WIDTH = 640
ALLOWED_PHYSICAL_GPUS = {6, 7}


@dataclass(frozen=True)
class SequenceEntry:
    row_id: str
    puzzle: str
    method: str
    sequence: str


def load_outcome_blind_sequences(csv_path: Path) -> list[SequenceEntry]:
    """Load only identifier and sequence columns; no mutant outcome enters memory."""
    frame = pd.read_csv(csv_path, usecols=list(OUTCOME_BLIND_COLUMNS))
    entries = []
    for row in frame.itertuples(index=False):
        sequence = str(row.sequence).upper().replace("T", "U")
        if not sequence or not set(sequence) <= set("ACGUN"):
            raise ValueError(f"unsupported RNA sequence in row {row.id}")
        entries.append(
            SequenceEntry(
                row_id=str(row.id),
                puzzle=str(row.puzzle),
                method=str(row.method),
                sequence=sequence,
            )
        )
    if len({entry.row_id for entry in entries}) != len(entries):
        raise ValueError("RNA-FM cache input contains duplicate row ids")
    return entries


def freeze_foundation(model: Any) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def assert_official_source_root(source_root: Path) -> Path:
    source_root = Path(source_root).resolve()
    if not (source_root / "fm" / "__init__.py").is_file():
        raise FileNotFoundError(f"RNA-FM source package is absent: {source_root}")
    head = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != OFFICIAL_REPOSITORY_COMMIT:
        raise RuntimeError(
            f"RNA-FM source HEAD {head} does not match frozen commit "
            f"{OFFICIAL_REPOSITORY_COMMIT}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("RNA-FM source checkout has local modifications")
    return source_root


def load_official_rnafm(
    model_location: Path, source_root: Path
) -> tuple[Any, Callable]:
    model_location = Path(model_location).resolve()
    if not model_location.is_file():
        raise FileNotFoundError(f"official RNA-FM checkpoint is absent: {model_location}")
    source_root = assert_official_source_root(source_root)
    sys.path.insert(0, str(source_root))
    try:
        import fm
    except ImportError as exc:
        raise RuntimeError(
            "official RNA-FM package is absent; install the pinned ml4bio/RNA-FM "
            "revision before cache generation"
        ) from exc
    module_path = Path(fm.__file__).resolve()
    if source_root not in module_path.parents:
        raise RuntimeError(f"RNA-FM imported from non-frozen source: {module_path}")
    model, alphabet = fm.pretrained.rna_fm_t12(str(model_location))
    freeze_foundation(model)
    return model, alphabet.get_batch_converter()


def extract_batch_embeddings(
    model: Any,
    batch_converter: Callable,
    entries: list[SequenceEntry],
    device: str,
) -> list[np.ndarray]:
    import torch

    data = [(entry.row_id, entry.sequence) for entry in entries]
    _labels, _strings, tokens = batch_converter(data)
    tokens = tokens.to(device)
    model = model.to(device)
    with torch.no_grad():
        result = model(tokens, repr_layers=[REPRESENTATION_LAYER])
    representation = result["representations"][REPRESENTATION_LAYER]
    output = []
    for index, entry in enumerate(entries):
        # RNA-FM prepends one special token; exclude all special/padding tokens.
        row = representation[index, 1 : len(entry.sequence) + 1]
        if row.shape != (len(entry.sequence), REPRESENTATION_WIDTH):
            raise RuntimeError(
                f"unexpected RNA-FM representation shape for {entry.row_id}: {tuple(row.shape)}"
            )
        output.append(row.detach().float().cpu().numpy())
    return output


def batches(entries: list[SequenceEntry], batch_size: int) -> Iterable[list[SequenceEntry]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(entries), batch_size):
        yield entries[start : start + batch_size]


def assert_cache_authority(repo_root: Path) -> None:
    active = yaml.safe_load(
        (repo_root / "configs/reactflow_delta/active_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    if active["authority"]["current_phase"] != "V4M1":
        raise RuntimeError("v4 outcome-blind cache is closed outside V4M1")
    if active.get("outcome_blind_cache_allowed") is not True:
        raise RuntimeError("v4 outcome-blind cache authority is absent")
    if active.get("training_allowed") is not False:
        raise RuntimeError("cache generation cannot share an authority that opens training")
    if active.get("new_external_outcome_access_allowed") is not False:
        raise RuntimeError("v4 cache requires external outcomes to remain locked")


def write_cache(
    *,
    entries: list[SequenceEntry],
    embeddings: Iterable[list[np.ndarray]],
    cache_path: Path,
    manifest_path: Path,
    model_location: Path,
    source_root: Path,
    foundation_parameter_count: int,
    foundation_trainable_parameter_count: int,
) -> dict[str, Any]:
    import h5py

    lengths = np.asarray([len(entry.sequence) for entry in entries], dtype=np.int32)
    max_length = int(lengths.max()) if len(lengths) else 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(cache_path, "w") as handle:
        handle.create_dataset(
            "row_ids", data=np.asarray([entry.row_id for entry in entries], dtype=object), dtype=string_dtype
        )
        handle.create_dataset("lengths", data=lengths)
        matrix = handle.create_dataset(
            "embeddings",
            shape=(len(entries), max_length, REPRESENTATION_WIDTH),
            dtype=np.float16,
            chunks=(1, max_length, REPRESENTATION_WIDTH),
            compression="lzf",
        )
        offset = 0
        for batch in embeddings:
            for row in batch:
                length = row.shape[0]
                matrix[offset, :length] = row.astype(np.float16, copy=False)
                offset += 1
        if offset != len(entries):
            raise RuntimeError(
                f"embedding batches produced {offset} rows for {len(entries)} sequence entries"
            )
    manifest = {
        "schema_version": SCHEMA,
        "evidence_status": "OUTCOME_BLIND_FROZEN_FOUNDATION_INPUT_ONLY",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "official_checkpoint_source": OFFICIAL_CHECKPOINT_SOURCE,
        "checkpoint_path_used": str(Path(model_location).resolve()),
        "package_source_root": str(Path(source_root).resolve()),
        "foundation_parameter_count": int(foundation_parameter_count),
        "foundation_trainable_parameter_count": int(
            foundation_trainable_parameter_count
        ),
        "loader": "fm.pretrained.rna_fm_t12",
        "representation_layer": REPRESENTATION_LAYER,
        "representation_width": REPRESENTATION_WIDTH,
        "csv_columns_read": list(OUTCOME_BLIND_COLUMNS),
        "n_sequences": len(entries),
        "max_sequence_length": max_length,
        "cache_path": str(cache_path),
        "external_outcome_accessed": False,
        "mutant_outcome_columns_loaded": False,
        "exact_openknot_pretraining_overlap": "UNKNOWN_NOT_ASSERTED",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m2-csv", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--model-location", type=Path, required=True)
    parser.add_argument("--rnafm-source-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    assert_cache_authority(args.repo_root.resolve())
    if args.physical_gpu not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError("v4 RNA-FM cache may use only physical GPU6 or GPU7")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("authorized v4 GPU is unavailable")
    entries = load_outcome_blind_sequences(args.m2_csv)
    model, converter = load_official_rnafm(
        args.model_location, args.rnafm_source_root
    )
    foundation_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    foundation_trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    embedding_batches = (
        extract_batch_embeddings(model, converter, batch, "cuda:0")
        for batch in batches(entries, args.batch_size)
    )
    write_cache(
        entries=entries,
        embeddings=embedding_batches,
        cache_path=args.cache_path,
        manifest_path=args.manifest_path,
        model_location=args.model_location,
        source_root=args.rnafm_source_root,
        foundation_parameter_count=foundation_parameter_count,
        foundation_trainable_parameter_count=foundation_trainable_parameter_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
