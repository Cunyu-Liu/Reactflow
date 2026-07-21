#!/usr/bin/env python3
"""Export missing frozen-feature shards from a shared on-disk work queue."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Optional, Sequence

import export_frozen_features as base
import export_frozen_shard_range as range_export


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export missing ReactFlow frozen shards from a shared queue")
    parser.add_argument("--sequences", required=True, type=Path, help="JSONL of {sequence,id,family}")
    parser.add_argument("--out", required=True, type=Path, help="parent sharded frozen directory")
    parser.add_argument("--backend", choices=("dry-run", "torch"), default="dry-run")
    parser.add_argument("--model", default="RibonanzaNet2", help="model name for provenance")
    parser.add_argument("--model-version", default="alpha-v1", help="model version tag")
    parser.add_argument("--limit", type=int, default=None, help="cap number of sequences")
    parser.add_argument("--d-single", type=int, default=384, help="per-nucleotide dim (dry-run)")
    parser.add_argument("--d-pair", type=int, default=128, help="pairwise dim (dry-run; 0 to omit)")
    parser.add_argument("--n-probe", type=int, default=2, help="reactivity probes (dry-run; 0 to omit)")
    parser.add_argument("--seed", type=int, default=0, help="dry-run master seed")
    parser.add_argument("--compress", action="store_true", help="DEFLATE the NPZ archive")
    parser.add_argument("--shard-size", required=True, type=int, help="records per shard")
    parser.add_argument("--shard-start", type=int, default=0, help="inclusive shard index")
    parser.add_argument("--shard-end", type=int, default=None, help="exclusive shard index")
    parser.add_argument("--claim-dir", type=Path, default=None, help="directory for atomic shard claims")
    parser.add_argument("--worker-id", default=None, help="identifier written into claim files")
    parser.add_argument("--stale-claim-seconds", type=int, default=3600, help="reclaim dead/stale claims after this many seconds")
    parser.add_argument("--batch-size", type=int, default=1, help="torch same-length mini-batch size")
    parser.add_argument("--network-dir", type=Path, default=None, help="dir with Network.py + weights")
    parser.add_argument("--config", type=Path, default=None, help="model config YAML")
    parser.add_argument("--weights", type=Path, default=None, help="model weights .pt")
    parser.add_argument("--device", default="cpu", help="torch device")
    return parser


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _claim_stale(path: Path, *, now: float, stale_seconds: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", -1))
        claimed_at = float(payload.get("claimed_at", 0.0))
    except Exception:
        return path.stat().st_mtime + stale_seconds < now
    if pid > 0 and _pid_alive(pid):
        return False
    return claimed_at + stale_seconds < now


def _try_claim(path: Path, payload: dict, *, stale_seconds: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if stale_seconds > 0 and _claim_stale(path, now=time.time(), stale_seconds=stale_seconds):
                try:
                    path.unlink()
                    continue
                except FileNotFoundError:
                    continue
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        return True


def _claim_next_shard(
    *,
    sequences: Sequence[tuple[str, str, Optional[str]]],
    out_dir: Path,
    provenance: base.FrozenFeatureProvenance,
    shard_size: int,
    shard_start: int,
    shard_end: int,
    claim_dir: Path,
    worker_id: str,
    stale_claim_seconds: int,
) -> Optional[tuple[int, Sequence[tuple[str, str, Optional[str]]]]]:
    for shard_index in range(shard_start, shard_end):
        start = shard_index * shard_size
        chunk = sequences[start : start + shard_size]
        if not chunk:
            continue
        shard_dir = out_dir / f"shard_{shard_index:05d}"
        existing = base._existing_shard_summary(  # noqa: SLF001 - recovery companion script.
            shard_dir,
            expected_record_count=len(chunk),
            expected_provenance=provenance,
        )
        if existing is not None:
            continue
        claim = claim_dir / f"shard_{shard_index:05d}.claim"
        payload = {
            "claimed_at": time.time(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "shard": shard_index,
            "worker_id": worker_id,
        }
        if _try_claim(claim, payload, stale_seconds=stale_claim_seconds):
            return shard_index, chunk
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be positive")
    if args.shard_start < 0:
        raise SystemExit("--shard-start must be non-negative")

    sequences = base.read_sequences(args.sequences, limit=args.limit)
    shard_end = args.shard_end
    if shard_end is None:
        shard_end = (len(sequences) + args.shard_size - 1) // args.shard_size
    if shard_end <= args.shard_start:
        raise SystemExit("--shard-end must be greater than --shard-start")

    backend, provenance = range_export._build_backend_and_provenance(args)  # noqa: SLF001
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    claim_dir = args.claim_dir or (out_dir / ".shard_claims")
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}"

    written = 0
    records_written = 0
    while True:
        claimed = _claim_next_shard(
            sequences=sequences,
            out_dir=out_dir,
            provenance=provenance,
            shard_size=args.shard_size,
            shard_start=args.shard_start,
            shard_end=shard_end,
            claim_dir=claim_dir,
            worker_id=worker_id,
            stale_claim_seconds=args.stale_claim_seconds,
        )
        if claimed is None:
            break
        shard_index, chunk = claimed
        shard_dir = out_dir / f"shard_{shard_index:05d}"
        print(
            f"pool: writing {shard_dir.name} records={len(chunk)} batch_size={args.batch_size}",
            flush=True,
        )
        encoded_arrays = base._encode_chunk(backend, chunk, batch_size=args.batch_size)  # noqa: SLF001
        records = [
            base.FrozenFeatureRecord(record_id=record_id, sequence=sequence, arrays=arrays, family=family)
            for (record_id, sequence, family), arrays in zip(chunk, encoded_arrays)
        ]
        finalized = base.write_frozen_shard(shard_dir, records, provenance, compress=args.compress)
        print(f"pool: wrote {shard_dir.name} records={finalized.record_count}", flush=True)
        written += 1
        records_written += finalized.record_count

    summary = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "claim_dir": str(claim_dir),
        "out": str(out_dir),
        "records_written": records_written,
        "shard_end": shard_end,
        "shard_size": args.shard_size,
        "shard_start": args.shard_start,
        "weights_sha256": provenance.weights_sha256,
        "worker_id": worker_id,
        "written_shards": written,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
