#!/usr/bin/env python3
"""Export a disjoint range of frozen-feature shards.

This companion to ``export_frozen_features.py`` is intended for recovery work:
multiple processes can safely cover non-overlapping shard index ranges on
different GPUs without racing on the parent ``sharded_manifest.json``.  A final
manifest rebuild should be run after all ranges are complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import export_frozen_features as base


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a range of ReactFlow frozen shards")
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
    parser.add_argument("--shard-start", required=True, type=int, help="inclusive shard index")
    parser.add_argument("--shard-end", required=True, type=int, help="exclusive shard index")
    parser.add_argument("--resume", action="store_true", help="skip complete matching shards")
    parser.add_argument("--batch-size", type=int, default=1, help="torch same-length mini-batch size")
    parser.add_argument("--network-dir", type=Path, default=None, help="dir with Network.py + weights")
    parser.add_argument("--config", type=Path, default=None, help="model config YAML")
    parser.add_argument("--weights", type=Path, default=None, help="model weights .pt")
    parser.add_argument("--device", default="cpu", help="torch device")
    return parser


def _build_backend_and_provenance(args: argparse.Namespace) -> tuple[object, base.FrozenFeatureProvenance]:
    d_pair = None if args.d_pair in (0, None) else args.d_pair
    n_probe = None if args.n_probe in (0, None) else args.n_probe
    if args.backend == "dry-run":
        backend: object = base.DryRunBackend(
            d_single=args.d_single,
            d_pair=d_pair,
            n_probe=n_probe,
            seed=args.seed,
        )
        provenance = base.build_provenance(
            model_name=args.model,
            model_version=args.model_version,
            weights_sha256="",
            d_single=args.d_single,
            d_pair=d_pair,
            n_probe=n_probe,
            notes="DRY RUN: deterministic random features, NOT real model weights",
        )
    else:
        missing = [
            name
            for name, value in (
                ("--network-dir", args.network_dir),
                ("--config", args.config),
                ("--weights", args.weights),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(f"--backend torch requires {', '.join(missing)}")
        backend = base.TorchRibonanzaBackend(
            network_dir=args.network_dir,
            config_path=args.config,
            weights_path=args.weights,
            device=args.device,
            include_pair=d_pair is not None,
            include_react=n_probe is not None,
        )
        d_single_real = getattr(backend, "d_single", args.d_single)
        provenance = base.build_provenance(
            model_name=args.model,
            model_version=args.model_version,
            weights_sha256=backend.weights_sha256,  # type: ignore[attr-defined]
            d_single=d_single_real,
            d_pair=d_pair,
            n_probe=n_probe,
            notes=f"real weights sha256={backend.weights_sha256[:12]}...",  # type: ignore[attr-defined]
        )
    return backend, provenance


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be positive")
    if args.shard_start < 0 or args.shard_end <= args.shard_start:
        raise SystemExit("--shard-start/--shard-end must define a non-empty forward range")

    sequences = base.read_sequences(args.sequences, limit=args.limit)
    backend, provenance = _build_backend_and_provenance(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    empty = 0
    records_written = 0
    for shard_index in range(args.shard_start, args.shard_end):
        start = shard_index * args.shard_size
        chunk = sequences[start : start + args.shard_size]
        if not chunk:
            empty += 1
            continue
        shard_dir = out_dir / f"shard_{shard_index:05d}"
        if args.resume:
            existing = base._existing_shard_summary(  # noqa: SLF001 - recovery companion script.
                shard_dir,
                expected_record_count=len(chunk),
                expected_provenance=provenance,
            )
            if existing is not None:
                print(f"range: skipping complete {shard_dir.name}", flush=True)
                skipped += 1
                continue
        print(
            f"range: writing {shard_dir.name} records={len(chunk)} batch_size={args.batch_size}",
            flush=True,
        )
        encoded_arrays = base._encode_chunk(backend, chunk, batch_size=args.batch_size)  # noqa: SLF001
        records = [
            base.FrozenFeatureRecord(record_id=record_id, sequence=sequence, arrays=arrays, family=family)
            for (record_id, sequence, family), arrays in zip(chunk, encoded_arrays)
        ]
        finalized = base.write_frozen_shard(shard_dir, records, provenance, compress=args.compress)
        print(f"range: wrote {shard_dir.name} records={finalized.record_count}", flush=True)
        written += 1
        records_written += finalized.record_count

    summary = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "empty_shards": empty,
        "out": str(out_dir),
        "records_written": records_written,
        "shard_end": args.shard_end,
        "shard_size": args.shard_size,
        "shard_start": args.shard_start,
        "skipped_shards": skipped,
        "weights_sha256": provenance.weights_sha256,
        "written_shards": written,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
