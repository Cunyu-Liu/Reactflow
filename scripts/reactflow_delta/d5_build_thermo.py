#!/usr/bin/env python3
"""Build ViennaRNA thermodynamic features for D5 GEO UTR sequences.

Computes MFE energy, partition function ensemble energy, per-position unpaired
probability, and per-position positional entropy for each unique UTR sequence
in D5 GEO datasets (GSE114002, GSE145046).

Output .npz arrays:
  - mfe_energy: (n_utrs,) float32 — MFE free energy (kcal/mol)
  - pf_energy: (n_utrs,) float32 — partition function ensemble free energy (kcal/mol)
  - unpaired_prob: (n_utrs, L) float32 — per-position unpaired probability
  - positional_entropy: (n_utrs, L) float32 — per-position entropy (bits)
  - utr: (n_utrs,) <U50 — UTR sequences (for record mapping)

Usage:
    python scripts/reactflow_delta/d5_build_thermo.py \\
        --input artifacts/reactflow_delta/d5/d5_gse114002_records.jsonl \\
        --output artifacts/reactflow_delta/d5/d5_gse114002_thermo.npz \\
        [--temperature 37.0] [--max-sequences N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def load_unique_utrs(jsonl_path: Path) -> tuple[list[str], int]:
    """Load unique UTR sequences from a D5 JSONL file.

    Auto-detects the sequence field name: ``utr`` (GSE114002) or ``seq``
    (GSE145046).

    Returns (unique_utrs, total_records).
    """

    utr_seen: set[str] = set()
    unique_utrs: list[str] = []
    total = 0
    seq_field: str | None = None
    with jsonl_path.open() as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            if seq_field is None:
                seq_field = "utr" if "utr" in rec else "seq"
                if seq_field not in rec:
                    raise KeyError(
                        f"record has neither 'utr' nor 'seq' field; keys={list(rec.keys())}"
                    )
            utr = rec[seq_field]
            if utr not in utr_seen:
                utr_seen.add(utr)
                unique_utrs.append(utr)
    return unique_utrs, total


def compute_thermo_features(
    sequences: list[str],
    *,
    temperature: float = 37.0,
    progress_every: int = 10000,
) -> dict[str, np.ndarray]:
    """Compute ViennaRNA thermo features for a list of RNA sequences.

    Returns dict with mfe_energy, pf_energy, unpaired_prob, positional_entropy.
    """

    import RNA  # noqa: E402

    n = len(sequences)
    if n == 0:
        raise ValueError("no sequences to process")

    L = len(sequences[0])
    # Verify uniform length
    for i, s in enumerate(sequences):
        if len(s) != L:
            raise ValueError(
                f"sequence {i} has length {len(s)}, expected {L} (uniform length required)"
            )

    mfe_energy = np.zeros(n, dtype=np.float32)
    pf_energy = np.zeros(n, dtype=np.float32)
    unpaired_prob = np.zeros((n, L), dtype=np.float32)
    positional_entropy = np.zeros((n, L), dtype=np.float32)

    md = RNA.md()
    md.temperature = temperature

    t0 = time.time()
    for i, seq in enumerate(sequences):
        seq_rna = seq.upper().replace("T", "U")
        fc = RNA.fold_compound(seq_rna, md)

        # MFE: (structure_string, energy_float)
        _, mfe_e = fc.mfe()
        mfe_energy[i] = mfe_e

        # Partition function: (structure_string, pf_energy_float)
        _, pf_e = fc.pf()
        pf_energy[i] = pf_e

        # BPP: (n+1)×(n+1), 1-indexed → extract n×n 0-indexed
        bpp_raw = fc.bpp()
        bpp = np.array(bpp_raw, dtype=np.float64)[1 : L + 1, 1 : L + 1]
        np.fill_diagonal(bpp, 0.0)  # self-pairing = 0

        # Paired probability per position: sum over partners
        paired = bpp.sum(axis=1)
        up = np.clip(1.0 - paired, 0.0, 1.0)
        unpaired_prob[i] = up.astype(np.float32)

        # Positional entropy (bits): S(i) = -sum_j p*log2(p) - p_unpaired*log2(p_unpaired)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_bpp = np.where(bpp > 1e-15, np.log2(bpp), 0.0)
            log_up = np.where(up > 1e-15, np.log2(up), 0.0)
        entropy = -(bpp * log_bpp).sum(axis=1) - (up * log_up)
        positional_entropy[i] = entropy.astype(np.float32)

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(
                f"  {i + 1}/{n} ({rate:.0f} seq/s, ETA {eta / 60:.1f} min)",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"  Done: {n} sequences in {elapsed / 60:.1f} min ({n / elapsed:.0f} seq/s)",
        flush=True,
    )

    return {
        "mfe_energy": mfe_energy,
        "pf_energy": pf_energy,
        "unpaired_prob": unpaired_prob,
        "positional_entropy": positional_entropy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="D5 JSONL records file"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output .npz file"
    )
    parser.add_argument(
        "--temperature", type=float, default=37.0, help="ViennaRNA temperature (°C)"
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Limit number of unique UTRs (0 = all, for testing)",
    )
    args = parser.parse_args()

    print(f"[d5_build_thermo] input={args.input}", flush=True)
    print(f"[d5_build_thermo] output={args.output}", flush=True)
    print(f"[d5_build_thermo] temperature={args.temperature}°C", flush=True)

    # Load unique UTRs
    print("Loading unique UTRs...", flush=True)
    t0 = time.time()
    unique_utrs, total_records = load_unique_utrs(args.input)
    n_unique = len(unique_utrs)
    print(
        f"  {n_unique} unique UTRs (from {total_records} total records) "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )

    if args.max_sequences > 0 and args.max_sequences < n_unique:
        unique_utrs = unique_utrs[: args.max_sequences]
        print(f"  limited to {args.max_sequences} sequences", flush=True)

    # Compute thermo features
    print("Computing thermo features...", flush=True)
    features = compute_thermo_features(
        unique_utrs, temperature=args.temperature
    )

    # Add UTR sequences for mapping back to records
    L = len(unique_utrs[0]) if unique_utrs else 1
    features["utr"] = np.array(unique_utrs, dtype=f"<U{L}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **features)

    file_size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {args.output} ({file_size_mb:.1f} MB)", flush=True)
    print(f"  arrays: {list(features.keys())}", flush=True)
    for k, v in features.items():
        print(f"    {k}: shape={v.shape} dtype={v.dtype}", flush=True)


if __name__ == "__main__":
    main()
