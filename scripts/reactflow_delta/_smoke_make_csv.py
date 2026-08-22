#!/usr/bin/env python3
"""Generate a small synthetic M2 CSV for engineering smoke of run_p2_v3.
Mirrors the official OpenKnot M2 row layout used by M2Universe (not scientific data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_csv(path: Path, n_puzzles: int = 4, methods=("Eterna", "Rosetta"),
             seq_len: int = 24, sub: tuple[int, int] = (8, 16)) -> None:
    rng = np.random.RandomState(7)
    bases = "ACGU"
    seq = "".join(bases[i] for i in rng.randint(0, 4, size=seq_len))
    rows = []
    full_start, full_end = sub
    for p in range(1, n_puzzles + 1):
        puzzle = f"P{p:02d}"
        for method in methods:
            rec = {"id": f"{puzzle}_{method}_wt", "sequence": seq,
                   "experiment_type": "2A3_MaP", "dataset_name": "synth",
                   "puzzle": puzzle, "method": method,
                   "sub_start": full_start + 1, "sub_end": full_end,
                   "design_length": full_end - full_start,
                   "design_sequence": seq[full_start:full_end],
                   "target_structure": "", "mutA": 0,
                   "M2_structure": "AAAA"}
            for i in range(1, seq_len + 1):
                rec[f"reactivity_{i:04d}"] = float((i % 5)) / 4 + 0.1
                rec[f"reactivity_error_{i:04d}"] = 0.1
            rows.append(rec)
            for full_pos in range(full_start, full_end):
                design_pos = full_pos - full_start
                ref = seq[full_pos]
                for alt in [b for b in bases if b != ref]:
                    m = dict(rec)
                    m["id"] = (
                        f"{puzzle}_{method}_mm_{design_pos}_{ref}_{alt}"
                    )
                    m["sequence"] = (
                        seq[:full_pos] + alt + seq[full_pos + 1:]
                    )
                    m["mutA"] = design_pos + 1
                    # small correlated mutation response signal
                    delta = rng.normal(0.0, 0.05, size=seq_len)
                    for i in range(1, seq_len + 1):
                        d = float((i - 1) - full_pos)
                        m[f"reactivity_{i:04d}"] = float((i % 5)) / 4 + 0.1 \
                            + 0.2 * np.exp(-0.5 * (d / 3.0) ** 2) + delta[i - 1]
                        m[f"reactivity_error_{i:04d}"] = 0.1
                    rows.append(m)
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/p2v3_smoke_m2.csv")
    n_p = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    make_csv(out, n_puzzles=n_p)
    print(f"wrote {out}")
