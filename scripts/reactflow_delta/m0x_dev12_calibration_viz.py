#!/usr/bin/env python3
"""dev12 calibration visualization: raw vs z-score-calibrated magnitude.

Plots, for a set of TYPICAL pairs (low / mid / high true burden), the per-position
predicted magnitude under RAW (|delta_r_hat|) and CALIBRATED (within-pair z-score,
the proxy that fixed the sign).  The aim is to confirm the calibration is NOT
overdone: i.e. the relative ordering of positions within a pair is preserved
(the same positions stay "hot") after standardization, and the z_max reflects a
genuinely elevated position rather than a degenerate 2-point artifact.

Outputs a multi-panel matplotlib figure to --out-dir.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
if (_HERE.parents[2] / "src").exists():
    sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(Path.cwd() / "src"))

from b0x_baselines import _pair_scale  # noqa: E402
from b0x_data import load_pairs, split_groups  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-jsonl", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--predictions-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=6)
    ap.add_argument("--method", default="Agg")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use(args.method)
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa
        print(f"matplotlib unavailable: {e}", file=sys.stderr)
        return 2

    pairs = load_pairs(args.canonical_jsonl, args.split_manifest,
                       splits={"validation"})
    groups = split_groups(pairs)
    val = groups.get("validation", [])

    preds_all = np.load(args.predictions_npz, allow_pickle=True)

    # Gather per-pair data
    records = []  # {pair_id, study, burden, mag, tr}
    for p in val:
        if p.pair_id not in preds_all.files:
            continue
        sc = np.asarray(preds_all[p.pair_id], dtype=np.float32)
        mask = np.asarray(p.mask, dtype=bool)
        d = np.asarray(p.delta, dtype=np.float64)
        scale = _pair_scale(p)
        elig = mask & np.isfinite(d) & (np.arange(len(mask)) < len(sc))
        if elig.sum() == 0:
            continue
        mag = np.abs(sc[elig])
        tr = np.abs(d[elig]) / scale
        records.append({"pair_id": p.pair_id, "study": p.study,
                        "burden": float(tr.mean()),
                        "mag": mag, "tr": tr})

    records.sort(key=lambda r: r["burden"])

    # Select typical pairs: low/mid/high burden quantiles
    n = len(records)
    q_idx = [int(n * q) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    sel_idx = sorted(set(q_idx + [0, n - 1]))[: args.n_pairs]
    sel = [records[i] for i in sel_idx if 0 <= i < n]

    rows = (len(sel) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    panels = {}
    for ax, rec in zip(axes, sel):
        mag = rec["mag"]
        mu, sd = mag.mean(), mag.std()
        z = (mag - mu) / sd if sd > 1e-9 else np.zeros_like(mag)
        order = np.argsort(-mag)
        x = np.arange(len(mag))
        ax.bar(x, mag, alpha=0.5, color="tab:blue", label="raw |pred|")
        ax.scatter(x, z, color="tab:red", s=18, zorder=3,
                   label="z-score (calibrated)")
        # mark max-z position
        zmax_i = int(np.argmax(z))
        ax.scatter([zmax_i], [z[zmax_i]], color="black", s=40, marker="*",
                   zorder=4, label=f"z_max @pos{zmax_i}")
        ax.set_title(f"{rec['pair_id']} ({rec['study']})\n"
                     f"true burden={rec['burden']:.3f}  z_max={z.max():.2f}")
        ax.set_xlabel("eligible position (sorted by |pred|)")
        ax.legend(fontsize=7)
        panels[rec["pair_id"]] = {
            "study": rec["study"], "burden": rec["burden"],
            "z_max": float(z.max()),
            "n_pos": int(len(mag)),
            "argmax_raw": int(np.argmax(mag)),
            "argmax_z": int(np.argmax(z)),
            "raw_top5_pos": [int(i) for i in np.argsort(-mag)[:5]],
            "z_top5_pos": [int(i) for i in np.argsort(-z)[:5]],
        }

    # hide any unused axes
    for ax in axes[len(sel):]:
        ax.set_visible(False)

    fig.suptitle("dev12 calibration: raw |delta_r_hat| vs within-pair z-score "
                 "(is z_max a genuine hot spot?)", fontsize=13)
    fig.tight_layout()
    fig_path = out_dir / "calibration_distribution.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"[done] figure -> {fig_path}", flush=True)

    (out_dir / "calibration_panels.json").write_text(
        json.dumps({"n_pairs": len(sel), "panels": panels}, indent=2),
        encoding="utf-8")

    print(f"\n{'pair':28s} {'burden':>7s} {'z_max':>6s} {'n_pos':>5s} "
          f"{'argmax_raw/z':>12s}  top5 raw==top5 z?")
    for pid, pnl in panels.items():
        same = set(pnl["raw_top5_pos"]) == set(pnl["z_top5_pos"])
        print(f"{pid:28s} {pnl['burden']:7.3f} {pnl['z_max']:6.2f} "
              f"{pnl['n_pos']:5d} {pnl['argmax_raw']:>5d}/{pnl['argmax_z']:<5d} "
              f"{'YES' if same else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())