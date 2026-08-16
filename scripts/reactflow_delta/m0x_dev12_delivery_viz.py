#!/usr/bin/env python3
"""M0-X EPRO_DEV_12 delivery summary chart.

Produces a two-panel PNG summarising the continuous-burden acceptance metrics:
  Panel A - model ranking on the unified within_pair_z_max burden proxy
            (7 models), with the raw mean-burden Spearman overlaid to show how
            the proxy re-ranks / flips each model.
  Panel B - dev12 raw mean-burden vs within_pair_z_max (the sign-recovery that
            is the headline of this iteration).

Evidence class: DEVELOPMENT_ONLY. Read-only; loads unified_zmax_compare.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Display / colour map
CAT_COLORS = {
    "supervised": "#e07b39",   # orange - our model
    "zero_shot": "#7a8ba8",    # slate - folding baselines
}
BEST_ZERO_COLOR = "#2f6f9f"    # blue - best zero-shot (efold)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.unified).read_text())
    models = data["models"]
    n_val = data.get("n_val_pairs", 548)

    names = list(models.keys())
    z_spear = {m: models[m]["within_pair_z_max"]["spearman"] for m in names}
    z_ndcg = {m: models[m]["within_pair_z_max"]["ndcg_at_10"] for m in names}
    raw_spear = {m: models[m]["raw_mean_burden"]["spearman"] for m in names}
    raw_ndcg = {m: models[m]["raw_mean_burden"]["ndcg_at_10"] for m in names}
    cat = {m: models[m]["category"] for m in names}

    # sort by within_pair_z_max Spearman descending
    order = sorted(names, key=lambda m: z_spear[m], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ---------------- Panel A: model ranking ----------------
    ax = axes[0]
    x = np.arange(len(order))
    zvals = [z_spear[m] for m in order]
    rawvals = [raw_spear[m] for m in order]
    colors = []
    for m in order:
        if cat[m] == "supervised":
            colors.append(CAT_COLORS["supervised"])
        elif m == "efold":
            colors.append(BEST_ZERO_COLOR)
        else:
            colors.append(CAT_COLORS["zero_shot"])

    bars = ax.bar(x, zvals, width=0.62, color=colors, alpha=0.92,
                  label="within_pair_z_max")
    # raw mean-burden overlaid as open markers (contrast to proxy re-ranking)
    ax.plot(x, rawvals, "o", color="#444", mfc="none", mew=1.4, ms=7,
            label="raw mean-burden")

    ax.axhline(0.0, color="#999", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Spearman vs true burden  mean(|Δ|/scale)")
    ax.set_title("M0-X EPRO_DEV_12: within_pair_z_max burden ranking (n=548 val)")
    for i, m in enumerate(order):
        ax.annotate(f"{z_spear[m]:+.3f}", (i, z_spear[m]),
                    xytext=(0, 4 if z_spear[m] >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=colors[i])
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_ylim(-0.35, 0.65)

    # ---------------- Panel B: dev12 sign recovery ----------------
    ax = axes[1]
    m12 = "epro_dev12"
    labels = ["raw mean-burden", "within_pair_z_max"]
    vals = [raw_spear[m12], z_spear[m12]]
    ndcgs = [raw_ndcg[m12], z_ndcg[m12]]
    bar_colors = ["#b04a4a", CAT_COLORS["supervised"]]
    b = ax.bar(labels, vals, width=0.5, color=bar_colors, alpha=0.92)
    ax.axhline(0.0, color="#999", lw=0.8, ls="--")
    for rect, v, nd in zip(b, vals, ndcgs):
        ax.annotate(f"Spearman {v:+.3f}\nNDCG@10 {nd:.3f}",
                    (rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 8 if v >= 0 else -16),
                    textcoords="offset points", ha="center", fontsize=9,
                    color="#1a1a1a")
    ax.set_ylabel("Spearman vs true burden")
    ax.set_ylim(-0.45, 0.6)
    ax.set_title("dev12: continuous-burden correlation recovery\n"
                 "raw → within_pair_z_max (sign fixed, test SEALED)")
    ax.annotate("cross-pair scale drift\nremoved by z-score", xy=(0.35, 0.0),
                xytext=(0.02, -0.28), fontsize=8, color="#666",
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))

    fig.suptitle("M0-X EPRO_DEV_12 acceptance summary  (evidence_class=DEVELOPMENT_ONLY, "
                 "no confirmatory claim)", fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[done] -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
