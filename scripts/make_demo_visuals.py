#!/usr/bin/env python3
"""Generate deterministic demo SVG diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reactflow.constraints import dotbracket_to_matrix  # noqa: E402
from reactflow.visualization import write_pair_heatmap_svg, write_profile_overlay_svg  # noqa: E402


def main() -> int:
    """Write demo heatmap/profile SVG files under ``outputs/``."""

    output_dir = ROOT / "outputs"
    matrix = dotbracket_to_matrix("(((...)))")
    write_pair_heatmap_svg(matrix, output_dir / "demo_pair_heatmap.svg", title="Demo RNA pair heatmap")
    write_profile_overlay_svg(
        predicted=(0.1, 0.2, 0.8, 1.0, 0.6, 0.2, 0.1, 0.3, 0.4),
        target=(0.0, 0.3, 0.7, 0.9, 0.5, 0.1, 0.2, 0.2, 0.5),
        path=output_dir / "demo_reactivity_overlay.svg",
        title="Demo reactivity consistency",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
