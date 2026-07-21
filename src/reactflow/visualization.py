"""Lightweight SVG visual diagnostics.

The project intentionally starts with dependency-free SVG writers rather than
notebook-only plotting.  Generated SVG files are deterministic, diffable, and
work on Linux/macOS/Windows without a display server.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Mapping, Sequence


def _color(value: float, vmin: float, vmax: float) -> str:
    """Map a scalar to a blue-white-red color.

    Complexity: O(1).
    """

    if not math.isfinite(value):
        return "#eeeeee"
    if vmax <= vmin:
        ratio = 0.5
    else:
        ratio = min(1.0, max(0.0, (value - vmin) / (vmax - vmin)))
    red = int(255 * ratio)
    blue = int(255 * (1.0 - ratio))
    green = int(255 * (1.0 - abs(ratio - 0.5) * 1.4))
    green = min(255, max(0, green))
    return f"#{red:02x}{green:02x}{blue:02x}"


def write_pair_heatmap_svg(
    matrix: Sequence[Sequence[float]],
    path: Path,
    *,
    cell_size: int = 14,
    title: str = "Pair score heatmap",
) -> Path:
    """Write an SVG heatmap for a pair matrix.

    Formula: each cell visualizes ``P_ij`` or a pair score.  This is a direct
    diagnostic for predicted pair probabilities, thermodynamic priors, or legal
    projection outputs.

    Complexity: O(L^2) time and output size.
    """

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("matrix must be square")
    values = [float(value) for row in matrix for value in row if math.isfinite(float(value))]
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    margin = 80
    width = margin + size * cell_size + 20
    height = margin + size * cell_size + 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="monospace" font-size="16">{html.escape(title)}</text>',
    ]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            x = margin + j * cell_size
            y = margin + i * cell_size
            color = _color(float(value), vmin, vmax)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="0.5"/>'
            )
    parts.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_profile_overlay_svg(
    predicted: Sequence[float],
    target: Sequence[float],
    path: Path,
    *,
    width: int = 900,
    height: int = 260,
    title: str = "Reactivity profile overlay",
) -> Path:
    """Write an SVG line plot comparing predicted and measured reactivity.

    The two profiles are scaled to the same visible y-axis.  Missing values are
    skipped, producing separate polyline segments.

    Complexity: O(L).
    """

    if len(predicted) != len(target):
        raise ValueError("predicted and target lengths must match")
    finite = [float(v) for v in list(predicted) + list(target) if math.isfinite(float(v))]
    if not finite:
        raise ValueError("at least one finite value is required")
    vmin, vmax = min(finite), max(finite)
    if vmax <= vmin:
        vmax = vmin + 1.0
    margin_left, margin_top, margin_bottom, margin_right = 60, 45, 35, 20
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def point(index: int, value: float) -> str:
        """Map a profile value to an SVG coordinate string.

        Complexity: O(1).
        """

        x = margin_left + (index / max(1, len(predicted) - 1)) * plot_w
        y = margin_top + (1.0 - (value - vmin) / (vmax - vmin)) * plot_h
        return f"{x:.2f},{y:.2f}"

    def polyline(values: Sequence[float]) -> str:
        """Return SVG polyline points after skipping missing values.

        Complexity: O(L) for L values.
        """

        return " ".join(point(i, float(value)) for i, value in enumerate(values) if math.isfinite(float(value)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="25" font-family="monospace" font-size="16">{html.escape(title)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>',
        f'<polyline fill="none" stroke="#d62728" stroke-width="2" points="{polyline(target)}"/>',
        f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{polyline(predicted)}"/>',
        f'<text x="{width - 210}" y="25" font-family="monospace" font-size="12" fill="#d62728">target</text>',
        f'<text x="{width - 130}" y="25" font-family="monospace" font-size="12" fill="#1f77b4">predicted</text>',
        "</svg>",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


_CURVE_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf")


def write_training_curves_svg(
    series: Mapping[str, Sequence[float]],
    path: Path,
    *,
    width: int = 900,
    height: int = 320,
    title: str = "Training curves",
) -> Path:
    """Write an SVG multi-line plot of per-epoch training metrics.

    Each named series is drawn against a shared epoch axis.  All series are
    scaled to a common visible y-range so relative trends (loss decreasing while
    F1 increases) are directly readable.  The renderer is dependency-free and
    deterministic, producing diffable output for CI artifact comparison.

    Complexity: O(S * E) for ``S`` series of length ``E``.
    """

    named = {str(name): [float(v) for v in values] for name, values in series.items()}
    if not named:
        raise ValueError("at least one series is required")
    lengths = {len(values) for values in named.values()}
    if len(lengths) != 1:
        raise ValueError("all series must have the same length")
    epochs = lengths.pop()
    if epochs == 0:
        raise ValueError("series must be non-empty")
    finite = [value for values in named.values() for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError("at least one finite value is required")
    vmin, vmax = min(finite), max(finite)
    if vmax <= vmin:
        vmax = vmin + 1.0
    margin_left, margin_top, margin_bottom, margin_right = 60, 45, 40, 160
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def point(index: int, value: float) -> str:
        """Map an (epoch, value) pair to SVG coordinates.

        Complexity: O(1).
        """

        x = margin_left + (index / max(1, epochs - 1)) * plot_w
        y = margin_top + (1.0 - (value - vmin) / (vmax - vmin)) * plot_h
        return f"{x:.2f},{y:.2f}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="25" font-family="monospace" font-size="16">{html.escape(title)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>',
        f'<text x="{margin_left - 55}" y="{margin_top + 10}" font-family="monospace" font-size="11">{vmax:.3f}</text>',
        f'<text x="{margin_left - 55}" y="{margin_top + plot_h}" font-family="monospace" font-size="11">{vmin:.3f}</text>',
        f'<text x="{margin_left}" y="{height - 12}" font-family="monospace" font-size="11">epoch 0</text>',
        f'<text x="{margin_left + plot_w - 60}" y="{height - 12}" font-family="monospace" font-size="11">epoch {epochs - 1}</text>',
    ]
    for index, (name, values) in enumerate(sorted(named.items())):
        color = _CURVE_COLORS[index % len(_CURVE_COLORS)]
        points = " ".join(point(i, value) for i, value in enumerate(values) if math.isfinite(value))
        legend_y = margin_top + 15 + index * 18
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        parts.append(
            f'<text x="{width - margin_right + 10}" y="{legend_y}" font-family="monospace" '
            f'font-size="12" fill="{color}">{html.escape(name)}</text>'
        )
    parts.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_guidance_scan_svg(
    etas: Sequence[float],
    series: Mapping[str, Sequence[float]],
    path: Path,
    *,
    width: int = 900,
    height: int = 340,
    title: str = "Guidance eta scan",
) -> Path:
    """Write an SVG plot of guidance-scan metrics against the ``eta`` axis.

    Unlike :func:`write_training_curves_svg`, which shares one y-axis across all
    series (appropriate for same-scale training losses), this writer normalizes
    *each* series to its own visible ``[0, 1]`` band because a guidance scan
    mixes quantities on very different scales (pairing energy in kcal/mol, pair
    counts, and F1 in ``[0, 1]``).  Each series therefore shows its own
    ``min``/``max`` in the legend so the true magnitudes remain auditable while
    the trends are visually comparable.  The x-axis is the (assumed ascending)
    ``eta`` grid rather than an integer epoch index.

    Per-series normalization maps value ``v`` to
    ``y_norm = (v - min_s) / (max_s - min_s)`` (constant series map to the mid
    line).  The renderer is dependency-free and deterministic.

    Complexity: O(S * E) for ``S`` series of length ``E = len(etas)``.
    """

    eta_values = [float(e) for e in etas]
    if not eta_values:
        raise ValueError("etas must be non-empty")
    named = {str(name): [float(v) for v in values] for name, values in series.items()}
    if not named:
        raise ValueError("at least one series is required")
    if any(len(values) != len(eta_values) for values in named.values()):
        raise ValueError("every series must have the same length as etas")
    eta_min, eta_max = min(eta_values), max(eta_values)
    if eta_max <= eta_min:
        eta_max = eta_min + 1.0
    margin_left, margin_top, margin_bottom, margin_right = 60, 45, 45, 230
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_coord(eta: float) -> float:
        """Map an ``eta`` value to an SVG x-coordinate.

        Complexity: O(1).
        """

        return margin_left + (eta - eta_min) / (eta_max - eta_min) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="25" font-family="monospace" font-size="16">{html.escape(title)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>',
        f'<text x="{margin_left}" y="{height - 14}" font-family="monospace" font-size="11">eta {eta_min:.2f}</text>',
        f'<text x="{margin_left + plot_w - 70}" y="{height - 14}" font-family="monospace" font-size="11">eta {eta_max:.2f}</text>',
    ]
    for index, (name, values) in enumerate(sorted(named.items())):
        color = _CURVE_COLORS[index % len(_CURVE_COLORS)]
        finite = [v for v in values if math.isfinite(v)]
        vmin = min(finite) if finite else 0.0
        vmax = max(finite) if finite else 1.0
        span = vmax - vmin
        coords = []
        for eta, value in zip(eta_values, values):
            if not math.isfinite(value):
                continue
            norm = 0.5 if span <= 0 else (value - vmin) / span
            y = margin_top + (1.0 - norm) * plot_h
            coords.append(f"{x_coord(eta):.2f},{y:.2f}")
        legend_y = margin_top + 15 + index * 20
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(coords)}"/>')
        parts.append(
            f'<text x="{width - margin_right + 10}" y="{legend_y}" font-family="monospace" '
            f'font-size="11" fill="{color}">{html.escape(name)} [{vmin:.2f},{vmax:.2f}]</text>'
        )
    parts.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
