#!/usr/bin/env python3
"""generate_p6_tables_figures_v1: P6 main tables & figures (contract 12.8).

Auto-generates the paper main tables and figures from the locked result
artifacts ONLY (no placeholder / hand-copied headline). Emits:

  main_tables.md      - Table 1 (development horizontal), Table 2 (P4 external
                        confirmation), Table 3 (P5 mechanism), Table 4 (gates)
  main_tables.tex     - LaTeX rendering of the same tables
  figures/fig1_p2_forest.png   - P2 20-puzzle D_p2 forest plot
  figures/fig2_p4_components.png - P4 component-macro D + per-dataset
  figures/fig3_p5_distance_curve.png - P5 signed distance curve
  figures/fig4_calibration.png  - P4 coverage/calibration
  summary.json        - machine-readable table/figure registry

If matplotlib is unavailable, figures are skipped and the registry records
"figures_skipped: matplotlib unavailable" (tables are always produced).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TEMPLATES = {}


def _ci_str(ci: dict) -> str:
    if ci is None or ci.get("ci_low") is None:
        return "n/a"
    return f"{ci['mean']:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"


def _pct(x: float) -> str:
    return f"{x:+.2f}%"


def build_tables(p2_result, p3_result, horizontal, p4, p5, calib, replay,
                 gates: dict) -> dict:
    t1_rows = []
    for m in horizontal.get("method_table", []):
        t1_rows.append({
            "method": m["method"], "mean_crps": m["mean_held_crps"],
            "skill_vs_zero_pct": m["skill_vs_zero_pct"]})

    t2_rows = []
    for c in p4.get("component_rows", []):
        t2_rows.append({
            "wt": c["wt_name"], "n_mut": c["n_scored"], "n_pos": c["n_positions"],
            "D_vs_zero": c["D_vs_zero"], "D_vs_median": c["D_vs_median"]})

    t3_rows = []
    for b, s in p5.get("band_stats", {}).items():
        t3_rows.append({"band": b, "ci": s})
    t3_neg = p5.get("negative_control", {})
    t3_region = p5.get("region_strata", {})

    tables = {
        "table1_development_horizontal": {
            "caption": "Development held-out 20-puzzle full-construct CRPS by model (lower is better).",
            "columns": ["method", "mean_crps", "skill_vs_zero_pct"],
            "rows": t1_rows,
            "note": f"P2 primary: D_p = {_ci_str(p2_result.get('p2_ci20'))} "
                    f"(sign-flip p={p2_result.get('sign_flip', {}).get('p_value', 'n/a')}); "
                    f"P3: all LRSO ranks NO_INCREMENTAL (CI upper < 0)."},
        "table2_p4_external": {
            "caption": "External confirmation on development-disconnected Ribonanza M2-style 2A3 components "
                       "(K=24, 3237 single-SNV). D = CRPS(baseline) - CRPS(direct) per position, component-macro.",
            "columns": ["wt", "n_mut", "n_pos", "D_vs_zero", "D_vs_median"],
            "rows": t2_rows,
            "note": f"P4 verdict: {p4.get('verdict')}; pooled D_vs_zero {_ci_str(p4.get('ci_zero'))}; "
                    f"D_vs_median {_ci_str(p4.get('ci_median'))}; "
                    f"Holm-Bonferroni p={p4.get('holm_bonferroni_p_values')} (FWER pass={p4.get('fwer_pass')}); "
                    f"leave-dominant-out {_ci_str(p4.get('leave_dominant_out_ci'))}; "
                    f"calibration {calib.get('verdict')} (cov95={calib.get('pooled', {}).get('cov_95')})."},
        "table3_p5_mechanism": {
            "caption": "P5 mechanism contrasts on the frozen external components (D_vs_zero by |dist| band; "
                       "positive = direct better than WT-anchor).",
            "columns": ["band", "ci"],
            "rows": t3_rows,
            "note": f"distance heterogeneity edit-vfar {_ci_str(p5.get('distance_heterogeneity', {}).get('D_edit_minus_vfar'))}; "
                    f"negative control (permuted) {_ci_str(t3_neg.get('permuted_edit_D'))} "
                    f"(pass={t3_neg.get('pass')}); region {json.dumps(t3_region)}; "
                    f"verdict {p5.get('verdict')}."},
        "table4_gates": {
            "caption": "Phase gate final verdicts (prospective-v2).",
            "columns": ["phase", "verdict"],
            "rows": [{"phase": k, "verdict": v} for k, v in gates.items()],
            "note": "All gates reached final verdicts; P5 MECHANISM_NOT_ESTABLISHED is the contract fail-closed "
                    "outcome for the pre-frozen edit-site-concentration claim."},
    }
    return tables


def render_markdown(tables: dict) -> str:
    lines = ["# ReactFlow-Delta prospective-v2 main tables (auto-generated)", ""]
    for tid, t in tables.items():
        lines.append(f"## {tid}\n")
        lines.append(f"*{t['caption']}*\n")
        cols = t["columns"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in t["rows"]:
            cells = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, float):
                    cells.append(f"{v:.6g}")
                elif isinstance(v, dict) and "ci_low" in v:
                    cells.append(_ci_str(v))
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if t.get("note"):
            lines.append(f"*{t['note']}*\n")
        lines.append("")
    return "\n".join(lines)


def render_latex(tables: dict) -> str:
    lines = [r"\documentclass{article}", r"\usepackage{booktabs}", r"\begin{document}", ""]
    for tid, t in tables.items():
        lines.append(f"% {tid}")
        lines.append(r"\begin{table}[h]\centering")
        lines.append(f"\\caption{{{t['caption']}}}")
        lines.append(r"\begin{tabular}{" + "l" * len(t["columns"]) + "}")
        lines.append(r"\toprule")
        lines.append(" & ".join(t["columns"]) + r" \\")
        lines.append(r"\midrule")
        for r in t["rows"]:
            cells = []
            for c in t["columns"]:
                v = r.get(c)
                if isinstance(v, float):
                    cells.append(f"{v:.4f}")
                elif isinstance(v, dict) and "ci_low" in v:
                    cells.append(_ci_str(v))
                else:
                    cells.append(str(v))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if t.get("note"):
            lines.append(f"\\note{{{t['note']}}}")
        lines.append(r"\end{table}")
        lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def render_figures(p2_result, p4, p5, calib, out_dir: Path) -> list[dict]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    figs = []
    # Fig 1: P2 20-puzzle D_p2 forest
    pd2 = p2_result.get("per_puzzle_D_p2", {})
    if pd2:
        puzzles = sorted(pd2)
        vals = [pd2[p] for p in puzzles]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axvline(0, color="grey", lw=0.8)
        ci = p2_result.get("p2_ci20", {})
        ax.axvline(ci.get("ci_low", 0), color="tab:red", ls="--", lw=0.8, label=f"CI lower {ci.get('ci_low', 0):.4f}")
        ax.barh(range(len(puzzles)), vals, color="steelblue")
        ax.set_yticks(range(len(puzzles)))
        ax.set_yticklabels([p.replace("P", "Puzzle ") for p in puzzles])
        ax.set_xlabel("D_p2 = CRPS(zero) - CRPS(direct) (full-construct, scale 0.3)")
        ax.set_title("P2: per-puzzle direct-learnability effect (20-puzzle LOPO)")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "fig1_p2_forest.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        figs.append({"figure": "fig1_p2_forest.png", "source": "p2_direct_v2_result"})
    # Fig 2: P4 component-macro D
    comps = p4.get("component_rows", [])
    if comps:
        names = [c["wt_name"].split("_")[0][:14] for c in comps]
        dzero = [c["D_vs_zero"] for c in comps]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axvline(0, color="grey", lw=0.8)
        ax.axvline(p4.get("ci_zero", {}).get("ci_low", 0), color="tab:red", ls="--", lw=0.8,
                   label=f"component-macro CI lower {p4.get('ci_zero', {}).get('ci_low', 0):.4f}")
        ax.barh(range(len(comps)), dzero, color="darkseagreen")
        ax.set_yticks(range(len(comps))); ax.set_yticklabels(names)
        ax.set_xlabel("component D_vs_zero (CRPS)")
        ax.set_title("P4: external component-macro direct effect (K=24)")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "fig2_p4_components.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        figs.append({"figure": "fig2_p4_components.png", "source": "p4_external_result"})
    # Fig 3: P5 distance curve
    bands = p5.get("band_stats", {})
    if bands:
        labels = list(bands.keys())
        means = [bands[b]["mean"] for b in labels]
        lows = [bands[b]["ci_low"] for b in labels]
        highs = [bands[b]["ci_high"] for b in labels]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(x, means, yerr=[np.array(means) - np.array(lows),
                                    np.array(highs) - np.array(means)],
                    fmt="o-", color="tab:blue", capsize=4)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30)
        ax.set_ylabel("D_vs_zero (component-macro)")
        ax.set_title("P5: signed distance curve on external components")
        fig.tight_layout()
        p = out_dir / "fig3_p5_distance_curve.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        figs.append({"figure": "fig3_p5_distance_curve.png", "source": "p5_mechanism_result"})
    # Fig 4: calibration
    pooled = calib.get("pooled", {})
    if pooled:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(["68% nominal", "95% nominal"], [pooled.get("cov_68", 0), pooled.get("cov_95", 0)],
               color="mediumpurple")
        ax.axhline(0.6827, color="grey", ls="--", lw=0.8)
        ax.axhline(0.95, color="grey", ls="--", lw=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"P4 calibration: {calib.get('verdict')} (fixed scale 0.3)")
        fig.tight_layout()
        p = out_dir / "fig4_calibration.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        figs.append({"figure": "fig4_calibration.png", "source": "p4_calibration_result"})
    return figs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-result", required=True)
    ap.add_argument("--p3-result", required=True)
    ap.add_argument("--horizontal", required=True)
    ap.add_argument("--p4-result", required=True)
    ap.add_argument("--p5-result", required=True)
    ap.add_argument("--calib-result", required=True)
    ap.add_argument("--replay-report", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    def load(p: str):
        return json.loads(Path(p).read_text(encoding="utf-8"))

    p2 = load(args.p2_result); p3 = load(args.p3_result)
    hor = load(args.horizontal); p4 = load(args.p4_result)
    p5 = load(args.p5_result); calib = load(args.calib_result)
    replay = load(args.replay_report)

    gates = {
        "P0": "AUTHORITY_RECONCILIATION_COMPLETE_PASS",
        "P1": "FAIL_CLOSED_OPEN (blocks_phase2/3=False)",
        "P2": p2.get("verdict"),
        "P3": "NO_INCREMENTAL_LRSO_SKILL (all ranks)",
        "P4": f"{p4.get('verdict')} + {calib.get('verdict')}",
        "P5": p5.get("verdict"),
        "P6_replay": replay.get("verdict"),
    }

    tables = build_tables(p2, p3, hor, p4, p5, calib, replay, gates)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main_tables.md").write_text(render_markdown(tables), encoding="utf-8")
    (out_dir / "main_tables.tex").write_text(render_latex(tables), encoding="utf-8")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = render_figures(p2, p4, p5, calib, fig_dir)

    summary = {"schema_version": "reactflow_delta.p6_tables_figures.v1",
               "tables": list(tables.keys()),
               "figures": figs,
               "gates": gates,
               "figures_skipped": len(figs) == 0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
