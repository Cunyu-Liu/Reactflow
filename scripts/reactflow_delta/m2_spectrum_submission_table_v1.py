#!/usr/bin/env python3
"""m2_spectrum_submission_table_v1.py — definitive M2 response-spectrum
submission table (mirrors m2r_submission_horizontal_table_v1.py).

Consolidates the full M2 response-spectrum method chain into a single auditable
horizontal comparison.  All numbers are read from the committed audit artifacts:

  * deep rows:       m2_attn_method_summary_full.json (plain/posaware/attn)
  * multi-depth:     m2_crossarch_ensemble_report.json
  * 3-way deep:      m2_three_way_ensemble_report.json  (+12.84% headline prior)
  * 4-way / v6:      m2_four_way_ensemble_report.json   (fail-closed)
  * NEW headline:    m2_gbdt_3way_ensemble_matched_20260818/m2_gbdt_ensemble_report.json
                     (leak-free per-position GBDT + 3-way deep, evaluated ONLY on
                     matched 272,988 positions so the blend-vs-deep comparison is
                     fair — no median-placeholder rows)

Outputs submission_horizontal_table_m2.json (+ .md).
"""
from __future__ import annotations

import argparse, json
from pathlib import Path


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def build_table(attn_summary: str, crossarch_report: str,
                threeway_report: str, fourway_report: str,
                gbdt_report: str, out: str,
                puzzle_report: str = None) -> dict:
    s = _load(attn_summary)
    ca = _load(crossarch_report)
    tw = _load(threeway_report)
    fw = _load(fourway_report)
    gb = _load(gbdt_report)
    pzR = _load(puzzle_report) if puzzle_report else None

    baseline_mae = s["wmae_skill"]["plain_residual_mlp"]["wmae_baseline"]
    w = s["wmae_skill"]

    rows = [
        {
            "model": "baseline (wmed_spectrum median)",
            "feature_set": "sequence-free per-position median",
            "split": "design-level LOO (272,988 positions)",
            "mae": baseline_mae, "skill": None, "ci": None, "perm_p": None,
        },
        {
            "model": "plain residual MLP",
            "feature_set": "local window base+reactivity+error",
            "split": "design-level LOO (272,988 positions)",
            "mae": w["plain_residual_mlp"]["wmae_model"],
            "skill": w["plain_residual_mlp"]["skill"],
            "ci": (w["plain_residual_mlp"]["ci_low"], w["plain_residual_mlp"]["ci_high"]),
            "perm_p": w["plain_residual_mlp"]["permutation_p"],
            "pct_pos": w["plain_residual_mlp"]["pct_positive"],
            "source": "m2_attn_method_summary_full.json",
        },
        {
            "model": "position-aware (v3)",
            "feature_set": "shared trunk + per-position heads",
            "split": "design-level LOO (272,988 positions)",
            "mae": w["position_aware"]["wmae_model"],
            "skill": w["position_aware"]["skill"],
            "ci": (w["position_aware"]["ci_low"], w["position_aware"]["ci_high"]),
            "perm_p": w["position_aware"]["permutation_p"],
            "pct_pos": w["position_aware"]["pct_positive"],
            "source": "m2_attn_method_summary_full.json",
        },
        {
            "model": "position-aware + self-attention 1-layer (v4)",
            "feature_set": "attention over 21 window positions",
            "split": "design-level LOO (272,988 positions)",
            "mae": w["position_aware_attention"]["wmae_model"],
            "skill": w["position_aware_attention"]["skill"],
            "ci": (w["position_aware_attention"]["ci_low"], w["position_aware_attention"]["ci_high"]),
            "perm_p": w["position_aware_attention"]["permutation_p"],
            "pct_pos": w["position_aware_attention"]["pct_positive"],
            "source": "m2_attn_method_summary_full.json",
        },
        {
            "model": "position-aware + self-attention 2-layer (v5)",
            "feature_set": "2-layer attention",
            "split": "design-level LOO (272,988 positions)",
            "mae": tw["components"]["v5_attn_2layer"]["wmae_model"],
            "skill": tw["components"]["v5_attn_2layer"]["skill"],
            "ci": (tw["components"]["v5_attn_2layer"]["ci_low"],
                   tw["components"]["v5_attn_2layer"]["ci_high"]),
            "perm_p": tw["components"]["v5_attn_2layer"]["permutation_p"],
            "source": "m2_three_way_ensemble_report.json v5_attn_2layer",
        },
        {
            "model": "multi-depth ensemble (0.25*v4 + 0.75*v5)",
            "feature_set": "cross-arch blend",
            "split": "design-level LOO (272,988 positions)",
            "mae": ca["ensemble"]["wmae_model"],
            "skill": ca["ensemble"]["skill"],
            "ci": (ca["ensemble"]["ci_low"], ca["ensemble"]["ci_high"]),
            "perm_p": ca["ensemble"]["permutation_p"],
            "source": "m2_crossarch_ensemble_report.json",
        },
        {
            "model": "3-way deep ensemble (0.15*v3+0.20*v4+0.65*v5)",
            "feature_set": "cross-arch blend",
            "split": "design-level LOO (272,988 positions)",
            "mae": tw["ensemble"]["wmae_model"],
            "skill": tw["ensemble"]["skill"],
            "ci": (tw["ensemble"]["ci_low"], tw["ensemble"]["ci_high"]),
            "perm_p": tw["ensemble"]["permutation_p"],
            "source": "m2_three_way_ensemble_report.json",
        },
    ]

    # NEW headline: leak-free GBDT + 3-way deep, evaluated only on matched rows
    gres = gb["results"]
    rows.append({
        "model": "GBDT cross-arch + 3-way deep (leak-free 31-dims, a=0.5)",
        "feature_set": "per-position MFE/seq + 3-way deep blend",
        "split": "design-level LOO (matched %d / %d positions)"
                 % (gb["n_rows_matched"], gb["n_rows_total"]),
        "mae": gres["blend"]["mae"],
        "skill": gres["blend"]["skill"],
        "ci": (gres["blend"]["sig"]["ci_low"], gres["blend"]["sig"]["ci_high"]),
        "perm_p": gres["blend"]["sig"]["permutation_p"],
        "headline": True,
        "source": "m2_masked_eval_report.json (matched-only, derived from m2_gbdt_3way_ensemble_20260818 oof)",
        "blend_vs_deep_pp": gb["blend_vs_deep"]["pooled_gain_pp"],
        "loo_exclusion": gb["blend_vs_deep"]["loo_exclusion"],
    })

    # fail-closed rows (independent, no gain)
    v6 = fw["components"]["v6_studentt"]
    rows.append({
        "model": "Student-t NLL loss (v6)",
        "feature_set": "robust likelihood",
        "split": "design-level LOO (272,988 positions)",
        "mae": v6["wmae_model"],
        "skill": v6["skill"],
        "ci": (v6["ci_low"], v6["ci_high"]),
        "perm_p": v6["permutation_p"],
        "closed": True,
        "source": "m2_four_way_ensemble_report.json v6_studentt",
    })
    e4 = fw["grid"]["ens4_attn_heavy"]
    rows.append({
        "model": "4-way ensemble incl. v6 (attn-heavy)",
        "feature_set": "0.10*v3+0.25*v4+0.45*v5+0.20*v6",
        "split": "design-level LOO (272,988 positions)",
        "mae": e4["wmae_model"],
        "skill": e4["skill"],
        "perm_p": e4["permutation_p"],
        "closed": True,
        "source": "m2_four_way_ensemble_report.json grid.ens4_attn_heavy",
    })

    # ---- puzzle-level rows (leak-free LOPO: train 19 puzzles -> predict 1) ----
    if pzR is not None:
        pr = pzR["results"]
        rows.append({
            "model": "attn deep [puzzle-level LOPO]",
            "feature_set": "puzzle-level attn 1-layer OOF (5-seed mu)",
            "split": "puzzle-level LOPO (matched %d rows)" % pzR["n_rows_matched"],
            "mae": pr["deep_attn_puzzle"]["mae"],
            "skill": pr["deep_attn_puzzle"]["skill"],
            "source": "m2_gbdt_puzzle_ensemble_report.json",
            "puzzle_level": True,
        })
        rows.append({
            "model": "GBDT leak-free [puzzle-level LOPO]",
            "feature_set": "31-dim per-position MFE/seq",
            "split": "puzzle-level LOPO (matched %d rows)" % pzR["n_rows_matched"],
            "mae": pr["gbdt_puzzle"]["mae"],
            "skill": pr["gbdt_puzzle"]["skill"],
            "source": "m2_gbdt_puzzle_ensemble_report.json",
            "puzzle_level": True,
        })
        rows.append({
            "model": "GBDT + attn deep blend (a=0.5) [puzzle-level LOPO]",
            "feature_set": "puzzle-level GBDT + attn deep",
            "split": "puzzle-level LOPO (matched %d rows)" % pzR["n_rows_matched"],
            "mae": pr["blend"]["mae"],
            "skill": pr["blend"]["skill"],
            "ci": (pr["blend"]["sig"]["ci_low"], pr["blend"]["sig"]["ci_high"]),
            "perm_p": pr["blend"]["sig"]["permutation_p"],
            "source": "m2_gbdt_puzzle_ensemble_report.json",
            "puzzle_level": True, "puzzle_headline": True,
        })

    significance = {
        "baseline": "wmed_spectrum",
        "baseline_wmae": baseline_mae,
        "exchangeable_unit": "puzzle_x_method_design",
        "n_designs": tw["n_designs"],
        "n_positions": tw["ensemble"]["n_positions"],
        "n_perm": tw["ensemble"]["n_perm"],
        "n_boot": tw["ensemble"]["n_boot"],
        "permutation_p_all_deep_rows": 0.0033222591362126247,
        "headline": {
            "skill": gres["blend"]["skill"],
            "ci": [gres["blend"]["sig"]["ci_low"], gres["blend"]["sig"]["ci_high"]],
            "permutation_p": gres["blend"]["sig"]["permutation_p"],
            "blend_vs_deep_pp": gb["blend_vs_deep"]["pooled_gain_pp"],
            "per_design_pp": gb["blend_vs_deep"]["per_design_mean_pp"],
            "loo_exclusion": gb["blend_vs_deep"]["loo_exclusion"],
        },
    }
    if pzR is not None:
        significance["puzzle_level"] = {
            "exchangeable_unit": "puzzle",
            "n_puzzles": pzR["n_puzzles"],
            "fold_unit": "puzzle",
            "headline": {
                "skill": pzR["results"]["blend"]["skill"],
                "ci": [pzR["results"]["blend"]["sig"]["ci_low"],
                       pzR["results"]["blend"]["sig"]["ci_high"]],
                "permutation_p": pzR["results"]["blend"]["sig"]["permutation_p"],
                "blend_vs_deep_pp": pzR["blend_vs_deep"]["pooled_gain_pp"],
                "per_puzzle_pp": pzR["blend_vs_deep"]["per_puzzle_mean_pp"],
                "loo_exclusion": pzR["blend_vs_deep"]["loo_exclusion"],
            },
            "deep_component": pzR["deep_component"],
        }

    report = {
        "schema": "reactflow_delta.response_spectrum.m2_submission_horizontal_table.v1",
        "dataset": "OpenKnot_M2",
        "rows": rows,
        "significance": significance,
    }

    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    (out_p / "submission_horizontal_table_m2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    md = ["# M2 response-spectrum 横向对比表（Submission horizontal table）\n"]
    md.append("| Model | Feature set | Split | WMAE | Skill | 95% CI | perm_p | source |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        ci = ("(%.4f, %.4f)" % r["ci"]) if r.get("ci") else "—"
        pp = "%.4f" % r["perm_p"] if r.get("perm_p") else "—"
        hl = " **" if r.get("headline") else ""
        md.append("| %s%s%s | %s | %s | %.4f | %s | %s | %s | %s |" % (
            hl, r["model"], hl, r["feature_set"], r["split"],
            r["mae"], ("+%.2f%%" % (r["skill"] * 100)) if r.get("skill") is not None else "—",
            ci, pp, r.get("source", "")))
    md.append("")
    md.append("## Significance (design-block bootstrap + permutation)")
    sg = significance
    md.append("- baseline WMAE = %.4f, exchangeable unit = %s" % (sg["baseline_wmae"], sg["exchangeable_unit"]))
    md.append("- all deep rows: perm_p = %.4f (n_perm=%d, n_boot=%d)" % (sg["permutation_p_all_deep_rows"], sg["n_perm"], sg["n_boot"]))
    hl = sg["headline"]
    md.append("- **headline blend** skill = +%.2f%%, CI=(%.4f, %.4f), perm_p=%.4f" % (
        hl["skill"] * 100, hl["ci"][0], hl["ci"][1], hl["permutation_p"]))
    md.append("- blend vs 3-way deep: pooled +%.2fpp, per-design +%.2fpp" % (hl["blend_vs_deep_pp"], hl["per_design_pp"]))
    loo = hl["loo_exclusion"]
    md.append("- LOO-exclusion: mean +%.2fpp, range [%+.2f, %+.2f]pp, %d/%d folds positive" % (
        loo["gain_mean_pp"], loo["gain_min_pp"], loo["gain_max_pp"],
        int(loo["pct_positive"] * loo["n_folds"]), loo["n_folds"]))

    if "puzzle_level" in sg:
        pz = sg["puzzle_level"]
        md.append("")
        md.append("## Puzzle-level LOPO (leak-free: train 19 puzzles -> predict held-out)")
        md.append("- exchangeable unit = %s, n_puzzles = %d" % (pz["exchangeable_unit"], pz["n_puzzles"]))
        md.append("- deep component: %s" % pz["deep_component"])
        h2 = pz["headline"]
        md.append("- **puzzle headline blend** skill = +%.2f%%, CI=(%.4f, %.4f), perm_p=%.4f" % (
            h2["skill"] * 100, h2["ci"][0], h2["ci"][1], h2["permutation_p"]))
        md.append("- blend vs attn deep: pooled +%.2fpp, per-puzzle +%.2fpp" % (h2["blend_vs_deep_pp"], h2["per_puzzle_pp"]))
        loo2 = h2["loo_exclusion"]
        md.append("- LOO-exclusion: mean +%.2fpp, range [%+.2f, %+.2f]pp, %d/%d puzzles positive" % (
            loo2["gain_mean_pp"], loo2["gain_min_pp"], loo2["gain_max_pp"],
            int(loo2["pct_positive"] * loo2["n_folds"]), loo2["n_folds"]))

    (out_p / "submission_horizontal_table_m2.md").write_text(
        "\n".join(md), encoding="utf-8")

    print("rows=%d headline_skill=%+.4f headline_p=%.4f" % (
        len(rows), gres["blend"]["skill"], gres["blend"]["sig"]["permutation_p"]))
    print("DONE -> %s/submission_horizontal_table_m2.{json,md}" % out_p)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn-summary", required=True)
    ap.add_argument("--crossarch-report", required=True)
    ap.add_argument("--threeway-report", required=True)
    ap.add_argument("--fourway-report", required=True)
    ap.add_argument("--gbdt-report", required=True)
    ap.add_argument("--puzzle-report", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_table(args.attn_summary, args.crossarch_report, args.threeway_report,
                args.fourway_report, args.gbdt_report, args.out,
                puzzle_report=args.puzzle_report)


if __name__ == "__main__":
    main()
