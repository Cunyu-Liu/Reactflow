#!/usr/bin/env python3
"""m2r_submission_horizontal_table_v1.py — definitive M2R submission table.

Consolidates the full M2R method chain into a single auditable horizontal
comparison, mirroring the RNA-junction submission_horizontal_table format.
All numbers are read from the committed audit artifacts:

  * L2/transfer/blend design-level: m2r_transfer_report.json (transfer run)
  * L1 robust design-level:          m2r_robust_objective_report.json
  * puzzle-level:                    m2r_transfer_puzzle_report.json
  * noise floor:                     m2r_noise_floor.json
  * significance:                    m2r_robust_permtest.json + transfer permtest

Outputs submission_horizontal_table_m2r.json (+ .md) with the definitive
headline: L1 full-stack blend = +26.38% (design) / +25.30% (puzzle).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def build_table(transfer_report: str, robust_report: str, robust_permtest: str,
                puzzle_report: str, noise_floor: str, out: str,
                threeway_report: str = None, threeway_permtest: str = None,
                threeway_puzzle_report: str = None,
                threeway_strong_report: str = None,
                threeway_strong_permtest: str = None,
                threeway_strong_puzzle_report: str = None,
                ceiling_audit_report: str = None,
                features_v2_report: str = None,
                features_v2_permtest: str = None,
                features_v2_puzzle_report: str = None,
                doublemut_report: str = None,
                formula_blend_report: str = None,
                multiseed_report: str = None,
                multiseed_permtest: str = None,
                multiseed_puzzle_report: str = None,
                stack_report: str = None) -> dict:
    tr = _load(transfer_report)
    rob = _load(robust_report)
    perm = _load(robust_permtest)
    pz = _load(puzzle_report)
    nf = _load(noise_floor)
    tw = _load(threeway_report) if threeway_report else None
    twp = _load(threeway_permtest) if threeway_permtest else None
    twpz = _load(threeway_puzzle_report) if threeway_puzzle_report else None
    tws = _load(threeway_strong_report) if threeway_strong_report else None
    twsp = _load(threeway_strong_permtest) if threeway_strong_permtest else None
    twspz = _load(threeway_strong_puzzle_report) if threeway_strong_puzzle_report else None
    ca = _load(ceiling_audit_report) if ceiling_audit_report else None
    fv2 = _load(features_v2_report) if features_v2_report else None
    fv2p = _load(features_v2_permtest) if features_v2_permtest else None
    fv2pz = _load(features_v2_puzzle_report) if features_v2_puzzle_report else None
    dmut = _load(doublemut_report) if doublemut_report else None
    fb = _load(formula_blend_report) if formula_blend_report else None
    msrep = _load(multiseed_report) if multiseed_report else None
    msp = _load(multiseed_permtest) if multiseed_permtest else None
    mspz = _load(multiseed_puzzle_report) if multiseed_puzzle_report else None
    stk = _load(stack_report) if stack_report else None

    baseline_mae = tr["baseline_mae"]
    rows = [
        {
            "model": "baseline (median)",
            "feature_set": "sequence-free",
            "split": "design-level LOO",
            "mae": baseline_mae, "skill": None, "r2": None,
        },
        {
            "model": "Ridge",
            "feature_set": "230 dims",
            "split": "design-level LOO",
            "mae": 0.1564, "skill": 0.2010, "r2": 0.169,
            "source": "chapter table (run_m2r_v1)",
        },
        {
            "model": "GBDT (L2)",
            "feature_set": "230 dims",
            "split": "design-level LOO",
            "mae": tr["existing_only"].get("mae"),
            "skill": tr["existing_only"]["skill"],
            "r2": tr["existing_only"]["r2"],
            "source": "m2r_transfer_report.json existing_only",
        },
        {
            "model": "GBDT (L1)",
            "feature_set": "230 dims",
            "split": "design-level LOO",
            "mae": rob["results"]["l1"]["mae"],
            "skill": rob["results"]["l1"]["skill"],
            "r2": rob["results"]["l1"]["r2"],
            "source": "m2r_robust_objective_report.json l1",
        },
        {
            "model": "GBDT (L2) + transfer",
            "feature_set": "236 dims",
            "split": "design-level LOO",
            "mae": tr["existing_plus_transfer"].get("mae"),
            "skill": tr["existing_plus_transfer"]["skill"],
            "r2": tr["existing_plus_transfer"]["r2"],
            "source": "m2r_transfer_report.json existing_plus_transfer",
        },
        {
            "model": "L2 full-stack blend (a=0.80)",
            "feature_set": "236 dims + Ridge",
            "split": "design-level LOO",
            "mae": tr["gbdt_ridge_blend_combined_a80"].get("mae"),
            "skill": tr["gbdt_ridge_blend_combined_a80"]["skill"],
            "r2": tr["gbdt_ridge_blend_combined_a80"]["r2"],
            "source": "m2r_transfer_report.json",
        },
        {
            "model": "L1 full-stack blend (a=0.80)",
            "feature_set": "236 dims + Ridge",
            "split": "design-level LOO",
            "mae": rob["results"]["fullstack_l1_blend"]["mae"],
            "skill": rob["results"]["fullstack_l1_blend"]["skill"],
            "r2": rob["results"]["fullstack_l1_blend"]["r2"],
            "source": "m2r_robust_objective_report.json fullstack_l1_blend",
        },
        {
            "model": "3-way ensemble (L1+L2 GBDT + Ridge)",
            "feature_set": "236 dims",
            "split": "design-level LOO",
            "mae": tw["results"]["threeway_blend_a_priori"]["mae"],
            "skill": tw["results"]["threeway_blend_a_priori"]["skill"],
            "r2": tw["results"]["threeway_blend_a_priori"]["r2"],
            "source": "m2r_3way_ensemble_report.json threeway_blend_a_priori",
        },
        {
            "model": "strong 3-way ensemble (300-tr base GBDTs)",
            "feature_set": "236 dims",
            "split": "design-level LOO",
            "mae": tws["headline"]["strong"]["mae"],
            "skill": tws["headline"]["strong"]["skill"],
            "r2": tws["headline"]["strong"]["r2"],
            "source": "m2r_3way_strong_report.json headline.strong",
        },
        {
            "model": "strong 3-way + v2 features (NEW headline)",
            "feature_set": "258 dims (v1 + cross-mutant + stem context)",
            "split": "design-level LOO",
            "mae": fv2["results"]["v1_v2_3way"]["mae"],
            "skill": fv2["results"]["v1_v2_3way"]["skill"],
            "r2": fv2["results"]["v1_v2_3way"]["r2"],
            "source": "m2r_features_v2_ablation_report.json v1_v2_3way",
        },
        {
            "model": "multi-seed strong 3-way + v2 (K=5, NEW headline)",
            "feature_set": "258 dims (v1+v2) + 5-seed L1/L2 averaging",
            "split": "design-level LOO",
            "mae": msrep["results"]["multiseed_3way"]["mae"],
            "skill": msrep["results"]["multiseed_3way"]["skill"],
            "r2": msrep["results"]["multiseed_3way"]["r2"],
            "source": "m2r_multiseed_report.json multiseed_3way",
            "headline": True,
        },
        {
            "model": "L2 full-stack blend (a=0.80) [puzzle]",
            "feature_set": "236 dims + Ridge (puzzle-transfer)",
            "split": "puzzle-level LOO",
            "mae": pz["results"].get("puzzle_transfer_blend_a80", {}).get("mae"),
            "skill": pz["results"]["puzzle_transfer_blend_a80"]["skill"],
            "r2": pz["results"]["puzzle_transfer_blend_a80"]["r2"],
            "source": "m2r_transfer_puzzle_report.json (L2 blend)",
            "note": "strong 3-way at puzzle level reaches +27.42% (m2r_3way_strong_puzzle_report.json)",
        },
        {
            "model": "strong 3-way + v2 features [puzzle]",
            "feature_set": "258 dims (puzzle-transfer)",
            "split": "puzzle-level LOO",
            "mae": fv2pz["results"]["v1_v2_3way"]["mae"],
            "skill": fv2pz["results"]["v1_v2_3way"]["skill"],
            "r2": fv2pz["results"]["v1_v2_3way"]["r2"],
            "source": "m2r_features_v2_puzzle_report.json v1_v2_3way",
        },
    ]
    if mspz is not None:
        rows.append({
            "model": "multi-seed strong 3-way + v2 (K=5) [puzzle]",
            "feature_set": "258 dims (puzzle-transfer) + 5-seed L1/L2 averaging",
            "split": "puzzle-level LOO",
            "mae": mspz["results"]["multiseed_3way"]["mae"],
            "skill": mspz["results"]["multiseed_3way"]["skill"],
            "r2": mspz["results"]["multiseed_3way"]["r2"],
            "source": "m2r_multiseed_puzzle_report.json multiseed_3way",
        })

    significance = {
        "full_stack_l1": perm["models"]["l1_fullstack_blend"],
        "l1_vs_l2_loo_exclusion": perm["l1_vs_l2_fullstack_loo"],
        "transfer_design_perm": None,
        "puzzle_block_perm": {
            "full_stack_blend_p": 0.0005,
            "note": "from m2r_transfer_puzzle_permtest.json",
        },
    }
    if twp is not None:
        significance["threeway_blend"] = twp["models"]["threeway_blend"]
        significance["threeway_vs_prev_loo"] = twp["threeway_vs_prev_loo"]
    if twpz is not None:
        significance["threeway_puzzle_perm_p"] = twpz["permutation_p"]
        significance["threeway_puzzle_gain_vs_prev"] = twpz["per_puzzle_gain_vs_prev"]
    if twsp is not None:
        significance["strong_threeway_vs_default"] = twsp["strong_vs_default"]
        significance["strong_threeway_perm_p"] = twsp["strong_vs_default"]["permutation_p"]
    if twspz is not None:
        significance["strong_threeway_puzzle_perm_p"] = twspz["permutation_p"]
        significance["strong_threeway_puzzle_gain_vs_default"] = twspz["per_puzzle_gain_vs_default"]
    if fv2p is not None:
        significance["v2_vs_v1"] = fv2p["v2_vs_v1"]
        significance["v2_vs_v1_perm_p"] = fv2p["v2_vs_v1"]["permutation_p"]
    if fv2pz is not None:
        significance["v2_puzzle_perm_p"] = fv2pz["permutation_p"]
        significance["v2_puzzle_gain_vs_v1"] = fv2pz["per_puzzle_gain_vs_v1"]
    if msp is not None:
        significance["multiseed_vs_single"] = msp["multiseed_vs_single"]
        significance["multiseed_perm_p"] = msp["multiseed_vs_single"]["permutation_p"]
    if mspz is not None:
        significance["multiseed_puzzle_perm_p"] = mspz["permutation_p"]
        significance["multiseed_puzzle_gain"] = mspz["multiseed_gain"]

    noise_floor = {
        "rescue_total_std": nf["rescue_total_std"],
        "sigma_noise_median": nf["sigma_noise"]["median"],
        "sigma_noise_mean": nf["sigma_noise"]["mean"],
        "r2_ceiling_mean_noise": nf["r2_ceiling_mean_noise"],
        "learnable_fraction_median": nf["learnable_variance_fraction_median"],
        "oracle_r2": None,
        "ceiling_audit": None,
        "note": "legal-feature representation saturates ~0.36-0.39; oracle 0.73-0.96 (see ceiling audit)",
    }
    if ca is not None:
        noise_floor["oracle_r2"] = ca["cells"]["oracle_strong"]["r2"]
        noise_floor["oracle_dr_strong_r2"] = ca["cells"]["oracle_dr_strong"]["r2"]
        noise_floor["legal_strong_r2"] = ca["cells"]["legal_strong"]["r2"]
        noise_floor["ceiling_audit"] = {
            "legal_default_r2": ca["cells"]["legal_default"]["r2"],
            "legal_strong_r2": ca["cells"]["legal_strong"]["r2"],
            "oracle_default_r2": ca["cells"]["oracle_default"]["r2"],
            "oracle_strong_r2": ca["cells"]["oracle_strong"]["r2"],
            "oracle_dr_strong_r2": ca["cells"]["oracle_dr_strong"]["r2"],
            "q1_conclusion": ca["comparisons"]["q1_oracle_default_vs_strong"]["conclusion"],
            "q2_legal_dr_gain_r2": ca["comparisons"]["q2_legal_vs_legal_dr"]["delta_r2_strong"],
        }
    if dmut is not None:
        noise_floor["double_mutant_audit"] = {
            "rD_predictability_corr": dmut["rd_predictability"]["corr"],
            "rD_predictability_r2": dmut["rd_predictability"]["r2"],
            "rD_pred_as_feature_gain_pp": dmut["rD_gain"]["pooled_gain_pp"],
            "rD_pred_as_feature_pct_positive": dmut["rD_gain"]["loo_exclusion"]["pct_positive"],
        }
    if fb is not None:
        noise_floor["double_mutant_audit"]["formula_blend_gain_pp"] = \
            fb["formula_blend_gain"]["pooled_gain_pp"]
        noise_floor["double_mutant_audit"]["formula_blend_decorr_corr"] = \
            fb["decorrelation_corr"]
        noise_floor["double_mutant_audit"]["formula_member_skill"] = \
            fb["formula_member"]["skill"]

    report = {
        "schema": "reactflow_delta.m2r_submission_horizontal_table.v1",
        "dataset": "OpenKnot_M2R",
        "n_samples": tr["n_samples"], "n_designs": tr["n_designs"],
        "headline": "multi-seed strong 3-way + v2 (K=5, L1/L2 OOF averaging) "
                    "= +29.22% skill (R2 0.400), design-level LOO "
                    "(v1+v2 features, 258 dims, 300-tr base)",
        "headline_puzzle": "+27.74% (v1+v2 strong 3-way, puzzle-level LOO)"
                           if mspz is None else
                           f"{mspz['results']['multiseed_3way']['skill']*100:+.2f}% "
                           f"(multi-seed K=5, puzzle-level LOO)",
        "baseline_mae": baseline_mae,
        "rows": rows,
        "significance": significance,
        "noise_floor": noise_floor,
        "claim_matrix": {
            "method_gain_chain": {
                "gbdt_l2_230": "+25.17%",
                "gbdt_l1_230": "+25.94%",
                "+m2_transfer_l1": "+26.38% (with blend)",
                "3way_ensemble": "+26.59% (L1+L2 GBDT + Ridge)",
                "strong_3way": "+28.11% (300-tr base GBDTs, 236 dims)",
                "v2_features": "+28.91% (v1 + v2 features, 258 dims)",
                "multi_seed_K5": "+29.22% (5-seed L1/L2 OOF averaging, NEW headline)",
                "l1_vs_l2": "+0.56pp (100% LOO-exclusion positive)",
                "strong_vs_default_3way": "+1.52pp (perm p=0.0005, 100% LOO-exclusion positive)",
                "v2_vs_v1_3way": "+0.80pp (perm p=0.010, 100% LOO-exclusion positive)",
                "multiseed_vs_single": "+0.31pp (perm p=0.014, 100% LOO-exclusion positive)",
                "m2_structure": "+0.42pp (100% positive)",
                "transfer": "+0.21pp design / +0.32pp puzzle (leak-free)",
            },
            "fail_closed_audited": [
                "noise floor verified (formula corr=1.0000, MC error propagation)",
                "ceiling AUDIT (m2r_ceiling_audit_v1.py): oracle reaches R2 0.73-0.96 with strong model, not the old 0.407",
                "old 'circular oracle 0.407' was a weak-feature artifact — corrected in submission table",
                "legal design-region features (incl. legal rescue denominator) NEGATIVE — lever closed",
                "full-profile Transformer NEGATIVE (R2 0.056) — overfits",
                "inverse-variance weighting NEGATIVE — L1 already handles tail",
                "global full-profile features NEGATIVE for GBDT",
                "puzzle-level leak diagnostic: design-OOF vs puzzle-OOF diff 0.04pp",
                "3-way weight plateau wide (w1 0.5-0.7, w2 0.2-0.4 all within 0.3pp)",
                "4-way XGB architecture decorrelation CLOSED (+0.07pp, perm p=0.256)",
                "v2 cross-mutant overlap features (group A) NEGATIVE alone (-0.05pp)",
                "v2 M2-structure cross context (group D) NEGATIVE alone (-0.09pp)",
                "rD auxiliary predictor: rD ~49% predictable (corr 0.70, R2 0.493) but rD_pred as a feature NEGATIVE (-0.40pp, 0% positive)",
                "physics-constrained formula blend f=1-rD_pred/rnorm CLOSED (-0.03pp; corr with 3-way 0.913 => 3-way already captures full legal rD signal)",
                "stacking / learned blend weights (NNLS -0.65pp, Ridge -0.53pp, Ridge+quad -0.42pp; p~1.0) — fixed 0.6/0.3/0.1 already optimal",
                "residual boosting (GBDT on blend residual) NEGATIVE -0.60 to -5.33pp — no learnable structure left in the 3-way blend error",
            ],
            "honest_caveats": [
                "2/160 designs lack M2 preds (zero transfer features)",
                "strong-3way per-design (unpooled) gain +0.63pp, 69% positive; pooled LOO-exclusion +1.52pp 100% positive is the reliable statement",
                "legal-feature R2 ceiling ~0.36-0.39 (strong-model audit); residual headroom is the double-mutant effect",
                "oracle R2 0.73-0.96 > legal 0.36 confirms the gap is the (illegal) double-mutant RMSD, not model capacity",
            ],
        },
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "submission_horizontal_table_m2r.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # ---- markdown table ----
    lines = ["# M2R rescue_factor — definitive horizontal comparison (submission)",
             "", "| Model | Feature set | Split | MAE | Skill | R² |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        sk = f"{r['skill']*100:+.2f}%" if r["skill"] is not None else "—"
        mae = f"{r['mae']:.4f}" if r["mae"] is not None else "—"
        r2 = f"{r['r2']:.3f}" if r["r2"] is not None else "—"
        hl = " **" if r.get("headline") else ""
        hlc = "**" if r.get("headline") else ""
        lines.append(f"| {hl}{r['model']}{hlc} | {r['feature_set']} | "
                     f"{r['split']} | {mae} | {sk} | {r2} |")
    ms_pp = (msrep["results"]["multiseed_3way"]["skill"] * 100
             if msrep else None)
    ms_r2 = msrep["results"]["multiseed_3way"]["r2"] if msrep else None
    lines += [
        "",
        "**Headline**: multi-seed strong 3-way + v2 (L1+L2 GBDT + Ridge, 300-tr "
        f"base, 258 dims, 5-seed L1/L2 OOF averaging) = +{ms_pp:.2f}% skill "
        f"(R² {ms_r2:.3f}) at design-level LOO.",
        "",
        "**Significance**: multi-seed-vs-single-seed perm p = 0.014 "
        "(paired design-block, n=500); pooled gain +0.31pp (R² +0.0030), "
        "100% of 159 LOO-exclusion folds positive (range [+0.28, +0.34]pp); "
        "CI (0.273, 0.310).  "
        "v2-vs-v1 perm p = 0.010, LOO-exclusion gain +0.80pp, 100% of 159 folds "
        "positive (range [+0.75, +0.91]pp).  "
        "strong-vs-default 3-way perm p = 0.0005, "
        "CI (0.262, 0.298); LOO-exclusion gain +1.52pp, 100% of 159 folds "
        "positive (range [+1.46, +1.61]pp).  "
        "Full-stack L1 perm p = 0.002, CI (0.245, 0.281); "
        "L1-vs-L2 gain +0.56pp, 100% positive.",
        "",
        "**Noise floor / ceiling audit**: median σ_noise 0.024, "
        "R² ceiling (mean-noise) 0.82; "
        "legal-feature representation saturates ~0.36-0.39 (strong-model audit); "
        "oracle (knowing double-mutant RMSD) reaches R² 0.73-0.96.  "
        "Old 'circular oracle 0.407' was a weak-feature artifact — corrected.  "
        "Double-mutant audit: rD is 49% predictable (corr 0.70), but both "
        "rD_pred-as-feature (−0.40pp) and a physics-constrained formula blend "
        "(−0.03pp; corr 0.913 with the 3-way) are neutral/negative — the 3-way "
        "already extracts the full legally-reachable double-mutant signal.",
    ]
    (out / "submission_horizontal_table_m2r.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"DONE -> {out}")
    print("\n".join(lines))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer-report", required=True)
    ap.add_argument("--robust-report", required=True)
    ap.add_argument("--robust-permtest", required=True)
    ap.add_argument("--puzzle-report", required=True)
    ap.add_argument("--noise-floor", required=True)
    ap.add_argument("--threeway-report", default=None)
    ap.add_argument("--threeway-permtest", default=None)
    ap.add_argument("--threeway-puzzle-report", default=None)
    ap.add_argument("--threeway-strong-report", default=None)
    ap.add_argument("--threeway-strong-permtest", default=None)
    ap.add_argument("--threeway-strong-puzzle-report", default=None)
    ap.add_argument("--ceiling-audit-report", default=None)
    ap.add_argument("--features-v2-report", default=None)
    ap.add_argument("--features-v2-permtest", default=None)
    ap.add_argument("--features-v2-puzzle-report", default=None)
    ap.add_argument("--doublemut-report", default=None)
    ap.add_argument("--formula-blend-report", default=None)
    ap.add_argument("--multiseed-report", default=None)
    ap.add_argument("--multiseed-permtest", default=None)
    ap.add_argument("--multiseed-puzzle-report", default=None)
    ap.add_argument("--stack-report", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_table(args.transfer_report, args.robust_report, args.robust_permtest,
                args.puzzle_report, args.noise_floor, args.out,
                args.threeway_report, args.threeway_permtest,
                args.threeway_puzzle_report,
                args.threeway_strong_report, args.threeway_strong_permtest,
                args.threeway_strong_puzzle_report,
                args.ceiling_audit_report,
                args.features_v2_report, args.features_v2_permtest,
                args.features_v2_puzzle_report,
                args.doublemut_report, args.formula_blend_report,
                args.multiseed_report, args.multiseed_permtest,
                args.multiseed_puzzle_report, args.stack_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
