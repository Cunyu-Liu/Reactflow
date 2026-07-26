#!/usr/bin/env python3
"""Generate significance_report.json comparing model vs ViennaRNA baseline.

Performs paired t-test and bootstrap confidence intervals on F1 scores
from multi-seed results.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


VIENNARNA_F1_IN_CLAN = 0.6819
VIENNARNA_F1_NOVEL_CLAN = 0.6822


def paired_t_test(x: list[float], y: float) -> dict:
    """One-sample t-test comparing x values against a constant y.

    Tests whether the mean of x is significantly different from y.
    """
    n = len(x)
    if n < 2:
        return {"t_stat": 0, "p_value": 1.0, "df": 0, "n": n, "significant": False}

    m = mean(x)
    s = stdev(x)
    if s == 0:
        return {"t_stat": float("inf") if m != y else 0, "p_value": 0.0 if m != y else 1.0, "df": n - 1, "n": n, "significant": m != y}

    t_stat = (m - y) / (s / math.sqrt(n))
    df = n - 1

    # Approximate p-value using normal distribution (good for df >= 30)
    # For small df, this is conservative
    from math import erf, sqrt
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))

    return {
        "t_stat": round(t_stat, 6),
        "p_value": round(p_value, 6),
        "df": df,
        "n": n,
        "mean": round(m, 6),
        "baseline": y,
        "mean_diff": round(m - y, 6),
        "significant": p_value < 0.05,
    }


def bootstrap_ci(values: list[float], n_bootstrap: int = 10000) -> dict:
    """Bootstrap 95% confidence interval for the mean."""
    if not values:
        return {"ci_low": 0, "ci_high": 0, "n_bootstrap": 0}

    import random
    n = len(values)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(values) for _ in range(n)]
        bootstrap_means.append(mean(sample))

    bootstrap_means.sort()
    ci_low = bootstrap_means[int(0.025 * n_bootstrap)]
    ci_high = bootstrap_means[int(0.975 * n_bootstrap)]

    return {
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "n_bootstrap": n_bootstrap,
    }


def generate_significance(artifacts_dir: Path) -> dict:
    """Generate significance report."""
    multiseed_path = artifacts_dir / "multiseed_results.json"
    if not multiseed_path.exists():
        return {"error": "multiseed_results.json not found", "tests": []}

    with open(multiseed_path) as f:
        multiseed = json.load(f)

    tests = []

    for cfg in multiseed.get("configs", []):
        tier = cfg.get("tier", "")
        f1_stats = cfg.get("f1_stats", {})
        f1_values = [r["mean_f1"] for r in cfg.get("individual_results", [])]

        baseline = VIENNARNA_F1_IN_CLAN if tier == "in_clan" else VIENNARNA_F1_NOVEL_CLAN

        # Paired t-test
        ttest = paired_t_test(f1_values, baseline)

        # Bootstrap CI
        boot_ci = bootstrap_ci(f1_values)

        tests.append({
            "test_name": f"model_vs_viennarna_{tier}",
            "config": cfg.get("config", ""),
            "tier": tier,
            "metric": "mean_f1",
            "model_mean": f1_stats.get("mean", 0),
            "model_std": f1_stats.get("std", 0),
            "model_n": f1_stats.get("n", 0),
            "baseline": baseline,
            "baseline_name": "ViennaRNA",
            "mean_diff": f1_stats.get("mean", 0) - baseline,
            "t_test": ttest,
            "bootstrap_ci": boot_ci,
            "beats_baseline": f1_stats.get("mean", 0) > baseline,
            "significant": ttest.get("significant", False),
            "p_value": ttest.get("p_value", 1.0),
        })

    # Overall verdict
    all_significant = all(t.get("significant", False) for t in tests) if tests else False
    all_beat = all(t.get("beats_baseline", False) for t in tests) if tests else False

    return {
        "schema_version": 1,
        "tests": tests,
        "overall": {
            "all_beat_baseline": all_beat,
            "all_significant": all_significant,
            "verdict": "PASS" if (all_beat and all_significant) else "FAIL",
            "num_tests": len(tests),
            "num_significant": sum(1 for t in tests if t.get("significant", False)),
            "num_beat": sum(1 for t in tests if t.get("beats_baseline", False)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate significance_report.json")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/c1_3"))
    args = parser.parse_args()

    report = generate_significance(args.artifacts_dir)

    output_path = args.artifacts_dir / "significance_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Written {output_path}")
    overall = report.get("overall", {})
    print(f"Verdict: {overall.get('verdict', 'UNKNOWN')}")
    print(f"Beat baseline: {overall.get('num_beat', 0)}/{overall.get('num_tests', 0)}")
    print(f"Significant: {overall.get('num_significant', 0)}/{overall.get('num_tests', 0)}")


if __name__ == "__main__":
    main()
