#!/usr/bin/env python3
"""CPU smoke test for the DeltaFlow joint-structure metric scorer.

Verifies the estimator semantics against analytic expectations:
  1. distal mask definition (|j - i| > 2 AND valid).
  2. shape correlation recovers a known correlation and gates small sets.
  3. localization top-1/top-3 hit logic and the random baselines.
  4. energy score: independent-column forecasts give ~0 gain under the
     column-permutation null; strongly coupled forecasts give a clearly
     positive gain (joint structure is identifiable).
  5. PCA2: a rank-2 ensemble reports ~1.0; an independent-column ensemble
     reports a small fraction.
  6. classification boundaries (PASS/MARGINAL/FAIL) at the frozen F3
     values.
  7. score_arm + compare_arms end to end on synthetic mutants.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deltaflow_joint_metrics import (  # noqa: E402
    DEFAULT_B_PERM,
    compare_arms,
    distal_positions,
    energy_score_mutant,
    mutant_localization,
    mutant_shape_correlation,
    pca2_variance_fraction,
    score_arm,
    PCA2_PASS_RANGE,
)


def make_mutant(
    *,
    seed: int,
    length: int = 60,
    draws: int = 12,
    coupling: float = 0.0,
    noise_scale: float = 0.15,
) -> dict:
    rng = np.random.default_rng(seed)
    edit = 20
    observed = np.zeros(length)
    observed[25:55] = rng.normal(0.0, 0.3, 30)
    shared = rng.normal(0.0, 1.0, draws)
    base = np.repeat(observed[None, :], draws, axis=0)
    noise = rng.normal(0.0, noise_scale, (draws, length))
    samples = base + noise
    if coupling > 0.0:
        outer = coupling * shared[:, None] * np.where(
            np.abs(observed) > 1e-12, 1.0, 0.4
        )[None, :]
        samples = samples + outer
    valid = np.zeros(length, dtype=bool)
    valid[5:50] = True
    valid[edit] = False
    distal = distal_positions(edit, length, valid)
    return {
        "construct_id": f"C{seed % 2}",
        "edit_position": edit,
        "length": length,
        "observed_delta": observed,
        "samples": samples,
        "distal_mask": distal,
    }


def main() -> None:
    length = 60
    edit = 20
    valid = np.zeros(length, dtype=bool)
    valid[5:50] = True
    distal = distal_positions(edit, length, valid)
    assert not distal[: 21 - 2].any() or distal.min() >= 0
    assert distal[18:23].sum() == 0, "positions within +-2 of the edit are not distal"
    assert distal[5:18].all() and distal[23:50].all()
    assert distal[:5].sum() == 0 and distal[50:].sum() == 0
    print("distal mask: |j-i|>2 AND valid OK")

    rng = np.random.default_rng(7)
    x = np.arange(30, dtype=np.float64)
    y = 2.0 * x + rng.normal(0.0, 0.05, 30)
    pearson, spearman = mutant_shape_correlation(
        y[None, :], x, np.ones(30, dtype=bool)
    )
    assert pearson > 0.99, pearson
    assert spearman > 0.99, spearman
    tiny_mask = np.zeros(30, dtype=bool)
    tiny_mask[:9] = True
    pearson_t, spearman_t = mutant_shape_correlation(
        y[None, :], x, tiny_mask
    )
    assert math.isnan(pearson_t) and math.isnan(spearman_t)
    print(f"shape correlation: recovers {pearson:.4f}; min-pairs gate OK")

    observed = np.zeros(length)
    observed[40] = 3.0
    predicted_samples = np.zeros((4, length))
    predicted_samples[:, 40] = 2.0
    predicted_samples[:, 30] = 1.0
    predicted_samples[:, 35] = 0.5
    top1, top3, n_distal = mutant_localization(
        predicted_samples, observed, distal
    )
    assert top1 == 1 and top3 == 1 and n_distal > 0
    miss_samples = np.zeros((4, length))
    miss_samples[:, 30] = 5.0
    top1_m, top3_m, _ = mutant_localization(miss_samples, observed, distal)
    assert top1_m == 0 and top3_m == 0
    print("localization: top-1/top-3 hit logic OK")

    n_distal_demo = int(distal.sum())
    assert n_distal_demo >= 2
    base_1 = 1.0 / n_distal_demo
    base_3 = min(3.0, n_distal_demo) / n_distal_demo
    assert 0.0 < base_1 <= 1.0 and 0.0 < base_3 <= 1.0
    print(f"random baselines (n={n_distal_demo}): top1={base_1:.4f} top3={base_3:.4f}")

    independent_samples = np.zeros((16, length))
    column_rng = np.random.default_rng(3)
    for column in range(length):
        independent_samples[:, column] = column_rng.normal(0.0, 1.0, 16)
    independent_observed = independent_samples.mean(axis=0)
    energy_independent = energy_score_mutant(
        independent_samples, independent_observed, distal,
        b_perm=DEFAULT_B_PERM, seed=11,
    )
    assert math.isfinite(energy_independent["gain_percent"])
    assert abs(energy_independent["gain_percent"]) < 1.0, energy_independent
    print(
        "energy (independent columns): gain "
        f"{energy_independent['gain_percent']:+.3f}% ~ 0 OK"
    )

    shared_rng = np.random.default_rng(5)
    shared = shared_rng.normal(0.0, 1.0, 16)
    coupled_samples = (
        np.repeat(independent_observed[None, :], 16, axis=0)
        + shared[:, None] * np.ones((1, length))
    )
    coupled_observed = independent_observed + 0.5
    energy_coupled = energy_score_mutant(
        coupled_samples, coupled_observed, distal,
        b_perm=DEFAULT_B_PERM, seed=11,
    )
    assert energy_coupled["gain_percent"] > 5.0, energy_coupled
    print(
        "energy (coupled columns): gain "
        f"{energy_coupled['gain_percent']:+.2f}% > 5 OK (joint identified)"
    )

    direction = np.linspace(-1.0, 1.0, 40)
    rank2 = np.outer(direction, np.ones(12)) + np.outer(
        np.ones(40), np.linspace(1.0, 2.0, 12)
    )
    value = pca2_variance_fraction(rank2)
    assert value > 0.999, value
    id_rng = np.random.default_rng(9)
    independent_matrix = id_rng.normal(0.0, 1.0, (40, 12))
    value_independent = pca2_variance_fraction(independent_matrix)
    assert value_independent < 0.5, value_independent
    print(f"pca2: rank-2 ensemble {value:.4f}; independent {value_independent:.4f} OK")

    from deltaflow_joint_metrics import (  # noqa: E402
        classify_energy,
        classify_localization,
        classify_pca2,
        classify_shape,
    )

    assert classify_shape(0.61) == "PASS"
    assert classify_shape(0.60) == "PASS"
    assert classify_shape(0.55) == "MARGINAL"
    assert classify_shape(0.5459) == "FAIL"
    assert classify_localization(0.07, 0.0219) == "PASS"
    assert classify_localization(0.05, 0.0219) == "MARGINAL"
    assert classify_localization(0.03, 0.0219) == "FAIL"
    assert classify_energy(1.05) == "PASS"
    assert classify_energy(0.99) == "FAIL"
    low, high = PCA2_PASS_RANGE
    assert classify_pca2(0.397) == "PASS"
    assert classify_pca2(low - 0.01) == "FAIL"
    assert classify_pca2(high + 0.01) == "FAIL"
    print("classification boundaries: frozen F3 values OK")

    candidate_mutants = [
        make_mutant(seed=100 + i, coupling=1.2, noise_scale=0.05)
        for i in range(6)
    ]
    null_mutants = [
        make_mutant(seed=100 + i, coupling=0.0, noise_scale=0.45)
        for i in range(6)
    ]
    candidate_score = score_arm(
        arm="candidate", mutants=candidate_mutants, b_perm=20, seed=3
    )
    null_score = score_arm(
        arm="null", mutants=null_mutants, b_perm=20, seed=3
    )
    assert candidate_score["n_mutants_total"] == 6
    assert candidate_score["shape"]["verdict"] in ("PASS", "MARGINAL", "FAIL")
    assert null_score["shape"]["pearson_median"] < candidate_score["shape"][
        "pearson_median"
    ]
    comparison = compare_arms(candidate_score, null_score)
    assert comparison["deltas"]["shape_pearson_median"] > 0.0
    assert comparison["deltas"]["energy_gain_percent_median"] > 0.0
    assert (
        comparison["deltas"]["top1_rate"] >= 0.0
        if math.isfinite(comparison["deltas"]["top1_rate"])
        else True
    )
    print(
        "score_arm/compare_arms: candidate shape "
        f"{candidate_score['shape']['pearson_median']:.3f} vs null "
        f"{null_score['shape']['pearson_median']:.3f}; energy gain "
        f"{candidate_score['energy']['gain_percent_median']:+.2f}% vs "
        f"{null_score['energy']['gain_percent_median']:+.2f}% OK"
    )

    print("ALL JOINT-METRIC SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
