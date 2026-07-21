import csv
import json
import math
import random

import pytest

from reactflow.cli import main
from reactflow.constraints import dotbracket_to_matrix
from reactflow.estimators import (
    gumbel_softmax_partner_probabilities,
    hard_partner_from_relaxed,
    mean_field_expected_reactivity,
    monte_carlo_expected_reactivity,
)
from reactflow.metrics import f1_score, matthews_corrcoef, mean_absolute_error, pair_confusion
from reactflow.reactivity import ReactivityForwardOperator
from reactflow.symbolic import run_all_symbolic_checks
from reactflow.visualization import write_pair_heatmap_svg, write_profile_overlay_svg, write_training_curves_svg


def test_mean_field_estimator_matches_forward_operator():
    sequence = "ACGU"
    q = (1.0, 0.0, 1.0, 0.0)

    estimated = mean_field_expected_reactivity(sequence, q, probe="2A3")
    direct = ReactivityForwardOperator().from_expectations(sequence, q, None, "2A3")

    assert estimated == direct


def test_monte_carlo_estimator_averages_structure_forward_values():
    sequence = "GGGAAACCC"
    paired = dotbracket_to_matrix("(((...)))")
    unpaired = tuple(tuple(0 for _ in sequence) for _ in sequence)

    estimated = monte_carlo_expected_reactivity(sequence, (paired, unpaired), probe="2A3")

    assert len(estimated) == len(sequence)
    assert estimated[3] > estimated[0]
    with pytest.raises(ValueError, match="at least one"):
        monte_carlo_expected_reactivity(sequence, (), probe="2A3")
    with pytest.raises(ValueError, match="length"):
        monte_carlo_expected_reactivity(sequence, (((0,),),), probe="2A3")


def test_gumbel_softmax_and_hard_partner_are_valid_distributions():
    probs = gumbel_softmax_partner_probabilities((0.0, 1.0, 2.0), temperature=0.5, rng=random.Random(1))
    hard = hard_partner_from_relaxed(probs)

    assert sum(probs) == pytest.approx(1.0)
    assert sum(hard) == 1
    assert len(hard) == 3
    with pytest.raises(ValueError, match="positive"):
        gumbel_softmax_partner_probabilities((1.0,), temperature=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        gumbel_softmax_partner_probabilities((), temperature=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        hard_partner_from_relaxed(())
    with pytest.raises(ValueError, match="positive"):
        hard_partner_from_relaxed((0.0, 0.0))


def test_pair_metrics_and_profile_mae():
    target = dotbracket_to_matrix("(((...)))")
    predicted = dotbracket_to_matrix("((.....))")

    confusion = pair_confusion(predicted, target)

    assert confusion == {"tp": 2, "fp": 0, "fn": 1, "tn": 33}
    assert f1_score(predicted, target) == pytest.approx(0.8)
    assert matthews_corrcoef(predicted, target) > 0
    assert mean_absolute_error((0.0, 1.0), (0.5, 1.5)) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="same shape"):
        pair_confusion(((0,),), ((0, 1),))
    with pytest.raises(ValueError, match="finite"):
        mean_absolute_error((float("nan"),), (1.0,))


def test_svg_visualization_tools_write_cross_platform_files(tmp_path):
    matrix = dotbracket_to_matrix("((..))")
    heatmap = write_pair_heatmap_svg(matrix, tmp_path / "heatmap.svg")
    overlay = write_profile_overlay_svg((0.1, 0.5, 0.2), (0.0, 0.4, 0.3), tmp_path / "overlay.svg")

    assert heatmap.read_text().startswith("<svg")
    assert "rect" in heatmap.read_text()
    assert "polyline" in overlay.read_text()
    with pytest.raises(ValueError, match="lengths must match"):
        write_profile_overlay_svg((1.0,), (1.0, 2.0), tmp_path / "bad.svg")


def test_symbolic_checks_return_zero_residuals():
    pytest.importorskip("sympy")

    results = run_all_symbolic_checks()

    assert results["affine_expectation"]["residual"] == "0"
    assert results["weighted_calibration"]["residual_alpha"] == "0"
    assert results["weighted_calibration"]["residual_gamma"] == "0"
    # C3 derivations: every residual across all new checks must simplify to zero.
    for name in (
        "softmax_cross_entropy_gradient",
        "softmax_jacobian",
        "mixture_path",
        "conditional_rate_master_equation",
        "reactivity_magnitude_gradient",
        "guidance_monotonicity_exchange",
        "adapter_gradient",
        "pearson_affine_invariance",
    ):
        residuals = {k: v for k, v in results[name].items() if k.startswith("residual")}
        assert residuals, f"{name} produced no residual entries"
        assert all(value == "0" for value in residuals.values()), (name, residuals)


def test_training_curves_svg_renders_all_series(tmp_path):
    series = {
        "total": [1.4, 1.2, 1.0, 0.9],
        "dfm": [1.2, 1.1, 1.0, 0.95],
        "mean_f1": [0.4, 0.5, 0.55, 0.6],
    }
    output = write_training_curves_svg(series, tmp_path / "curves.svg")
    text = output.read_text()

    assert text.startswith("<svg")
    assert text.count("polyline") == 3
    assert "total" in text and "mean_f1" in text


def test_training_curves_svg_validation(tmp_path):
    with pytest.raises(ValueError, match="at least one series"):
        write_training_curves_svg({}, tmp_path / "empty.svg")
    with pytest.raises(ValueError, match="same length"):
        write_training_curves_svg({"a": [1.0, 2.0], "b": [1.0]}, tmp_path / "bad.svg")
    with pytest.raises(ValueError, match="non-empty"):
        write_training_curves_svg({"a": []}, tmp_path / "zero.svg")


def test_cli_train_writes_artifacts_and_summary(tmp_path, capsys):
    out_dir = tmp_path / "train_out"

    assert main(["train", "--epochs", "3", "--samples", "3", "--output-dir", str(out_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["epochs"] == 3
    assert payload["samples"] == 3
    assert (out_dir / "training_curves.svg").exists()
    assert (out_dir / "reactivity_overlay.svg").exists()
    assert (out_dir / "pairing_marginals.svg").exists()
    assert payload["last"]["total"] <= payload["first"]["total"]


def test_cli_plot_and_symbolic_commands(tmp_path, capsys):
    pytest.importorskip("sympy")
    heatmap = tmp_path / "cli_heatmap.svg"
    overlay = tmp_path / "cli_overlay.svg"

    assert main(["plot-dotbracket", "((..))", str(heatmap)]) == 0
    assert main(
        [
            "plot-profiles",
            "--predicted",
            "0.1,0.2,0.3",
            "--target",
            "0.0,0.2,0.4",
            "--output",
            str(overlay),
        ]
    ) == 0
    assert main(["verify-symbolic"]) == 0
    output = capsys.readouterr().out

    assert heatmap.exists()
    assert overlay.exists()
    assert "affine_expectation" in output


def test_cli_validate_csv_outputs_validation_and_feature_json(tmp_path, capsys):
    path = tmp_path / "mini.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "sequence_id",
                "sequence",
                "experiment_type",
                "reads",
                "signal_to_noise",
                "reactivity_0001",
                "reactivity_0002",
            ]
        )
        writer.writerow(["s1", "AC", "DMS_MaP", "200", "2.0", "0.2", "0.5"])

    assert main(["validate-csv", str(path), "--limit", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation"]["valid_positions"] == 2
    assert payload["features"]["gc_fraction"] == 0.5


def test_visualization_rejects_non_square_and_all_missing_inputs(tmp_path):
    with pytest.raises(ValueError, match="square"):
        write_pair_heatmap_svg(((0.0, 1.0),), tmp_path / "bad_heatmap.svg")
    with pytest.raises(ValueError, match="finite"):
        write_profile_overlay_svg((math.nan,), (math.nan,), tmp_path / "bad_overlay.svg")
