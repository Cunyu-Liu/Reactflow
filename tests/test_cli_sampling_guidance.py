import json

import pytest

from reactflow.cli import main
from reactflow.symbolic import run_all_symbolic_checks


def test_cli_sample_reports_full_legality_and_writes_heatmap(tmp_path, capsys):
    out_dir = tmp_path / "sample_out"
    exit_code = main(
        [
            "sample",
            "--epochs",
            "3",
            "--samples",
            "3",
            "--num-samples",
            "20",
            "--num-steps",
            "8",
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["num_samples"] == 20
    assert payload["legal_count"] == 20
    assert payload["legality_rate"] == 1.0
    assert len(payload["ensemble_unpaired_probability"]) == len(payload["sequence"])
    assert (out_dir / "ensemble_pairing_frequency.svg").exists()


def test_cli_sample_accepts_explicit_sequence_and_no_pseudoknot(tmp_path, capsys):
    out_dir = tmp_path / "sample_seq"
    exit_code = main(
        [
            "sample",
            "--sequence",
            "GGGAAACCC",
            "--epochs",
            "2",
            "--num-samples",
            "12",
            "--num-steps",
            "6",
            "--no-pseudoknot",
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["sequence"] == "GGGAAACCC"
    assert payload["legality_rate"] == 1.0


def test_cli_guidance_scan_exact_is_legal_and_monotone(tmp_path, capsys):
    out_dir = tmp_path / "scan_out"
    exit_code = main(
        [
            "guidance-scan",
            "--sequence",
            "UAGUUGUGCCGCAG",
            "--reference",
            "((((......))))",
            "--etas",
            "0.0,0.25,0.5,1.0,2.0",
            "--epochs",
            "3",
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["exact_projection"] is True
    assert payload["legal_throughout"] is True
    assert payload["pair_energy_monotone_non_increasing"] is True
    assert len(payload["points"]) == 5
    energies = [point["pair_energy"] for point in payload["points"]]
    assert energies == sorted(energies, reverse=True)  # non-increasing
    assert (out_dir / "guidance_eta_scan.svg").exists()


def test_cli_guidance_scan_greedy_flag_runs(tmp_path, capsys):
    out_dir = tmp_path / "scan_greedy"
    exit_code = main(
        [
            "guidance-scan",
            "--sequence",
            "GGGAAACCC",
            "--etas",
            "0.0,1.0",
            "--greedy",
            "--epochs",
            "2",
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code in (0, 1)  # greedy has no monotonicity guarantee
    assert payload["exact_projection"] is False
    assert payload["legal_throughout"] is True


def test_cli_guidance_scan_defaults_to_pilot_sequence_and_structure(tmp_path, capsys):
    # No --sequence and no --reference: the handler falls back to the first pilot
    # sample and uses its ground-truth structure as the F1 reference.
    out_dir = tmp_path / "scan_default"
    exit_code = main(
        [
            "guidance-scan",
            "--etas",
            "0.0,1.0",
            "--epochs",
            "2",
            "--samples",
            "3",
            "--output-dir",
            str(out_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["sequence"]  # a non-empty pilot sequence was selected
    assert all(point["f1_to_reference"] is not None for point in payload["points"])
    assert (out_dir / "guidance_eta_scan.svg").exists()


def test_cli_verify_symbolic_includes_c4_checks(capsys):
    pytest.importorskip("sympy")

    assert main(["verify-symbolic"]) == 0
    output = capsys.readouterr().out

    assert "thermo_mse_gradient" in output
    assert "thermo_kl_gradient" in output
    assert "guidance_monotonicity_exchange" in output


def test_new_symbolic_checks_have_zero_residuals():
    pytest.importorskip("sympy")

    results = run_all_symbolic_checks()

    for name in ("thermo_mse_gradient", "thermo_kl_gradient", "guidance_monotonicity_exchange"):
        residuals = {k: v for k, v in results[name].items() if k.startswith("residual")}
        assert residuals, f"{name} produced no residual entries"
        assert all(value == "0" for value in residuals.values()), (name, residuals)
