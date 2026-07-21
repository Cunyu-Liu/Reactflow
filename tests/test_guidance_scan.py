import math

import pytest

from reactflow.constraints import (
    dotbracket_to_matrix,
    matrix_to_pairs,
    project_greedy_matching,
    project_max_weight_nested,
    validate_pair_matrix,
)
from reactflow.thermo import (
    GuidanceScanPoint,
    energy_guided_scores,
    guidance_eta_scan,
    guidance_scan_is_monotone,
    structure_pair_energy,
    thermo_logit_gradient,
    thermo_unpaired_kl,
    thermo_unpaired_mse,
)
from reactflow.visualization import write_guidance_scan_svg


def _pair_weight(matrix, scores):
    """Sum the score-matrix weight of the pairs selected by ``matrix``."""

    return sum(scores[i][j] for i, j in matrix_to_pairs(matrix))


def test_nested_dp_finds_global_optimum_that_greedy_misses():
    # All-canonical 8-mer.  Greedy grabs the single locally-best pair (0,5)=3.0,
    # which crosses (2,7); the exact nested optimum is {(0,7),(2,5)} = 4.0.
    sequence = "GCGCGCGC"
    size = len(sequence)
    scores = [[0.0] * size for _ in range(size)]
    scores[0][5] = scores[5][0] = 3.0
    scores[2][7] = scores[7][2] = 2.9
    scores[0][7] = scores[7][0] = 2.0
    scores[2][5] = scores[5][2] = 2.0

    nested = project_max_weight_nested(sequence, scores, min_loop=1)
    greedy = project_greedy_matching(sequence, scores, min_loop=1, allow_pseudoknot=False, min_score=0.1)

    assert matrix_to_pairs(nested) == ((0, 7), (2, 5))
    assert _pair_weight(nested, scores) == pytest.approx(4.0)
    # The exact optimum is strictly better than the greedy heuristic here.
    assert _pair_weight(nested, scores) > _pair_weight(greedy, scores)


def test_nested_dp_output_is_legal_and_nested_over_random_matrices():
    import random

    rng = random.Random(2024)
    sequence = "GGGAAACCCUUUGGGCCC"
    size = len(sequence)
    for _ in range(50):
        scores = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                value = rng.uniform(-1.0, 2.0)
                scores[i][j] = scores[j][i] = value
        matrix = project_max_weight_nested(sequence, scores, min_loop=3)
        result = validate_pair_matrix(sequence, matrix, min_loop=3, allow_pseudoknot=False)
        assert result.valid


def test_nested_dp_respects_min_loop_and_size_checks():
    sequence = "GGGGCCCC"
    size = len(sequence)
    scores = [[5.0] * size for _ in range(size)]
    # min_loop = 3 forbids (3,4) style tight turns; only wide pairs may survive.
    matrix = project_max_weight_nested(sequence, scores, min_loop=3)
    for i, j in matrix_to_pairs(matrix):
        assert j - i > 3
    assert project_max_weight_nested("", []) == ()
    with pytest.raises(ValueError, match="size differ"):
        project_max_weight_nested("AC", ((0.0,),))


def test_structure_pair_energy_sums_only_pair_free_energies():
    sequence = "GGGAAACCC"
    matrix = dotbracket_to_matrix("(((...)))")
    # Three G-C pairs at -3.0 each, no loop/crossing bookkeeping.
    assert structure_pair_energy(sequence, matrix) == pytest.approx(-9.0)
    assert structure_pair_energy(sequence, tuple(tuple(0 for _ in sequence) for _ in sequence)) == 0.0


def test_structure_pair_energy_rejects_disallowed_pairs():
    # A forced A-C pair is chemically disallowed (energy = +inf).
    matrix = ((0, 0, 0, 0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (1, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="disallowed"):
        structure_pair_energy("ACGUC", matrix)


def test_guidance_eta_scan_pair_energy_is_monotone_non_increasing_exact():
    # Competitive-stem sequence: data-optimal (eta=0) pairing differs from the
    # thermodynamic minimum, so the pairing energy strictly steps down with eta.
    sequence = "UAGUUGUGCCGCAG"
    size = len(sequence)
    scores = [[0.0] * size for _ in range(size)]
    # Reward a weaker but data-supported outer stem so eta=0 is not already MFE.
    scores[0][11] = scores[11][0] = 1.0
    scores[1][10] = scores[10][1] = 1.0
    etas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

    points = guidance_eta_scan(sequence, scores, etas, exact=True)

    assert [p.eta for p in points] == etas
    assert all(p.legal for p in points)
    assert all(p.crossing_count == 0 for p in points)
    # Pairing energy must never rise as eta increases (monotonicity theorem).
    for previous, current in zip(points, points[1:]):
        assert current.pair_energy <= previous.pair_energy + 1e-9
    assert guidance_scan_is_monotone(points)
    # The scan must actually move: eta should reduce the pairing energy overall.
    assert points[-1].pair_energy < points[0].pair_energy


def test_guidance_scan_is_monotone_detects_violations_and_illegality():
    ok = (
        GuidanceScanPoint(0.0, True, 2, -1.0, -1.0, 0, None),
        GuidanceScanPoint(1.0, True, 3, -2.0, -2.0, 0, None),
    )
    assert guidance_scan_is_monotone(ok)

    rising = (
        GuidanceScanPoint(0.0, True, 2, -2.0, -2.0, 0, None),
        GuidanceScanPoint(1.0, True, 1, -1.0, -1.0, 0, None),  # energy rose
    )
    assert not guidance_scan_is_monotone(rising)

    illegal = (GuidanceScanPoint(0.0, False, 0, 0.0, 0.0, 0, None),)
    assert not guidance_scan_is_monotone(illegal)

    with pytest.raises(ValueError, match="non-empty"):
        guidance_scan_is_monotone(())


def test_guidance_eta_scan_reports_f1_and_validates_shapes():
    sequence = "GGGAAACCC"
    reference = dotbracket_to_matrix("(((...)))")
    size = len(sequence)
    scores = [[0.0] * size for _ in range(size)]
    points = guidance_eta_scan(sequence, scores, [0.0, 1.0], reference=reference, exact=True)

    assert all(p.f1_to_reference is not None for p in points)
    assert all(0.0 <= p.f1_to_reference <= 1.0 for p in points)
    with pytest.raises(ValueError, match="size must match"):
        guidance_eta_scan(sequence, [[0.0]], [1.0])
    with pytest.raises(ValueError, match="reference matrix shape"):
        guidance_eta_scan(sequence, scores, [1.0], reference=[[0.0]])
    with pytest.raises(ValueError, match="at least one value"):
        guidance_eta_scan(sequence, scores, [])


def test_greedy_projection_scan_can_break_monotonicity():
    # Documented negative control: the greedy heuristic is not a global optimizer,
    # so its pairing energy need not be monotone in eta.  We only assert the scan
    # runs and stays legal; the exact path is the one with the guarantee.
    sequence = "UAGUUGUGCCGCAG"
    size = len(sequence)
    scores = [[0.0] * size for _ in range(size)]
    scores[0][11] = scores[11][0] = 1.0
    scores[1][10] = scores[10][1] = 1.0

    points = guidance_eta_scan(sequence, scores, [0.0, 0.25, 0.5, 1.0], exact=False)

    assert all(p.legal for p in points)


def test_energy_guided_scores_follows_boltzmann_shift():
    sequence = "GGGAAACCC"
    size = len(sequence)
    scores = [[0.0] * size for _ in range(size)]
    guided = energy_guided_scores(sequence, scores, eta=1.0, temperature_kelvin=310.15)

    # s'_ij = s_ij - eta * e(x_i,x_j) / (R T); G-C pair energy is -3.0.
    rt = 0.00198720425864083 * 310.15
    assert guided[0][8] == pytest.approx(0.0 - (-3.0) / rt)
    assert guided[3][4] == float("-inf")  # inside min_loop -> masked


def test_thermo_unpaired_losses_match_closed_form():
    q = (0.8, 0.2, 0.5)
    t = (0.6, 0.3, 0.5)

    mse = thermo_unpaired_mse(q, t)
    expected_mse = ((0.8 - 0.6) ** 2 + (0.2 - 0.3) ** 2 + 0.0) / 3
    assert mse == pytest.approx(expected_mse)

    kl = thermo_unpaired_kl(q, t)
    expected_kl = 0.0
    for qi, ti in zip(q, t):
        if ti > 0.0:
            expected_kl += ti * math.log(ti / qi)
        if ti < 1.0:
            expected_kl += (1 - ti) * math.log((1 - ti) / (1 - qi))
    expected_kl /= 3
    assert kl == pytest.approx(expected_kl)

    with pytest.raises(ValueError, match="equal length"):
        thermo_unpaired_mse((0.1,), (0.1, 0.2))
    with pytest.raises(ValueError, match="at least one"):
        thermo_unpaired_kl((), ())


def _softmax(row):
    exps = [math.exp(v) for v in row]
    total = sum(exps)
    return [v / total for v in exps]


@pytest.mark.parametrize("mode", ["mse", "kl"])
def test_thermo_logit_gradient_matches_finite_difference(mode):
    logits = [[0.4, -0.1, 0.2], [-0.3, 0.5, 0.0]]
    marginals = [_softmax(row) for row in logits]
    target = (0.7, 0.35)
    lam = 1.3

    analytic = thermo_logit_gradient(marginals, target, lam, mode=mode)

    def loss(rows):
        m = [_softmax(r) for r in rows]
        q = [row[0] for row in m]
        if mode == "mse":
            return lam * thermo_unpaired_mse(q, target)
        return lam * thermo_unpaired_kl(q, target)

    eps = 1e-6
    for i in range(2):
        for k in range(3):
            plus = [list(r) for r in logits]
            minus = [list(r) for r in logits]
            plus[i][k] += eps
            minus[i][k] -= eps
            numeric = (loss(plus) - loss(minus)) / (2 * eps)
            assert analytic[i][k] == pytest.approx(numeric, abs=1e-6)


def test_thermo_logit_gradient_rejects_bad_mode_and_shape():
    marginals = [[0.5, 0.3, 0.2]]
    with pytest.raises(ValueError, match="mse.*kl|mode"):
        thermo_logit_gradient(marginals, (0.5,), 1.0, mode="huber")
    with pytest.raises(ValueError, match="length must match"):
        thermo_logit_gradient(marginals, (0.5, 0.5), 1.0)


def test_write_guidance_scan_svg_renders_per_series_bands(tmp_path):
    etas = [0.0, 0.5, 1.0]
    series = {
        "pair_energy": [-2.0, -4.0, -6.0],
        "pair_count": [2.0, 3.0, 4.0],
        "f1": [0.5, 0.5, 0.5],  # constant series -> mid-line, [min,max] shown
    }
    output = write_guidance_scan_svg(etas, series, tmp_path / "scan.svg")
    text = output.read_text()

    assert text.startswith("<svg")
    assert text.count("polyline") == 3
    assert "pair_energy" in text and "pair_count" in text and "f1" in text
    # Per-series legend must expose the true magnitude range.
    assert "[-6.00,-2.00]" in text


def test_write_guidance_scan_svg_validation(tmp_path):
    with pytest.raises(ValueError, match="etas must be non-empty"):
        write_guidance_scan_svg([], {"a": []}, tmp_path / "a.svg")
    with pytest.raises(ValueError, match="at least one series"):
        write_guidance_scan_svg([0.0, 1.0], {}, tmp_path / "b.svg")
    with pytest.raises(ValueError, match="same length as etas"):
        write_guidance_scan_svg([0.0, 1.0], {"a": [1.0]}, tmp_path / "c.svg")
