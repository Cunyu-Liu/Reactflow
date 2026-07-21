"""Physics-inspired thermodynamic priors for RNA 2D structures.

Scope
-----
This module deliberately implements a lightweight, fully auditable surrogate for
Turner-style thermodynamics.  It is not advertised as a replacement for
ViennaRNA.  The purpose is to provide a complete differentiable/guidance-ready
prior for early ReactFlow pilots and tests.

Energy model
------------
For a structure ``S``:

    E(S|x) = sum_(i,j in S) e(x_i, x_j)
             + loop_penalty * number_of_pairs
             + crossing_penalty * number_of_crossings.

Canonical pairs have negative energies, so lower energy is preferred.  Guidance
adds ``-eta * DeltaE / RT`` to logits, increasing scores for energetically
favorable pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from reactflow.constraints import (
    is_allowed_pair,
    matrix_to_pairs,
    project_greedy_matching,
    project_max_weight_nested,
    validate_pair_matrix,
)
from reactflow.metrics import f1_score


PAIR_ENERGY_KCAL = {
    ("G", "C"): -3.0,
    ("C", "G"): -3.0,
    ("A", "U"): -2.0,
    ("U", "A"): -2.0,
    ("G", "U"): -1.0,
    ("U", "G"): -1.0,
}
R_KCAL_PER_MOL_K = 0.00198720425864083


def pair_energy(base_i: str, base_j: str) -> float:
    """Return surrogate free energy contribution for one base pair.

    Formula: ``e(AU)=e(UA)=-2``, ``e(GC)=e(CG)=-3``,
    ``e(GU)=e(UG)=-1`` kcal/mol; disallowed pairs return ``+inf``.

    Complexity: O(1).
    """

    return PAIR_ENERGY_KCAL.get((base_i.upper(), base_j.upper()), math.inf)


def crossing_count(pairs: Sequence[Tuple[int, int]]) -> int:
    """Count pseudoknot crossings among pairs.

    A crossing exists when ``a<i<b<j`` for pairs ``(a,b)`` and ``(i,j)``.

    Complexity: O(P^2), P = number of pairs.
    """

    normalized = [(min(i, j), max(i, j)) for i, j in pairs]
    crossings = 0
    for idx, (a, b) in enumerate(normalized):
        for i, j in normalized[idx + 1 :]:
            if (a < i < b < j) or (i < a < j < b):
                crossings += 1
    return crossings


def structure_energy(
    sequence: str,
    pair_matrix: Sequence[Sequence[float]],
    *,
    loop_penalty: float = 0.2,
    crossing_penalty: float = 5.0,
) -> float:
    """Compute surrogate structure energy.

    Invalid pair matrices raise ``ValueError`` instead of returning an arbitrary
    energy.  This is important for training diagnostics because illegal samples
    should be fixed by projection, not silently scored.

    Complexity: O(L^2 + P^2).
    """

    validation = validate_pair_matrix(sequence, pair_matrix, allow_pseudoknot=True)
    if not validation.valid:
        raise ValueError("; ".join(validation.violations))
    pairs = matrix_to_pairs(pair_matrix)
    energy = 0.0
    for i, j in pairs:
        contribution = pair_energy(sequence[i], sequence[j])
        if math.isinf(contribution):
            raise ValueError(f"disallowed pair ({i},{j})")
        energy += contribution + loop_penalty
    energy += crossing_penalty * crossing_count(pairs)
    return energy


def energy_guided_scores(
    sequence: str,
    scores: Sequence[Sequence[float]],
    *,
    eta: float = 1.0,
    temperature_kelvin: float = 310.15,
    min_loop: int = 3,
    allow_wobble: bool = True,
) -> Tuple[Tuple[float, ...], ...]:
    """Apply thermodynamic guidance to pair logits/scores.

    For candidate pair ``(i,j)``, adding the pair changes energy by
    ``DeltaE=e(x_i,x_j)`` under the surrogate model.  The guided score is

        s'_ij = s_ij - eta * DeltaE / (R T).

    Since favorable pairs have ``DeltaE<0``, their scores increase.  Illegal
    candidates receive ``-inf`` so downstream projection cannot select them.

    Complexity: O(L^2).
    """

    size = len(scores)
    if len(sequence) != size or any(len(row) != size for row in scores):
        raise ValueError("sequence length and score matrix size must match")
    rt = R_KCAL_PER_MOL_K * temperature_kelvin
    guided = [[float("-inf") for _ in range(size)] for _ in range(size)]
    for i in range(size):
        guided[i][i] = float("-inf")
        for j in range(i + 1, size):
            if j - i <= min_loop or not is_allowed_pair(sequence[i], sequence[j], allow_wobble=allow_wobble):
                continue
            delta = pair_energy(sequence[i], sequence[j])
            value = float(scores[i][j]) - eta * delta / rt
            guided[i][j] = value
            guided[j][i] = value
    return tuple(tuple(row) for row in guided)


def _pair_weight(sequence: str, i: int, j: int, temperature_kelvin: float) -> float:
    """Return Boltzmann weight for one legal pair.

    Complexity: O(1).
    """

    energy = pair_energy(sequence[i], sequence[j])
    if math.isinf(energy):
        return 0.0
    return math.exp(-energy / (R_KCAL_PER_MOL_K * temperature_kelvin))


def nested_partition_table(
    sequence: str,
    *,
    temperature_kelvin: float = 310.15,
    min_loop: int = 3,
    allow_wobble: bool = True,
) -> Tuple[Tuple[float, ...], ...]:
    """Compute a Nussinov-style partition table for nested structures.

    Recurrence for interval ``[i,j]``:

        Z[i,j] = Z[i,j-1] + sum_{k=i}^{j-min_loop-1}
                 Z[i,k-1] * w(k,j) * Z[k+1,j-1].

    Empty intervals have partition 1.  The recurrence enumerates all nested
    matchings under the simplified energy model.

    Complexity: O(L^3) time and O(L^2) memory.
    """

    sequence = sequence.upper()
    size = len(sequence)
    if size == 0:
        return tuple()
    z = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        z[i][i] = 1.0

    def interval(a: int, b: int) -> float:
        """Return partition value for an interval, with empty interval = 1.

        Formula: ``Z[a,b]=1`` for empty intervals and ``Z[a,b]=z[a][b]``
        otherwise.  Complexity: O(1).
        """

        if a > b:
            return 1.0
        return z[a][b]

    for span in range(1, size):
        for i in range(0, size - span):
            j = i + span
            total = interval(i, j - 1)
            for k in range(i, j - min_loop):
                if not is_allowed_pair(sequence[k], sequence[j], allow_wobble=allow_wobble):
                    continue
                weight = _pair_weight(sequence, k, j, temperature_kelvin)
                total += interval(i, k - 1) * weight * interval(k + 1, j - 1)
            z[i][j] = total
    return tuple(tuple(row) for row in z)


def sample_nested_structure(
    sequence: str,
    partition: Sequence[Sequence[float]],
    *,
    rng: random.Random,
    temperature_kelvin: float = 310.15,
    min_loop: int = 3,
    allow_wobble: bool = True,
) -> Tuple[Tuple[int, ...], ...]:
    """Sample one nested structure exactly from the DP distribution.

    The backtracking probabilities are proportional to the same terms used in
    ``nested_partition_table``.  Therefore samples are exact for the simplified
    nested model up to floating-point error.

    Complexity: O(L^2) expected time per sample and O(L^2) output memory.
    """

    sequence = sequence.upper()
    size = len(sequence)
    matrix = [[0 for _ in range(size)] for _ in range(size)]

    def interval(a: int, b: int) -> float:
        """Return cached partition value for sampling backtracking.

        Formula: ``Z[a,b]=1`` for empty intervals and ``partition[a][b]``
        otherwise.  Complexity: O(1).
        """

        if a > b:
            return 1.0
        return float(partition[a][b])

    def backtrack(i: int, j: int) -> None:
        """Recursively sample whether ``j`` is unpaired or paired to ``k``.

        Formula: each option is drawn with probability ``weight / sum(weight)``;
        paired options use ``Z[i,k-1] * exp(-E(k,j)/RT) * Z[k+1,j-1]``.
        Complexity: O(L^2) over the full recursive traceback.
        """

        if i >= j:
            return
        options: List[Tuple[Optional[int], float]] = [(None, interval(i, j - 1))]
        for k in range(i, j - min_loop):
            if is_allowed_pair(sequence[k], sequence[j], allow_wobble=allow_wobble):
                weight = interval(i, k - 1) * _pair_weight(sequence, k, j, temperature_kelvin) * interval(k + 1, j - 1)
                if weight > 0:
                    options.append((k, weight))
        total = sum(weight for _, weight in options)
        if total <= 0:
            return
        draw = rng.random() * total
        cumulative = 0.0
        chosen: Optional[int] = None
        for candidate, weight in options:
            cumulative += weight
            if draw <= cumulative:
                chosen = candidate
                break
        if chosen is None:
            backtrack(i, j - 1)
        else:
            matrix[chosen][j] = 1
            matrix[j][chosen] = 1
            backtrack(i, chosen - 1)
            backtrack(chosen + 1, j - 1)

    if size > 0:
        backtrack(0, size - 1)
    return tuple(tuple(row) for row in matrix)


def monte_carlo_unpaired_prior(
    sequence: str,
    *,
    samples: int = 256,
    seed: int = 0,
    temperature_kelvin: float = 310.15,
    min_loop: int = 3,
    allow_wobble: bool = True,
) -> Tuple[float, ...]:
    """Estimate thermodynamic unpaired probabilities with DP sampling.

    This is an unbiased Monte Carlo estimator for the simplified nested model:

        q_i ~= (1/M) sum_m 1[position i is unpaired in S_m].

    Complexity: O(L^3 + M L^2) time and O(L^2) memory.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    partition = nested_partition_table(
        sequence,
        temperature_kelvin=temperature_kelvin,
        min_loop=min_loop,
        allow_wobble=allow_wobble,
    )
    rng = random.Random(seed)
    counts = [0 for _ in sequence]
    for _ in range(samples):
        matrix = sample_nested_structure(
            sequence,
            partition,
            rng=rng,
            temperature_kelvin=temperature_kelvin,
            min_loop=min_loop,
            allow_wobble=allow_wobble,
        )
        for index, row in enumerate(matrix):
            if sum(row) == 0:
                counts[index] += 1
    return tuple(count / samples for count in counts)


_THERMO_EPS = 1e-12


def thermo_unpaired_mse(
    unpaired_probs: Sequence[float],
    target_unpaired: Sequence[float],
) -> float:
    """Return the mean-squared thermodynamic semi-supervision loss.

    With model unpaired probabilities ``q_i = pi_i[0]`` and a Turner-style
    unpaired prior ``t_i = q_i^Turner`` (from :func:`monte_carlo_unpaired_prior`)
    the loss is

        ell_thermo^MSE = (1/L) sum_i (q_i - t_i)^2.

    This is a soft, differentiable pull of the model's marginal unpaired
    probability toward the thermodynamic ensemble average.  It is *semi*
    supervised because ``t_i`` is a physics prior, not a measured label.

    Complexity: O(L).
    """

    if len(unpaired_probs) != len(target_unpaired):
        raise ValueError("unpaired_probs and target_unpaired must have equal length")
    if not unpaired_probs:
        raise ValueError("at least one position is required")
    size = len(unpaired_probs)
    total = 0.0
    for q, t in zip(unpaired_probs, target_unpaired):
        diff = float(q) - float(t)
        total += diff * diff
    return total / size


def thermo_unpaired_kl(
    unpaired_probs: Sequence[float],
    target_unpaired: Sequence[float],
) -> float:
    """Return the mean Bernoulli-KL thermodynamic semi-supervision loss.

    Treating each position as a Bernoulli variable ``unpaired ~ Bern(.)`` the
    loss is the forward KL from the Turner prior to the model,

        ell_thermo^KL = (1/L) sum_i KL( Bern(t_i) || Bern(q_i) )
                      = (1/L) sum_i [ t_i log(t_i / q_i)
                                      + (1 - t_i) log((1 - t_i)/(1 - q_i)) ],

    with the convention ``0 log 0 = 0``.  Forward KL (prior as reference) is
    mode-covering: it strongly penalizes the model assigning low unpaired
    probability where the thermodynamic prior expects the base to be unpaired.
    ``q_i`` is clamped to ``[eps, 1-eps]`` so the logarithms stay finite.

    Complexity: O(L).
    """

    if len(unpaired_probs) != len(target_unpaired):
        raise ValueError("unpaired_probs and target_unpaired must have equal length")
    if not unpaired_probs:
        raise ValueError("at least one position is required")
    size = len(unpaired_probs)
    total = 0.0
    for q_raw, t_raw in zip(unpaired_probs, target_unpaired):
        q = min(1.0 - _THERMO_EPS, max(_THERMO_EPS, float(q_raw)))
        t = float(t_raw)
        if t > 0.0:
            total += t * math.log(t / q)
        if t < 1.0:
            total += (1.0 - t) * math.log((1.0 - t) / (1.0 - q))
    return total / size


def thermo_logit_gradient(
    marginals: Sequence[Sequence[float]],
    target_unpaired: Sequence[float],
    lambda_thermo: float,
    *,
    mode: str = "mse",
) -> List[List[float]]:
    """Exact gradient of ``lambda_thermo * ell_thermo`` into every logit row.

    The chain rule factors through the model unpaired probability
    ``q_i = pi_i[0] = softmax(logit_i)[0]`` using the class-0 row of the softmax
    Jacobian ``d pi_i[0] / d logit_i[k] = pi_i[0] (1[k=0] - pi_i[k])``:

        d ell_thermo / d logit_i[k]
            = lambda_thermo * (1/L) * g_i * pi_i[0] * (1[k=0] - pi_i[k]),

    where the per-position sensitivity ``g_i = d ell_i / d q_i`` is

        MSE:  g_i = 2 (q_i - t_i),
        KL :  g_i = -t_i / q_i + (1 - t_i) / (1 - q_i).

    For the KL mode ``q_i`` is clamped to ``[eps, 1-eps]`` to keep the reciprocals
    finite, matching :func:`thermo_unpaired_kl`.  The structure of this gradient
    is identical to the reactivity-magnitude gradient
    (:func:`reactflow.train._reactivity_logit_gradient`); both are verified with
    finite differences and SymPy.

    Complexity: O(L * K).
    """

    if mode not in ("mse", "kl"):
        raise ValueError("mode must be 'mse' or 'kl'")
    size = len(marginals)
    if len(target_unpaired) != size:
        raise ValueError("target_unpaired length must match number of positions")
    grads: List[List[float]] = []
    for i in range(size):
        row = marginals[i]
        num_classes = len(row)
        q0 = float(row[0])
        t = float(target_unpaired[i])
        if mode == "mse":
            g_i = 2.0 * (q0 - t)
        else:
            q_clamped = min(1.0 - _THERMO_EPS, max(_THERMO_EPS, q0))
            g_i = -t / q_clamped + (1.0 - t) / (1.0 - q_clamped)
        coeff = float(lambda_thermo) * g_i / size
        grad_row = [0.0 for _ in range(num_classes)]
        for k in range(num_classes):
            indicator = 1.0 if k == 0 else 0.0
            grad_row[k] = coeff * q0 * (indicator - float(row[k]))
        grads.append(grad_row)
    return grads


def guided_projection(
    sequence: str,
    scores: Sequence[Sequence[float]],
    *,
    eta: float = 1.0,
    temperature_kelvin: float = 310.15,
    min_loop: int = 3,
    allow_wobble: bool = True,
    allow_pseudoknot: bool = True,
) -> Tuple[Tuple[int, ...], ...]:
    """Apply energy guidance then greedy legal projection.

    This is the inference-time hook described in the research plan:
    ``scores -> thermodynamic guidance -> hard legality projection``.

    Complexity: O(L^2 log L).
    """

    guided = energy_guided_scores(
        sequence,
        scores,
        eta=eta,
        temperature_kelvin=temperature_kelvin,
        min_loop=min_loop,
        allow_wobble=allow_wobble,
    )
    return project_greedy_matching(
        sequence,
        guided,
        min_loop=min_loop,
        allow_wobble=allow_wobble,
        allow_pseudoknot=allow_pseudoknot,
        min_score=0.0,
    )


def structure_pair_energy(sequence: str, pair_matrix: Sequence[Sequence[float]]) -> float:
    """Return the pure base-pairing energy ``sum_(i,j) e(x_i, x_j)``.

    This is the *guided* part of :func:`structure_energy`: it drops the
    ``loop_penalty`` and ``crossing_penalty`` book-keeping terms and keeps only
    the sum of pairwise free energies that energy guidance actually manipulates
    (``s'_ij = s_ij - eta * e(x_i,x_j) / RT``).  It is therefore the quantity for
    which the exact guidance scan is provably monotone in ``eta``.

    Complexity: O(L^2) to extract pairs, O(P) to sum.
    """

    pairs = matrix_to_pairs(pair_matrix)
    total = 0.0
    for i, j in pairs:
        contribution = pair_energy(sequence[i], sequence[j])
        if math.isinf(contribution):
            raise ValueError(f"disallowed pair ({i},{j})")
        total += contribution
    return total


@dataclass(frozen=True)
class GuidanceScanPoint:
    """One point of a thermodynamic-guidance ``eta`` sweep.

    Attributes:
        eta: the guidance strength used for this point.
        legal: whether the projected structure passed
            :func:`~reactflow.constraints.validate_pair_matrix`; guaranteed
            ``True`` by construction for both the exact and greedy projectors.
        pair_count: number of base pairs in the projected structure.
        structure_energy: full surrogate energy including loop/crossing penalties
            (:func:`structure_energy`).
        pair_energy: pure base-pairing energy :func:`structure_pair_energy`; this
            is the quantity guaranteed non-increasing in ``eta`` for the exact
            projector.
        crossing_count: number of pseudoknot crossings (0 for the nested exact
            projector).
        f1_to_reference: pair-level F1 against a supplied reference structure, or
            ``None`` when no reference was given.

    Complexity: O(1) summary storage.
    """

    eta: float
    legal: bool
    pair_count: int
    structure_energy: float
    pair_energy: float
    crossing_count: int
    f1_to_reference: Optional[float]


def guidance_eta_scan(
    sequence: str,
    scores: Sequence[Sequence[float]],
    etas: Sequence[float],
    *,
    reference: Optional[Sequence[Sequence[float]]] = None,
    temperature_kelvin: float = 310.15,
    min_loop: int = 3,
    allow_wobble: bool = True,
    loop_penalty: float = 0.2,
    crossing_penalty: float = 5.0,
    exact: bool = True,
) -> Tuple[GuidanceScanPoint, ...]:
    """Sweep the inference-time guidance strength ``eta`` and record the tradeoff.

    For each ``eta`` in ``etas`` the pipeline is

        scores --(energy guidance)--> s'_ij = s_ij - eta * e(x_i,x_j) / RT
               --(legal projection)--> nested structure S(eta),

    where the projector is the *exact* maximum-weight nested optimizer
    :func:`~reactflow.constraints.project_max_weight_nested` when ``exact=True``
    and the greedy heuristic :func:`~reactflow.constraints.project_greedy_matching`
    otherwise.  Every point records legality, pair count, full structure energy,
    pure pairing energy and (optionally) F1 to a reference.

    Monotonicity theorem (exact projector)
    --------------------------------------
    Write ``f(S) = sum_{(i,j) in S} s_ij`` and ``g(S) = sum_{(i,j) in S}
    e(x_i,x_j)`` (the pairing energy).  The projected structure is the exact
    maximizer ``S(eta) = argmax_S [ f(S) - (eta / RT) g(S) ]`` over the
    *eta-independent* set of legal nested structures.  For ``eta_1 < eta_2`` with
    optima ``S_1, S_2`` optimality gives

        f(S_1) - (eta_1/RT) g(S_1) >= f(S_2) - (eta_1/RT) g(S_2),
        f(S_2) - (eta_2/RT) g(S_2) >= f(S_1) - (eta_2/RT) g(S_1).

    Adding and simplifying yields ``(eta_2 - eta_1)/RT * (g(S_1) - g(S_2)) >= 0``,
    hence ``g(S_1) >= g(S_2)``: the pairing energy is **non-increasing** in
    ``eta``.  This is exactly the ``eta`` scan curve required by the research
    plan.  The greedy heuristic (``exact=False``) is *not* a global optimizer, so
    it can violate this and is provided only to demonstrate the gap.

    The scan uses ``min_score = -inf`` for the exact projector so the feasible
    set does not depend on ``eta`` (a prerequisite of the theorem); the exact DP
    still only adds a pair when it strictly improves the objective, so a floor is
    unnecessary.

    Complexity: ``O(E * L^3)`` for ``E`` values of ``eta`` (each ``eta`` costs an
    ``O(L^2)`` guidance pass plus an ``O(L^3)`` exact projection).
    """

    sequence = sequence.upper()
    size = len(scores)
    if len(sequence) != size or any(len(row) != size for row in scores):
        raise ValueError("sequence length and score matrix size must match")
    if reference is not None and (len(reference) != size or any(len(row) != size for row in reference)):
        raise ValueError("reference matrix shape must match the score matrix")
    if not etas:
        raise ValueError("etas must contain at least one value")

    points: List[GuidanceScanPoint] = []
    for eta in etas:
        guided = energy_guided_scores(
            sequence,
            scores,
            eta=float(eta),
            temperature_kelvin=temperature_kelvin,
            min_loop=min_loop,
            allow_wobble=allow_wobble,
        )
        if exact:
            matrix = project_max_weight_nested(
                sequence,
                guided,
                min_loop=min_loop,
                allow_wobble=allow_wobble,
                min_score=float("-inf"),
            )
            allow_pseudoknot = False
        else:
            matrix = project_greedy_matching(
                sequence,
                guided,
                min_loop=min_loop,
                allow_wobble=allow_wobble,
                allow_pseudoknot=False,
                min_score=0.0,
            )
            allow_pseudoknot = False
        validation = validate_pair_matrix(
            sequence,
            matrix,
            min_loop=min_loop,
            allow_wobble=allow_wobble,
            allow_pseudoknot=allow_pseudoknot,
        )
        pairs = matrix_to_pairs(matrix)
        total_energy = structure_energy(
            sequence,
            matrix,
            loop_penalty=loop_penalty,
            crossing_penalty=crossing_penalty,
        )
        pairing_energy = structure_pair_energy(sequence, matrix)
        crossings = crossing_count(pairs)
        f1 = float(f1_score(matrix, reference)) if reference is not None else None
        points.append(
            GuidanceScanPoint(
                eta=float(eta),
                legal=validation.valid,
                pair_count=validation.pair_count,
                structure_energy=total_energy,
                pair_energy=pairing_energy,
                crossing_count=crossings,
                f1_to_reference=f1,
            )
        )
    return tuple(points)


def guidance_scan_is_monotone(points: Sequence[GuidanceScanPoint], *, tol: float = 1e-9) -> bool:
    """Return whether a scan is legal throughout and non-increasing in pairing energy.

    The check encodes the two C4 acceptance criteria for the guidance sweep:
    (1) every projected structure is legal, and (2) the pairing energy
    ``g(S(eta))`` never increases as ``eta`` grows (the monotonicity theorem in
    :func:`guidance_eta_scan`).  A small tolerance ``tol`` absorbs floating-point
    noise.  The caller is expected to have sorted ``points`` by ascending
    ``eta``.

    Complexity: O(E).
    """

    if not points:
        raise ValueError("points must be non-empty")
    if not all(point.legal for point in points):
        return False
    for previous, current in zip(points, points[1:]):
        if current.pair_energy > previous.pair_energy + tol:
            return False
    return True
