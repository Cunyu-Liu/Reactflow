"""Masked continuous-time-Markov-chain (CTMC) sampler for RNA 2D structures.

This module realizes the *generative* half of discrete flow matching for the
ReactFlow denoiser.  Training (see :mod:`reactflow.train`) fits the per-position
denoiser posterior ``p^theta_{1|t}(x1 | x_t)``; sampling integrates the induced
CTMC from the source distribution at ``t = 0`` to the data distribution at
``t = 1`` and then projects the per-position categorical draw onto a *legal*
joint secondary structure.

State space
-----------
For a sequence of length ``L`` every position ``i`` carries a partner class in
``{0, 1, ..., L}`` (``K = L + 1`` classes): class ``0`` is *unpaired* and class
``j + 1`` means *paired with sequence index ``j``*.  The generative model is
per-position factorized; global legality (symmetry, at most one partner,
canonical/wobble chemistry, minimum loop, optional no-pseudoknot) is restored by
the greedy projection of :mod:`reactflow.constraints` at the terminal time.

Generative CTMC
---------------
The unconditional sampling rate marginalizes the Campbell conditional rate over
the denoiser posterior (Campbell et al., ICML 2024; Gat et al., NeurIPS 2024):

    R^theta_t(z -> j) = sum_{x1} p^theta_{1|t}(x1 | x_t = z) * R*_t(z -> j | x1),

where ``R*_t`` is :func:`reactflow.dfm.conditional_rate_matrix`.  Given a single
current state ``z`` this is exactly :func:`reactflow.dfm.posterior_transition_rates`.
A continuous-time Markov jump process with generator ``R^theta`` transports the
uniform source to the model's data distribution as ``t: 0 -> 1``.

Euler transition kernel
-----------------------
Integrating the master equation ``d/dt p = p R`` for a state that is a point
mass at ``z`` over one small step ``dt`` gives the explicit-Euler transition row

    T_dt(z -> j) = delta_{zj} + dt * R^theta_t(z -> j).

Because ``R`` is a valid generator (rows sum to zero) this row is already
normalized: ``sum_j T_dt(z -> j) = 1 + dt * 0 = 1``.  For ``dt`` large enough to
make the self-transition ``1 + dt * R(z -> z)`` negative we clamp negatives to
zero and renormalize, mirroring the distribution-level
:func:`reactflow.dfm.euler_step_distribution`.  ``T_dt`` is then a categorical
distribution from which the next state is drawn.  This is the standard
tau-leaping (first-order) DFM sampler step.

Legality guarantee
------------------
The per-position draws at ``t = 1`` need not agree (two positions may pick the
same partner, or pick each other asymmetrically, or cross).  The final structure
is therefore obtained by *maximum-weight greedy projection*
(:func:`reactflow.constraints.project_greedy_matching`), whose output is legal by
construction: it only ever enumerates canonical/wobble candidates that respect
``min_loop``, greedily enforces the matching constraint (each index used once),
and optionally rejects crossings.  Consequently
:func:`reactflow.constraints.validate_pair_matrix` returns ``valid=True`` for
every sample and every seed; the test-suite verifies this empirically.

Ensemble semantics
------------------
ReactFlow's scientific claim is that a chemical-probing reactivity profile is a
first moment of the structure ensemble.  :func:`sample_structures` draws ``M``
independent trajectories and :func:`pairing_frequency_matrix` returns the Monte
Carlo estimate of the ensemble pairing probability
``\\hat P_{ij} = (1/M) sum_m S^{(m)}_{ij}``, an unbiased estimator of the model's
marginal pairing distribution.

The module uses only the Python standard library so the generative core is
deterministic, unit-testable and reproducible across Linux/Windows/macOS.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, List, Optional, Sequence, Tuple

from reactflow.constraints import (
    PairValidationResult,
    project_greedy_matching,
    validate_pair_matrix,
)
from reactflow.dfm import posterior_transition_rates, uniform_source
from reactflow.model import PairwiseDenoiser, marginal_pair_matrix
from reactflow.train import build_features


@dataclass(frozen=True)
class SampledStructure:
    """One CTMC sample and its legal projection.

    Attributes:
        sequence: the RNA sequence over ``{A,C,G,U}``.
        partner_classes: terminal per-position CTMC states ``x1_i`` (class 0 =
            unpaired, class ``j+1`` = paired to index ``j``); these are the raw,
            possibly-inconsistent draws before projection.
        pair_matrix: the legal binary symmetric structure after greedy
            projection.
        validation: the :class:`~reactflow.constraints.PairValidationResult` of
            ``pair_matrix``; ``validation.valid`` is guaranteed ``True``.
        num_steps: number of Euler integration steps used.

    Complexity: O(L^2) storage because ``pair_matrix`` is dense.
    """

    sequence: str
    partner_classes: Tuple[int, ...]
    pair_matrix: Tuple[Tuple[int, ...], ...]
    validation: PairValidationResult
    num_steps: int


def euler_transition_kernel(
    current_index: int,
    rate_row: Sequence[float],
    dt: float,
) -> Tuple[float, ...]:
    """Return the one-step Euler transition distribution ``T_dt(z -> .)``.

    Given the current state ``z = current_index`` and its sampling-rate row
    ``R(z -> .)`` (with ``R(z -> z) = -sum_{j != z} R(z -> j)``) the explicit
    Euler update of the master equation for a point mass at ``z`` is

        T_dt(z -> j) = 1[j = z] + dt * R(z -> j).

    Row-sum preservation: ``sum_j T_dt(z -> j) = 1 + dt * sum_j R(z -> j) = 1``
    because the rate row sums to zero.  When ``dt`` is large enough that the
    self-transition ``1 + dt * R(z -> z)`` turns negative, negative entries are
    clamped to zero and the row is renormalized (identical correction to
    :func:`reactflow.dfm.euler_step_distribution`, restricted to a single source
    row).  The result is a valid categorical distribution over ``K`` classes.

    Complexity: O(K).
    """

    num_classes = len(rate_row)
    if not 0 <= current_index < num_classes:
        raise ValueError("current_index out of range")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    transition: List[float] = []
    for j in range(num_classes):
        base = 1.0 if j == current_index else 0.0
        value = base + dt * float(rate_row[j])
        transition.append(max(0.0, value))
    total = sum(transition)
    if total <= 0.0:
        raise ValueError("euler transition kernel produced a non-positive mass")
    return tuple(value / total for value in transition)


def _categorical(probabilities: Sequence[float], rng: random.Random) -> int:
    """Draw one index from a categorical distribution by inverse-CDF sampling.

    ``draw ~ U[0,1)`` selects the first index whose cumulative probability meets
    or exceeds ``draw``.  The final index is returned if floating-point roundoff
    leaves a tiny residual mass, so the function never falls through.

    Complexity: O(K).
    """

    draw = rng.random()
    cumulative = 0.0
    for index, prob in enumerate(probabilities):
        cumulative += float(prob)
        if draw <= cumulative:
            return index
    return len(probabilities) - 1


def _projection_scores(
    states: Sequence[int],
    soft_matrix: Sequence[Sequence[float]],
    soft_weight: float,
) -> List[List[float]]:
    """Assemble the max-weight-matching score matrix for projection.

    The score of candidate pair ``(i, j)`` blends the discrete CTMC vote with a
    small soft prior taken from the terminal denoiser marginals:

        score_ij = vote_ij + soft_weight * P^soft_ij,
        vote_ij  = 0.5 * ( 1[state_i = j+1] + 1[state_j = i+1] ) in {0, 0.5, 1}.

    ``vote_ij`` measures how many of the two endpoints selected the pair during
    the CTMC rollout; the soft term (``P^soft`` from
    :func:`reactflow.model.marginal_pair_matrix`) breaks ties using the model's
    terminal confidence without overriding the sampled structure.  Different
    seeds yield different votes, which is what makes the projected ensemble
    diverse.

    Complexity: O(L^2).
    """

    size = len(states)
    scores = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            vote = 0.5 * (
                (1.0 if states[i] == j + 1 else 0.0)
                + (1.0 if states[j] == i + 1 else 0.0)
            )
            scores[i][j] = vote + soft_weight * float(soft_matrix[i][j])
    return scores


def sample_structure(
    model: PairwiseDenoiser,
    sequence: str,
    *,
    num_steps: int = 50,
    seed: int = 0,
    allow_pseudoknot: bool = True,
    soft_weight: float = 0.1,
    min_vote: float = 0.5,
    feature_builder: Optional[
        Callable[[str, float, Sequence[int]], Sequence[Sequence[float]]]
    ] = None,
) -> SampledStructure:
    """Draw one legal 2D structure by integrating the denoiser CTMC.

    Algorithm:

    1. initialize each position from the uniform source ``p0(z) = 1/K``;
    2. for ``s = 0 .. num_steps - 1`` at flow time ``t = s / num_steps``:
       build features from the current states, run the denoiser to obtain the
       posterior ``p^theta_{1|t}(. | x_t)``, form the sampling-rate row
       :func:`reactflow.dfm.posterior_transition_rates`, advance each position
       with :func:`euler_transition_kernel`, and resample the state;
    3. at the terminal time build the soft pairing matrix from the denoiser
       marginals and greedily project the CTMC votes onto a legal matching.

    The projection output is legal by construction, so the returned
    ``validation.valid`` is always ``True`` (checked in the tests across seeds).

    Args:
        model: a trained (or initialized) :class:`~reactflow.model.PairwiseDenoiser`.
        sequence: RNA sequence over ``{A,C,G,U}``.
        num_steps: number of explicit-Euler integration steps ``>= 1``.
        seed: master seed for the trajectory (initial states + all draws).
        allow_pseudoknot: whether the projection may keep crossing pairs.
        soft_weight: weight of the terminal soft prior in the projection score.
        min_vote: minimum score for a candidate to be projected; the default
            ``0.5`` requires at least one endpoint to have selected the pair.

    Complexity: ``O(num_steps * (L^3 + L^2 H^2))`` for the rollout (each step is
    ``L`` rate rows at ``O(L^2)`` plus one ``O(L^2 H^2)`` forward pass) plus
    ``O(L^2 log L)`` for the projection.
    """

    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    sequence = sequence.upper()
    size = len(sequence)
    if size == 0:
        raise ValueError("sequence must be non-empty")
    num_classes = size + 1
    source = uniform_source(num_classes)
    rng = random.Random(seed)
    dt = 1.0 / num_steps

    states = [_categorical(source, rng) for _ in range(size)]
    for step in range(num_steps):
        t = step / num_steps
        features = (
            feature_builder(sequence, t, states)
            if feature_builder is not None
            else build_features(sequence, t, states)
        )
        forward = model.forward(sequence, features)
        next_states: List[int] = []
        for i in range(size):
            rate_row = posterior_transition_rates(t, states[i], forward.marginals[i], source)
            transition = euler_transition_kernel(states[i], rate_row, dt)
            next_states.append(_categorical(transition, rng))
        states = next_states

    terminal_features = (
        feature_builder(sequence, 1.0, states)
        if feature_builder is not None
        else build_features(sequence, 1.0, states)
    )
    terminal_forward = model.forward(sequence, terminal_features)
    soft_matrix = marginal_pair_matrix(terminal_forward.marginals)
    scores = _projection_scores(states, soft_matrix, soft_weight)

    matrix = project_greedy_matching(
        sequence,
        scores,
        min_loop=model.min_loop,
        allow_wobble=model.allow_wobble,
        allow_pseudoknot=allow_pseudoknot,
        min_score=min_vote,
    )
    validation = validate_pair_matrix(
        sequence,
        matrix,
        min_loop=model.min_loop,
        allow_wobble=model.allow_wobble,
        allow_pseudoknot=allow_pseudoknot,
    )
    return SampledStructure(
        sequence=sequence,
        partner_classes=tuple(states),
        pair_matrix=matrix,
        validation=validation,
        num_steps=num_steps,
    )


def sample_structures(
    model: PairwiseDenoiser,
    sequence: str,
    *,
    num_samples: int,
    num_steps: int = 50,
    seed: int = 0,
    allow_pseudoknot: bool = True,
    soft_weight: float = 0.1,
    min_vote: float = 0.5,
    feature_builder: Optional[
        Callable[[str, float, Sequence[int]], Sequence[Sequence[float]]]
    ] = None,
) -> Tuple[SampledStructure, ...]:
    """Draw an ensemble of ``num_samples`` independent legal structures.

    Sample ``m`` uses seed ``seed + m`` so the ensemble is reproducible yet the
    trajectories differ.  Each element is guaranteed legal.

    Formula: ``S_m = sample_structure(model, sequence; seed + m)`` for
    ``m=0..num_samples-1``.

    Complexity: ``num_samples`` times :func:`sample_structure`.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    return tuple(
        sample_structure(
            model,
            sequence,
            num_steps=num_steps,
            seed=seed + offset,
            allow_pseudoknot=allow_pseudoknot,
            soft_weight=soft_weight,
            min_vote=min_vote,
            feature_builder=feature_builder,
        )
        for offset in range(num_samples)
    )


def pairing_frequency_matrix(structures: Sequence[SampledStructure]) -> Tuple[Tuple[float, ...], ...]:
    """Return the Monte Carlo ensemble pairing probability matrix.

    For an ensemble ``{S^{(m)}}`` the estimator is

        \\hat P_{ij} = (1/M) sum_{m=1}^{M} S^{(m)}_{ij},

    an unbiased estimate of the model's marginal probability that ``i`` and ``j``
    pair.  This first moment is the object the reactivity-consistency loss ties
    to chemical-probing data.

    Complexity: O(M L^2).
    """

    if not structures:
        raise ValueError("at least one structure is required")
    size = len(structures[0].pair_matrix)
    for structure in structures:
        if len(structure.pair_matrix) != size:
            raise ValueError("all structures must share the same length")
    frequency = [[0.0 for _ in range(size)] for _ in range(size)]
    for structure in structures:
        matrix = structure.pair_matrix
        for i in range(size):
            for j in range(size):
                frequency[i][j] += float(matrix[i][j])
    count = len(structures)
    return tuple(tuple(value / count for value in row) for row in frequency)


def ensemble_unpaired_probability(structures: Sequence[SampledStructure]) -> Tuple[float, ...]:
    """Return the ensemble unpaired probability per position.

    From the pairing-frequency matrix ``\\hat P`` the unpaired probability is

        \\hat q_i = 1 - sum_{j != i} \\hat P_{ij},

    clamped to ``[0, 1]``.  Because every sample is a legal matching, the sum has
    at most one non-zero term per sample, so ``\\hat q_i`` is well defined.  This
    is the ensemble first moment compared against measured reactivity.

    Complexity: O(M L^2).
    """

    frequency = pairing_frequency_matrix(structures)
    size = len(frequency)
    unpaired: List[float] = []
    for i in range(size):
        paired_mass = sum(frequency[i][j] for j in range(size) if j != i)
        unpaired.append(min(1.0, max(0.0, 1.0 - paired_mass)))
    return tuple(unpaired)
