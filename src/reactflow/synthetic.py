"""Deterministic synthetic pilot dataset for end-to-end C3 wiring.

WARNING - THIS IS SYNTHETIC PILOT DATA, NOT REAL EXPERIMENTAL DATA
------------------------------------------------------------------
The samples produced here are generated from a known nested structure and the
project's own affine forward operator.  They exist solely to prove that the
end-to-end training loop (L_DFM + reactivity-consistency) runs, decreases, and
does not collapse to a marginal-only degenerate solution.  They are **not** a
substitute for real chemical-probing data and must never be reported as such.

The real training corpus (documented with verifiable download links in
:mod:`reactflow.data`) is the public **Ribonanza2** release
(Kaggle ``rhijudas/ribonanza2-training-data``) and the Stanford **Ribonanza**
v1 competition data (Kaggle ``stanford-ribonanza-rna-folding``).  Those are
consumed in cycle C5 on a machine with network + storage for the 174 GB HDF5
shards; the pilot below is a stand-in for pure-CPU, offline reproducibility.

Construction
------------
For each sample we deterministically build:

1. a sequence that admits a clean nested hairpin/stem (canonical + wobble);
2. its ground-truth pair matrix ``S`` (dot-bracket parsed);
3. per-position clean partner-class targets ``x1_i`` for DFM;
4. a synthetic reactivity profile ``r = f(S)`` from the affine forward
   operator with a fixed, reproducible additive perturbation so the shape is
   correlated-but-not-identical to a perfect forward pass;
5. observation weights from probe/base validity.

All randomness is seeded, so the dataset is identical on Linux, Windows and
macOS.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional, Sequence, Tuple

from reactflow.constraints import dotbracket_to_matrix
from reactflow.reactivity import ReactivityForwardOperator, masked_unit_weights


COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


@dataclass(frozen=True)
class SyntheticSample:
    """One synthetic pilot record.

    Attributes:
        sequence: RNA sequence over ``{A,C,G,U}``.
        dotbracket: ground-truth nested structure in dot-bracket notation.
        pair_matrix: binary symmetric pair matrix of the structure.
        partner_classes: DFM clean targets ``x1_i`` (0 = unpaired, else j+1).
        reactivity: synthetic per-position reactivity for the given probe.
        weights: per-position observation weights (probe/base validity).
        probe: probe name (``DMS`` or ``2A3``).
        source_id: optional public-record identifier carried through real-data
            caches for reproducibility.
        family: optional Rfam family/clan label used by family-balanced training
            schedules.
        cluster: optional sequence-identity cluster label used by MMseqs-balanced
            training schedules.

    Complexity: O(L^2) storage because the pair matrix is dense.
    """

    sequence: str
    dotbracket: str
    pair_matrix: Tuple[Tuple[int, ...], ...]
    partner_classes: Tuple[int, ...]
    reactivity: Tuple[float, ...]
    weights: Tuple[float, ...]
    probe: str
    source_id: Optional[str] = None
    family: Optional[str] = None
    cluster: Optional[str] = None
    reactivity_source: str = "unknown"
    reactivity_error: Optional[Tuple[float, ...]] = None
    reactivity_snr: Optional[float] = None
    reactivity_quality: Optional[str] = None
    parent_source_id: Optional[str] = None
    window_start: Optional[int] = None
    window_end: Optional[int] = None
    parent_length: Optional[int] = None


def _build_stem_sequence(stem: int, loop: int, rng: random.Random) -> Tuple[str, str]:
    """Build a single hairpin: ``stem`` base pairs enclosing a ``loop`` region.

    The 5' arm is drawn randomly; the 3' arm is its reverse complement so every
    closing pair is canonical.  Returns ``(sequence, dotbracket)``.

    Complexity: O(stem + loop).
    """

    if stem <= 0 or loop < 3:
        raise ValueError("stem must be positive and loop at least 3")
    left = [rng.choice("ACGU") for _ in range(stem)]
    right = [COMPLEMENT[base] for base in reversed(left)]
    loop_bases = [rng.choice("ACGU") for _ in range(loop)]
    sequence = "".join(left + loop_bases + right)
    dotbracket = "(" * stem + "." * loop + ")" * stem
    return sequence, dotbracket


def _partner_classes(pair_matrix: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    """Convert a pair matrix to per-position partner classes.

    Class ``0`` denotes unpaired; class ``j+1`` denotes a pair with index ``j``.

    Complexity: O(L^2).
    """

    size = len(pair_matrix)
    classes: List[int] = []
    for i in range(size):
        partner = 0
        for j in range(size):
            if i != j and int(pair_matrix[i][j]) > 0:
                partner = j + 1
                break
        classes.append(partner)
    return tuple(classes)


def make_sample(
    *,
    stem: int,
    loop: int,
    probe: str = "2A3",
    seed: int = 0,
    noise: float = 0.05,
    operator: Optional[ReactivityForwardOperator] = None,
) -> SyntheticSample:
    """Create one deterministic synthetic sample.

    The reactivity is ``r_i = f_i(S) + delta_i`` with a bounded seeded
    perturbation ``delta_i in [-noise, noise]`` so the training target is not a
    trivially perfect copy of the operator output.

    Complexity: O(L^2) dominated by matrix operations.
    """

    rng = random.Random(seed)
    sequence, dotbracket = _build_stem_sequence(stem, loop, rng)
    pair_matrix = dotbracket_to_matrix(dotbracket)
    partner_classes = _partner_classes(pair_matrix)
    op = operator or ReactivityForwardOperator()
    clean = op.from_structure(sequence, pair_matrix, probe)
    reactivity = tuple(value + (rng.random() * 2.0 - 1.0) * noise for value in clean)
    weights = masked_unit_weights(sequence, probe, reactivity)
    return SyntheticSample(
        sequence=sequence,
        dotbracket=dotbracket,
        pair_matrix=pair_matrix,
        partner_classes=partner_classes,
        reactivity=reactivity,
        weights=weights,
        probe=probe,
    )


def make_dataset(
    *,
    count: int = 6,
    stem: int = 4,
    loop: int = 4,
    probe: str = "2A3",
    seed: int = 0,
    noise: float = 0.05,
) -> Tuple[SyntheticSample, ...]:
    """Create a deterministic list of synthetic samples.

    Each sample uses a distinct derived seed so sequences differ, while the whole
    dataset is reproducible from the single top-level ``seed``.

    Complexity: O(count * L^2).
    """

    if count <= 0:
        raise ValueError("count must be positive")
    operator = ReactivityForwardOperator()
    samples: List[SyntheticSample] = []
    for index in range(count):
        samples.append(
            make_sample(
                stem=stem,
                loop=loop,
                probe=probe,
                seed=seed * 1000 + index,
                noise=noise,
                operator=operator,
            )
        )
    return tuple(samples)
