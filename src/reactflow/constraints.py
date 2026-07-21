"""RNA 2D structure legality constraints and projection algorithms.

Representation
--------------
An RNA secondary structure of length ``L`` is represented by a symmetric matrix
``P in {0,1}^{L x L}``.  A legal 2D matching satisfies

    P_ij = P_ji,
    P_ii = 0,
    sum_j P_ij <= 1.

Additional chemistry constraints disallow non-canonical base pairs and loops
shorter than ``min_loop`` nucleotides.  The greedy projection in this module is
not a replacement for exact dynamic programming folding; it is a deterministic
projection layer for model logits during sampling and testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


CANONICAL_PAIRS = frozenset({("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")})
WOBBLE_PAIRS = frozenset({("G", "U"), ("U", "G")})


@dataclass(frozen=True)
class PairValidationResult:
    """Validation result for a pair matrix.

    Complexity: O(1) metadata plus O(V) violation strings.
    """

    valid: bool
    violations: Tuple[str, ...]
    pair_count: int


def _matrix_size(matrix: Sequence[Sequence[float]]) -> int:
    """Return square matrix size or raise.

    Complexity: O(L) to check row lengths.
    """

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("pair matrix must be square")
    return size


def is_allowed_pair(base_i: str, base_j: str, *, allow_wobble: bool = True) -> bool:
    """Return whether two nucleotides can form an allowed pair.

    Formula: the allowed set is ``C union W`` where
    ``C={AU,UA,GC,CG}`` and ``W={GU,UG}`` if wobble is enabled.

    Complexity: O(1).
    """

    pair = (base_i.upper(), base_j.upper())
    return pair in CANONICAL_PAIRS or (allow_wobble and pair in WOBBLE_PAIRS)


def would_cross(existing_pairs: Iterable[Tuple[int, int]], i: int, j: int) -> bool:
    """Check whether candidate pair ``(i,j)`` crosses any existing pair.

    For normalized pairs ``a<b`` and ``i<j``, a pseudoknot crossing exists iff
    ``a < i < b < j`` or ``i < a < j < b``.

    Complexity: O(P) where P is the number of accepted pairs.
    """

    if i > j:
        i, j = j, i
    for a, b in existing_pairs:
        if a > b:
            a, b = b, a
        if (a < i < b < j) or (i < a < j < b):
            return True
    return False


def validate_pair_matrix(
    sequence: str,
    matrix: Sequence[Sequence[float]],
    *,
    min_loop: int = 3,
    allow_wobble: bool = True,
    allow_pseudoknot: bool = True,
) -> PairValidationResult:
    """Validate matching, symmetry and chemical constraints.

    A violation is reported rather than fixed.  This separation keeps training
    code honest: projection can be applied during sampling, while validation can
    still expose illegal model outputs.

    Complexity: O(L^2 + P^2) time and O(P) memory, where P is the number of
    non-zero upper-triangle pairs.
    """

    sequence = sequence.upper()
    size = _matrix_size(matrix)
    violations: List[str] = []
    if len(sequence) != size:
        violations.append(f"sequence length {len(sequence)} differs from matrix size {size}")
    pairs: List[Tuple[int, int]] = []
    partner_count = [0 for _ in range(size)]

    for i in range(size):
        if matrix[i][i] not in (0, 0.0, False):
            violations.append(f"diagonal position ({i},{i}) must be zero")
        for j in range(i + 1, size):
            left = matrix[i][j]
            right = matrix[j][i]
            if abs(float(left) - float(right)) > 1e-9:
                violations.append(f"matrix is not symmetric at ({i},{j})")
            if float(left) > 0.5 or float(right) > 0.5:
                pairs.append((i, j))
                partner_count[i] += 1
                partner_count[j] += 1
                if j - i <= min_loop:
                    violations.append(f"pair ({i},{j}) violates min_loop={min_loop}")
                if i < len(sequence) and j < len(sequence):
                    if not is_allowed_pair(sequence[i], sequence[j], allow_wobble=allow_wobble):
                        violations.append(f"pair ({i},{j})={sequence[i]}-{sequence[j]} is not canonical/wobble")

    for index, count in enumerate(partner_count):
        if count > 1:
            violations.append(f"position {index} has {count} partners")

    if not allow_pseudoknot:
        for idx, (i, j) in enumerate(pairs):
            if would_cross(pairs[:idx] + pairs[idx + 1 :], i, j):
                violations.append(f"pair ({i},{j}) participates in pseudoknot crossing")
                break

    return PairValidationResult(valid=not violations, violations=tuple(violations), pair_count=len(pairs))


def allowed_pair_mask(
    sequence: str,
    *,
    min_loop: int = 3,
    allow_wobble: bool = True,
) -> Tuple[Tuple[bool, ...], ...]:
    """Build a legal candidate mask before matching projection.

    ``mask[i][j]=True`` iff ``i<j``, ``j-i>min_loop`` and the bases are allowed.

    Complexity: O(L^2) time and memory.
    """

    sequence = sequence.upper()
    size = len(sequence)
    mask: List[List[bool]] = [[False for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + min_loop + 1, size):
            if is_allowed_pair(sequence[i], sequence[j], allow_wobble=allow_wobble):
                mask[i][j] = True
                mask[j][i] = True
    return tuple(tuple(row) for row in mask)


def project_greedy_matching(
    sequence: str,
    scores: Sequence[Sequence[float]],
    *,
    min_loop: int = 3,
    allow_wobble: bool = True,
    allow_pseudoknot: bool = True,
    min_score: float = 0.0,
) -> Tuple[Tuple[int, ...], ...]:
    """Project dense pair scores to a legal binary matching.

    Algorithm:
    1. enumerate candidates ``(score_ij, i, j)`` satisfying chemistry and loop
       constraints;
    2. sort by descending score;
    3. accept a candidate if both endpoints are unused and, when
       ``allow_pseudoknot=False``, it does not cross accepted pairs.

    The procedure greedily approximates maximum-weight matching under RNA
    constraints.  It is deterministic and useful as a sampling projection layer.

    Complexity: O(L^2 log L) time from sorting and O(L^2) memory.
    """

    sequence = sequence.upper()
    size = _matrix_size(scores)
    if len(sequence) != size:
        raise ValueError("sequence length and score matrix size differ")

    candidates: List[Tuple[float, int, int]] = []
    for i in range(size):
        for j in range(i + min_loop + 1, size):
            score = float(scores[i][j])
            if score < min_score:
                continue
            if is_allowed_pair(sequence[i], sequence[j], allow_wobble=allow_wobble):
                candidates.append((score, i, j))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used = [False for _ in range(size)]
    accepted: List[Tuple[int, int]] = []
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for _, i, j in candidates:
        if used[i] or used[j]:
            continue
        if not allow_pseudoknot and would_cross(accepted, i, j):
            continue
        matrix[i][j] = 1
        matrix[j][i] = 1
        used[i] = True
        used[j] = True
        accepted.append((i, j))
    return tuple(tuple(row) for row in matrix)


def project_max_weight_nested(
    sequence: str,
    scores: Sequence[Sequence[float]],
    *,
    min_loop: int = 3,
    allow_wobble: bool = True,
    min_score: float = 0.0,
) -> Tuple[Tuple[int, ...], ...]:
    """Project dense pair scores to the *exact* maximum-weight nested matching.

    Unlike :func:`project_greedy_matching`, which is a fast but sub-optimal
    greedy heuristic, this routine solves the pseudoknot-free maximum-weight
    matching *exactly* with a Nussinov-style dynamic program.  Exactness is what
    makes energy guidance behave monotonically (see
    :func:`reactflow.thermo.guidance_eta_scan`): the greedy heuristic can raise
    the total energy when guidance perturbs local scores, whereas the exact
    optimizer cannot.

    Dynamic program
    ---------------
    Let ``s_kj`` be the (already guided) score of the legal candidate pair
    ``(k, j)`` and let ``W[i][j]`` be the maximum achievable score of a nested
    matching restricted to the sub-interval ``[i, j]``.  The recurrence is

        W[i][j] = max(
            W[i][j-1],                                        # j is unpaired
            max_{k in [i, j-min_loop-1], (k,j) legal, s_kj >= min_score}
                W[i][k-1] + s_kj + W[k+1][j-1]                # j pairs with k
        ),

    with the empty-interval convention ``W[a][b] = 0`` for ``a > b``.  The first
    branch leaves position ``j`` unpaired; the second closes a pair ``(k, j)``
    that splits the interval into the independent sub-problems ``[i, k-1]``
    (outside/left of the pair) and ``[k+1, j-1]`` (enclosed by the pair).
    Because a nested structure forbids crossings, these sub-problems do not
    interact, so the DP is exact.

    Determinism and ties
    ---------------------
    The base case for each ``[i, j]`` is "``j`` unpaired"; a pairing ``k`` only
    replaces the incumbent when it *strictly* improves the score
    (``cand > best + 1e-12``).  Therefore ties deterministically prefer (a) fewer
    pairs and (b) the smallest partner index ``k``.  The output is identical on
    every platform.

    Legality
    --------
    Only candidates that satisfy ``j - k > min_loop``,
    :func:`is_allowed_pair` and ``s_kj >= min_score`` are ever considered, so the
    reconstructed matching is legal by construction and never crosses (nested).
    :func:`validate_pair_matrix` with ``allow_pseudoknot=False`` returns
    ``valid=True`` for the output.

    Complexity: ``O(L^3)`` time (interval span x left end x split point) and
    ``O(L^2)`` memory, the standard Nussinov cost.  Traceback is iterative (an
    explicit stack) so it is safe for long sequences regardless of the Python
    recursion limit.
    """

    sequence = sequence.upper()
    size = _matrix_size(scores)
    if len(sequence) != size:
        raise ValueError("sequence length and score matrix size differ")
    if size == 0:
        return tuple()

    best_weight = [[0.0 for _ in range(size)] for _ in range(size)]
    # ``split[i][j]`` records the traceback decision for interval [i, j]:
    #   -1 -> position j is unpaired, recurse on [i, j-1];
    #    k -> j pairs with k, recurse on [i, k-1] and [k+1, j-1].
    split = [[-1 for _ in range(size)] for _ in range(size)]

    def interval_weight(a: int, b: int) -> float:
        """Return ``W[a][b]`` with the empty-interval convention ``a>b -> 0``.

        Formula: ``W[a,b] = 0`` when the interval is empty, otherwise the DP
        table value.  Complexity: O(1).
        """

        if a > b:
            return 0.0
        return best_weight[a][b]

    for span in range(1, size):
        for i in range(0, size - span):
            j = i + span
            best = best_weight[i][j - 1]  # j unpaired
            choice = -1
            for k in range(i, j - min_loop):
                if not is_allowed_pair(sequence[k], sequence[j], allow_wobble=allow_wobble):
                    continue
                candidate_score = float(scores[k][j])
                if candidate_score < min_score:
                    continue
                total = interval_weight(i, k - 1) + candidate_score + interval_weight(k + 1, j - 1)
                if total > best + 1e-12:
                    best = total
                    choice = k
            best_weight[i][j] = best
            split[i][j] = choice

    matrix = [[0 for _ in range(size)] for _ in range(size)]
    stack: List[Tuple[int, int]] = [(0, size - 1)]
    while stack:
        i, j = stack.pop()
        if i >= j:
            continue
        k = split[i][j]
        if k < 0:
            stack.append((i, j - 1))
        else:
            matrix[k][j] = 1
            matrix[j][k] = 1
            stack.append((i, k - 1))
            stack.append((k + 1, j - 1))
    return tuple(tuple(row) for row in matrix)


def unpaired_indicators(matrix: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    """Return ``u_i = 1 - sum_j P_ij`` for a binary/soft pair matrix.

    Values are clamped to [0, 1] so a noisy soft matrix cannot produce invalid
    reactivity probabilities.

    Complexity: O(L^2) time and O(L) memory.
    """

    size = _matrix_size(matrix)
    indicators: List[float] = []
    for i in range(size):
        paired_mass = sum(float(matrix[i][j]) for j in range(size) if j != i)
        indicators.append(min(1.0, max(0.0, 1.0 - paired_mass)))
    return tuple(indicators)


def edge_context_indicators(matrix: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    """Approximate helix-edge/fraying context ``e_i``.

    ``e_i`` is 1 when position i is paired but at least one sequence neighbor is
    unpaired.  This captures the common chemistry observation that paired bases
    near helix ends can remain partially reactive.

    Complexity: O(L^2) due to unpaired indicator computation.
    """

    size = _matrix_size(matrix)
    unpaired = unpaired_indicators(matrix)
    contexts: List[float] = []
    for i in range(size):
        paired = 1.0 - unpaired[i]
        left_unpaired = unpaired[i - 1] if i > 0 else 1.0
        right_unpaired = unpaired[i + 1] if i + 1 < size else 1.0
        contexts.append(1.0 if paired > 0.5 and max(left_unpaired, right_unpaired) > 0.5 else 0.0)
    return tuple(contexts)


def dotbracket_to_matrix(dotbracket: str) -> Tuple[Tuple[int, ...], ...]:
    """Parse simple nested dot-bracket notation into a pair matrix.

    The parser supports ``.()``, not pseudoknot bracket alphabets.  It raises on
    unbalanced brackets rather than guessing.

    Complexity: O(L) time and O(L^2) output memory.
    """

    size = len(dotbracket)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    stack: List[int] = []
    for index, char in enumerate(dotbracket):
        if char == "(":
            stack.append(index)
        elif char == ")":
            if not stack:
                raise ValueError(f"unmatched closing bracket at {index}")
            partner = stack.pop()
            matrix[index][partner] = 1
            matrix[partner][index] = 1
        elif char != ".":
            raise ValueError(f"unsupported dot-bracket character '{char}' at {index}")
    if stack:
        raise ValueError(f"unmatched opening bracket at {stack[-1]}")
    return tuple(tuple(row) for row in matrix)


def matrix_to_pairs(matrix: Sequence[Sequence[float]]) -> Tuple[Tuple[int, int], ...]:
    """Return sorted upper-triangle pairs from a matrix.

    Complexity: O(L^2).
    """

    size = _matrix_size(matrix)
    pairs: List[Tuple[int, int]] = []
    for i in range(size):
        for j in range(i + 1, size):
            if float(matrix[i][j]) > 0.5:
                pairs.append((i, j))
    return tuple(pairs)


def pairs_to_matrix(pairs: Iterable[Tuple[int, int]], size: int) -> Tuple[Tuple[int, ...], ...]:
    """Build a symmetric binary pair matrix from an explicit list of pairs.

    This is the inverse of :func:`matrix_to_pairs` and the bridge used to turn
    external base-pair lists (for example the ``structure: [[i, j], ...]`` field
    of the eFold/RNAndria JSON files) into the ``P in {0,1}^{L x L}`` matrices
    that the rest of the package consumes.

    Each input pair ``(i, j)`` sets ``P_ij = P_ji = 1``.  A pair with ``i == j``
    is rejected (the diagonal must stay zero, matching
    :func:`validate_pair_matrix`) and out-of-range indices raise, so silent
    truncation of malformed external data is impossible.  The routine does *not*
    enforce the matching cardinality ``sum_j P_ij <= 1`` -- that legality check
    remains the responsibility of :func:`validate_pair_matrix`, keeping parsing
    and validation separate.

    Complexity: O(L^2) to allocate the matrix plus O(P) to set the pairs, for a
    total of O(L^2 + P) time and O(L^2) memory.
    """

    if size < 0:
        raise ValueError("size must be non-negative")
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for pair in pairs:
        i, j = int(pair[0]), int(pair[1])
        if i == j:
            raise ValueError(f"pair ({i},{j}) lies on the diagonal")
        if not (0 <= i < size and 0 <= j < size):
            raise ValueError(f"pair ({i},{j}) is out of range for size {size}")
        matrix[i][j] = 1
        matrix[j][i] = 1
    return tuple(tuple(row) for row in matrix)
