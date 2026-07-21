"""Per-position RNA 2D denoiser with exact hand-written backpropagation.

Model
-----
Given a sequence of length ``L`` and per-position feature vectors
``feat_i in R^F`` (base identity, flow time ``t``, and the noised state ``x_t``),
the denoiser predicts, for each position ``i``, a categorical posterior over
``K = L + 1`` partner classes (class 0 = unpaired, class ``j+1`` = paired to
``j``).  The computation graph is fully explicit so every gradient below is
derived by hand and validated against finite differences in the test-suite.

Forward pass
------------
1. Hidden embedding (shared input projection + tanh):

       a_i = W feat_i + b,     h_i = tanh(a_i) in R^H.

2. Bilinear pair score (asymmetric ``M``) plus a canonical-compatibility term:

       s_ij = h_i^T M h_j + c_pair * compat_ij,     i != j.

3. Linear unpaired score:

       s_i^u = v . h_i + b_u.

4. Per-position logits over ``K`` classes with legality masking:

       logit_i[0]   = s_i^u,
       logit_i[j+1] = s_ij   if (i,j) is a legal pair else -inf,

   and marginals ``pi_i = softmax(logit_i)``.

Backward pass (exact)
---------------------
Given upstream gradients ``g_i = dL/dlogit_i`` (length ``K``) the parameter
gradients are, writing ``compat`` and legality as fixed masks:

* unpaired head:  dL/dv += g_i[0] h_i,   dL/db_u += g_i[0];
* pair head:      dL/dM += sum_{i != j} g_i[j+1] * (h_i h_j^T),
                  dL/dc_pair += sum_{i != j} g_i[j+1] compat_ij;
* hidden (query + key roles of the same ``h_i``):

      dL/dh_i = g_i[0] v
                + sum_{j != i} g_i[j+1] (M h_j)      # i as query in s_ij
                + sum_{j != i} g_j[i+1] (M^T h_j)     # i as key   in s_ji

* projection through tanh:  dL/da_i = dL/dh_i ⊙ (1 - h_i^2),
                            dL/dW  += dL/da_i feat_i^T,  dL/db += dL/da_i.

* input features (for external upstream producers such as the C5 frozen-encoder
  adapter):  dL/dfeat_i = W^T dL/da_i,   i.e.
  ``grad_feat_i[f] = sum_p dL/da_i[p] * W[p][f]``.  This is not a model
  parameter -- it is the signal a feature producer needs to train itself.

The key/query split is the only subtle point: because ``M`` is not symmetric,
``h_i`` enters both ``s_ij`` (as the left/query factor) and ``s_ji`` (as the
right/key factor), so its gradient accumulates from both.

Complexity
----------
Forward and backward are ``O(L^2 H^2)`` time (each of ``O(L^2)`` pairs costs an
``O(H^2)`` bilinear form) and ``O(L^2 + H^2)`` memory.  For pilot-scale ``L`` and
small ``H`` this is entirely tractable and, crucially, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Optional, Sequence, Tuple

from reactflow.constraints import is_allowed_pair
from reactflow.dfm import softmax


NEG_INF = -1.0e9


@dataclass
class DenoiserParameters:
    """Learnable parameters of :class:`PairwiseDenoiser`.

    Attributes are plain nested lists so gradients can be accumulated and SGD
    updates applied without any tensor dependency.

    Formula: hidden activations use ``h_i = tanh(W x_i + b)``; pair logits use a
    bilinear score ``h_i^T M h_j`` plus canonical-pair compatibility, and
    unpaired logits use ``v^T h_i + b_u``.  Complexity: O(HF + H^2) parameters.
    """

    input_weight: List[List[float]]  # W: H x F
    input_bias: List[float]  # b: H
    pair_matrix: List[List[float]]  # M: H x H
    pair_compat: float  # c_pair scalar
    unpaired_weight: List[float]  # v: H
    unpaired_bias: float  # b_u scalar

    @property
    def hidden_size(self) -> int:
        """Return hidden dimension ``H``.

        Formula: ``H = len(input_bias)``.  Complexity: O(1).
        """

        return len(self.input_bias)

    @property
    def feature_size(self) -> int:
        """Return feature dimension ``F``.

        Formula: ``F = input_weight.shape[1]`` when the matrix is non-empty.
        Complexity: O(1).
        """

        return len(self.input_weight[0]) if self.input_weight else 0

    @staticmethod
    def random_init(feature_size: int, hidden_size: int, *, seed: int = 0, scale: float = 0.2) -> "DenoiserParameters":
        """Deterministically initialize parameters with small uniform values.

        A fixed ``seed`` guarantees identical parameters across platforms, which
        is required for the reproducibility and cross-platform goals.

        Complexity: O(H*F + H^2).
        """

        if feature_size <= 0 or hidden_size <= 0:
            raise ValueError("feature_size and hidden_size must be positive")
        rng = random.Random(seed)

        def uniform() -> float:
            """Sample one symmetric initialization coefficient.

            Formula: ``u = (2 * U[0,1) - 1) * scale``.  Complexity: O(1).
            """

            return (rng.random() * 2.0 - 1.0) * scale

        input_weight = [[uniform() for _ in range(feature_size)] for _ in range(hidden_size)]
        input_bias = [0.0 for _ in range(hidden_size)]
        pair_matrix = [[uniform() for _ in range(hidden_size)] for _ in range(hidden_size)]
        unpaired_weight = [uniform() for _ in range(hidden_size)]
        return DenoiserParameters(
            input_weight=input_weight,
            input_bias=input_bias,
            pair_matrix=pair_matrix,
            pair_compat=0.5,
            unpaired_weight=unpaired_weight,
            unpaired_bias=0.0,
        )


@dataclass
class DenoiserGradients:
    """Gradient container mirroring :class:`DenoiserParameters`.

    ``grad_features`` is the gradient of the loss with respect to the per-position
    input feature vectors, ``dL/dfeat_i in R^F``.  It is *not* a model parameter
    gradient (so :func:`sgd_update` ignores it); it is the upstream signal that an
    external feature producer -- e.g. the C5 frozen-encoder ``FeatureAdapter`` --
    needs to backpropagate into its own parameters.  It defaults to ``None`` so
    all existing gradient constructions remain valid.

    Complexity: O(HF + H^2 + optional L F) storage.
    """

    input_weight: List[List[float]]
    input_bias: List[float]
    pair_matrix: List[List[float]]
    pair_compat: float
    unpaired_weight: List[float]
    unpaired_bias: float
    grad_features: Optional[List[List[float]]] = None


@dataclass
class ForwardResult:
    """Cached forward-pass tensors needed by the backward pass.

    Complexity: O(LK + LH + L^2) storage for K partner classes and H hidden units.
    """

    logits: Tuple[Tuple[float, ...], ...]
    marginals: Tuple[Tuple[float, ...], ...]
    hidden: Tuple[Tuple[float, ...], ...]
    legal_pair: Tuple[Tuple[bool, ...], ...]
    compat: Tuple[Tuple[float, ...], ...]


class PairwiseDenoiser:
    """A small, exactly-differentiable per-position partner-class denoiser.

    Formula: row ``i`` predicts categorical partner classes where class 0 is
    unpaired and class ``j+1`` means paired to position ``j``.  Legal pair logits
    are ``h_i^T M h_j + c_pair * compat(i,j)`` and illegal pairs are masked before
    softmax.  Complexity: O(L^2 H^2) for the dense pairwise forward/backward path.
    """

    def __init__(
        self,
        parameters: DenoiserParameters,
        *,
        min_loop: int = 3,
        allow_wobble: bool = True,
    ) -> None:
        """Store parameters and legality settings."""

        self.parameters = parameters
        self.min_loop = min_loop
        self.allow_wobble = allow_wobble

    def _legality(self, sequence: str) -> Tuple[Tuple[Tuple[bool, ...], ...], Tuple[Tuple[float, ...], ...]]:
        """Build the legal-pair mask and canonical-compatibility feature.

        ``legal[i][j]`` is True iff ``i != j``, ``|i-j| > min_loop`` and the
        bases can pair.  ``compat`` equals the same indicator as a float feature.

        Complexity: O(L^2).
        """

        sequence = sequence.upper()
        size = len(sequence)
        legal = [[False for _ in range(size)] for _ in range(size)]
        compat = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if i == j or abs(i - j) <= self.min_loop:
                    continue
                if is_allowed_pair(sequence[i], sequence[j], allow_wobble=self.allow_wobble):
                    legal[i][j] = True
                    compat[i][j] = 1.0
        return tuple(tuple(row) for row in legal), tuple(tuple(row) for row in compat)

    def forward(self, sequence: str, features: Sequence[Sequence[float]]) -> ForwardResult:
        """Run the forward pass and cache intermediates for backprop.

        Complexity: O(L^2 H^2).
        """

        params = self.parameters
        size = len(sequence)
        if len(features) != size:
            raise ValueError("features length must match sequence length")
        hidden_size = params.hidden_size
        feature_size = params.feature_size

        hidden: List[List[float]] = []
        for feat in features:
            if len(feat) != feature_size:
                raise ValueError("feature vector has wrong dimension")
            activation = []
            for row, bias in zip(params.input_weight, params.input_bias):
                total = bias
                for weight, value in zip(row, feat):
                    total += weight * float(value)
                activation.append(math.tanh(total))
            hidden.append(activation)

        legal, compat = self._legality(sequence)

        logits: List[List[float]] = []
        marginals: List[Tuple[float, ...]] = []
        for i in range(size):
            row = [0.0 for _ in range(size + 1)]
            unpaired = params.unpaired_bias
            for value, weight in zip(hidden[i], params.unpaired_weight):
                unpaired += weight * value
            row[0] = unpaired
            for j in range(size):
                if not legal[i][j]:
                    row[j + 1] = NEG_INF
                    continue
                score = params.pair_compat * compat[i][j]
                for p in range(hidden_size):
                    hip = hidden[i][p]
                    if hip == 0.0:
                        continue
                    m_row = params.pair_matrix[p]
                    acc = 0.0
                    for q in range(hidden_size):
                        acc += m_row[q] * hidden[j][q]
                    score += hip * acc
                row[j + 1] = score
            logits.append(row)
            marginals.append(softmax(row))

        return ForwardResult(
            logits=tuple(tuple(row) for row in logits),
            marginals=tuple(marginals),
            hidden=tuple(tuple(row) for row in hidden),
            legal_pair=legal,
            compat=compat,
        )

    def marginals(self, sequence: str, features: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
        """Return only the per-position class marginals ``pi_i``.

        Complexity: O(L^2 H^2).
        """

        return self.forward(sequence, features).marginals

    def backward(
        self,
        forward_result: ForwardResult,
        features: Sequence[Sequence[float]],
        grad_logits: Sequence[Sequence[float]],
    ) -> DenoiserGradients:
        """Backpropagate upstream logit gradients into parameter gradients.

        ``grad_logits[i]`` is ``dL/dlogit_i`` (length ``K = L+1``).  The returned
        gradients follow the exact derivation in the module docstring.  The
        returned :class:`DenoiserGradients` also carries ``grad_features``, the
        gradient ``dL/dfeat_i = W^T dL/da_i`` w.r.t. the input feature vectors,
        so an external feature producer (the C5 frozen-encoder adapter) can chain
        its own backward pass.

        Complexity: O(L^2 H^2).
        """

        params = self.parameters
        hidden = [list(row) for row in forward_result.hidden]
        size = len(hidden)
        hidden_size = params.hidden_size
        compat = forward_result.compat
        legal = forward_result.legal_pair
        if len(grad_logits) != size:
            raise ValueError("grad_logits length must match sequence length")

        grad_hidden = [[0.0 for _ in range(hidden_size)] for _ in range(size)]
        grad_pair_matrix = [[0.0 for _ in range(hidden_size)] for _ in range(hidden_size)]
        grad_pair_compat = 0.0
        grad_unpaired_weight = [0.0 for _ in range(hidden_size)]
        grad_unpaired_bias = 0.0

        for i in range(size):
            g_row = grad_logits[i]
            if len(g_row) != size + 1:
                raise ValueError("each grad_logits row must have length L+1")
            g_unpaired = float(g_row[0])
            grad_unpaired_bias += g_unpaired
            for p in range(hidden_size):
                grad_unpaired_weight[p] += g_unpaired * hidden[i][p]
                grad_hidden[i][p] += g_unpaired * params.unpaired_weight[p]

            for j in range(size):
                if not legal[i][j]:
                    continue
                g = float(g_row[j + 1])
                if g == 0.0:
                    continue
                grad_pair_compat += g * compat[i][j]
                for p in range(hidden_size):
                    hip = hidden[i][p]
                    m_row = params.pair_matrix[p]
                    for q in range(hidden_size):
                        grad_pair_matrix[p][q] += g * hip * hidden[j][q]
                        grad_hidden[i][p] += g * m_row[q] * hidden[j][q]
                        grad_hidden[j][q] += g * hip * m_row[q]

        grad_input_weight = [[0.0 for _ in range(params.feature_size)] for _ in range(hidden_size)]
        grad_input_bias = [0.0 for _ in range(hidden_size)]
        grad_features = [[0.0 for _ in range(params.feature_size)] for _ in range(size)]
        for i in range(size):
            feat = features[i]
            grad_feat_row = grad_features[i]
            for p in range(hidden_size):
                grad_a = grad_hidden[i][p] * (1.0 - hidden[i][p] * hidden[i][p])
                grad_input_bias[p] += grad_a
                weight_row = params.input_weight[p]
                row = grad_input_weight[p]
                for f_index, value in enumerate(feat):
                    row[f_index] += grad_a * float(value)
                    grad_feat_row[f_index] += grad_a * weight_row[f_index]

        return DenoiserGradients(
            input_weight=grad_input_weight,
            input_bias=grad_input_bias,
            pair_matrix=grad_pair_matrix,
            pair_compat=grad_pair_compat,
            unpaired_weight=grad_unpaired_weight,
            unpaired_bias=grad_unpaired_bias,
            grad_features=grad_features,
        )


def sgd_update(parameters: DenoiserParameters, gradients: DenoiserGradients, learning_rate: float) -> None:
    """Apply an in-place SGD step ``theta <- theta - lr * grad``.

    Complexity: O(H*F + H^2).
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    for row, grad_row in zip(parameters.input_weight, gradients.input_weight):
        for index in range(len(row)):
            row[index] -= learning_rate * grad_row[index]
    for index in range(len(parameters.input_bias)):
        parameters.input_bias[index] -= learning_rate * gradients.input_bias[index]
    for row, grad_row in zip(parameters.pair_matrix, gradients.pair_matrix):
        for index in range(len(row)):
            row[index] -= learning_rate * grad_row[index]
    parameters.pair_compat -= learning_rate * gradients.pair_compat
    for index in range(len(parameters.unpaired_weight)):
        parameters.unpaired_weight[index] -= learning_rate * gradients.unpaired_weight[index]
    parameters.unpaired_bias -= learning_rate * gradients.unpaired_bias


def unpaired_probabilities(marginals: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    """Return per-position unpaired probability ``q_i = pi_i[0]``.

    Complexity: O(L).
    """

    return tuple(float(row[0]) for row in marginals)


def marginal_pair_matrix(marginals: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
    """Assemble a symmetric expected pairing matrix from class marginals.

    ``P_ij = 0.5 (pi_i[j+1] + pi_j[i+1])`` symmetrizes the two per-position
    marginals; the diagonal is zero.  This soft matrix feeds visualization and
    the greedy legal projection at sampling time.

    Complexity: O(L^2).
    """

    size = len(marginals)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            value = 0.5 * (float(marginals[i][j + 1]) + float(marginals[j][i + 1]))
            matrix[i][j] = value
    return tuple(tuple(row) for row in matrix)
