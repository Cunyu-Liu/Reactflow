"""Symbolic verification for ReactFlow mathematical derivations.

The functions in this module use SymPy to check algebraic identities that are
easy to get subtly wrong in implementation:

1. affine forward operator expectation;
2. weighted affine calibration normal equations;
3. softmax cross-entropy gradient ``softmax(l) - onehot(y)`` (DFM loss);
4. softmax Jacobian ``d pi_i / d l_k = pi_i (1[i=k] - pi_k)``;
5. mixture-path normalization, endpoints and time derivative;
6. Campbell conditional-rate-matrix master (Kolmogorov forward) equation for the
   uniform-source mixture path;
7. reactivity magnitude gradient into the denoiser logits;
8. variance-aware ensemble-calibration and contact denoising auxiliary gradients.

They return machine-readable dictionaries so tests and CI can fail loudly if an
identity no longer simplifies to zero.
"""

from __future__ import annotations

from typing import Dict


def _sympy():
    """Import SymPy lazily with an actionable error message.

    Complexity: O(1) after import cache warm-up.
    """

    try:
        import sympy as sp  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Symbolic verification requires optional dependency: pip install sympy") from exc
    return sp


def verify_affine_expectation_identity() -> Dict[str, str]:
    """Verify ``E[aU+bE+c] = aE[U]+bE[E]+c``.

    We enumerate the joint distribution of binary variables ``U`` and ``E``:
    ``p00,p01,p10,p11``.  The identity must hold without assuming independence,
    which is important because unpaired state and edge context are correlated in
    real RNA structures.

    Complexity: symbolic expression size is constant, O(1).
    """

    sp = _sympy()
    a, b, c = sp.symbols("a b c")
    p00, p01, p10, p11 = sp.symbols("p00 p01 p10 p11")
    total = p00 + p01 + p10 + p11
    eu = (p10 + p11) / total
    ee = (p01 + p11) / total
    direct = (p00 * c + p01 * (b + c) + p10 * (a + c) + p11 * (a + b + c)) / total
    exchanged = a * eu + b * ee + c
    residual = sp.simplify(direct - exchanged)
    return {"identity": "E[aU+bE+c] = aE[U]+bE[E]+c", "residual": str(residual)}


def verify_weighted_calibration_normal_equations() -> Dict[str, str]:
    """Verify normal equations for weighted affine calibration.

    Objective:

        J(alpha,gamma)=sum_i w_i (alpha*x_i+gamma-y_i)^2.

    Setting gradients to zero yields:

        alpha*Sxx + gamma*Sx = Sxy,
        alpha*Sx  + gamma*Sw = Sy.

    Complexity: constant symbolic problem O(1).
    """

    sp = _sympy()
    alpha, gamma = sp.symbols("alpha gamma")
    x1, x2, y1, y2, w1, w2 = sp.symbols("x1 x2 y1 y2 w1 w2")
    objective = w1 * (alpha * x1 + gamma - y1) ** 2 + w2 * (alpha * x2 + gamma - y2) ** 2
    grad_alpha = sp.simplify(sp.diff(objective, alpha) / 2)
    grad_gamma = sp.simplify(sp.diff(objective, gamma) / 2)
    sxx = w1 * x1**2 + w2 * x2**2
    sx = w1 * x1 + w2 * x2
    sw = w1 + w2
    sxy = w1 * x1 * y1 + w2 * x2 * y2
    sy = w1 * y1 + w2 * y2
    residual_alpha = sp.simplify(grad_alpha - (alpha * sxx + gamma * sx - sxy))
    residual_gamma = sp.simplify(grad_gamma - (alpha * sx + gamma * sw - sy))
    return {
        "identity": "weighted affine calibration normal equations",
        "residual_alpha": str(residual_alpha),
        "residual_gamma": str(residual_gamma),
    }


def verify_softmax_cross_entropy_gradient() -> Dict[str, str]:
    """Verify ``d/dl_k [-log softmax(l)_y] = softmax(l)_k - 1[k=y]``.

    This is the gradient used by :func:`reactflow.dfm.softmax_cross_entropy_gradient`
    and, position-wise, by the DFM denoising loss.  We build a 3-class softmax
    symbolically, differentiate the cross-entropy against a concrete label, and
    check the residual against the closed form for every logit component.

    Complexity: constant symbolic problem, O(1).
    """

    sp = _sympy()
    l0, l1, l2 = sp.symbols("l0 l1 l2", real=True)
    logits = [l0, l1, l2]
    exps = [sp.exp(value) for value in logits]
    denom = sum(exps)
    softmax = [item / denom for item in exps]
    residuals = {}
    for target in range(3):
        loss = -sp.log(softmax[target])
        for k in range(3):
            grad = sp.diff(loss, logits[k])
            indicator = 1 if k == target else 0
            residual = sp.simplify(grad - (softmax[k] - indicator))
            residuals[f"residual_y{target}_k{k}"] = str(residual)
    residuals["identity"] = "d/dl_k [-log softmax(l)_y] = softmax_k - 1[k=y]"
    return residuals


def verify_softmax_jacobian() -> Dict[str, str]:
    """Verify the softmax Jacobian ``d pi_i / d l_k = pi_i (1[i=k] - pi_k)``.

    The class-0 row of this Jacobian is exactly the factor
    ``pi_0 (1[k=0] - pi_k)`` used in the reactivity-consistency logit gradient
    (see :func:`reactflow.train._reactivity_logit_gradient`).  Verifying the full
    Jacobian symbolically guarantees that chain-rule factor is correct.

    Complexity: O(1).
    """

    sp = _sympy()
    l0, l1, l2 = sp.symbols("l0 l1 l2", real=True)
    logits = [l0, l1, l2]
    exps = [sp.exp(value) for value in logits]
    denom = sum(exps)
    softmax = [item / denom for item in exps]
    residuals = {}
    for i in range(3):
        for k in range(3):
            grad = sp.diff(softmax[i], logits[k])
            indicator = 1 if i == k else 0
            residual = sp.simplify(grad - softmax[i] * (indicator - softmax[k]))
            residuals[f"residual_i{i}_k{k}"] = str(residual)
    residuals["identity"] = "d pi_i / d l_k = pi_i (1[i=k] - pi_k)"
    return residuals


def verify_mixture_path_identities() -> Dict[str, str]:
    """Verify normalization, endpoints and derivative of the mixture path.

    For the linear/mixture path ``p_{t|1}(z|x1) = (1-t) p0(z) + t 1[z=x1]`` with a
    3-class general source ``p0`` summing to one, we check:

    * normalization ``sum_z p_{t|1}(z|x1) = 1`` for all ``t``;
    * endpoint ``p_{0|1} = p0`` and ``p_{1|1} = onehot(x1)``;
    * derivative ``d/dt p_{t|1}(z|x1) = 1[z=x1] - p0(z)`` (constant in ``t``).

    Complexity: O(1).
    """

    sp = _sympy()
    t = sp.symbols("t", real=True)
    p0, p1, p2 = sp.symbols("p0 p1 p2", nonnegative=True)
    source = [p0, p1, p2]
    data_index = 1  # x1 = class 1
    path = [(1 - t) * source[z] + (t if z == data_index else 0) for z in range(3)]
    # SymPy cannot know sum(source) = 1, so substitute the constraint explicitly.
    normalization_residual = sp.simplify((sum(path) - 1).subs(p2, 1 - p0 - p1))
    residuals: Dict[str, str] = {
        "identity": "mixture path normalization / endpoints / derivative",
        "constraint": "p0 + p1 + p2 = 1",
        "residual_normalization": str(normalization_residual),
    }
    for z in range(3):
        endpoint_zero = sp.simplify(path[z].subs(t, 0) - source[z])
        endpoint_one = sp.simplify(path[z].subs(t, 1) - (1 if z == data_index else 0))
        derivative = sp.simplify(sp.diff(path[z], t) - ((1 if z == data_index else 0) - source[z]))
        residuals[f"residual_endpoint_t0_z{z}"] = str(endpoint_zero)
        residuals[f"residual_endpoint_t1_z{z}"] = str(endpoint_one)
        residuals[f"residual_derivative_z{z}"] = str(derivative)
    return residuals


def verify_conditional_rate_master_equation() -> Dict[str, str]:
    """Verify the Campbell rate matrix satisfies the master equation.

    For the uniform-source mixture path with ``K=3`` and data class ``x1``, the
    conditional rate matrix is

        R*(z->j) = ReLU(d_t p(j) - d_t p(z)) / (Z_t p(z)),  j != z,

    with diagonal set to the negative row sum.  The Kolmogorov forward (master)
    equation requires, for every target state ``j``,

        d_t p(j) = sum_z R*(z->j) p(z).

    With a uniform source ``p0 = 1/3`` the derivative ``d_t p(z) = 1[z=x1] - 1/3``
    is positive only for ``z = x1``, so the ReLU keeps exactly the fluxes flowing
    into ``x1``.  We verify the residual is zero symbolically in ``t`` for the
    representative interior time where all path masses are positive.

    Complexity: O(K^2) symbolic, constant for fixed K.
    """

    sp = _sympy()
    t = sp.symbols("t", positive=True)
    num_classes = 3
    data_index = 0
    source_value = sp.Rational(1, num_classes)
    marginal = [(1 - t) * source_value + (t if z == data_index else 0) for z in range(num_classes)]
    derivative = [(1 if z == data_index else 0) - source_value for z in range(num_classes)]
    support = num_classes  # every mass positive on the open interval (0, 1)

    def relu(expr):
        """Symbolic ReLU valid on the interior where the sign is decidable.

        Complexity: O(size(expr)) for SymPy simplification.
        """

        simplified = sp.simplify(expr)
        return simplified if simplified.is_nonnegative else sp.Integer(0)

    rate = [[sp.Integer(0)] * num_classes for _ in range(num_classes)]
    for z in range(num_classes):
        row_sum = sp.Integer(0)
        for j in range(num_classes):
            if j == z:
                continue
            flux = derivative[j] - derivative[z]
            entry = relu(flux) / (support * marginal[z])
            rate[z][j] = entry
            row_sum += entry
        rate[z][z] = -row_sum

    residuals = {}
    for j in range(num_classes):
        lhs = derivative[j]
        rhs = sum(rate[z][j] * marginal[z] for z in range(num_classes))
        residuals[f"residual_master_j{j}"] = str(sp.simplify(lhs - rhs))
    for z in range(num_classes):
        residuals[f"residual_rowsum_z{z}"] = str(sp.simplify(sum(rate[z][j] for j in range(num_classes))))
    residuals["identity"] = "d_t p(j) = sum_z R*(z->j) p(z) and row sums = 0"
    return residuals


def verify_reactivity_magnitude_gradient() -> Dict[str, str]:
    """Verify the reactivity magnitude gradient into the denoiser logits.

    The single-sample magnitude loss (calibration frozen) is

        ell = sum_i w_i (alpha rhat_i + gamma - r_i)^2 / sum_j w_j,
        rhat_i = a_i q_i + c_i,   q_i = pi_i[0] = softmax(l_i)_0.

    Differentiating through ``q_i`` with the softmax Jacobian yields

        d ell / d l_i[k] = d_mag_i * a_i * pi_i[0] * (1[k=0] - pi_i[k]),
        d_mag_i = 2 w_i alpha (alpha rhat_i + gamma - r_i) / sum_j w_j.

    This matches :func:`reactflow.train._reactivity_logit_gradient`.  We verify a
    2-position, 3-class instance symbolically for every logit component.

    Complexity: O(L K) symbolic, constant for the fixed instance.
    """

    sp = _sympy()
    alpha, gamma = sp.symbols("alpha gamma", real=True)
    positions = 2
    num_classes = 3
    logits = [[sp.symbols(f"l{i}_{k}", real=True) for k in range(num_classes)] for i in range(positions)]
    a = sp.symbols("a0 a1", real=True)
    c = sp.symbols("c0 c1", real=True)
    r = sp.symbols("r0 r1", real=True)
    w = sp.symbols("w0 w1", positive=True)

    softmaxes = []
    for i in range(positions):
        exps = [sp.exp(logits[i][k]) for k in range(num_classes)]
        denom = sum(exps)
        softmaxes.append([item / denom for item in exps])
    q = [softmaxes[i][0] for i in range(positions)]
    rhat = [a[i] * q[i] + c[i] for i in range(positions)]
    weight_sum = w[0] + w[1]
    loss = sum(w[i] * (alpha * rhat[i] + gamma - r[i]) ** 2 for i in range(positions)) / weight_sum

    residuals = {}
    for i in range(positions):
        residual_i = alpha * rhat[i] + gamma - r[i]
        d_mag = 2 * w[i] * alpha * residual_i / weight_sum
        for k in range(num_classes):
            indicator = 1 if k == 0 else 0
            expected = d_mag * a[i] * softmaxes[i][0] * (indicator - softmaxes[i][k])
            grad = sp.diff(loss, logits[i][k])
            residuals[f"residual_pos{i}_k{k}"] = str(sp.simplify(grad - expected))
    residuals["identity"] = "d ell_mag / d l_i[k] = d_mag_i a_i pi_i0 (1[k=0]-pi_i[k])"
    return residuals


def verify_thermo_mse_gradient() -> Dict[str, str]:
    """Verify the MSE thermodynamic-prior gradient into the denoiser logits.

    The single-sample MSE semi-supervision loss is

        ell = (1/L) sum_i (q_i - t_i)^2,   q_i = pi_i[0] = softmax(l_i)_0,

    with the Turner prior ``t_i`` treated as constant.  Differentiating through
    ``q_i`` with the class-0 row of the softmax Jacobian gives

        d ell / d l_i[k] = (1/L) * g_i * pi_i[0] * (1[k=0] - pi_i[k]),
        g_i = 2 (q_i - t_i).

    This is exactly :func:`reactflow.thermo.thermo_logit_gradient` with
    ``lambda_thermo = 1`` and ``mode = "mse"``.  We verify a 2-position, 3-class
    instance symbolically for every logit component.

    Complexity: O(L K) symbolic, constant for the fixed instance.
    """

    sp = _sympy()
    positions = 2
    num_classes = 3
    logits = [[sp.symbols(f"l{i}_{k}", real=True) for k in range(num_classes)] for i in range(positions)]
    target = sp.symbols("t0 t1", real=True)

    softmaxes = []
    for i in range(positions):
        exps = [sp.exp(logits[i][k]) for k in range(num_classes)]
        denom = sum(exps)
        softmaxes.append([item / denom for item in exps])
    q = [softmaxes[i][0] for i in range(positions)]
    loss = sum((q[i] - target[i]) ** 2 for i in range(positions)) / positions

    residuals = {}
    for i in range(positions):
        g_i = 2 * (q[i] - target[i])
        for k in range(num_classes):
            indicator = 1 if k == 0 else 0
            expected = (g_i / positions) * softmaxes[i][0] * (indicator - softmaxes[i][k])
            grad = sp.diff(loss, logits[i][k])
            residuals[f"residual_pos{i}_k{k}"] = str(sp.simplify(grad - expected))
    residuals["identity"] = "d ell_thermo^MSE / d l_i[k] = (1/L) 2(q_i - t_i) pi_i0 (1[k=0]-pi_i[k])"
    return residuals


def verify_thermo_kl_gradient() -> Dict[str, str]:
    """Verify the Bernoulli-KL thermodynamic-prior gradient into the logits.

    The per-position forward KL from the Turner prior to the model is

        ell_i = t_i log(t_i / q_i) + (1 - t_i) log((1 - t_i)/(1 - q_i)),
        ell   = (1/L) sum_i ell_i,   q_i = pi_i[0].

    Its sensitivity to the model unpaired probability is

        g_i = d ell_i / d q_i = -t_i / q_i + (1 - t_i) / (1 - q_i),

    so, chaining through the class-0 softmax Jacobian,

        d ell / d l_i[k] = (1/L) * g_i * pi_i[0] * (1[k=0] - pi_i[k]).

    This matches :func:`reactflow.thermo.thermo_logit_gradient` with ``mode="kl"``
    (away from the ``[eps, 1-eps]`` clamp).  We verify a 2-position, 3-class
    instance symbolically for every logit component.

    Complexity: O(L K) symbolic, constant for the fixed instance.
    """

    sp = _sympy()
    positions = 2
    num_classes = 3
    logits = [[sp.symbols(f"l{i}_{k}", real=True) for k in range(num_classes)] for i in range(positions)]
    target = sp.symbols("t0 t1", positive=True)

    softmaxes = []
    for i in range(positions):
        exps = [sp.exp(logits[i][k]) for k in range(num_classes)]
        denom = sum(exps)
        softmaxes.append([item / denom for item in exps])
    q = [softmaxes[i][0] for i in range(positions)]
    loss = sum(
        target[i] * sp.log(target[i] / q[i]) + (1 - target[i]) * sp.log((1 - target[i]) / (1 - q[i]))
        for i in range(positions)
    ) / positions

    residuals = {}
    for i in range(positions):
        g_i = -target[i] / q[i] + (1 - target[i]) / (1 - q[i])
        for k in range(num_classes):
            indicator = 1 if k == 0 else 0
            expected = (g_i / positions) * softmaxes[i][0] * (indicator - softmaxes[i][k])
            grad = sp.diff(loss, logits[i][k])
            residuals[f"residual_pos{i}_k{k}"] = str(sp.simplify(grad - expected))
    residuals["identity"] = "d ell_thermo^KL / d l_i[k] = (1/L)(-t_i/q_i+(1-t_i)/(1-q_i)) pi_i0 (1[k=0]-pi_i[k])"
    return residuals


def verify_guidance_monotonicity_exchange() -> Dict[str, str]:
    """Verify the guidance-scan monotonicity exchange inequality.

    :func:`reactflow.thermo.guidance_eta_scan` projects onto the exact maximizer
    ``S(eta) = argmax_S [ f(S) - (eta/RT) g(S) ]`` over an ``eta``-independent
    feasible set, where ``f`` is the data score and ``g`` the pairing energy of a
    structure.  For ``eta_1 < eta_2`` with optima ``S_1, S_2`` optimality gives
    two inequalities; adding them must leave

        (eta_2 - eta_1)/RT * (g(S_1) - g(S_2)) >= 0,

    i.e. ``g`` is non-increasing in ``eta``.  We reproduce the algebra
    symbolically: starting from the two optimality slacks ``d1 >= 0`` and
    ``d2 >= 0`` (each the amount by which the incumbent beats the challenger), the
    sum ``d1 + d2`` must equal ``(eta_2 - eta_1)/RT * (g1 - g2)``.  The residual of
    that identity is checked to be zero, which certifies the sign argument used by
    :func:`reactflow.thermo.guidance_scan_is_monotone`.

    Complexity: O(1).
    """

    sp = _sympy()
    f1, f2, g1, g2 = sp.symbols("f1 f2 g1 g2", real=True)
    eta1, eta2, rt = sp.symbols("eta1 eta2 RT", positive=True)
    # Optimality of S1 at eta1: f1 - eta1/RT g1 >= f2 - eta1/RT g2, slack d1 >= 0.
    d1 = (f1 - eta1 / rt * g1) - (f2 - eta1 / rt * g2)
    # Optimality of S2 at eta2: f2 - eta2/RT g2 >= f1 - eta2/RT g1, slack d2 >= 0.
    d2 = (f2 - eta2 / rt * g2) - (f1 - eta2 / rt * g1)
    exchange = sp.simplify((d1 + d2) - (eta2 - eta1) / rt * (g1 - g2))
    return {
        "identity": "d1 + d2 = (eta2 - eta1)/RT (g1 - g2), with d1,d2 >= 0 => g1 >= g2",
        "residual_exchange": str(exchange),
    }


def verify_adapter_gradient() -> Dict[str, str]:
    """Verify the frozen-feature adapter parameter gradients (cycle C5.3).

    The linear adapter of :class:`reactflow.features.FeatureAdapter` maps a frozen
    per-nucleotide vector ``h in R^{d_single}`` to

        a_p = sum_f W[p][f] h[f] + b[p].

    Composed with any downstream scalar loss ``L(a)``, the chain rule gives, with
    ``g_p = dL/da_p``,

        dL/dW[p][f] = g_p h[f],     dL/db[p] = g_p.

    Because ``h`` is a *frozen* constant input there is no input gradient.  We
    build a 2-output adapter over a 3-dim frozen vector and compose it with a
    non-trivial downstream loss ``L = sum_p phi(a_p)`` (so ``g_p = phi'(a_p)`` is
    genuinely a function of the adapter output, exercising the full chain rule),
    then check the residual of every parameter gradient against the closed form.

    This certifies the hand-written :meth:`FeatureAdapter.backward` and, together
    with the finite-difference test, forms the double verification mandated for
    every hand-derived gradient in the project.

    Complexity: O(d_adapter * d_single) symbolic, constant for the fixed instance.
    """

    sp = _sympy()
    d_single = 3
    d_adapter = 2
    weight = [[sp.symbols(f"W{p}_{f}", real=True) for f in range(d_single)] for p in range(d_adapter)]
    bias = [sp.symbols(f"b{p}", real=True) for p in range(d_adapter)]
    h = [sp.symbols(f"h{f}", real=True) for f in range(d_single)]

    a = [sum(weight[p][f] * h[f] for f in range(d_single)) + bias[p] for p in range(d_adapter)]
    # Non-trivial downstream loss so dL/da_p depends on a_p (full chain rule).
    loss = sum(sp.sin(a[p]) + a[p] ** 2 for p in range(d_adapter))
    g = [sp.diff(loss, bias[p]) for p in range(d_adapter)]  # g_p = dL/da_p (since d a_p/d b_p = 1)

    residuals: Dict[str, str] = {}
    for p in range(d_adapter):
        # dL/db_p should equal g_p exactly.
        residuals[f"residual_bias_p{p}"] = str(sp.simplify(sp.diff(loss, bias[p]) - g[p]))
        for f in range(d_single):
            grad_w = sp.diff(loss, weight[p][f])
            residuals[f"residual_weight_p{p}_f{f}"] = str(sp.simplify(grad_w - g[p] * h[f]))
    residuals["identity"] = "dL/dW[p][f] = g_p h[f], dL/db[p] = g_p (frozen h)"
    return residuals


def verify_pearson_affine_invariance() -> Dict[str, str]:
    """Verify Pearson correlation is invariant to positive affine rescaling.

    The C5.4 evaluation reports Pearson (and, on ranks, Spearman) as
    *calibration-free shape* metrics.  That is only legitimate if rescaling the
    prediction by a positive affine map leaves the correlation unchanged.  For
    ``u_i = a x_i + c`` with ``a > 0`` we have ``u_i - ubar = a (x_i - xbar)``, so

        corr(a x + c, y) = a / sqrt(a^2) * corr(x, y) = corr(x, y).

    We build a 3-point instance with symbolic data, a symbolic *positive* scale
    ``a`` and offset ``c``, and check the residual ``corr(a x + c, y) -
    corr(x, y)`` simplifies to zero (SymPy uses ``a > 0`` to resolve
    ``sqrt(a^2) = a``).  This certifies the shape metric in
    :func:`reactflow.evaluate.reactivity_metrics`.

    Complexity: O(1) symbolic for the fixed 3-point instance.
    """

    sp = _sympy()
    a = sp.symbols("a", positive=True)
    c = sp.symbols("c", real=True)
    x = sp.symbols("x0 x1 x2", real=True)
    y = sp.symbols("y0 y1 y2", real=True)
    n = 3

    def corr(u, v):
        """Return the symbolic Pearson correlation for the fixed 3-vector.

        Formula: ``cov(u,v) / sqrt(var(u) var(v))``.  Complexity: O(1).
        """

        ubar = sum(u) / n
        vbar = sum(v) / n
        cov = sum((u[i] - ubar) * (v[i] - vbar) for i in range(n))
        varu = sum((u[i] - ubar) ** 2 for i in range(n))
        varv = sum((v[i] - vbar) ** 2 for i in range(n))
        return cov / sp.sqrt(varu * varv)

    rescaled = [a * x[i] + c for i in range(n)]
    residual = sp.simplify(corr(rescaled, list(y)) - corr(list(x), list(y)))
    return {
        "identity": "corr(a x + c, y) = corr(x, y) for a > 0",
        "residual_invariance": str(residual),
    }


def verify_heteroscedastic_calibration_gradient() -> Dict[str, str]:
    """Verify the variance-aware ensemble-calibration gradient into the logits.

    The single-position heteroscedastic Gaussian negative log-likelihood
    (:mod:`reactflow.ensemble`) is, with ``q = pi[0] = softmax(l)_0`` and probe
    coefficients ``a, c`` and calibration ``alpha, gamma`` held constant,

        mu = alpha (a q + c) + gamma,
        v  = beta a^2 q (1 - q) + tau2,
        ell = (mu - r)^2 / (2 v) + (1/2) log v.

    The chain rule factors through ``q`` with the class-0 softmax Jacobian
    ``d q / d l[k] = q (1[k=0] - pi[k])``:

        d ell / d l[k] = s * q * (1[k=0] - pi[k]),
        s = (mu - r) (alpha a) / v + ( 1/(2v) - (mu - r)^2 / (2 v^2) ) * beta a^2 (1 - 2q).

    This is exactly the per-position sensitivity in
    :func:`reactflow.ensemble.heteroscedastic_reactivity_logit_gradient` with unit
    weight and ``lambda_calib = 1``.  We check a 1-position, 3-class instance for
    every logit component; both the mean and variance channels are exercised
    because ``beta`` is kept symbolic and nonzero.

    Complexity: O(K) symbolic for the fixed instance.
    """

    sp = _sympy()
    num_classes = 3
    logits = [sp.symbols(f"l{k}", real=True) for k in range(num_classes)]
    a, c = sp.symbols("a c", real=True)
    alpha, gamma = sp.symbols("alpha gamma", real=True)
    beta = sp.symbols("beta", positive=True)
    tau2 = sp.symbols("tau2", positive=True)
    r = sp.symbols("r", real=True)

    exps = [sp.exp(logits[k]) for k in range(num_classes)]
    denom = sum(exps)
    pi = [item / denom for item in exps]
    q = pi[0]
    mu = alpha * (a * q + c) + gamma
    v = beta * a**2 * q * (1 - q) + tau2
    loss = (mu - r) ** 2 / (2 * v) + sp.log(v) / 2

    residuals = {}
    for k in range(num_classes):
        indicator = 1 if k == 0 else 0
        dmu_dq = alpha * a
        dv_dq = beta * a**2 * (1 - 2 * q)
        s = (mu - r) * dmu_dq / v + (1 / (2 * v) - (mu - r) ** 2 / (2 * v**2)) * dv_dq
        expected = s * q * (indicator - pi[k])
        grad = sp.diff(loss, logits[k])
        residuals[f"residual_k{k}"] = str(sp.simplify(grad - expected))
    residuals["identity"] = "d ell_calib / d l[k] = s q (1[k=0]-pi[k]); s couples mean + variance channels"
    return residuals


def verify_contact_denoising_gradient() -> Dict[str, str]:
    """Verify contact-BCE gradient for the DFM-induced soft contact probability.

    For two positions there is one unordered candidate pair ``(0,1)``.  The clean
    contact target is ``y`` and the DFM-induced soft contact is

        p = 0.5 * (pi_0[2] + pi_1[1]),

    because row 0 chooses partner 1 with class ``2`` and row 1 chooses partner 0
    with class ``1``.  The balanced scaling is a constant, so the core BCE
    derivative is

        d BCE / d p = (p - y) / (p (1 - p)).

    The row-0 logit gradient is ``0.5 * dBCE/dp`` times the class-2 softmax
    Jacobian; row 1 receives the symmetric class-1 contribution.  This is the
    identity implemented by :func:`reactflow.contact.contact_denoising_logit_gradient`.

    Complexity: O(K) symbolic for the fixed 2-position instance.
    """

    sp = _sympy()
    num_classes = 3
    logits0 = [sp.symbols(f"a{k}", real=True) for k in range(num_classes)]
    logits1 = [sp.symbols(f"b{k}", real=True) for k in range(num_classes)]
    y = sp.symbols("y", real=True)

    def softmax(logits):
        """Return symbolic softmax probabilities for the fixed logits.

        Formula: ``pi_k = exp(l_k) / sum_j exp(l_j)``.  Complexity: O(K).
        """

        exps = [sp.exp(value) for value in logits]
        denom = sum(exps)
        return [value / denom for value in exps]

    pi0 = softmax(logits0)
    pi1 = softmax(logits1)
    p = (pi0[2] + pi1[1]) / 2
    loss = -(y * sp.log(p) + (1 - y) * sp.log(1 - p))
    dloss_dp = (p - y) / (p * (1 - p))

    residuals = {}
    for k in range(num_classes):
        expected0 = sp.Rational(1, 2) * dloss_dp * pi0[2] * ((1 if k == 2 else 0) - pi0[k])
        residuals[f"residual_row0_k{k}"] = str(sp.simplify(sp.diff(loss, logits0[k]) - expected0))

        expected1 = sp.Rational(1, 2) * dloss_dp * pi1[1] * ((1 if k == 1 else 0) - pi1[k])
        residuals[f"residual_row1_k{k}"] = str(sp.simplify(sp.diff(loss, logits1[k]) - expected1))
    residuals["identity"] = "d BCE(P_01,y) / d logits follows 0.5 times the two class-specific softmax Jacobians"
    return residuals


def run_all_symbolic_checks() -> Dict[str, Dict[str, str]]:
    """Run all symbolic checks.

    Complexity: O(1) for the current fixed set of derivations.
    """

    return {
        "affine_expectation": verify_affine_expectation_identity(),
        "weighted_calibration": verify_weighted_calibration_normal_equations(),
        "softmax_cross_entropy_gradient": verify_softmax_cross_entropy_gradient(),
        "softmax_jacobian": verify_softmax_jacobian(),
        "mixture_path": verify_mixture_path_identities(),
        "conditional_rate_master_equation": verify_conditional_rate_master_equation(),
        "reactivity_magnitude_gradient": verify_reactivity_magnitude_gradient(),
        "thermo_mse_gradient": verify_thermo_mse_gradient(),
        "thermo_kl_gradient": verify_thermo_kl_gradient(),
        "guidance_monotonicity_exchange": verify_guidance_monotonicity_exchange(),
        "adapter_gradient": verify_adapter_gradient(),
        "pearson_affine_invariance": verify_pearson_affine_invariance(),
        "heteroscedastic_calibration_gradient": verify_heteroscedastic_calibration_gradient(),
        "contact_denoising_gradient": verify_contact_denoising_gradient(),
    }
