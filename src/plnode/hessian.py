"""Matrix-free projected Hessian of J at nu* (Meta-Theorem 1, quantitative).

Do NOT assemble: (K+1) M ~ 3.4e4 variables would give a ~9 GB dense matrix.
The gradient of the discrete cost comes almost for free from one forward pass
+ cocycle:

    G_{k,l}(nu) = w_k [ Psi_l(t_k) + eps (log nu_{k,l} - log nu^inf_l) ],

modulo a constant per time slice (normal direction of the simplex constraint),
removed by projecting to zero mean per slice.

Hessian-vector products are SPLIT into an exact part and a smooth
finite-difference part:

  * entropic part, analytic and diagonal:
        (H_ent v)_{k,l} = w_k eps v_{k,l} / nu*_{k,l};
  * terminal part, G_Psi(nu)_{k,l} = w_k Psi_l(t_k; nu), differentiated by
    CENTRAL directional differences
        H_Psi v ~ [G_Psi(nu* + h v) - G_Psi(nu* - h v)] / (2 h).
    Psi(nu) involves NO logarithms (forward pass + cocycle are smooth in nu,
    and tolerate slightly negative masses), so no clipping is needed and the
    difference is well conditioned.

Rationale for the split: differencing the FULL gradient would difference
eps log nu across near-zero nodes (masses ~ e^{-40}), where any FD step h
crosses zero and clipping destroys both accuracy and the symmetry of H.
The split keeps the stiff term exact and differences only the smooth one.

v lives in the tangent space (zero mean per slice); we project before and
after.  Each product costs two forward passes.  lambda_min via Lanczos
(scipy eigsh on a LinearOperator, which='SA').

Note: the exact diagonal eps w_k / nu* is huge at near-zero nodes (that IS
the curvature of the entropic term there), so the spectrum is very spread;
lambda_min lives on directions supported where nu* has mass.  For Lanczos
efficiency an optional `support_threshold` restricts the operator to nodes
with nu* >= threshold (the complement has enormous positive curvature and
cannot carry lambda_min).

Internal checks:
  * symmetry <Hu, v> ~ <u, Hv> on random pairs;
  * coherence with Pontryagin: <H eta, eta> / 2 must predict the measured
    excess J(nu* + eta) - J* for small smooth perturbations eta.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .fisher import psi_values, time_weights
from .solver import cocycle_gradients, cost, forward_pass, log_prior

LOG_CLIP = 1e-300


# =============================================================================
# Tangent-space projection and gradients
# =============================================================================
def project_tangent(v, mask=None):
    """Remove the per-slice mean: tangent space of {sum_l nu_{k,l} = 1}.

    With a boolean support mask, the mean is taken (and removed) on the
    support only; entries off support are zeroed.
    """
    if mask is None:
        return v - v.mean(axis=1, keepdims=True)
    out = np.where(mask[None, :] if mask.ndim == 1 else mask, v, 0.0)
    denom = max(int(mask.sum()), 1)
    out = out - (out.sum(axis=1, keepdims=True) / denom)
    return np.where(mask[None, :] if mask.ndim == 1 else mask, out, 0.0)


def gradient_psi_part(nu, x0, y, A, d1, *, T=1.0, grad_fn=None):
    """G_Psi(nu)_{k,l} = w_k Psi_l(t_k; nu)  (smooth part of the gradient)."""
    K = nu.shape[0] - 1
    dt = T / K
    X, Phi = forward_pass(nu, x0, A, d1, dt, K)
    p = cocycle_gradients(X, Phi, y, grad_fn=grad_fn)
    Psi = psi_values(X, p, A, d1)
    return time_weights(K, dt)[:, None] * Psi


def gradient(nu, x0, y, A, d1, eps, *, T=1.0, c1=0.05, c2=0.5,
             grad_fn=None, log_nu_inf=None):
    """Full Euclidean gradient G(nu) of the discrete cost, projected."""
    K = nu.shape[0] - 1
    dt = T / K
    if log_nu_inf is None:
        log_nu_inf = log_prior(A, c1, c2)
    G_psi = gradient_psi_part(nu, x0, y, A, d1, T=T, grad_fn=grad_fn)
    w = time_weights(K, dt)
    log_nu = np.log(np.clip(nu, LOG_CLIP, None))
    G = G_psi + w[:, None] * eps * (log_nu - log_nu_inf[None, :])
    return project_tangent(G)


# =============================================================================
# Hessian operator
# =============================================================================
def hessian_operator(nu_star, x0, y, A, d1, eps, *, T=1.0, c1=0.05, c2=0.5,
                     grad_fn=None, h=1e-5, support_threshold=None,
                     sigma_null=100.0, verbose=False):
    """LinearOperator for v -> H v at nu_star (split exact + central FD).

    support_threshold: if given, restrict to nodes with nu* >= threshold on
    every slice (mask shared across slices for a well-defined tangent space).
    Counter attribute `.n_products` tracks the number of Hv products.

    sigma_null: penalty added on the COMPLEMENT of the tangent subspace
    (per-slice constants, and off-support nodes when restricted).  Without
    it the projected operator has a large null space with eigenvalue
    0 < lambda_min, and smallest-eigenvalue solvers converge to those
    spurious modes instead of lambda_min.  The penalty moves the whole
    complement to sigma_null >> lambda_min; vectors inside the tangent
    subspace are unaffected.
    """
    K1, M = nu_star.shape
    K = K1 - 1
    dt = T / K
    w = time_weights(K, dt)
    n = K1 * M
    counter = {"n": 0}

    mask = None
    if support_threshold is not None:
        mask = (nu_star >= support_threshold).all(axis=0)   # (M,)
        if verbose:
            print(f"    support restriction: {int(mask.sum())}/{M} nodes")

    diag_ent = w[:, None] * eps / np.clip(nu_star, LOG_CLIP, None)

    def matvec(v_flat):
        v_raw = np.asarray(v_flat, dtype=float).reshape(K1, M)
        v = project_tangent(v_raw, mask)
        null_part = v_raw - v                     # complement of the tangent
        scale = np.abs(v).max()
        if scale == 0:
            return (sigma_null * null_part).ravel()
        step = h / scale
        Gp = gradient_psi_part(nu_star + step * v, x0, y, A, d1, T=T,
                               grad_fn=grad_fn)
        Gm = gradient_psi_part(nu_star - step * v, x0, y, A, d1, T=T,
                               grad_fn=grad_fn)
        Hv = (Gp - Gm) / (2.0 * step) + diag_ent * v
        Hv = project_tangent(Hv, mask) + sigma_null * null_part
        counter["n"] += 1
        if verbose and counter["n"] % 10 == 0:
            print(f"    Hv products: {counter['n']}", flush=True)
        return Hv.ravel()

    op = LinearOperator((n, n), matvec=matvec, dtype=float)
    op.n_products = lambda: counter["n"]
    op.mask = mask
    op.diag_ent = diag_ent
    op.sigma_null = sigma_null
    return op


# =============================================================================
# Eigenvalues and checks
# =============================================================================
def _check_tangent(op, vecs, K1, M):
    """Warn if a returned eigenvector lives in the penalized complement."""
    for j in range(vecs.shape[1]):
        v = vecs[:, j].reshape(K1, M)
        vt = project_tangent(v, op.mask)
        frac = np.linalg.norm(v - vt) / max(np.linalg.norm(v), 1e-300)
        if frac > 1e-6:
            raise RuntimeError(
                f"eigenvector {j} has {frac:.2e} of its norm outside the "
                f"tangent subspace — lambda_min may exceed sigma_null="
                f"{op.sigma_null}; increase sigma_null and rerun")


def lambda_min(op, k=1, tol=1e-6, maxiter=None, v0=None):
    """Smallest algebraic eigenvalue(s) of the projected Hessian via Lanczos.

    Requires an operator built with sigma_null > lambda_min so that the
    complement of the tangent subspace (per-slice constants, off-support
    nodes) does not pollute the bottom of the spectrum; this is verified on
    the returned eigenvectors.  Returns (vals, vecs).
    """
    n = op.shape[0]
    K1M = op.diag_ent.shape
    vals, vecs = eigsh(op, k=k, which="SA", tol=tol, maxiter=maxiter, v0=v0)
    _check_tangent(op, vecs, *K1M)
    assert n == K1M[0] * K1M[1]
    return vals, vecs


def lambda_min_lobpcg(op, k=1, tol=1e-4, maxiter=300, seed=0, verbose=False):
    """lambda_min via LOBPCG with the exact entropic diagonal as Jacobi
    preconditioner (the natural preconditioner here: the spectrum spread is
    dominated by eps w_k / nu*).  Usually converges in tens of Hv products
    where plain Lanczos needs hundreds.
    """
    from scipy.sparse.linalg import lobpcg

    K1, M = op.diag_ent.shape
    n = K1 * M
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n, k))
    for j in range(k):
        X[:, j] = project_tangent(X[:, j].reshape(K1, M), op.mask).ravel()

    diag = op.diag_ent + 1.0                      # regularized Jacobi diagonal

    def precond(v):
        v = np.asarray(v)
        if v.ndim == 1:
            return project_tangent((v.reshape(K1, M) / diag), op.mask).ravel()
        cols = [project_tangent((v[:, j].reshape(K1, M) / diag),
                                op.mask).ravel() for j in range(v.shape[1])]
        return np.stack(cols, axis=1)

    Minv = LinearOperator((n, n), matvec=precond, matmat=precond, dtype=float)
    vals, vecs = lobpcg(op, X, M=Minv, largest=False, tol=tol,
                        maxiter=maxiter, verbosityLevel=1 if verbose else 0)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    _check_tangent(op, vecs, K1, M)
    return vals, vecs


def symmetry_check(op, K1, M, n_pairs=3, seed=0):
    """max relative asymmetry |<Hu,v> - <u,Hv>| / (|Hu||v| + |u||Hv|)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_pairs):
        u = project_tangent(rng.standard_normal((K1, M)), op.mask).ravel()
        v = project_tangent(rng.standard_normal((K1, M)), op.mask).ravel()
        Hu, Hv = op @ u, op @ v
        num = abs(Hu @ v - u @ Hv)
        den = np.linalg.norm(Hu) * np.linalg.norm(v) \
            + np.linalg.norm(u) * np.linalg.norm(Hv)
        worst = max(worst, num / den if den > 0 else 0.0)
    return worst


def quadratic_prediction_check(op, nu_star, x0, y, A, d1, eps, eta, *,
                               T=1.0, c1=0.05, c2=0.5, grad_fn=None,
                               terminal_fn=None):
    """Pontryagin coherence: <H eta, eta>/2 vs measured J(nu*+eta) - J*.

    eta must be a small tangent perturbation keeping nu* + eta > 0 (use e.g.
    a lognormal tilt of nu*): the quadratic model is meaningless across the
    boundary of the simplex.  Returns (quad_pred, excess_measured).
    """
    K = nu_star.shape[0] - 1
    dt = T / K
    log_nu_inf = log_prior(A, c1, c2)

    def J_of(nu):
        X, _ = forward_pass(nu, x0, A, d1, dt, K)
        val, _, _ = cost(X, y, nu, log_nu_inf, eps, dt, terminal_fn=terminal_fn)
        return val

    eta_t = project_tangent(eta, op.mask)
    quad = 0.5 * float((op @ eta_t.ravel()) @ eta_t.ravel())
    excess = J_of(nu_star + eta_t) - J_of(nu_star)
    return quad, excess
