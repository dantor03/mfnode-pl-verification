"""Explicit solver for the Gibbs formula of Daudin-Delarue (2025), Thm 1.4.

Ported (behavior-preserving) from the memoria code ``explicit_nd.py``.

Setting: prototype vector field b(x, a) = tanh(a1 . x + a2) * a0 with
x in R^{d1}, a = (a0, a1, a2) in A = R^{d'}, d' = 2 d1 + 1, and Gibbs prior
nu^inf ~ exp(-ell(a)), ell(a) = c1 |a|^4 + c2 |a|^2.

The optimal control satisfies (Thm 1.4)

    nu*_t(a) ~ exp( -ell(a) - (1/eps) int b(x,a) . grad_x u*_t dgamma*_t ),

with (gamma*, u*) solving the coupled forward-backward system.  The backward
transport PDE is solved *by characteristics*: u* is constant along them, so

    grad_x u*_t(X_i(t)) = [Phi_i(T) Phi_i(t)^{-1}]^T grad_x L(X_i(T), y_i)

(the Jacobian cocycle), where Phi_i(t) = dX_i(t)/dX_i(0) solves the matrix
variational equation along the forward flow.

The fixed point is computed by damped Picard iteration

    nu^{k+1} = (1 - omega) nu^k + omega nu_Gibbs(nu^k),

with logsumexp stabilization, stopping when diffTV < tol.  NOTE the key
calibration finding of the memoria: diffTV measures the *damped* step, so the
true distance to the fixed point is diffTV / omega.

For small eps the Gibbs map loses contractivity; we use continuation in eps
(warm start along a schedule, default (0.5, 0.2, 0.1, 0.05, 0.02)).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

EPS_SCHEDULE = (0.5, 0.2, 0.1, 0.05, 0.02)

# Asymmetric production grid of the memoria: better resolution on the active
# axes (a0^(1), a0^(2), a1^(2)); M = 7*9*5*7*5 = 11025.  All entries odd so
# that a = 0 (mode of the prior, dynamically inert point) is a grid node.
MP_ASYM = (7, 9, 5, 7, 5)
# Uniform pilot grid used for the gradient-flow experiments: M = 5^5 = 3125.
MP_PILOT = (5, 5, 5, 5, 5)


# =============================================================================
# Vector field and derivatives
# =============================================================================
def b_nd(x, A, d1):
    """b(x, a) = tanh(a1 . x + a2) * a0.

    x : (N, d1), A : (M, 2 d1 + 1) with columns [a0 (d1), a1 (d1), a2 (1)]
    -> (N, M, d1)
    """
    a0 = A[:, :d1]
    a1 = A[:, d1:2 * d1]
    a2 = A[:, 2 * d1]
    pre = x @ a1.T + a2[None, :]                     # (N, M)
    s = np.tanh(pre)
    return s[:, :, None] * a0[None, :, :]            # (N, M, d1)


def db_dx_nd(x, A, d1):
    """(db_k/dx_j)(x, a) = sech^2(a1 . x + a2) a1_j a0_k -> (N, M, d1, d1)."""
    a0 = A[:, :d1]
    a1 = A[:, d1:2 * d1]
    a2 = A[:, 2 * d1]
    pre = x @ a1.T + a2[None, :]
    sech2 = 1.0 - np.tanh(pre) ** 2                  # (N, M)
    out = a0[:, :, None] * a1[:, None, :]            # (M, d1, d1)
    return sech2[:, :, None, None] * out[None, :, :, :]


def ell_nd(A, c1=0.05, c2=0.5):
    """Prior potential ell(a) = c1 |a|^4 + c2 |a|^2 (strongly convex, quartic)."""
    n2 = (A ** 2).sum(axis=1)
    return c1 * n2 ** 2 + c2 * n2


def grad_ell_nd(A, c1=0.05, c2=0.5):
    """grad ell(a) = (4 c1 |a|^2 + 2 c2) a  -> (M, d')."""
    n2 = (A ** 2).sum(axis=1)
    return (4.0 * c1 * n2 + 2.0 * c2)[:, None] * A


# =============================================================================
# Parameter grids over A = [-R, R]^{d'}
# =============================================================================
def build_grid(mp_per_axis, R=2.0):
    """Cartesian grid on [-R, R]^D with mp_per_axis[j] points on axis j.

    Every mp_per_axis[j] must be ODD so that the origin a = 0 is a node
    (mode of the prior and dynamically inert point: b(x,0)=0, d_x b(x,0)=0).

    Returns (A, axis_grids) with A of shape (prod(mp), D), 'ij' ordering, so
    A.reshape(*mp, D)[..., j] varies along tensor index j.
    """
    mp_per_axis = tuple(int(m) for m in mp_per_axis)
    assert all(m % 2 == 1 for m in mp_per_axis), (
        f"mp_per_axis must be odd per axis to include the origin; got {mp_per_axis}"
    )
    axis_grids = [np.linspace(-R, R, m) for m in mp_per_axis]
    mesh = np.meshgrid(*axis_grids, indexing="ij")
    A = np.stack([g.ravel() for g in mesh], axis=1)
    return A, axis_grids


def log_prior(A, c1=0.05, c2=0.5):
    """Normalized log prior masses on the grid nodes (uniform cell volume)."""
    log_w = -ell_nd(A, c1, c2)
    return log_w - logsumexp(log_w)


# =============================================================================
# Forward pass: particles + variational Jacobians, RK4
# =============================================================================
def forward_pass(nu, x0, A, d1, dt, K):
    """Integrate (X_i, Phi_i) on [0, T]: dX/dt = bbar(X, nu_t),
    dPhi/dt = (d_x bbar)|_X Phi, Phi(0) = I.

    nu : (K+1, M), x0 : (N, d1)
    -> X : (K+1, N, d1), Phi : (K+1, N, d1, d1)
    """
    N = x0.shape[0]
    X = np.zeros((K + 1, N, d1))
    Phi = np.zeros((K + 1, N, d1, d1))
    X[0] = x0
    Phi[0] = np.broadcast_to(np.eye(d1), (N, d1, d1)).copy()

    def deriv(x, ph, nu_k):
        bX = b_nd(x, A, d1)                              # (N, M, d1)
        dbX = db_dx_nd(x, A, d1)                          # (N, M, d1, d1)
        bbar = np.einsum('nmd,m->nd', bX, nu_k)
        dbar = np.einsum('nmkj,m->nkj', dbX, nu_k)
        dph = np.einsum('nkj,nji->nki', dbar, ph)
        return bbar, dph

    for k in range(K):
        nu_k, nu_mid, nu_end = nu[k], 0.5 * (nu[k] + nu[k + 1]), nu[k + 1]
        x, ph = X[k], Phi[k]
        kx1, kp1 = deriv(x, ph, nu_k)
        kx2, kp2 = deriv(x + 0.5 * dt * kx1, ph + 0.5 * dt * kp1, nu_mid)
        kx3, kp3 = deriv(x + 0.5 * dt * kx2, ph + 0.5 * dt * kp2, nu_mid)
        kx4, kp4 = deriv(x + dt * kx3, ph + dt * kp3, nu_end)
        X[k + 1] = x + dt / 6 * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
        Phi[k + 1] = ph + dt / 6 * (kp1 + 2 * kp2 + 2 * kp3 + kp4)
    return X, Phi


def cocycle_gradients(X, Phi, y, grad_fn=None):
    """grad_x u_t at the particles via the Jacobian cocycle.

    p[k, i] = [Phi_i(T) Phi_i(t_k)^{-1}]^T grad_x L(X_i(T), y_i)  -> (K+1, N, d1)

    grad_fn: optional callable X_T -> grad_x L(X_T, y) of shape (N, d1);
    defaults to full MSE, grad = X(T) - y.
    """
    K1 = X.shape[0]
    dxu_T = (X[-1] - y) if grad_fn is None else grad_fn(X[-1])
    PhiT = Phi[-1]
    p = np.zeros_like(X)
    for k in range(K1):
        Pk_inv = np.linalg.inv(Phi[k])
        M_t = np.einsum('nij,njk->nik', PhiT, Pk_inv)
        p[k] = np.einsum('nij,ni->nj', M_t, dxu_T)
    return p


def gibbs_update(X, p, A, d1, eps, log_nu_inf):
    """Apply the Gibbs formula at each t_k given cocycle gradients p.

    Also returns Psi[k, l] = (1/N) sum_i b(X_i(t_k), a_l) . p[k, i]
    (the 'term' entering both the Gibbs exponent and the Fisher functional).
    """
    K1, N, _ = X.shape
    M = A.shape[0]
    new_nu = np.zeros((K1, M))
    Psi = np.zeros((K1, M))
    for k in range(K1):
        bXk = b_nd(X[k], A, d1)                          # (N, M, d1)
        Psi[k] = np.einsum('nmd,nd->m', bXk, p[k]) / N
        log_w = log_nu_inf - Psi[k] / eps
        new_nu[k] = np.exp(log_w - logsumexp(log_w))
    return new_nu, Psi


# =============================================================================
# Discrete cost
# =============================================================================
def cost(X, y, nu, log_nu_inf, eps, dt, terminal_fn=None):
    """J_h = terminal + eps * sum_k w_k KL(nu_k || nu^inf), trapezoidal w_k.

    terminal_fn: optional callable X_T -> scalar; defaults to full MSE
    0.5 E |X_T - y|^2.

    NOTE: the memoria used a rectangle rule (dt on all K+1 slices, i.e. an
    effective horizon T + dt for the entropic term).  The trapezoidal rule is
    used here so that the discrete identity dJ_h/ds = -I_h holds with matching
    quadratures (I_h uses the same w_k).  nu* is unaffected (the Gibbs fixed
    point is independent of the quadrature); only reported J values shift
    slightly with respect to the memoria.
    """
    if terminal_fn is None:
        terminal = 0.5 * ((X[-1] - y) ** 2).sum(axis=1).mean()
    else:
        terminal = terminal_fn(X[-1])
    K1 = nu.shape[0]
    w = np.full(K1, dt)
    w[0] = w[-1] = dt / 2
    log_nu = np.log(np.clip(nu, 1e-300, None))
    kl = (nu * (log_nu - log_nu_inf[None, :])).sum(axis=1)
    entropic = eps * float(w @ kl)
    return terminal + entropic, terminal, entropic


# =============================================================================
# Damped Picard iteration + continuation in eps
# =============================================================================
@dataclass
class PicardResult:
    nu: np.ndarray                 # (K+1, M) fixed point (last iterate)
    X: np.ndarray                  # (K+1, N, d1) forward particles at nu
    Phi: np.ndarray                # (K+1, N, d1, d1) variational Jacobians
    cost_history: list = field(default_factory=list)
    diff_history: list = field(default_factory=list)
    iters: int = 0
    converged: bool = False
    eps: float = np.nan
    omega: float = np.nan

    @property
    def diff_final(self):
        return self.diff_history[-1] if self.diff_history else np.nan

    @property
    def dist_to_fixed_point(self):
        """Calibrated TV distance to the true fixed point: diffTV / omega."""
        return self.diff_final / self.omega


def picard(x0, y, A, d1, *, T=1.0, K=10, eps=0.5,
           c1=0.05, c2=0.5, omega=0.2, max_iter=200, tol=1e-4,
           verbose=True, nu_init=None, log_nu_inf=None,
           grad_fn=None, terminal_fn=None):
    """Damped Picard iteration for the Gibbs fixed point at fixed eps.

    diffTV = max_k TV(nu^{k+1}_t, nu^k_t) measures the *damped* step; the true
    distance to the fixed point is diffTV / omega (see PicardResult).
    """
    dt = T / K
    if log_nu_inf is None:
        log_nu_inf = log_prior(A, c1, c2)
    nu_inf = np.exp(log_nu_inf)
    nu = nu_init.copy() if nu_init is not None else np.tile(nu_inf, (K + 1, 1))

    cost_hist, diff_hist = [], []
    converged = False
    it = -1
    for it in range(max_iter):
        X, Phi = forward_pass(nu, x0, A, d1, dt, K)
        p = cocycle_gradients(X, Phi, y, grad_fn=grad_fn)
        nu_new, _ = gibbs_update(X, p, A, d1, eps, log_nu_inf)
        nu_next = (1 - omega) * nu + omega * nu_new
        nu_next /= nu_next.sum(axis=1, keepdims=True)
        diff = np.abs(nu_next - nu).sum(axis=1).max() / 2
        Jval, term, ent = cost(X, y, nu_next, log_nu_inf, eps, dt,
                               terminal_fn=terminal_fn)
        cost_hist.append(Jval)
        diff_hist.append(diff)
        if verbose and (it < 5 or it % 10 == 0):
            print(f"  it {it:3d}  J={Jval:.5f}  term={term:.5f}  "
                  f"ent={ent:.5f}  diffTV={diff:.2e}")
        nu = nu_next
        if diff < tol:
            converged = True
            if verbose:
                print(f"  converged at iter {it} (diffTV={diff:.2e}, "
                      f"dist ~ diffTV/omega = {diff / omega:.2e})")
            break

    X, Phi = forward_pass(nu, x0, A, d1, dt, K)
    return PicardResult(nu=nu, X=X, Phi=Phi,
                        cost_history=cost_hist, diff_history=diff_hist,
                        iters=it + 1, converged=converged, eps=eps, omega=omega)


def picard_continuation(x0, y, A, d1, eps_schedule=EPS_SCHEDULE, *, T=1.0, K=10,
                        c1=0.05, c2=0.5, omega=0.2, max_iter=200, tol=1e-4,
                        verbose=True, log_nu_inf=None,
                        grad_fn=None, terminal_fn=None):
    """Continuation in eps with warm start: returns [(eps, PicardResult), ...]."""
    results, nu_warm = [], None
    for i, eps in enumerate(eps_schedule):
        if verbose:
            print(f"\n[continuation {i + 1}/{len(eps_schedule)}] eps={eps}")
        res = picard(x0, y, A, d1, T=T, K=K, eps=eps, c1=c1, c2=c2,
                     omega=omega, max_iter=max_iter, tol=tol,
                     verbose=verbose, nu_init=nu_warm, log_nu_inf=log_nu_inf,
                     grad_fn=grad_fn, terminal_fn=terminal_fn)
        results.append((eps, res))
        nu_warm = res.nu
    return results


# =============================================================================
# Correctness check: exact a <-> -a symmetry (memoria, Prop. de simetria)
# =============================================================================
def symmetry_error(nu, mp_per_axis):
    """max |nu(a) - nu(-a)| over nodes and time slices.

    With tanh odd and ell even, b(x, .) is even in a => nu*_t(-a) = nu*_t(a)
    exactly.  On a symmetric grid the map a -> -a reverses every axis index,
    so the check is a pure array reversal.  Machine precision (~1e-16) expected.
    """
    mp = tuple(int(m) for m in mp_per_axis)
    K1 = nu.shape[0]
    nu_t = nu.reshape((K1,) + mp)
    flipped = nu_t[(slice(None),) + tuple(slice(None, None, -1) for _ in mp)]
    return float(np.abs(nu_t - flipped).max())
