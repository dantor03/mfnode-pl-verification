"""Discretized Fisher-information functional I(nu) (Daudin-Delarue, Sec. 5).

First variation of J with respect to nu_t:

    dJ/dnu_t(a) = Psi_t(a) + eps log(nu_t(a) / nu^inf(a)) + const,
    Psi_t(a)    = int b(x, a) . grad_x u_t(x, y) dgamma_t(x, y),

with (gamma, u) solved with the *current* control nu (not the optimum).  The
functional driving the Wasserstein gradient flow (dJ/ds = -I) is

    I(nu) = int_0^T int_A | eps grad_a log nu_t(a) + eps grad_a ell(a)
                            + grad_a Psi_t(a) |^2  dnu_t(a) dt
          = int_0^T int_A | grad_a ( eps log(nu_t/nu^inf) + Psi_t ) |^2 dnu_t dt.

Discretization with trapezoidal weights w_0 = w_K = dt/2, w_k = dt:

    I_h(nu) = sum_k w_k sum_l | G_{k,l} |^2 nu_{k,l},

where the integrand field G admits two discretizations (mode=):

  * "fd" (DEFAULT, discretely consistent): apply the SAME finite-difference
    operator D_a to every term,
        G = D_a[ eps log nu + eps ell + Psi ].
    Because the discrete Gibbs fixed point satisfies eps log nu* = -eps ell
    - Psi + const per slice EXACTLY, the integrand vanishes at nu* to solver
    tolerance: I_h(nu*_eps) ~ (diffTV_final / omega)^2, the Phase-1
    calibration test.  This is also the discretization dual to the
    Scharfetter-Gummel flow (gradient_flow.py), for which nu* is an exact
    steady state.

  * "analytic": grad ell and grad_a Psi analytic (exact per point), only
    grad_a log nu by finite differences.  Pointwise more accurate for the
    CONTINUOUS integrand, but the Gibbs cancellation at nu* then holds only
    up to the O(h^2) mismatch between D_a and the exact gradient of the
    quartic-plus-Psi potential, leaving an O(h^4)-scale floor in I_h(nu*).
    Kept for accuracy diagnostics on fine grids.

Analytic ingredients (shared with the Hessian and the flow):
  * grad ell(a) = (4 c1 |a|^2 + 2 c2) a;
  * grad_a Psi by blocks of the prototype b(x,a) = tanh(a1.x + a2) a0,
    reusing the cocycle gradients p[k, i] from the forward pass:
        z = a1 . X_i(t_k) + a2,
        block a0: tanh(z) p_{i,k}
        block a1: (1 - tanh^2 z) (a0 . p_{i,k}) X_i(t_k)
        block a2: (1 - tanh^2 z) (a0 . p_{i,k})
    Cost O(N M d1) per time slice; no finite differences, no extra error.

Finite-difference safeguards (memoria):
  (a) nu_l are node MASSES, not densities, but the cell volume is constant,
      so log(cell volume) is constant and killed by D_a: differentiate
      log nu_l directly.
  (b) work in log space (logsumexp upstream), never log(exp(...)).
  (c) floor max(log nu, LOG_FLOOR) at ~zero-mass nodes: their contribution is
      weighted by nu and negligible, but without the floor they produce NaNs
      when squared.
"""
import numpy as np

from .solver import (b_nd, cocycle_gradients, ell_nd, forward_pass,
                     grad_ell_nd)

LOG_FLOOR = -40.0


# =============================================================================
# Analytic grad_a Psi by blocks / Psi values
# =============================================================================
def grad_a_psi(X, p, A, d1):
    """grad_a Psi_{t_k}(a_l) = (1/N) sum_i [grad_a b(X_i(t_k), a_l)]^T p_{i,k}.

    X : (K+1, N, d1) forward particles
    p : (K+1, N, d1) cocycle gradients grad_x u_t(X_i(t))
    A : (M, 2 d1 + 1) arbitrary parameter points (not necessarily a grid)
    -> (K+1, M, d') with d' = 2 d1 + 1
    """
    K1, N, _ = X.shape
    M = A.shape[0]
    dpr = 2 * d1 + 1
    a0 = A[:, :d1]
    a1 = A[:, d1:2 * d1]
    a2 = A[:, 2 * d1]

    out = np.zeros((K1, M, dpr))
    for k in range(K1):
        Xk, pk = X[k], p[k]                          # (N, d1)
        z = Xk @ a1.T + a2[None, :]                  # (N, M)
        th = np.tanh(z)
        sech2 = 1.0 - th ** 2
        a0p = pk @ a0.T                              # (N, M) = a0_l . p_{i,k}
        out[k, :, :d1] = np.einsum('nm,nd->md', th, pk) / N
        out[k, :, d1:2 * d1] = np.einsum('nm,nm,nd->md', sech2, a0p, Xk) / N
        out[k, :, 2 * d1] = np.einsum('nm,nm->m', sech2, a0p) / N
    return out


def psi_values(X, p, A, d1):
    """Psi_{t_k}(a_l) = (1/N) sum_i b(X_i(t_k), a_l) . p_{i,k}  -> (K+1, M).

    Same quantity as the Gibbs exponent 'term' in the solver; also feeds the
    drift potential of the flow and the Hessian gradient G(nu)."""
    K1, N, _ = X.shape
    M = A.shape[0]
    Psi = np.zeros((K1, M))
    for k in range(K1):
        bXk = b_nd(X[k], A, d1)
        Psi[k] = np.einsum('nmd,nd->m', bXk, p[k]) / N
    return Psi


# =============================================================================
# Finite-difference gradient on the tensor grid
# =============================================================================
def fd_gradient(field, mp_per_axis, R=2.0):
    """Per-axis finite differences of a grid function (centered interior,
    one-sided boundary).  field : (K+1, M) -> (K+1, M, D)."""
    mp = tuple(int(m) for m in mp_per_axis)
    D = len(mp)
    K1, M = field.shape
    assert M == int(np.prod(mp)), f"grid mismatch: M={M} vs prod(mp)={np.prod(mp)}"
    hs = [2.0 * R / (m - 1) for m in mp]
    f_t = field.reshape((K1,) + mp)
    out = np.empty((K1,) + mp + (D,))
    for j in range(D):
        out[..., j] = np.gradient(f_t, hs[j], axis=1 + j, edge_order=1)
    return out.reshape(K1, M, D)


def grad_a_log_nu(nu, mp_per_axis, R=2.0, log_floor=LOG_FLOOR, log_nu=None):
    """D_a[log nu_k](a_l) with the safeguards (a)-(c).  -> (K+1, M, D)."""
    if log_nu is None:
        with np.errstate(divide="ignore"):
            log_nu = np.log(nu)
    log_nu = np.maximum(log_nu, log_floor)           # safeguard (c)
    return fd_gradient(log_nu, mp_per_axis, R=R)


# =============================================================================
# Assembly
# =============================================================================
def time_weights(K, dt):
    """Trapezoidal weights on t: w_0 = w_K = dt/2, w_k = dt otherwise."""
    w = np.full(K + 1, dt)
    w[0] = w[-1] = dt / 2
    return w


def fisher_integrand_field(nu, mp_per_axis, A, eps, *, R=2.0, c1=0.05, c2=0.5,
                           mode="fd", Psi=None, gpsi=None, log_nu=None):
    """Integrand field G_{k,l,:} (see module docstring)  -> (K+1, M, D)."""
    glog = grad_a_log_nu(nu, mp_per_axis, R=R, log_nu=log_nu)
    if mode == "fd":
        assert Psi is not None, "mode='fd' needs the Psi values"
        pot = Psi + eps * ell_nd(A, c1, c2)[None, :]
        return eps * glog + fd_gradient(pot, mp_per_axis, R=R)
    elif mode == "analytic":
        assert gpsi is not None, "mode='analytic' needs grad_a Psi"
        return eps * glog + eps * grad_ell_nd(A, c1, c2)[None, :, :] + gpsi
    raise ValueError(f"unknown mode {mode!r}")


def fisher_information(nu, mp_per_axis, A, eps, dt, *, R=2.0, c1=0.05,
                       c2=0.5, mode="fd", Psi=None, gpsi=None, log_nu=None,
                       return_field=False):
    """Assemble I_h(nu) = sum_k w_k sum_l |G_{k,l}|^2 nu_{k,l}."""
    G = fisher_integrand_field(nu, mp_per_axis, A, eps, R=R, c1=c1, c2=c2,
                               mode=mode, Psi=Psi, gpsi=gpsi, log_nu=log_nu)
    K1 = nu.shape[0]
    w = time_weights(K1 - 1, dt)
    dens = (G ** 2).sum(axis=2) * nu                 # (K+1, M)
    I_h = float(np.einsum('k,km->', w, dens))
    if return_field:
        return I_h, G
    return I_h


def fisher_from_control(nu, x0, y, A, d1, mp_per_axis, eps, *, T=1.0,
                        R=2.0, c1=0.05, c2=0.5, grad_fn=None, mode="fd",
                        return_state=False):
    """One-call evaluation of I_h(nu): forward pass + cocycle with the current
    control nu, then assembly in the requested mode."""
    K = nu.shape[0] - 1
    dt = T / K
    X, Phi = forward_pass(nu, x0, A, d1, dt, K)
    p = cocycle_gradients(X, Phi, y, grad_fn=grad_fn)
    Psi = psi_values(X, p, A, d1) if mode == "fd" else None
    gpsi = grad_a_psi(X, p, A, d1) if mode == "analytic" else None
    I_h = fisher_information(nu, mp_per_axis, A, eps, dt, R=R, c1=c1, c2=c2,
                             mode=mode, Psi=Psi, gpsi=gpsi)
    if return_state:
        return I_h, {"X": X, "Phi": Phi, "p": p, "Psi": Psi, "gpsi": gpsi}
    return I_h
