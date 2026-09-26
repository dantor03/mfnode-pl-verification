"""Wasserstein gradient flow of J via finite-volume Fokker-Planck steps.

For each time slice t_k (the K+1 slices evolve in parallel in descent time s,
coupled through Psi, which depends on the whole curve nu^s):

    d_s nu_t^s = div_a( nu_t^s grad_a Phi_t^{(s)} ) + eps Lap_a nu_t^s,
    Phi_t(a) := Psi_t(a) + eps ell(a)        (drift potential),

on [-R, R]^{d'} with NO-FLUX boundary conditions.

Scheme: finite volume on the tensor grid (node masses, constant cell volume),
explicit Euler in s, with the SCHARFETTER-GUMMEL (exponential-fitting) face
flux.  For the face between neighboring cells L, R along axis j (spacing h):

    z      = (Phi_L - Phi_R) / eps                    (face Peclet number)
    F_{LR} = (eps / h^2) [ B(-z) m_L - B(z) m_R ],    B(x) = x / (e^x - 1),

which reduces to centered diffusion for z -> 0 and to upwind for |z| >> 1,
conserves mass exactly, preserves positivity under the explicit CFL, and --
crucially -- its steady state satisfies m_R / m_L = e^{(Phi_L - Phi_R)/eps},
i.e. the DISCRETE Gibbs measure: the fixed point nu* of the Picard solver is
an exact equilibrium of the discrete flow.  (A plain upwind drift, the naive
choice, has an O(h) numerically-diffused steady state != discrete Gibbs, which
would put an artificial floor on the J - J* decay curves of E2.)

The upwind scheme is kept as an option (flux="upwind") for comparison.

CFL: the exact positivity bound is computed from the assembled outflow
coefficients, ds <= safety / max_cell sum_faces(outflow rate).

Cost-control tricks (see paper Sec. 3.3):
  * pilot uniform grid (M = 5^5) for flow experiments; asymmetric M = 11025
    reserved for nu* and E1;
  * FREEZE Psi for `refresh_every` steps of s, recomputing the forward pass
    only then (Psi is Lipschitz in nu; per-step change is O(ds)).

The a <-> -a symmetry is preserved by the flow (odd drift potential gradient,
even diffusion, symmetric grid): used as a drift-in-s correctness test.
"""
from dataclasses import dataclass, field

import numpy as np

from .fisher import fisher_information, grad_a_psi, psi_values, time_weights
from .solver import cocycle_gradients, cost, ell_nd, forward_pass, log_prior


# =============================================================================
# Drift potential and face coefficients
# =============================================================================
def drift_potential(Psi, A, eps, c1=0.05, c2=0.5):
    """Phi_{k,l} = Psi_k(a_l) + eps ell(a_l)  -> (K+1, M)."""
    return Psi + eps * ell_nd(A, c1, c2)[None, :]


def _bernoulli(x):
    """B(x) = x / (e^x - 1), stable near 0 and for large |x|."""
    out = np.empty_like(x)
    small = np.abs(x) < 1e-8
    out[small] = 1.0 - 0.5 * x[small]
    xs = x[~small]
    out[~small] = xs / np.expm1(xs)
    return out


def _face_rates(pot, mp, eps, R):
    """Per-axis face rate coefficients (rL, rR) of the SG flux.

    Flux through a face (mass/time): F = rL * m_L - rR * m_R with
    rL = (eps/h^2) B(-z), rR = (eps/h^2) B(z), z = (Phi_L - Phi_R)/eps.
    Returns a list over axes of (rL, rR) arrays shaped like the face slabs.
    """
    D = len(mp)
    K1 = pot.shape[0]
    hs = [2.0 * R / (m - 1) for m in mp]
    pot_t = pot.reshape((K1,) + mp)
    rates = []
    for j in range(D):
        ax = 1 + j
        lo = [slice(None)] * (D + 1)
        hi = [slice(None)] * (D + 1)
        lo[ax] = slice(None, -1)
        hi[ax] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        z = (pot_t[lo] - pot_t[hi]) / eps
        coef = eps / hs[j] ** 2
        rates.append((coef * _bernoulli(-z), coef * _bernoulli(z), lo, hi))
    return rates


def stable_ds(rates, mp, K1, safety=0.9):
    """Exact positivity bound: ds <= safety / max_cell (total outflow rate)."""
    outflow = np.zeros((K1,) + tuple(mp))
    for rL, rR, lo, hi in rates:
        outflow[lo] += rL       # cell L loses at rate rL m_L through this face
        outflow[hi] += rR       # cell R loses at rate rR m_R
    return safety / float(outflow.max())


def fp_step(nu, rates, mp, ds):
    """One explicit Euler step with precomputed face rates (mass form)."""
    K1 = nu.shape[0]
    m_t = nu.reshape((K1,) + tuple(mp))
    dm = np.zeros_like(m_t)
    for rL, rR, lo, hi in rates:
        flux = rL * m_t[lo] - rR * m_t[hi]     # positive: mass L -> R
        dm[lo] -= ds * flux
        dm[hi] += ds * flux
    return (m_t + dm).reshape(K1, -1)


def upwind_rates(nu_shape, gpsi, A, mp, eps, R, c1=0.05, c2=0.5):
    """Legacy upwind + centered-diffusion face rates (kept for comparison).

    Same (rL, rR) interface as _face_rates: with face velocity u (average of
    node drift -V), rL = max(u,0)/h + eps/h^2, rR = max(-u,0)/h + eps/h^2.
    """
    from .solver import grad_ell_nd
    D = len(mp)
    K1 = nu_shape[0]
    hs = [2.0 * R / (m - 1) for m in mp]
    V = gpsi + eps * grad_ell_nd(A, c1, c2)[None, :, :]
    rates = []
    for j in range(D):
        ax = 1 + j
        lo = [slice(None)] * (D + 1)
        hi = [slice(None)] * (D + 1)
        lo[ax] = slice(None, -1)
        hi[ax] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        Vj = V[..., j].reshape((K1,) + tuple(mp))
        u = -0.5 * (Vj[lo] + Vj[hi])
        rL = np.maximum(u, 0.0) / hs[j] + eps / hs[j] ** 2
        rR = np.maximum(-u, 0.0) / hs[j] + eps / hs[j] ** 2
        rates.append((rL, rR, lo, hi))
    return rates


# =============================================================================
# Full descent loop
# =============================================================================
@dataclass
class FlowResult:
    nu: np.ndarray                       # final control (K+1, M)
    s_grid: np.ndarray                   # descent times where J, I were logged
    J_history: np.ndarray
    I_history: np.ndarray
    mass_error: float                    # max |sum_l nu - 1| along the run
    min_mass: float                      # most negative node mass seen
    ds: float
    extras: dict = field(default_factory=dict)


def gradient_flow(nu0, x0, y, A, d1, mp_per_axis, eps, *, T=1.0, R=2.0,
                  c1=0.05, c2=0.5, n_steps=1000, ds=None, refresh_every=5,
                  grad_fn=None, terminal_fn=None, renormalize=True,
                  flux="sg", fisher_mode="fd", verbose=False):
    """Integrate the Fokker-Planck gradient flow from nu0 for n_steps.

    Psi (hence the drift potential and the forward pass) is refreshed every
    `refresh_every` steps; J and I_h are logged at each refresh.
    """
    mp = tuple(int(m) for m in mp_per_axis)
    K = nu0.shape[0] - 1
    dt = T / K
    log_nu_inf = log_prior(A, c1, c2)
    nu = nu0.copy()

    s_log, J_log, I_log = [], [], []
    mass_err, min_mass = 0.0, np.inf
    s, step = 0.0, 0
    ds_used = ds

    while step < n_steps:
        # --- refresh Psi with the current control ---------------------------
        X, Phi = forward_pass(nu, x0, A, d1, dt, K)
        p = cocycle_gradients(X, Phi, y, grad_fn=grad_fn)
        Psi = psi_values(X, p, A, d1)
        pot = drift_potential(Psi, A, eps, c1, c2)
        if flux == "sg":
            rates = _face_rates(pot, mp, eps, R)
        else:
            gpsi = grad_a_psi(X, p, A, d1)
            rates = upwind_rates(nu.shape, gpsi, A, mp, eps, R, c1, c2)
        if ds_used is None:
            ds_used = stable_ds(rates, mp, K + 1)

        Jval, _, _ = cost(X, y, nu, log_nu_inf, eps, dt, terminal_fn=terminal_fn)
        gpsi_I = grad_a_psi(X, p, A, d1) if fisher_mode == "analytic" else None
        I_h = fisher_information(nu, mp, A, eps, dt, R=R, c1=c1, c2=c2,
                                 mode=fisher_mode, Psi=Psi, gpsi=gpsi_I)
        s_log.append(s)
        J_log.append(Jval)
        I_log.append(I_h)
        if verbose:
            print(f"  s={s:9.4f}  step={step:6d}  J={Jval:.8f}  I_h={I_h:.3e}")

        # --- frozen-Psi inner steps ----------------------------------------
        for _ in range(min(refresh_every, n_steps - step)):
            nu = fp_step(nu, rates, mp, ds_used)
            min_mass = min(min_mass, float(nu.min()))
            mass_err = max(mass_err, float(np.abs(nu.sum(axis=1) - 1.0).max()))
            if renormalize:
                nu = np.clip(nu, 0.0, None)
                nu /= nu.sum(axis=1, keepdims=True)
            s += ds_used
            step += 1

    return FlowResult(nu=nu, s_grid=np.array(s_log), J_history=np.array(J_log),
                      I_history=np.array(I_log), mass_error=mass_err,
                      min_mass=min_mass, ds=ds_used,
                      extras={"refresh_every": refresh_every, "eps": eps,
                              "flux": flux})


# =============================================================================
# The J-I-flow consistency test (paper figure, Phase 2 step 9)
# =============================================================================
def djds_test(nu, x0, y, A, d1, mp_per_axis, eps, *, T=1.0, R=2.0,
              c1=0.05, c2=0.5, ds=1e-4, grad_fn=None, terminal_fn=None,
              flux="sg", fisher_mode="fd"):
    """Verify [J(nu^{s+ds}) - J(nu^s)] / ds  ~  -I_h(nu^s).

    Closes the circle J - I - flow.  Returns (lhs, rhs, rel_err).
    """
    mp = tuple(int(m) for m in mp_per_axis)
    K = nu.shape[0] - 1
    dt = T / K
    log_nu_inf = log_prior(A, c1, c2)

    X, Phi = forward_pass(nu, x0, A, d1, dt, K)
    p = cocycle_gradients(X, Phi, y, grad_fn=grad_fn)
    Psi = psi_values(X, p, A, d1)
    pot = drift_potential(Psi, A, eps, c1, c2)
    if flux == "sg":
        rates = _face_rates(pot, mp, eps, R)
    else:
        rates = upwind_rates(nu.shape, grad_a_psi(X, p, A, d1), A, mp, eps, R,
                             c1, c2)
    J0, _, _ = cost(X, y, nu, log_nu_inf, eps, dt, terminal_fn=terminal_fn)
    gpsi_I = grad_a_psi(X, p, A, d1) if fisher_mode == "analytic" else None
    I_h = fisher_information(nu, mp, A, eps, dt, R=R, c1=c1, c2=c2,
                             mode=fisher_mode, Psi=Psi, gpsi=gpsi_I)

    nu1 = fp_step(nu, rates, mp, ds)
    nu1 = np.clip(nu1, 0.0, None)
    nu1 /= nu1.sum(axis=1, keepdims=True)
    X1, _ = forward_pass(nu1, x0, A, d1, dt, K)
    J1, _, _ = cost(X1, y, nu1, log_nu_inf, eps, dt, terminal_fn=terminal_fn)

    lhs = (J1 - J0) / ds
    rhs = -I_h
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-300)
    return lhs, rhs, rel


def kl_to(nu, nu_ref, dt):
    """Integrated relative entropy int KL(nu_t || nu_ref_t) dt (lambda^2 of DD)."""
    K1 = nu.shape[0]
    w = time_weights(K1 - 1, dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nu > 0, nu * np.log(np.where(nu > 0, nu, 1.0)
                                             / np.clip(nu_ref, 1e-300, None)), 0.0)
    return float(np.einsum('k,km->', w, ratio))
