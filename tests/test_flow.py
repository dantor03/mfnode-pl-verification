import numpy as np

from plnode import gradient_flow, perturbations, solver
from plnode.fisher import psi_values
from plnode.solver import cocycle_gradients, forward_pass


def _rates(nu, t):
    K = nu.shape[0] - 1
    X, Phi = forward_pass(nu, t["x0"], t["A"], 2, 1.0 / K, K)
    p = cocycle_gradients(X, Phi, t["y"])
    Psi = psi_values(X, p, t["A"], 2)
    pot = gradient_flow.drift_potential(Psi, t["A"], t["eps"])
    return gradient_flow._face_rates(pot, t["mp"], t["eps"], t["R"])


def test_fp_step_conserves_mass_and_positivity(tiny):
    t = tiny
    rng = np.random.default_rng(0)
    nu = perturbations.mixture_perturbation(t["res"].nu, 0.3, rng)
    rates = _rates(nu, t)
    ds = gradient_flow.stable_ds(rates, t["mp"], nu.shape[0])
    nu1 = gradient_flow.fp_step(nu, rates, t["mp"], ds)
    np.testing.assert_allclose(nu1.sum(axis=1), 1.0, atol=1e-12)
    assert nu1.min() >= 0.0, f"positivity violated: {nu1.min():.2e}"


def test_gibbs_is_steady_state_of_sg_flux(tiny):
    """The discrete Gibbs measure of the CURRENT potential is an exact
    equilibrium of the Scharfetter-Gummel step (the design property)."""
    from scipy.special import logsumexp
    t = tiny
    nu = t["res"].nu
    K = nu.shape[0] - 1
    X, Phi = forward_pass(nu, t["x0"], t["A"], 2, 1.0 / K, K)
    p = cocycle_gradients(X, Phi, t["y"])
    Psi = psi_values(X, p, t["A"], 2)
    pot = gradient_flow.drift_potential(Psi, t["A"], t["eps"])
    # Gibbs of this exact potential
    log_g = -pot / t["eps"]
    log_g -= logsumexp(log_g, axis=1, keepdims=True)
    nu_g = np.exp(log_g)
    rates = gradient_flow._face_rates(pot, t["mp"], t["eps"], t["R"])
    ds = gradient_flow.stable_ds(rates, t["mp"], nu.shape[0])
    nu1 = gradient_flow.fp_step(nu_g, rates, t["mp"], ds)
    np.testing.assert_allclose(nu1, nu_g, atol=1e-14)


def test_flow_preserves_symmetry(tiny):
    """Odd drift + even diffusion + symmetric grid preserve nu(-a) = nu(a)."""
    t = tiny
    rng = np.random.default_rng(1)
    nu0 = perturbations.symmetric_perturbation(t["res"].nu, t["mp"], 0.3, rng)
    assert solver.symmetry_error(nu0, t["mp"]) < 1e-14
    res = gradient_flow.gradient_flow(
        nu0, t["x0"], t["y"], t["A"], 2, t["mp"], t["eps"],
        T=1.0, R=t["R"], n_steps=50, refresh_every=5)
    err = solver.symmetry_error(res.nu, t["mp"])
    assert err < 1e-12, f"symmetry drift in s: {err:.2e}"


def test_flow_decreases_J(tiny):
    t = tiny
    rng = np.random.default_rng(2)
    nu0 = perturbations.mixture_perturbation(t["res"].nu, 0.3, rng)
    res = gradient_flow.gradient_flow(
        nu0, t["x0"], t["y"], t["A"], 2, t["mp"], t["eps"],
        T=1.0, R=t["R"], n_steps=100, refresh_every=5)
    assert res.J_history[-1] < res.J_history[0]
    # I_h should also shrink as we approach the minimizer
    assert res.I_history[-1] < res.I_history[0]


def test_djds_identity(tiny):
    """[J(nu+ds) - J(nu)] / ds ~ -I_h(nu) with matching discretizations."""
    t = tiny
    rng = np.random.default_rng(3)
    nu0 = perturbations.mixture_perturbation(t["res"].nu, 0.2, rng)
    lhs, rhs, rel = gradient_flow.djds_test(
        nu0, t["x0"], t["y"], t["A"], 2, t["mp"], t["eps"],
        T=1.0, R=t["R"], ds=1e-4)
    assert lhs < 0 and rhs < 0
    assert rel < 0.15, f"dJ/ds vs -I_h relative mismatch {rel:.3f}"


def test_flow_from_fixed_point_stays(tiny):
    """Starting AT nu*, the SG flow must stay there (exact equilibrium)."""
    t = tiny
    res = gradient_flow.gradient_flow(
        t["res"].nu, t["x0"], t["y"], t["A"], 2, t["mp"], t["eps"],
        T=1.0, R=t["R"], n_steps=20, refresh_every=5)
    assert abs(res.J_history[-1] - res.J_history[0]) < 1e-6, (
        res.J_history[0], res.J_history[-1])


def test_flow_tracks_relative_entropy_to_nu_star(tiny):
    """With nu_ref = nu*, the logged int KL(nu^s || nu*) dt starts at the
    perturbation's lambda^2 and decreases along the descent."""
    t = tiny
    rng = np.random.default_rng(2)
    nu0 = perturbations.mixture_perturbation(t["res"].nu, 0.2, rng)
    res = gradient_flow.gradient_flow(
        nu0, t["x0"], t["y"], t["A"], 2, t["mp"], t["eps"],
        T=1.0, R=t["R"], n_steps=100, refresh_every=5, nu_ref=t["res"].nu)
    kl = res.extras["kl_history"]
    assert kl.shape == res.J_history.shape
    assert np.all(kl >= 0.0)
    np.testing.assert_allclose(kl[0], gradient_flow.kl_to(nu0, t["res"].nu,
                                                          1.0 / t["K"]))
    assert kl[-1] < 0.5 * kl[0]
