import numpy as np

from plnode import solver


def test_picard_converges(tiny):
    assert tiny["res"].converged
    assert tiny["res"].diff_final < 1e-6


def test_symmetry_machine_precision(tiny):
    """With tanh odd and ell even, nu*(-a) = nu*(a) exactly (memoria Prop.)."""
    err = solver.symmetry_error(tiny["res"].nu, tiny["mp"])
    assert err < 1e-13, f"symmetry error {err:.2e} above machine precision"


def test_nu_is_probability(tiny):
    nu = tiny["res"].nu
    assert np.all(nu >= 0)
    np.testing.assert_allclose(nu.sum(axis=1), 1.0, atol=1e-12)


def test_gibbs_consistency(tiny):
    """At the fixed point, one Gibbs update moves nu by <= diffTV/omega."""
    t = tiny
    res = t["res"]
    log_nu_inf = solver.log_prior(t["A"])
    p = solver.cocycle_gradients(res.X, res.Phi, t["y"])
    nu_gibbs, _ = solver.gibbs_update(res.X, p, t["A"], 2, t["eps"], log_nu_inf)
    tv = 0.5 * np.abs(nu_gibbs - res.nu).sum(axis=1).max()
    assert tv <= 2.0 * res.dist_to_fixed_point + 1e-12


def test_cocycle_identity(tiny):
    """Phi(T) Phi(t)^-1 at t = T is the identity -> p[T] = grad L(X_T)."""
    t = tiny
    res = t["res"]
    p = solver.cocycle_gradients(res.X, res.Phi, t["y"])
    np.testing.assert_allclose(p[-1], res.X[-1] - t["y"], atol=1e-10)
