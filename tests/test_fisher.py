import numpy as np

from plnode import fisher, perturbations, solver


def _fisher_at(nu, t):
    return fisher.fisher_from_control(
        nu, t["x0"], t["y"], t["A"], 2, t["mp"], t["eps"], T=1.0, R=t["R"])


def test_fisher_nonnegative_and_small_at_fixed_point(tiny):
    t = tiny
    I_star = _fisher_at(t["res"].nu, t)
    assert I_star >= 0.0
    rng = np.random.default_rng(0)
    nu_pert = perturbations.mixture_perturbation(t["res"].nu, 0.2, rng)
    I_pert = _fisher_at(nu_pert, t)
    assert I_pert > 10 * I_star, (
        f"I at fixed point ({I_star:.3e}) not << I at perturbation "
        f"({I_pert:.3e})")


def test_grad_a_psi_matches_finite_differences(tiny):
    """Analytic grad_a Psi vs central finite differences of Psi(a).

    Psi(a) can be evaluated at ARBITRARY parameter points (not only grid
    nodes), so the analytic block formulas are checked against small-step
    (h = 1e-5) central differences at random points — an exact test, immune
    to the coarseness of the tensor grid.
    """
    t = tiny
    res = t["res"]
    p = solver.cocycle_gradients(res.X, res.Phi, t["y"])

    rng = np.random.default_rng(0)
    A0 = rng.uniform(-t["R"], t["R"], size=(20, 5))
    gpsi = fisher.grad_a_psi(res.X, p, A0, 2)          # (K1, 20, 5)

    h = 1e-5
    for j in range(5):
        e = np.zeros(5)
        e[j] = h
        Pp = fisher.psi_values(res.X, p, A0 + e, 2)
        Pm = fisher.psi_values(res.X, p, A0 - e, 2)
        fd = (Pp - Pm) / (2 * h)                       # (K1, 20)
        np.testing.assert_allclose(gpsi[..., j], fd, rtol=1e-5, atol=1e-8,
                                   err_msg=f"axis {j}")


def test_gradient_matches_directional_derivative(tiny):
    """<G(nu), v> ~ [J(nu + h v) - J(nu - h v)] / 2h on a tangent direction."""
    from plnode.hessian import gradient, project_tangent
    from plnode.solver import cost, forward_pass, log_prior

    t = tiny
    nu = perturbations.mixture_perturbation(
        t["res"].nu, 0.1, np.random.default_rng(1))
    K = nu.shape[0] - 1
    dt = 1.0 / K
    log_nu_inf = log_prior(t["A"])

    def J_of(nu_):
        X, _ = forward_pass(nu_, t["x0"], t["A"], 2, dt, K)
        return cost(X, t["y"], nu_, log_nu_inf, t["eps"], dt)[0]

    rng = np.random.default_rng(2)
    v = project_tangent(rng.standard_normal(nu.shape))
    v /= np.abs(v).max()
    h = 1e-6
    lhs = (J_of(nu + h * v) - J_of(nu - h * v)) / (2 * h)
    G = gradient(nu, t["x0"], t["y"], t["A"], 2, t["eps"], T=1.0)
    rhs = float((G * v).sum())
    assert abs(lhs - rhs) / (abs(lhs) + 1e-12) < 1e-3, (lhs, rhs)


def test_time_weights_trapezoid():
    w = fisher.time_weights(4, 0.25)
    assert w[0] == w[-1] == 0.125
    np.testing.assert_allclose(w.sum(), 1.0)
