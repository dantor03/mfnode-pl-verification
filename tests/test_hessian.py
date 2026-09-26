import numpy as np

from plnode import hessian, perturbations


def _op(t, h=1e-5):
    return hessian.hessian_operator(
        t["res"].nu, t["x0"], t["y"], t["A"], d1=2, eps=t["eps"], T=1.0, h=h)


def test_hessian_symmetry(tiny):
    t = tiny
    K1, M = t["res"].nu.shape
    op = _op(t)
    asym = hessian.symmetry_check(op, K1, M, n_pairs=2, seed=0)
    assert asym < 5e-3, f"Hessian asymmetry {asym:.2e}"


def test_hessian_psd_at_minimizer(tiny):
    """<H eta, eta> > 0 for random tangent directions at nu* (stability)."""
    t = tiny
    op = _op(t)
    rng = np.random.default_rng(0)
    for _ in range(3):
        v = hessian.project_tangent(rng.standard_normal(t["res"].nu.shape))
        v /= np.linalg.norm(v)
        q = float((op @ v.ravel()) @ v.ravel())
        assert q > 0, f"negative curvature {q:.3e} at the minimizer"


def test_quadratic_prediction(tiny):
    """<H eta, eta>/2 predicts J(nu* + eta) - J* for small SMOOTH eta.

    A multiplicative lognormal tilt keeps eta proportional to nu*, staying
    inside the quadratic regime (a mixture perturbation would dump mass on
    near-zero nodes, where J is dominated by the log nonlinearity)."""
    t = tiny
    op = _op(t)
    rng = np.random.default_rng(1)
    eta = perturbations.lognormal_tilt(t["res"].nu, 0.02, rng) - t["res"].nu
    quad, excess = hessian.quadratic_prediction_check(
        op, t["res"].nu, t["x0"], t["y"], t["A"], 2, t["eps"], eta, T=1.0)
    assert excess > 0
    assert 0.7 < quad / excess < 1.4, (quad, excess)


def test_lambda_min_lobpcg_matches_dense(tiny):
    """LOBPCG lambda_min vs brute-force dense assembly on the tiny grid."""
    t = tiny
    K1, M = t["res"].nu.shape
    op = hessian.hessian_operator(
        t["res"].nu, t["x0"], t["y"], t["A"], d1=2, eps=t["eps"], T=1.0,
        h=1e-5, support_threshold=1e-6, sigma_null=1e4)
    # dense assembly by applying op to the canonical basis
    n = K1 * M
    H = np.zeros((n, n))
    eye = np.eye(n)
    for j in range(n):
        H[:, j] = op @ eye[:, j]
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    # smallest TANGENT eigenvalue: skip the penalized complement (~sigma)
    tangent = np.array([
        np.linalg.norm(V[:, i][~np.tile(op.mask, K1)]) < 1e-8
        and abs(V[:, i].reshape(K1, M).mean(axis=1)).max() < 1e-8
        for i in range(n)])
    lam_dense = w[tangent].min()
    vals, _ = hessian.lambda_min_lobpcg(op, k=1, tol=1e-6, seed=0)
    assert abs(vals[0] - lam_dense) / abs(lam_dense) < 1e-3, (
        vals[0], lam_dense)
