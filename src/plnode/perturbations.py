"""Perturbation generator around nu* (reused by E1, E2, E4).

Generates random admissible perturbations at controlled TV distance, in the
spirit of the Pontryagin test of the memoria: nu_delta = (1 - delta) nu* +
delta mu with mu a random probability on the grid, plus a multiplicative
log-normal tilt variant that stays inside the simplex and explores directions
where nu* is small.

TV distance per slice: TV(nu_delta, nu*) = 0.5 |nu_delta - nu*|_1.
"""
import numpy as np


def mixture_perturbation(nu_star, delta, rng, concentrate=None):
    """nu_delta = (1 - delta) nu* + delta mu, mu ~ Dirichlet-like random.

    concentrate: optional exponent alpha; mu ~ normalized U^alpha which for
    alpha > 1 concentrates the random mass on few nodes.
    TV(nu_delta, nu*) <= delta with equality when supp(mu) misses nu*'s mass.
    """
    K1, M = nu_star.shape
    U = rng.random((K1, M))
    if concentrate is not None:
        U = U ** concentrate
    mu = U / U.sum(axis=1, keepdims=True)
    return (1.0 - delta) * nu_star + delta * mu


def lognormal_tilt(nu_star, sigma, rng):
    """nu_delta ~ nu* exp(sigma xi), xi ~ N(0,1) i.i.d., renormalized.

    Multiplicative tilt: explores the directions weighted by nu* itself and
    keeps the relative entropy to nu* finite and O(sigma^2).
    """
    K1, M = nu_star.shape
    xi = rng.standard_normal((K1, M))
    log_nu = np.log(np.clip(nu_star, 1e-300, None)) + sigma * xi
    log_nu -= log_nu.max(axis=1, keepdims=True)
    nu = np.exp(log_nu)
    return nu / nu.sum(axis=1, keepdims=True)


def symmetric_perturbation(nu_star, mp_per_axis, delta, rng, kind="mixture"):
    """Perturbation that preserves the exact a <-> -a symmetry of nu*.

    Useful to test the symmetry-preservation of the gradient flow while
    starting away from nu*.
    """
    mp = tuple(int(m) for m in mp_per_axis)
    if kind == "mixture":
        nu = mixture_perturbation(nu_star, delta, rng)
    else:
        nu = lognormal_tilt(nu_star, delta, rng)
    K1 = nu.shape[0]
    nu_t = nu.reshape((K1,) + mp)
    flipped = nu_t[(slice(None),) + tuple(slice(None, None, -1) for _ in mp)]
    nu_sym = 0.5 * (nu_t + flipped)
    nu_sym = nu_sym.reshape(K1, -1)
    return nu_sym / nu_sym.sum(axis=1, keepdims=True)


def tv_distance(nu, nu_star):
    """max_k TV(nu_k, nu*_k)."""
    return float(0.5 * np.abs(nu - nu_star).sum(axis=1).max())


def perturbation_cloud(nu_star, deltas, n_per_delta, seed=0, kinds=("mixture",
                                                                    "tilt")):
    """Yield (delta, kind, nu_delta) over the E1 cloud."""
    rng = np.random.default_rng(seed)
    for delta in deltas:
        for _ in range(n_per_delta):
            if "mixture" in kinds:
                yield delta, "mixture", mixture_perturbation(nu_star, delta, rng)
            if "tilt" in kinds:
                yield delta, "tilt", lognormal_tilt(nu_star, delta, rng)
