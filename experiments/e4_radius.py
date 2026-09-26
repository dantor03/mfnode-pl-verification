"""E4 (optional) — Radius of the PL neighborhood.

Sweep increasing perturbation amplitudes until the ratio I / (J - J*) drops
below c(eps) (from E1) or the descent stops being exponential: the amplitude
where this happens estimates the radius of the PL neighborhood.

Usage:
    python experiments/e4_radius.py --problem moons --grid asym --eps 0.05 --c 0.1
"""
import argparse
import json

import numpy as np

from common import GRIDS, R_GRID, RESULTS, T, get_problem, load_ckpt
from plnode import fisher, perturbations
from plnode.gradient_flow import kl_to
from plnode.solver import cost, forward_pass, log_prior

AMPLITUDES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
N_PER_AMP = 6


def main(problem_name, grid_name, eps, c_ref, seed=0):
    prob = get_problem(problem_name)
    mp = GRIDS[grid_name]
    d = load_ckpt(problem_name, grid_name, eps)
    nu_star, A = d["nu"], d["A"]
    K = nu_star.shape[0] - 1
    dt = T / K
    log_nu_inf = log_prior(A)

    X, _ = forward_pass(nu_star, prob["x0"], A, 2, dt, K)
    J_star, _, _ = cost(X, prob["y"], nu_star, log_nu_inf, eps, dt,
                        terminal_fn=prob["terminal_fn"])

    rng = np.random.default_rng(seed)
    rows = []
    print(f"{'delta':>7s} {'min ratio':>10s} {'frac < c_ref':>12s}")
    for delta in AMPLITUDES:
        ratios = []
        for _ in range(N_PER_AMP):
            nu = perturbations.mixture_perturbation(nu_star, delta, rng)
            Xp, _ = forward_pass(nu, prob["x0"], A, 2, dt, K)
            J, _, _ = cost(Xp, prob["y"], nu, log_nu_inf, eps, dt,
                           terminal_fn=prob["terminal_fn"])
            I_h = fisher.fisher_from_control(
                nu, prob["x0"], prob["y"], A, 2, mp, eps, T=T, R=R_GRID,
                grad_fn=prob["grad_fn"])
            gap = J - J_star
            if gap > 0:
                ratios.append(I_h / gap)
                rows.append(dict(delta=delta, ratio=I_h / gap, J_gap=gap,
                                 lambda2=kl_to(nu, nu_star, dt)))
        frac = float(np.mean([r < c_ref for r in ratios])) if ratios else np.nan
        print(f"{delta:7.3f} {min(ratios):10.4f} {frac:12.2f}")

    path = RESULTS / f"e4_radius_{problem_name}_{grid_name}_eps{eps:g}.json"
    path.write_text(json.dumps(dict(c_ref=c_ref, eps=eps, rows=rows), indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--grid", choices=list(GRIDS), default="asym")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--c", type=float, required=True,
                    help="c(eps) reference from E1")
    a = ap.parse_args()
    main(a.problem, a.grid, a.eps, a.c)
