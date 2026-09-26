"""E2 — Gradient descent near nu*: measured exponential rate c-hat vs c(eps).

Initialize the Wasserstein gradient flow at controlled TV distances from nu*,
integrate until J - J* drops 3-4 orders of magnitude, fit the slope of
log(J(nu^s) - J*) vs s, and compare c-hat with the PL prediction (decay
>= e^{-2 c s}).  Along the same descents the integrated relative entropy
int KL(nu^s_t || nu*_t) dt (the distance of DD Thm 5.1) is recorded and its
exponential rate fitted as well: convergence of the trajectory, not only of
the cost.  Several seeds and amplitudes: insensitivity of the rate to
the initial direction IS the PL phenomenon.

Runs on the pilot grid by design (cost decision of the plan).

Usage:
    python experiments/e2_descent.py --problem moons --eps 0.05 --n-steps 3000
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG_DIR, GRIDS, R_GRID, RESULTS, T, get_problem, load_ckpt
from plnode import gradient_flow, perturbations
from plnode.solver import cost, forward_pass, log_prior


def fit_rate(s, g, return_window=False):
    """Exponential rate of the decaying stretch of (s, gap).

    The gap decays exponentially and then floors at the solver-tolerance
    level (~(diffTV/omega)^2 x curvature), around which it may drift
    non-monotonically at negligible amplitude.  Estimator:
      floor := median of the last 10% of the trajectory;
      window := decaying stretch (s <= argmin g) with
                30*floor < g < g[0]/3
    (past the initial transient, well above the floor).  Log-linear fit on
    the window; NaN if fewer than 4 points qualify.
    """
    tail = max(1, len(g) // 10)
    floor = max(float(np.median(g[-tail:])), 1e-16)
    decaying = np.arange(len(g)) <= int(np.argmin(g))
    sel = decaying & (g > 30 * floor) & (g < g[0] / 3)
    if sel.sum() < 4:
        return (np.nan, sel) if return_window else np.nan
    rate = -np.polyfit(s[sel], np.log(g[sel]), 1)[0]
    return (rate, sel) if return_window else rate


def main(problem_name, eps, deltas=(0.02, 0.05, 0.1), seeds=(0, 1, 2),
         n_steps=3000, refresh_every=5, grid_name="pilot"):
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
    print(f"J* = {J_star:.8f}  (eps={eps}, {problem_name}/{grid_name})")

    fig, ax = plt.subplots(figsize=(6, 4.4))
    fig_kl, ax_kl = plt.subplots(figsize=(6, 4.4))
    runs = []
    for delta in deltas:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            nu0 = perturbations.mixture_perturbation(nu_star, delta, rng)
            res = gradient_flow.gradient_flow(
                nu0, prob["x0"], prob["y"], A, d1=2, mp_per_axis=mp, eps=eps,
                T=T, R=R_GRID, n_steps=n_steps, refresh_every=refresh_every,
                grad_fn=prob["grad_fn"], terminal_fn=prob["terminal_fn"],
                nu_ref=nu_star)
            gap = res.J_history - J_star
            pos = gap > 1e-16
            s, g = res.s_grid[pos], gap[pos]
            rate = fit_rate(s, g)
            kl = res.extras["kl_history"]
            kl_pos = kl > 1e-16
            kl_rate = fit_rate(res.s_grid[kl_pos], kl[kl_pos])
            runs.append(dict(delta=delta, seed=seed, rate=float(rate),
                             s=list(map(float, s)), gap=list(map(float, g)),
                             kl_rate=float(kl_rate),
                             s_kl=list(map(float, res.s_grid[kl_pos])),
                             kl=list(map(float, kl[kl_pos]))))
            ax_kl.semilogy(res.s_grid[kl_pos], kl[kl_pos], alpha=0.75)
            ax.semilogy(s, g, alpha=0.75,
                        label=rf"$\delta$={delta}, seed={seed}, "
                              rf"$\hat c$={rate:.3g}")
            print(f"  delta={delta:5.3f} seed={seed}  c-hat = {rate:.4f}  "
                  f"KL rate = {kl_rate:.4f}  (mass_err={res.mass_error:.1e})")

    ax.set_xlabel("descent time $s$")
    ax.set_ylabel(r"$J(\nu^s) - J^*$")
    ax.set_title(rf"E2 — descent, {problem_name}, $\varepsilon$={eps}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    out = FIG_DIR / f"e2_descent_{problem_name}_eps{eps:g}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")

    ax_kl.set_xlabel("descent time $s$")
    ax_kl.set_ylabel(r"$\int \mathcal{E}(\nu^s_t\,|\,\nu^*_t)\,dt$")
    ax_kl.set_title(rf"E2 — trajectory, {problem_name}, $\varepsilon$={eps}")
    ax_kl.grid(alpha=0.3)
    out_kl = FIG_DIR / f"e2_kl_{problem_name}_eps{eps:g}.png"
    fig_kl.tight_layout()
    fig_kl.savefig(out_kl, dpi=150)
    print(f"-> {out_kl}")

    out_json = RESULTS / f"e2_{problem_name}_eps{eps:g}.json"
    with open(out_json, "w") as fh:
        json.dump(dict(J_star=float(J_star), eps=eps, runs=runs), fh)
    print(f"-> {out_json}")
    return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--n-steps", type=int, default=3000)
    ap.add_argument("--refresh-every", type=int, default=5)
    ap.add_argument("--grid", default="pilot",
                    help="checkpoint grid (pilot/asym or *flow refined)")
    a = ap.parse_args()
    main(a.problem, a.eps, n_steps=a.n_steps, refresh_every=a.refresh_every,
         grid_name=a.grid)
