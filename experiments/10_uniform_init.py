"""Task: initialize Picard with the UNIFORM distribution instead of the prior.

Replaces nu^(0) = nu^inf (Gibbs prior) by nu^(0) = 1/M (uniform on the grid,
no node privileged) and runs Picard *directly* at each eps in EPS_PROBE --
deliberately without continuation, so this also probes the basin of
attraction from far away.  Each run is compared against the frozen
continuation checkpoint at the same eps (results/nu_star/): if both reach the
same fixed point (TV ~ tol/omega), the fixed point does not depend on the
initialization.

Usage:
    python experiments/10_uniform_init.py                  # both problems
    python experiments/10_uniform_init.py --problem moons
"""
import argparse
import time

import numpy as np

from common import (GRIDS, K, OMEGA, R_GRID, RESULTS, T, TOL, get_problem,
                    load_ckpt)
from plnode import solver

EPS_PROBE = (0.5, 0.1, 0.05)
OUT_DIR = RESULTS / "uniform_init"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run(problem_name, grid_name="pilot", max_iter=200):
    prob = get_problem(problem_name)
    mp = GRIDS[grid_name]
    A, _ = solver.build_grid(mp, R=R_GRID)
    M = A.shape[0]
    nu_uniform = np.full((K + 1, M), 1.0 / M)

    print(f"\n=== uniform init: {problem_name} on {grid_name} (M={M}) ===",
          flush=True)
    print(f"{'eps':>6s} {'iters':>6s} {'conv':>5s} {'J_unif':>10s} "
          f"{'J_ckpt':>10s} {'TV(unif,ckpt)':>14s} {'sym_err':>9s}",
          flush=True)
    for eps in EPS_PROBE:
        out = OUT_DIR / f"{problem_name}_{grid_name}_eps{eps:g}.npz"
        if out.exists():
            print(f"{eps:6.3f}  -- already done ({out.name}) --", flush=True)
            continue
        t0 = time.time()
        res = solver.picard(
            prob["x0"], prob["y"], A, d1=2, T=T, K=K, eps=eps,
            omega=OMEGA, max_iter=max_iter, tol=TOL, verbose=False,
            nu_init=nu_uniform,
            grad_fn=prob["grad_fn"], terminal_fn=prob["terminal_fn"])
        ref = load_ckpt(problem_name, grid_name, eps)
        tv = 0.5 * np.abs(res.nu - ref["nu"]).sum(axis=1).max()
        sym = solver.symmetry_error(res.nu, mp)
        np.savez_compressed(
            out, nu=res.nu, eps=eps, iters=res.iters,
            converged=res.converged, diff_final=res.diff_history[-1],
            cost_final=res.cost_history[-1], tv_to_ckpt=tv,
            cost_ckpt=float(ref["cost_final"]))
        print(f"{eps:6.3f} {res.iters:6d} {str(res.converged):>5s} "
              f"{res.cost_history[-1]:10.5f} {float(ref['cost_final']):10.5f} "
              f"{tv:14.2e} {sym:9.1e}   [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"])
    ap.add_argument("--grid", default="pilot")
    args = ap.parse_args()
    problems = [args.problem] if args.problem else ["moons", "housing"]
    for p in problems:
        run(p, args.grid)
    print("\nDONE uniform-init", flush=True)
