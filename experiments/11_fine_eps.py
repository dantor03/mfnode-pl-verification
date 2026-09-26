"""Task: refine the eps-continuation between 0.05 and 0.02.

The coarse schedule (0.5, 0.2, 0.1, 0.05, 0.02) fails to converge at
eps = 0.02 (the 0.05 -> 0.02 jump is too large once the Gibbs map loses
contractivity).  Here we warm-start from the frozen eps = 0.05 checkpoint and
continue with the finer schedule EPS_FINE = (0.04, 0.03, 0.025, 0.02),
checking whether smaller continuation steps restore convergence at 0.02.

Checkpoints are saved under the grid tag '<grid>fine' so the frozen paper
outputs are never overwritten; the script is crash-safe (per-eps resume).
For moons, the final iterate is also compared against the flow-refined exact
fixed point (grid tag '<grid>flow') when available.

Usage:
    python experiments/11_fine_eps.py                  # both problems, pilot
    python experiments/11_fine_eps.py --problem moons
"""
import argparse
import time

import numpy as np

from common import (GRIDS, K, OMEGA, R_GRID, T, TOL, ckpt_path, get_problem,
                    load_ckpt, save_ckpt)
from plnode import solver

EPS_FINE = (0.04, 0.03, 0.025, 0.02)


def run(problem_name, grid_name="pilot", max_iter=200, max_iter_last=300):
    prob = get_problem(problem_name)
    mp = GRIDS[grid_name]
    A, _ = solver.build_grid(mp, R=R_GRID)
    fine = grid_name + "fine"

    nu_warm = load_ckpt(problem_name, grid_name, 0.05)["nu"]
    print(f"\n=== fine-eps continuation: {problem_name} on {grid_name} "
          f"(M={A.shape[0]}), warm start eps=0.05 ===", flush=True)
    print(f"{'eps':>6s} {'iters':>6s} {'conv':>5s} {'J':>10s} "
          f"{'diffTV/om':>10s} {'sym_err':>9s} {prob['metric_name']:>9s}",
          flush=True)
    for eps in EPS_FINE:
        path = ckpt_path(problem_name, fine, eps)
        if path.exists():
            d = np.load(path)
            nu_warm = d["nu"]
            print(f"{eps:6.3f}  -- resumed from existing checkpoint --",
                  flush=True)
            continue
        t0 = time.time()
        mi = max_iter_last if eps == EPS_FINE[-1] else max_iter
        res = solver.picard(
            prob["x0"], prob["y"], A, d1=2, T=T, K=K, eps=eps,
            omega=OMEGA, max_iter=mi, tol=TOL, verbose=False,
            nu_init=nu_warm,
            grad_fn=prob["grad_fn"], terminal_fn=prob["terminal_fn"])
        save_ckpt(problem_name, fine, eps, res, A, mp)   # crash-safe
        nu_warm = res.nu
        sym = solver.symmetry_error(res.nu, mp)
        metric = prob["metric"](res.X[-1])
        print(f"{eps:6.3f} {res.iters:6d} {str(res.converged):>5s} "
              f"{res.cost_history[-1]:10.5f} {res.dist_to_fixed_point:10.2e} "
              f"{sym:9.1e} {metric:9.3f}   [{time.time()-t0:.0f}s]",
              flush=True)

    # --- comparisons at eps = 0.02 -----------------------------------------
    nu_fine = np.load(ckpt_path(problem_name, fine, 0.02))["nu"]
    ref_coarse = load_ckpt(problem_name, grid_name, 0.02)
    tv_coarse = 0.5 * np.abs(nu_fine - ref_coarse["nu"]).sum(axis=1).max()
    print(f"TV(fine, coarse-schedule iterate) at eps=0.02: {tv_coarse:.3e}",
          flush=True)
    flow_path = ckpt_path(problem_name, grid_name + "flow", 0.02)
    if flow_path.exists():
        nu_flow = np.load(flow_path)["nu"]
        tv_flow = 0.5 * np.abs(nu_fine - nu_flow).sum(axis=1).max()
        print(f"TV(fine, flow-refined exact fixed point):   {tv_flow:.3e}",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"])
    ap.add_argument("--grid", default="pilot")
    args = ap.parse_args()
    problems = [args.problem] if args.problem else ["moons", "housing"]
    for p in problems:
        run(p, args.grid)
    print("\nDONE fine-eps", flush=True)
