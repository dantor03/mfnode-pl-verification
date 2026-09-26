"""Phase 0.2 — Freeze a reproducible nu* for both benchmarks and all eps.

Runs the eps-continuation (0.5, 0.2, 0.1, 0.05, 0.02) on the requested grid
and saves one checkpoint per (problem, grid, eps) under results/nu_star/.

Usage:
    python experiments/00_compute_nu_star.py --problem moons   --grid pilot
    python experiments/00_compute_nu_star.py --problem housing --grid asym
    python experiments/00_compute_nu_star.py --all              # everything

Cost guide (memoria): pilot (M=3125) ~ 10 s/Picard iteration equivalents,
asym (M=11025) ~ 35 s/it; a full continuation on asym takes ~2-3 h.
"""
import argparse
import time

import numpy as np

from common import (GRIDS, K, MAX_ITER, OMEGA, R_GRID, T, TOL, get_problem,
                    save_ckpt)
from plnode import solver


def run(problem_name, grid_name, eps_schedule=solver.EPS_SCHEDULE,
        max_iter=MAX_ITER, verbose=True):
    """Continuation with per-eps checkpointing and resume.

    Each eps is saved AS SOON as its Picard finishes (crash-safe), and
    existing checkpoints are reused as warm starts instead of recomputed.
    """
    import numpy as np

    from common import ckpt_path

    prob = get_problem(problem_name)
    mp = GRIDS[grid_name]
    A, _ = solver.build_grid(mp, R=R_GRID)
    print(f"\n=== nu* for {problem_name} on grid {grid_name} "
          f"(mp={mp}, M={A.shape[0]}) ===", flush=True)
    t0 = time.time()

    header = (f"{'eps':>6s} {'iters':>6s} {'conv':>5s} {'J':>10s} "
              f"{'diffTV/om':>10s} {'sym_err':>9s} {prob['metric_name']:>9s}")
    print(header, flush=True)
    nu_warm = None
    results = []
    for eps in eps_schedule:
        path = ckpt_path(problem_name, grid_name, eps)
        if path.exists():
            d = np.load(path)
            nu_warm = d["nu"]
            print(f"{eps:6.3f}  -- resumed from existing checkpoint --",
                  flush=True)
            continue
        res = solver.picard(
            prob["x0"], prob["y"], A, d1=2, T=T, K=K, eps=eps,
            omega=OMEGA, max_iter=max_iter, tol=TOL, verbose=verbose,
            nu_init=nu_warm,
            grad_fn=prob["grad_fn"], terminal_fn=prob["terminal_fn"])
        save_ckpt(problem_name, grid_name, eps, res, A, mp)   # crash-safe
        nu_warm = res.nu
        results.append((eps, res))
        sym = solver.symmetry_error(res.nu, mp)
        metric = prob["metric"](res.X[-1])
        print(f"{eps:6.3f} {res.iters:6d} {str(res.converged):>5s} "
              f"{res.cost_history[-1]:10.5f} {res.dist_to_fixed_point:10.2e} "
              f"{sym:9.1e} {metric:9.3f}", flush=True)
    print(f"total continuation time: {time.time() - t0:.1f}s", flush=True)
    print("checkpoints saved to results/nu_star/", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"])
    ap.add_argument("--grid", choices=list(GRIDS), default="pilot")
    ap.add_argument("--all", action="store_true",
                    help="both problems on both grids (long: hours on asym)")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    args = ap.parse_args()

    if args.all:
        for pb in ("moons", "housing"):
            for gr in ("pilot", "asym"):
                run(pb, gr, max_iter=args.max_iter)
    else:
        if not args.problem:
            ap.error("--problem required unless --all")
        run(args.problem, args.grid, max_iter=args.max_iter)
    np.random.seed(0)  # no-op; explicit reminder that everything above is deterministic
