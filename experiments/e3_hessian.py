"""E3 — lambda_min(eps) of the projected Hessian at nu* (matrix-free Lanczos).

Each Hv product = 2 forward passes; expect ~50-150 products per eigenvalue.
Includes the internal checks: symmetry of H and the Pontryagin coherence
<H eta, eta>/2 vs measured excess.

Usage:
    python experiments/e3_hessian.py --problem moons --grid pilot --eps 0.05
    python experiments/e3_hessian.py --problem moons --grid pilot --all-eps
    python experiments/e3_hessian.py --problem moons --grid pilotflow \
        --eps 0.02 --maxiter 1200

The residual |H v - lambda v| of the returned vector is printed: lambda is a
Rayleigh quotient, hence an upper bound for lambda_min, and lies within the
residual of an eigenvalue.
"""
import argparse
import json
import time

import numpy as np

from common import GRIDS, RESULTS, T, get_problem, load_ckpt
from plnode import hessian, perturbations
from plnode.solver import EPS_SCHEDULE


def one_eps(problem_name, grid_name, eps, *, h=1e-5, tol=1e-4, checks=True,
            support_threshold=1e-6, maxiter=300):
    """lambda_min at nu*_eps.

    support_threshold: restrict the operator to nodes carrying mass on every
    slice.  The entropic diagonal eps*w/nu* reaches ~1/nu ~ 1e12 on
    numerically empty nodes; those directions have enormous POSITIVE
    curvature (they cannot carry lambda_min) but they spread the spectrum
    and stall Lanczos.  The restriction removes them.

    The Pontryagin coherence check uses a multiplicative (lognormal-tilt)
    perturbation: eta proportional to nu* stays in the quadratic regime,
    whereas a mixture perturbation dumps mass on near-zero nodes where J is
    dominated by the log nonlinearity and the quadratic model is meaningless.
    """
    prob = get_problem(problem_name)
    d = load_ckpt(problem_name, grid_name, eps)
    nu_star, A = d["nu"], d["A"]
    K1, M = nu_star.shape

    op = hessian.hessian_operator(
        nu_star, prob["x0"], prob["y"], A, d1=2, eps=eps, T=T,
        grad_fn=prob["grad_fn"], h=h, support_threshold=support_threshold,
        verbose=True)

    if checks:
        sym = hessian.symmetry_check(op, K1, M, n_pairs=2)
        print(f"  symmetry check: max rel asym = {sym:.2e}", flush=True)
        rng = np.random.default_rng(0)
        eta = perturbations.lognormal_tilt(nu_star, 0.02, rng) - nu_star
        quad, excess = hessian.quadratic_prediction_check(
            op, nu_star, prob["x0"], prob["y"], A, 2, eps, eta, T=T,
            grad_fn=prob["grad_fn"], terminal_fn=prob["terminal_fn"])
        print(f"  Pontryagin coherence: <H eta,eta>/2 = {quad:.4e}  "
              f"vs measured excess = {excess:.4e}  "
              f"(ratio {quad/excess:.3f})", flush=True)

    t0 = time.time()
    vals, vecs = hessian.lambda_min_lobpcg(op, k=1, tol=tol, maxiter=maxiter)
    lam, vec = float(vals[0]), vecs[:, 0]
    n_hv = op.n_products()
    resid = float(np.linalg.norm(op @ vec - lam * vec) / np.linalg.norm(vec))
    print(f"  lambda_min = {lam:.6e}   residual = {resid:.1e}   "
          f"({n_hv} Hv products, {time.time()-t0:.0f}s)", flush=True)
    return lam


def main(args):
    eps_list = list(EPS_SCHEDULE) if args.all_eps else [args.eps]
    out = {}
    for eps in eps_list:
        print(f"\n=== lambda_min at eps={eps} "
              f"({args.problem}/{args.grid}) ===")
        out[eps] = one_eps(args.problem, args.grid, eps, h=args.h,
                           maxiter=args.maxiter)
    path = RESULTS / f"e3_lambda_min_{args.problem}_{args.grid}.json"
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update({str(k): v for k, v in out.items()})
    path.write_text(json.dumps(existing, indent=1))
    print(f"\n-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--grid", choices=list(GRIDS), default="pilot")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--all-eps", action="store_true")
    ap.add_argument("--h", type=float, default=1e-5)
    ap.add_argument("--maxiter", type=int, default=300)
    main(ap.parse_args())
