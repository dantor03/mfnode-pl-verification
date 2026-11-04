"""Genericity probe (Daudin-Delarue, Thm 4.1(b): O is open and dense).

The other experiments fix one initial condition gamma_0 per benchmark.  Here
we redraw the data: if O is open, a stable minimizer persists (with nearby
lambda_min) under small perturbations of gamma_0; if it is dense, a randomly
drawn data set should fall in O.  For each data variant:

  1. eps-continuation on the pilot grid down to eps = 0.05 (from the prior);
  2. lambda_min of the projected Hessian at eps in HESS_EPS;
  3. a reduced E1 cloud (CLOUD_DELTAS x CLOUD_N x {mixture, tilt}) at the same
     eps, giving c(eps) = inf I_h / (J - J*) over the admissible points.

Variants: moons with sampling seeds 0-4 (noise 0.12) and noise 0.08 / 0.16
(seed 42); housing with training subsamples 0-4 and the features fixed to
(MedInc, AveOccup) so that only the sample changes.  The last variant,
moons_base, is the reference data of the other experiments (seed 42, noise
0.12) run through the same probe: on the pilot grid with this 30-point cloud
c(eps) is not on the scale of E1 (asymmetric grid), so the variants are
compared with it rather than with E1.

Results are appended per variant to results/genericity.json (crash-safe,
resumable); checkpoints go to results/genericity/.  Variants are independent,
so several processes can run in parallel with --variant.

Usage:
    python experiments/13_genericity.py --list
    python experiments/13_genericity.py --variant 0 1 2
    python experiments/13_genericity.py --all
    python experiments/13_genericity.py --relam 0 --eps 0.05 --maxiter 1500

--relam re-solves lambda_min at one eps from the saved checkpoint with a
larger LOBPCG iteration budget and records the residual norm
|H v - lambda v| (the Rayleigh quotient is an upper bound for lambda_min and
lies within the residual of an eigenvalue).  The result goes to
results/genericity_parts/relam-<tag>-eps<eps>.json.
"""
import argparse
import json
import time

import numpy as np

from common import GRIDS, MAX_ITER, OMEGA, R_GRID, RESULTS, T, TOL
from plnode import datasets, fisher, hessian, perturbations, solver
from plnode.gradient_flow import kl_to

EPS_SCHEDULE = (0.5, 0.2, 0.1, 0.05)
HESS_EPS = (0.1, 0.05)
CLOUD_DELTAS = (0.01, 0.05, 0.1)
CLOUD_N = 5                                   # per (delta, kind): 30 points
GRID = "pilot"
OUT_DIR = RESULTS / "genericity"
OUT_JSON = RESULTS / "genericity.json"
HOUSING_FEATURES = ("MedInc", "AveOccup")

VARIANTS = (
    [dict(tag=f"moons_seed{s}", problem="moons", seed=s, noise=0.12)
     for s in range(5)]
    + [dict(tag=f"moons_noise{n:g}", problem="moons", seed=datasets.SEED,
            noise=n) for n in (0.08, 0.16)]
    + [dict(tag=f"housing_sub{s}", problem="housing", seed=s)
       for s in range(5)]
    + [dict(tag="moons_base", problem="moons", seed=datasets.SEED,
            noise=0.12)]
)


def build_problem(v):
    if v["problem"] == "moons":
        return datasets.moons_problem(seed=v["seed"], noise=v["noise"])
    return datasets.housing_problem(seed=v["seed"],
                                    features=list(HOUSING_FEATURES))


def J_of(nu, prob, A, eps, log_nu_inf, K):
    dt = T / K
    X, _ = solver.forward_pass(nu, prob["x0"], A, 2, dt, K)
    val, _, _ = solver.cost(X, prob["y"], nu, log_nu_inf, eps, dt,
                            terminal_fn=prob["terminal_fn"])
    return val


def reduced_cloud(nu_star, prob, A, mp, eps, log_nu_inf, seed=0):
    """c(eps) over a 30-point cloud, with the E1 admissibility rule."""
    K = nu_star.shape[0] - 1
    dt = T / K
    J_star = J_of(nu_star, prob, A, eps, log_nu_inf, K)
    I_star = fisher.fisher_from_control(nu_star, prob["x0"], prob["y"], A, 2,
                                        mp, eps, T=T, R=R_GRID,
                                        grad_fn=prob["grad_fn"])
    ratios, lam2 = [], []
    n_total = n_nonpositive = 0
    for _, _, nu in perturbations.perturbation_cloud(nu_star, CLOUD_DELTAS,
                                                     CLOUD_N, seed=seed):
        n_total += 1
        gap = J_of(nu, prob, A, eps, log_nu_inf, K) - J_star
        I_h = fisher.fisher_from_control(nu, prob["x0"], prob["y"], A, 2, mp,
                                         eps, T=T, R=R_GRID,
                                         grad_fn=prob["grad_fn"])
        n_nonpositive += gap <= 0
        if gap > 0 and I_h > 10 * I_star:
            ratios.append(I_h / gap)
            lam2.append(kl_to(nu, nu_star, dt))
    return dict(c=float(min(ratios)) if ratios else None,
                n_admissible=len(ratios), n_total=n_total,
                n_nonpositive_gap=int(n_nonpositive),
                lambda2_max=float(max(lam2)) if lam2 else None,
                I_star=float(I_star), J_star=float(J_star))


def run_variant(v, with_hessian=True):
    prob = build_problem(v)
    mp = GRIDS[GRID]
    A, _ = solver.build_grid(mp, R=R_GRID)
    log_nu_inf = solver.log_prior(A)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = dict(v, continuation=[], per_eps={})
    print(f"\n##### {v['tag']} #####", flush=True)

    nu_warm = None
    for eps in EPS_SCHEDULE:
        path = OUT_DIR / f"{v['tag']}_eps{eps:g}.npz"
        if path.exists():
            d = np.load(path)
            nu_warm = d["nu"]
            rec["continuation"].append(json.loads(str(d["summary"])))
            print(f"  eps={eps}: checkpoint found", flush=True)
            continue
        t0 = time.time()
        res = solver.picard(prob["x0"], prob["y"], A, d1=2, T=T, K=10,
                            eps=eps, omega=OMEGA, max_iter=MAX_ITER, tol=TOL,
                            verbose=False, nu_init=nu_warm,
                            log_nu_inf=log_nu_inf, grad_fn=prob["grad_fn"],
                            terminal_fn=prob["terminal_fn"])
        metric = prob["metric"](res.X[-1])
        summary = dict(
            eps=eps, iters=res.iters, converged=bool(res.converged),
            dist_to_fixed_point=float(res.dist_to_fixed_point),
            J=float(res.cost_history[-1]), metric=float(metric),
            symmetry_error=solver.symmetry_error(res.nu, mp),
            seconds=time.time() - t0)
        np.savez_compressed(path, nu=res.nu, converged=res.converged,
                            summary=json.dumps(summary))
        rec["continuation"].append(summary)
        print(f"  eps={eps}: iters={res.iters} conv={res.converged} "
              f"diffTV/omega={res.dist_to_fixed_point:.1e} "
              f"{prob['metric_name']}={metric:.3f}", flush=True)
        nu_warm = res.nu

    for eps in HESS_EPS:
        d = np.load(OUT_DIR / f"{v['tag']}_eps{eps:g}.npz")
        nu_star = d["nu"]
        out = dict(converged=bool(d["converged"]))
        cl = reduced_cloud(nu_star, prob, A, mp, eps, log_nu_inf)
        out.update(cloud=cl)
        print(f"  eps={eps}: c = {cl['c']}  "
              f"({cl['n_admissible']}/{cl['n_total']} admissible)", flush=True)
        if with_hessian:
            op = hessian.hessian_operator(nu_star, prob["x0"], prob["y"], A,
                                          d1=2, eps=eps, T=T,
                                          grad_fn=prob["grad_fn"],
                                          support_threshold=1e-6)
            vals, _ = hessian.lambda_min_lobpcg(op, k=1, tol=1e-4)
            out.update(lambda_min=float(vals[0]), hv_products=op.n_products())
            print(f"  eps={eps}: lambda_min = {vals[0]:.4e}", flush=True)
        rec["per_eps"][str(eps)] = out
    return rec


def relam(i, eps, maxiter):
    v = VARIANTS[i]
    prob = build_problem(v)
    A, _ = solver.build_grid(GRIDS[GRID], R=R_GRID)
    nu_star = np.load(OUT_DIR / f"{v['tag']}_eps{eps:g}.npz")["nu"]
    t0 = time.time()
    op = hessian.hessian_operator(nu_star, prob["x0"], prob["y"], A, d1=2,
                                  eps=eps, T=T, grad_fn=prob["grad_fn"],
                                  support_threshold=1e-6)
    vals, vecs = hessian.lambda_min_lobpcg(op, k=1, tol=1e-4,
                                           maxiter=maxiter)
    lam, vec = float(vals[0]), vecs[:, 0]
    resid = float(np.linalg.norm(op @ vec - lam * vec) / np.linalg.norm(vec))
    out = dict(tag=v["tag"], eps=eps, lambda_min=lam, residual=resid,
               hv_products=op.n_products(), maxiter=maxiter,
               seconds=time.time() - t0)
    print(f"  {v['tag']} eps={eps}: lambda_min = {lam:.4e}  "
          f"residual = {resid:.1e}  ({out['hv_products']} Hv)", flush=True)
    path = RESULTS / "genericity_parts" / f"relam-{v['tag']}-eps{eps:g}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}", flush=True)


def main(indices, with_hessian=True):
    records = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for i in indices:
        v = VARIANTS[i]
        if v["tag"] in records:
            print(f"{v['tag']}: already in {OUT_JSON.name}, skipping")
            continue
        rec = run_variant(v, with_hessian=with_hessian)
        # re-read before writing: parallel processes append to the same file
        records = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
        records[v["tag"]] = rec
        OUT_JSON.write_text(json.dumps(records, indent=1))
        print(f"-> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, nargs="+")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-hessian", action="store_true")
    ap.add_argument("--relam", type=int, metavar="VARIANT")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--maxiter", type=int, default=1500)
    a = ap.parse_args()
    if a.relam is not None:
        relam(a.relam, a.eps, a.maxiter)
    elif a.list:
        for i, v in enumerate(VARIANTS):
            print(i, v["tag"])
    else:
        idx = range(len(VARIANTS)) if a.all else (a.variant or [])
        main(list(idx), with_hessian=not a.no_hessian)
