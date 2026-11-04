"""Jacobi condition (Daudin-Delarue, Thm 4.1(a)).

For any minimizer nu* and any t1 > t0, the restriction (nu*_t)_{t >= t1} is
the UNIQUE minimizer of the problem started from (t1, gamma*_{t1}), and it is
STABLE.  Test, at fixed eps:

  for k1 in K1_LIST:
    1. restart the problem at t_{k1}: particles at X*(t_{k1}), horizon
       [t_{k1}, T] with K - k1 slices;
    2. solve it from the PRIOR (no warm start) by damped Picard;
    3. uniqueness: TV distance between the restarted fixed point and the
       restriction nu*[k1:] (dynamic programming + uniqueness => ~ tol/omega);
    4. stability: lambda_min of the projected Hessian of the restarted
       problem at its fixed point (> 0 predicted).

Note on quadrature: the first slice of the restarted problem carries the
trapezoidal weight dt/2 instead of dt.  The Gibbs fixed point does not depend
on the time weights, so (3) is unaffected; the Hessian of (4) is the Hessian
of the restarted problem with its own quadrature, which is what Thm 4.1(a)
is about.

Usage:
    python experiments/14_jacobi.py --problem moons --eps 0.05
    python experiments/14_jacobi.py --problem moons --eps 0.05 --no-hessian
    python experiments/14_jacobi.py --problem moons --eps 0.05 --plot-only
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG_DIR, MAX_ITER, OMEGA, RESULTS, T, TOL, get_problem, load_ckpt
from plnode import hessian, solver

K1_LIST = (2, 4, 6, 8)


def restarted_problem(X_star, k1, K):
    """Data of the problem started at t_{k1}: (x0, T', K')."""
    dt = T / K
    return X_star[k1], T - k1 * dt, K - k1


def main(problem_name, eps, grid_name="pilot", k1_list=K1_LIST,
         with_hessian=True, tol_eig=1e-4):
    prob = get_problem(problem_name)
    d = load_ckpt(problem_name, grid_name, eps)
    nu_star, A = d["nu"], d["A"]
    K = nu_star.shape[0] - 1
    dt = T / K
    X_star, _ = solver.forward_pass(nu_star, prob["x0"], A, 2, dt, K)

    out_json = RESULTS / f"jacobi_{problem_name}_{grid_name}_eps{eps:g}.json"
    records = json.loads(out_json.read_text()) if out_json.exists() else []
    done = {r["k1"] for r in records}

    for k1 in k1_list:
        if k1 in done:
            print(f"k1={k1}: already done, skipping")
            continue
        x0_r, T_r, K_r = restarted_problem(X_star, k1, K)
        print(f"\n=== restart at t={k1 * dt:.2f} (k1={k1}, K'={K_r}) ===",
              flush=True)
        t0 = time.time()
        res = solver.picard(x0_r, prob["y"], A, d1=2, T=T_r, K=K_r, eps=eps,
                            omega=OMEGA, max_iter=MAX_ITER, tol=TOL,
                            verbose=False, grad_fn=prob["grad_fn"],
                            terminal_fn=prob["terminal_fn"])
        tv = float(0.5 * np.abs(res.nu - nu_star[k1:]).sum(axis=1).max())
        rec = dict(k1=k1, t1=k1 * dt, iters=res.iters,
                   converged=bool(res.converged),
                   dist_to_fixed_point=float(res.dist_to_fixed_point),
                   tv_to_restriction=tv, picard_seconds=time.time() - t0)
        print(f"  Picard: iters={res.iters} conv={res.converged} "
              f"diffTV/omega={res.dist_to_fixed_point:.2e}  "
              f"TV(restart, restriction)={tv:.2e}", flush=True)

        if with_hessian:
            op = hessian.hessian_operator(
                res.nu, x0_r, prob["y"], A, d1=2, eps=eps, T=T_r,
                grad_fn=prob["grad_fn"], support_threshold=1e-6)
            t0 = time.time()
            vals, _ = hessian.lambda_min_lobpcg(op, k=1, tol=tol_eig)
            rec.update(lambda_min=float(vals[0]), hv_products=op.n_products(),
                       hessian_seconds=time.time() - t0)
            print(f"  lambda_min = {vals[0]:.6e}  ({op.n_products()} Hv)",
                  flush=True)

        records.append(rec)
        out_json.write_text(json.dumps(records, indent=1))   # crash-safe

    print(f"\n-> {out_json}")

    plot(problem_name, eps, grid_name, records)
    return records


def plot(problem_name, eps, grid_name, records):
    """Two panels: uniqueness (TV to the restriction) and stability."""
    records = sorted(records, key=lambda r: r["t1"])
    fig, (ax_tv, ax_lam) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    t1 = [r["t1"] for r in records]
    ax_tv.semilogy(t1, [r["tv_to_restriction"] for r in records], "o-",
                   color="#2a78d6")
    ax_tv.axhline(TOL / OMEGA, color="0.4", ls="--", lw=1)
    ax_tv.text(0.02, TOL / OMEGA * 1.15, r"tol$/\omega$", color="0.35",
               fontsize=8, transform=ax_tv.get_yaxis_transform())
    ax_tv.set_ylim(1e-5, 1e-2)
    ax_tv.set_ylabel(r"TV to $(\nu^*_t)_{t\geq t_1}$")
    ax_tv.set_title("(i) uniqueness", fontsize=10)

    lam = [(r["t1"], r["lambda_min"]) for r in records if "lambda_min" in r]
    e3 = RESULTS / f"e3_lambda_min_{problem_name}_{grid_name}.json"
    full = json.loads(e3.read_text()).get(str(eps)) if e3.exists() else None
    if lam:
        ax_lam.plot([x for x, _ in lam], [y for _, y in lam], "o-",
                    color="#2a78d6", label=r"restarted on $[t_1,T]$")
    if full is not None:
        ax_lam.plot([0.0], [full], "s", color="#eb6834", ms=7,
                    label=r"full problem ($t_1=0$)")
    top = max([y for _, y in lam] + ([full] if full is not None else []))
    ax_lam.set_ylim(0, 1.2 * top)
    ax_lam.set_ylabel(r"$\lambda_{\min}$")
    ax_lam.set_title("(ii) stability", fontsize=10)
    ax_lam.legend(fontsize=8, loc="lower left")
    for ax in (ax_tv, ax_lam):
        ax.set_xlabel(r"restart time $t_1$")
        ax.set_xlim(-0.05, 0.9)
        ax.grid(alpha=0.3)
    fig.suptitle(rf"Jacobi condition, {problem_name}, $\varepsilon={eps:g}$",
                 fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / f"jacobi_{problem_name}_eps{eps:g}.png"
    fig.savefig(out, dpi=200)
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--grid", default="pilot")
    ap.add_argument("--k1", type=int, nargs="+", default=list(K1_LIST))
    ap.add_argument("--no-hessian", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from the saved JSON")
    a = ap.parse_args()
    if a.plot_only:
        path = RESULTS / f"jacobi_{a.problem}_{a.grid}_eps{a.eps:g}.json"
        plot(a.problem, a.eps, a.grid, json.loads(path.read_text()))
        raise SystemExit
    main(a.problem, a.eps, grid_name=a.grid, k1_list=a.k1,
         with_hessian=not a.no_hessian)
