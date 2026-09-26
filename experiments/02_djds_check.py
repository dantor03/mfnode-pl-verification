"""Phase 2.9 — dJ/ds = -I test (closes the J - I - flow circle; paper figure).

Takes nu* (or a perturbation of it), performs one small Fokker-Planck step
Delta-s of the gradient flow, and compares [J(nu^{s+ds}) - J(nu^s)] / ds with
-I_h(nu^s) over a range of ds (expected agreement O(ds) + O(h^2)).

Also calibrates the Psi-freezing trick: J decrement vs refresh_every m.

Usage:
    python experiments/02_djds_check.py --problem moons --grid pilot --eps 0.05
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG_DIR, GRIDS, R_GRID, T, get_problem, load_ckpt
from plnode import gradient_flow, perturbations


def main(problem_name, grid_name, eps, delta=0.05, seed=0):
    prob = get_problem(problem_name)
    mp = GRIDS[grid_name]
    d = load_ckpt(problem_name, grid_name, eps)
    nu_star, A = d["nu"], d["A"]

    rng = np.random.default_rng(seed)
    nu0 = perturbations.mixture_perturbation(nu_star, delta, rng)

    ds_list = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    print(f"\ndJ/ds vs -I_h at TV~{delta} from nu* — {problem_name}/{grid_name},"
          f" eps={eps}")
    print(f"{'ds':>9s} {'[J1-J0]/ds':>14s} {'-I_h':>14s} {'rel err':>9s}")
    errs = []
    for ds in ds_list:
        lhs, rhs, rel = gradient_flow.djds_test(
            nu0, prob["x0"], prob["y"], A, d1=2, mp_per_axis=mp, eps=eps,
            T=T, R=R_GRID, ds=ds, grad_fn=prob["grad_fn"],
            terminal_fn=prob["terminal_fn"])
        errs.append(rel)
        print(f"{ds:9.1e} {lhs:14.6e} {rhs:14.6e} {rel:9.2e}")

    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.loglog(ds_list, errs, "o-", label="relative error")
    ax.loglog(ds_list, [errs[-1] * d_ / ds_list[-1] for d_ in ds_list],
              "k--", alpha=0.5, label=r"O($\Delta s$) reference")
    ax.set_xlabel(r"$\Delta s$")
    ax.set_ylabel(r"rel. err. of $\Delta J/\Delta s \approx -\mathcal{I}_h$")
    ax.set_title(f"{problem_name}, eps={eps}: dJ/ds = -I  (grid {grid_name})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    out = FIG_DIR / f"djds_{problem_name}_{grid_name}_eps{eps:g}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")

    # --- Psi-freezing calibration ------------------------------------------
    print("\nPsi-freezing: J decrement after fixed s-horizon vs refresh_every m")
    horizon_steps = 100
    base = None
    for m in (1, 5, 10, 20):
        res = gradient_flow.gradient_flow(
            nu0, prob["x0"], prob["y"], A, d1=2, mp_per_axis=mp, eps=eps,
            T=T, R=R_GRID, n_steps=horizon_steps, refresh_every=m,
            grad_fn=prob["grad_fn"], terminal_fn=prob["terminal_fn"])
        dec = res.J_history[0] - res.J_history[-1]
        if base is None:
            base = dec
        print(f"  m={m:3d}  J decrement={dec:.6e}  ratio to m=1: {dec/base:.4f}"
              f"  mass_err={res.mass_error:.1e}  min_mass={res.min_mass:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--grid", choices=list(GRIDS), default="pilot")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args()
    main(a.problem, a.grid, a.eps, a.delta)
