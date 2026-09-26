"""E1 — Empirical PL constant c(eps).

For each eps in the schedule: generate a cloud of perturbations of nu* at
several amplitudes, evaluate I_h and J - J* for each, scatter I vs J - J*,
and extract c(eps) = inf of the ratio I / (J - J*).  Log-log plot of c vs eps
gives the degeneration exponent.

Caveat (agreed mitigation): report c(eps) only where I_h(nu*_eps) is
negligible against the cloud's I values — at eps = 0.02 Picard does not fully
converge and nu* is not an exact fixed point.

Usage:
    python experiments/e1_pl_constant.py --problem moons --grid asym
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (FIG_DIR, GRIDS, R_GRID, RESULTS, T, get_problem,
                    load_all_eps)
from plnode import fisher, perturbations
from plnode.gradient_flow import kl_to
from plnode.solver import cost, forward_pass, log_prior

DELTAS = (0.005, 0.01, 0.02, 0.05, 0.1)
N_PER_DELTA = 10


def J_of(nu, prob, A, eps, log_nu_inf, K):
    dt = T / K
    X, _ = forward_pass(nu, prob["x0"], A, 2, dt, K)
    val, _, _ = cost(X, prob["y"], nu, log_nu_inf, eps, dt,
                     terminal_fn=prob["terminal_fn"])
    return val


def main(problem_name, grid_name, seed=0, only_eps=None):
    prob = get_problem(problem_name)
    mp = GRIDS[grid_name]
    if only_eps is not None:
        from common import load_ckpt
        ckpts = {only_eps: load_ckpt(problem_name, grid_name, only_eps)}
    else:
        ckpts = load_all_eps(problem_name, grid_name)
    log_nu_inf = None
    records = []

    for eps, d in ckpts.items():
        nu_star, A = d["nu"], d["A"]
        K = nu_star.shape[0] - 1
        dt = T / K
        if log_nu_inf is None:
            log_nu_inf = log_prior(A)
        J_star = J_of(nu_star, prob, A, eps, log_nu_inf, K)
        I_star = fisher.fisher_from_control(
            nu_star, prob["x0"], prob["y"], A, 2, mp, eps, T=T, R=R_GRID,
            grad_fn=prob["grad_fn"])
        print(f"\n--- eps={eps}:  J*={J_star:.6f}  I_h(nu*)={I_star:.3e} ---")

        for delta, kind, nu in perturbations.perturbation_cloud(
                nu_star, DELTAS, N_PER_DELTA, seed=seed):
            J = J_of(nu, prob, A, eps, log_nu_inf, K)
            I_h = fisher.fisher_from_control(
                nu, prob["x0"], prob["y"], A, 2, mp, eps, T=T, R=R_GRID,
                grad_fn=prob["grad_fn"])
            gap = J - J_star
            lam2 = kl_to(nu, nu_star, dt)
            records.append(dict(eps=eps, delta=delta, kind=kind, J_gap=gap,
                                I=I_h, lambda2=lam2, I_star=I_star,
                                J_star=J_star))
        n_eps = [r for r in records if r["eps"] == eps]
        valid = [r for r in n_eps if r["J_gap"] > 0 and r["I"] > 10 * I_star]
        if valid:
            c_eps = min(r["I"] / r["J_gap"] for r in valid)
            print(f"    cloud size={len(n_eps)}  usable={len(valid)}  "
                  f"c(eps) = inf I/(J-J*) = {c_eps:.4f}", flush=True)
        else:
            print("    no usable points (I not >> I_star) — refine schedule",
                  flush=True)
        # crash-safe: dump everything accumulated so far after each eps
        out_json = RESULTS / f"e1_{problem_name}_{grid_name}.json"
        with open(out_json, "w") as fh:
            json.dump(records, fh, indent=1)

    print(f"\n-> {out_json}")

    # --- figures -------------------------------------------------------------
    eps_list = sorted({r["eps"] for r in records}, reverse=True)
    fig, axes = plt.subplots(1, len(eps_list), figsize=(3.6 * len(eps_list), 3.4),
                             sharey=False)
    c_of_eps = {}
    for ax, eps in zip(np.atleast_1d(axes), eps_list):
        rs = [r for r in records if r["eps"] == eps]
        gaps = np.array([r["J_gap"] for r in rs])
        Is = np.array([r["I"] for r in rs])
        I_star = rs[0]["I_star"]
        ok = (gaps > 0) & (Is > 10 * I_star)
        ax.loglog(gaps[ok], Is[ok], "o", ms=4, alpha=0.7)
        if ok.any():
            c_eps = float((Is[ok] / gaps[ok]).min())
            c_of_eps[eps] = c_eps
            g = np.geomspace(gaps[ok].min(), gaps[ok].max(), 50)
            ax.loglog(g, c_eps * g, "r--", lw=1,
                      label=rf"$c(\varepsilon)={c_eps:.3g}$")
            ax.legend(fontsize=8)
        ax.set_title(rf"$\varepsilon={eps}$", fontsize=10)
        ax.set_xlabel(r"$J-J^*$")
        ax.grid(alpha=0.3, which="both")
    np.atleast_1d(axes)[0].set_ylabel(r"$\mathcal{I}_h$")
    fig.suptitle(f"E1 — PL scatter, {problem_name} ({grid_name} grid)")
    fig.tight_layout()
    out = FIG_DIR / f"e1_scatter_{problem_name}_{grid_name}.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")

    if len(c_of_eps) >= 2:
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        e = np.array(sorted(c_of_eps))
        c = np.array([c_of_eps[x] for x in e])
        ax.loglog(e, c, "o-")
        slope = np.polyfit(np.log(e), np.log(c), 1)[0]
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"$c(\varepsilon)$")
        ax.set_title(rf"E1 — $c(\varepsilon)$, slope $\approx$ {slope:.2f}")
        ax.grid(alpha=0.3, which="both")
        out = FIG_DIR / f"e1_c_vs_eps_{problem_name}_{grid_name}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"-> {out}  (degeneration exponent ~ {slope:.2f})")

    return records, c_of_eps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--grid", choices=list(GRIDS), default="asym")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=None,
                    help="run a single eps (e.g. flow-refined recovery)")
    a = ap.parse_args()
    main(a.problem, a.grid, a.seed, only_eps=a.eps)
