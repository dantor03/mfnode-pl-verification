"""Refine a non-converged Picard endpoint with the SG gradient flow.

Motivation (E2 finding): at eps = 0.02 damped Picard stalls at
diffTV/omega ~ 0.5 and its iterate is NOT the minimizer --- every descent
initialized nearby drives J below J(nu*_Picard).  The Scharfetter--Gummel
flow, whose exact discrete equilibrium is the self-consistent Gibbs fixed
point, can therefore be used as a POLISHER where the fixed-point map loses
contractivity: integrate the flow from the Picard iterate until I_h
plateaus.

Saves the refined control as a standard checkpoint under the pseudo-grid
name '<grid>flow' (e.g. moons_asymflow_eps0.02.npz), so every experiment
runner consumes it unchanged via --grid asymflow / pilotflow.

Diagnostics reported: J and I_h before/after, TV(refined, Picard),
symmetry error, and the fixed-point residual TV(Gibbs(nu), nu) (exactly 0
at the true fixed point, regardless of contractivity).

Usage:
    python experiments/03_refine_flow.py --problem moons --grid pilot --eps 0.02
    python experiments/03_refine_flow.py --problem moons --grid asym  --eps 0.02
"""
import argparse
import time

import numpy as np

from common import GRIDS, K, NU_STAR_DIR, R_GRID, T, get_problem, load_ckpt
from plnode import gradient_flow, solver
from plnode.fisher import fisher_from_control


def refine(problem_name, grid_name, eps, *, chunk=100, refresh_every=5,
           rel_tol=1e-3, max_chunks=60):
    """Run the SG flow in chunks until the J-decrement per chunk stalls."""
    prob = get_problem(problem_name)
    mp = GRIDS[grid_name]
    d = load_ckpt(problem_name, grid_name, eps)
    nu, A = d["nu"].copy(), d["A"]

    # crash-safe: per-chunk partial state (resumed if present)
    part = NU_STAR_DIR / f"{problem_name}_{grid_name}flow_eps{eps:g}.part.npz"
    s_resumed = 0.0
    if part.exists():
        pd = np.load(part)
        nu = pd["nu"].copy()
        s_resumed = float(pd["s_total"])
        print(f"resumed from {part.name} (s_total={s_resumed:.1f}, "
              f"chunk {int(pd['chunk'])})", flush=True)

    def I_of(nu_):
        return fisher_from_control(nu_, prob["x0"], prob["y"], A, 2, mp, eps,
                                   T=T, R=R_GRID, grad_fn=prob["grad_fn"])

    I0 = I_of(nu)
    print(f"start: I_h = {I0:.4e}   diffTV/om (Picard) = "
          f"{float(d['diff_final']) / float(d['omega']):.3f}", flush=True)

    t0 = time.time()
    J_prev, s_total = None, s_resumed
    for chunk_i in range(max_chunks):
        res = gradient_flow.gradient_flow(
            nu, prob["x0"], prob["y"], A, 2, mp, eps, T=T, R=R_GRID,
            n_steps=chunk, refresh_every=refresh_every,
            grad_fn=prob["grad_fn"], terminal_fn=prob["terminal_fn"])
        nu = res.nu
        s_total += res.s_grid[-1] + res.ds * refresh_every
        J_now = res.J_history[-1]
        print(f"  chunk {chunk_i:3d}  s={s_total:8.2f}  J={J_now:.8f}  "
              f"I_h={res.I_history[-1]:.3e}  ({time.time()-t0:.0f}s)",
              flush=True)
        np.savez_compressed(part, nu=nu, s_total=s_total, chunk=chunk_i)
        if J_prev is not None and abs(J_prev - J_now) < rel_tol * abs(
                J_prev - res.J_history[0] + 1e-300) + 1e-12:
            # decrement this chunk negligible vs total progress
            if abs(J_prev - J_now) < 1e-9:
                break
        J_prev = J_now

    # --- diagnostics --------------------------------------------------------
    I1 = I_of(nu)
    sym = solver.symmetry_error(nu, mp)
    tv_move = float(0.5 * np.abs(nu - d["nu"]).sum(axis=1).max())
    # fixed-point residual: one undamped Gibbs update
    dt = T / K
    X, Phi = solver.forward_pass(nu, prob["x0"], A, 2, dt, K)
    p = solver.cocycle_gradients(X, Phi, prob["y"], grad_fn=prob["grad_fn"])
    nu_g, _ = solver.gibbs_update(X, p, A, 2, eps, solver.log_prior(A))
    tv_fix = float(0.5 * np.abs(nu_g - nu).sum(axis=1).max())
    log_nu_inf = solver.log_prior(A)
    J_ref, _, _ = solver.cost(X, prob["y"], nu, log_nu_inf, eps, dt,
                              terminal_fn=prob["terminal_fn"])
    metric = prob["metric"](X[-1])

    print(f"\nrefined: J = {J_ref:.8f}   I_h = {I1:.4e}  (was {I0:.4e})",
          flush=True)
    print(f"TV(refined, Picard) = {tv_move:.4f}   sym_err = {sym:.1e}   "
          f"TV(Gibbs(nu), nu) = {tv_fix:.4e}   "
          f"{prob['metric_name']} = {metric:.3f}", flush=True)

    out = NU_STAR_DIR / f"{problem_name}_{grid_name}flow_eps{eps:g}.npz"
    np.savez_compressed(
        out, nu=nu, X=X, A=A, mp_per_axis=np.array(mp), R=R_GRID, eps=eps,
        omega=float(d["omega"]), diff_final=tv_fix,
        converged=bool(tv_fix < 1e-3), iters=-1, cost_final=J_ref,
        refined_from="picard", s_total=s_total)
    part.unlink(missing_ok=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--grid", choices=["pilot", "asym"], default="pilot")
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--chunk", type=int, default=100)
    a = ap.parse_args()
    refine(a.problem, a.grid, a.eps, chunk=a.chunk)
