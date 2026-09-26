"""Phase 1.6 — Consistency test of the discretized Fisher functional.

At nu*_eps the Gibbs formula cancels the integrand of I exactly, so
I_h(nu*_eps) should be small and scale like (diffTV_final / omega)^2 — the
calibrated squared distance to the true fixed point.  If this fails there is
a bug upstream; do not proceed to E1.

Usage:
    python experiments/01_fisher_consistency.py --problem moons --grid pilot
"""
import argparse

from common import GRIDS, R_GRID, T, get_problem, load_all_eps
from plnode import fisher


def main(problem_name, grid_name):
    prob = get_problem(problem_name)
    mp = GRIDS[grid_name]
    ckpts = load_all_eps(problem_name, grid_name)

    print(f"\nI_h(nu*_eps) vs (diffTV/omega)^2 — {problem_name}/{grid_name}")
    print(f"{'eps':>6s} {'I_h(nu*)':>12s} {'(diff/om)^2':>12s} {'ratio':>8s}")
    rows = []
    for eps, d in ckpts.items():
        nu, A = d["nu"], d["A"]
        I_h = fisher.fisher_from_control(
            nu, prob["x0"], prob["y"], A, d1=2, mp_per_axis=mp, eps=eps,
            T=T, R=R_GRID, grad_fn=prob["grad_fn"])
        cal = (float(d["diff_final"]) / float(d["omega"])) ** 2
        ratio = I_h / cal if cal > 0 else float("nan")
        rows.append((eps, I_h, cal, ratio))
        print(f"{eps:6.3f} {I_h:12.4e} {cal:12.4e} {ratio:8.2f}")
    print("\nExpected: ratio O(1) across the schedule (same order of magnitude).")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--grid", choices=list(GRIDS), default="pilot")
    a = ap.parse_args()
    main(a.problem, a.grid)
