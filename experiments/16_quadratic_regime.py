"""Size of the quadratic regime around nu*: the Pontryagin coherence check
of e3_hessian.py repeated at several perturbation amplitudes.

For a lognormal tilt eta of amplitude a, the excess J(nu* + eta) - J* is
compared with the quadratic model <H eta, eta>/2.  Evaluating both +eta and
-eta splits the excess into its even part (quadratic + quartic terms) and
its odd part (a residual first variation, and cubic terms): a first
variation shows up as an odd part proportional to a, cubic terms as one
proportional to a^3.

Writes results/quadratic_regime.json.

Usage:
    python experiments/16_quadratic_regime.py --problem moons --grid pilotflow --eps 0.02
"""
import argparse
import json
import time

import numpy as np

from common import RESULTS, T, get_problem, load_ckpt
from plnode import hessian, perturbations

AMPLITUDES = (0.02, 0.01, 0.005, 0.0025)


def main(problem, grid, eps, amplitudes=AMPLITUDES):
    prob = get_problem(problem)
    d = load_ckpt(problem, grid, eps)
    nu_star, A = d["nu"], d["A"]
    op = hessian.hessian_operator(
        nu_star, prob["x0"], prob["y"], A, d1=2, eps=eps, T=T,
        grad_fn=prob["grad_fn"], h=1e-5, support_threshold=1e-6, verbose=True)

    def excess(eta):
        return hessian.quadratic_prediction_check(
            op, nu_star, prob["x0"], prob["y"], A, 2, eps, eta, T=T,
            grad_fn=prob["grad_fn"], terminal_fn=prob["terminal_fn"])

    rows = []
    for a in amplitudes:
        t0 = time.time()
        eta = perturbations.lognormal_tilt(nu_star, a, np.random.default_rng(0)) \
            - nu_star
        quad, ex_p = excess(eta)
        _, ex_m = excess(-eta)
        row = dict(amplitude=a, quad=quad, excess_plus=ex_p, excess_minus=ex_m,
                   even=(ex_p + ex_m) / 2, odd=(ex_p - ex_m) / 2,
                   ratio=quad / ex_p, ratio_even=quad / ((ex_p + ex_m) / 2))
        rows.append(row)
        print(f"  a = {a:<7g} quad = {quad:.4e}  excess = {ex_p:.4e}  "
              f"ratio = {row['ratio']:.3f}  even ratio = {row['ratio_even']:.3f}"
              f"  odd = {row['odd']:.2e}  ({time.time() - t0:.0f}s)", flush=True)

    path = RESULTS / "quadratic_regime.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out[f"{problem}_{grid}_eps{eps:g}"] = rows
    path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--grid", default="pilotflow")
    ap.add_argument("--eps", type=float, default=0.02)
    a = ap.parse_args()
    main(a.problem, a.grid, a.eps)
