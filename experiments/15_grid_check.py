"""Grid dependence of the cloud envelope c(eps).

E1 runs on the asymmetric grid, E2/E3 and the genericity probe on the pilot
grid.  This check runs the genericity probe's 30-point cloud
(13_genericity.reduced_cloud) on the reference data at the frozen fixed
points of both grids, so that the level of c can be compared across grids
with the cloud held fixed.

Writes results/grid_check.json.

Usage:
    python experiments/15_grid_check.py --problem moons --eps 0.1
    python experiments/15_grid_check.py --problem moons --eps 0.1 --grids asym
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

from common import GRIDS, RESULTS, R_GRID, get_problem, load_ckpt
from plnode import solver

_spec = importlib.util.spec_from_file_location(
    "genericity", Path(__file__).with_name("13_genericity.py"))
genericity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(genericity)


def main(problem, eps, grids=("pilot", "asym")):
    prob = get_problem(problem)
    path = RESULTS / "grid_check.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for grid in grids:
        mp = GRIDS[grid]
        A, _ = solver.build_grid(mp, R=R_GRID)
        nu_star = load_ckpt(problem, grid, eps)["nu"]
        t0 = time.time()
        cl = genericity.reduced_cloud(nu_star, prob, A, mp, eps,
                                      solver.log_prior(A))
        cl.update(M=int(A.shape[0]), seconds=time.time() - t0)
        print(f"  {problem} {grid} eps={eps}: c = {cl['c']:.4f}  "
              f"({cl['n_admissible']}/{cl['n_total']} admissible, "
              f"M = {cl['M']})", flush=True)
        out[f"{problem}_{grid}_eps{eps:g}"] = cl
        path.write_text(json.dumps(out, indent=1))
    print(f"-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--grids", nargs="+", default=["pilot", "asym"],
                    choices=["pilot", "asym"])
    a = ap.parse_args()
    main(a.problem, a.eps, grids=a.grids)
