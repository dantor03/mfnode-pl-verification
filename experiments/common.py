"""Shared helpers for the experiment scripts.

nu* checkpoints live in results/nu_star/{problem}_{grid}_eps{eps}.npz with
keys: nu, X, A, mp_per_axis, R, eps, omega, diff_final, converged, plus the
problem metadata needed to rebuild the benchmark deterministically.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plnode import datasets, solver  # noqa: E402

RESULTS = ROOT / "results"
NU_STAR_DIR = RESULTS / "nu_star"
FIG_DIR = ROOT / "paper" / "figures"
for _d in (NU_STAR_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

GRIDS = {"pilot": solver.MP_PILOT, "asym": solver.MP_ASYM,
         # pseudo-grids: flow-refined checkpoints (03_refine_flow.py) on the
         # same meshes; every runner consumes them via --grid <name>flow
         "pilotflow": solver.MP_PILOT, "asymflow": solver.MP_ASYM}
R_GRID = 2.0
T, K = 1.0, 10
OMEGA, TOL, MAX_ITER = 0.2, 1e-4, 200


def get_problem(name):
    return datasets.PROBLEMS[name]()


def ckpt_path(problem, grid, eps):
    return NU_STAR_DIR / f"{problem}_{grid}_eps{eps:g}.npz"


def save_ckpt(problem, grid, eps, res, A, mp):
    np.savez_compressed(
        ckpt_path(problem, grid, eps),
        nu=res.nu, X=res.X, A=A, mp_per_axis=np.array(mp), R=R_GRID,
        eps=eps, omega=res.omega, diff_final=res.diff_final,
        converged=res.converged, iters=res.iters,
        cost_final=res.cost_history[-1] if res.cost_history else np.nan,
    )


def load_ckpt(problem, grid, eps):
    path = ckpt_path(problem, grid, eps)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run experiments/00_compute_nu_star.py first")
    return np.load(path)


def load_all_eps(problem, grid, eps_schedule=solver.EPS_SCHEDULE):
    return {eps: load_ckpt(problem, grid, eps) for eps in eps_schedule}
