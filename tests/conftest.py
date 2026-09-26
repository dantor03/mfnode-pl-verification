"""Tiny synthetic problem for fast correctness tests (seconds, not hours).

Grid 3^5 = 243 nodes, N = 40 moons points, K = 5 time slices, eps = 0.5
(strongly contractive regime -> Picard converges in a few iterations).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plnode import datasets, solver  # noqa: E402

MP_TINY = (3, 3, 3, 3, 3)
R_TINY = 2.0
K_TINY = 5
EPS_TINY = 0.5


@pytest.fixture(scope="session")
def tiny():
    X, y = datasets.get_moons(n=40)
    y_emb = datasets.embed_labels(y)
    A, _ = solver.build_grid(MP_TINY, R=R_TINY)
    res = solver.picard(X, y_emb, A, d1=2, T=1.0, K=K_TINY, eps=EPS_TINY,
                        omega=0.2, max_iter=200, tol=1e-6, verbose=False)
    return dict(x0=X, y=y_emb, A=A, mp=MP_TINY, R=R_TINY, K=K_TINY,
                eps=EPS_TINY, res=res)
