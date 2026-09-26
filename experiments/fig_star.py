"""Star figure — lambda_min(eps), c(eps) and c-hat(eps) on the same axes.

Three independent estimators of the same degeneration as eps -> 0+; if they
collapse onto a common power law, that is the central result of the paper.

Reads the JSON outputs of E1 (c), E2 (c-hat, one JSON per eps) and E3
(lambda_min) and overlays them in log-log.

Usage:
    python experiments/fig_star.py --problem moons --grid asym
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG_DIR, RESULTS
from plnode.solver import EPS_SCHEDULE


def load_c_of_eps(problem, grid):
    path = RESULTS / f"e1_{problem}_{grid}.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text())
    out = {}
    for eps in sorted({r["eps"] for r in records}):
        rs = [r for r in records
              if r["eps"] == eps and r["J_gap"] > 0 and r["I"] > 10 * r["I_star"]]
        if rs:
            out[eps] = min(r["I"] / r["J_gap"] for r in rs)
    return out


def load_chat_of_eps(problem):
    out = {}
    for eps in EPS_SCHEDULE:
        path = RESULTS / f"e2_{problem}_eps{eps:g}.json"
        if path.exists():
            d = json.loads(path.read_text())
            rates = [r["rate"] for r in d["runs"] if np.isfinite(r["rate"])]
            if rates:
                out[eps] = (float(np.median(rates)),
                            float(np.min(rates)), float(np.max(rates)))
    return out


def load_lambda_min(problem, grid):
    path = RESULTS / f"e3_lambda_min_{problem}_{grid}.json"
    if not path.exists():
        return {}
    return {float(k): v for k, v in json.loads(path.read_text()).items()}


def main(problem, grid):
    c = load_c_of_eps(problem, grid)
    chat = load_chat_of_eps(problem)
    lam = load_lambda_min(problem, grid)
    if not (c or chat or lam):
        raise SystemExit("no E1/E2/E3 results found — run those first")

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    if c:
        e = np.array(sorted(c))
        ax.loglog(e, [c[x] for x in e], "o-", label=r"$c(\varepsilon)$ (E1)")
    if chat:
        e = np.array(sorted(chat))
        med = np.array([chat[x][0] for x in e])
        lo = np.array([chat[x][1] for x in e])
        hi = np.array([chat[x][2] for x in e])
        ax.loglog(e, med, "s-", label=r"$\hat c(\varepsilon)$ (E2)")
        ax.fill_between(e, lo, hi, alpha=0.2)
    if lam:
        e = np.array(sorted(lam))
        ax.loglog(e, [lam[x] for x in e], "^-",
                  label=r"$\lambda_{\min}(\varepsilon)$ (E3)")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("estimator")
    ax.set_title(f"Degeneration of the PL constant — {problem}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    out = FIG_DIR / f"star_{problem}_{grid}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["moons", "housing"], default="moons")
    ap.add_argument("--grid", default="asym")
    a = ap.parse_args()
    main(a.problem, a.grid)
