"""Star figure — lambda_min(eps), c(eps) and c-hat(eps) on the same axes.

Three independent estimators of the same degeneration as eps -> 0+; if they
collapse onto a common power law, that is the central result of the paper.

Reads the JSON outputs of E1 (c; the flow-refined cloud replaces its eps, see
fig_e1.py), E2 (c-hat, one JSON per eps; at eps = 0.02 relative to the
flow-refined endpoint) and E3 (lambda_min on the pilot grid, plus the
flow-refined endpoint when it exists), and overlays them in log-log, one
panel per benchmark.

Usage:
    python experiments/fig_star.py --grid asym
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from common import FIG_DIR, RESULTS
from fig_e1 import c_of_eps, plain_log_ticks
from plnode.solver import EPS_SCHEDULE

PROBLEMS = ("moons", "housing")
COLORS = {"c": "#2a78d6", "chat": "#eb6834", "lam": "#1baf7a"}


def load_c_of_eps(problem, grid):
    return c_of_eps(problem, grid)


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


def load_lambda_min(problem, grid="pilot"):
    out = {}
    for g in (grid, f"{grid}flow"):
        path = RESULTS / f"e3_lambda_min_{problem}_{g}.json"
        if path.exists():
            out.update({float(k): v for k, v in
                        json.loads(path.read_text()).items()})
    return out


def slope(d):
    e = np.array(sorted(d))
    return float(np.polyfit(np.log(e), np.log([d[x] for x in e]), 1)[0])


def panel(ax, problem, grid):
    c = load_c_of_eps(problem, grid)
    chat = load_chat_of_eps(problem)
    lam = load_lambda_min(problem)
    if c:
        e = np.array(sorted(c))
        lab = r"$c(\varepsilon)$ (E1)"
        if len(c) > 1:
            lab += f", slope {slope(c):.2f}"
        ax.loglog(e, [c[x] for x in e], "o-", color=COLORS["c"], label=lab)
    if chat:
        e = np.array(sorted(chat))
        med = np.array([chat[x][0] for x in e])
        lab = r"$\hat c(\varepsilon)$ (E2)"
        if len(chat) > 1:
            lab += f", slope {slope({x: chat[x][0] for x in e}):.2f}"
        ax.loglog(e, med, "s-", color=COLORS["chat"], label=lab)
        ax.fill_between(e, [chat[x][1] for x in e], [chat[x][2] for x in e],
                        color=COLORS["chat"], alpha=0.2, lw=0)
    if lam:
        e = np.array(sorted(lam))
        lab = r"$\lambda_{\min}(\varepsilon)$ (E3)"
        if len(lam) > 1:
            lab += f", slope {slope(lam):.2f}"
        ax.loglog(e, [lam[x] for x in e], "^-", color=COLORS["lam"], label=lab)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_title(problem, fontsize=10)
    plain_log_ticks(ax.xaxis, (0.02, 0.05, 0.1, 0.2, 0.5))
    ax.yaxis.set_major_locator(ticker.LogLocator(numticks=10))
    ax.grid(alpha=0.3, which="major")
    ax.legend(fontsize=8, loc="lower right")
    return bool(c or chat or lam)


def main(grid):
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(9.6, 4.0),
                             sharey=True)
    if not any([panel(ax, p, grid) for ax, p in zip(axes, PROBLEMS)]):
        raise SystemExit("no E1/E2/E3 results found — run those first")
    axes[0].set_ylabel("estimator")
    fig.tight_layout()
    out = FIG_DIR / f"star_{grid}.png"
    fig.savefig(out, dpi=200)
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="asym")
    a = ap.parse_args()
    main(a.grid)
