"""E1 figures from the saved clouds (no recomputation, no data download).

Reads results/e1_<problem>_<grid>.json and, where it exists, the cloud
measured against the flow-refined endpoint, results/e1_<problem>_<grid>flow.json,
which replaces the records of the same eps (at eps = 0.02 the fixed-point
iterate is not converged and admits no cloud point).

Writes
  * e1_scatter_<grid>.png   I_h vs J - J* for every eps, one row per benchmark,
                            with the envelope c(eps) = inf I_h / (J - J*);
  * e1_c_vs_eps_<grid>.png  c(eps) in log-log for both benchmarks, with the
                            fitted slopes.

Usage:
    python experiments/fig_e1.py --grid asym
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from common import FIG_DIR, RESULTS

PROBLEMS = ("moons", "housing")
COLORS = {"moons": "#2a78d6", "housing": "#eb6834"}
KIND_MARKERS = {"mixture": "o", "tilt": "s"}


def load_records(problem, grid):
    """E1 records keyed by eps; the flow-refined cloud overrides its eps."""
    by_eps = {}
    for name in (f"e1_{problem}_{grid}.json", f"e1_{problem}_{grid}flow.json"):
        path = RESULTS / name
        if not path.exists():
            continue
        recs = json.loads(path.read_text())
        for eps in {r["eps"] for r in recs}:
            by_eps[eps] = [r for r in recs if r["eps"] == eps]
    return dict(sorted(by_eps.items(), reverse=True))


def admissible(recs):
    return [r for r in recs if r["J_gap"] > 0 and r["I"] > 10 * r["I_star"]]


def envelope(recs):
    ok = admissible(recs)
    return min(r["I"] / r["J_gap"] for r in ok) if ok else None


def c_of_eps(problem, grid):
    out = {}
    for eps, recs in load_records(problem, grid).items():
        c = envelope(recs)
        if c is not None:
            out[eps] = c
    return dict(sorted(out.items()))


def plain_log_ticks(axis, values):
    """Label a short log axis at the given values, without minor labels."""
    axis.set_major_locator(ticker.FixedLocator(values))
    axis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    axis.set_minor_formatter(ticker.NullFormatter())


def slope(c):
    e = np.array(sorted(c))
    return float(np.polyfit(np.log(e), np.log([c[x] for x in e]), 1)[0])


def scatter(grid):
    data = {p: load_records(p, grid) for p in PROBLEMS}
    eps_all = sorted({e for d in data.values() for e in d}, reverse=True)
    fig, axes = plt.subplots(len(PROBLEMS), len(eps_all),
                             figsize=(3.0 * len(eps_all), 2.7 * len(PROBLEMS)),
                             squeeze=False)
    for row, problem in enumerate(PROBLEMS):
        for col, eps in enumerate(eps_all):
            ax = axes[row, col]
            recs = data[problem].get(eps)
            if not recs:
                ax.text(0.5, 0.5, "not computed", ha="center", va="center",
                        transform=ax.transAxes, color="0.45")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ok = admissible(recs)
                for kind, marker in KIND_MARKERS.items():
                    pts = [r for r in ok if r["kind"] == kind]
                    if pts:
                        ax.loglog([r["J_gap"] for r in pts],
                                  [r["I"] for r in pts], marker, ms=3.5,
                                  ls="none", alpha=0.75,
                                  color=COLORS[problem],
                                  mfc="none" if kind == "tilt" else None,
                                  label=kind)
                c = envelope(recs)
                if c is not None:
                    g = np.geomspace(min(r["J_gap"] for r in ok),
                                     max(r["J_gap"] for r in ok), 50)
                    ax.loglog(g, c * g, "k--", lw=1)
                    ax.text(0.04, 0.93, rf"$c={c:.3g}$", transform=ax.transAxes,
                            va="top", fontsize=9)
                ax.text(0.96, 0.05, f"{len(ok)}/{len(recs)}", ha="right",
                        transform=ax.transAxes, fontsize=8, color="0.35")
                ax.grid(alpha=0.25, which="major")
            if row == 0:
                ax.set_title(rf"$\varepsilon={eps:g}$", fontsize=10)
            if row == len(PROBLEMS) - 1:
                ax.set_xlabel(r"$J-J^*$")
            if col == 0:
                ax.set_ylabel(rf"{problem}: $\mathcal{{I}}_h$")
    axes[0, 0].legend(fontsize=7, loc="lower right", bbox_to_anchor=(1, 0.12))
    fig.tight_layout()
    out = FIG_DIR / f"e1_scatter_{grid}.png"
    fig.savefig(out, dpi=200)
    print(f"-> {out}")


def c_plot(grid):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for problem in PROBLEMS:
        c = c_of_eps(problem, grid)
        if len(c) < 2:
            continue
        e = np.array(sorted(c))
        s = slope(c)
        ax.loglog(e, [c[x] for x in e], "o-", color=COLORS[problem], ms=5,
                  lw=1.5, label=f"{problem} (slope {s:.2f})")
        print(f"  {problem}: c = {c}  slope = {s:.3f}")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$c(\varepsilon)$")
    plain_log_ticks(ax.xaxis, (0.02, 0.05, 0.1, 0.2, 0.5))
    plain_log_ticks(ax.yaxis, (0.1, 0.2, 0.5, 1, 2, 5))
    ax.grid(alpha=0.3, which="major")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / f"e1_c_vs_eps_{grid}.png"
    fig.savefig(out, dpi=200)
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="asym")
    a = ap.parse_args()
    scatter(a.grid)
    c_plot(a.grid)
