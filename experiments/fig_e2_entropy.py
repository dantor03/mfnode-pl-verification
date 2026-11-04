"""E2 figure: cost gap and integrated relative entropy along the same descent.

For one descent per panel (amplitude delta = 0.05, seed 0) both quantities
are normalized by their initial value, so that their decay rates can be
compared directly.  At moderate eps they decay together; at small eps the
entropy develops a slow tail of small amplitude that the cost does not show.

Reads results/e2_<problem>_eps<eps>.json (written by e2_descent.py).

Usage:
    python experiments/fig_e2_entropy.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG_DIR, RESULTS

PANELS = (("moons", 0.1), ("moons", 0.02), ("housing", 0.02))
COST, ENTROPY = "#eb6834", "#6a3d9a"


def one_run(problem, eps, delta=0.05, seed=0):
    d = json.loads((RESULTS / f"e2_{problem}_eps{eps:g}.json").read_text())
    rates = [r["rate"] for r in d["runs"]]
    kl_rates = [r["kl_rate"] for r in d["runs"]]
    run = next(r for r in d["runs"] if r["delta"] == delta and r["seed"] == seed)
    return run, float(np.median(rates)), float(np.median(kl_rates))


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.0 * len(PANELS), 3.3),
                             sharey=True)
    for ax, (problem, eps) in zip(axes, PANELS):
        run, chat, kl_rate = one_run(problem, eps)
        s, g = np.array(run["s"]), np.array(run["gap"])
        sk, kl = np.array(run["s_kl"]), np.array(run["kl"])
        ax.semilogy(s, g / g[0], color=COST, lw=1.8,
                    label=rf"$J-J^*$ (median rate {chat:.3g})")
        ax.semilogy(sk, kl / kl[0], color=ENTROPY, lw=1.8, ls="--",
                    label=rf"$\int\mathcal{{E}}(\nu^s_t|\nu^*_t)\,dt$ "
                          rf"(median rate {kl_rate:.3g})")
        ax.set_title(rf"{problem}, $\varepsilon={eps:g}$", fontsize=10)
        ax.set_xlabel("descent time $s$")
        ax.grid(alpha=0.3, which="major")
        ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_ylabel("relative to the initial value")
    fig.tight_layout()
    out = FIG_DIR / "e2_entropy.png"
    fig.savefig(out, dpi=200)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
