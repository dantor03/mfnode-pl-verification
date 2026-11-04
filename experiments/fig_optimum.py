"""Figures and table of Section 4.2 ("What the optimal control looks like").

Regenerates, from the checkpoints of this repository (never from a
non-converged iterate):

  * optimum_trajectories_<grid>_eps<eps>.png   particle trajectories under the
    optimal mean field, moons (by class) and housing (by target);
  * optimum_marginals_<problem>_<grid>_eps<eps>.png   1D marginals of nu*_t
    along the layers vs the prior;
  * results/concentration.json + LaTeX rows for the concentration table
    (effective support, n_50%, top-10 overlap t=0 vs t=T) on the pilot grid,
    including the flow-refined eps = 0.02 endpoint when it exists.

Usage:
    python experiments/fig_optimum.py                      # asym, eps = 0.05
    python experiments/fig_optimum.py --grid asymflow --eps 0.02
    python experiments/fig_optimum.py --table-only
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG_DIR, R_GRID, RESULTS, T, ckpt_path, get_problem
from plnode import solver

AXIS_LABELS = [r"$a_0^{(1)}$", r"$a_0^{(2)}$", r"$a_1^{(1)}$", r"$a_1^{(2)}$",
               r"$a_2$"]
TABLE_CASES = [("pilot", 0.5), ("pilot", 0.1), ("pilot", 0.05),
               ("pilotflow", 0.02)]


def load(problem, grid, eps):
    path = ckpt_path(problem, grid, eps)
    if not path.exists():
        print(f"  (missing {path.name}, skipped)")
        return None
    d = np.load(path)
    return d["nu"], d["A"], tuple(int(m) for m in d["mp_per_axis"])


def trajectories(grid, eps):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, name in zip(axes, ("moons", "housing")):
        out = load(name, grid, eps)
        if out is None:
            ax.set_visible(False)
            continue
        nu, A, _ = out
        prob = get_problem(name)
        K = nu.shape[0] - 1
        X, _ = solver.forward_pass(nu, prob["x0"], A, 2, T / K, K)
        if name == "moons":
            colors = np.where(prob["labels"] > 0, "tab:red", "tab:blue")
            for i in range(X.shape[1]):
                ax.plot(X[:, i, 0], X[:, i, 1], color=colors[i], lw=0.4,
                        alpha=0.35)
            ax.scatter(X[-1, :, 0], X[-1, :, 1], c=colors, s=6)
            ax.plot([-1, 1], [0, 0], "kx", ms=9, mew=2)
            ax.axvline(0, color="k", lw=0.6, ls="--")
            ax.set_title(rf"moons, $\varepsilon$={eps} "
                         rf"(acc = {prob['metric'](X[-1]):.3f})")
        else:
            tgt = prob["targets"]
            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(tgt.min(), tgt.max())
            for i in range(X.shape[1]):
                ax.plot(X[:, i, 0], X[:, i, 1], color=cmap(norm(tgt[i])),
                        lw=0.4, alpha=0.35)
            sc = ax.scatter(X[-1, :, 0], X[-1, :, 1], c=tgt, cmap=cmap,
                            norm=norm, s=6)
            fig.colorbar(sc, ax=ax, label="target $y$")
            ax.set_title(rf"housing, $\varepsilon$={eps} "
                         rf"($R^2$ = {prob['metric'](X[-1]):.3f})")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / f"optimum_trajectories_{grid}_eps{eps:g}.png"
    fig.savefig(out, dpi=200)
    print(f"-> {out}")


def marginals(problem, grid, eps):
    out = load(problem, grid, eps)
    if out is None:
        return
    nu, A, mp = out
    K1 = nu.shape[0]
    prior = np.exp(solver.log_prior(A)).reshape(mp)
    nu_t = nu.reshape((K1,) + mp)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(mp), figsize=(3.0 * len(mp), 2.8))
    for j, ax in enumerate(axes):
        grid_j = np.linspace(-R_GRID, R_GRID, mp[j])
        other = tuple(i for i in range(len(mp)) if i != j)
        for k in range(K1):
            ax.plot(grid_j, nu_t[k].sum(axis=other), "o-", ms=2.5, lw=1,
                    color=cmap(k / (K1 - 1)))
        ax.plot(grid_j, prior.sum(axis=other), "k--", lw=1.2, label="prior")
        ax.set_xlabel(AXIS_LABELS[j])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("marginal mass")
    axes[0].legend(fontsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T))
    fig.colorbar(sm, ax=axes, label="layer $t$", fraction=0.02)
    fig.suptitle(rf"{problem}: marginals of $\nu^*_t$, $\varepsilon$={eps} "
                 f"({grid} grid)", fontsize=10)
    out_path = FIG_DIR / f"optimum_marginals_{problem}_{grid}_eps{eps:g}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"-> {out_path}")


def top_k_inclusion(w, k=10, rtol=1e-9):
    """Probability that each node is among the k heaviest of ``w``.

    The a <-> -a symmetry (and, at large eps, the permutation symmetry of the
    grid) makes many weights exactly equal, so a plain argsort cut at k splits
    tied nodes arbitrarily and the result depends on the sorting algorithm.
    Nodes tied with the k-th weight share the remaining slots equally, which
    is the inclusion probability under uniform random tie-breaking.
    """
    cut = np.sort(w)[::-1][k - 1]
    tie = np.abs(w - cut) <= rtol * cut
    p = ((w > cut) & ~tie).astype(float)
    p[tie] = (k - p.sum()) / tie.sum()
    return p


def concentration(nu):
    """Concentration statistics of the time-averaged mass (Table 4.2)."""
    M = nu.shape[1]
    wbar = nu.mean(axis=0)
    srt = np.sort(wbar)[::-1]
    H = -(wbar * np.log(np.maximum(wbar, 1e-300))).sum()
    csum = np.cumsum(srt)
    # expected size of top10(t=0) & top10(t=T) under random tie-breaking
    overlap = float(top_k_inclusion(nu[0]) @ top_k_inclusion(nu[-1]))
    return dict(top1_mass=float(srt[: max(1, M // 100)].sum()),
                n_eff=float(np.exp(H)),
                n50=int(np.searchsorted(csum, 0.5) + 1),
                top10_overlap=overlap, M=M)


def table():
    rows = {}
    for problem in ("moons", "housing"):
        for grid, eps in TABLE_CASES:
            out = load(problem, grid, eps)
            if out is None:
                continue
            st = concentration(out[0])
            rows[f"{problem}_{grid}_eps{eps:g}"] = dict(problem=problem,
                                                        grid=grid, eps=eps,
                                                        **st)
            print(f"\\emph{{{problem}}} & {eps:g} & {st['top1_mass']:.2f} & "
                  f"{st['n_eff']:.0f} & {st['n50']} & "
                  f"{st['top10_overlap']:.1f}/10\\\\")
    out = RESULTS / "concentration.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="asym")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    if not a.table_only:
        trajectories(a.grid, a.eps)
        for p in ("moons", "housing"):
            marginals(p, a.grid, a.eps)
    table()
