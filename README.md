# Numerical verification of stability and local Polyak–Łojasiewicz inequalities for entropic mean-field Neural ODEs

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Preprint and code by **Daniel Torres González** (ICMAT–CSIC, JAE Intro fellowship).

📄 **Preprint (draft):** [`paper/main.pdf`](paper/main.pdf)

The preprint is a numerical companion to

> S. Daudin, F. Delarue, *Genericity of Polyak–Łojasiewicz inequalities for
> entropic mean-field Neural ODEs*, [arXiv:2507.08486](https://arxiv.org/abs/2507.08486) (2025).

It computes the optimal control ν\* of the entropic mean-field control problem
behind continuum-depth ResNets and tests the two results of that paper:

- **Stability and uniqueness** (Meta-Theorem 1.3): independence of the fixed
  point from the initialization, smallest Hessian eigenvalue λ_min(ε) > 0,
  genericity across perturbed data, and the Jacobi condition.
- **Local PL inequality and the conjectured exponential convergence**
  (Meta-Theorem 1.4): the empirical PL constant c(ε), the descent rate ĉ(ε),
  and λ_min(ε), three independent estimators of the degeneration as ε → 0⁺.

Status: **draft**. Results still to be computed are marked in orange in the
PDF; see [`ROADMAP.md`](ROADMAP.md).

## Layout

```
paper/            preprint (main.tex, references.bib, figures/, main.pdf)
src/plnode/       library
  solver.py          explicit Gibbs solver: damped fixed point + ε-continuation,
                     backward equation by the Jacobian cocycle
  fisher.py          discretized Fisher functional 𝓘_h (compatible assembly)
  gradient_flow.py   Scharfetter–Gummel Wasserstein gradient flow
  hessian.py         matrix-free projected Hessian, LOBPCG λ_min
  perturbations.py   perturbation clouds around ν*
  datasets.py        frozen benchmarks (moons / California Housing top-2)
experiments/      runners (00–03 solver & gates, 10–12 uniqueness/continuation/
                  concentration, 13 genericity, 14 Jacobi condition,
                  e1–e4 PL experiments, fig_star, fig_optimum)
results/          frozen ν* checkpoints (nu_star/), experiment outputs, logs
tests/            correctness suite (pytest)
```

## Reproduce

```bash
pip install -e ".[dev]"
pytest tests/                                   # ~4 s
python experiments/00_compute_nu_star.py --all  # or use the frozen checkpoints
python experiments/e1_pl_constant.py --problem moons --grid asym
python experiments/e2_descent.py --problem moons --eps 0.05 --n-steps 600
python experiments/e3_hessian.py --problem moons --grid pilot --all-eps
cd paper && latexmk -pdf main.tex
```

Everything runs on a laptop CPU (numpy/scipy); timings are listed in the
paper's reproducibility appendix.

## License

MIT (code). The preprint text and figures are © the author.
