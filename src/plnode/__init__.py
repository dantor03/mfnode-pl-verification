"""plnode — Numerical verification of local Polyak-Lojasiewicz inequalities
for entropic mean-field Neural ODEs (Daudin & Delarue, arXiv:2507.08486).

Modules
-------
solver         Explicit Gibbs solver for Thm 1.4 (damped Picard + eps-continuation).
datasets       Frozen benchmark datasets (make_moons anchors, California top-2).
fisher         Discretized Fisher-information functional I(nu) and grad_a Psi.
gradient_flow  Finite-volume Fokker-Planck step for the Wasserstein gradient flow.
hessian        Matrix-free projected Hessian at nu* (Hv products + Lanczos).
perturbations  Symmetric/random perturbation generator around nu* (E1/E2/E4).
"""
from . import solver, datasets, fisher, gradient_flow, hessian, perturbations  # noqa: F401

__version__ = "0.1.0"
