"""Neuronas mas activadas: atomos del grid con mayor masa bajo nu*_t.

Para cada dataset (moons, housing) y varios eps, calcula:
  - masa promediada en el tiempo w_j = (1/K1) sum_k nu_k[j]
  - top atomos agrupados en pares +/-a (simetria exacta nu(a)=nu(-a))
  - concentracion: masa del top-1% / top-5% de atomos, soporte efectivo exp(H)
  - estabilidad temporal del ranking (top-10 en t=0 vs t=T)
"""
from pathlib import Path

import numpy as np

BASE = str(Path(__file__).resolve().parents[1] / "results" / "nu_star")

CASES = [
    ("moons",   [0.5, 0.1, 0.05]),
    ("housing", [0.5, 0.1, 0.05]),
]


def pair_index(A):
    """Para cada atomo j, indice de su companero -a (el grid es simetrico)."""
    M = A.shape[0]
    # grid 5^5 en [-R,R]: -a esta en el grid exactamente
    key = {tuple(np.round(a, 10)): j for j, a in enumerate(A)}
    return np.array([key[tuple(np.round(-A[j], 10))] for j in range(M)])


def analyze(tag, eps):
    d = np.load(f"{BASE}/{tag}_pilot_eps{eps}.npz", allow_pickle=True)
    nu, A = d["nu"], d["A"]          # (K1, M), (M, 5)
    K1, M = nu.shape
    wbar = nu.mean(axis=0)           # masa promedio temporal

    pidx = pair_index(A)

    # agrupar pares (j, pidx[j]) con j < pidx[j]; auto-pares (a=0-simetricos) solos
    seen = np.zeros(M, bool)
    pairs = []
    for j in np.argsort(-wbar):
        if seen[j]:
            continue
        seen[j] = True
        jp = pidx[j]
        mass = wbar[j] if jp == j else wbar[j] + wbar[jp]
        seen[jp] = True
        pairs.append((mass, j))
    pairs.sort(key=lambda t: -t[0])

    # concentracion
    srt = np.sort(wbar)[::-1]
    top1 = srt[: max(1, M // 100)].sum()
    top5 = srt[: max(1, M // 20)].sum()
    H = -(wbar * np.log(np.maximum(wbar, 1e-300))).sum()
    neff = np.exp(H)
    csum = np.cumsum(srt)
    n50 = int(np.searchsorted(csum, 0.5) + 1)
    n90 = int(np.searchsorted(csum, 0.9) + 1)

    # estabilidad temporal del top-10
    top0 = set(np.argsort(-nu[0])[:10])
    topT = set(np.argsort(-nu[-1])[:10])
    overlap = len(top0 & topT)

    print(f"\n=== {tag}  eps={eps}  (M={M}, K1={K1}) ===")
    print(f"masa top-1% atomos: {top1:.3f}   top-5%: {top5:.3f}   "
          f"soporte efectivo exp(H) = {neff:.0f} atomos ({100*neff/M:.1f}%)")
    print(f"atomos para cubrir 50% de la masa: {n50}   90%: {n90}")
    print(f"solape top-10 entre t=0 y t=T: {overlap}/10")
    print("top-5 pares (+/-a), masa conjunta promedio temporal:")
    print("   masa    a0 (salida)        a1 (entrada)      a2 (sesgo)")
    for mass, j in pairs[:5]:
        a = A[j]
        print(f"  {mass:.4f}  ({a[0]:+.1f}, {a[1]:+.1f})    "
              f"({a[2]:+.1f}, {a[3]:+.1f})    {a[4]:+.1f}")
    # variacion temporal de la masa del par top
    m0, j0 = pairs[0]
    jp0 = pidx[j0]
    tmass = nu[:, j0] + (nu[:, jp0] if jp0 != j0 else 0)
    print(f"masa del par top a lo largo de t: min={tmass.min():.4f} "
          f"max={tmass.max():.4f}")
    return pairs, wbar


if __name__ == "__main__":
    for tag, eps_list in CASES:
        for eps in eps_list:
            analyze(tag, eps)
