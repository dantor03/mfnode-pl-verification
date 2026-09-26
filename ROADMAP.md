# Hoja de ruta para terminar el preprint

Estado de partida: `paper/main.tex` (16 págs.). Los resultados que faltan aparecen en **naranja** (`\pending{}`) y las tareas editoriales en **rojo** (`\todo{}`). No hay ningún número inventado.

Cómputo estimado en total: ~1,5–2 semanas de CPU de portátil, que en buena parte se pueden correr en paralelo. Las tareas están ordenadas por dependencia y prioridad.

---

## Bloque 0 — Preparación ✅ HECHO

| # | Tarea | Estado |
|---|---|---|
| 0.1 | Entorno y tests | `pytest tests/`: 21 tests en verde (2 nuevos: KL a lo largo del flujo; reinicio en capa intermedia = restricción de ν*, la base discreta del test de Jacobi) |
| 0.2 | KL en E2 | `gradient_flow(..., nu_ref=nu_star)` registra $\int\mathcal E(\nu^s_t\mid\nu^*_t)\,dt$ en cada refresco; `e2_descent.py` guarda `kl`, `s_kl` y `kl_rate` por run, y la figura `e2_kl_*.png` |
| 0.3 | `experiments/fig_optimum.py` | Figuras de trayectorias y marginales desde los checkpoints; tabla de concentración (`results/concentration.json`). **Ya ejecutada la tabla**: reproduce la de la memoria y añade *moons* ε=0.02 (endpoint refinado): $N_{\rm eff}=81$, $n_{50\%}=17$. Metida en el preprint |
| 0.4 | `experiments/13_genericity.py` | 12 variantes (`--list`), reanudable, paralelizable con `--variant`. `datasets.housing_problem` acepta ahora `features=` para fijar (MedInc, AveOccup) al cambiar la submuestra |
| 0.5 | `experiments/14_jacobi.py` | Reinicio en $k_1\in\{2,4,6,8\}$, TV frente a la restricción y $\lambda_{\min}$; figura `jacobi_*.png` (incluye $t_1=0$ desde E3 si existe) |

Comandos que faltan de este bloque (rápidos, minutos):

```bash
cd experiments
python fig_optimum.py                              # trayectorias + marginales, asym ε=0.05
python fig_optimum.py --grid asymflow --eps 0.02   # sólo moons hasta que exista housing asymflow
```

Luego, en `paper/main.tex`, sustituir `flow_trajectories.png` y
`02_explicit_moons_mse_marginals_asym.png` por
`optimum_trajectories_asym_eps0.05.png` y `optimum_marginals_moons_asym_eps0.05.png`
y quitar los `\todo{regenerate …}`.

## Bloque 1 — *Housing* (obligatorio; lo esencial para el revisor)

```bash
# endpoint ε = 0.02 refinado por el flujo (igual que en moons)
python experiments/03_refine_flow.py --problem housing --grid pilot --eps 0.02
python experiments/03_refine_flow.py --problem housing --grid asym  --eps 0.02   # ~7 h
# E1 (5 valores de ε; en 0.02 contra el endpoint refinado)
python experiments/e1_pl_constant.py --problem housing --grid asym
python experiments/e1_pl_constant.py --problem housing --grid asymflow --eps 0.02
# E2 (tras la tarea 0.2, que añade la KL)
for e in 0.5 0.2 0.1 0.05; do python experiments/e2_descent.py --problem housing --eps $e --n-steps 600; done
python experiments/e2_descent.py --problem housing --eps 0.02 --grid pilotflow --n-steps 1200
# E3 (al menos 0.5, 0.1, 0.05)
python experiments/e3_hessian.py --problem housing --grid pilot --all-eps
# figura estrella con dos paneles
python experiments/fig_star.py --problem housing --grid asym
```

Coste: ~10 h (E1) + ~8 h (E2) + ~10 h (E3) + ~8 h (flow-refine).

## Bloque 2 — Completar *moons*

- Repetir E2 con la KL registrada (tarea 0.2): ~6 h. Da §6.2(d) y reemplaza los JSON actuales con el mismo contenido y la KL añadida.
- E3 en ε=0.02 contra `pilotflow`: ~3 h. Rellena la celda que falta en la tabla de estimadores.

## Bloque 3 — Experimentos nuevos

```bash
cd experiments
# genericidad: 12 variantes independientes; p.ej. 3 procesos en paralelo
python 13_genericity.py --variant 0 1 2 3 &
python 13_genericity.py --variant 4 5 6 7 &
python 13_genericity.py --variant 8 9 10 11 &
# (prueba rápida sin Hessiana: --no-hessian)

# Jacobi (moons, ε=0.05; la fila t1=0 sale de e3_lambda_min_moons_pilot.json)
python 14_jacobi.py --problem moons --eps 0.05
```

- Genericidad: ~2–3 días en total (continuación ~30 min + 2 λ_min ~2–3 h + nubes ~30 min por variante). Rellena `tab:genericity`. Si varios procesos terminan a la vez escriben el mismo `results/genericity.json`; el script relee antes de escribir, pero es mejor que no coincidan.
- Jacobi: ~4 cortes × (Picard ~5 min + λ_min ~2 h) ≈ 1 día. Da el texto de §5.4 y la figura `jacobi_moons_eps0.05.png`.

## Bloque 4 — Redacción final

1. Rellenar cada `\pending{}` con los números nuevos (`grep -n pending paper/main.tex`).
2. Abstract e introducción: frase sobre *housing*, genericidad y Jacobi.
3. Discusión: contrastar la heurística de concentración con *housing*, donde $\nu^*$ apenas se concentra. Es la predicción interesante: ¿la pendiente se queda en ~1 sin endurecerse?
4. Agradecimientos: nombre del tutor y referencia de la beca JAE Intro.
5. Revisión: Rafael. Decidir si antes de subir se envía un email a Daudin–Delarue.
6. Comprobar el tamaño: objetivo ≈ 15–16 págs. con referencias.

## Bloque 5 — Subida a arXiv

1. `latexmk -pdf paper/main.tex` sin warnings; quitar las macros `\todo`/`\pending` del preámbulo (deben quedar sin usos).
2. Commit final y tag `v1-arxiv`, y actualizar la URL del apartado de reproducibilidad si el repositorio cambia de nombre.
3. arXiv: categoría principal **math.OC**; cruzadas **math.NA** y **cs.LG**. Subir el `.tex`, el `.bbl` y las figuras (no el `.bib`).
4. Tras la aceptación: añadir el identificador arXiv al README.

## Riesgos

| Riesgo | Plan |
|---|---|
| E1 de *housing* en ε=0.02 con pocos puntos admisibles | Usar el endpoint `asymflow`, como en *moons* |
| Genericidad: alguna variante no converge en 0.05 | Reportarlo tal cual. Si hace falta, continuación fina 0.1→0.05 |
| Jacobi: diferencia de TV por el peso trapezoidal del primer corte | Comparar costes con la misma cuadratura; documentarlo |
| Tiempo de cómputo | Bloques 1 y 3 en paralelo (8 núcleos); los checkpoints permiten reanudar |
