# Hoja de ruta para terminar el preprint

Estado de partida: `paper/main.tex` (16 págs.). Los resultados que faltan aparecen en **naranja** (`\pending{}`) y las tareas editoriales en **rojo** (`\todo{}`). No hay ningún número inventado.

Cómputo estimado en total: ~1,5–2 semanas de CPU de portátil, que en buena parte se pueden correr en paralelo. Las tareas están ordenadas por dependencia y prioridad.

---

## Bloque 0 — Preparación (½ día, sin cómputo largo)

| # | Tarea | Detalle |
|---|---|---|
| 0.1 | Entorno | `pip install -e ".[dev]"` y `pytest tests/`: los 18 tests deben pasar |
| 0.2 | Registrar la KL en E2 | Modificar `experiments/e2_descent.py` para guardar en cada `run` la curva $\int\mathcal E(\nu^s_t\mid\nu^*_t)\,dt$ con `gradient_flow.kl_to` (ya existe). Es necesario para §6.2(d). Los JSON actuales de E2 no la tienen |
| 0.3 | Nuevo script `experiments/fig_optimum.py` | Regenera las figuras de §4.2 (trayectorias y marginales) y la tabla de concentración **desde los checkpoints de este repo**: ε=0.05 asym y ε=0.02 `asymflow`. Las figuras actuales vienen de la memoria y se hicieron sobre el iterado **no convergido** de ε=0.02 |
| 0.4 | Nuevo script `experiments/13_genericity.py` | Variantes de datos: *moons* con semillas 0–4 (ruido 0.12) y ruido 0.08/0.16; *housing* con subsamples 0–4. Para cada una: continuación pilot hasta 0.05, $\lambda_{\min}$ en 0.1 y 0.05, nube E1 reducida (30 puntos) en 0.1 y 0.05. Necesita que `datasets.py` acepte `seed`/`noise`/`subsample` como parámetros (hoy están fijos a SEED=42) |
| 0.5 | Nuevo script `experiments/14_jacobi.py` | *moons*, ε=0.05, pilot. Para $k_1\in\{2,4,6,8\}$: partículas iniciales $X^*(t_{k_1})$ (en el checkpoint, clave `X`), horizonte $[t_{k_1},T]$ con $K-k_1$ cortes, Picard desde el prior, TV frente a la restricción de $\nu^*$ y $\lambda_{\min}$ del problema recortado. Ojo al peso trapezoidal del primer corte |

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

- Genericidad (0.4): ~2–3 días en total. Rellena la Tabla `tab:genericity`.
- Jacobi (0.5): ~4 cortes × (Picard ~5 min + E3 ~2 h) ≈ 1 día. Da el texto de §5.4 y una figura pequeña o una fila de tabla.

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
