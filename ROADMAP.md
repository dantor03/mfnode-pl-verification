# Hoja de ruta del preprint

Estado: **borrador completo**. Todos los resultados de `paper/main.tex` están
calculados; no queda ningún `\pending{}`. Solo queda un `\todo{}` editorial
(el tag del commit de envío).

## Hecho

| Bloque | Contenido |
|---|---|
| Preparación | 21 tests en verde; KL a lo largo del flujo en E2; figuras del óptimo y tabla de concentración; scripts de genericidad (13) y Jacobi (14) |
| *Housing* | E1 en la malla asimétrica (5 valores de ε, ε=0.02 contra el endpoint refinado), E2 con KL, E3 en la malla piloto y en el endpoint refinado; figura estrella con dos paneles |
| *Moons* | E2 con KL; E3 en ε=0.02 contra el endpoint refinado (λ_min = 0.0081) |
| Genericidad | 12 variantes de datos + la referencia: continuación, λ_min en ε=0.1 y 0.05 y nube reducida de 30 puntos |
| Jacobi | Reinicio desde capas intermedias en *moons*, ε=0.05 |
| Comprobaciones | Dependencia de malla de c (`15_grid_check.py`): la misma nube da una c el doble de grande en la malla asimétrica que en la piloto; solo se comparan exponentes entre mallas. Régimen cuadrático (`16_quadratic_regime.py`): en *moons* se encoge más de cuatro veces entre ε=0.05 y 0.02. Cola lenta de la entropía en ε=0.02 (`fig_e2_entropy.py`) |
| Autovalores | Repetición con 1200–1500 iteraciones de LOBPCG de los λ_min que no habían convergido; los residuos están en el texto y en `results/genericity_parts/relam-*.json` |

Parte del cálculo (*housing*, genericidad y las repeticiones de autovalores)
se hizo en runners de GitHub Actions; los logs son
`results/logs/actions_*.log`.

## Puntos abiertos (declarados en el preprint)

- El nivel de los estimadores depende de la malla; los exponentes no. Un
  estudio de convergencia en malla del nivel de c queda abierto.
- La cola lenta de la entropía relativa en ε=0.02 necesitaría un coste de
  referencia con precisión muy por debajo de 1e-9.
- El radio del entorno PL: E4 no ve frontera en ε=0.05; lo que sí se encoge
  en *moons* es la región donde el coste es cuadrático.

## Pendiente (decisiones de Daniel)

1. Revisión de Rafael.
2. Decidir si antes de subir se envía un email a Daudin–Delarue.
3. arXiv: categoría principal **math.OC**; cruzadas **math.NA** y
   **cs.LG**. Subir el `.tex`, el `.bbl` y las figuras (no el `.bib`).
4. Commit final y tag `v1-arxiv`; quitar el `\todo` del apéndice de
   reproducibilidad y las macros `\todo`/`\pending` del preámbulo.
5. Tras la aceptación: añadir el identificador arXiv al README.
