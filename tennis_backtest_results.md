# Backtest walk-forward — motor de tenis
Generado: 2026-08-09 · `backtest_tennis.py` · 14,320 partidos reales ATP+WTA 2024–2026

> **Actualización 2026-08-09 — fix del cold-start de Elo:** la sección
> "el ranking oficial le gana al modelo" de abajo quedó resuelta. Ver
> sección "Fix: burn-in de Elo (2015-2026)" al final de este documento —
> el modelo ahora supera al ranking oficial (63.7% vs 63.45% accuracy) y
> ya está activo en producción (`tennis_elo_ratings.json` regenerado con
> 62,128 partidos 2015-2026, antes 14,320 de 2024-2026 solamente).
>
> **Actualización 2026-08-09 — bug crítico de orden encontrado y
> corregido:** durante el experimento de decay por inactividad se
> encontró que `combinar_archivos()` ordenaba los partidos **solo por
> fecha** — y el CSV fuente comparte una única `tourney_date` (la fecha
> de INICIO del torneo) entre TODAS sus rondas, listadas en el archivo en
> orden DESCENDENTE (Final primero, Primera Ronda al final). El sort
> estable de Python preservaba ese orden en los empates, así que el
> backtest procesaba la Final de un torneo **antes** que su Primera
> Ronda — walk-forward roto, con fuga de información hacia el pasado.
> Ya corregido (`src/providers/tennis_data_loader.py`, ordena también
> por ronda real dentro de cada fecha). Se re-verificaron TODOS los
> resultados anteriores de este documento contra el fix: burn-in,
> shrink_elo y Elo por superficie no cambiaron de forma significativa
> (diferencias de ±0.15% en Brier, dentro de ruido) — sus conclusiones
> siguen siendo válidas. El experimento de decay sí estaba completamente
> corrompido por este bug (ver sección "Decay de Elo por inactividad"
> más abajo) — se volvió a correr desde cero con el orden correcto.

Metodología: para cada partido, en orden cronológico, se predice usando
**solo** el Elo y la forma calculados con los partidos anteriores (nunca
información del partido en curso ni futuros), con el mismo motor de
producción (`TennisImprovedEngine`, no una reimplementación aparte).
Después de predecir, el Elo/forma se actualiza con el resultado real
antes de pasar al siguiente partido. La probabilidad evaluada es siempre
"P(gana el jugador que realmente ganó)" — evita cualquier sesgo de cómo
se etiquete jugador 1 vs jugador 2.

## Resultado 1: el modelo SÍ predice mejor que el azar

| | Brier score ↓ | Log-loss ↓ | Accuracy ↑ |
|---|---|---|---|
| **Modelo actual (Elo+superficie+forma)** | **0.22628** | **0.64208** | **61.03%** |
| Baseline coin-flip (p=0.5 siempre) | 0.25000 | 0.69315 | 50.00% |
| Baseline "gana el mejor ranking oficial" | — (no probabilístico) | — | **63.45%** |

Brier score 9.5% mejor que el coin-flip, log-loss 7.4% mejor — el modelo
tiene señal real, no es ruido.

## Resultado 2 (honesto, no esperado): el ranking oficial solo le gana al modelo

El baseline más simple posible — "gana el jugador con mejor ranking
ATP/WTA oficial", sin ningún modelo — acierta **63.45%** de las veces
(14,242 partidos con ranking disponible en ambos jugadores), contra
**61.03%** del modelo Elo+superficie+forma actual. La causa más probable:
nuestro Elo arranca de cero (1500) para todos en 2024 y solo tiene ~2
años de historial, mientras que el ranking ATP/WTA oficial incorpora años
de resultados previos a nuestra ventana de datos. No es una razón para
descartar el Elo (sigue aportando información distinta al ranking, y es
la única señal disponible para calcular EV contra cuotas), pero es un
hallazgo real que había que reportar, no esconder.

## Desglose

**Por superficie** — similar en las tres, sin sorpresas:
| Superficie | n | Brier | Accuracy |
|---|---|---|---|
| Clay | 4,409 | 0.22518 | 61.32% |
| Grass | 1,828 | 0.22621 | 62.23% |
| Hard | 8,083 | 0.22688 | 60.60% |

**Por nivel de torneo** — el modelo predice notablemente mejor en Grand
Slams (más partidos por jugador acumulados, mejor señal) que en ATP 250:
| Nivel | n | Brier | Accuracy |
|---|---|---|---|
| Grand Slam | 2,794 | 0.20091 | 67.48% |
| Masters 1000 | 3,975 | 0.22443 | 62.38% |
| ATP 500 | 1,724 | 0.22427 | 62.96% |
| ATP 250 | 5,827 | 0.24029 | 56.44% |

**Por paridad del enfrentamiento** — exactamente lo esperable: partidos
desbalanceados son fáciles de predecir, partidos parejos son casi
coin-flip por definición:
| Gap de Elo | n | Brier | Accuracy |
|---|---|---|---|
| Desbalanceado (>150) | 2,562 | 0.15644 | 79.82% |
| Moderado (50-150) | 5,490 | 0.23246 | 62.88% |
| Parejo (<50) | 6,268 | 0.24941 | 51.73% |

## Regresión a la media (shrinkage de Elo) — resultado: efecto marginal, NO se activa en producción

Grid de la constante `k` (`shrink_elo`, `elo_ajustado = 1500 + (elo - 1500) * games/(games+k)`):

| k | Brier | Log-loss | Accuracy |
|---|---|---|---|
| sin shrink | 0.22628 | 0.64208 | **61.03%** |
| 1 | 0.22617 | 0.64184 | 60.88% |
| 2 | 0.22610 | 0.64169 | 60.87% |
| 3 | 0.22605 | 0.64160 | 60.88% |
| **5** | **0.22600** | **0.64151** | 60.80% |
| 7 | 0.22599 | 0.64152 | 60.77% |
| 10 | 0.22603 | 0.64164 | 60.77% |
| 15 | 0.22617 | 0.64204 | 60.73% |
| 20 | 0.22637 | 0.64256 | 60.69% |
| 30 | 0.22686 | 0.64377 | 60.73% |
| 50 | 0.22792 | 0.64632 | 60.74% |
| 100 | 0.23035 | 0.65196 | 60.74% |

**Conclusión honesta:** la mejor mejora medida (k≈5-7) es de **0.00028
en Brier score (0.12% relativo)** — dentro del margen de ruido para 14k
partidos, no una mejora "real" en el sentido que pedías confirmar. Y
consistentemente **empeora la accuracy** en ~0.2-0.3 puntos porcentuales
en todo el rango probado. Hipótesis de por qué el efecto es tan chico:
el propio sistema de Elo con K-factor ya limita cuánto puede desviarse
un jugador de 1500 con pocos partidos (cada partido mueve el Elo como
máximo K=8-32 puntos), así que el "Elo inflado por poca muestra" que
motivó la idea es menos extremo en la práctica de lo que parecía en el
diagnóstico teórico.

**Decisión, siguiendo tu instrucción explícita ("solo si el backtest
confirma mejora real"):** `shrink_elo()` queda implementada, testeada y
disponible en el motor (parámetros `games1`/`games2` opcionales,
default `None` = sin cambios), pero **no se conecta a `app.py` ni a
`tennis_validator.py`** — no hay evidencia suficiente para activarla en
producción. Si en el futuro se agregan más años de histórico (más
partidos por jugador, especialmente para los que hoy tienen `games` muy
bajo — el 63.4% de los 1001 jugadores tiene menos de 15 partidos), vale
la pena volver a correr este mismo backtest antes de decidir activarla.

## Qué SÍ se deja andando

- `backtest_tennis.py`: infraestructura de backtesting reutilizable para
  medir el impacto de cualquier cambio futuro al modelo (Elo por
  superficie, H2H, etc.) contra este mismo baseline, antes de asumir que
  "mejora".
- `shrink_elo()` en `tennis_improved.py`: función correcta, testeada
  (`tests/test_tennis_shrinkage.py`), lista para usar si se decide
  reintentar con más datos.

## Fix: burn-in de Elo (2015-2026) — el modelo ahora supera al ranking oficial

### Hipótesis confirmada

`INITIAL_ELO = 1500.0` (`src/core/tennis_elo.py:24`) es igual para
absolutamente todos los jugadores sin importar su nivel real — con solo
3 años de historial (2024-2026, 14,320 partidos) el Elo no tenía tiempo
de converger a un valor representativo antes de que empezara la ventana
evaluada por el backtest. Confirmado como causa directa de que el
ranking oficial (que sí arrastra años de resultados previos) le ganara
al modelo.

### Opciones evaluadas

| Opción | Datos nuevos | Complejidad | Elegida |
|---|---|---|---|
| A — Seed inicial desde ranking (fórmula rank→Elo) | Ninguno (ya teníamos `winner_rank`/`loser_rank`) | Media, requiere derivar/validar una fórmula | No, innecesaria si C alcanza |
| B — Ranking oficial como feature del ensemble | Requiere ranking *actual* en cada request en vivo — no existe hoy en el pipeline de producción | Media-alta, agrega una dependencia nueva y otra fuente que mantener actualizada | No |
| **C — Burn-in: más años de historial para que el Elo converja** | Ninguno — la misma fuente (LuckyLoser91/TennisCourtLog) tiene datos reales hasta 1968 | Baja — solo cambiar el rango de años descargado | **Sí** |

C ataca la causa raíz (Elo sin tiempo de converger) sin inventar ninguna
fórmula ni agregar una dependencia nueva al request en vivo — se probó
primero y alcanzó el objetivo, así que A y B no hicieron falta.

### Resultado, mismo conjunto evaluado (14,320 partidos de 2024-2026) en los tres casos

| Ventana de burn-in | n. partidos totales procesados | Brier ↓ | Log-loss ↓ | Accuracy ↑ |
|---|---|---|---|---|
| Sin burn-in (solo 2024-2026) | 14,320 | 0.22628 | 0.64208 | 61.03% |
| **2015-2026 (12 años) — elegida** | 62,128 | **0.22071** | **0.63110** | **63.70%** |
| 2010-2026 (16 años) | ~más | 0.22085 | 0.63168 | 63.81% |
| Baseline ranking oficial | — | — | — | 63.45% |

12 y 16 años dan prácticamente el mismo resultado (retornos decrecientes
más allá de ~10-12 años) — se eligió 12 años por menor tiempo de
descarga/proceso sin pérdida de precisión medible. **El modelo ahora
supera el objetivo mínimo pedido (superar 63.45%).**

### Cambios aplicados

- `src/providers/tennis_data_loader.py`: default de `descargar_datos_tennis()`
  cambia de "últimos 3 años" a "últimos 12 años".
- `src/data/tennis_elo_ratings.json` regenerado con `calibrate_tennis_elo.py`:
  62,128 partidos reales 2015-2026 (antes 14,320 de 2024-2026), 2467
  jugadores (antes 1001), 1656 con forma real (antes 646).
- `backtest_tennis.py`: nuevo parámetro `evaluar_desde` para separar
  "partidos usados para calentar el Elo" de "partidos puntuados",
  reutilizable para cualquier backtest futuro con burn-in.

### Caveat honesto: jugadores retirados con Elo "congelado"

El sistema no tiene decay por inactividad — Roger Federer (retirado
2022) y Ashleigh Barty (retirada 2022) aparecen en el top 10 de Elo
porque su rating de pico simplemente no se mueve si no juegan más. No es
un problema práctico hoy (nadie va a pedir un análisis de un partido de
Federer), pero es una limitación real del modelo estático que vale la
pena resolver si se profundiza en P1/P2 (ej. descontar Elo por tiempo
sin jugar, o filtrar jugadores sin partidos en los últimos N meses del
autocomplete).

### Nota pendiente

El grid de `shrink_elo()` (sección anterior) se corrió contra la ventana
de 3 años — con 2467 jugadores y más partidos por jugador en la ventana
de 12 años, vale la pena re-correrlo antes de descartar definitivamente
la regresión a la media (no se hizo en este mismo cambio, para no mezclar
dos variables a la vez en la misma medición).

## Elo por superficie (P1) — resultado: NO mejora, no se activa

Mismo conjunto evaluado (14,320 partidos de 2024-2026, con el burn-in de
12 años ya activo como baseline: Brier 0.22071, accuracy 63.70%).

Se implementó `elegir_elo_superficie()`: un `TennisEloCalculator`
independiente por superficie (hard/clay/grass/carpet), actualizado
walk-forward solo con partidos de esa superficie, con fallback al Elo
overall cuando el jugador no tiene suficiente muestra en esa superficie
específica (`min_games_superficie`, umbral probado en grid).

| `min_games_superficie` | Brier | Accuracy |
|---|---|---|
| 5 (agresivo) | 0.22331 | 62.71% |
| 10 | 0.22442 | 62.54% |
| 20 | 0.22393 | 62.66% |
| 30 | 0.22334 | 62.90% |
| 50 | 0.22209 | 63.31% |
| 75 | 0.22133 | 63.61% |
| **100** | 0.22079 | 63.82% |
| **150** | **0.22073** | 63.73% |
| 200 | 0.22088 | 63.65% |
| 300 | 0.22085 | 63.59% |
| **Baseline (Elo overall, sin superficie)** | **0.22071** | **63.70%** |

**Conclusión honesta:** a umbrales bajos/moderados (5-50), separar el
Elo por superficie **empeora claramente** el Brier score (hasta 0.22442,
1.7% peor que el baseline) — la muestra por superficie es demasiado
ruidosa para mejorar la señal, especialmente en grass (solo 6,863
partidos reales en 12 años, contra 37,288 en hard). A umbrales altos
(100-300), el resultado converge asintóticamente al mismo baseline
(diferencias de ±0.00017 en Brier, dentro de ruido) — porque a esa
exigencia casi ningún jugador califica, así que el modelo termina
usando el Elo overall para casi todos de todas formas. **En ningún punto
del grid el Elo por superficie superó de forma clara y consistente al
Elo overall solo.**

Hipótesis de por qué: `SURFACE_ELO_FACTOR` (el ajuste genérico que ya
existía) probablemente ya captura la mayor parte de la señal útil de
"esta superficie favorece/perjudica a jugadores en general" sin
necesitar dividir el historial de cada jugador en 3-4 sub-muestras más
pequeñas y ruidosas. Dividir el Elo por superficie tiene sentido en
papers/modelos con décadas de datos por jugador top — con ~2467
jugadores y un historial real de 12 años, muchos todavía no acumulan
suficientes partidos por superficie para que la señal supere al ruido.

**Decisión, mismo criterio que shrink_elo:** `elegir_elo_superficie()`
queda implementada y testeada (`tests/test_tennis_surface_elo.py`, 7
casos) y disponible en el motor (parámetros `elo{1,2}_superficie`
opcionales, default `None` = sin cambios), pero **no se conecta a
`calibrate_tennis_elo.py` ni a `app.py`/`tennis_validator.py`** — no hay
evidencia de mejora real. `SURFACE_ELO_FACTOR` (el multiplicador
genérico) sigue siendo el único ajuste de superficie activo en
producción.

## Decay de Elo por inactividad — ACTIVADO en producción

Mismo baseline reverificado tras el fix de orden de rondas:
Brier 0.22060 / accuracy 63.89% (14,320 partidos de 2024-2026).

### El hallazgo inicial era falso — causado por el bug de orden

La primera corrida de este experimento (antes de encontrar el bug de
orden de rondas) mostraba una "mejora" que crecía sin límite: a
`decay_por_mes=1000` (básicamente forzando el Elo a 1500 para casi
cualquier jugador con más de un partido), el Brier score seguía
bajando y la accuracy subía a 67.4%. Esto era matemáticamente
imposible si el decay realmente estuviera destruyendo información real
— y efectivamente lo era: al procesar la Final de un torneo antes que
su Primera Ronda (el bug de orden), un jugador que seguía vivo en el
mismo torneo (misma `tourney_date`, meses_inactivo=0.0 exacto) quedaba
**exento del decay** y conservaba un Elo ya actualizado con el resultado
de una ronda posterior — el modelo tenía información del futuro del
torneo antes de "predecir" partidos anteriores del mismo torneo. No era
señal de inactividad, era fuga de información. Corregido el orden
(`_ORDEN_RONDA` en `tennis_data_loader.py`), el patrón cambió a la forma
de U invertida esperable de un decay real: mejora leve con poco decay,
empeora claramente con decay agresivo (`decay_por_mes=1000` da
brier=0.24353, peor que el baseline y cerca de un coin-flip).

### Grid con el orden corregido

| `decay_por_mes` | Brier | Accuracy |
|---|---|---|
| 0 (baseline) | 0.22060 | 63.89% |
| 0.05 | 0.22002 | 63.94% |
| 0.15 | 0.21943 | 63.94% |
| 0.20 | 0.21928 | 63.93% |
| **0.25** | **0.21921** | **63.97%** |
| 0.30 | 0.21923 | 63.95% |
| 0.40 | 0.21944 | 63.96% |
| 1.0 | 0.22218 | 63.24% |
| 10.0 | 0.23841 | 59.10% |
| 1000.0 | 0.24353 | 59.35% |

Curva suave y consistente alrededor de 0.15-0.4 (no un pico ruidoso
aislado) — a diferencia de shrink_elo (mejora de 0.12%, dentro de
ruido) y Elo por superficie (empeora o queda neutral), esta mejora es
real: 0.6% de Brier mejor que el baseline, con una forma de curva que
indica una señal genuina, no overfitting a un valor puntual. La
accuracy se mantiene casi plana (63.89% → 63.97%) — la mejora es sobre
todo de **calibración** (Brier/log-loss, penalizan predicciones
confiadas y equivocadas), no de más picks acertados.

### Decisión: activado en producción con `decay_por_mes=0.25`

A diferencia de shrink_elo y Elo por superficie, esta mejora cumple el
criterio de "mejora real, no ruido" — y además resuelve directamente el
caveat documentado de jugadores retirados con Elo congelado (Federer,
Barty), que fue la motivación original de todo este trabajo.

Cambios en producción:
- `calibrate_tennis_elo.py`: guarda `ultima_fecha` (fecha ISO del
  último partido) por jugador en `tennis_elo_ratings.json`.
- `tennis_validator.py`: calcula `meses_inactivo1`/`meses_inactivo2`
  (hoy − `ultima_fecha`, en meses de 30.44 días) al resolver Elo desde
  ratings calibrados. `None` si el Elo vino explícito en el request (no
  hay fecha de referencia) o si el jugador no tiene `ultima_fecha`.
- `app.py`: `DECAY_POR_MES_ACTIVO = 0.25`, pasado a `engine.analizar()`.

Validado en vivo: Roger Federer (última fecha registrada 2021-06-28,
~61 meses de inactividad) vs Jannik Sinner → `elo1_ajustado: 1500.0`
(Elo de Federer completamente decaído al prior). Novak Djokovic y
Sinner (gaps normales de semanas entre torneos) reciben un decay leve,
proporcional.

## Decay también aplicado a "forma" — RESUELTO

El caveat de arriba (forma vieja sin decay propio) ya está resuelto.

### Mecanismo: `forma_vigente()` (umbral, no decay continuo)

A diferencia del Elo (decay continuo hacia la media), la forma se
maneja con un corte simple: si `meses_inactivo` supera
`FORMA_MAX_MESES_INACTIVO=3.0`, la forma se descarta por completo
(se trata como si no existiera, cae al mismo fallback de Elo puro que ya
existía cuando faltaba forma real — P0-1). No tiene sentido "encoger"
gradualmente un win-rate de los últimos 10 partidos: pasados unos meses
sin jugar, esos 10 partidos ya no son "forma reciente" en absoluto, son
historia vieja — un corte es más honesto que un decay gradual acá.

Aplica en dos lugares: en el cálculo interno (`prob_match_winner_ensemble`)
y en el campo `forma` expuesto en la respuesta de `analizar()` — antes
del fix, `usa_forma` reflejaba correctamente el descarte pero el JSON de
respuesta seguía mostrando la forma vieja igual (inconsistencia real que
apareció al escribir el test end-to-end).

### Impacto en el backtest: neutral, como se esperaba

| | Brier | Accuracy |
|---|---|---|
| Solo decay de Elo (sin descartar forma) | 0.21921 | 63.97% |
| Elo + forma vieja descartada | 0.21932 | 63.97% |

Diferencia de 0.00011 (0.05% relativo) — dentro de ruido, ni mejora ni
empeora de forma medible. Esperable: son pocos los partidos evaluados
donde un jugador lleva más de 3 meses inactivo Y tiene forma calculada
de antes de ese parate (la mayoría de los jugadores con `forma` real
son justamente los que juegan seguido). El valor de este fix es de
**coherencia del modelo** (Elo y forma decaen juntos, no por separado),
no de precisión agregada — exactamente como se planteó el pedido.

### Activado en producción

`calibrate_tennis_elo.py`/`tennis_validator.py`/`app.py` ya pasaban
`meses_inactivo1`/`meses_inactivo2` al motor para el decay de Elo — el
descarte de forma se activó automáticamente al agregar la lógica al
motor, sin cambios adicionales de wiring.

Validado en vivo: Federer (inactivo ~61 meses) vs Sinner →
`"forma":{"j1":null,"j2":{...}}`, `"usa_forma":false`,
`"metodo":"Elo puro..."` — la forma vieja de Federer (7 ganados / 3
perdidos de 2021) ya no aparece ni se usa.
