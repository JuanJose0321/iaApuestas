# Backtest walk-forward — motor de tenis
Generado: 2026-08-09 · `backtest_tennis.py` · 14,320 partidos reales ATP+WTA 2024–2026

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
