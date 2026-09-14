# Reporte semanal de fútbol -- 2026-09-14

Generado automáticamente por `src/scripts/reporte_semanal_futbol.py` a partir de `apuestas` (Supabase, todo lo que no tiene `liga = "Tenis"`). Solo lectura -- no cambia nada de `src/engines/football.py` ni de la config de producción. Cualquier decisión sobre lo que sigue es manual.

## Panorama general

- Apuestas de fútbol resueltas (ganada/perdida) a la fecha: **2**
- Nuevas resueltas desde el último reporte: **N/A (primer reporte, no hay corrida anterior para comparar)**
- Accuracy global acumulada: **100.00%** (2/2)

## Accuracy por liga

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| Eredivisie | 2 | 2 | 100.00% |

## Accuracy por tipo de pick (directa / dupla / tripleta)

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| DIRECTA | 1 | 1 | 100.00% |
| DUPLA | 1 | 1 | 100.00% |

## Accuracy por badge de confianza

Sin rango "esperado" para alertar (a diferencia de tenis) -- `futbol_diagnostico_completo.md` encontró que el modelo no le gana al mercado, así que no hay un piso de accuracy validado como bueno. Esta tabla es solo descriptiva.

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| verde | 2 | 2 | 100.00% |

## Accuracy por umbral de probabilidad

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| todos | 2 | 2 | 100.00% |
| prob>=50% | 2 | 2 | 100.00% |
| prob>=65% | 1 | 1 | 100.00% |
| prob>=75% | 0 | 0 | 0.00% |

## Chequeo del EV (solo seguimiento -- ya no decide qué se muestra, ver commits fee1a01/5ddb82d)

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| ev_positivo | 2 | 2 | 100.00% |
| ev_negativo_o_cero | 0 | 0 | 0.00% |

## Modelo vs. probabilidad implícita de la cuota real

Referencia del backtest walk-forward (`futbol_diagnostico_completo.md`, n=7,008): modelo 53.94% / Brier 0.5827 -- mercado 54.14% / Brier 0.5767. El backtest concluyó que el modelo **no le gana al mercado**. Esta sección compara lo mismo, pero con apuestas reales de producción ya resueltas -- para ver si eso sigue siendo así.

- n con cuota y probabilidad válidas: **2**
- Accuracy modelo: **100.00%** -- Brier modelo: **0.1593**
- Accuracy implícita de la cuota (1/cuota, con vig): **50.00%** -- Brier: **0.2836**
- Accuracy cuando el modelo está MÁS confiado que la cuota implícita: 100.00% (n=2)
- Accuracy cuando el modelo está IGUAL o MENOS confiado: 0.00% (n=0)

## Tendencia reciente

Últimos 14 días (por `fecha_registro`) vs. acumulado histórico:

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| histórico (todo) | 2 | 2 | 100.00% |
| últimos 14 días | 0 | 0 | 0.00% |

## Chequeos de calidad de datos

- Duplicados exactos: ninguno.
- Campos numéricos en filas resueltas: sin vacíos.

## Para decidir

Nada para decidir todavía -- todo dentro de lo esperado y sin problemas de datos.
