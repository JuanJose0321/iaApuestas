# Reporte semanal de tenis -- 2026-09-01

Generado automáticamente por `src/scripts/reporte_semanal_tenis.py` a partir de `apuestas` (Supabase, `liga = "Tenis"`). Solo lectura -- no cambia nada del motor ni de la config de producción. Cualquier decisión sobre lo que sigue es manual.

## Panorama general

- Apuestas de tenis resueltas (ganada/perdida) a la fecha: **192**
- Nuevas resueltas desde el último reporte: **N/A (primer reporte, no hay corrida anterior para comparar)**
- Accuracy global acumulada: **55.73%** (107/192)

## Accuracy por badge

Rangos de referencia validados en `tennis_validacion_filtro_ev.md`: `verde` ~71%+, `amarillo` ~60-67%. Solo se alerta con `n >= 20` para no generar alarmas falsas con poca muestra.

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| amarillo | 20 | 11 | 55.00% |
| manual | 164 | 93 | 56.71% |
| verde | 8 | 3 | 37.50% |

Sin alertas de badge (todo dentro de rango, o sin muestra suficiente todavía).

## Accuracy por umbral de probabilidad

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| todos | 192 | 107 | 55.73% |
| prob>=55% | 125 | 75 | 60.00% |
| prob>=60% | 63 | 40 | 63.49% |
| prob>=65% | 37 | 25 | 67.57% |

## Chequeo del EV (solo seguimiento -- ya no decide qué se muestra)

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| ev_positivo | 91 | 40 | 43.96% |
| ev_negativo_o_cero | 101 | 67 | 66.34% |

## Tendencia reciente

Últimos 14 días (por `fecha_registro`) vs. acumulado histórico:

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| histórico (todo) | 192 | 107 | 55.73% |
| últimos 14 días | 104 | 55 | 52.88% |

## Chequeos de calidad de datos

- ⚠️ **8 grupo(s) de filas duplicadas exactas:**
  - Mees Rottgering vs Tomas Machac (Tomas Machac gana) -- 2 copias
  - Colton Smith vs Andrea Pellegrino (Andrea Pellegrino gana) -- 2 copias
  - Yeon Woo Ku vs Astra Sharma (Yeon Woo Ku gana) -- 2 copias
  - Yue Yuan vs Hanyu Guo (Yue Yuan gana) -- 2 copias
  - Bu Yunchaokete vs Rio Noguchi (Bu Yunchaokete gana) -- 3 copias
  - Luca Van Assche vs Aleksandar Kovacevic (Luca Van Assche gana) -- 2 copias
  - Storm Hunter vs Maddison Inglis (Storm Hunter gana) -- 2 copias
  - Katie Swan vs Katrina Scott (Katrina Scott gana) -- 2 copias
- Coherencia local/visitante vs. pick_descripcion: OK.
- Campos numéricos en filas resueltas: sin vacíos.

## Para decidir

**Esto podría valer la pena revisar con Juan:**
- 8 grupo(s) de duplicados exactos (ver Chequeos de calidad).
