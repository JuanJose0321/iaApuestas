# Reporte semanal de tenis -- 2026-09-14

Generado automáticamente por `src/scripts/reporte_semanal_tenis.py` a partir de `apuestas` (Supabase, `liga = "Tenis"`). Solo lectura -- no cambia nada del motor ni de la config de producción. Cualquier decisión sobre lo que sigue es manual.

## Panorama general

- Apuestas de tenis resueltas (ganada/perdida) a la fecha: **205**
- Nuevas resueltas desde el último reporte: **13 (de 192 a 205)**
- Accuracy global acumulada: **55.61%** (114/205)

## Accuracy por badge

Rangos de referencia validados en `tennis_validacion_filtro_ev.md`: `verde` ~71%+, `amarillo` ~60-67%. Solo se alerta con `n >= 20` para no generar alarmas falsas con poca muestra.

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| amarillo | 27 | 17 | 62.96% |
| manual | 169 | 93 | 55.03% |
| verde | 9 | 4 | 44.44% |

Sin alertas de badge (todo dentro de rango, o sin muestra suficiente todavía).

## Accuracy por umbral de probabilidad

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| todos | 205 | 114 | 55.61% |
| prob>=55% | 137 | 82 | 59.85% |
| prob>=60% | 71 | 47 | 66.20% |
| prob>=65% | 43 | 30 | 69.77% |

## Chequeo del EV (solo seguimiento -- ya no decide qué se muestra)

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| ev_positivo | 99 | 43 | 43.43% |
| ev_negativo_o_cero | 106 | 71 | 66.98% |

## Tendencia reciente

Últimos 14 días (por `fecha_registro`) vs. acumulado histórico:

| Categoría | n | Aciertos | Accuracy |
|---|---|---|---|
| histórico (todo) | 205 | 114 | 55.61% |
| últimos 14 días | 18 | 11 | 61.11% |

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
