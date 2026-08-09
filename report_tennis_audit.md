# AUDITORÍA COMPLETA — MÓDULO TENIS BETBRAIN
Fecha: 2026-08-08

> **Actualización 2026-08-08 (mismo día):** P0-1 y P0-2 implementados y
> validados (ver commit correspondiente). Resumen: el ensemble ya no
> diluye con una "forma" de 50% inventada (cae a Elo puro si no hay datos
> reales de ambos jugadores); los ratings se recalibraron con 14,320
> partidos reales ATP+WTA 2024–2026 (1001 jugadores, 646 con forma real de
> los últimos 10 partidos). La fuente original (JeffSackmann/tennis_atp y
> tennis_wta en GitHub) ya no existe — se usó el mirror activo
> `LuckyLoser91/TennisCourtLog`, con la salvedad de que su licencia no está
> declarada explícitamente en GitHub.
>
> **Actualización 2026-08-08 (P1):** `std_dev` de total de games también
> calibrado contra los 14,320 partidos reales (BO3: 5.91 vs 4.5
> hardcodeado; BO5: 9.39 vs 6.0) — ver `calibrate_tennis_std_dev.py`. Con
> el valor real, más ancho, el pick "Total Games Over 22.5" del caso
> Djokovic vs Sinner pasa de EV 46% → 36% (tras P0-1/P0-2) → 27% (tras
> P1), y ya no supera el umbral de confianza para mostrarse como pick —
> confirma que el EV original era un artefacto de una distribución
> artificialmente angosta, no una ineficiencia real del mercado. El resto
> de los hallazgos P1/P2 (Elo por superficie, H2H, tracking separado por
> deporte, etc.) permanece sin resolver.

## 1. ESTADO ACTUAL

### Arquitectura / flujo
- **Endpoint:** `POST /api/analizar_tenis` (`app.py:425-465`), rate limit 20/min.
- **Validación:** `src/engines/tennis_validator.py::validar_entrada_tenis` — valida jugadores, superficie (`clay/hard/grass/carpet`), formato (`best_of_3/best_of_5`), rango de Elo (500–3000) y estructura/márgenes de cada mercado de cuotas (match_winner, primer_set, set_handicap, total_games, game_handicap, sets_winners).
- **Motor:** `src/engines/tennis_improved.py::TennisImprovedEngine`, instanciado una vez (lazy singleton) en `app.py:84-107`.
- **Datos de Elo:** `src/data/tennis_elo_ratings.json` (39 jugadores calibrados).
- **Lista de jugadores para autocomplete:** `src/data/jugadores_por_genero.json` vía `src/providers/player_manager.py` (226 nombres: 111 ATP + 115 WTA), usada por `GET /api/players?genero=...`.
- **Frontend:** `templates/index.html` — tab "🎾 Tenis" con formulario (`#tennisFormCard`), autocomplete de jugador, selector de superficie/formato, tarjetas de pick verde/amarillo.

### Motor de Predicción
- **Tipo:** Elo logístico estándar + "Forma" reciente, combinados en ensemble fijo **70% Elo / 30% Forma**. **No usa XGBoost ni calibración isotónica** (eso solo existe en el motor de fútbol, `src/core/calibration.py`).
- **K-factor:** definido en `src/core/tennis_elo.py` (32/24/20/16/12/8 según nivel de torneo) — pero **solo se usa en el script offline `calibrate_tennis_elo.py`**, no en tiempo de request. El endpoint sirve Elo estático desde el JSON, no lo recalcula.
- **Ajuste por superficie:** multiplicador global fijo sobre el delta de Elo (`clay=1.20, hard=1.00, grass=0.85, carpet=0.90`) — **igual para todos los jugadores**, no hay Elo específico por superficie por jugador (no distingue a un especialista de clay de alguien mediocre en clay).
- **Umbrales de confianza:** `UMBRAL_VERDE=0.65`, `UMBRAL_AMARILLO=0.50` — **inconsistentes con fútbol**, que usa `UMBRAL_VERDE=0.75`, `UMBRAL_AMARILLO=0.65` (`src/core/confidence.py:10-11`). El tenis exige una barra más baja para "verde" que el fútbol, pese a tener un modelo objetivamente más simple y sin backtesting conocido.
- **Distribución de total de games:** Normal con `std_dev` **constante** por formato (4.5 para BO3, 6.0 para BO5) — no depende de los jugadores ni de la superficie, solo del formato.

### Datos disponibles
- **Jugadores con Elo calibrado:** 39 (mezcla ATP+WTA sin campo de género en el JSON).
- **Jugadores en autocomplete (sin Elo calibrado propio):** 226 (111 ATP + 115 WTA) — los ~187 que no están en los 39 calibrados caen al **default fijo 1500** en `validar_entrada_tenis` (`tennis_validator.py:48-50`), es decir, la mayoría de los partidos posibles vía el formulario usan un Elo inventado idéntico para ambos jugadores si ninguno está en la lista corta.
- **Última calibración:** `2026-05-08` (**3 meses desactualizado** respecto a hoy, 2026-08-08).
- **Fuente de los "500 matches_procesados" que generaron el Elo actual:** ⚠️ **son partidos sintéticos/aleatorios**, no históricos reales. `generar_datos_tenis_sinteticos.py` genera resultados con `random.random() < prob_p1` a partir de la posición en una lista fija de 20 jugadores por género (no rankings ATP/WTA reales, no resultados reales) y los guarda en `src/data/tennis/matches_atp_sintetic.csv`. `calibrate_tennis_elo.py` (el script "real", que sí descarga de GitHub/Jeff Sackmann vía `tennis_data_loader.py`) no dejó rastro de haberse ejecutado con éxito — `src/data/tennis/` no existe en el repo, solo el output sintético parece haber alimentado el JSON de producción.
- **Rango de Elo:** 1338.9 (Madison Keys) – 1681.6 (Jannik Sinner). Rango angosto (~343 puntos) típico de una calibración con pocos partidos, no de un sistema con historial profundo (el Elo ATP real de Sackmann typically spans 1300-2400+).
- **Features por jugador:** solo `elo` y `games` (nº de partidos usados en la calibración). No hay ranking oficial, edad, mano dominante, superficie preferida, lesiones, ni fecha de último partido.

### Performance (medido en esta sesión)
- **Tests:** 99/99 passing en la suite completa (`pytest tests/`) — pero **0 de esos 99 tests tocan el módulo de tenis**. No existe ningún archivo `tests/test_tennis*.py`. `grep -k tennis` devuelve 0 tests seleccionados.
- **Endpoint response time:** ~220ms (`curl` con `time`, servidor Flask dev local) — rápido, sin llamadas a red externas.
- **Error rate:** 0% en los casos probados; el validador rechaza correctamente cuotas con margen fuera de rango (probado: `{"1":1.5,"2":1.5}` → error 400 "Margen de match_winner fuera de rango (1.333)").
- **`TODO`/`FIXME`/`BUG` en archivos de tenis:** ninguno encontrado (el código está "limpio" superficialmente, pero eso es porque los bugs reales no están marcados, están silenciosos — ver Gaps Críticos).

## 2. FORTALEZAS ACTUALES

1. **Validación de entrada robusta**: `tennis_validator.py` chequea rangos de cuota, márgenes de casa de apuestas (overround) por mercado, líneas válidas por formato (BO3 vs BO5) — mejor cubierto que muchos endpoints similares.
2. **Cobertura amplia de mercados**: match winner, primer set, hándicap de sets, total de games, hándicap de games, ganador por set — la validación ya soporta 6 mercados aunque el motor (`analizar()`) hoy solo genera picks para 2 (match winner y total games).
3. **Gestión de riesgo consistente con fútbol**: mismo esquema Kelly fraccionado (25%) y estructura de picks verde/amarillo/rojo, lo que mantiene coherencia de UX entre deportes.
4. **Infraestructura de calibración ya escrita**: `TennisEloCalculator` con K-factor adaptativo por nivel de torneo y `tennis_data_loader.py` apuntando a la fuente correcta (Jeff Sackmann GitHub, gratuita y confiable) — el diseño es razonable, solo falta ejecutarlo con datos reales y automatizarlo.
5. **Endpoint rápido y sin dependencias externas en el hot path** (a diferencia de fútbol, que encadena llamadas a APIs externas) — ideal para el límite de 10s de Vercel Hobby mencionado en el README.

## 3. LIMITACIONES & GAPS

### Críticos (rompen picks o causan falsos positivos)

- [ ] **`form_stats` está hardcodeado vacío en producción** (`app.py:92`: `form_stats = {}`, nunca se puebla). El "30% Forma" del ensemble "Elo (70%) + Forma (30%)" **siempre evalúa a exactamente 0.5** para ambos jugadores (verificado en vivo: Djokovic vs Sinner devolvió `"p_form":0.5` con `"forma":{"ganados":0,"perdidos":0}`). Esto no es "falta de una feature" — es una feature que el código anuncia como activa (`"metodo": "Ensemble Elo (70%) + Forma (30%)"` en cada response) y que en realidad **diluye el Elo real con una constante sin información**, empeorando la calibración respecto a usar Elo puro al 100%. Es engañoso para cualquiera que lea el output creyendo que el modelo usa forma reciente.
- [ ] **Los ratings de Elo en producción se calibraron con partidos sintéticos, no reales.** `generar_datos_tenis_sinteticos.py` produce resultados con `random.random()` ponderado por posición en una lista fija de 20 nombres por género — no son resultados históricos. Si el JSON actual (`_meta.matches_procesados: 500`, fecha `2026-05-08`) vino de ese generador y no de `calibrate_tennis_elo.py` con datos reales de Sackmann (no hay evidencia de que ese script se haya corrido con éxito — `src/data/tennis/*.csv` no existe en el repo), **todos los Elo actuales son esencialmente ruido con un sesgo hacia el ranking "de memoria" que el desarrollador tipeó a mano**, no datos.
- [ ] **`std_dev` de total de games es una constante fija por formato (4.5 / 6.0), no derivada de los jugadores ni la superficie.** Combinado con el punto anterior, esto genera picks de alta confianza artificialmente: en la prueba en vivo (Djokovic vs Sinner, hard, BO3) el motor devolvió un pick "Over 22.5 games" con **EV = 45.98%** y confianza "verde" (0.687) a una cuota de 1.90 — un EV de esa magnitud en un mercado de total de games de tenis profesional es sistemáticamente inverosímil y apunta a mala calibración del modelo de distribución, no a una ineficiencia real del mercado.
- [ ] **187 de los 226 jugadores del autocomplete no tienen Elo calibrado propio** y caen al default `1500` para ambos si ninguno está en la lista de 39 — el motor entonces predice un partido 50/50 "informado" cuando en realidad no tiene ninguna señal, y ese 50/50 puede seguir generando picks de "valor" en mercados de totales por la razón del punto anterior.

### Importantes (afectan precisión)

- [ ] **Sin H2H (head-to-head)**, ni general ni específico por superficie. El validador ni siquiera acepta un campo de H2H en el request.
- [ ] **Sin ajuste por superficie específico del jugador** — el `SURFACE_ELO_FACTOR` es un multiplicador global aplicado a todos los jugadores por igual; no diferencia a un especialista de clay de alguien flojo en clay.
- [ ] **Sin recency real** — el campo "forma" existe en el modelo de datos pero, aparte de estar vacío en producción (ver crítico arriba), tampoco hay ponderación tipo "últimos 5 pesan más que los 5 anteriores"; es un ratio simple ganados/perdidos.
- [ ] **Sin contexto de torneo en tiempo de predicción** — el K-factor por nivel de torneo (`Grand Slam` vs `ATP 250`) solo se aplica en la calibración offline, no afecta la predicción del partido actual (un Grand Slam y un Challenger se tratan igual en `analizar()`).
- [ ] **Sin lesiones/retiros/fatiga/back-to-back** — no hay ningún campo para esto ni en el modelo ni en el request.
- [ ] **Sin edad ni mano dominante** como factor.
- [ ] **Umbrales de confianza inconsistentes entre deportes** (tenis: 0.65/0.50 vs fútbol: 0.75/0.65) sin justificación documentada — si es intencional (tenis tiene menos varianza esperada) debería estar explicado; si no, es una inconsistencia de producto que hace que un "verde" en tenis no signifique lo mismo que un "verde" en fútbol.
- [ ] **Sin tracking/backtesting específico de tenis** — `src/services/tracking.py` registra picks con `liga:"Tenis"` genérico (visto en el frontend, `pick_tipo:"TENIS"`), pero no hay ninguna métrica separada de win rate o ROI por deporte; todo se mezcla en `/api/metricas`.

### Nice-to-have

- [ ] Predicción de sets individuales / hándicap de sets (el validador ya acepta el mercado, el motor no lo resuelve).
- [ ] Intervalos de confianza explícitos por pick (hoy solo hay un score 0-1).
- [ ] Filtro de ranking/seed en el autocomplete del frontend.
- [ ] Mostrar fecha de última actualización de Elo en el frontend (hoy no se expone en el JSON de respuesta ni en la UI).
- [ ] Actualización automática de ratings (equivalente al workflow de GitHub Actions que ya existe para los CSV de fútbol, según el commit reciente `875f0e6`).

## 4. RECOMENDACIONES PRIORIZADAS

### PRIORIDAD 0 — IMPLEMENTAR YA (1-2 horas)

1. **Eliminar el componente "Forma" fantasma o implementarlo de verdad**
   - Qué: mientras `form_stats` esté vacío, `prob_from_form` siempre retorna 0.5 y diluye el Elo real en 30%. Solución mínima: si no hay datos de forma reales, usar **100% Elo** (no un ensemble falso), y sólo activar el ensemble cuando `form_stats` tenga datos reales por jugador.
   - Impacto: elimina un sesgo sistemático hacia 0.5 en cada predicción — hoy cada probabilidad final está "encogida" ~30% hacia el empate sin ninguna razón estadística.
   - Esfuerzo: 30 min.
   - Ejemplo: en `TennisImprovedEngine.prob_match_winner_ensemble`, chequear `if player1 in self.form_stats and player2 in self.form_stats: usar ensemble; else: p_j1 = p_elo`.

2. **Corregir el string `"metodo"` en la respuesta hasta que la forma esté implementada de verdad**
   - Qué: no anunciar `"Ensemble Elo (70%) + Forma (30%)"` cuando la forma es constante — es información falsa hacia cualquier consumidor del JSON (frontend, logs, tú mismo revisando picks).
   - Impacto: evita confianza injustificada en la explicación del modelo.
   - Esfuerzo: 5 min (parte del fix anterior).

3. **Auditar el origen real de `tennis_elo_ratings.json`**
   - Qué: correr `calibrate_tennis_elo.py` de punta a punta y confirmar que `descargar_datos_tennis()` efectivamente trae CSVs reales de Sackmann (revisar `src/data/tennis/` después de correrlo). Si falla la descarga (posible por certificados/proxy/rate limit de GitHub raw), documentarlo y no dejar que el fallback caiga silenciosamente en datos sintéticos.
   - Impacto: es la base de todo el motor — si el Elo es ruido, ninguna otra mejora importa.
   - Esfuerzo: 1 hora (incluye verificar manualmente que el CSV descargado tiene partidos reales de 2024-2026 y no está vacío).

4. **Marcar/loguear explícitamente cuando un jugador usa Elo default (1500)**
   - Qué: en `validar_entrada_tenis`, cuando ninguno de los dos Elo viene del JSON calibrado, agregar un flag `"elo_estimado": true` en la respuesta o rechazar/advertir en vez de generar picks "verdes" con datos inventados.
   - Impacto: evita picks de alta confianza basados en 1500 vs 1500 + variación aleatoria de forma.
   - Esfuerzo: 30 min.

### PRIORIDAD 1 — PRÓXIMA SEMANA (3-4 horas)

1. **Calibrar `std_dev` de total de games con datos reales** en vez de la constante 4.5/6.0 — calcular desviación estándar empírica de totales de games por superficie/formato a partir del histórico real de Sackmann una vez esté disponible (P0-3). Esto es lo que está generando el EV de 46% inverosímil visto en la prueba en vivo.
   - Esfuerzo: 2 horas (incluye script de análisis exploratorio sobre el CSV histórico).

2. **Elo específico por superficie** (o al menos un ajuste por jugador, no solo un multiplicador global) — mantener 3 Elos por jugador (clay/hard/grass) actualizados independientemente al procesar el histórico, en vez de un único Elo + factor genérico de superficie.
   - Esfuerzo: 1.5 horas (extender `TennisEloCalculator` + regenerar el JSON con 3 ratings por jugador).

3. **Escribir tests para el módulo de tenis** — `tests/test_tennis_engine.py` (unit tests de `TennisImprovedEngine`: Elo puro, ensemble, evaluar_value, niveles de confianza) y `tests/test_tennis_endpoint.py` (integración contra `/api/analizar_tenis`, incluyendo el caso de jugador sin Elo calibrado y el caso de margen de cuota inválido). Hoy: 0 de 99 tests cubren este módulo.
   - Esfuerzo: 1.5 horas.

4. **Unificar umbrales de confianza entre fútbol y tenis** (o documentar explícitamente por qué difieren) en un único lugar (`src/core/confidence.py`), en vez de que `tennis_improved.py` tenga su propia copia hardcodeada de `UMBRAL_VERDE`/`UMBRAL_AMARILLO`.
   - Esfuerzo: 30 min.

### PRIORIDAD 2 — FUTURO (5+ horas)

1. **H2H específico por superficie** — requiere guardar el histórico de enfrentamientos directos (no solo el resultado agregado de Elo) y exponerlo como input adicional al ensemble.
2. **Automatizar actualización diaria/semanal de Elo** vía GitHub Actions (mismo patrón que el workflow de fútbol del commit `875f0e6`), corriendo `calibrate_tennis_elo.py` contra los CSV más recientes de Sackmann.
3. **Resolver picks para los mercados ya validados pero no implementados** (primer set, hándicap de sets, hándicap de games, ganador por set) — el validador ya los acepta, `analizar()` los ignora.
4. **Tracking y métricas separadas por deporte** en `/api/metricas` (win rate, ROI, calibración real vs. predicha) para poder medir si el modelo de tenis es rentable, que es literalmente el propósito declarado del proyecto entero (ver README).
5. **Contexto de torneo en tiempo real** (Grand Slam vs 250) como input al modelo de predicción, no solo a la calibración offline.

## 5. ROADMAP DE IMPLEMENTACIÓN

**Semana 1 (P0):**
- [ ] Desactivar/arreglar el componente Forma fantasma (100% Elo mientras no haya datos reales)
- [ ] Corregir el string "metodo" en la respuesta
- [ ] Correr y verificar `calibrate_tennis_elo.py` con datos reales de Sackmann
- [ ] Flag de "elo_estimado" para jugadores sin calibración propia

**Semana 2 (P1):**
- [ ] Calibrar `std_dev` de total de games con datos reales
- [ ] Elo por superficie
- [ ] Tests de tenis (unit + integración)
- [ ] Unificar umbrales de confianza

**Futuro (P2):**
- [ ] H2H por superficie
- [ ] Automatización de actualización de Elo (GitHub Actions)
- [ ] Resolver mercados pendientes (primer set, hándicaps, ganador por set)
- [ ] Métricas de tenis separadas en `/api/metricas`
- [ ] Contexto de torneo en tiempo de predicción

## 6. MÉTRICAS SUGERIDAS

No hay datos históricos de picks de tenis registrados de forma separable hoy (todo se mezcla bajo `liga:"Tenis"` genérico en el tracking), así que **no se puede calcular accuracy/precision/ROI real actual** — esta es en sí misma una brecha (ver P2-4). Antes de optimizar el modelo más, conviene:
1. Implementar el fix P0-1 (eliminar la forma fantasma).
2. Dejar correr el sistema en modo "solo tracking" un tiempo con Elo real (post P0-3).
3. Medir accuracy/calibración real contra resultados antes de agregar más features — agregar H2H o fatiga sobre una base de Elo sintético (el problema P0-2) sería construir sobre arena.

## 7. SCRIPTS PENDIENTES

- [ ] **Actualización periódica de ratings**: existe `calibrate_tennis_elo.py` pero no está automatizado (no hay `.github/workflows/tennis-elo.yml` equivalente al de fútbol).
- [ ] **Tracking de picks separado por deporte**: falta filtrar/agrupar por `liga:"Tenis"` en `calcular_metricas()` (`src/services/tracking.py`) para poder responder "¿el modelo de tenis es rentable?", que es la pregunta que el README dice que el proyecto entero busca responder.
- [ ] **Análisis de calibración**: no existe ningún script que compare `prob_predicha` vs resultado real para tenis (si existe para fútbol, replicar el patrón).

## 8. PRÓXIMOS PASOS

1. Empezar por **P0-1** (eliminar el ensemble de Forma fantasma) — es el cambio de menor esfuerzo y mayor impacto porque hoy está silenciosamente sesgando cada predicción del modelo.
2. Ejecutar desde terminal:
   ```bash
   python calibrate_tennis_elo.py
   ```
   y verificar en el log que descarga CSVs reales (no debe decir "matches_procesados: 500" otra vez si la descarga real trae miles de partidos de 3 años de ATP+WTA).
3. Testear en: `http://localhost:5000/api/analizar_tenis` con el jugador Djokovic vs Sinner (hard, BO3) y confirmar que `"p_form"` deja de aparecer fijo en 0.5 y que el EV de "Total Games Over" baja a un rango creíble (<15%, no 46%).

> **Actualización 2026-08-09:** P0-1 (backtesting walk-forward) y P0-2
> (regresión a la media) de esta sección implementados — ver
> `tennis_backtest_results.md`. Resumen: el modelo actual sí predice
> mejor que el azar (Brier 0.226 vs 0.25 coin-flip), pero el ranking
> oficial ATP/WTA solo (63.45% accuracy) le gana al modelo (61.03%) —
> hallazgo honesto, no esperado. La regresión a la media (`shrink_elo()`)
> se implementó y testeó pero el backtest mostró una mejora marginal
> (~0.12% relativo en Brier, empeora accuracy) — **no se activó en
> producción** siguiendo la instrucción explícita de solo hacerlo si el
> backtest confirmaba mejora real. Queda disponible para reintentar con
> más años de histórico.
>
> **Actualización 2026-08-09 (fix del cold-start):** confirmado que
> `INITIAL_ELO=1500` uniforme + solo 3 años de histórico era la causa de
> que el ranking oficial le ganara al modelo. Fix elegido: burn-in con
> 12 años de historial real (2015-2026, 62,128 partidos, la misma fuente
> ya tenía datos hasta 1968) en vez de una fórmula rank→Elo inventada o
> agregar el ranking como feature en vivo. Resultado: accuracy 63.70% —
> supera el 63.45% del ranking oficial. `tennis_elo_ratings.json` ya
> regenerado y en producción (2467 jugadores, antes 1001). Ver detalle
> completo en `tennis_backtest_results.md`.
>
> **Actualización 2026-08-09 (P1, Elo por superficie — NO activado):**
> implementado y probado contra el mismo baseline (0.22071 Brier /
> 63.70% accuracy). Resultado: en todo el grid de umbrales probados
> (5-300 partidos mínimos por superficie), nunca superó de forma clara
> al Elo overall — a umbrales bajos empeora (hasta 1.7% peor Brier), a
> umbrales altos converge al mismo resultado sin mejorarlo. Mismo
> criterio que `shrink_elo()`: queda implementado y testeado
> (`elegir_elo_superficie()`, `tests/test_tennis_surface_elo.py`), pero
> **no se activa en producción** — sigue en pie solo `SURFACE_ELO_FACTOR`
> (el multiplicador genérico). Detalle en `tennis_backtest_results.md`.
>
> **Actualización 2026-08-09 (decay por inactividad — SÍ ACTIVADO):**
> resuelve el caveat de jugadores retirados (Federer/Barty) con Elo
> congelado. Bug importante encontrado en el camino: `combinar_archivos()`
> ordenaba solo por fecha, pero el CSV comparte una única fecha (inicio
> del torneo) entre todas sus rondas, listadas en orden descendente
> (Final primero) — walk-forward roto, con fuga de información hacia el
> pasado. Corregido (`_ORDEN_RONDA` en `tennis_data_loader.py`); se
> reverificaron burn-in/shrink/superficie contra el fix y sus
> conclusiones no cambiaron. Con el orden correcto, el decay muestra una
> mejora real y consistente (curva suave, no un pico aislado): Brier
> 0.6% mejor (0.22060 → 0.21921) con `decay_por_mes=0.25`, accuracy
> prácticamente igual. Activado en producción: `calibrate_tennis_elo.py`
> guarda `ultima_fecha` por jugador, `tennis_validator.py` calcula
> meses de inactividad, `app.py` usa `DECAY_POR_MES_ACTIVO=0.25`.
> Validado en vivo: Federer (inactivo desde 2021) cae a Elo 1500 puro.
> Detalle completo en `tennis_backtest_results.md`.
>
> **Actualización 2026-08-09 (forma también decae — resuelto):** el
> caveat pendiente ("forma vieja de Federer seguía activa aunque su Elo
> ya decayó") quedó resuelto — `forma_vigente()` descarta la forma
> completa (corte simple, no decay gradual) cuando `meses_inactivo`
> supera 3 meses, reusando el mismo fallback de Elo puro de P0-1.
> Impacto en el backtest: neutral (Brier 0.21921→0.21932, dentro de
> ruido) — es un fix de coherencia del modelo, no de precisión agregada.
> Validado en vivo: Federer ya no muestra `forma` en la respuesta.
>
> **Actualización 2026-08-09 (H2H — SÍ ACTIVADO, última pieza de esta
> sección):** grid de peso (0.10-0.30) × enfrentamientos previos mínimos
> (2/3/5) sobre el baseline con decay ya activo (Brier 0.21921 / accuracy
> 63.97%). Mejor punto: `min_partidos=2, weight=0.18` → Brier 0.21895
> (+0.12%), accuracy 64.21% (+0.24pp) — curva suave confirmada con grid
> fino, no un pico aislado. A diferencia de shrink_elo (Brier similar
> pero accuracy siempre peor), H2H mejora ambas métricas de forma
> consistente. Activado: `calibrate_tennis_elo.py` exporta
> `tennis_h2h.json` (38,095 pares), `tennis_validator.py` lo consulta,
> `app.py` usa `H2H_WEIGHT_ACTIVO=0.18`. Con esto quedan resueltas las 4
> mejoras P0/P1 de esta sección (backtesting, regresión a la media,
> decay de inactividad, H2H) — Elo por superficie fue la única que no
> se activó por falta de evidencia. Detalle en `tennis_backtest_results.md`.

## 9. AUDITORÍA DE PRECISIÓN — FASE 2 (2026-08-09)

Diagnóstico solicitado explícitamente sin implementación. Todo lo citado
abajo está verificado contra el código actual (`src/engines/tennis_improved.py`,
`src/data/tennis_elo_ratings.json`, `src/data/tennis_std_dev_calibrated.json`,
`src/core/backtest.py`, `src/core/model.py`) en el momento de esta auditoría.

### Corrección al contexto de partida

El pedido asumía que el motor ya tiene "Elo específico por superficie
(hard/clay/grass/carpet)". **No es así** — verificado en
`tennis_elo_ratings.json`: cada jugador tiene un único campo `elo` global
(ej. `{"elo": 1616.2, "games": 113, "forma": {...}}`). Lo que existe es
`SURFACE_ELO_FACTOR` (`tennis_improved.py:28-33`), un multiplicador
**global e idéntico para cualquier jugador** (clay×1.20, hard×1.00,
grass×0.85, carpet×0.90) aplicado sobre la diferencia de Elo — no es Elo
por superficie, es un ajuste genérico de superficie. Sigue siendo
exactamente el mismo gap ya identificado en la Fase 1 de esta auditoría,
sin implementar todavía.

### 9.1 Qué SÍ considera el modelo hoy (verificado línea por línea)

| Feature | Dónde | Detalle |
|---|---|---|
| Elo dinámico global | `tennis_elo_ratings.json` | 1001 jugadores, calibrado con 14,320 partidos reales 2024-2026 |
| K-factor por nivel de torneo | `src/core/tennis_elo.py:15-22` | Grand Slam=32, Masters 1000=24, ATP 500=20, ATP 250=16, Challenger=12, ITF=8 — **solo se usa en la calibración offline** (`calibrate_tennis_elo.py`), no afecta la predicción de un partido puntual |
| Ajuste genérico de superficie | `tennis_improved.py:28-33,96-101` | Multiplicador fijo sobre el delta de Elo, igual para todos los jugadores (no es Elo por superficie) |
| Forma reciente real | `tennis_elo_ratings.json` campo `forma` | Últimos 10 partidos, calculado en la misma pasada cronológica que el Elo (`calibrate_tennis_elo.py`), mínimo 3 partidos para reportarse — 646/1001 jugadores la tienen |
| Ensemble Elo+Forma condicional | `tennis_improved.py:120-163` | 70% Elo + 30% Forma solo si AMBOS jugadores tienen forma real; si no, 100% Elo (fix P0-1) |
| std_dev de Total Games calibrado | `tennis_std_dev_calibrated.json` | Desviación muestral real por formato (BO3=5.91, BO5=9.39), separado por la columna `best_of` real, excluyendo partidos incompletos (fix P1) |
| Kelly fraccionado (25%) | `tennis_improved.py:198-217,337-345` | Igual que fútbol |

### 9.2 Gaps identificados

#### Crítico — bloquea saber si el modelo sirve

**No existe backtesting ni calibración estadística para tenis, en absoluto.**
Verificado: `grep -rl "brier\|log_loss\|backtest"` sobre todo el proyecto
solo encuentra `src/core/backtest.py`, `src/core/model.py` y `src/cli.py` —
los tres son 100% del motor de **fútbol** (XGBoost + Poisson). `model.py`
sí calcula `log_loss` (líneas 56, 142-152), pero comparando el modelo de
fútbol calibrado vs. sin calibrar vs. el mercado — cero relación con tenis.
`tracking.py::calcular_metricas()` sí calcula `calibracion` (prob promedio
vs. tasa de acierto real), pero mezcla fútbol y tenis en el mismo pool sin
separar por deporte (gap ya señalado en la Fase 1, sección 3).

**Consecuencia concreta:** ni vos ni yo sabemos hoy si el Elo+superficie+forma
actual predice mejor que Elo puro, que un coin-flip, o que las cuotas del
mercado. Todos los ajustes hechos hasta ahora (P0-1, P0-2, P1) son
correcciones de bugs obvios (sesgo hacia 0.5, datos sintéticos, std_dev
demasiado angosto) — mejoras seguras por construcción — pero nadie midió
si el modelo resultante tiene Brier score / log-loss mejor que la cuota
implícita del mercado. Sin esto, cualquier feature nueva que se agregue
de acá en más se está afinando a ciegas.

#### Importantes

- **Sin regresión a la media para muestra pequeña.** El campo `games` ya
  existe en `tennis_elo_ratings.json` (partidos usados en la calibración
  de cada jugador) pero **no se usa en ningún lado del motor** — un
  jugador con `games: 3` se trata con la misma confianza que uno con
  `games: 500`, pese a que su Elo es mucho más ruidoso. `_calc_confianza()`
  (`tennis_improved.py:219-225`) no tiene ningún input de tamaño de
  muestra.
- **H2H (head-to-head) no se persiste.** `combinar_archivos()` (loader)
  sí trae el historial completo de partidos con nombres de ganador/perdedor,
  pero `calibrate_tennis_elo.py` lo descarta después de actualizar el Elo
  — no queda ningún índice de enfrentamientos directos por par de
  jugadores. Los datos ya están disponibles (no requiere fuente nueva),
  solo no se guardan.
- **Sin actualización automática de Elo.** A diferencia de fútbol, que ya
  tiene un GitHub Actions diario (`.github/workflows/...`, commit
  `875f0e6`) para refrescar los CSV, tenis no tiene ningún equivalente —
  `calibrate_tennis_elo.py` hay que correrlo a mano. El Elo se desactualiza
  cada semana que pasa sin recalibrar.
- **Calibración isotónica/Platt de las probabilidades del Elo:** técnicamente
  correcto pedirlo, pero es un subproducto directo del backtesting (punto
  crítico de arriba) — no se puede calibrar sin primero tener pares
  (prob predicha, resultado real) del histórico, que es exactamente lo que
  falta. Es el mismo trabajo, no un ítem aparte.

#### Necesitan fuente de datos nueva (no la tenemos hoy, ni en CSV ni en ningún otro lado del proyecto)

- **Fatiga / días desde el último partido:** los CSV históricos sí tienen
  fecha, pero el endpoint `POST /api/analizar_tenis` no recibe fecha del
  partido a predecir ni torneo/ronda — haría falta cambiar el schema del
  request y tener un calendario de torneos en vivo.
- **Rendimiento en el torneo actual (sets ganados en rondas previas):**
  mismo problema — requiere saber en qué torneo/ronda está el partido a
  predecir, dato que hoy no se pide ni se tiene.
- **Lesiones/retiros recientes:** el histórico solo tiene ganador/perdedor
  y score — ningún reporte de estado físico. Requiere una fuente externa
  nueva (no existe ningún candidato evaluado todavía).
- **Altitud/condiciones del torneo:** no hay metadata de ciudad/altitud en
  ningún archivo del proyecto. Impacto probablemente bajo salvo casos
  puntuales (Madrid, Bogotá) — no prioritario aun si se consiguiera la
  fuente.
- **Estilo de juego / matchup:** no hay ningún dato de estilo en el
  proyecto. Además de requerir fuente nueva, es un dato difícil de
  cuantificar sistemáticamente sin un dataset especializado — impacto
  incierto.
- **Momentum en vivo / in-play:** el motor es 100% pre-partido, sin
  ningún input de estado del partido en curso. Implementarlo sería
  esencialmente un producto distinto (modelo in-play), no una mejora
  incremental — requiere datos de partido en vivo que no están en el
  alcance actual del proyecto.

### 9.3 Reporte priorizado

#### P0 — hacer antes que cualquier otra cosa

1. **Backtesting walk-forward + Brier score / log-loss para tenis**
   - Qué resuelve: hoy no hay forma de saber si el modelo predice mejor
     que el mercado o que un coin-flip.
   - Impacto: crítico — es la precondición para que cualquier otra mejora
     tenga sentido medirla.
   - Esfuerzo: 3-4h. Requiere reproducir el Elo **paso a paso** guardando
     el Elo de cada jugador *antes* de cada partido (no solo el Elo final
     que hoy se guarda) para poder generar la predicción "honesta" que el
     modelo hubiera dado en ese momento, y compararla contra el resultado
     real con Brier score / log-loss, contra la cuota implícita del
     mercado si se consigue histórico de cuotas (no lo tenemos — sin eso,
     al menos comparar contra un baseline de Elo puro vs. Bookmaker
     consensus no es posible, pero sí contra un baseline naive 50/50 y
     contra ranking ATP/WTA oficial si está en los CSV).
   - Datos: ya los tenemos (los mismos 14,320 partidos).

2. **Regresión a la media por tamaño de muestra**
   - Qué resuelve: Elo con pocos partidos (`games` bajo) es ruidoso pero
     se usa con la misma confianza que uno con historial largo.
   - Impacto: medio-alto, esfuerzo bajo.
   - Esfuerzo: 1h. Ej.: encoger el Elo hacia 1500 proporcional a
     `1/sqrt(games)`, o ensanchar el `std_dev` efectivo de la predicción
     cuando `games` es bajo.
   - Datos: ya los tenemos (`games` ya está en el JSON).

#### P1 — con impacto real, esfuerzo medio, datos ya disponibles

3. **Elo por superficie por jugador** (el gap que el pedido asumía ya resuelto)
   - Impacto: medio-alto — la intuición de que el ranking cambia fuerte
     por superficie es real en tenis, y ya tenemos `surface` en los CSV
     históricos.
   - Esfuerzo: 2h — extender `TennisEloCalculator` para trackear 3 Elos
     por jugador (o un Elo overall + delta por superficie) y regenerar
     `tennis_elo_ratings.json`.
   - Bloqueado por: nada, se puede hacer ya. Pero su impacto real solo se
     puede confirmar después de P0-1 (backtesting).

4. **H2H (head-to-head)**
   - Impacto: medio — en la literatura de tenis moderna el Elo ya captura
     la mayoría de la señal de H2H; el aporte incremental es real pero
     modesto salvo matchups de estilo muy marcados.
   - Esfuerzo: 2-3h — nueva estructura para persistir enfrentamientos
     directos por par de jugadores durante la misma pasada de calibración.
   - Datos: ya los tenemos, solo no se guardan hoy.

5. **Automatizar recalibración de Elo (GitHub Actions)**
   - Impacto: alto en el tiempo (mitiga que el Elo se desactualice cada
     semana), esfuerzo bajo-medio.
   - Esfuerzo: 2h — replicar el patrón que ya existe para fútbol
     (commit `875f0e6`).

#### P2 — impacto incierto o requiere fuente de datos nueva

6. Calibración isotónica de probabilidades — mismo trabajo que P0-1, no
   es un ítem separado.
7. Fatiga/descanso — requiere calendario de torneos en vivo + cambiar el
   schema del request.
8. Rendimiento en torneo actual — mismo bloqueo que el anterior.
9. Lesiones/retiros — necesita fuente de datos nueva, no evaluada.
10. Altitud/condiciones — necesita fuente de datos nueva, impacto bajo.
11. Estilo de juego/matchup — necesita fuente de datos nueva, impacto incierto.
12. Momentum in-play — cambio de alcance de producto, no una mejora incremental.

### 9.4 Recomendación

Implementar **P0-1 (backtesting) antes que cualquier feature nueva**,
incluido el Elo por superficie que parecía la mejora "obvia" siguiente.
Sin un número de referencia (Brier/log-loss actual), no hay forma de
confirmar si agregar superficie por jugador o H2H realmente mejora algo,
o si solo agrega ruido/overfitting a un histórico de apenas 2 años. Una
vez que exista ese baseline, P0-2 (regresión a la media) y P1-3 (Elo por
superficie) son los siguientes candidatos naturales porque no requieren
datos nuevos y su efecto se puede medir contra el mismo baseline.

## 10. FIX: Selector de jugadores desincronizado del Elo real (2026-08-09)

No relacionado a la precisión del modelo (secciones 1-9) — un bug de
sincronización de datos encontrado por el usuario: el selector de
jugadores de la app (`GET /api/players`) nunca usó
`tennis_elo_ratings.json`. Usaba `src/data/jugadores_por_genero.json`,
una lista **curada a mano** de 226 jugadores ("actualizado mayo 2026"
según su propio `_meta`) que nunca se sincronizó con el Elo real
calibrado. Tras el burn-in de 12 años (sección 9), 2,282 jugadores con
Elo real quedaron invisibles en el selector — incluyendo Nadal, Federer,
Serena Williams, Osaka, Halep, Barty, Tiafoe.

**Fix:** `jugadores_por_genero.json` ya no se mantiene a mano — se
deriva de `tennis_elo_ratings.json` (`generar_jugadores_por_genero.py`,
corrido automáticamente al final de `calibrate_tennis_elo.py`). El
género de cada jugador se deriva de la fuente real (si jugó en archivos
`atp_matches_*` o `wta_matches_*`, agregado como campo `"genero"` en
`tennis_elo_ratings.json` durante la calibración), no de una heurística
por nombre.

Resultado: selector pasa de 226 a **2,467 jugadores** (1,247 masculino +
1,220 femenino). Validado en vivo: Nadal, Federer, Serena Williams,
Osaka y Barty ya aparecen y son analizables. 6 tests nuevos
(`tests/test_generar_jugadores_por_genero.py`).

## 11. FIX CRÍTICO: sesgo sistemático de total_esp — ACTIVADO (2026-08-09)

Diagnóstico del usuario confirmado con datos: Total Games generaba EV
inflado (36-56% con cuotas justas) casi siempre, mientras Match Winner
casi nunca aparecía como pick. Causa: `total_esp` (media de la
distribución de total de games) nunca fue calibrada — a diferencia de
`std_dev` (P1) — y sobreestimaba el total real en **+2.81 games (BO3) /
+3.10 games (BO5)**, con Brier score de Over/Under en umbrales bajos
**peor que adivinar a ciegas** (0.310 vs 0.25).

Fix: `total_esp = a + b*p_base*(1-p_base)`, coeficientes ajustados por
regresión (mínimos cuadrados) contra games reales, walk-forward, misma
metodología que ya validó Elo/H2H/decay. Sesgo baja de +2.81/+3.10 a
0.00 (por construcción), MAE mejora ~13%/~7%. `backtest_tennis.py` ahora
valida este mercado de forma permanente. Validado en vivo: los mismos 3
partidos de ejemplo con cuotas justas pasan de EV 36-56% a **sin pick**,
igual que Match Winner siempre se comportó correctamente. Detalle
completo en `tennis_backtest_results.md`.

## 12. FIX: el motor nunca evaluaba si jugador2 tenía valor (2026-08-09)

Segunda causa (independiente de la sección 11) de por qué "Match Winner
casi nunca aparecía como pick": `analizar()` solo evaluaba
`cuotas["match_winner"]["1"]` (jugador1) — nunca chequeaba `["2"]`, sin
importar cuánto valor real tuviera jugador2 aunque fuera el favorito del
modelo. Confirmado leyendo el código, no una hipótesis.

Fix: mismo bloque de evaluación, ahora recorre `(player1, "1", prob_j1)`
y `(player2, "2", prob_j2)` de forma independiente (mismo patrón que ya
usa Total Games para Over — cada lado se evalúa por separado, no es
"el mejor de los dos"). No cambia `prob_j1`/`prob_j2` — confirmado con
backtest: Brier 0.21895 / accuracy 64.21% idéntico al valor de
referencia, porque este fix es de qué se **muestra**, no de cómo
**predice** el modelo.

Validado en vivo: Rafael Jodar (jugador1) vs Jannik Sinner (jugador2,
favorito real del modelo, prob=0.846) con cuota 1.35 para Sinner
(generosa frente a la cuota justa 1.18) — antes del fix, cero picks
sin importar el valor real; después, `"Jannik Sinner gana"`, EV=14.2%,
verde. 4 tests nuevos (`tests/test_tennis_pick_ambos_jugadores.py`).

## 13. DIAGNÓSTICO: consolidación de 6 cabos sueltos (2026-08-09)

Diagnóstico de estado puro — sin activar nada. Baseline de referencia
en todos los backtests: producción actual (decay=0.25 + H2H
weight=0.18/min=2), evaluado sobre partidos desde 2024-01-01 (n=14,320):
**Brier 0.21895, accuracy 64.21%** (vs. coinflip 50%/0.25 y ranking
oficial ATP/WTA 63.45%). Detalle completo, con las 3 corridas de grid,
en `tennis_backtest_results.md`.

1. **shrink_elo() sobre baseline actual — RESUELTO, no activar.**
   Antes solo se había probado contra un baseline sin decay/H2H. Re-
   corrido con `sobre_baseline=True`, grid k=10/20/30/50: brier y
   accuracy empeoran de forma monótona con k (accuracy 64.06% → 63.74%
   a medida que k crece). Confirma la conclusión original con el
   baseline correcto: no aporta.

2. **Elo por superficie sobre baseline actual — RESUELTO, no activar.**
   Mismo re-test, grid min_games=5/10/20: empeora en los tres casos
   (accuracy 62.6-62.8% vs 64.21% del baseline) — peor que en el test
   original contra el baseline viejo. Confirmado, no activar.

3. **WTA vs ATP — IRRELEVANTE, sin asimetría real.** Desglose por
   género sobre el baseline: femenino brier 0.21787/accuracy 64.43%,
   masculino brier 0.21993/accuracy 64.01%. Diferencia de 0.4pp,
   dentro de ruido. No hay sesgo de género que corregir.

4. **Jugadores de muestra chica — OPORTUNIDAD REAL PENDIENTE (con
   matiz).** Desglose por games jugados del jugador con menos historial
   de los dos: `<10` games (n=1684) accuracy 62.62%/brier 0.219,
   `10-20` (n=1167) accuracy 60.50%/**brier 0.241 (peor de los 4
   buckets)**, `20-50` (n=2460) accuracy 63.19%, `50+` (n=9009)
   accuracy 65.27%/brier 0.215 (mejor). No hay degradación catastrófica
   ni sobreconfianza — el bucket `<10` no es el peor, lo es `10-20`
   (probablemente zona de transición donde el burn-in ya soltó al
   jugador pero el Elo aún no convergió). Con n=1167 el intervalo de
   confianza es ancho; vale la pena investigar pero no está confirmado
   como problema estadísticamente sólido todavía.

5. **Mercados: validador soporta 6, `analizar()` solo resuelve 2 —
   OPORTUNIDAD REAL PENDIENTE, con un bug concreto de por medio.**
   `tennis_validator.py` acepta `match_winner`, `primer_set`,
   `set_handicap`, `total_games`, `game_handicap`, `sets_winners`.
   `analizar()` (tennis_improved.py:544-604) solo genera picks para
   `match_winner` y `total_games` — `primer_set`/`set_handicap`/
   `game_handicap`/`sets_winners` se validan pero nunca se convierten
   en picks (código muerto desde la UI, que ni siquiera los junta en
   el payload). Además, dentro de `total_games` **solo se evalúa
   `"over"`** (línea 581: `if "over" in tg`) — pero el frontend
   (`templates/index.html:1381`) sí junta `cuotas.total_games.under`
   si el usuario lo completa. Si alguien carga solo la cuota de
   "under" (sin "over"), hoy se descarta en silencio sin generar pick,
   con valor real potencialmente ignorado — mismo patrón de bug que la
   sección 12, sin arreglar todavía.

6. **Umbrales verde/amarillo — VALIDADO (con matiz sobre dónde cae el
   corte real).** Desglose por certeza del modelo (`max(pred,1-pred)`,
   proxy de la "confianza" real que además suma el EV): 0.50-0.55
   accuracy 52.46%, 0.55-0.65 (≈amarillo) 57.22%, 0.65-0.70 (≈verde)
   62.58%, 0.70-0.80 accuracy 71.37%, 0.80+ accuracy 83.25%/brier
   0.138. La relación es monótona y sin inversiones — los umbrales SÍ
   separan tramos de accuracy real, no son arbitrarios. Matiz: el
   salto de calidad grande está en ~0.70, no en 0.65 — el tramo
   0.65-0.70 (justo el mínimo de "verde") todavía ronda 62.6% de
   accuracy, no muy por encima del amarillo alto. No es un error, pero
   subir el corte de verde a 0.70 daría una separación más limpia.

**Prioridad para implementar (ninguna implementada todavía):**
1. Arreglar el bug de `total_games` "under" nunca evaluado (punto 5) —
   mismo patrón exacto que la sección 12, bajo esfuerzo, bug real y
   concreto (no solo una hipótesis de mejora).
2. Investigar el bucket 10-20 games con más datos/otro corte de bucket
   antes de decidir una acción concreta (punto 4) — todavía no hay
   fix claro, solo una señal a seguir.
3. Evaluar subir `UMBRAL_VERDE` de 0.65 a 0.70 (punto 6) — bajo
   esfuerzo, pero cambia qué se le muestra al usuario como "alta
   confianza"; conviene decidirlo con el usuario, no solo por Brier.
4. Mercados sin resolver (primer_set/set_handicap/game_handicap/
   sets_winners, punto 5) — esfuerzo alto (UI + backtest de cada
   mercado nuevo), sin urgencia mientras no haya demanda de usarlos.
5. Puntos 1, 2 y 3: cerrados, no requieren acción.

## 14. FIX: Total Games "Under" nunca se evaluaba + FEATURE: log de
predicciones para trackear precisión en vivo (2026-08-09)

**Fix.** Mismo bug que la sección 12 (Match Winner J1/J2), ahora en
Total Games: `analizar()` solo miraba `cuotas["total_games"]["over"]`
— `"under"` se descartaba en silencio aunque el frontend ya lo manda
si el usuario completa esa cuota. Se agregó el bloque simétrico que
evalúa `"under"` de forma independiente (mismo patrón: no es "el mejor
de los dos", cada lado tiene su propio EV/confianza/pick). Como el fix
es de qué se **muestra**, no de cómo el modelo **predice**, se
reconfirmó el baseline de producción sin cambios: Brier 0.21895 /
accuracy 64.21% idénticos al valor de referencia de la sección 13.
Ejemplo real: Djokovic (1650) vs rival (1500), best_of_3, línea 26.5 —
p(under)=0.765, cuota 1.50 (generosa sobre la justa 1.31) → antes del
fix, cero picks sin importar el valor; después, `"Under 26.5 games"`,
EV=14.8%, amarillo. 5 tests nuevos
(`tests/test_tennis_total_games_under.py`).

**Feature: log de predicciones.** Hasta ahora solo se registraban en
`apuestas` las predicciones donde el usuario decidió apostar. Nuevo
módulo `src/services/tennis_predictions.py` (mismo patrón dual
CSV/Supabase que `tracking.py`) registra **cada** análisis de tenis
corrido desde `/api/analizar_tenis` — tenga pick de value o no — y
permite cargar el resultado real después (`POST /api/tenis/resultado`)
para calcular automáticamente:
- `acerto_ganador`: `favorito predicho == ganador real`.
- `acerto_total`: `|total_games_real - total_esp| <= std_dev` del
  formato (no un umbral arbitrario — la misma desviación estándar
  calibrada que ya usa el modelo para su propia distribución).

Nueva pestaña "Precisión Tenis" en `/historial` (`templates/historial.html`):
tabla de todas las predicciones logueadas, botones para cargar
ganador/total real por fila, y dos métricas destacadas arriba
("Precisión: quién gana" / "Precisión: total de games", cada una con su
propio N de partidos con resultado cargado — no siempre se cargan
juntos).

Tabla `predicciones_tenis` en Supabase (`supabase/schema.sql`) —
`std_dev` no se guarda por fila porque es constante por formato, se
deriva de `STD_GAMES` al cargar el resultado. 11 tests nuevos
(`tests/test_tennis_predictions.py`).

**Bug encontrado y corregido durante el trabajo (no en producción):**
`_init()` del nuevo módulo tocaba el CSV por defecto sin importar el
`csv_path` explícito del caller — hacía que corridas de test (que sí
pasan un `csv_path` temporal) igual crearan
`src/data/predicciones_tenis.csv` vacío en disco. Corregido para que
`_init()` respete el `csv_path` recibido. De paso, se detectó que
`src/data/apuestas_registradas.csv` (historial real de apuestas) quedó
fuera de `.gitignore` desde que se sacó de git tracking — sigue sin
subirse, pero conviene agregarlo al `.gitignore` para que no vuelva a
aparecer como "untracked" en cada `git status`.
