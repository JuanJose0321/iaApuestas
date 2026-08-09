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
> declarada explícitamente en GitHub. El resto de los hallazgos (P1/P2,
> incluyendo el `std_dev` fijo de total de games que sigue generando EV
> inflado) permanece sin resolver — quedan fuera del alcance de este fix.

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
