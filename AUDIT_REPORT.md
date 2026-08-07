━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUDIT REPORT - BetBrain
Fecha: 2026-08-06

## FINDINGS POR SEVERIDAD

### CRÍTICO (Bloquea producción)

- **API key real en `.env.example` sin comitear** | `.env.example:17` (working tree, no estaba en ningún commit) | El archivo `.env.example` tenía en disco una key de Groq real y funcional (prefijo `gsk_...`, redactada aquí a propósito) en vez de un placeholder. Verificación posterior con `git grep gsk_ <commit>` contra cada commit confirmó que **nunca llegó a comitearse ni a subirse a GitHub** — solo vivía en el archivo local sin trackear. | Reemplazar por placeholder (`TU_KEY_AQUI`), rotar la key igualmente por higiene (estuvo en texto plano en disco), y añadir un pre-commit hook de secret scanning (gitleaks/truffleHog) para prevenir que esto llegue a un commit en el futuro. — **Estado: corregido en Fase 2** (placeholder en `.env.example`, hook de gitleaks añadido; no requirió purga de historial).

- **Suite de tests rota / no ejecutable** | `tests/test_fetch_leagues.py:31`, `tests/test_modelos.py:10` | Ambos archivos ejecutan `sys.exit(1)` a nivel de módulo (son scripts standalone viejos, no tests de pytest). Al importarlos, pytest los trata como `SystemExit`, lo cual **aborta la recolección completa** (`INTERNALERROR`) y detiene la ejecución de TODO el resto de la suite. Hoy, `pytest` no corre de forma confiable ni un solo test. | Convertir estos scripts a tests reales con `assert` (no `sys.exit`), o moverlos fuera de `tests/` a una carpeta `scripts/diagnostics/` no recolectada por pytest.

- **pytest sin scope: recolecta el venv completo** | (raíz del proyecto, sin `pytest.ini`/`pyproject.toml`) | No existe configuración de `testpaths`/`rootdir`. Ejecutar `pytest` desde la raíz escanea `Lib/site-packages/` y recolecta **127,877 "tests"** de dependencias de terceros (sklearn, etc.) además de los tests reales del proyecto — inutilizable en la práctica y esconde fallos reales entre ruido. | Crear `pyproject.toml`/`pytest.ini` con `testpaths = ["tests"]` y `norecursedirs = ["Lib", "Scripts", "Include", ".git"]`.

- **Entorno virtual y artefactos versionados en git** | raíz del repo (`Lib/`, `Scripts/`, `Include/`, `__pycache__/`, `data/api_cache/*.json`) | El `.gitignore` solo excluye `Lib/`, `venv/`, `.env` — pero **27 archivos de `Lib/Scripts/Include` ya están commiteados** (probablemente antes de crear el `.gitignore`), junto con **27 `__pycache__`** y **260 archivos de caché de API** (`data/api_cache/*.json`). Esto hace el repo no reproducible (binarios `.exe` de Windows en `Scripts/`), infla el historial y genera diffs de ruido en cada commit. | `git rm -r --cached Lib Scripts Include "**/__pycache__" data/api_cache`, confirmar que están en `.gitignore`, y purgar del historial si el tamaño lo justifica. — **Estado: corregido en Fase 2** (desrastreados, `.gitignore` actualizado).

### ALTO (Antes de deploy)

- **Motor de tenis duplicado/ambiguo** | `src/engines/tennis.py` (482 líneas) vs `src/engines/tennis_improved.py` (usado por `app.py:45`) | `app.py` usa `TennisImprovedEngine`, pero `src/engines/tennis.py` (el motor viejo, sin Elo calibrado) sigue en el repo y aún es importado por `test_app_imports.py:17`. No hay señal de deprecación. Riesgo de que alguien reintroduzca el motor viejo por error. | Eliminar `tennis.py` si está reemplazado, o renombrarlo/marcarlo `@deprecated` y actualizar el test. — **Estado: corregido en Fase 4** (`src/engines/tennis.py` eliminado, sin referencias restantes).

- **Test con import a ruta inexistente** | `tests/test_fetch_leagues.py:135` | Importa `from src.fetch_leagues import ...`, pero el módulo real está en `src/providers/fetch_leagues.py`. El test nunca fue actualizado tras el refactor a `src/providers/` — evidencia de que la suite no corre en CI ni se mantiene junto al código. | Eliminar o reescribir el test contra la ruta actual. — **Estado: corregido en Fase 3** (archivo eliminado; su función real, `get_teams_by_league` vía `LIGAS_CONFIG`, sigue cubierta indirectamente por los tests de `/api/teams`).

- **Bug real: `src/data/equipos_por_liga.json` incompleto** | `src/providers/league_manager.py:27` | Descubierto al arreglar los tests en Fase 3: había dos copias del archivo de equipos — la de `src/data/` (la que el código realmente lee) solo tenía 5 de 14 ligas, dejando `/api/teams` roto para Liga MX, Champions League, MLS, Brasileirao, Eredivisie, Primeira Liga, Championship y Liga Profesional Argentina. | **Estado: corregido en Fase 3** (`src/data/equipos_por_liga.json` reemplazado por la copia completa de `data/equipos_por_liga.json`).

- **Bug real: filtro de riesgo `_check_marginal_ou` perdido en el refactor** | `src/engines/football.py` (ausente hasta Fase 3) | El `app.py` pre-refactor (`a968ea0`) descartaba picks de Over/Under 2.5 cuando el xG modelado caía en zona marginal (±0.30 de la línea) sin datos reales que confirmaran la tendencia — un filtro de gestión de riesgo real, no cosmético. Se perdió al mover la lógica a `src/engines/football.py`. | **Estado: corregido en Fase 3** (reimplementado como `check_marginal_ou`, expuesto en `debug_filtrado.descartados` con motivo `"marginal_ou"`).

- **Bug real: `evaluar_coherencia()` llamada con argumentos equivocados** | `app.py` (llamada a `evaluar_coherencia`) | Se le pasaba `prob_1x2_final` en vez de `cuotas`, y `cuotas["1X2"]` en vez de `lambdas` — causaba `KeyError: 'home'` capturado silenciosamente por un `except Exception: pass`. Resultado: la narrativa del LLM **nunca se generaba**, aunque hubiera picks válidos, desde que se hizo el refactor. | **Estado: corregido en Fase 3** (argumentos corregidos; la excepción ahora también se loguea en vez de tragarse en silencio).

- **Sin CI/CD** | (no existe `.github/workflows/`, ni ningún pipeline) | Nada bloquea un merge con tests rotos, secretos filtrados o código sin lint. | Añadir GitHub Actions mínimo: lint + pytest + secret-scan en cada push/PR.

- **Dependencias sin pin, sin lockfile** | `requirements.txt` (todas con `>=`) | `flask>=3.0.0`, `xgboost>=2.0.0`, etc. sin cotas superiores ni hashes. Un `pip install` en dos momentos distintos puede traer versiones incompatibles del modelo XGBoost calibrado (`models/*.pkl`), rompiendo la predicción silenciosamente. | Generar lockfile (`pip-compile` / `poetry.lock`) con versiones exactas, separar `requirements-dev.txt`.

- **Servidor Flask de desarrollo como único punto de entrada** | `app.py:418` (`app.run(...)`) | No hay WSGI de producción (gunicorn/waitress), ni límites de tamaño de request. El propio Flask lo advierte como "no apto para producción". | Servir con `waitress` detrás de un proceso supervisado (planeado para Fase 8). `SECRET_KEY` ya corregido en Fase 2 (`config.py`, desde entorno con fallback aleatorio por proceso).

- **Endpoints sin autenticación ni rate limiting** | `app.py` (todas las rutas `/api/*`, `/chat`) | Cualquiera con acceso de red puede escribir en `data/ledger.json`/CSV vía `/api/registrar_apuesta`, o agotar la cuota de APIs externas (Groq, API-Football) vía `/chat`. Aceptable en local; no en producción. | Añadir auth (API key / login) y rate limiting (Flask-Limiter) antes de exponer el servicio fuera de `127.0.0.1`.

### MEDIO (Pronto)

- **Tests duplicados/dispersos** | raíz (`test_api_teams.py`, `test_app_imports.py`, `test_frontend_debug.py`, `test_tennis_integration.py`) + `tests/` | Dos convenciones de ubicación coexisten sin razón aparente; complica saber qué se ejecuta con `pytest` por defecto. | Consolidar todo bajo `tests/`. — **Estado: corregido en Fase 3** (tests reales consolidados en `tests/`; scripts de diagnóstico manual movidos a `diagnostics/`, fuera de la recolección de pytest).

- **Sin README.md** | raíz del proyecto | Existen 10 `.md` de estado ad-hoc (`FIX_SUMMARY.md`, `INTEGRATION_REPORT.md`, `MOTOR_STATUS.md`, `SYSTEM_OVERVIEW.md`, etc.) pero ningún `README.md` canónico — punto de entrada esperado por cualquier herramienta/dev nuevo. | Crear `README.md` que consolide instalación, arquitectura y enlace al resto como histórico.

- **`except Exception` amplio y silencioso** | `app.py:193`, `app.py:225-226`, `app.py:240-241` | Fallos de contexto de API, coherencia y narrativa LLM se tragan sin registrar el stacktrace (solo `debug` o nada), lo que puede esconder degradación sistemática (ej. la API externa lleva días caída) detrás de un "sin datos" silencioso. Este exact patrón escondió el bug real de `evaluar_coherencia()` (ver arriba) durante todo el refactor. | Loguear a nivel `warning`/`error` con el stacktrace incluso cuando el fallback continúa. — **Estado: parcialmente corregido** (el bloque de `coherencia` ya loguea en Fase 3; los de `contexto_api` y `narrativa` LLM quedan para Fase 5).

- **Efectos secundarios al importar `config.py`** | `config.py:23-24` | `mkdir` se ejecuta en tiempo de import, no de uso — dificulta testear el módulo sin tocar el filesystem real y puede fallar en entornos read-only (contenedores). | Mover la creación de directorios a una función `ensure_dirs()` invocada explícitamente en el arranque.

- **Type hints inconsistentes** | scripts de raíz (`calibrate_*.py`, `generar_*.py`) vs `src/` | Los módulos de `src/` tienen hints parciales; los scripts de raíz prácticamente ninguno. No hay `mypy`/`pyright` configurado para exigirlo. | Añadir `mypy` con modo incremental empezando por `src/core` y `src/engines`.

### BAJO (Nice to have)

- **Logs versionados en git** | `api_calls.log`, `logs/betbrain.log` | Archivos de log no deberían commitearse; crecen indefinidamente en el historial. | Añadir a `.gitignore`, eliminar del tracking. — **Estado: corregido en Fase 2** (`*.log` en `.gitignore`, desrastreados).

- **Sin LICENSE** | raíz | Ambiguo para cualquier colaborador externo sobre términos de uso. | Añadir `LICENSE` si se planea abrir el repo.

- **Prints con emojis en scripts operativos** | `generar_*.py`, `calibrate_*.py`, mensajes en `SYSTEM_OVERVIEW.md` | Cómodo para uso manual en CLI, pero no es logging estructurado; dificulta parsear salida en un pipeline automatizado. | Migrar prints de scripts operativos a `logging` con niveles.

- **Múltiples `.md` de estado redundantes** | raíz (10 archivos: `FIX_SUMMARY.md`, `INTEGRATION_REPORT.md`, `INTEGRATION_SUMMARY.md`, `INTEGRATION_SUMMARY_TENNIS.md`, `MOTOR_STATUS.md`, `SYSTEM_OVERVIEW.md`, `QUICK_START.md`, `SETUP_LIGAS.md`, `TESTING_GUIDE.md`, `UPGRADE_GUIDE.md`, `FRONTEND_IMPROVEMENTS.md`, `EJECUTAR_AHORA.md`) | Documentación fragmentada y con fechas/estados que se contradicen entre sí con el tiempo (algunos ya desactualizados frente al código actual). | Consolidar en `README.md` + `docs/CHANGELOG.md`; archivar el resto.

## FASE 4 — BARRIDO DE CÓDIGO MUERTO (`vulture`)

Con confianza ≥80% (candidatos de alta certeza), `vulture` solo encontró una
variable local sin usar (`max_years` en `src/providers/tennis_data_loader.py:25`) —
irrelevante, no se tocó.

Con confianza ≥60% aparecen ~55 símbolos más, pero son en su mayoría falsos
positivos esperables en este tipo de proyecto:
- **Rutas Flask** (`api_teams`, `api_players`, `not_found`, etc. en `app.py`) —
  se invocan vía decorador `@app.route`, `vulture` no rastrea eso.
- **Métodos de utilidad usados solo por tests** (`src/providers/manager.py`:
  `get_h2h`, `csv_info`, `sportsmonk_disponible`, `reset_stats`, etc.) —
  cubiertos por `tests/test_multi_source.py`, `vulture` no cruza con la suite.
- **`TennisEloCalculator`** (`src/core/tennis_elo.py`) — parece huérfano visto
  solo desde `src/`, pero lo usan `calibrate_elo_simple.py` y
  `calibrate_tennis_elo.py` (scripts de calibración en la raíz).
- **Todo `src/nba/`** (`NBAEngine`, `nba_validator.py`) — genuinamente no está
  conectado a ninguna ruta de `app.py` todavía, pero es trabajo en progreso
  documentado (ver `SYSTEM_OVERVIEW.md`: "NBA en desarrollo"), no código muerto
  a eliminar.

No se borró nada de esta lista — son candidatos para que el usuario revise
con contexto de producto, no código muerto confirmado.

## ESTADÍSTICAS

- Total archivos Python (excluyendo venv `Lib/`/`Scripts/`/`Include/`): **54**
- Líneas de código totales (todo el proyecto, sin venv): **10,888**
- Líneas de código del núcleo (`app.py` + `config.py` + `src/`, sin tests/scripts): **7,907**
- Coverage de tests: **No medible** — la suite no completa la recolección con `pytest` (ver hallazgo crítico de tests rotos); efectivamente 0% ejecutable de forma automatizada hoy.
- Archivos con docstring de módulo: **28/32** relevantes (87.5%) — nota: mide solo docstring a nivel de módulo, no cobertura de funciones/clases individuales.
- Archivos de caché/API commiteados en git: **260** (`data/api_cache/`)
- Archivos de venv commiteados por error: **27**

## TOP 3 PRIORIDADES

1. **Revocar y rotar la API key de Groq filtrada en `.env.example`** — es explotable ahora mismo por cualquiera con acceso de lectura al repo.
2. **Arreglar la suite de tests** (eliminar los `sys.exit()` a nivel de módulo en `tests/test_fetch_leagues.py` y `tests/test_modelos.py`, y añadir `pytest.ini` con `testpaths`) — sin esto, no hay forma confiable de saber si algo se rompe.
3. **Sacar el venv, `__pycache__` y caché de API del control de versiones** — bloquea cualquier limpieza de historial posterior y sigue ensuciando cada commit futuro.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
