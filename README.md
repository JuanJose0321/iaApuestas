# BetBrain

Herramienta de análisis de apuestas deportivas (fútbol y tenis, NBA en
desarrollo). Compara la probabilidad estimada por el modelo contra la cuota
que ofrece la casa de apuestas, calcula el valor esperado (EV) y sugiere
stake vía Kelly fraccionado. No es un producto para terceros — es una
herramienta personal para validar si el modelo produce un win rate rentable
a largo plazo.

## Arquitectura

- **`app.py`** — servidor Flask. Rutas principales: `/chat` (analiza un
  partido de fútbol), `/api/analizar_tenis`, `/api/teams`, `/api/players`,
  y el tracking de apuestas (`/api/registrar_apuesta`, `/api/historial`,
  `/api/metricas`).
- **`src/core/`** — cálculos puros: distribución de Poisson
  (`probability.py`), sistema de confianza (`confidence.py`), coherencia
  modelo↔mercado (`coherence.py`), Elo de tenis (`tennis_elo.py`).
- **`src/engines/`** — motores de predicción: `football.py` (Poisson +
  XGBoost calibrado), `tennis_improved.py` (Elo calibrado por superficie).
- **`src/providers/`** — conectores a fuentes de datos externas
  (TheSportsDB, API-Football, Sportmonks, CSVs históricos de
  football-data.co.uk), con caché en disco y fallback si una fuente falla.
- **`src/services/`** — `bankroll.py` (Kelly fraccionado), `tracking.py`
  (registro de apuestas y métricas), `analyst.py` (narrativa vía LLM,
  opcional).
- **`templates/` + `static/`** — frontend tipo chat.

## Instalación

Requiere Python 3.14 (ver `pyvenv.cfg`).

```bash
pip install -r requirements.txt          # producción
pip install --no-deps xgboost==3.2.0     # aparte, para no arrastrar scipy (ver "Despliegue en Vercel")
pip install -r requirements-dev.txt      # dev local/CI: ya incluye xgboost normal, + pytest/mypy/vulture/pre-commit
```

`requirements.txt` ya no incluye `xgboost` directamente — se instala aparte con
`--no-deps` para no arrastrar `scipy` (135MB) como dependencia transitiva
innecesaria (el motor de fútbol usa la API nativa de XGBoost, sin
`scikit-learn`/`scipy`). `requirements-dev.txt` ya se encarga de esto por vos
si vas a correr tests o development local.

Copia `.env.example` a `.env` y rellena tus claves (todas opcionales excepto
`SECRET_KEY`, que puede generarse con
`python -c "import secrets; print(secrets.token_hex(32))"`). Sin las APIs
externas configuradas, el sistema sigue funcionando con los CSV/JSON locales
como respaldo.

## Correr la app

```bash
python app.py       # desarrollo — servidor de Flask, con auto-reload si FLASK_DEBUG=1
python wsgi.py       # "producción" local — mismo Flask app servido con waitress
```

Sirve en `http://127.0.0.1:5000` por defecto (configurable con `FLASK_PORT`).
Los endpoints que escriben datos o llaman APIs externas (`/chat`,
`/api/analizar_tenis`, `/api/registrar_apuesta`) tienen rate limiting
(Flask-Limiter). Sigue pensado para uso local/personal, no para exponerse
públicamente sin añadir autenticación (ver `AUDIT_REPORT.md`).

## Despliegue en Vercel + Supabase

Vercel detecta automáticamente la instancia `Flask` llamada `app` en
`app.py` en la raíz y la despliega como una única función — no hace falta
reescribir rutas a mano. Lo único que cambia es la persistencia de
apuestas: el filesystem de Vercel es de solo lectura fuera de `/tmp`, así
que `src/services/tracking.py` usa Supabase en vez de CSV/JSON local
cuando detecta `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` configurados
(si no están, sigue usando CSV local — no rompe el flujo de desarrollo).

Pasos:
1. Crea un proyecto en [supabase.com](https://supabase.com) y corre
   `supabase/schema.sql` en **SQL Editor** (no se puede automatizar con
   solo la API key — hace falta un token de acceso personal de Supabase).
2. En [vercel.com](https://vercel.com) → Import Project → conecta el repo.
3. En Project Settings → Environment Variables, agrega `SUPABASE_URL`,
   `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY` (ver `.env.local`,
   nunca commiteado) y las que uses de fútbol/tenis (`GROQ_API_KEY`, etc.).
4. Deploy.

### Tamaño del bundle

El motor de fútbol usa `polars` (no `pandas`) y la API nativa de XGBoost con
calibración isotónica propia (`src/core/calibration.py`, sin `scikit-learn`/
`scipy` — ver ese archivo para el porqué). `vercel.json` instala `xgboost`
aparte con `--no-deps` para que `pip` no reinstale `scipy` como dependencia
transitiva declarada (135MB que el código no usa en absoluto).

`requirements.txt` usa `polars-lts-cpu`, no `polars` a secas: el paquete
`polars` normal viene partido en dos (`polars` + `polars-runtime-32`, este
último con el binario nativo de Rust) y ese runtime solo pesaba ~176MB en
mediciones reales — más que `pandas`, `scikit-learn` y `scipy` juntos.
`polars-lts-cpu` es un solo wheel autocontenido, sin la optimización AVX2
que no necesitamos para el volumen de datos de este proyecto (miles de
filas de CSV). Medido en un venv limpio (mismo proceso que usa
`vercel.json`): 426.7MB → 378.3MB.

`xgboost` (140MB) sigue siendo lo más pesado — es el motor de predicción,
no hay forma de sacarlo sin perder el ensemble ML. Si el build en Vercel
vuelve a fallar por tamaño, las opciones que quedan son: reducir el motor a
solo-Poisson para el despliegue serverless (el código ya soporta ese
fallback), o usar el despliegue tradicional (`wsgi.py` + waitress, más
arriba) en un host sin límite de bundle — no tiene esta restricción porque
no es un modelo serverless.

El plan Hobby de Vercel limita cada función a 10s por defecto (`vercel.json`
pide 30s vía `maxDuration`, pero el techo real depende del plan) — `/chat`
encadena llamadas a APIs externas y puede tardar más que eso en el peor caso.

## Tests

```bash
pytest tests/            # suite automatizada (84 tests, sin dependencias externas)
mypy src/core src/engines
```

Los scripts en `diagnostics/` **no** son tests de pytest — son scripts
manuales que requieren el servidor corriendo o claves de API reales. Ver
`diagnostics/README.md`.

## CLI

```bash
python src/cli.py analyze "Real Madrid" "Barcelona" --c1 2.1 --cx 3.4 --c2 3.2
python src/cli.py backtest
python src/cli.py clear-cache
```

## Documentación adicional

Hay más detalle histórico del desarrollo (informes de integración, guías de
setup por feature, etc.) archivado en `docs/archive/` — puede estar
desactualizado frente al código actual; este README es la fuente de verdad.
Ver también `AUDIT_REPORT.md` para el estado de deuda técnica conocida.
