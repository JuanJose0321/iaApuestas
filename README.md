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
pip install -r requirements-dev.txt      # + pytest, mypy, vulture, pre-commit
```

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

**Riesgos conocidos de esta ruta**: el bundle incluye `xgboost` +
`scikit-learn` + `pandas` + `numpy` (pesados) — cerca del límite de 500MB
del plan gratuito, revisa el log de build si falla por tamaño. El plan
Hobby limita cada función a 10s por defecto (`vercel.json` pide 30s vía
`maxDuration`, pero el techo real depende del plan) — `/chat` encadena
llamadas a APIs externas y puede tardar más que eso en el peor caso.

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
