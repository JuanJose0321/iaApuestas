"""
Diagnóstico completo del sistema BetBrain.
Ejecutar: Scripts/python diagnostico.py

Prueba en orden:
  1. Variables de entorno y configuración
  2. Sportmonks (API real con token)
  3. CSV football-data (datos locales)
  4. DataSourceManager (cadena de fallback)
  5. Engine de picks con partidos reales de Sportmonks
  6. LLM Groq
"""
import json
import logging
import sys
import time
from pathlib import Path

# Forzar UTF-8 en consola Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Logging en consola ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)-22s] %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
# Silenciar urllib3 para no spamear
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

log = logging.getLogger("diagnostico")

SEP = "=" * 70
SEP2 = "-" * 70

def ok(msg): log.info("  [OK]   %s", msg)
def fail(msg): log.error("  [ERR]  %s", msg)
def info(msg): log.info("         %s", msg)
def warn(msg): log.warning("  [WARN] %s", msg)
def titulo(msg): log.info("\n%s\n  %s\n%s", SEP, msg, SEP)
def subtitulo(msg): log.info("\n%s\n  %s", SEP2, msg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
titulo("1. CONFIGURACIÓN Y VARIABLES DE ENTORNO")
try:
    from config import (
        API_FOOTBALL_KEY, API_FOOTBALL_HOST, SPORTMONKS_TOKEN,
        GROQ_API_KEY, GROQ_MODEL,
        BANKROLL_INICIAL, KELLY_FRACTION, MIN_EV,
        CACHE_DIR, CACHE_TTL_HORAS, PROMEDIO_GOLES_LIGA,
    )
    ok(f"config.py cargado")
    info(f"API_FOOTBALL_KEY : {'presente' if API_FOOTBALL_KEY else 'VACIO (api-football deshabilitado)'}")
    info(f"SPORTMONKS_TOKEN : {'presente' if SPORTMONKS_TOKEN else 'VACIO (sportsmonk deshabilitado)'}")
    info(f"GROQ_API_KEY     : {'presente' if GROQ_API_KEY else 'VACIO (LLM deshabilitado)'}")
    info(f"GROQ_MODEL       : {GROQ_MODEL}")
    info(f"BANKROLL_INICIAL : {BANKROLL_INICIAL}")
    info(f"KELLY_FRACTION   : {KELLY_FRACTION}")
    info(f"MIN_EV           : {MIN_EV}")
    info(f"CACHE_DIR        : {CACHE_DIR}")
    info(f"CACHE_TTL_HORAS  : {CACHE_TTL_HORAS}h")
except Exception as e:
    fail(f"Error cargando config: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPORTMONKS
# ─────────────────────────────────────────────────────────────────────────────
titulo("2. SPORTMONKS API (fuente live principal)")

partidos_hoy_sm = []

if not SPORTMONKS_TOKEN:
    warn("SPORTMONKS_TOKEN no configurado — saltando pruebas de Sportmonks")
else:
    try:
        import src.sportsmonk as sm
        ok(f"src/sportsmonk.py importado OK")
        info(f"disponible(): {sm.disponible()}")

        # Test: buscar equipo conocido
        subtitulo("2a. Búsqueda de equipo")
        t0 = time.time()
        team_id = sm.search_team("Barcelona")
        elapsed = time.time() - t0
        if team_id:
            ok(f"search_team('Barcelona') id={team_id} ({elapsed:.2f}s)")
        else:
            warn("search_team('Barcelona') = None (puede ser limite de plan free)")

        # Test: forma del equipo
        if team_id:
            subtitulo("2b. Forma del equipo (últimos 5 partidos)")
            t0 = time.time()
            forma = sm.get_team_form(team_id, last=5)
            elapsed = time.time() - t0
            if forma:
                ok(f"get_team_form(Barcelona) OK ({elapsed:.2f}s)")
                info(f"  Partidos  : {forma['partidos']}")
                info(f"  W/D/L     : {forma['W']}/{forma['D']}/{forma['L']}")
                info(f"  GF/GC avg : {forma['gf_promedio']} / {forma['gc_promedio']}")
                info(f"  BTTS rate : {forma['btts_rate']:.0%}")
                info(f"  Over2.5   : {forma['over_25_rate']:.0%}")
                info(f"  Secuencia : {forma['secuencia']}")
            else:
                warn(f"get_team_form = None ({elapsed:.2f}s). Plan gratuito puede no incluir fixtures detallados.")

        # Test: partidos de hoy via Sportmonks
        subtitulo("2c. Partidos disponibles hoy (Sportmonks)")
        import requests as _req
        from datetime import date
        hoy = date.today().isoformat()
        # Sportmonks v3: filtros van en el path, no como query params
        resp = _req.get(
            f"https://api.sportmonks.com/v3/football/fixtures/date/{hoy}",
            params={
                "api_token": SPORTMONKS_TOKEN,
                "include": "participants;league",
                "per_page": 50,
            },
            timeout=20,
        )
        if resp.status_code == 200:
            fixtures_data = resp.json().get("data", [])
            ok(f"GET /fixtures/date/{hoy} = {len(fixtures_data)} partidos")
            for fix in fixtures_data[:10]:
                parts = fix.get("participants", [])
                home_name = away_name = "?"
                for p in parts:
                    loc = p.get("meta", {}).get("location", "")
                    if loc == "home": home_name = p.get("name", "?")
                    elif loc == "away": away_name = p.get("name", "?")
                league = fix.get("league", {}).get("name", "?") if fix.get("league") else "?"
                hora = (fix.get("starting_at") or "")[:16]
                info(f"  [{hora}] {league}: {home_name} vs {away_name}")
                partidos_hoy_sm.append({
                    "home": home_name, "away": away_name,
                    "liga": league, "hora": hora,
                })
            if len(fixtures_data) > 10:
                info(f"  ... y {len(fixtures_data)-10} mas")
        elif resp.status_code == 403:
            warn("403 Forbidden: plan gratuito de Sportmonks no incluye /fixtures por fecha")
            info("  Para partidos en vivo necesitas plan de pago en sportmonks.com")
        else:
            warn(f"GET /fixtures/date/{hoy} = HTTP {resp.status_code}")
            info(f"  Respuesta: {resp.text[:300]}")

    except Exception as e:
        fail(f"Sportmonks excepción: {e}")
        log.exception("Traceback completo:")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FOOTBALL-DATA CSV
# ─────────────────────────────────────────────────────────────────────────────
titulo("3. FOOTBALL-DATA CSV (datos locales sin API key)")
try:
    from src.football_data_api import info_cobertura, get_team_form_csv, get_h2h_csv
    cob = info_cobertura()
    if cob.get("partidos", 0) > 0:
        ok(f"CSV: {cob['partidos']} partidos, {cob['equipos']} equipos, {cob['ligas']} ligas")
        info(f"  Rango fechas: {cob.get('rango_fechas', 'N/A')}")
        info(f"  Ligas: {', '.join(cob.get('lista_ligas', []))}")

        # Test forma desde CSV
        subtitulo("3a. Forma desde CSV — Real Madrid")
        forma_rm = get_team_form_csv("Real Madrid", last=5)
        if forma_rm:
            ok("Forma Real Madrid desde CSV")
            info(f"  Partidos: {forma_rm['partidos']} | {forma_rm['W']}W {forma_rm['D']}D {forma_rm['L']}L")
            info(f"  GF: {forma_rm['gf_promedio']} | GC: {forma_rm['gc_promedio']}")
            info(f"  Secuencia: {forma_rm['secuencia']}")
        else:
            warn("Real Madrid no encontrado en CSV (nombre distinto en la liga)")

        # Test H2H desde CSV
        subtitulo("3b. H2H desde CSV — Real Madrid vs Barcelona")
        h2h = get_h2h_csv("Real Madrid", "Barcelona", last=10)
        if h2h:
            ok(f"H2H Real Madrid vs Barcelona: {h2h['n']} partidos")
            info(f"  Goles promedio: {h2h['goles_promedio']}")
            info(f"  BTTS: {h2h['btts_rate']:.0%} | Over2.5: {h2h['over_25_rate']:.0%}")
        else:
            warn("Sin H2H en CSV para este par de equipos")
    else:
        warn("CSV sin datos — ejecuta: Scripts/python src/data_loader.py para descargar")
        info(f"  LIGAS configuradas: {cob.get('lista_ligas', [])}")
except Exception as e:
    fail(f"football_data_api excepción: {e}")
    log.exception("Traceback:")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA SOURCE MANAGER
# ─────────────────────────────────────────────────────────────────────────────
titulo("4. DATA SOURCE MANAGER (cadena de fallback)")
try:
    from src.data_source_manager import dsm
    ok(f"DSM importado OK — fuente_default='{dsm.fuente_default}'")
    info(f"  sportsmonk_disponible: {dsm.sportsmonk_disponible()}")

    subtitulo("4a. contexto_partido_completo — Real Madrid vs Barcelona")
    t0 = time.time()
    ctx = dsm.contexto_partido_completo("Real Madrid", "Barcelona")
    elapsed = time.time() - t0
    ok(f"DSM respondió en {elapsed:.2f}s")
    info(f"  fuente        : {ctx.get('fuente')}")
    info(f"  api_disponible: {ctx.get('api_disponible')}")
    info(f"  forma_home    : {'SI' if ctx.get('forma_home') else 'NO'}")
    info(f"  forma_away    : {'SI' if ctx.get('forma_away') else 'NO'}")
    info(f"  h2h           : {'SI' if ctx.get('h2h') else 'NO'}")
    if ctx.get("notas"):
        for nota in ctx["notas"]:
            info(f"  nota: {nota}")

    subtitulo("4b. Estadísticas del DSM")
    stats = dsm.stats()
    for k, v in stats.items():
        info(f"  {k:20s}: {v}")

except Exception as e:
    fail(f"DSM excepción: {e}")
    log.exception("Traceback:")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENGINE DE PICKS — análisis real de partidos
# ─────────────────────────────────────────────────────────────────────────────
titulo("5. ENGINE DE PICKS — análisis con cuotas reales")

PARTIDOS_TEST = [
    # (home, away, liga, cuotas)  — cuotas de ejemplo
    ("Real Madrid", "Barcelona", "LaLiga", {
        "1X2":    {"1": 2.20, "X": 3.50, "2": 3.30},
        "OU_2.5": {"Over": 1.85, "Under": 1.95},
        "BTTS":   {"Yes": 1.72, "No": 2.08},
    }),
    ("Manchester City", "Arsenal", "Premier League", {
        "1X2":    {"1": 1.95, "X": 3.60, "2": 4.00},
        "OU_2.5": {"Over": 1.90, "Under": 1.90},
        "BTTS":   {"Yes": 1.80, "No": 2.00},
    }),
    ("Bayern Munich", "Borussia Dortmund", "Bundesliga", {
        "1X2":    {"1": 1.70, "X": 3.90, "2": 5.00},
        "OU_2.5": {"Over": 1.75, "Under": 2.05},
        "BTTS":   {"Yes": 1.85, "No": 1.95},
    }),
]

# Si tenemos partidos reales de Sportmonks, analizar el primero también
if partidos_hoy_sm:
    p = partidos_hoy_sm[0]
    if p["home"] != "?" and p["away"] != "?":
        PARTIDOS_TEST.insert(0, (p["home"], p["away"], p["liga"], {
            "1X2":    {"1": 2.30, "X": 3.30, "2": 3.00},
            "OU_2.5": {"Over": 1.85, "Under": 1.95},
            "BTTS":   {"Yes": 1.75, "No": 2.05},
        }))
        info(f"Añadido partido real de Sportmonks: {p['home']} vs {p['away']}")

try:
    from src.engine import BettingEngine, leg_legible
    from src.confidence import calcular_confianza, nivel_confianza, UMBRAL_CONFIANZA
    engine = BettingEngine()
    ok(f"Engine cargado  calibrator={'SI' if engine.calibrator else 'NO'}  df_stats={'SI' if engine.df_stats is not None and not engine.df_stats.empty else 'NO'}")

    for home, away, liga, cuotas in PARTIDOS_TEST:
        subtitulo(f"Partido: {home} vs {away}  [{liga}]")
        try:
            from config import PROMEDIO_GOLES_LIGA
            promedio = PROMEDIO_GOLES_LIGA.get(liga, PROMEDIO_GOLES_LIGA["Default"])

            # Obtener contexto real
            ctx = dsm.contexto_partido_completo(home, away)
            fuente = ctx.get("fuente", "desconocida")

            # Ajustar promedio con forma real
            fh = ctx.get("forma_home")
            fa = ctx.get("forma_away")
            if fh and fa:
                xg_real = fh["gf_promedio"] + fa["gf_promedio"]
                promedio = round(0.70 * promedio + 0.30 * xg_real, 2)
                info(f"  Promedio ajustado con forma real: {promedio}")

            # Engine
            res = engine.pick_multileg(home, away, cuotas, promedio_goles_liga=promedio)
            lambdas = res.get("lambdas", {})
            xg_total = lambdas.get("home", 0) + lambdas.get("away", 0)

            info(f"  Fuente datos : {fuente} (api_disponible={ctx.get('api_disponible')})")
            info(f"  xG estimado  : {lambdas.get('home', '?')} – {lambdas.get('away', '?')}")
            info(f"  xG total     : {xg_total:.2f}")
            info(f"  Fuente 1X2   : {res.get('fuente_1x2')}")

            # Mostrar probabilidades
            p1x2 = res.get("prob_1x2_final", {})
            if p1x2:
                info(f"  Prob 1/X/2   : {p1x2.get('1',0):.1%} / {p1x2.get('X',0):.1%} / {p1x2.get('2',0):.1%}")

            picks_encontrados = 0
            tiene_datos = ctx.get("api_disponible") and (fh or fa)
            factor_datos = 1.0 if tiene_datos else 0.6

            for etiqueta, key in [("DIRECTA", "directa"), ("DUPLA", "dupla"), ("TRIPLETA", "tripleta")]:
                p_pick = res.get(key)
                if p_pick is None:
                    continue
                legs_txt = " + ".join(
                    f"{leg_legible(leg, home, away)} @{leg['cuota']}"
                    for leg in p_pick.get("legs", [])
                )
                prob = p_pick["prob_conjunta"]
                ev   = p_pick["ev"]
                cuota_total = p_pick["cuota_total"]
                conf = calcular_confianza(prob, ev, factor_datos)
                nivel = nivel_confianza(conf)

                picks_encontrados += 1
                pasó = conf >= UMBRAL_CONFIANZA
                marca = "[PICK]" if pasó else "[bajo umbral]"
                info(f"  [{etiqueta}] {legs_txt}")
                info(f"    cuota={cuota_total:.2f} prob={prob:.1%} EV={ev:.1%} confianza={conf:.0%} {nivel} {marca}")

            if picks_encontrados == 0:
                info(f"  Sin picks con EV positivo en estas cuotas")

        except Exception as e:
            fail(f"Engine excepción para {home} vs {away}: {e}")
            log.exception("Traceback:")

except Exception as e:
    fail(f"Engine import excepción: {e}")
    log.exception("Traceback:")


# ─────────────────────────────────────────────────────────────────────────────
# 6. GROQ LLM
# ─────────────────────────────────────────────────────────────────────────────
titulo("6. GROQ LLM (narrador)")
if not GROQ_API_KEY:
    warn("GROQ_API_KEY no configurado — LLM deshabilitado")
else:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        t0 = time.time()
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Di solo: OK"}],
            max_tokens=5,
        )
        elapsed = time.time() - t0
        answer = resp.choices[0].message.content.strip()
        ok(f"Groq respondió en {elapsed:.2f}s: '{answer}'")
        info(f"  Modelo: {GROQ_MODEL}")
        info(f"  Tokens usados: {resp.usage.total_tokens}")
    except Exception as e:
        fail(f"Groq excepción: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
titulo("7. RESUMEN DEL SISTEMA")

componentes = {
    "API-Football key":  bool(API_FOOTBALL_KEY),
    "Sportmonks token":  bool(SPORTMONKS_TOKEN),
    "Groq LLM key":      bool(GROQ_API_KEY),
}

try:
    from src.football_data_api import info_cobertura
    cob = info_cobertura()
    componentes["CSV datos locales"] = cob.get("partidos", 0) > 0
except Exception:
    componentes["CSV datos locales"] = False

try:
    from src.engine import BettingEngine
    eng = BettingEngine()
    componentes["ML calibrator"] = eng.calibrator is not None
    componentes["df_stats (CSV)"] = eng.df_stats is not None and not eng.df_stats.empty
except Exception:
    componentes["ML calibrator"] = False
    componentes["df_stats (CSV)"] = False

for comp, estado in componentes.items():
    marca = "[OK]" if estado else "[--]"
    nivel = "OK" if estado else "DESHABILITADO"
    info(f"  {marca} {comp:30s} {nivel}")

print()
log.info("Diagnóstico completo finalizado.")
