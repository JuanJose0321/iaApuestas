"""
Flask app de BetBrain.
Usa el motor unificado (Poisson + XGBoost calibrado) para detectar value bets.
El LLM (Groq) es OPCIONAL y solo narra/explica — nunca genera picks.

Mejoras v2:
- Validación de cuotas con errores 400 claros
- Integración real de football_data + coherence + analyst
- Sistema de confianza (filtra picks < 70%)
- Detección de contradicciones en combos
- Liga configurable con promedio de goles correcto
"""
import logging
import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

from config import (
    GROQ_API_KEY, GROQ_MODEL, FLASK_DEBUG, FLASK_PORT,
    BANKROLL_INICIAL, PROMEDIO_GOLES_LIGA,
)
from src.engine import BettingEngine, leg_legible
from src.bankroll import Ledger, stake_recomendado
from src.confidence import (
    calcular_confianza, nivel_confianza,
    verificar_contradicciones_combo,
    UMBRAL_CONFIANZA, UMBRAL_CONFIANZA_SIN_API,
)
from src.data_source_manager import dsm

# ── Logging estructurado ────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BetBrain] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "betbrain.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
_log = logging.getLogger("betbrain")

# ── Startup diagnostics ─────────────────────────────────────────────────
_log.info("=== BetBrain arrancando ===")
_log.info("GROQ_API_KEY: %s", "presente" if GROQ_API_KEY else "AUSENTE — LLM deshabilitado")
_log.info("API_FOOTBALL_KEY: %s", "presente" if os.getenv("API_FOOTBALL_KEY") else "AUSENTE")

app = Flask(__name__)
engine = BettingEngine()
ledger = Ledger()

_log.info("calibrator cargado: %s", engine.calibrator is not None)
_log.info("df_stats cargado: %s", engine.df_stats is not None and not engine.df_stats.empty
          if engine.df_stats is not None else False)

# LLM opcional
llm_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        llm_client = Groq(api_key=GROQ_API_KEY)
        _log.info("Groq LLM client inicializado OK")
    except ImportError:
        _log.warning("groq no instalado - LLM narrador deshabilitado")
else:
    _log.warning("GROQ_API_KEY vacía — narrativa usará fallback de reglas")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _validar_cuotas_1x2(c_1x2: dict) -> str | None:
    """Valida el dict de cuotas 1X2. Devuelve mensaje de error o None."""
    for k in ("1", "X", "2"):
        if k not in c_1x2:
            return f"Falta la cuota '{k}' en 1X2"
        try:
            v = float(c_1x2[k])
        except (TypeError, ValueError):
            return f"Cuota 1X2['{k}'] no es un número válido"
        if v <= 1.0:
            return (f"Cuota 1X2['{k}'] = {v} es inválida "
                    f"(debe ser > 1.0 para tener sentido matemático)")
    return None


def _validar_cuotas_2way(mercado: str, d: dict) -> str | None:
    for k, v in d.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return f"Cuota {mercado}['{k}'] no es un número válido"
        if fv <= 1.0:
            return f"Cuota {mercado}['{k}'] = {fv} es inválida (debe ser > 1.0)"
    return None


def _validar_entrada(data: dict) -> tuple:
    """
    Valida los campos comunes del body JSON.
    Devuelve (home, away, cuotas, liga, promedio, error_msg).
    error_msg es None si todo es válido.
    """
    home  = (data.get("home") or "").strip()
    away  = (data.get("away") or "").strip()
    cuotas = data.get("cuotas") or {}
    liga   = (data.get("liga") or "Default").strip()

    if not home:
        return None, None, None, None, None, "Falta el campo 'home'"
    if not away:
        return None, None, None, None, None, "Falta el campo 'away'"
    if "1X2" not in cuotas:
        return None, None, None, None, None, "Falta el mercado '1X2' en cuotas"

    err = _validar_cuotas_1x2(cuotas["1X2"])
    if err:
        return None, None, None, None, None, err

    if "OU_2.5" in cuotas:
        err = _validar_cuotas_2way("OU_2.5", cuotas["OU_2.5"])
        if err:
            return None, None, None, None, None, err

    if "BTTS" in cuotas:
        err = _validar_cuotas_2way("BTTS", cuotas["BTTS"])
        if err:
            return None, None, None, None, None, err

    promedio = PROMEDIO_GOLES_LIGA.get(liga, PROMEDIO_GOLES_LIGA["Default"])
    return home, away, cuotas, liga, promedio, None


def _ajustar_promedio_con_forma(promedio_base: float, ctx_api: dict) -> float:
    """Ajusta el promedio de goles usando la forma reciente real (blend 70/30)."""
    fh = ctx_api.get("forma_home")
    fa = ctx_api.get("forma_away")
    if not fh or not fa:
        return promedio_base
    xg_real = fh["gf_promedio"] + fa["gf_promedio"]  # suma de GF promedios
    return round(0.70 * promedio_base + 0.30 * xg_real, 2)


# ── Filtro de picks OU_2.5 en zona marginal ─────────────────────────────
_MARGEN_OU_MIN = 0.30       # |xg_total - 2.5| mínimo para considerar el pick fiable
_EV_MARGINAL_MIN = 0.10     # EV mínimo cuando el xG está en zona marginal (con datos API)


def _check_marginal_ou(raw_pick: dict, xg_total: float,
                       tiene_datos_reales: bool) -> str | None:
    """
    Devuelve un motivo de descarte si el pick tiene un leg de OU_2.5 y el xG
    del modelo cae en la zona marginal (|xg - 2.5| < _MARGEN_OU_MIN).

    - Sin datos API reales: siempre descarta (no hay forma real que confirme la tendencia).
    - Con datos API reales: descarta solo si EV < _EV_MARGINAL_MIN (10%).
    Devuelve None si el pick es válido.
    """
    has_ou = any(leg.get("mercado") == "OU_2.5" for leg in raw_pick.get("legs", []))
    if not has_ou:
        return None

    margen = abs(xg_total - 2.5)
    if margen >= _MARGEN_OU_MIN:
        return None  # xG claro: Over o Under con suficiente margen

    if not tiene_datos_reales:
        return (f"marginal_sin_datos (xg_total={xg_total:.2f}, "
                f"margen={margen:.2f} < {_MARGEN_OU_MIN} — sin forma real que confirme tendencia)")

    ev = raw_pick.get("ev", 0)
    if ev < _EV_MARGINAL_MIN:
        return (f"ev_insuficiente_zona_marginal (xg={xg_total:.2f}, margen={margen:.2f} < {_MARGEN_OU_MIN}, "
                f"ev={ev:.1%} < {_EV_MARGINAL_MIN:.0%} requerido en zona dudosa)")

    return None


def _formatear_pick(pick: dict, etiqueta: str, home: str, away: str,
                    bankroll: float, factor_datos: float) -> dict | None:
    if pick is None:
        return None
    legs_txt = [
        {"texto": leg_legible(leg, home, away), "cuota": leg["cuota"]}
        for leg in pick["legs"]
    ]
    # Detección de contradicciones en el combo
    contradicciones = verificar_contradicciones_combo(pick["legs"])

    prob   = pick["prob_conjunta"]
    ev     = pick["ev"]
    stake  = stake_recomendado(bankroll, prob, pick["cuota_total"], "kelly_frac")
    conf   = calcular_confianza(prob, ev, factor_datos)

    return {
        "tipo":              etiqueta,
        "partido":           f"{home} vs {away}",
        "legs":              legs_txt,
        "cuota_total":       pick["cuota_total"],
        "prob":              prob,
        "prob_implicita":    pick["prob_implicita"],
        "ev":                ev,
        "stake_sugerido":    stake,
        "kelly_pct":         pick["kelly_stake_pct"],
        "payout":            round(stake * pick["cuota_total"], 2) if stake else 0,
        "confianza":         conf,
        "confianza_nivel":   nivel_confianza(conf),
        "contradicciones":   contradicciones,
    }


def _narrar_llm(partido: str, cuotas: dict, prob_1x2: dict,
                picks: list, ctx_api: dict, coherencia: dict) -> str | None:
    """Llama al analyst LLM con contexto completo. None si no hay LLM."""
    try:
        from src.analyst import analizar as analizar_llm, fallback_sin_llm
        resultado = analizar_llm(partido, cuotas, prob_1x2,
                                 picks, ctx_api, coherencia)
        if resultado:
            return resultado
        return fallback_sin_llm(coherencia, picks)
    except Exception as e:
        return f"(Analista LLM no disponible: {e})"


# -----------------------------------------------------------------------
# Rutas
# -----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        bankroll=ledger.bankroll_actual(),
        bankroll_inicial=ledger.bankroll_inicial,
    )


@app.route("/analizar", methods=["POST"])
def analizar():
    data = request.get_json(force=True)
    home, away, cuotas, liga, promedio, err = _validar_entrada(data)
    if err:
        return jsonify({"error": err}), 400

    try:
        resultado = engine.analizar(home, away, cuotas,
                                    promedio_goles_liga=promedio)
    except Exception as e:
        return jsonify({"error": f"Engine: {e}"}), 500

    bankroll = ledger.bankroll_actual()
    for vb in resultado["value_bets"]:
        vb["stake_sugerido"] = stake_recomendado(
            bankroll, vb["prob_modelo"], vb["cuota"], "kelly_frac"
        )

    resultado["bankroll_actual"] = bankroll
    resultado["liga"] = liga
    return jsonify(resultado)


@app.route("/pick", methods=["POST"])
def pick():
    data = request.get_json(force=True)
    home, away, cuotas, liga, promedio, err = _validar_entrada(data)
    if err:
        return jsonify({"error": err}), 400

    try:
        res = engine.pick_simple(
            home, away, cuotas,
            cuota_min=float(data.get("cuota_min", 1.70)),
            cuota_max=float(data.get("cuota_max", 2.50)),
            promedio_goles_liga=promedio,
        )
    except Exception as e:
        return jsonify({"error": f"Engine: {e}"}), 500

    bankroll = ledger.bankroll_actual()
    if res["directa"]:
        res["directa"]["stake_sugerido"] = stake_recomendado(
            bankroll, res["directa"]["prob_modelo"],
            res["directa"]["cuota"], "kelly_frac",
        )
    if res["dupla"]:
        res["dupla"]["stake_sugerido"] = stake_recomendado(
            bankroll, res["dupla"]["prob_conjunta"],
            res["dupla"]["cuota_total"], "kelly_frac",
        )
    res["bankroll_actual"] = bankroll
    return jsonify(res)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint principal de la UI tipo chat.

    Body JSON:
    {
      "home":  "Real Madrid",
      "away":  "Barcelona",
      "liga":  "LaLiga",          ← NUEVO (opcional, default "Default")
      "cuotas": {
        "1X2":    {"1": 2.30, "X": 3.40, "2": 3.10},
        "OU_2.5": {"Over": 1.80, "Under": 2.00},
        "BTTS":   {"Yes": 1.75, "No": 2.05}
      }
    }
    """
    data = request.get_json(force=True)
    home, away, cuotas, liga, promedio, err = _validar_entrada(data)
    if err:
        _log.warning("/chat validación fallida: %s", err)
        return jsonify({"error": err}), 400

    _log.info("/chat START partido=%s vs %s liga=%s", home, away, liga)

    # ── 1. Datos reales de la API (degradación elegante si no hay key) ──
    try:
        ctx_api = dsm.contexto_partido_completo(home, away)
        promedio = _ajustar_promedio_con_forma(promedio, ctx_api)
        f_lesiones_home = dsm.factor_ajuste_lesiones(ctx_api.get("injuries_home", []))
        f_lesiones_away = dsm.factor_ajuste_lesiones(ctx_api.get("injuries_away", []))
        _log.info("API-Football: disponible=%s forma_home=%s forma_away=%s h2h=%s notas=%s",
                  ctx_api.get("api_disponible"),
                  ctx_api.get("forma_home") is not None,
                  ctx_api.get("forma_away") is not None,
                  ctx_api.get("h2h") is not None,
                  ctx_api.get("notas", []))
    except Exception as e:
        _log.error("API-Football excepción: %s", e, exc_info=True)
        ctx_api = {"api_disponible": False, "notas": [str(e)],
                   "forma_home": None, "forma_away": None, "h2h": None}
        f_lesiones_home = f_lesiones_away = 1.0

    # ── 2. Motor de picks ──
    try:
        res = engine.pick_multileg(
            home, away, cuotas,
            cuota_min=float(data.get("cuota_min", 1.70)),
            cuota_max=float(data.get("cuota_max", 2.50)),
            cuota_min_tripleta=float(data.get("cuota_min_tripleta", 2.50)),
            cuota_max_tripleta=float(data.get("cuota_max_tripleta", 6.00)),
            promedio_goles_liga=promedio,
        )
        combos_uniformes = res.get("combos_bloqueados_uniforme", [])
        _log.info("Engine: fuente=%s lambdas=%s directa=%s dupla=%s tripleta=%s combos_uniformes=%d",
                  res.get("fuente_1x2"),
                  res.get("lambdas"),
                  res.get("directa") is not None,
                  res.get("dupla") is not None,
                  res.get("tripleta") is not None,
                  len(combos_uniformes))
        if combos_uniformes:
            _log.warning("Combos bloqueados por uniformidad: %s", combos_uniformes)
    except Exception as e:
        _log.error("Engine excepción: %s", e, exc_info=True)
        return jsonify({"error": f"Engine: {e}"}), 500

    # ── 3. Coherencia modelo ↔ mercado ──
    try:
        from src.coherence import evaluar_coherencia
        coherencia = evaluar_coherencia(
            cuotas, res["lambdas"],
            forma_home=ctx_api.get("forma_home"),
            forma_away=ctx_api.get("forma_away"),
            h2h=ctx_api.get("h2h"),
        )
        _log.info("Coherencia: confianza=%s flags=%s",
                  coherencia.get("confianza_modelo"), coherencia.get("flags"))
    except Exception as exc:
        _log.error("Coherencia excepción: %s", exc, exc_info=True)
        coherencia = {"flags": [], "mensajes": [], "market_xg_total": None,
                      "model_xg_total": 0, "confianza_modelo": "media"}

    # ── 4. Factor de datos: si tenemos forma real, el modelo es más fiable ──
    tiene_datos_reales = (
        ctx_api.get("api_disponible") and
        (ctx_api.get("forma_home") is not None or
         ctx_api.get("forma_away") is not None)
    )
    factor_datos = 1.0 if tiene_datos_reales else 0.6
    umbral = UMBRAL_CONFIANZA if tiene_datos_reales else UMBRAL_CONFIANZA_SIN_API
    _log.info("factor_datos=%.1f umbral=%.2f (datos_reales=%s)",
              factor_datos, umbral, tiene_datos_reales)

    bankroll = ledger.bankroll_actual()

    # ── 5. Formatear y filtrar picks por confianza ──
    picks_totales = []
    picks_filtrados = []
    picks_descartados = []
    xg_total = res["lambdas"]["home"] + res["lambdas"]["away"]

    for etiqueta, key in [("DIRECTA", "directa"),
                           ("DUPLA (x2)", "dupla"),
                           ("TRIPLETA (x3)", "tripleta")]:
        p = res.get(key)
        if p is None:
            _log.info("Pick %s: engine devolvió None (sin candidatos en rango/EV)", key)
            continue

        # ── Filtro OU_2.5 marginal (antes de formatear) ──
        motivo_marginal = _check_marginal_ou(p, xg_total, tiene_datos_reales)
        if motivo_marginal:
            _log.warning("Pick %s DESCARTADO por OU marginal: %s", etiqueta, motivo_marginal)
            picks_descartados.append({
                "tipo":    etiqueta,
                "motivo":  "marginal_ou",
                "detalle": motivo_marginal,
            })
            continue

        fmt = _formatear_pick(p, etiqueta, home, away, bankroll, factor_datos)
        if fmt is None:
            continue
        picks_totales.append(fmt)
        _log.info("Pick %s: prob=%.3f ev=%.4f confianza=%.4f umbral=%.2f contradicciones=%s",
                  etiqueta, fmt["prob"], fmt["ev"], fmt["confianza"],
                  umbral, fmt["contradicciones"])
        if fmt["contradicciones"]:
            _log.warning("Pick %s DESCARTADO por contradicciones: %s",
                         etiqueta, fmt["contradicciones"])
            picks_descartados.append({
                "tipo": etiqueta,
                "motivo": "contradiccion",
                "detalle": fmt["contradicciones"],
                "confianza": fmt["confianza"],
                "umbral": umbral,
            })
            continue
        if fmt["confianza"] >= umbral:
            picks_filtrados.append(fmt)
        else:
            _log.warning("Pick %s DESCARTADO por confianza baja: %.4f < %.2f",
                         etiqueta, fmt["confianza"], umbral)
            picks_descartados.append({
                "tipo": etiqueta,
                "motivo": "confianza_baja",
                "detalle": (f"confianza={fmt['confianza']:.4f} < umbral={umbral:.2f} "
                            f"({'con' if tiene_datos_reales else 'sin'} datos API)"),
                "confianza": fmt["confianza"],
                "umbral": umbral,
            })

    _log.info("Picks totales=%d filtrados=%d descartados=%d",
              len(picks_totales), len(picks_filtrados), len(picks_descartados))

    # ── 6. Narrativa LLM ──
    _llm_disponible = llm_client is not None
    narrativa = _narrar_llm(
        f"{home} vs {away}",
        cuotas,
        res.get("prob_1x2_final", {}),
        picks_filtrados,
        ctx_api,
        coherencia,
    )
    _log.info("Narrativa: fuente=%s largo=%d chars",
              "LLM" if _llm_disponible else "fallback_reglas",
              len(narrativa) if narrativa else 0)

    # ── 7. Mensaje principal ──
    if not picks_filtrados:
        if picks_totales:
            motivos = "; ".join(
                f"{d['tipo']}: {d['detalle']}" for d in picks_descartados
            )
            mensaje = (
                f"Análisis de **{home} vs {away}** listo.\n\n"
                f"El motor encontró {len(picks_totales)} pick(s) con EV positivo, "
                f"pero ninguno superó el umbral de confianza del "
                f"{int(umbral*100)}% "
                f"({'con datos API' if tiene_datos_reales else 'sin datos API — umbral reducido a ' + str(int(umbral*100)) + '%'}).\n\n"
                f"Motivos de descarte: {motivos}\n\n"
                f"Hoy no hay picks de valor suficiente — "
                f"mejor esperar otro partido que forzar."
            )
        else:
            mensaje = (
                f"Análisis de **{home} vs {away}** listo.\n\n"
                f"El modelo no encontró value bets (EV ≥ 5%) en estas cuotas. "
                f"La casa parece tener precios eficientes para este partido. "
                f"Prueba con otro partido o con cuotas de otra casa."
            )
    else:
        fuente_txt = res["fuente_1x2"]
        promedio_txt = f"{promedio:.1f} goles/partido"
        liga_txt = f" ({liga})" if liga != "Default" else ""
        mensaje = (
            f"**{home} vs {away}**\n\n"
            f"xG estimado: {res['lambdas']['home']} – {res['lambdas']['away']} "
            f"(fuente: {fuente_txt}, liga avg: {promedio_txt}{liga_txt})\n\n"
            f"**{len(picks_filtrados)} recomendación(es)** con EV positivo y "
            f"confianza ≥ {int(umbral*100)}%. "
            f"Stake calculado con Kelly fraccional sobre bankroll de {bankroll:.2f}."
        )

    _log.info("/chat END partido=%s vs %s picks_enviados=%d narrativa=%s",
              home, away, len(picks_filtrados), "sí" if narrativa else "vacía")

    return jsonify({
        "partido":        f"{home} vs {away}",
        "liga":           liga,
        "fuente":         res["fuente_1x2"],
        "lambdas":        res["lambdas"],
        "picks":          picks_filtrados,
        "mensaje":        mensaje,
        "narrativa":      narrativa,
        "coherencia":     coherencia,
        "contexto_api":   ctx_api,
        "bankroll":       bankroll,
        "debug_filtrado": {
            "factor_datos":      factor_datos,
            "umbral_usado":      umbral,
            "datos_api_reales":  tiene_datos_reales,
            "xg_total":          round(xg_total, 2),
            "margen_ou":         round(abs(xg_total - 2.5), 2),
            "picks_totales":     len(picks_totales),
            "picks_pasaron":     len(picks_filtrados),
            "descartados":       picks_descartados,
            "combos_descartados": combos_uniformes,
        },
    })


@app.route("/api/teams")
def api_teams():
    """
    Devuelve la lista de equipos de una liga, para poblar el dropdown del UI.

    Uso:  GET /api/teams?liga=Liga%20MX
    Si no se pasa ?liga, devuelve el dict completo {liga: [equipos...]}.

    Fuente: data/equipos_por_liga.json (editable a mano).
    """
    import json
    ruta = Path(__file__).parent / "data" / "equipos_por_liga.json"
    if not ruta.exists():
        return jsonify({"error": "equipos_por_liga.json no encontrado",
                        "ligas": {}}), 500
    try:
        with ruta.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        _log.error("Error leyendo equipos_por_liga.json: %s", e)
        return jsonify({"error": str(e), "ligas": {}}), 500

    # Quitar la entrada _meta si la pide como dict completo
    ligas_dict = {k: v for k, v in data.items() if not k.startswith("_")}

    liga = (request.args.get("liga") or "").strip()
    if liga:
        equipos = ligas_dict.get(liga)
        if equipos is None:
            return jsonify({"liga": liga, "equipos": [],
                            "error": f"Liga '{liga}' no tiene equipos registrados"}), 404
        return jsonify({"liga": liga, "equipos": equipos, "total": len(equipos)})

    return jsonify({"ligas": ligas_dict, "total_ligas": len(ligas_dict)})


@app.route("/api/fixtures-today")
def fixtures_today():
    """
    Devuelve los partidos del día de las ligas principales,
    agrupados por liga, listos para mostrar en la UI.
    Parámetro opcional: ?liga=Premier+League para filtrar.
    """
    try:
        todos = dsm.get_fixtures_today()
    except Exception as e:
        return jsonify({"error": f"API-Football: {e}", "partidos": []}), 500

    ligas_top = {
        "La Liga", "Premier League", "Bundesliga", "Serie A", "Ligue 1",
        "Champions League", "Europa League", "Eredivisie",
        "Primeira Liga", "Championship", "Liga MX", "MLS",
    }
    filtro_liga = request.args.get("liga", "").strip()

    partidos_limpios = []
    for p in todos:
        liga_nombre = p.get("league", {}).get("name", "")
        if filtro_liga and liga_nombre != filtro_liga:
            continue
        if not filtro_liga and liga_nombre not in ligas_top:
            continue

        teams   = p.get("teams",   {})
        fixture = p.get("fixture", {})
        goals   = p.get("goals",   {})
        status  = fixture.get("status", {})

        partidos_limpios.append({
            "fixture_id":  fixture.get("id"),
            "fecha":       fixture.get("date", "")[:16],
            "estado":      status.get("long", ""),
            "estado_corto": status.get("short", ""),
            "liga":        liga_nombre,
            "pais":        p.get("league", {}).get("country", ""),
            "local":       teams.get("home", {}).get("name", ""),
            "visitante":   teams.get("away", {}).get("name", ""),
            "goles_local": goals.get("home"),
            "goles_visit": goals.get("away"),
        })

    # Ordenar: primero por estado (en curso primero), luego por hora
    orden_estado = {"1H": 0, "HT": 1, "2H": 2, "ET": 3, "NS": 4, "FT": 5}
    partidos_limpios.sort(key=lambda x: (
        orden_estado.get(x["estado_corto"], 9),
        x["fecha"]
    ))

    # Agrupar por liga
    por_liga: dict = {}
    for p in partidos_limpios:
        liga = p["liga"]
        por_liga.setdefault(liga, []).append(p)

    return jsonify({
        "fecha":         __import__("datetime").date.today().isoformat(),
        "total":         len(partidos_limpios),
        "por_liga":      por_liga,
        "partidos":      partidos_limpios,
    })


@app.route("/registrar", methods=["POST"])
def registrar():
    """Guarda una apuesta en el ledger."""
    d = request.get_json(force=True)
    required = ["partido", "mercado", "seleccion", "cuota", "stake",
                "prob_modelo", "ev"]
    for field in required:
        if field not in d:
            return jsonify({"error": f"Falta campo '{field}'"}), 400

    try:
        cuota = float(d["cuota"])
        stake = float(d["stake"])
        prob  = float(d["prob_modelo"])
        ev    = float(d["ev"])
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Valor numérico inválido: {e}"}), 400

    if cuota <= 1.0:
        return jsonify({"error": "cuota debe ser > 1.0"}), 400
    if stake < 0:
        return jsonify({"error": "stake no puede ser negativo"}), 400

    a = ledger.registrar(
        partido=str(d["partido"]), mercado=str(d["mercado"]),
        seleccion=str(d["seleccion"]),
        cuota=cuota, stake=stake, prob_modelo=prob, ev=ev,
    )
    return jsonify({"ok": True, "apuesta": a.__dict__,
                    "resumen": ledger.resumen()})


@app.route("/ledger")
def ver_ledger():
    return jsonify({
        "resumen":  ledger.resumen(),
        "apuestas": [a.__dict__ for a in ledger.apuestas],
    })


@app.route("/liquidar", methods=["POST"])
def liquidar():
    d = request.get_json(force=True)
    try:
        indice = int(d["indice"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Campo 'indice' requerido y debe ser entero"}), 400

    resultado = d.get("resultado", "")
    if resultado not in ("ganada", "perdida", "push", "pendiente"):
        return jsonify({"error": "resultado debe ser: ganada | perdida | push | pendiente"}), 400

    if indice < 0 or indice >= len(ledger.apuestas):
        return jsonify({"error": f"Índice {indice} fuera de rango"}), 400

    a = ledger.liquidar(indice, resultado)
    return jsonify({"ok": True, "apuesta": a.__dict__,
                    "resumen": ledger.resumen()})


@app.route("/historial")
def historial():
    return render_template("historial.html")


@app.route("/api/registrar_apuesta", methods=["POST"])
def api_registrar_apuesta():
    """Registra una apuesta en el CSV de tracking."""
    from src.tracking import registrar_apuesta
    d = request.get_json(force=True)
    required = ["liga", "local", "visitante", "pick_tipo", "pick_descripcion", "cuota", "stake"]
    for f in required:
        if f not in d:
            return jsonify({"error": f"Falta campo '{f}'"}), 400
    try:
        cuota = float(d["cuota"])
        stake = float(d["stake"])
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Valor numérico inválido: {e}"}), 400
    if cuota <= 1.0:
        return jsonify({"error": "cuota debe ser > 1.0"}), 400
    if stake < 0:
        return jsonify({"error": "stake no puede ser negativo"}), 400
    return jsonify(registrar_apuesta(d))


@app.route("/api/historial")
def api_historial():
    """Devuelve el historial de apuestas con filtros opcionales."""
    from src.tracking import leer_historial
    rows = leer_historial()
    liga      = request.args.get("liga", "").strip()
    resultado = request.args.get("resultado", "").strip()
    fecha_desde = request.args.get("fecha_desde", "").strip()
    if liga:
        rows = [r for r in rows if r.get("liga") == liga]
    if resultado:
        rows = [r for r in rows if r.get("resultado") == resultado]
    if fecha_desde:
        rows = [r for r in rows if r.get("fecha_partido", "") >= fecha_desde]
    return jsonify({"apuestas": rows, "total": len(rows)})


@app.route("/api/actualizar_resultado", methods=["POST"])
def api_actualizar_resultado():
    """Actualiza el resultado de una apuesta y recalcula el bankroll."""
    from src.tracking import actualizar_resultado
    d = request.get_json(force=True)
    try:
        id_apuesta = int(d["id"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Campo 'id' requerido y debe ser entero"}), 400
    resultado = d.get("resultado", "")
    if resultado not in ("ganada", "perdida", "void", "cashout", "pendiente"):
        return jsonify({"error": "resultado debe ser: ganada | perdida | void | cashout | pendiente"}), 400
    try:
        return jsonify(actualizar_resultado(id_apuesta, resultado, d.get("notas", "")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/metricas")
def api_metricas():
    """Devuelve métricas completas del historial."""
    from src.tracking import calcular_metricas
    return jsonify(calcular_metricas())


@app.route("/api/verificar_resultados")
def api_verificar_resultados():
    """
    Revisa apuestas pendientes con fixture_id_api y actualiza resultados
    automáticamente consultando la API de football.
    """
    from src.tracking import leer_historial, actualizar_resultado, determinar_resultado_pick
    rows       = leer_historial()
    pendientes = [r for r in rows if r.get("resultado") == "pendiente"
                  and r.get("fixture_id_api")]
    actualizadas = 0
    errores = []

    for r in pendientes:
        fid = r["fixture_id_api"]
        try:
            res_api = dsm.get_fixture_result(int(fid))
            if not res_api or not res_api.get("terminado"):
                continue
            res = determinar_resultado_pick(
                r.get("pick_descripcion", ""),
                r.get("local", ""),
                r.get("visitante", ""),
                int(res_api["goles_local"]     or 0),
                int(res_api["goles_visitante"] or 0),
            )
            if res == "unknown":
                continue
            actualizar_resultado(int(r["id"]), res,
                                  notas=f"Auto: {res_api['status']}")
            actualizadas += 1
        except Exception as e:
            errores.append(f"fixture {fid}: {e}")

    return jsonify({
        "pendientes_revisadas": len(pendientes),
        "actualizadas":         actualizadas,
        "errores":              errores,
    })


@app.route("/api/dsm/stats")
def dsm_stats():
    """Estadísticas de uso del DataSourceManager (por fuente, coalesced, errores)."""
    return jsonify({
        "dsm_stats":   dsm.stats(),
        "csv_cobertura": dsm.csv_info(),
        "fuente_default": dsm.fuente_default,
    })


@app.route("/api/dsm/fuente", methods=["POST"])
def dsm_set_fuente():
    """
    Cambia la fuente de datos en caliente sin reiniciar el servidor.
    Body: {"fuente": "auto" | "api-football" | "football-data" | "merged"}
    """
    data = request.get_json(force=True) or {}
    nueva = data.get("fuente", "").strip()
    validas = {"auto", "api-football", "football-data", "merged"}
    if nueva not in validas:
        return jsonify({"error": f"fuente inválida. Usa: {sorted(validas)}"}), 400
    anterior = dsm.fuente_default
    dsm.fuente_default = nueva  # type: ignore[assignment]
    _log.info("DSM fuente cambiada: %s → %s", anterior, nueva)
    return jsonify({"anterior": anterior, "nueva": nueva})


if __name__ == "__main__":
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT)
