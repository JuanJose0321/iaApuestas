"""
BetBrain — Flask app.

Routes
------
GET  /                          chat UI
GET  /historial                 historial UI
POST /chat                      analizar partido de fútbol
GET  /api/teams                 equipos por liga
POST /api/registrar_apuesta     registrar apuesta
GET  /api/metricas              métricas
GET  /api/historial             historial filtrado
POST /api/actualizar_resultado  actualizar resultado
GET  /api/verificar_resultados  verificar pendientes
POST /api/analizar_tenis        analizar partido de tenis
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FLASK_DEBUG, FLASK_PORT, SECRET_KEY, BANKROLL_INICIAL, PROMEDIO_GOLES_LIGA,
    LOGS_DIR, ensure_dirs,
)

ensure_dirs()

_handlers: list[logging.Handler] = [logging.StreamHandler()]
try:
    _handlers.append(
        RotatingFileHandler(LOGS_DIR / "betbrain.log", maxBytes=5 * 1024 * 1024,
                            backupCount=3, encoding="utf-8")
    )
except OSError:
    pass  # filesystem de solo lectura (Vercel/serverless): log solo a stdout/stderr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=_handlers,
)
_log = logging.getLogger("betbrain.app")
from src.core.confidence import (
    calcular_confianza, nivel_confianza, verificar_contradicciones_combo,
    UMBRAL_VERDE, UMBRAL_AMARILLO,
)
from src.services.tracking import (
    registrar_apuesta as _registrar,
    actualizar_resultado as _actualizar,
    calcular_metricas,
    leer_historial,
    leer_config,
)
from src.providers.league_manager import get_teams
from src.providers.player_manager import get_players as get_tennis_players
from src.engines.tennis_improved import TennisImprovedEngine
from src.engines.tennis_validator import validar_entrada_tenis

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB: los payloads de /chat son pequeños

limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")


@limiter.request_filter
def _sin_limite_en_tests() -> bool:
    """Exime de rate limiting cuando app.config['TESTING'] está activo."""
    return app.testing

# Football engine: lazy (XGBoost + CSV son pesados)
_football_engine = None
# Tennis engine: mejorado con Elo calibrado
_tennis_engine = None

def _get_tennis_engine():
    """Carga motor de tenis con Elo calibrado."""
    global _tennis_engine
    if _tennis_engine is None:
        import json
        elo_path = Path(__file__).parent / "src" / "data" / "tennis_elo_ratings.json"

        elo_ratings = {}
        form_stats = {}

        # Cargar Elo ratings si existen
        if elo_path.exists():
            try:
                with open(elo_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for player_name, stats in data.get("jugadores", {}).items():
                        elo_ratings[player_name] = stats.get("elo", 1500.0)
                _log.info(f"✅ Cargados {len(elo_ratings)} Elo ratings para tenis")
            except Exception as e:
                _log.warning(f"⚠️ Error cargando Elo ratings: {e}")

        _tennis_engine = TennisImprovedEngine(elo_ratings=elo_ratings, form_stats=form_stats)

    return _tennis_engine


def _get_football_engine():
    global _football_engine
    if _football_engine is None:
        from src.engines.football import BettingEngine
        _football_engine = BettingEngine()
    return _football_engine


# ── Helpers de picks de fútbol ────────────────────────────────────────────────

def _transformar_pick(raw: dict, home: str, away: str,
                      bankroll: float, factor_datos: float) -> dict:
    """Formato motor → formato UI del template."""
    from src.engines.football import leg_legible
    legs_ui = [
        {"texto": leg_legible(leg, home, away),
         "cuota": leg["cuota"],
         "mercado": leg["mercado"]}
        for leg in raw["legs"]
    ]
    prob      = raw["prob_conjunta"]
    ev        = raw["ev"]
    confianza = calcular_confianza(prob, ev, factor_datos)
    contras   = verificar_contradicciones_combo(raw["legs"])
    stake     = round(bankroll * raw["kelly_stake_pct"], 2)
    return {
        "tipo":            raw["tipo"].upper(),
        "partido":         f"{home} vs {away}",
        "legs":            legs_ui,
        "cuota_total":     raw["cuota_total"],
        "prob":            prob,
        "ev":              ev,
        "stake_sugerido":  stake,
        "confianza":       confianza,
        "confianza_nivel": nivel_confianza(confianza),
        "contradicciones": contras,
    }


def _clasificar_picks(picks_raw: list, home: str, away: str,
                      bankroll: float, factor_datos: float):
    verdes, amarillos, descartados = [], [], []
    for raw in picks_raw:
        if raw is None:
            continue
        p    = _transformar_pick(raw, home, away, bankroll, factor_datos)
        conf = p["confianza"]
        if conf >= UMBRAL_VERDE:
            verdes.append(p)
        elif conf >= UMBRAL_AMARILLO:
            amarillos.append(p)
        else:
            descartados.append({
                **p,
                "umbral":  UMBRAL_VERDE,
                "motivo":  "contradiccion" if p["contradicciones"] else "confianza_baja",
                "detalle": f"Confianza {conf:.0%} < umbral {UMBRAL_VERDE:.0%}. EV={p['ev']:+.1%}",
            })
    return verdes, amarillos, descartados


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    cfg      = leer_config()
    bankroll = float(cfg.get("bankroll_actual", BANKROLL_INICIAL))
    return render_template("index.html", bankroll=bankroll)


@app.route("/historial")
def historial():
    return render_template("historial.html")


# ── Fútbol ────────────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
@limiter.limit("20 per minute")
def chat():
    data   = request.get_json(silent=True) or {}
    home   = (data.get("home") or "").strip()
    away   = (data.get("away") or "").strip()
    liga   = data.get("liga", "Default")
    cuotas = data.get("cuotas") or {}

    if not home:
        return jsonify({"error": "Falta el equipo local"}), 400
    if not away:
        return jsonify({"error": "Falta el equipo visitante"}), 400
    if not cuotas.get("1X2"):
        return jsonify({"error": "Faltan cuotas 1X2"}), 400

    try:
        c1x2 = cuotas["1X2"]
        for k in ("1", "X", "2"):
            if float(c1x2[k]) <= 1.0:
                raise ValueError(f"Cuota {k} debe ser > 1.0")
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Cuotas 1X2 inválidas: {exc}"}), 400

    cfg      = leer_config()
    bankroll = float(cfg.get("bankroll_actual", BANKROLL_INICIAL))
    promedio = PROMEDIO_GOLES_LIGA.get(liga, PROMEDIO_GOLES_LIGA["Default"])

    # Contexto API (falla silenciosa para el usuario, pero logueada)
    contexto_api: dict = {"api_disponible": False}
    try:
        from src.providers.manager import dsm
        ctx = dsm.contexto_partido_completo(home, away)
        if ctx:
            contexto_api = ctx
    except Exception as exc:
        _log.warning("contexto_api no disponible: %s", exc, exc_info=True)

    tiene_datos_reales = bool(
        contexto_api.get("api_disponible") and
        (contexto_api.get("forma_home") is not None or
         contexto_api.get("forma_away") is not None)
    )
    factor_datos = 1.0 if contexto_api.get("api_disponible") else 0.6

    # Motor de fútbol
    try:
        from src.engines.football import check_marginal_ou
        eng    = _get_football_engine()
        result = eng.pick_multileg(home, away, cuotas,
                                   promedio_goles_liga=promedio)
    except Exception as exc:
        _log.exception("Error en pick_multileg")
        return jsonify({"error": f"Error del motor: {type(exc).__name__}: {exc}"}), 500

    lambdas  = result.get("lambdas", {"home": 0, "away": 0})
    xg_total = lambdas["home"] + lambdas["away"]
    margen_ou = abs(xg_total - 2.5)

    picks_raw = []
    for key in ("directa", "dupla", "tripleta"):
        if result.get(key):
            picks_raw.append(result[key])
    for key in ("directa_alt", "dupla_alt", "tripleta_alt"):
        picks_raw.extend(p for p in (result.get(key) or []) if p)

    # Filtro de zona marginal OU_2.5: descarta antes de clasificar por confianza
    picks_validos = []
    descartados_marginal = []
    for raw in picks_raw:
        motivo = check_marginal_ou(raw, xg_total, tiene_datos_reales)
        if motivo:
            descartados_marginal.append({
                "tipo":    raw["tipo"].upper(),
                "motivo":  "marginal_ou",
                "detalle": motivo,
            })
        else:
            picks_validos.append(raw)

    verdes, amarillos, descartados = _clasificar_picks(
        picks_validos, home, away, bankroll, factor_datos
    )
    descartados = descartados_marginal + descartados

    # Coherencia (falla silenciosa, pero logueada)
    coherencia: dict = {}
    try:
        from src.core.coherence import evaluar_coherencia
        coherencia = evaluar_coherencia(
            cuotas, lambdas,
            forma_home=contexto_api.get("forma_home"),
            forma_away=contexto_api.get("forma_away"),
            h2h=contexto_api.get("h2h"),
        )
    except Exception as exc:
        _log.warning("evaluar_coherencia falló: %s", exc, exc_info=True)

    # Narrativa LLM (falla silenciosa para el usuario, pero logueada)
    narrativa = ""
    todos     = verdes + amarillos
    if todos and coherencia:
        try:
            from src.services.analyst import analizar as _llm, fallback_sin_llm
            narrativa = (
                _llm(f"{home} vs {away}", cuotas,
                     result.get("prob_1x2_final", {}),
                     todos, contexto_api, coherencia)
                or fallback_sin_llm(coherencia, todos)
            )
        except Exception as exc:
            _log.warning("Narrativa LLM falló: %s", exc, exc_info=True)

    n       = len(verdes) + len(amarillos)
    mensaje = (
        f"Encontré {n} pick(s) con valor para {home} vs {away} "
        f"({result.get('fuente_1x2', 'Poisson')})."
        if n else
        f"Sin picks con valor suficiente para {home} vs {away}."
    )

    return jsonify({
        "mensaje":         mensaje,
        "picks_verdes":    verdes,
        "picks_amarillos": amarillos,
        "debug_filtrado":  {
            "descartados":        descartados,
            "xg_total":           round(xg_total, 2),
            "margen_ou":          round(margen_ou, 2),
            "combos_descartados": result.get("combos_bloqueados_uniforme", []),
            "factor_datos":       factor_datos,
            "datos_api_reales":   tiene_datos_reales,
        },
        "contexto_api":    contexto_api,
        "narrativa":       narrativa,
        "bankroll":        bankroll,
    })


@app.route("/api/teams")
def api_teams():
    liga = request.args.get("liga", "").strip()
    if not liga or liga == "Default":
        return jsonify({"error": "Liga no especificada"})
    try:
        equipos = get_teams(liga)
        if not equipos:
            return jsonify({"error": f"Sin equipos para '{liga}'"})
        return jsonify({"liga": liga, "equipos": sorted(equipos)})
    except Exception as exc:
        return jsonify({"error": str(exc)})


@app.route("/api/players")
def api_players():
    genero = request.args.get("genero", "").strip()
    if not genero or genero == "Default":
        return jsonify({"error": "Género no especificado"})
    try:
        jugadores = get_tennis_players(genero)
        if not jugadores:
            return jsonify({"error": f"Sin jugadores para '{genero}'"})
        return jsonify({"genero": genero, "jugadores": sorted(jugadores)})
    except Exception as exc:
        return jsonify({"error": str(exc)})


# ── Tracking ──────────────────────────────────────────────────────────────────

@app.route("/api/registrar_apuesta", methods=["POST"])
@limiter.limit("30 per minute")
def api_registrar_apuesta():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(_registrar(data))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/metricas")
def api_metricas():
    try:
        return jsonify(calcular_metricas())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/historial")
def api_historial():
    resultado = request.args.get("resultado", "")
    liga      = request.args.get("liga", "")
    desde     = request.args.get("fecha_desde", "")
    try:
        rows = leer_historial()
        if resultado:
            rows = [r for r in rows if r.get("resultado") == resultado]
        if liga:
            rows = [r for r in rows if r.get("liga") == liga]
        if desde:
            rows = [r for r in rows if (r.get("fecha_registro") or "") >= desde]
        return jsonify({"apuestas": rows})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/actualizar_resultado", methods=["POST"])
def api_actualizar_resultado():
    data      = request.get_json(silent=True) or {}
    id_ap     = data.get("id")
    resultado = (data.get("resultado") or "").strip()
    if not id_ap:
        return jsonify({"error": "Falta id"}), 400
    if not resultado:
        return jsonify({"error": "Falta resultado"}), 400
    try:
        return jsonify(_actualizar(int(id_ap), resultado))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/verificar_resultados")
def api_verificar_resultados():
    try:
        rows       = leer_historial()
        pendientes = [r for r in rows if r.get("resultado") == "pendiente"]
        return jsonify({
            "pendientes_revisadas": len(pendientes),
            "actualizadas":         0,
            "nota": "Actualización automática no disponible. Usa el historial.",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Tenis ─────────────────────────────────────────────────────────────────────

@app.route("/api/analizar_tenis", methods=["POST"])
@limiter.limit("20 per minute")
def api_analizar_tenis():
    """
    Analiza un partido de tenis y devuelve picks clasificados.

    Body JSON:
      "jugador1": "Carlos Alcaraz", "jugador2": "Jannik Sinner",
      "elo1": 1650, "elo2": 1680, "superficie": "hard", "formato": "best_of_3",
      "cuotas": {"match_winner": {"1": 1.85, "2": 1.95}, "total_games": {"linea": 22.5, "over": 1.85, "under": 1.95}}
    """
    data = request.get_json(silent=True) or {}

    j1, j2, elo1, elo2, superficie, formato, cuotas, err = validar_entrada_tenis(data)
    if err:
        return jsonify({"error": err}), 400

    cuota_min = float(data.get("cuota_min") or 1.20)
    cuota_max = float(data.get("cuota_max") or 6.00)

    cfg      = leer_config()
    bankroll = float(cfg.get("bankroll_actual", BANKROLL_INICIAL))

    try:
        engine = _get_tennis_engine()
        resultado = engine.analizar(
            j1, j2, elo1, elo2, superficie, formato, cuotas,
            cuota_min=cuota_min, cuota_max=cuota_max,
        )
    except Exception as exc:
        _log.exception("Error en TennisEngine.analizar")
        return jsonify({"error": f"Error del motor de tenis: {exc}"}), 500

    # Añadir stake_sugerido
    for pick in (resultado["picks_verdes"] + resultado["picks_amarillos"]):
        pick["stake_sugerido"] = engine.stake_recomendado(
            bankroll, pick["prob"], pick["cuota"]
        )

    resultado["bankroll"] = bankroll
    return jsonify(resultado)


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Ruta no encontrada"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Error interno del servidor"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _log.info("=== BetBrain arrancando en puerto %d ===", FLASK_PORT)
    app.run(host="127.0.0.1", port=FLASK_PORT, debug=FLASK_DEBUG, use_reloader=False)
