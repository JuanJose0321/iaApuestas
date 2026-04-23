"""
Cliente API-Football (api-sports.io) con:
  - Caché en memoria (10 min TTL) para ahorrar cuota diaria
  - Reintentos con backoff en errores 429/5xx
  - Log de cada llamada a api_calls.log
  - Degradación elegante: devuelve None si no hay key o falla la API

Free tier: 100 req/día.
Headers para acceso directo (api-sports.io):
  x-apisports-key: TU_KEY

NOTA: football_data.py usa RapidAPI headers (deprecated).
      Este módulo usa la API directa de api-sports.io.
"""
import hashlib
import json
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import API_FOOTBALL_KEY, API_FOOTBALL_HOST, CACHE_DIR, CACHE_TTL_HORAS

# -----------------------------------------------------------------------
# Logging a archivo
# -----------------------------------------------------------------------
_log_path = Path(__file__).resolve().parent.parent / "api_calls.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [API-Football] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(_log_path, encoding="utf-8"),
    ],
)
_log = logging.getLogger("api_football")

BASE_URL = f"https://{API_FOOTBALL_HOST}"

# Headers para API directa de api-sports.io
def _headers():
    return {"x-apisports-key": API_FOOTBALL_KEY or ""}


# -----------------------------------------------------------------------
# Caché en MEMORIA (10 min TTL) — para llamadas repetidas en misma sesión
# -----------------------------------------------------------------------
_mem_cache: dict = {}
_MEM_TTL = 600  # 10 minutos


def _mem_get(key: str):
    entry = _mem_cache.get(key)
    if entry and (time.time() - entry["ts"]) < _MEM_TTL:
        return entry["data"]
    return None


def _mem_set(key: str, data):
    _mem_cache[key] = {"ts": time.time(), "data": data}


# -----------------------------------------------------------------------
# Caché en DISCO (CACHE_TTL_HORAS) — persiste entre reinicios del servidor
# -----------------------------------------------------------------------
def _disk_key(endpoint: str, params: dict) -> Path:
    raw = json.dumps({"e": endpoint, "p": params}, sort_keys=True)
    h = hashlib.md5(raw.encode()).hexdigest()
    return CACHE_DIR / f"af_{h}.json"


def _disk_get(endpoint: str, params: dict):
    p = _disk_key(endpoint, params)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        edad_h = (time.time() - raw.get("_ts", 0)) / 3600
        if edad_h > CACHE_TTL_HORAS:
            return None
        return raw["data"]
    except Exception:
        return None


def _disk_set(endpoint: str, params: dict, data):
    p = _disk_key(endpoint, params)
    try:
        p.write_text(json.dumps({"_ts": time.time(), "data": data}), encoding="utf-8")
    except Exception:
        pass


# -----------------------------------------------------------------------
# HTTP con reintentos
# -----------------------------------------------------------------------
def _get(endpoint: str, params: dict | None = None) -> dict | None:
    if not API_FOOTBALL_KEY:
        _log.warning("API_FOOTBALL_KEY no configurada — sin datos reales")
        return None

    params = params or {}
    cache_key = json.dumps({"e": endpoint, "p": params}, sort_keys=True)

    # 1. Caché memoria
    cached = _mem_get(cache_key)
    if cached is not None:
        _log.debug("cache-mem HIT %s %s", endpoint, params)
        return cached

    # 2. Caché disco
    cached = _disk_get(endpoint, params)
    if cached is not None:
        _log.debug("cache-disk HIT %s %s", endpoint, params)
        _mem_set(cache_key, cached)
        return cached

    # 3. Llamada real a la API con reintentos
    url = f"{BASE_URL}/{endpoint}"
    for intento in range(3):
        try:
            _log.info("GET /%s %s (intento %d)", endpoint, params, intento + 1)
            r = requests.get(url, headers=_headers(), params=params, timeout=15)

            if r.status_code == 429:
                wait = 2 ** (intento + 1)
                _log.warning("Rate limit (429) — esperando %ds", wait)
                time.sleep(wait)
                continue

            if r.status_code >= 500:
                _log.error("Error servidor %d en %s", r.status_code, endpoint)
                time.sleep(1)
                continue

            if r.status_code == 404:
                _log.warning("404 en %s %s", endpoint, params)
                return None

            if r.status_code != 200:
                _log.error("HTTP %d en %s", r.status_code, endpoint)
                return None

            data = r.json()
            _mem_set(cache_key, data)
            _disk_set(endpoint, params, data)
            _log.info("OK /%s → %d resultados", endpoint,
                      len(data.get("response", [])))
            return data

        except requests.exceptions.Timeout:
            _log.warning("Timeout en %s (intento %d)", endpoint, intento + 1)
            time.sleep(1)
        except Exception as exc:
            _log.error("Excepción en %s: %s", endpoint, exc)
            return None

    _log.error("Máx reintentos alcanzados para %s", endpoint)
    return None


# -----------------------------------------------------------------------
# Funciones públicas
# -----------------------------------------------------------------------

def search_team(name: str) -> int | None:
    """Busca team_id por nombre. Prefiere coincidencia exacta."""
    data = _get("teams", {"search": name})
    if not data or not data.get("response"):
        return None
    for r in data["response"]:
        t = r.get("team", {})
        if t.get("name", "").lower() == name.lower():
            return t.get("id")
    # Fallback: primer resultado
    return data["response"][0]["team"]["id"]


def get_team_form(team_id: int, last: int = 5) -> dict | None:
    """
    Últimos `last` partidos jugados del equipo (resultados finales).
    Devuelve: {partidos, W, D, L, gf_promedio, gc_promedio,
               btts_rate, over_25_rate, secuencia}

    Nota: el plan gratuito de api-sports.io requiere 'season' en vez de 'last'.
    Se obtienen todos los partidos FT de la temporada actual y se toman los N más recientes.
    """
    # El plan gratuito de api-sports.io solo da acceso a temporadas hasta 2024.
    # Intentamos desde la más reciente permitida hacia atrás hasta encontrar datos.
    data = None
    for season in (2024, 2023, 2022):
        candidate = _get("fixtures", {"team": team_id, "season": season, "status": "FT"})
        if candidate and candidate.get("response") and not candidate.get("errors"):
            data = candidate
            break

    if not data or not data.get("response"):
        return None

    # Ordenar por fecha descendente y tomar los N más recientes
    partidos_resp = sorted(
        data["response"],
        key=lambda x: x.get("fixture", {}).get("date", ""),
        reverse=True
    )[:last]

    w = d = l = 0
    gf = gc = 0
    btts_yes = over_25 = 0
    seq = []

    for p in partidos_resp:
        goals = p.get("goals") or {}
        gh, ga = goals.get("home"), goals.get("away")
        if gh is None or ga is None:
            continue
        teams = p.get("teams") or {}
        es_local = teams.get("home", {}).get("id") == team_id
        propios = gh if es_local else ga
        ajenos  = ga if es_local else gh
        gf += propios; gc += ajenos
        if propios > ajenos:   w += 1; seq.append("W")
        elif propios < ajenos: l += 1; seq.append("L")
        else:                  d += 1; seq.append("D")
        if gh > 0 and ga > 0: btts_yes += 1
        if gh + ga > 2:       over_25  += 1

    n = len(seq)
    if n == 0:
        return None
    return {
        "partidos": n, "W": w, "D": d, "L": l,
        "gf_promedio": round(gf / n, 2),
        "gc_promedio": round(gc / n, 2),
        "btts_rate":   round(btts_yes / n, 2),
        "over_25_rate": round(over_25 / n, 2),
        "secuencia": "".join(seq),
    }


def get_head_to_head(team1_id: int, team2_id: int, last: int = 10) -> dict | None:
    """H2H: stats agregadas de los últimos N cruces directos."""
    import datetime
    date_from = (datetime.date.today() - datetime.timedelta(days=1825)).isoformat()  # 5 años
    date_to   = datetime.date.today().isoformat()

    data = _get("fixtures/headtohead", {
        "h2h":  f"{team1_id}-{team2_id}",
        "from": date_from,
        "to":   date_to,
    })
    if not data or not data.get("response"):
        return None

    total = gh_sum = ga_sum = btts = over25 = 0
    w_h1 = w_h2 = empates = 0

    for p in data["response"]:
        goals = p.get("goals") or {}
        gh, ga = goals.get("home"), goals.get("away")
        if gh is None or ga is None:
            continue
        total += 1
        teams  = p.get("teams") or {}
        hid    = teams.get("home", {}).get("id")
        if hid == team1_id:
            gh_sum += gh; ga_sum += ga
            if gh > ga: w_h1 += 1
            elif gh < ga: w_h2 += 1
            else: empates += 1
        else:
            gh_sum += ga; ga_sum += gh
            if ga > gh: w_h1 += 1
            elif ga < gh: w_h2 += 1
            else: empates += 1
        if gh > 0 and ga > 0: btts  += 1
        if gh + ga > 2:       over25 += 1

    if total == 0:
        return None
    return {
        "n": total,
        "goles_promedio":    round((gh_sum + ga_sum) / total, 2),
        "btts_rate":         round(btts   / total, 2),
        "over_25_rate":      round(over25 / total, 2),
        "wins_local_actual": w_h1,
        "empates":           empates,
        "wins_visit_actual": w_h2,
    }


def get_team_statistics(team_id: int, league_id: int, season: int) -> dict | None:
    """Stats de temporada completa de un equipo (goles, forma, etc.)."""
    data = _get("teams/statistics",
                {"team": team_id, "league": league_id, "season": season})
    if not data or not data.get("response"):
        return None
    return data["response"]


def get_fixtures_today(league_id: int | None = None) -> list:
    """Partidos del día de hoy. Filtra por liga si se especifica."""
    import datetime
    today = datetime.date.today().isoformat()
    params = {"date": today}
    if league_id:
        params["league"] = league_id
    data = _get("fixtures", params)
    if not data or not data.get("response"):
        return []
    return data["response"]


def get_injuries(team_id: int, fixture_id: int | None = None) -> list:
    """
    Lesiones y bajas. Requiere fixture_id para datos precisos.
    Sin fixture_id busca lesiones recientes del equipo.
    """
    params = {"team": team_id}
    if fixture_id:
        params["fixture"] = fixture_id
    data = _get("injuries", params)
    if not data or not data.get("response"):
        return []
    return data["response"]


def get_lineups(fixture_id: int) -> list:
    """Alineaciones confirmadas de un partido."""
    data = _get("fixtures/lineups", {"fixture": fixture_id})
    if not data or not data.get("response"):
        return []
    return data["response"]


def contexto_partido_completo(home: str, away: str) -> dict:
    """
    Función orquestadora: devuelve todo el contexto real para analizar
    un partido. Si la API no está disponible, degrada con gracia.
    """
    out = {
        "api_disponible": bool(API_FOOTBALL_KEY),
        "home": home,
        "away": away,
        "home_id": None,
        "away_id": None,
        "forma_home": None,
        "forma_away": None,
        "h2h": None,
        "injuries_home": [],
        "injuries_away": [],
        "notas": [],
    }

    if not API_FOOTBALL_KEY:
        out["notas"].append("API_FOOTBALL_KEY no configurada — análisis sin datos reales")
        return out

    hid = search_team(home)
    aid = search_team(away)
    out["home_id"] = hid
    out["away_id"] = aid

    if hid is None:
        out["notas"].append(f"No se encontró team_id para '{home}'")
    if aid is None:
        out["notas"].append(f"No se encontró team_id para '{away}'")

    if hid:
        out["forma_home"]    = get_team_form(hid, last=5)
        out["injuries_home"] = get_injuries(hid)
    if aid:
        out["forma_away"]    = get_team_form(aid, last=5)
        out["injuries_away"] = get_injuries(aid)
    if hid and aid:
        out["h2h"] = get_head_to_head(hid, aid, last=10)

    return out


def get_fixture_result(fixture_id: int) -> dict | None:
    """
    Devuelve el resultado de un partido terminado.
    Returns {terminado, goles_local, goles_visitante, status, status_short} or None.
    """
    data = _get("fixtures", {"id": fixture_id})
    if not data or not data.get("response"):
        return None
    f = data["response"][0]
    fixture = f.get("fixture", {})
    goals   = f.get("goals", {})
    status  = fixture.get("status", {})
    short   = status.get("short", "")
    return {
        "terminado":        short in ("FT", "AET", "PEN", "AWD", "WO"),
        "goles_local":      goals.get("home"),
        "goles_visitante":  goals.get("away"),
        "status":           status.get("long", ""),
        "status_short":     short,
    }


def factor_ajuste_lesiones(injuries: list) -> float:
    """
    Devuelve un factor multiplicativo para el lambda del equipo
    basado en el número de lesionados confirmados.
    Más lesionados → factor < 1 → λ reducido.
    """
    n = len([i for i in injuries
             if i.get("player", {}).get("type", "").lower() in
             ("injured", "suspended", "questionable")])
    if n >= 4:  return 0.85
    if n >= 2:  return 0.93
    if n == 1:  return 0.97
    return 1.0
