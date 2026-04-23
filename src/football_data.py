"""
Cliente API-Football con caché en disco.
Free tier: 100 req/día, por eso cacheamos todo.

Uso:
    from src.football_data import contexto_partido
    ctx = contexto_partido("Real Madrid", "Barcelona", liga="La Liga")

Si API_FOOTBALL_KEY no está configurada, devuelve None (degrada con gracia).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import API_FOOTBALL_KEY, API_FOOTBALL_HOST, CACHE_DIR, CACHE_TTL_HORAS


BASE_URL = f"https://{API_FOOTBALL_HOST}"
HEADERS = {
    "x-rapidapi-host": API_FOOTBALL_HOST,
    "x-rapidapi-key": API_FOOTBALL_KEY,
}


# -------------------------------------------------------------------
# Caché en disco
# -------------------------------------------------------------------
def _cache_path(endpoint: str, params: dict) -> Path:
    key = json.dumps({"e": endpoint, "p": params}, sort_keys=True)
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _cache_get(endpoint: str, params: dict):
    p = _cache_path(endpoint, params)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    edad_h = (time.time() - raw.get("_ts", 0)) / 3600
    if edad_h > CACHE_TTL_HORAS:
        return None
    return raw["data"]


def _cache_set(endpoint: str, params: dict, data):
    p = _cache_path(endpoint, params)
    p.write_text(json.dumps({"_ts": time.time(), "data": data}),
                 encoding="utf-8")


# -------------------------------------------------------------------
# HTTP
# -------------------------------------------------------------------
def _get(endpoint: str, params: dict | None = None, use_cache: bool = True):
    if not API_FOOTBALL_KEY:
        return None
    params = params or {}
    if use_cache:
        cached = _cache_get(endpoint, params)
        if cached is not None:
            return cached
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS,
                         params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    _cache_set(endpoint, params, data)
    return data


# -------------------------------------------------------------------
# Helpers de alto nivel
# -------------------------------------------------------------------
def buscar_team_id(nombre: str) -> int | None:
    """Busca un equipo por nombre y devuelve su team_id (o None)."""
    data = _get("teams", {"search": nombre})
    if not data or not data.get("response"):
        return None
    # Preferir match exacto
    for r in data["response"]:
        t = r.get("team", {})
        if t.get("name", "").lower() == nombre.lower():
            return t.get("id")
    # Fallback: primer resultado
    return data["response"][0]["team"]["id"]


def forma_reciente(team_id: int, last: int = 5) -> dict | None:
    """
    Devuelve stats de los últimos `last` partidos del equipo:
    wins/draws/losses, goles a favor, goles en contra, BTTS rate, Over 2.5 rate,
    y una cadena "WDLWW" legible.
    """
    data = _get("fixtures", {"team": team_id, "last": last})
    if not data or not data.get("response"):
        return None
    partidos = data["response"]
    if not partidos:
        return None

    w = d = l = 0
    gf = gc = 0
    btts_yes = over_25 = 0
    seq = []
    for p in partidos:
        goals = p.get("goals") or {}
        gh, ga = goals.get("home"), goals.get("away")
        if gh is None or ga is None:
            continue
        teams = p.get("teams") or {}
        es_local = teams.get("home", {}).get("id") == team_id
        propios = gh if es_local else ga
        ajenos = ga if es_local else gh
        gf += propios
        gc += ajenos
        if propios > ajenos:
            w += 1; seq.append("W")
        elif propios < ajenos:
            l += 1; seq.append("L")
        else:
            d += 1; seq.append("D")
        if gh > 0 and ga > 0:
            btts_yes += 1
        if gh + ga > 2:
            over_25 += 1

    n = len(seq)
    if n == 0:
        return None
    return {
        "partidos": n,
        "W": w, "D": d, "L": l,
        "gf_promedio": round(gf / n, 2),
        "gc_promedio": round(gc / n, 2),
        "btts_rate": round(btts_yes / n, 2),
        "over_25_rate": round(over_25 / n, 2),
        "secuencia": "".join(seq),   # "WDLWW"
    }


def h2h(home_id: int, away_id: int, last: int = 10) -> dict | None:
    """Head-to-head: estadísticas agregadas de los últimos N encuentros."""
    data = _get("fixtures/headtohead",
                {"h2h": f"{home_id}-{away_id}", "last": last})
    if not data or not data.get("response"):
        return None
    partidos = data["response"]
    if not partidos:
        return None

    total = 0
    gh_sum = ga_sum = 0
    btts = over25 = 0
    w_local_actual = w_visit_actual = empates = 0
    for p in partidos:
        goals = p.get("goals") or {}
        gh, ga = goals.get("home"), goals.get("away")
        if gh is None or ga is None:
            continue
        total += 1
        teams = p.get("teams") or {}
        hid = teams.get("home", {}).get("id")
        aid = teams.get("away", {}).get("id")
        # Normalizar: contar desde la perspectiva del home_id actual
        if hid == home_id:
            gh_sum += gh; ga_sum += ga
            if gh > ga: w_local_actual += 1
            elif gh < ga: w_visit_actual += 1
            else: empates += 1
        elif hid == away_id:
            # Invertir
            gh_sum += ga; ga_sum += gh
            if ga > gh: w_local_actual += 1  # home_id (jugando visitante) ganó
            elif ga < gh: w_visit_actual += 1
            else: empates += 1
        if gh > 0 and ga > 0:
            btts += 1
        if gh + ga > 2:
            over25 += 1

    if total == 0:
        return None
    return {
        "n": total,
        "goles_promedio": round((gh_sum + ga_sum) / total, 2),
        "btts_rate": round(btts / total, 2),
        "over_25_rate": round(over25 / total, 2),
        "wins_local_actual": w_local_actual,
        "empates": empates,
        "wins_visit_actual": w_visit_actual,
    }


def contexto_partido(home: str, away: str) -> dict:
    """
    Función de alto nivel: devuelve todo el contexto útil para analizar
    un partido. Si la API no está disponible o algo falla, devuelve
    un dict con banderas.
    """
    out = {
        "api_disponible": bool(API_FOOTBALL_KEY),
        "home": home,
        "away": away,
        "forma_home": None,
        "forma_away": None,
        "h2h": None,
        "notas": [],
    }
    if not API_FOOTBALL_KEY:
        out["notas"].append("API_FOOTBALL_KEY no configurada — sin datos reales")
        return out

    hid = buscar_team_id(home)
    aid = buscar_team_id(away)
    if hid is None:
        out["notas"].append(f"No encontré team_id para '{home}'")
    if aid is None:
        out["notas"].append(f"No encontré team_id para '{away}'")

    if hid:
        out["forma_home"] = forma_reciente(hid, last=5)
    if aid:
        out["forma_away"] = forma_reciente(aid, last=5)
    if hid and aid:
        out["h2h"] = h2h(hid, aid, last=10)

    return out


if __name__ == "__main__":
    import json
    print(json.dumps(contexto_partido("Real Madrid", "Barcelona"),
                     indent=2, ensure_ascii=False))
