"""
Cliente Sportmonks API v3 — fuente secundaria opcional.

Rol en el sistema
─────────────────
  FALLBACK: solo se usa cuando api-football no tiene datos o falla.
  Si SPORTMONKS_TOKEN no está configurado, todas las funciones devuelven
  None / [] / contexto vacío sin levantar excepciones.

API
───
  Base URL : https://api.sportmonks.com/v3/football
  Auth     : query param ?api_token=TOKEN
  Docs     : https://docs.sportmonks.com/football
  Free tier: acceso limitado a ligas/temporadas

Caché
─────
  Misma estrategia que api_football:
    - Memoria   : 10 min TTL (evita duplicados en la misma sesión Flask)
    - Disco     : CACHE_TTL_HORAS TTL  (prefijo "sm_" para no colisionar con "af_")

Output
──────
  Todas las funciones públicas devuelven el mismo schema que api_football,
  de modo que el DataSourceManager puede intercambiarlas sin conocer la fuente.
"""
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SPORTMONKS_TOKEN, CACHE_DIR, CACHE_TTL_HORAS

_log = logging.getLogger("betbrain.sportsmonk")

BASE_URL = "https://api.sportmonks.com/v3/football"

# ──────────────────────────────────────────────────────────────────────────────
# Caché en MEMORIA (10 min)
# ──────────────────────────────────────────────────────────────────────────────

_mem_cache: dict = {}
_MEM_TTL = 600


def _mem_get(key: str):
    entry = _mem_cache.get(key)
    if entry and (time.time() - entry["ts"]) < _MEM_TTL:
        return entry["data"]
    return None


def _mem_set(key: str, data) -> None:
    _mem_cache[key] = {"ts": time.time(), "data": data}


# ──────────────────────────────────────────────────────────────────────────────
# Caché en DISCO (CACHE_TTL_HORAS) — prefijo "sm_" ≠ "af_"
# ──────────────────────────────────────────────────────────────────────────────

def _disk_key(endpoint: str, params: dict) -> Path:
    raw = json.dumps({"e": endpoint, "p": params}, sort_keys=True)
    h = hashlib.md5(raw.encode()).hexdigest()
    return CACHE_DIR / f"sm_{h}.json"


def _disk_get(endpoint: str, params: dict):
    p = _disk_key(endpoint, params)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if (time.time() - raw.get("_ts", 0)) / 3600 > CACHE_TTL_HORAS:
            return None
        return raw["data"]
    except Exception:
        return None


def _disk_set(endpoint: str, params: dict, data) -> None:
    try:
        _disk_key(endpoint, params).write_text(
            json.dumps({"_ts": time.time(), "data": data}), encoding="utf-8"
        )
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# HTTP con reintentos
# ──────────────────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None) -> dict | None:
    """GET autenticado con cache mem -> disco -> HTTP. Devuelve None si falla."""
    if not SPORTMONKS_TOKEN:
        return None

    params = {**(params or {}), "api_token": SPORTMONKS_TOKEN}
    cache_key = json.dumps({"e": endpoint, "p": params}, sort_keys=True)

    cached = _mem_get(cache_key)
    if cached is not None:
        _log.debug("sm cache-mem HIT %s", endpoint)
        return cached

    cached = _disk_get(endpoint, params)
    if cached is not None:
        _log.debug("sm cache-disk HIT %s", endpoint)
        _mem_set(cache_key, cached)
        return cached

    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(3):
        try:
            _log.info("sm GET /%s (intento %d)", endpoint, attempt + 1)
            r = requests.get(url, params=params, timeout=15)

            if r.status_code == 429:
                wait = 2 ** (attempt + 1)
                _log.warning("sm rate-limit -> esperando %ds", wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                _log.error("sm error servidor %d en /%s", r.status_code, endpoint)
                time.sleep(1)
                continue
            if r.status_code == 401:
                _log.error("sm 401 Unauthorized — verifica SPORTMONKS_TOKEN")
                return None
            if r.status_code == 404:
                _log.warning("sm 404 /%s", endpoint)
                return None
            if r.status_code != 200:
                _log.error("sm HTTP %d en /%s", r.status_code, endpoint)
                return None

            data = r.json()
            _mem_set(cache_key, data)
            _disk_set(endpoint, params, data)
            _log.info("sm OK /%s -> %d items",
                      endpoint, len(data.get("data", [])) if isinstance(data.get("data"), list) else 1)
            return data

        except requests.exceptions.Timeout:
            _log.warning("sm timeout /%s (intento %d)", endpoint, attempt + 1)
            time.sleep(1)
        except Exception as exc:
            _log.error("sm excepción /%s: %s", endpoint, exc)
            return None

    _log.error("sm máx reintentos para /%s", endpoint)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Parseo de scores Sportmonks v3
# ──────────────────────────────────────────────────────────────────────────────

def _extraer_goles(fixture: dict) -> tuple[int | None, int | None]:
    """
    Extrae (goles_home, goles_away) de un fixture Sportmonks v3.
    Busca en `scores` (include) con description=CURRENT.
    Devuelve (None, None) si no hay scores disponibles.
    """
    scores = fixture.get("scores", [])
    if not scores:
        return None, None

    gh = ga = None
    for s in scores:
        if s.get("description") != "CURRENT":
            continue
        score_obj = s.get("score", {})
        participant = score_obj.get("participant", "")
        goals = score_obj.get("goals")
        if goals is None:
            continue
        if participant == "home":
            gh = int(goals)
        elif participant == "away":
            ga = int(goals)

    return gh, ga


def _equipo_es_local(fixture: dict, team_id: int) -> bool | None:
    """
    Devuelve True si team_id jugó como local en el fixture, False si visitante.
    Devuelve None si no se puede determinar.
    """
    for p in fixture.get("participants", []):
        if p.get("id") == team_id:
            return p.get("meta", {}).get("location", "") == "home"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Funciones públicas — mismo schema que api_football
# ──────────────────────────────────────────────────────────────────────────────

def search_team(name: str) -> int | None:
    """
    Busca team_id por nombre. Prefiere coincidencia exacta.
    Devuelve None si no hay token o no encuentra el equipo.
    """
    data = _get(f"teams/search/{name}")
    if not data or not isinstance(data.get("data"), list):
        return None
    teams = data["data"]
    if not teams:
        return None
    # Preferir coincidencia exacta (case-insensitive)
    for t in teams:
        if t.get("name", "").lower() == name.lower():
            return t.get("id")
    return teams[0].get("id")


def get_team_form(team_id: int, last: int = 5) -> dict | None:
    """
    Últimos `last` partidos del equipo (finalizados).
    Devuelve el mismo schema que api_football.get_team_form.
    """
    # Sportmonks v3: /fixtures/teams/{id}/last/{n}?include=scores;participants
    data = _get(
        f"fixtures/teams/{team_id}/last/{last}",
        {"include": "scores;participants"},
    )
    if not data or not isinstance(data.get("data"), list):
        return None

    fixtures = data["data"]
    if not fixtures:
        return None

    w = d = l = 0
    gf = gc = btts_yes = over_25 = 0
    seq: list[str] = []

    for fix in fixtures:
        gh, ga = _extraer_goles(fix)
        if gh is None or ga is None:
            continue
        es_local = _equipo_es_local(fix, team_id)
        if es_local is None:
            continue
        propios = gh if es_local else ga
        ajenos  = ga if es_local else gh
        gf += propios
        gc += ajenos
        if propios > ajenos:   w += 1; seq.append("W")
        elif propios < ajenos: l += 1; seq.append("L")
        else:                  d += 1; seq.append("D")
        if gh > 0 and ga > 0: btts_yes += 1
        if gh + ga > 2:       over_25  += 1

    n = len(seq)
    if n == 0:
        return None

    return {
        "partidos":    n,
        "W": w, "D": d, "L": l,
        "gf_promedio": round(gf / n, 2),
        "gc_promedio": round(gc / n, 2),
        "btts_rate":   round(btts_yes / n, 2),
        "over_25_rate": round(over_25 / n, 2),
        "secuencia":   "".join(seq),
        "_fuente":     "sportsmonk",
    }


def get_head_to_head(team1_id: int, team2_id: int, last: int = 10) -> dict | None:
    """
    H2H entre dos equipos. Normaliza perspectiva desde team1_id como local.
    Devuelve el mismo schema que api_football.get_head_to_head.
    """
    data = _get(
        f"fixtures/head-to-head/{team1_id}/{team2_id}",
        {"include": "scores;participants"},
    )
    if not data or not isinstance(data.get("data"), list):
        return None

    fixtures = sorted(
        data["data"],
        key=lambda x: x.get("starting_at", ""),
        reverse=True,
    )[:last]

    if not fixtures:
        return None

    total = gh_sum = ga_sum = btts = over25 = 0
    w_t1 = w_t2 = empates = 0

    for fix in fixtures:
        gh, ga = _extraer_goles(fix)
        if gh is None or ga is None:
            continue

        # Normalizar: perspectiva desde team1_id
        t1_es_local = _equipo_es_local(fix, team1_id)
        if t1_es_local is None:
            continue
        g1 = gh if t1_es_local else ga
        g2 = ga if t1_es_local else gh

        total  += 1
        gh_sum += g1
        ga_sum += g2
        if g1 > g2:   w_t1 += 1
        elif g1 < g2: w_t2 += 1
        else:         empates += 1
        if gh > 0 and ga > 0: btts   += 1
        if gh + ga > 2:       over25 += 1

    if total == 0:
        return None

    return {
        "n":                 total,
        "goles_promedio":    round((gh_sum + ga_sum) / total, 2),
        "btts_rate":         round(btts   / total, 2),
        "over_25_rate":      round(over25 / total, 2),
        "wins_local_actual": w_t1,
        "empates":           empates,
        "wins_visit_actual": w_t2,
        "_fuente":           "sportsmonk",
    }


def contexto_partido_completo(home: str, away: str) -> dict:
    """
    Contexto completo para analizar un partido.
    Compatible en schema con api_football.contexto_partido_completo.
    Si SPORTMONKS_TOKEN no está configurado, devuelve ctx con api_disponible=False.
    """
    out: dict = {
        "api_disponible": bool(SPORTMONKS_TOKEN),
        "fuente":         "sportsmonk",
        "home":           home,
        "away":           away,
        "home_id":        None,
        "away_id":        None,
        "forma_home":     None,
        "forma_away":     None,
        "h2h":            None,
        "injuries_home":  [],   # Sportmonks injuries requiere plan de pago
        "injuries_away":  [],
        "notas":          [],
    }

    if not SPORTMONKS_TOKEN:
        out["notas"].append("SPORTMONKS_TOKEN no configurado — fuente deshabilitada")
        return out

    hid = search_team(home)
    aid = search_team(away)
    out["home_id"] = hid
    out["away_id"] = aid

    if hid is None:
        out["notas"].append(f"Sportmonks: equipo '{home}' no encontrado")
    if aid is None:
        out["notas"].append(f"Sportmonks: equipo '{away}' no encontrado")

    if hid:
        out["forma_home"] = get_team_form(hid, last=5)
        if out["forma_home"] is None:
            out["notas"].append(f"Sportmonks: sin forma reciente para '{home}'")
    if aid:
        out["forma_away"] = get_team_form(aid, last=5)
        if out["forma_away"] is None:
            out["notas"].append(f"Sportmonks: sin forma reciente para '{away}'")
    if hid and aid:
        out["h2h"] = get_head_to_head(hid, aid, last=10)
        if out["h2h"] is None:
            out["notas"].append(f"Sportmonks: sin H2H para '{home}' vs '{away}'")

    return out


def disponible() -> bool:
    """True si SPORTMONKS_TOKEN está configurado."""
    return bool(SPORTMONKS_TOKEN)
