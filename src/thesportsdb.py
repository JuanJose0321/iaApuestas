"""
Cliente TheSportsDB API v1 — fuente gratuita sin key para datos de liga.

Rol en el sistema
─────────────────
  COMPLEMENTO: cubre ligas que api-football/sportsmonk no tienen en free tier
  (Liga MX, MLS, Brasileirao, etc.) mediante la API pública gratuita v1.
  Se agrega a la cadena de fallback del DataSourceManager.

API
───
  Base URL  : https://www.thesportsdb.com/api/v1/json/123/
  Auth      : key=123 (pública, sin registro)
  Free tier : sin limite diario, sin livescore
  Ligas MX  : id=4350 (Liga MX / Mexican Primera League)
  Docs      : https://www.thesportsdb.com/api.php

Limitaciones free tier
──────────────────────
  - Sin livescore en tiempo real (requiere Patreon v2)
  - Historico limitado (ultimo partido por equipo en eventslast)
  - Sin odds
  - eventsh2h requiere IDs exactos

Cache
─────
  Memoria 10 min + Disco CACHE_TTL_HORAS  (prefijo "tsdb_")
"""
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CACHE_DIR, CACHE_TTL_HORAS

_log = logging.getLogger("betbrain.thesportsdb")

BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"

LIGA_MX_ID = "4350"

# ──────────────────────────────────────────────────────────────────────────────
# Cache memoria (10 min)
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
# Cache disco
# ──────────────────────────────────────────────────────────────────────────────
def _disk_key(endpoint: str, params: dict) -> Path:
    raw = json.dumps({"e": endpoint, "p": params}, sort_keys=True)
    h = hashlib.md5(raw.encode()).hexdigest()
    return CACHE_DIR / f"tsdb_{h}.json"


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
    params = params or {}
    cache_key = json.dumps({"e": endpoint, "p": params}, sort_keys=True)

    cached = _mem_get(cache_key)
    if cached is not None:
        _log.debug("tsdb cache-mem HIT %s", endpoint)
        return cached

    cached = _disk_get(endpoint, params)
    if cached is not None:
        _log.debug("tsdb cache-disk HIT %s", endpoint)
        _mem_set(cache_key, cached)
        return cached

    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(3):
        try:
            _log.info("tsdb GET /%s (intento %d)", endpoint, attempt + 1)
            r = requests.get(url, params=params, timeout=15)

            if r.status_code == 429:
                wait = 2 ** (attempt + 1)
                _log.warning("tsdb rate-limit -> esperando %ds", wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                _log.error("tsdb error servidor %d en /%s", r.status_code, endpoint)
                time.sleep(1)
                continue
            if r.status_code != 200:
                _log.error("tsdb HTTP %d en /%s", r.status_code, endpoint)
                return None

            data = r.json()
            _mem_set(cache_key, data)
            _disk_set(endpoint, params, data)
            return data

        except requests.exceptions.Timeout:
            _log.warning("tsdb timeout /%s (intento %d)", endpoint, attempt + 1)
            time.sleep(1)
        except Exception as exc:
            _log.error("tsdb excepcion /%s: %s", endpoint, exc)
            return None

    _log.error("tsdb max reintentos para /%s", endpoint)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Parseo de resultados
# ──────────────────────────────────────────────────────────────────────────────
def _parse_score(event: dict) -> tuple[int | None, int | None]:
    """Extrae (home_goals, away_goals) de un evento. None si no terminado."""
    try:
        hg = event.get("intHomeScore")
        ag = event.get("intAwayScore")
        if hg is None or ag is None:
            return None, None
        return int(hg), int(ag)
    except (TypeError, ValueError):
        return None, None


def _equipo_es_local(event: dict, team_name: str) -> bool | None:
    home = (event.get("strHomeTeam") or "").lower()
    away = (event.get("strAwayTeam") or "").lower()
    name_lower = team_name.lower()
    if name_lower in home:
        return True
    if name_lower in away:
        return False
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Funciones publicas — mismo schema que api_football / sportsmonk
# ──────────────────────────────────────────────────────────────────────────────

def search_team(name: str) -> tuple[str | None, str | None]:
    """
    Busca equipo por nombre. Devuelve (idTeam, strTeam) o (None, None).
    Prefiere coincidencia exacta.
    """
    data = _get("searchteams.php", {"t": name})
    if not data:
        return None, None
    teams = data.get("teams") or []
    if not teams:
        return None, None

    name_lower = name.lower()
    for t in teams:
        if (t.get("strTeam") or "").lower() == name_lower:
            return t.get("idTeam"), t.get("strTeam")
    return teams[0].get("idTeam"), teams[0].get("strTeam")


def get_team_form(team_id: str, team_name: str, last: int = 5) -> dict | None:
    """
    Ultimos partidos del equipo (API devuelve los ultimos disponibles).
    Devuelve mismo schema que api_football.get_team_form.
    """
    data = _get("eventslast.php", {"id": team_id})
    if not data:
        return None

    events = data.get("results") or []
    if not events:
        return None

    # Ordenar por fecha descendente y tomar los ultimos N
    events_sorted = sorted(events, key=lambda e: e.get("dateEvent", ""), reverse=True)[:last]

    w = d = l = 0
    gf = gc = btts_yes = over_25 = 0
    seq: list[str] = []

    for ev in events_sorted:
        gh, ga = _parse_score(ev)
        if gh is None or ga is None:
            continue
        es_local = _equipo_es_local(ev, team_name)
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
        if gh + ga > 2:       over_25 += 1

    n = len(seq)
    if n == 0:
        return None

    return {
        "partidos":     n,
        "W": w, "D": d, "L": l,
        "gf_promedio":  round(gf / n, 2),
        "gc_promedio":  round(gc / n, 2),
        "btts_rate":    round(btts_yes / n, 2),
        "over_25_rate": round(over_25 / n, 2),
        "secuencia":    "".join(seq),
        "_fuente":      "thesportsdb",
    }


def get_head_to_head(team1_id: str, team2_id: str, team1_name: str,
                     last: int = 10) -> dict | None:
    """
    H2H entre dos equipos. Normaliza perspectiva desde team1 como referencia.
    Devuelve mismo schema que api_football.get_head_to_head.
    """
    data = _get("eventsh2h.php", {"id": team1_id, "id2": team2_id})
    if not data:
        return None

    events = data.get("results") or []
    if not events:
        return None

    events_sorted = sorted(events, key=lambda e: e.get("dateEvent", ""), reverse=True)[:last]

    total = gh_sum = ga_sum = btts = over25 = 0
    w_t1 = w_t2 = empates = 0

    for ev in events_sorted:
        gh, ga = _parse_score(ev)
        if gh is None or ga is None:
            continue

        t1_local = _equipo_es_local(ev, team1_name)
        if t1_local is None:
            continue

        g1 = gh if t1_local else ga
        g2 = ga if t1_local else gh

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
        "_fuente":           "thesportsdb",
    }


def get_fixtures_today(league_id: str = LIGA_MX_ID) -> list[dict]:
    """
    Proximos partidos de la liga (no hay endpoint de 'hoy' exacto en v1 free).
    Devuelve lista de dicts con home, away, fecha, id_evento.
    """
    from datetime import date
    hoy = date.today().isoformat()

    data = _get("eventsnextleague.php", {"id": league_id})
    events = (data.get("events") or []) if data else []

    resultado = []
    for ev in events:
        fecha = ev.get("dateEvent", "")
        if not fecha.startswith(hoy):
            continue
        resultado.append({
            "id_evento":  ev.get("idEvent"),
            "home":       ev.get("strHomeTeam", ""),
            "away":       ev.get("strAwayTeam", ""),
            "liga":       ev.get("strLeague", ""),
            "fecha":      fecha,
            "hora":       ev.get("strTime", ""),
        })
    return resultado


def contexto_partido_completo(home: str, away: str) -> dict:
    """
    Contexto completo para analizar un partido.
    Compatible en schema con api_football.contexto_partido_completo.
    """
    out: dict = {
        "api_disponible": True,
        "fuente":         "thesportsdb",
        "home":           home,
        "away":           away,
        "home_id":        None,
        "away_id":        None,
        "forma_home":     None,
        "forma_away":     None,
        "h2h":            None,
        "injuries_home":  [],
        "injuries_away":  [],
        "notas":          [],
    }

    hid, hname = search_team(home)
    aid, aname = search_team(away)
    out["home_id"] = hid
    out["away_id"] = aid

    if hid is None:
        out["notas"].append(f"TheSportsDB: equipo '{home}' no encontrado")
        out["api_disponible"] = False
    if aid is None:
        out["notas"].append(f"TheSportsDB: equipo '{away}' no encontrado")
        out["api_disponible"] = False

    if hid:
        out["forma_home"] = get_team_form(hid, hname or home)
        if out["forma_home"] is None:
            out["notas"].append(f"TheSportsDB: sin forma reciente para '{home}'")

    if aid:
        out["forma_away"] = get_team_form(aid, aname or away)
        if out["forma_away"] is None:
            out["notas"].append(f"TheSportsDB: sin forma reciente para '{away}'")

    if hid and aid:
        out["h2h"] = get_head_to_head(hid, aid, hname or home)
        if out["h2h"] is None:
            out["notas"].append(f"TheSportsDB: sin H2H para '{home}' vs '{away}'")

    if not (out["forma_home"] or out["forma_away"]):
        out["api_disponible"] = False

    return out


def disponible() -> bool:
    """TheSportsDB v1 siempre disponible (sin key)."""
    return True
