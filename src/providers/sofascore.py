"""
Proveedor SofaScore (fallback) — API no oficial, sin documentación pública.

Usado solo cuando api-football → sportsmonk → thesportsdb fallan (ver
DataSourceManager._ctx_auto en manager.py). No hay endpoint de H2H directo
por nombres (SofaScore requiere un ID de partido específico) ni de
lesiones por equipo (probado, 404) — se dejan en None/[] igual que hacen
los demás proveedores cuando no tienen ese dato.

Evaluado manualmente antes de escribir esto: search/all y
team/{id}/events/last/0 devuelven datos reales; scheduled-events/{date},
search (sin /all) y las dos rutas de lesiones probadas dieron 404 —
endpoints no documentados, pueden cambiar sin aviso.

CONOCIDO: en el smoke test de esta integración, api.sofascore.com devolvió
403 Forbidden a peticiones hechas con requests (Python), mientras que la
misma URL con el mismo User-Agent vía curl devolvió 200 — indica bloqueo
por fingerprint TLS (WAF/Cloudflare detectando la librería, no los headers),
no un error de este código. En la práctica esto puede dejar este paso del
fallback inerte (api_disponible=False) hasta que cambien las reglas del WAF
o SofaScore quede accesible de nuevo. Se mantiene igual porque falla en modo
seguro (ver contexto_partido_completo: nunca lanza excepción, siempre
devuelve el schema esperado con api_disponible=False y notas explicativas)
y no agrega peso al bundle (requests ya es dependencia del proyecto).
"""
import logging
import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

_log = logging.getLogger("betbrain.sofascore")

BASE_URL = "https://api.sofascore.com/api/v1"
TIMEOUT = 8
# SofaScore respondió 301/404 con un User-Agent genérico en las pruebas
# manuales; con uno de navegador real funcionó de forma consistente.
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
}

_session = requests.Session()
_session.headers.update(_HEADERS)


def _buscar_equipo(nombre: str) -> Optional[tuple[int, str]]:
    """Busca un equipo por nombre. Devuelve (id, nombre_exacto) o None."""
    try:
        r = _session.get(f"{BASE_URL}/search/all", params={"q": nombre}, timeout=TIMEOUT)
        r.raise_for_status()
        for result in r.json().get("results", []):
            if result.get("type") != "team":
                continue
            entity = result.get("entity", {})
            team_name = (entity.get("name") or "").lower()
            if nombre.lower() in team_name or team_name in nombre.lower():
                return entity.get("id"), entity.get("name")
    except Exception as exc:
        _log.debug("SofaScore búsqueda de '%s' falló: %s", nombre, exc)
    return None


def _eventos_recientes(team_id: int) -> list[dict]:
    """Últimos partidos finalizados de un equipo (más recientes primero)."""
    try:
        r = _session.get(f"{BASE_URL}/team/{team_id}/events/last/0", timeout=TIMEOUT)
        r.raise_for_status()
        eventos = r.json().get("events", [])
        return [e for e in eventos if e.get("status", {}).get("type") == "finished"]
    except Exception as exc:
        _log.debug("SofaScore eventos de team %s fallaron: %s", team_id, exc)
        return []


def get_team_form(team_id: int, last: int = 5) -> Optional[dict]:
    """
    Forma reciente de un equipo a partir de sus últimos `last` partidos.

    Mismo schema que football_csv.get_team_form_csv / api_football.get_team_form:
      {partidos, W, D, L, gf_promedio, gc_promedio, btts_rate, over_25_rate, secuencia}
    None si no hay partidos parseables — nunca se inventa un resultado
    default (a diferencia de un placeholder tipo "DDDDD").
    """
    eventos = _eventos_recientes(team_id)[:last]
    if not eventos:
        return None

    w = d = l = 0
    gf = gc = btts_yes = over_25 = 0
    seq = []

    for evt in eventos:
        home_id = evt.get("homeTeam", {}).get("id")
        away_id = evt.get("awayTeam", {}).get("id")
        hs = evt.get("homeScore", {}).get("current")
        as_ = evt.get("awayScore", {}).get("current")
        if hs is None or as_ is None:
            continue

        es_local = team_id == home_id
        if not es_local and team_id != away_id:
            continue
        propios = hs if es_local else as_
        ajenos  = as_ if es_local else hs

        gf += propios
        gc += ajenos
        if propios > ajenos:   w += 1; seq.append("W")
        elif propios < ajenos: l += 1; seq.append("L")
        else:                  d += 1; seq.append("D")
        if propios > 0 and ajenos > 0: btts_yes += 1
        if propios + ajenos > 2:       over_25  += 1

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
        "_fuente":      "sofascore",
    }


def contexto_partido_completo(home: str, away: str) -> dict:
    """
    Contexto de partido desde SofaScore. Compatible en schema con
    api_football.contexto_partido_completo (mismas claves).

    Sin H2H (requiere resolver un ID de evento específico entre ambos
    equipos, no un endpoint directo por nombres) ni lesiones (no se
    encontró un endpoint funcional en la evaluación manual).
    """
    notas: list[str] = []

    home_res = _buscar_equipo(home)
    away_res = _buscar_equipo(away)
    home_id = home_res[0] if home_res else None
    away_id = away_res[0] if away_res else None

    if home_id is None:
        notas.append(f"SofaScore: equipo '{home}' no encontrado")
    if away_id is None:
        notas.append(f"SofaScore: equipo '{away}' no encontrado")

    forma_home = get_team_form(home_id, last=5) if home_id else None
    forma_away = get_team_form(away_id, last=5) if away_id else None

    if forma_home is None and home_id:
        notas.append(f"SofaScore: sin partidos recientes para '{home}'")
    if forma_away is None and away_id:
        notas.append(f"SofaScore: sin partidos recientes para '{away}'")

    api_disponible = forma_home is not None or forma_away is not None

    return {
        "api_disponible": api_disponible,
        "fuente":         "sofascore",
        "home":           home,
        "away":           away,
        "home_id":        home_id,
        "away_id":        away_id,
        "forma_home":     forma_home,
        "forma_away":     forma_away,
        "h2h":            None,   # requiere ID de evento específico — no implementado
        "injuries_home":  [],     # sin endpoint funcional encontrado
        "injuries_away":  [],
        "notas":          notas,
    }
