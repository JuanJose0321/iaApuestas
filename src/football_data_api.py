"""
Proveedor de contexto basado en los CSVs locales de football-data.co.uk.

Fuente      : data/raw/*.csv  (descargados por src/data_loader.py)
Ventajas    : sin límite de requests, siempre disponible, cubre 4 temporadas
Limitaciones: solo las 5 ligas configuradas en LIGAS, sin lesiones/alineaciones,
              frescura = última descarga manual de los CSVs

Devuelve el mismo esquema que api_football.contexto_partido_completo
para usarse como drop-in fallback transparente.

Funciones públicas
──────────────────
  contexto_partido_completo(home, away) → dict   (mismo schema que api_football)
  get_team_form_csv(team_name, last=5)  → dict | None
  get_h2h_csv(home, away, last=10)      → dict | None
  buscar_nombre_equipo(name)            → str | None   (nombre normalizado en CSV)
"""
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR

_log = logging.getLogger("betbrain.football_data_api")

# ──────────────────────────────────────────────────────────────────────────────
# Carga y caché del DataFrame global (se carga una sola vez al importar)
# ──────────────────────────────────────────────────────────────────────────────

_df_cache: pd.DataFrame | None = None


def _cargar_df() -> pd.DataFrame:
    """Carga y concatena todos los CSVs de RAW_DATA_DIR. Resultado en módulo cache."""
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    frames = []
    requeridas = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    for csv in sorted(RAW_DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv, encoding="latin-1", on_bad_lines="skip")
            if not all(c in df.columns for c in requeridas):
                continue
            liga_val = csv.stem.split("_")[0]
            subset = df[requeridas].copy()
            subset["_liga"] = liga_val
            frames.append(subset)
        except Exception as exc:
            _log.warning("CSV %s no se pudo cargar: %s", csv.name, exc)

    if not frames:
        _log.error("No hay CSVs en %s. Ejecuta: python src/data_loader.py", RAW_DATA_DIR)
        _df_cache = pd.DataFrame(columns=requeridas + ["_liga"])
        return _df_cache

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=requeridas).sort_values("Date").reset_index(drop=True)
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    _df_cache = df
    _log.info("CSV provider: %d partidos cargados de %d archivos", len(df), len(frames))
    return _df_cache


def reload_csv() -> None:
    """Fuerza recarga del DataFrame (útil tras descargar CSVs nuevos)."""
    global _df_cache
    _df_cache = None
    _cargar_df()


# ──────────────────────────────────────────────────────────────────────────────
# Normalización y búsqueda de nombres de equipo
# ──────────────────────────────────────────────────────────────────────────────

def _normalizar(name: str) -> str:
    """Lowercase sin acentos ni puntuación extra para comparar nombres."""
    name = name.lower().strip()
    # reemplazos comunes de acentos que aparecen en inglés/español en los CSVs
    for src, dst in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),
                     ("ü","u"),("ñ","n"),("ç","c")]:
        name = name.replace(src, dst)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return " ".join(name.split())


@lru_cache(maxsize=256)
def _todos_los_equipos() -> list[str]:
    df = _cargar_df()
    equipos = set(df["HomeTeam"].dropna().tolist()) | set(df["AwayTeam"].dropna().tolist())
    return sorted(equipos)


def buscar_nombre_equipo(name: str) -> str | None:
    """
    Busca el nombre exacto del equipo tal como aparece en los CSVs.
    Estrategia: exacto → startswith → contains.
    Devuelve None si no encuentra nada razonable.
    """
    if not name:
        return None
    norm_input = _normalizar(name)
    equipos = _todos_los_equipos()
    norm_equipo = {_normalizar(e): e for e in equipos}

    # 1. Coincidencia exacta
    if norm_input in norm_equipo:
        return norm_equipo[norm_input]

    # 2. Comienza con el input (ej. "man united" → "Manchester United")
    candidatos = [orig for norm, orig in norm_equipo.items()
                  if norm.startswith(norm_input) or norm_input.startswith(norm)]
    if len(candidatos) == 1:
        return candidatos[0]

    # 3. Contiene el input completo
    candidatos = [orig for norm, orig in norm_equipo.items() if norm_input in norm]
    if len(candidatos) == 1:
        return candidatos[0]

    # 4. Todas las palabras del input presentes en el nombre del equipo
    palabras = norm_input.split()
    candidatos = [orig for norm, orig in norm_equipo.items()
                  if all(p in norm for p in palabras)]
    if len(candidatos) == 1:
        return candidatos[0]
    if len(candidatos) > 1:
        # Priorizar el más corto (más específico)
        candidatos.sort(key=lambda x: len(x))
        return candidatos[0]

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Forma reciente
# ──────────────────────────────────────────────────────────────────────────────

def get_team_form_csv(team_name: str, last: int = 5) -> dict | None:
    """
    Calcula la forma del equipo a partir de sus últimos `last` partidos
    registrados en los CSVs.

    Devuelve el mismo esquema que api_football.get_team_form:
      {partidos, W, D, L, gf_promedio, gc_promedio, btts_rate, over_25_rate, secuencia}
    """
    df = _cargar_df()
    if df.empty:
        return None

    nombre_csv = buscar_nombre_equipo(team_name)
    if nombre_csv is None:
        _log.debug("CSV: equipo '%s' no encontrado", team_name)
        return None

    mask = (df["HomeTeam"] == nombre_csv) | (df["AwayTeam"] == nombre_csv)
    partidos = df[mask].sort_values("Date", ascending=False).head(last)

    if partidos.empty:
        return None

    w = d = l = 0
    gf = gc = btts_yes = over_25 = 0
    seq = []

    for _, row in partidos.iterrows():
        es_local = row["HomeTeam"] == nombre_csv
        propios = int(row["FTHG"]) if es_local else int(row["FTAG"])
        ajenos  = int(row["FTAG"]) if es_local else int(row["FTHG"])
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
        "partidos":    n,
        "W": w, "D": d, "L": l,
        "gf_promedio": round(gf / n, 2),
        "gc_promedio": round(gc / n, 2),
        "btts_rate":   round(btts_yes / n, 2),
        "over_25_rate": round(over_25 / n, 2),
        "secuencia":   "".join(seq),
        "_fuente":     "csv",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Head-to-Head
# ──────────────────────────────────────────────────────────────────────────────

def get_h2h_csv(home: str, away: str, last: int = 10) -> dict | None:
    """
    H2H entre home y away desde los CSVs históricos.
    Normaliza perspectiva desde el punto de vista del equipo local actual (home).

    Devuelve el mismo esquema que api_football.get_head_to_head:
      {n, goles_promedio, btts_rate, over_25_rate,
       wins_local_actual, empates, wins_visit_actual}
    """
    df = _cargar_df()
    if df.empty:
        return None

    nombre_h = buscar_nombre_equipo(home)
    nombre_a = buscar_nombre_equipo(away)

    if nombre_h is None or nombre_a is None:
        _log.debug("CSV H2H: no encontrado home='%s' away='%s'", home, away)
        return None

    mask = (
        ((df["HomeTeam"] == nombre_h) & (df["AwayTeam"] == nombre_a)) |
        ((df["HomeTeam"] == nombre_a) & (df["AwayTeam"] == nombre_h))
    )
    enfrentamientos = df[mask].sort_values("Date", ascending=False).head(last)

    if enfrentamientos.empty:
        return None

    total = gh_sum = ga_sum = btts = over25 = 0
    w_h = w_a = empates = 0

    for _, row in enfrentamientos.iterrows():
        gh, ga = int(row["FTHG"]), int(row["FTAG"])
        # Normalizar perspectiva: home es siempre nombre_h
        if row["HomeTeam"] == nombre_h:
            goles_h, goles_a = gh, ga
        else:  # nombre_h jugó de visitante
            goles_h, goles_a = ga, gh

        gh_sum += goles_h
        ga_sum += goles_a
        total  += 1

        if goles_h > goles_a:   w_h += 1
        elif goles_h < goles_a: w_a += 1
        else:                   empates += 1

        if gh > 0 and ga > 0:  btts  += 1
        if gh + ga > 2:        over25 += 1

    if total == 0:
        return None

    return {
        "n":                 total,
        "goles_promedio":    round((gh_sum + ga_sum) / total, 2),
        "btts_rate":         round(btts   / total, 2),
        "over_25_rate":      round(over25 / total, 2),
        "wins_local_actual": w_h,
        "empates":           empates,
        "wins_visit_actual": w_a,
        "_fuente":           "csv",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Contexto completo — mismo esquema que api_football.contexto_partido_completo
# ──────────────────────────────────────────────────────────────────────────────

def contexto_partido_completo(home: str, away: str) -> dict:
    """
    Devuelve el contexto de análisis completo construido desde los CSVs locales.
    Compatible en schema con api_football.contexto_partido_completo.

    Sin lesiones (los CSVs no tienen esa información).
    """
    df = _cargar_df()
    notas: list[str] = []
    api_disponible = not df.empty

    if df.empty:
        notas.append("CSV provider: no hay archivos en data/raw/. Ejecuta data_loader.py")
        return {
            "api_disponible": False,
            "fuente":         "csv",
            "home": home, "away": away,
            "home_id": None, "away_id": None,
            "forma_home": None, "forma_away": None,
            "h2h": None,
            "injuries_home": [], "injuries_away": [],
            "notas": notas,
        }

    nombre_h = buscar_nombre_equipo(home)
    nombre_a = buscar_nombre_equipo(away)

    if nombre_h is None:
        notas.append(f"CSV: equipo '{home}' no encontrado en los CSVs locales")
    if nombre_a is None:
        notas.append(f"CSV: equipo '{away}' no encontrado en los CSVs locales")

    forma_home = get_team_form_csv(home, last=5)  if nombre_h else None
    forma_away = get_team_form_csv(away, last=5)  if nombre_a else None
    h2h        = get_h2h_csv(home, away, last=10) if (nombre_h and nombre_a) else None

    if forma_home is None and nombre_h:
        notas.append(f"CSV: sin partidos recientes para '{home}'")
    if forma_away is None and nombre_a:
        notas.append(f"CSV: sin partidos recientes para '{away}'")
    if h2h is None and nombre_h and nombre_a:
        notas.append(f"CSV: sin H2H registrado entre '{home}' y '{away}'")

    return {
        "api_disponible": api_disponible,
        "fuente":         "csv",
        "home":           home,
        "away":           away,
        "home_id":        nombre_h,    # en CSV usamos nombre, no ID numérico
        "away_id":        nombre_a,
        "forma_home":     forma_home,
        "forma_away":     forma_away,
        "h2h":            h2h,
        "injuries_home":  [],          # no disponible en CSVs
        "injuries_away":  [],
        "notas":          notas,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de diagnóstico
# ──────────────────────────────────────────────────────────────────────────────

def info_cobertura() -> dict:
    """Devuelve estadísticas de los CSVs cargados (útil para debug)."""
    df = _cargar_df()
    if df.empty:
        return {"partidos": 0, "equipos": 0, "ligas": [], "rango_fechas": None}
    return {
        "partidos": len(df),
        "equipos":  len(_todos_los_equipos()),
        "ligas":    sorted(df["_liga"].unique().tolist()),
        "rango_fechas": {
            "desde": df["Date"].min().date().isoformat(),
            "hasta": df["Date"].max().date().isoformat(),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Ejecución directa — ejemplo de uso y smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=== CSV Provider — Cobertura ===")
    print(json.dumps(info_cobertura(), indent=2, ensure_ascii=False))

    print("\n=== Búsqueda de equipos ===")
    for nombre in ["Real Madrid", "Barcelona", "Man United", "Arsenal",
                   "Bayern", "PSG", "Juventus", "Atletico"]:
        encontrado = buscar_nombre_equipo(nombre)
        print(f"  {nombre!r:20s} → {encontrado!r}")

    print("\n=== Contexto Real Madrid vs Barcelona ===")
    ctx = contexto_partido_completo("Real Madrid", "Barcelona")
    print(f"  fuente       : {ctx['fuente']}")
    print(f"  api_disponible: {ctx['api_disponible']}")
    print(f"  home_id      : {ctx['home_id']}")
    print(f"  away_id      : {ctx['away_id']}")
    if ctx["forma_home"]:
        fh = ctx["forma_home"]
        print(f"  forma_home   : {fh['secuencia']} GF={fh['gf_promedio']} GC={fh['gc_promedio']}")
    if ctx["forma_away"]:
        fa = ctx["forma_away"]
        print(f"  forma_away   : {fa['secuencia']} GF={fa['gf_promedio']} GC={fa['gc_promedio']}")
    if ctx["h2h"]:
        h2 = ctx["h2h"]
        print(f"  h2h          : n={h2['n']} goles={h2['goles_promedio']} "
              f"W={h2['wins_local_actual']} D={h2['empates']} L={h2['wins_visit_actual']}")
    if ctx["notas"]:
        print(f"  notas        : {ctx['notas']}")
