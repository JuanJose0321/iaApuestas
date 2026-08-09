"""
Cargador de datos históricos ATP/WTA reales.

Fuente: LuckyLoser91/TennisCourtLog (GitHub), que replica la estructura y
formato del proyecto histórico de Jeff Sackmann (JeffSackmann/tennis_atp y
tennis_wta) — esos dos repos originales dejaron de existir en GitHub (404
confirmado en auditoría de 2026-08-08; la cuenta JeffSackmann hoy solo
tiene 1 repo público). TennisCourtLog mantiene el mismo esquema de columnas
y convención de nombres de archivo, y se actualiza activamente. Los CSV
están en Git LFS, por eso se sirven desde media.githubusercontent.com (el
host normal raw.githubusercontent.com solo devuelve el puntero LFS, no el
contenido).

Nota de procedencia: a diferencia de los datos originales de Sackmann
(CC BY-NC-SA), la licencia de este mirror no está declarada explícitamente
en GitHub ("Other/NOASSERTION"). Uso apropiado para esta herramienta
personal, no para redistribución.
"""
import csv
import logging
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List

_log = logging.getLogger("betbrain.tennis_loader")

GITHUB_LFS_BASE = "https://media.githubusercontent.com/media/LuckyLoser91/TennisCourtLog/main"
ATP_BASE = f"{GITHUB_LFS_BASE}/tennis_atp"
WTA_BASE = f"{GITHUB_LFS_BASE}/tennis_wta"

DATA_DIR = Path(__file__).parent.parent / "data" / "tennis"

_TIEBREAK_RE = re.compile(r"\(\d+\)")


def descargar_archivo(url: str, output_path: Path) -> bool:
    """Descarga un archivo CSV. Devuelve True si tuvo éxito."""
    try:
        _log.info(f"Descargando {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "BetBrain/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read().decode("utf-8")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(data)
            _log.info(f"Guardado: {output_path} ({len(data)} bytes)")
            return True
    except urllib.error.URLError as e:
        _log.error(f"Error descargando {url}: {e}")
        return False
    except Exception as e:
        _log.error(f"Error inesperado descargando {url}: {e}")
        return False


def descargar_datos_tennis(genero: str = "ambos", years: List[int] | None = None) -> bool:
    """
    Descarga histórico real de partidos ATP/WTA.

    Args:
        genero: "atp", "wta" o "ambos"
        years: años a descargar (default: últimos 12, incluyendo el actual)

    Returns:
        True si al menos un archivo se descargó exitosamente.

    Nota sobre el default de 12 años: con solo 3 años (2024-2026) el Elo
    arranca "en frío" (INITIAL_ELO=1500 para todo el mundo, ver
    src/core/tennis_elo.py) y no le da tiempo a converger — un backtest
    walk-forward confirmó que con esa ventana corta el modelo (61.0%
    accuracy) perdía contra el baseline ingenuo "gana el mejor ranking
    oficial" (63.45%). Con 12 años de burn-in (partidos anteriores a la
    ventana evaluada, usados solo para calentar el Elo) el modelo sube a
    63.7-63.8% y supera ese baseline. Ver tennis_backtest_results.md.
    Probar con 16 años no mejoró más (retornos decrecientes) — 12 es el
    punto de equilibrio entre precisión y tiempo de descarga/proceso.
    """
    if years is None:
        current_year = datetime.now().year
        years = list(range(current_year - 11, current_year + 1))

    success = False

    if genero in ("atp", "ambos"):
        for year in years:
            url = f"{ATP_BASE}/atp_matches_{year}.csv"
            filename = DATA_DIR / f"atp_matches_{year}.csv"
            if descargar_archivo(url, filename):
                success = True

    if genero in ("wta", "ambos"):
        for year in years:
            url = f"{WTA_BASE}/wta_matches_{year}.csv"
            filename = DATA_DIR / f"wta_matches_{year}.csv"
            if descargar_archivo(url, filename):
                success = True

    return success


def _fecha_iso(tourney_date: str) -> str:
    """
    Normaliza fechas para orden cronológico correcto.

    El mismo repo mezcla tres formatos entre archivos (confirmado
    inspeccionando los CSV descargados): 'YYYY-MM-DD' ya-formateado,
    'YYYY/M/D' sin zero-padding (TennisCourtLog en 2026) y 'YYYYMMDD'
    compacto (formato Sackmann clásico). Un sort lexicográfico directo de
    'YYYY/M/D' sin padding ordena mal (ej. '2026/8/3' quedaría después de
    '2026/12/1'), así que todo se normaliza a 'YYYY-MM-DD'.
    """
    s = (tourney_date or "").strip()
    if not s:
        return "0000-00-00"
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            y, m, d = s.split("-")
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        if "/" in s:
            y, m, d = s.split("/")
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        if len(s) == 8 and s.isdigit():
            return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    except ValueError:
        pass
    return "0000-00-00"


def extraer_sets(score_str: str) -> tuple[int, int]:
    """
    Cuenta sets ganados por cada jugador a partir del string de score real
    (ej. '6-4 3-6 7-6(4)', '6-2 6-1 RET', 'W/O').

    A diferencia de la versión anterior (que solo parseaba el formato
    numérico inventado por el generador de datos sintéticos, sin guiones),
    esta maneja guiones, tiebreaks entre paréntesis y retiros/walkovers.
    """
    s = (score_str or "").strip()
    if not s or s.upper() in ("W/O", "WALKOVER"):
        return (0, 0)

    sets_w = 0
    sets_l = 0
    for token in s.split():
        token = _TIEBREAK_RE.sub("", token).strip()
        if not token or "-" not in token:
            continue  # ignora tokens como 'RET', 'DEF', 'ABD'
        try:
            g1_str, g2_str = token.split("-", 1)
            g1, g2 = int(g1_str), int(g2_str)
        except ValueError:
            continue
        if g1 > g2:
            sets_w += 1
        elif g2 > g1:
            sets_l += 1

    return (sets_w, sets_l)


def contar_games_totales(score_str: str) -> int | None:
    """
    Suma el total de games jugados en un partido a partir del score real
    (ej. '6-4 7-6(3) 6-2' → 10 + 13 + 8 = 31).

    Devuelve None para partidos incompletos (retiro, walkover, default):
    su total de games no es representativo de "cuántos games se juegan si
    el partido se completa normalmente", que es la pregunta que responde
    el mercado de Total de Games — incluirlos sesgaría la calibración
    hacia abajo con partidos cortados a mitad de un set.
    """
    s = (score_str or "").strip()
    if not s:
        return None
    su = s.upper()
    if any(tag in su for tag in ("RET", "W/O", "DEF", "ABD", "WALKOVER")):
        return None

    total = 0
    for token in s.split():
        token = _TIEBREAK_RE.sub("", token).strip()
        if "-" not in token:
            return None
        try:
            g1_str, g2_str = token.split("-", 1)
            total += int(g1_str) + int(g2_str)
        except ValueError:
            return None
    return total


# Orden cronológico real de las rondas dentro de un mismo torneo. El CSV
# fuente comparte una única "tourney_date" (la fecha de INICIO del
# torneo) entre TODAS sus rondas — confirmado inspeccionando el Brisbane
# 2024 real: R32, R16, QF, SF y F aparecen todos con la misma fecha. Y el
# archivo lista las rondas en orden DESCENDENTE (Final primero, R32 al
# final). Sin esto, un sort por fecha sola (con Python "stable sort"
# preservando el orden del archivo en los empates) procesa la Final
# antes que la Primera Ronda del mismo torneo — walk-forward roto: el
# modelo "vería" el resultado de la Final antes de predecir partidos que
# en la realidad pasaron antes. Confirmado que esto invalidaba por
# completo un experimento de backtest (decay por inactividad) que
# explotaba exactamente este agujero.
_ORDEN_RONDA = {
    "RR": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "BR": 7, "F": 8,
}


def combinar_archivos(patron_prefijos: tuple[str, ...] = ("atp_matches_", "wta_matches_")) -> List[Dict]:
    """
    Combina todos los CSV de matches descargados en DATA_DIR, en orden
    cronológico (necesario para que el cálculo de Elo sea correcto).
    Dentro de partidos con la misma fecha (mismo torneo), ordena además
    por ronda real (_ORDEN_RONDA) — ver nota arriba.

    Returns:
        Lista de partidos como dicts con: date (YYYY-MM-DD), tournament,
        level, surface, winner, loser, score, best_of, winner_rank,
        loser_rank, round, genero ("M" o "F", derivado de si el archivo
        es atp_matches_* o wta_matches_* — señal confiable de la fuente,
        no una heurística por nombre).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    matches = []

    for csv_file in sorted(DATA_DIR.glob("*.csv")):
        if not csv_file.name.startswith(patron_prefijos):
            continue
        genero = "M" if csv_file.name.startswith("atp_matches_") else "F"
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    winner = (row.get("winner_name") or "").strip()
                    loser = (row.get("loser_name") or "").strip()
                    if not winner or not loser:
                        continue
                    matches.append({
                        "date": _fecha_iso(row.get("tourney_date", "")),
                        "tournament": row.get("tourney_name", ""),
                        "level": row.get("tourney_level", "ATP250"),
                        "surface": (row.get("surface") or "hard").lower(),
                        "winner": winner,
                        "loser": loser,
                        "score": row.get("score", ""),
                        "best_of": row.get("best_of", ""),
                        "winner_rank": row.get("winner_rank", ""),
                        "loser_rank": row.get("loser_rank", ""),
                        "round": row.get("round", ""),
                        "genero": genero,
                    })
                    count += 1
            _log.info(f"Cargados {count} partidos de {csv_file.name}")
        except Exception as e:
            _log.error(f"Error leyendo {csv_file}: {e}")

    matches.sort(key=lambda m: (m["date"], _ORDEN_RONDA.get(m["round"], 3)))
    return matches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _log.info("Descargando datos históricos reales de tenis...")
    if descargar_datos_tennis("ambos"):
        _log.info("Descarga completada")
        matches = combinar_archivos()
        _log.info(f"Total de partidos cargados: {len(matches)}")
    else:
        _log.error("No se pudieron descargar archivos")
