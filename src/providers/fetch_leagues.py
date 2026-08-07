"""
Script para actualizar automaticamente equipos por liga desde TheSportsDB.

Uso:
    python src/fetch_leagues.py          # Actualiza todas las ligas
    python src/fetch_leagues.py --ligas LaLiga "Premier League"  # Ligas especificas

Ejecutar periodicamente (ej: cada semana) para mantener datos frescos.
"""
import requests
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("fetch_leagues")

BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"

# Mapeo de ligas: nombre_local → (thesportsdb_id, nombre_en_API)
# IDs corregidos y verificados en TheSportsDB (abril 2026)
LIGAS_CONFIG = {
    "Premier League": ("133602", "English Premier League"),
    "Championship": ("133604", "English Championship"),
    "LaLiga": ("133613", "Spanish La Liga"),
    "Bundesliga": ("133610", "German Bundesliga"),
    "Serie A": ("133612", "Italian Serie A"),
    "Ligue 1": ("133618", "French Ligue 1"),
    "Eredivisie": ("133621", "Dutch Eredivisie"),
    "Primeira Liga": ("133622", "Portuguese Primeira Liga"),
    "Brasileirao": ("133616", "Brazilian Serie A"),
    "Liga MX": ("133614", "Mexican Liga MX"),
    "MLS": ("133703", "Major League Soccer"),
    "Champions League": ("133633", "UEFA Champions League"),
    "Europa League": ("133635", "UEFA Europa League"),
    "Liga Profesional Argentina": ("133620", "Argentine Primera Division"),
}


def get_teams_by_league(league_id, league_name):
    """
    Obtiene equipos de una liga desde TheSportsDB.

    Args:
        league_id: ID de la liga en TheSportsDB
        league_name: Nombre para logging

    Returns:
        Lista de nombres de equipos
    """
    try:
        url = f"{BASE_URL}/lookup_all_teams.php?id={league_id}"
        _log.info(f"Fetching {league_name} from TheSportsDB...")

        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            _log.error(f"HTTP {response.status_code} para {league_name}")
            return []

        data = response.json()
        teams = data.get("teams", [])

        if not teams:
            _log.warning(f"No teams found for {league_name}")
            return []

        # Extraer nombres de equipos
        team_names = [t.get("strTeam") for t in teams if t.get("strTeam")]
        _log.info(f"OK - {league_name}: {len(team_names)} equipos")

        return team_names

    except requests.exceptions.Timeout:
        _log.error(f"Timeout para {league_name}")
        return []
    except Exception as e:
        _log.error(f"Error en {league_name}: {e}")
        return []


def update_equipos_json(output_file=None, ligas=None):
    """
    Actualiza el archivo equipos_por_liga.json con datos frescos.

    Args:
        output_file: Ruta del JSON (default: data/equipos_por_liga.json)
        ligas: Lista de ligas a actualizar (default: todas)
    """
    if output_file is None:
        output_file = Path(__file__).parent.parent / "data" / "equipos_por_liga.json"

    if not output_file.exists():
        _log.error(f"Archivo no encontrado: {output_file}")
        return False

    # Cargar JSON existente como base
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        _log.error(f"Error cargando JSON: {e}")
        return False

    # Determinar cuales ligas actualizar
    ligas_a_actualizar = ligas if ligas else list(LIGAS_CONFIG.keys())

    # Actualizar cada liga
    for liga_nombre in ligas_a_actualizar:
        if liga_nombre not in LIGAS_CONFIG:
            _log.warning(f"Liga no reconocida: {liga_nombre}")
            continue

        league_id, tsdb_name = LIGAS_CONFIG[liga_nombre]

        # Especial: Champions League y Europa League no usan lookup_all_teams
        if liga_nombre in ["Champions League", "Europa League"]:
            _log.info(f"SKIP - {liga_nombre} (manual, no actualizable desde TheSportsDB)")
            continue

        # Obtener equipos
        teams = get_teams_by_league(league_id, liga_nombre)

        if teams:
            data[liga_nombre] = teams

    # Actualizar metadata
    data["_meta"]["actualizado"] = datetime.now().isoformat()
    data["_meta"]["nota"] = (
        "Actualizado automaticamente desde TheSportsDB (src/fetch_leagues.py). "
        "Champions League y Europa League se actualizan manualmente."
    )

    # Guardar
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _log.info(f"OK - JSON actualizado: {output_file}")
        return True
    except Exception as e:
        _log.error(f"Error guardando JSON: {e}")
        return False


def main():
    """Punto de entrada."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Actualizar equipos por liga desde TheSportsDB"
    )
    parser.add_argument(
        "--ligas",
        nargs="+",
        help="Ligas especificas a actualizar (default: todas)"
    )
    parser.add_argument(
        "--output",
        help="Ruta del archivo JSON de salida"
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    ligas = args.ligas if args.ligas else None

    success = update_equipos_json(output_path, ligas)

    if success:
        _log.info("OK - Actualizacion completada")
        return 0
    else:
        _log.error("ERROR - Actualizacion fallo")
        return 1


if __name__ == "__main__":
    sys.exit(main())
