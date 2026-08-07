"""
Gestor de ligas con fallback inteligente.

Prioridad de fuentes (sin API-Football):
1. TheSportsDB (fallback gratis, recomendado)
2. JSON local (respaldo final, actualizado vía fetch_leagues.py)

API-Football descartado por limitaciones de cuenta.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger("betbrain.league_manager")


class LeagueManager:
    """Gestor de ligas y equipos con fallback multi-fuente."""

    def __init__(self):
        self.cache = {}
        self._load_json_backup()

    def _load_json_backup(self):
        """Carga equipos desde JSON local como respaldo."""
        json_path = Path(__file__).parent.parent / "data" / "equipos_por_liga.json"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Filtrar _meta
                self.cache = {k: v for k, v in data.items() if not k.startswith("_")}
            _log.info(f"✅ JSON local cargado: {len(self.cache)} ligas")
        except Exception as e:
            _log.error(f"Error cargando JSON: {e}")
            self.cache = {}

    def get_teams(self, liga: str, force_refresh: bool = False) -> List[str]:
        """
        Obtiene equipos de una liga.

        NOTA: TheSportsDB tiene problemas con mapeo de ligas, usamos JSON local como fuente principal.

        Args:
            liga: Nombre de la liga
            force_refresh: Si True, recarga JSON del disco

        Returns:
            Lista de equipos ordenada alfabéticamente
        """
        # PRIMERO: Usar JSON local (es confiable y actualizado)
        equipos_local = self.cache.get(liga, [])
        if equipos_local:
            _log.info(f"✅ Liga '{liga}' obtenida del JSON local ({len(equipos_local)} equipos)")
            return sorted(equipos_local)

        _log.warning(f"⚠️ Liga '{liga}' no encontrada en JSON local")
        return []

    def _get_from_api_football(self, liga: str) -> Optional[List[str]]:
        """API-Football descartado por limitaciones de cuenta."""
        _log.debug(f"API-Football no disponible para '{liga}' — usando fallback a TheSportsDB/JSON")
        return None

    def _get_from_thesportsdb(self, liga: str) -> Optional[List[str]]:
        """Obtiene equipos desde TheSportsDB (gratis, sin key, tiempo real)."""
        try:
            import requests
            from src.providers.fetch_leagues import LIGAS_CONFIG

            # Verificar si la liga está configurada
            if liga not in LIGAS_CONFIG:
                _log.debug(f"Liga '{liga}' no configurada en TheSportsDB")
                return None

            league_id, league_name = LIGAS_CONFIG[liga]

            # Hacer petición a TheSportsDB
            url = f"https://www.thesportsdb.com/api/v1/json/123/lookup_all_teams.php?id={league_id}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                _log.debug(f"TheSportsDB retornó {response.status_code} para '{liga}'")
                return None

            data = response.json()
            teams = data.get("teams", [])

            if not teams:
                _log.debug(f"TheSportsDB no devolvió equipos para '{liga}'")
                return None

            # Extraer nombres de equipos
            team_names = [t.get("strTeam") for t in teams if t.get("strTeam")]

            if team_names:
                _log.info(f"✅ Liga '{liga}' obtenida de TheSportsDB en tiempo real ({len(team_names)} equipos)")
                return team_names

        except requests.exceptions.Timeout:
            _log.debug(f"Timeout conectando a TheSportsDB para '{liga}'")
        except requests.exceptions.ConnectionError:
            _log.debug(f"Error de conexión a TheSportsDB para '{liga}'")
        except Exception as e:
            _log.debug(f"Error en TheSportsDB para '{liga}': {type(e).__name__}: {e}")

        return None

    def refresh_from_json(self):
        """Recarga desde JSON local."""
        self._load_json_backup()
        _log.info("✅ Caché actualizado desde JSON")

    def listar_ligas(self) -> List[str]:
        """Devuelve lista de ligas disponibles."""
        return sorted(self.cache.keys())

    def __repr__(self) -> str:
        return f"LeagueManager({len(self.cache)} ligas)"


# Singleton global
_manager = None


def get_league_manager() -> LeagueManager:
    """Devuelve la instancia global de LeagueManager."""
    global _manager
    if _manager is None:
        _manager = LeagueManager()
    else:
        # Recargar JSON por si se actualizó
        _manager._load_json_backup()
    return _manager


def get_teams(liga: str, force_refresh: bool = False) -> List[str]:
    """Helper para obtener equipos de una liga."""
    return get_league_manager().get_teams(liga, force_refresh)


def listar_ligas() -> List[str]:
    """Helper para listar ligas disponibles."""
    return get_league_manager().listar_ligas()
