"""
Gestor de jugadores de tenis con fallback inteligente.

Fuente única:
- JSON local (jugadores_por_genero.json, actualizado manualmente)
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger("betbrain.player_manager")


class PlayerManager:
    """Gestor de jugadores de tenis por género."""

    def __init__(self):
        self.cache = {}
        self._load_json_backup()

    def _load_json_backup(self):
        """Carga jugadores desde JSON local."""
        json_path = Path(__file__).parent.parent / "data" / "jugadores_por_genero.json"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Filtrar _meta
                self.cache = {k: v for k, v in data.items() if not k.startswith("_")}
            _log.info(f"✅ JSON de jugadores cargado: {len(self.cache)} géneros")
        except Exception as e:
            _log.error(f"Error cargando JSON de jugadores: {e}")
            self.cache = {}

    def get_players(self, genero: str, force_refresh: bool = False) -> List[str]:
        """
        Obtiene jugadores de un género (masculino/femenino).

        Args:
            genero: 'masculino' o 'femenino'
            force_refresh: Si True, recarga JSON del disco

        Returns:
            Lista de jugadores ordenada alfabéticamente
        """
        if force_refresh:
            self._load_json_backup()

        # Normalizar género
        genero_norm = genero.lower().strip() if genero else ""

        jugadores_local = self.cache.get(genero_norm, [])
        if jugadores_local:
            _log.info(f"✅ Género '{genero_norm}' obtenido del JSON ({len(jugadores_local)} jugadores)")
            return sorted(jugadores_local)

        _log.warning(f"⚠️ Género '{genero_norm}' no encontrado en JSON")
        return []

    def refresh_from_json(self):
        """Recarga desde JSON local."""
        self._load_json_backup()
        _log.info("✅ Caché de jugadores actualizado desde JSON")

    def listar_generos(self) -> List[str]:
        """Devuelve lista de géneros disponibles."""
        return sorted(self.cache.keys())

    def __repr__(self) -> str:
        return f"PlayerManager({len(self.cache)} géneros)"


# Singleton global
_manager = None


def get_player_manager() -> PlayerManager:
    """Devuelve la instancia global de PlayerManager."""
    global _manager
    if _manager is None:
        _manager = PlayerManager()
    else:
        # Recargar JSON por si se actualizó
        _manager._load_json_backup()
    return _manager


def get_players(genero: str, force_refresh: bool = False) -> List[str]:
    """Helper para obtener jugadores de un género."""
    return get_player_manager().get_players(genero, force_refresh)


def listar_generos() -> List[str]:
    """Helper para listar géneros disponibles."""
    return get_player_manager().listar_generos()
