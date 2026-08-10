"""
Cliente de Supabase para persistencia de apuestas en despliegues serverless
(Vercel), donde el filesystem local no persiste entre invocaciones.

El esquema de la tabla `apuestas` replica exactamente src.services.tracking.COLUMNS
para que calcular_metricas() y el resto de la lógica de negocio en tracking.py
no tengan que cambiar — solo cambia de dónde se leen/escriben las filas.
Ver supabase/schema.sql para el DDL.
"""
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

_client = None


def disponible() -> bool:
    """True si hay credenciales de Supabase configuradas."""
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


def _get_client():
    """Cliente de Supabase (lazy, usa la service_role key: solo backend)."""
    global _client
    if _client is None:
        if not disponible():
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY no configurados")
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client


def insertar_apuesta(fila: dict) -> dict:
    """Inserta una fila en `apuestas` (mismas claves que tracking.COLUMNS, sin 'id')."""
    fila = {k: v for k, v in fila.items() if k != "id"}
    result = _get_client().table("apuestas").insert(fila).execute()
    return result.data[0] if result.data else {}


def leer_apuestas() -> list[dict]:
    """Todas las filas de `apuestas`, más recientes primero."""
    result = _get_client().table("apuestas").select("*").order("id", desc=True).execute()
    return result.data or []


def actualizar_apuesta(id_apuesta: int, cambios: dict) -> Optional[dict]:
    """Actualiza columnas de una apuesta por id. None si no existía."""
    result = (
        _get_client().table("apuestas")
        .update(cambios).eq("id", id_apuesta).execute()
    )
    return result.data[0] if result.data else None


def eliminar_apuesta(id_apuesta: int) -> bool:
    """Elimina una apuesta por id. True si existía y se borró."""
    result = (
        _get_client().table("apuestas")
        .delete().eq("id", id_apuesta).execute()
    )
    return bool(result.data)


def insertar_prediccion(fila: dict) -> dict:
    """Inserta una fila en `predicciones_tenis` (mismas claves que
    tennis_predictions.COLUMNS, sin 'id')."""
    fila = {k: v for k, v in fila.items() if k != "id"}
    result = _get_client().table("predicciones_tenis").insert(fila).execute()
    return result.data[0] if result.data else {}


def leer_predicciones_tenis() -> list[dict]:
    """Todas las filas de `predicciones_tenis`, orden cronológico (igual
    que leer el CSV de arriba a abajo) — quien llama decide si invertir."""
    result = (
        _get_client().table("predicciones_tenis").select("*")
        .order("id", desc=False).execute()
    )
    return result.data or []


def actualizar_prediccion(id_prediccion: int, cambios: dict) -> Optional[dict]:
    """Actualiza columnas de una predicción por id. None si no existía."""
    result = (
        _get_client().table("predicciones_tenis")
        .update(cambios).eq("id", id_prediccion).execute()
    )
    return result.data[0] if result.data else None


def leer_config() -> Optional[dict]:
    """Fila única de `bankroll_config` (id=1). None si aún no existe."""
    result = (
        _get_client().table("bankroll_config").select("*").eq("id", 1).execute()
    )
    return result.data[0] if result.data else None


def guardar_config(cfg: dict) -> dict:
    """Upsert de la fila única de `bankroll_config`."""
    payload: dict[str, Any] = {**cfg, "id": 1}
    result = _get_client().table("bankroll_config").upsert(payload).execute()
    return result.data[0] if result.data else payload
