"""
Keep-alive de Supabase: evita que el plan gratuito pause el proyecto.

Supabase pausa los proyectos del plan gratuito tras ~7 días sin actividad
(ya nos pasó una vez). Este script hace una consulta mínima y de SOLO
LECTURA sobre `apuestas` -- un count exacto del lado del servidor, sin
bajarse ninguna fila -- para que el proyecto siempre figure "en uso" aunque
esa semana nadie entre a la app.

No escribe, no borra y no toca ninguna otra ruta de producción: la única
operación es el SELECT del count.

Uso
---
    python src/scripts/supabase_keepalive.py

Sale con código 0 si el ping funcionó y 1 si falló (credenciales ausentes o
error de conexión/consulta), para que GitHub Actions marque la corrida en
rojo y el problema sea visible en la pestaña Actions.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.services import supabase_client as _sb


def _ahora() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> int:
    ts = _ahora()

    if not _sb.disponible():
        print(
            f"[{ts}] Ping FALLO - SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY no configurados",
            file=sys.stderr,
        )
        return 1

    try:
        filas = _sb.contar_apuestas()
    except Exception as exc:  # noqa: BLE001 -- cualquier fallo debe salir != 0
        print(f"[{ts}] Ping FALLO - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"[{ts}] Ping OK - apuestas tiene {filas} filas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
