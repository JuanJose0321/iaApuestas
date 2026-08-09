#!/usr/bin/env python3
"""
Genera src/data/jugadores_por_genero.json DERIVADO de
src/data/tennis_elo_ratings.json — no se mantiene más a mano.

Antes: jugadores_por_genero.json era una lista curada manualmente (226
jugadores, "actualizado mayo 2026" según su propio _meta) que nunca se
sincronizó con el Elo real calibrado — 2,282 jugadores con Elo real
(incluyendo Nadal, Federer, Serena Williams, Osaka, Barty...) quedaron
fuera del selector de la app, invisibles para el usuario aunque el
motor sí tuviera su Elo calculado.

El género de cada jugador viene de tennis_elo_ratings.json (campo
"genero": "M"/"F"), que a su vez lo deriva calibrate_tennis_elo.py de
la fuente real (si el jugador jugó en archivos atp_matches_* o
wta_matches_*) — no es una heurística por nombre.

Se corre automáticamente al final de calibrate_tennis_elo.py, pero
también puede correrse solo (por ejemplo, si tennis_elo_ratings.json ya
está actualizado y solo hace falta regenerar el selector):

    python generar_jugadores_por_genero.py
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
_log = logging.getLogger("generar_jugadores_por_genero")

ELO_RATINGS_FILE = Path(__file__).parent / "src" / "data" / "tennis_elo_ratings.json"
OUTPUT_FILE = Path(__file__).parent / "src" / "data" / "jugadores_por_genero.json"


def generar(elo_ratings_path: Path = ELO_RATINGS_FILE,
            output_path: Path = OUTPUT_FILE) -> bool:
    """Lee tennis_elo_ratings.json y regenera jugadores_por_genero.json."""
    if not elo_ratings_path.exists():
        _log.error(f"No existe {elo_ratings_path} — corré calibrate_tennis_elo.py primero.")
        return False

    with open(elo_ratings_path, "r", encoding="utf-8") as f:
        ratings = json.load(f).get("jugadores", {})

    masculino = sorted(name for name, stats in ratings.items() if stats.get("genero") == "M")
    femenino = sorted(name for name, stats in ratings.items() if stats.get("genero") == "F")
    sin_genero = [name for name, stats in ratings.items() if stats.get("genero") not in ("M", "F")]

    if sin_genero:
        _log.warning(f"{len(sin_genero)} jugadores sin género derivado (Elo calibrado con una "
                      f"versión anterior sin este campo) — quedan fuera del selector. "
                      f"Recalibrá con calibrate_tennis_elo.py para incluirlos.")

    salida = {
        "masculino": masculino,
        "femenino": femenino,
        "_meta": {
            "total_generos": 2,
            "masculino_count": len(masculino),
            "femenino_count": len(femenino),
            "generado": f"derivado automáticamente de tennis_elo_ratings.json "
                        f"({datetime.now(timezone.utc).isoformat()})",
            "nota": "Ya no se mantiene a mano — se regenera en cada recalibración "
                    "de Elo (ver calibrate_tennis_elo.py, PASO 6).",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    _log.info(f"Guardado: {output_path} ({len(masculino)} masculino, {len(femenino)} femenino)")
    return True


if __name__ == "__main__":
    success = generar()
    sys.exit(0 if success else 1)
