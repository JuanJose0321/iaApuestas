#!/usr/bin/env python3
"""
Descarga datos históricos reales de tenis y calibra Elo + forma reciente.

Pasos:
1. Descarga datos ATP/WTA reales (ver src/providers/tennis_data_loader.py
   para la fuente y por qué ya no es JeffSackmann/tennis_atp directamente)
2. Procesa partidos en orden cronológico
3. Calcula Elo dinámico y forma reciente (últimos N partidos) por jugador
4. Exporta a JSON para usar en la app

Uso:
    python calibrate_tennis_elo.py
"""
import sys
import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.tennis_elo import TennisEloCalculator
from src.providers.tennis_data_loader import (
    descargar_datos_tennis, combinar_archivos, extraer_sets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
_log = logging.getLogger("calibrate_elo")

OUTPUT_FILE = Path(__file__).parent / "src" / "data" / "tennis_elo_ratings.json"

FORMA_VENTANA = 10       # últimos N partidos considerados para "forma"
FORMA_MIN_PARTIDOS = 3   # con menos que esto, no se reporta forma (ruido)


def mapear_nivel_torneo(level_str: str) -> str:
    """Mapea nivel de torneo (ATP/WTA, cualquier convención de nombres) a K-factor."""
    level_str = (level_str or "").upper()

    if "GRAND" in level_str or "SLAM" in level_str:
        return "Grand Slam"
    elif "MASTERS 1000" in level_str or "1000" in level_str:
        return "Masters 1000"
    elif "500" in level_str:
        return "ATP 500"
    elif "250" in level_str or "CHALLENGER" in level_str:
        return "ATP 250"
    else:
        return "ATP 250"  # default


def calibrar_elo():
    """Descarga datos reales y calibra Elo + forma reciente."""

    _log.info("=" * 70)
    _log.info("CALIBRANDO ELO DE TENISTAS CON DATOS REALES")
    _log.info("=" * 70)

    _log.info("\n[PASO 1] Descargando datos históricos reales...")
    if not descargar_datos_tennis("ambos"):
        _log.warning("No se pudieron descargar archivos nuevos, "
                      "intentando con los que ya existan localmente...")

    _log.info("\n[PASO 2] Cargando y ordenando partidos por fecha...")
    matches = combinar_archivos()

    if not matches:
        _log.error("No hay partidos para procesar. Verifica la descarga "
                    "(ver src/providers/tennis_data_loader.py).")
        return False

    hoy = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    antes = len(matches)
    matches = [m for m in matches if m["date"] != "0000-00-00" and m["date"] <= hoy]
    if len(matches) != antes:
        _log.warning(f"Descartados {antes - len(matches)} partidos con fecha "
                      f"inválida o futura (error de origen en la fuente).")

    _log.info(f"{len(matches)} partidos cargados "
              f"({matches[0]['date']} a {matches[-1]['date']})")

    _log.info("\n[PASO 3] Calibrando Elo y forma reciente...")
    elo_calc = TennisEloCalculator()
    recientes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=FORMA_VENTANA))
    ultima_fecha: Dict[str, str] = {}

    procesados = 0
    omitidos = 0
    for i, match in enumerate(matches):
        try:
            winner = match["winner"]
            loser = match["loser"]

            winner_sets, loser_sets = extraer_sets(match.get("score", ""))
            if winner_sets == 0 and loser_sets == 0:
                winner_sets = 2  # score no parseable (RET temprano, W/O, etc.)

            level_mapped = mapear_nivel_torneo(match.get("level", "ATP 250"))

            elo_calc.update_elo(
                winner, loser,
                tournament_level=level_mapped,
                winner_sets=winner_sets,
                loser_sets=loser_sets,
            )

            recientes[winner].append(True)
            recientes[loser].append(False)
            ultima_fecha[winner] = match["date"]
            ultima_fecha[loser] = match["date"]
            procesados += 1

            if (i + 1) % 5000 == 0:
                _log.info(f"  Procesados {i + 1}/{len(matches)} partidos...")

        except Exception as e:
            omitidos += 1
            _log.debug(f"Error procesando partido {i}: {e}")
            continue

    _log.info(f"Calibración completada: {procesados} partidos procesados, "
              f"{omitidos} omitidos, {len(elo_calc.players)} jugadores")

    _log.info("\n[PASO 4] Exportando ratings + forma...")

    ratings_export = {
        "jugadores": {},
        "_meta": {
            "fecha": datetime.now(timezone.utc).isoformat(),
            "total_jugadores": len(elo_calc.players),
            "matches_procesados": procesados,
            "rango_fechas": f"{matches[0]['date']} a {matches[-1]['date']}",
            "metodo": "Elo dinámico con K-factor adaptativo + forma real (últimos "
                      f"{FORMA_VENTANA} partidos, mínimo {FORMA_MIN_PARTIDOS})",
            "fuente": "LuckyLoser91/TennisCourtLog (mirror activo del formato "
                      "Jeff Sackmann; JeffSackmann/tennis_atp y tennis_wta ya no "
                      "existen en GitHub — ver src/providers/tennis_data_loader.py)",
        }
    }

    for name, player_elo in elo_calc.players.items():
        entry = {
            "elo": round(player_elo.elo, 1),
            "games": player_elo.games_played,
        }
        historial = recientes.get(name)
        if historial and len(historial) >= FORMA_MIN_PARTIDOS:
            ganados = sum(1 for w in historial if w)
            total = len(historial)
            entry["forma"] = {
                "ganados": ganados,
                "perdidos": total - ganados,
                "porcentaje": round(100.0 * ganados / total, 1),
            }
        if name in ultima_fecha:
            entry["ultima_fecha"] = ultima_fecha[name]
        ratings_export["jugadores"][name] = entry

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ratings_export, f, indent=2, ensure_ascii=False)

    _log.info(f"Guardado: {OUTPUT_FILE}")

    con_forma = sum(1 for v in ratings_export["jugadores"].values() if "forma" in v)
    _log.info(f"Jugadores con forma real calculada: {con_forma}/{len(elo_calc.players)}")

    _log.info("\n[TOP 10 JUGADORES POR ELO]")
    top = elo_calc.get_ranking(10)
    for rank, (name, elo) in enumerate(top, 1):
        _log.info(f"  {rank:2d}. {name:30s} {elo:7.1f}")

    return True


if __name__ == "__main__":
    success = calibrar_elo()
    sys.exit(0 if success else 1)
