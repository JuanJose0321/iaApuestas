#!/usr/bin/env python3
"""
Script para descargar datos históricos de tenis y calibrar Elo.

Pasos:
1. Descarga datos ATP/WTA 2024-2026
2. Procesa partidos en orden cronológico
3. Calcula Elo dinámico para cada jugador
4. Exporta a JSON para usar en la app

Uso:
    python calibrate_tennis_elo.py
"""
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.tennis_elo import TennisEloCalculator
from src.providers.tennis_data_loader import descargar_datos_tennis, combinar_archivos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
_log = logging.getLogger("calibrate_elo")

OUTPUT_FILE = Path(__file__).parent / "src" / "data" / "tennis_elo_ratings.json"


def mapear_nivel_torneo(level_str: str) -> str:
    """Mapea nivel de torneo a K-factor."""
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


def extraer_sets(score_str: str) -> tuple:
    """
    Extrae sets del score.

    Ej: "2 6 4 6 3 6" → (3, 2)
    """
    try:
        parts = score_str.strip().split()
        if not parts:
            return (0, 0)

        # Contar sets ganados (cada 2 números = 1 set)
        sets_j1 = 0
        sets_j2 = 0

        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                g1 = int(parts[i])
                g2 = int(parts[i + 1])
                if g1 > g2:
                    sets_j1 += 1
                else:
                    sets_j2 += 1

        return (sets_j1, sets_j2)
    except:
        return (0, 0)


def calibrar_elo():
    """Descarga datos y calibra Elo."""

    _log.info("=" * 70)
    _log.info("CALIBRANDO ELO DE TENISTAS")
    _log.info("=" * 70)

    # Paso 1: Descargar datos
    _log.info("\n[PASO 1] Descargando datos históricos...")
    if not descargar_datos_tennis("ambos"):
        _log.warning("⚠️ No se pudieron descargar algunos archivos, intentando con locales...")

    # Paso 2: Cargar y procesar matches
    _log.info("\n[PASO 2] Cargando partidos...")
    matches = combinar_archivos()

    if not matches:
        _log.error("❌ No hay partidos para procesar. Verifica la descarga.")
        return False

    _log.info(f"✅ {len(matches)} partidos cargados")

    # Paso 3: Ordenar por fecha (importante para Elo)
    _log.info("\n[PASO 3] Ordenando por fecha...")
    matches.sort(key=lambda x: x.get('date', ''))

    # Paso 4: Crear calculador Elo
    _log.info("\n[PASO 4] Calibrando Elo...")
    elo_calc = TennisEloCalculator()

    # Procesar cada match
    for i, match in enumerate(matches):
        try:
            winner = match.get('winner', '').strip()
            loser = match.get('loser', '').strip()

            if not winner or not loser:
                continue

            # Extraer sets
            score = match.get('score', '')
            winner_sets, loser_sets = extraer_sets(score)

            # Si no hay score, asumir 2-0
            if winner_sets == 0 and loser_sets == 0:
                winner_sets = 2

            # Mapear nivel de torneo
            level = match.get('level', 'ATP 250')
            level_mapped = mapear_nivel_torneo(level)

            # Actualizar Elo
            elo_calc.update_elo(
                winner, loser,
                tournament_level=level_mapped,
                winner_sets=winner_sets,
                loser_sets=loser_sets
            )

            # Progreso cada 1000 matches
            if (i + 1) % 1000 == 0:
                _log.info(f"  Procesados {i + 1}/{len(matches)} matches...")

        except Exception as e:
            _log.debug(f"Error procesando match {i}: {e}")
            continue

    _log.info(f"✅ Calibración completada: {len(elo_calc.players)} jugadores")

    # Paso 5: Exportar Elo ratings
    _log.info("\n[PASO 5] Exportando ratings...")

    ratings_export = {
        "jugadores": {},
        "_meta": {
            "fecha": datetime.now().isoformat(),
            "total_jugadores": len(elo_calc.players),
            "metodo": "Elo dinámico con K-factor adaptativo",
        }
    }

    for name, player_elo in elo_calc.players.items():
        ratings_export["jugadores"][name] = {
            "elo": round(player_elo.elo, 1),
            "games": player_elo.games_played,
        }

    # Crear directorio si no existe
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Guardar JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ratings_export, f, indent=2, ensure_ascii=False)

    _log.info(f"✅ Guardado: {OUTPUT_FILE}")

    # Mostrar top 10
    _log.info("\n[TOP 10 JUGADORES POR ELO]")
    top = elo_calc.get_ranking(10)
    for rank, (name, elo) in enumerate(top, 1):
        _log.info(f"  {rank:2d}. {name:30s} {elo:7.1f}")

    return True


if __name__ == "__main__":
    success = calibrar_elo()
    sys.exit(0 if success else 1)
