#!/usr/bin/env python3
"""
Calibración simple de Elo usando datos sintéticos.
"""
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.tennis_elo import TennisEloCalculator

OUTPUT_FILE = Path(__file__).parent / "src" / "data" / "tennis_elo_ratings.json"
DATA_FILE = Path(__file__).parent / "src" / "data" / "tennis" / "matches_atp_sintetic.csv"


def calibrar_elo_desde_csv():
    """Calibra Elo desde CSV de matches."""

    print("=" * 70)
    print("CALIBRANDO ELO DE TENISTAS")
    print("=" * 70)

    # Crear calculador
    print("\n[PASO 1] Inicializando calculador Elo...")
    elo_calc = TennisEloCalculator()

    # Cargar matches
    print(f"\n[PASO 2] Cargando matches desde {DATA_FILE}...")
    if not DATA_FILE.exists():
        print(f"❌ Archivo no existe: {DATA_FILE}")
        return False

    matches = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            matches.append(row)

    print(f"✅ {len(matches)} matches cargados")

    # Procesar matches
    print(f"\n[PASO 3] Calibrando Elo ({len(matches)} matches)...")
    for i, match in enumerate(matches):
        try:
            winner = match.get('winner_name', '').strip()
            loser = match.get('loser_name', '').strip()
            level = match.get('tourney_level', 'ATP 250')

            if not winner or not loser:
                continue

            # Extraer sets del score
            score = match.get('score', '')
            parts = score.split() if score else []

            # Contar sets
            sets_w = 0
            sets_l = 0
            for j in range(0, len(parts), 2):
                if j + 1 < len(parts):
                    g1, g2 = int(parts[j]), int(parts[j + 1])
                    if g1 > g2:
                        sets_w += 1
                    else:
                        sets_l += 1

            if sets_w == 0 and sets_l == 0:
                sets_w = 2

            # Actualizar Elo
            elo_calc.update_elo(winner, loser, level, sets_w, sets_l)

            if (i + 1) % 100 == 0:
                print(f"  Procesados {i + 1}/{len(matches)}...")

        except Exception as e:
            print(f"❌ Error en match {i}: {e}")
            continue

    print(f"✅ {len(elo_calc.players)} jugadores calibrados")

    # Exportar
    print(f"\n[PASO 4] Exportando Elo ratings...")

    ratings = {
        "jugadores": {},
        "_meta": {
            "fecha": datetime.now().isoformat(),
            "total_jugadores": len(elo_calc.players),
            "matches_procesados": len(matches),
            "metodo": "Elo dinámico",
        }
    }

    for name, player in elo_calc.players.items():
        ratings["jugadores"][name] = {
            "elo": round(player.elo, 1),
            "games": player.games_played,
        }

    # Guardar
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ratings, f, indent=2, ensure_ascii=False)

    print(f"✅ Guardado: {OUTPUT_FILE}")

    # Top 15
    print(f"\n[TOP 15 JUGADORES]")
    top = elo_calc.get_ranking(15)
    for rank, (name, elo) in enumerate(top, 1):
        player = elo_calc.players[name]
        print(f"  {rank:2d}. {name:30s} ELO {elo:7.1f} ({player.games_played} games)")

    return True


if __name__ == "__main__":
    try:
        success = calibrar_elo_desde_csv()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
