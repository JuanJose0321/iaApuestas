#!/usr/bin/env python3
"""
Test de integración del motor mejorado de tenis.
Verifica que _get_tennis_engine() carga las ratings y analiza correctamente.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import _get_tennis_engine
from config import BANKROLL_INICIAL

def test_tennis_integration():
    print("=" * 70)
    print("TEST: Integración de Tennis Improved Engine")
    print("=" * 70)

    # Paso 1: Cargar el motor
    print("\n[PASO 1] Cargando motor de tenis...")
    try:
        engine = _get_tennis_engine()
        print(f"✅ Motor cargado exitosamente")
        print(f"   Elo ratings cargados: {len(engine.elo_ratings)}")
    except Exception as e:
        print(f"❌ Error cargando motor: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Paso 2: Verificar Elo ratings
    print("\n[PASO 2] Verificando Elo ratings...")
    top_players = sorted(engine.elo_ratings.items(), key=lambda x: x[1], reverse=True)[:5]
    for player, elo in top_players:
        print(f"   {player:30s} ELO {elo:7.1f}")

    # Paso 3: Test con Alcaraz vs Sinner
    print("\n[PASO 3] Analizando: Alcaraz vs Sinner...")
    try:
        j1, j2 = "Carlos Alcaraz", "Jannik Sinner"
        elo1 = engine.get_elo(j1)
        elo2 = engine.get_elo(j2)

        print(f"   {j1:30s} ELO {elo1:7.1f}")
        print(f"   {j2:30s} ELO {elo2:7.1f}")

        cuotas = {
            "match_winner": {"1": 1.85, "2": 1.95},
            "total_games": {"linea": 22.5, "over": 1.85, "under": 1.95}
        }

        resultado = engine.analizar(
            j1, j2, elo1, elo2,
            superficie="hard",
            formato="best_of_3",
            cuotas=cuotas,
            cuota_min=1.20,
            cuota_max=6.00
        )

        print(f"\n   Resultado del análisis:")
        print(f"   - Picks verdes: {len(resultado['picks_verdes'])}")
        print(f"   - Picks amarillos: {len(resultado['picks_amarillos'])}")
        print(f"   - Resumen: {resultado['resumen']}")

        # Mostrar detalles del modelo
        modelo = resultado['modelo']
        print(f"\n   Modelo: {modelo['metodo']}")
        print(f"   - P(J1 gana) ensemble: {modelo['p_base_j1']:.2%}")
        print(f"   - Match winner prob: {modelo['match_winner']['prob_j1']:.2%}")
        print(f"   - Total games esperado: {modelo['total_games']['total_esp']} games")

        # Mostrar picks si los hay
        if resultado['picks_verdes']:
            print(f"\n   ✅ PICKS VERDES:")
            for pick in resultado['picks_verdes']:
                print(f"      {pick['pick']:40s} Prob {pick['prob']:.2%} @ {pick['cuota']} EV {pick['ev']:+.2%}")

        if resultado['picks_amarillos']:
            print(f"\n   ⚠️  PICKS AMARILLOS:")
            for pick in resultado['picks_amarillos']:
                print(f"      {pick['pick']:40s} Prob {pick['prob']:.2%} @ {pick['cuota']} EV {pick['ev']:+.2%}")

        print(f"\n✅ Test completado exitosamente")
        return True

    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tennis_integration()
    sys.exit(0 if success else 1)
