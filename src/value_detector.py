"""
Atajo CLI para analizar un partido pasándole cuotas y recibir los value bets.
Uso:
    python src/value_detector.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.engine import BettingEngine


def pedir_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).replace(",", "."))
        except ValueError:
            print("Número inválido, inténtalo de nuevo.")


def main():
    eng = BettingEngine()
    print("\n--- Análisis de value bet ---")
    home = input("Equipo local: ").strip()
    away = input("Equipo visitante: ").strip()

    print("\nCuotas 1X2:")
    c1 = pedir_float("  Cuota Local (1): ")
    cx = pedir_float("  Cuota Empate (X): ")
    c2 = pedir_float("  Cuota Visitante (2): ")

    cuotas = {"1X2": {"1": c1, "X": cx, "2": c2}}

    if input("\n¿Tienes cuotas de Over/Under 2.5? (s/n): ").lower().startswith("s"):
        o = pedir_float("  Cuota Over 2.5: ")
        u = pedir_float("  Cuota Under 2.5: ")
        cuotas["OU_2.5"] = {"Over": o, "Under": u}

    if input("\n¿Tienes cuotas de BTTS? (s/n): ").lower().startswith("s"):
        y = pedir_float("  Cuota BTTS Sí: ")
        n = pedir_float("  Cuota BTTS No: ")
        cuotas["BTTS"] = {"Yes": y, "No": n}

    res = eng.analizar(home, away, cuotas)

    print(f"\n📊 {res['partido']}")
    print(f"   Fuente: {res['fuente_prob_1x2']}")
    print(f"   Prob 1X2:  L {res['prob_1x2']['1']:.1%}  "
          f"X {res['prob_1x2']['X']:.1%}  "
          f"V {res['prob_1x2']['2']:.1%}")

    if not res["value_bets"]:
        print("\n❌ No se detectó value bet con EV ≥ MIN_EV.")
        return

    print("\n🎯 VALUE BETS DETECTADOS:")
    for vb in res["value_bets"]:
        print(f"   [{vb['mercado']}] {vb['seleccion']:<6} | "
              f"Cuota {vb['cuota']:.2f} | "
              f"Prob modelo {vb['prob_modelo']:.1%} vs mercado {vb['prob_implicita']:.1%} | "
              f"EV {vb['ev']:+.1%} | "
              f"Stake Kelly {vb['kelly_stake_pct']:.2%} del bankroll")


if __name__ == "__main__":
    main()
