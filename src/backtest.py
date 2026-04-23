"""
Backtest del modelo: simula apostar partido por partido con el set de test
y reporta ROI, yield, hit rate y curva de capital.

Uso:
    python src/backtest.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CALIBRATOR_PATH, BANKROLL_INICIAL, MIN_EV, KELLY_FRACTION
from src.model import preparar_dataset, split_temporal, FEATURES, MAPPING
from probability_engine import calcular_ev
from src.bankroll import kelly


def backtest(metodo: str = "kelly_frac",
             min_ev: float = MIN_EV,
             bankroll_inicial: float = BANKROLL_INICIAL) -> dict:
    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(f"Falta {CALIBRATOR_PATH}. Corre: python src/model.py")

    cal = joblib.load(CALIBRATOR_PATH)
    df = preparar_dataset()
    _, test = split_temporal(df, test_frac=0.2)
    test = test.sort_values("Date").reset_index(drop=True)

    probs = cal.predict_proba(test[FEATURES])
    cuotas = test[["B365H", "B365D", "B365A"]].to_numpy()
    resultados_reales = test["FTR"].map(MAPPING).to_numpy()

    bankroll = bankroll_inicial
    curva = [bankroll]
    apuestas_totales = 0
    ganadas = 0
    total_staked = 0.0
    pnl = 0.0

    for i in range(len(test)):
        # Revisar cada una de las 3 selecciones
        for sel_idx in range(3):
            p = probs[i, sel_idx]
            cuota = cuotas[i, sel_idx]
            ev = calcular_ev(p, cuota)

            if ev < min_ev:
                continue

            # Stake
            if metodo == "kelly_frac":
                frac = kelly(p, cuota, fraction=KELLY_FRACTION)
                stake = bankroll * frac
            elif metodo == "flat_1pct":
                stake = bankroll * 0.01
            elif metodo == "flat_2pct":
                stake = bankroll * 0.02
            else:
                raise ValueError(metodo)

            if stake <= 0:
                continue

            apuestas_totales += 1
            total_staked += stake

            if resultados_reales[i] == sel_idx:
                ganancia = stake * (cuota - 1)
                bankroll += ganancia
                pnl += ganancia
                ganadas += 1
            else:
                bankroll -= stake
                pnl -= stake

        curva.append(bankroll)

        if bankroll <= 0:
            print(f"💀 Bankroll agotado en el partido {i}")
            break

    return {
        "metodo": metodo,
        "min_ev": min_ev,
        "bankroll_inicial": bankroll_inicial,
        "bankroll_final": round(bankroll, 2),
        "apuestas": apuestas_totales,
        "total_staked": round(total_staked, 2),
        "pnl": round(pnl, 2),
        "roi": round(pnl / bankroll_inicial, 4) if bankroll_inicial else 0,
        "yield": round(pnl / total_staked, 4) if total_staked else 0,
        "hit_rate": round(ganadas / apuestas_totales, 4) if apuestas_totales else 0,
        "max_bankroll": round(max(curva), 2),
        "min_bankroll": round(min(curva), 2),
        "curva": curva,
    }


if __name__ == "__main__":
    print("🔬 Corriendo backtest (Kelly fraccional + EV mínimo)...")
    for metodo in ["kelly_frac", "flat_1pct", "flat_2pct"]:
        r = backtest(metodo=metodo)
        print(f"\n--- {metodo.upper()} ---")
        print(f"  Apuestas      : {r['apuestas']}")
        print(f"  Hit rate      : {r['hit_rate'] * 100:.2f}%")
        print(f"  Total staked  : {r['total_staked']:.2f}")
        print(f"  PnL           : {r['pnl']:+.2f}")
        print(f"  ROI           : {r['roi'] * 100:+.2f}%")
        print(f"  Yield         : {r['yield'] * 100:+.2f}%")
        print(f"  Bankroll final: {r['bankroll_final']:.2f}")
        print(f"  Max / Min     : {r['max_bankroll']:.2f} / {r['min_bankroll']:.2f}")
