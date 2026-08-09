#!/usr/bin/env python3
"""
Calibra el std_dev de la distribución Normal de "total de games" del motor
de tenis (`TennisImprovedEngine.prob_total_games`) contra partidos reales,
en vez de la constante hardcodeada (4.5 para BO3, 6.0 para BO5).

Por qué la media no se toca: `total_esp` (la media) ya se calcula por
partido a partir de la probabilidad de Elo (`prob_total_games()` en
tennis_improved.py) — varía según cuán parejo esté el enfrentamiento. Lo
que estaba mal era el ANCHO de esa distribución (std_dev), fijo sin importar
los jugadores. Este script calcula el std_dev empírico real de "games
totales por partido completado", separado por formato (BO3/BO5), a partir
del mismo histórico real usado para calibrar Elo (ver
src/providers/tennis_data_loader.py — la fuente original de Jeff Sackmann
en GitHub ya no existe, se usa un mirror activo).

Nota de método: se usa la desviación estándar muestral directa de los
games reales (estimador correcto y no sesgado para el ancho de una Normal),
en vez de ajustar el std_dev minimizando el error contra un puñado de
líneas de Over/Under arbitrarias — ese enfoque indirecto es más ruidoso
(depende de qué thresholds se elijan) y no aporta nada que el std muestral
no dé ya de forma más robusta. La tabla de validación al final compara
igual el % real de Over vs el predicho por el modelo calibrado, para
verificar que el ajuste es razonable.

Uso:
    python calibrate_tennis_std_dev.py
"""
import sys
import json
import logging
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.providers.tennis_data_loader import combinar_archivos, contar_games_totales

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
_log = logging.getLogger("calibrate_std_dev")

OUTPUT_FILE = Path(__file__).parent / "src" / "data" / "tennis_std_dev_calibrated.json"

DEFAULTS = {"best_of_3": 4.5, "best_of_5": 6.0}
MIN_MUESTRAS = 100  # por debajo de esto, no hay señal suficiente para calibrar

THRESHOLDS = {
    "best_of_3": [20.5, 22.5, 24.5, 26.5],
    "best_of_5": [26.5, 28.5, 30.5, 32.5],
}


def calibrar_formato(games: list[int], formato: str) -> dict:
    n = len(games)
    if n < MIN_MUESTRAS:
        _log.warning(f"{formato}: solo {n} partidos completos (< {MIN_MUESTRAS}), "
                      f"insuficiente para calibrar — se mantiene el default "
                      f"{DEFAULTS[formato]}.")
        return {"std_dev": DEFAULTS[formato], "mean": None, "n": n, "calibrado": False}

    mean_obs = float(np.mean(games))
    std_obs = float(np.std(games, ddof=1))  # ddof=1: desviación muestral, no poblacional

    _log.info(f"{formato} — n={n}  media={mean_obs:.2f}  std={std_obs:.2f}  "
              f"(antes hardcodeado: {DEFAULTS[formato]})")

    _log.info(f"  {'Línea':<10}{'Over real':<12}{'Over predicho':<16}{'Error abs'}")
    for th in THRESHOLDS[formato]:
        p_real = sum(1 for g in games if g > th) / n
        p_pred = 1.0 - norm.cdf(th, loc=mean_obs, scale=std_obs)
        _log.info(f"  {th:<10}{p_real*100:>7.1f}%    {p_pred*100:>10.1f}%      "
                  f"{abs(p_real - p_pred):.3f}")

    return {"std_dev": round(std_obs, 2), "mean": round(mean_obs, 2), "n": n, "calibrado": True}


def calibrar_std_dev():
    _log.info("=" * 70)
    _log.info("CALIBRANDO STD_DEV DE TOTAL DE GAMES CON DATOS REALES")
    _log.info("=" * 70)

    matches = combinar_archivos()
    if not matches:
        _log.error("No hay partidos locales — corré calibrate_tennis_elo.py primero "
                    "(descarga los CSV reales) o descargalos manualmente.")
        return False

    por_formato: dict[str, list[int]] = defaultdict(list)
    incompletos = 0
    for m in matches:
        total = contar_games_totales(m.get("score", ""))
        if total is None:
            incompletos += 1
            continue
        bo = (m.get("best_of") or "").strip()
        if bo == "3":
            por_formato["best_of_3"].append(total)
        elif bo == "5":
            por_formato["best_of_5"].append(total)

    _log.info(f"\n{len(matches)} partidos totales, {incompletos} incompletos "
              f"(RET/W/O/DEF, excluidos), "
              f"{len(por_formato['best_of_3'])} BO3 completos, "
              f"{len(por_formato['best_of_5'])} BO5 completos\n")

    resultado_bo3 = calibrar_formato(por_formato["best_of_3"], "best_of_3")
    resultado_bo5 = calibrar_formato(por_formato["best_of_5"], "best_of_5")

    config = {
        "STD_DEV_BO3": resultado_bo3["std_dev"],
        "STD_DEV_BO5": resultado_bo5["std_dev"],
        "_meta": {
            "fecha": datetime.now(timezone.utc).isoformat(),
            "metodo": "Desviación estándar muestral de games totales reales por "
                      "partido completo (RET/W/O/DEF excluidos), separado por best_of.",
            "best_of_3": resultado_bo3,
            "best_of_5": resultado_bo5,
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    _log.info(f"\nGuardado: {OUTPUT_FILE}")
    _log.info(f"STD_DEV_BO3: {config['STD_DEV_BO3']} (antes: {DEFAULTS['best_of_3']})")
    _log.info(f"STD_DEV_BO5: {config['STD_DEV_BO5']} (antes: {DEFAULTS['best_of_5']})")

    return True


if __name__ == "__main__":
    success = calibrar_std_dev()
    sys.exit(0 if success else 1)
