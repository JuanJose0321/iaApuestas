#!/usr/bin/env python3
"""
Calibra la distribución Normal de "total de games" del motor de tenis
(`TennisImprovedEngine.prob_total_games`) contra partidos reales: el
ANCHO (std_dev, calibrado en P1) y la MEDIA (total_esp, calibrada acá).

Historia: `total_esp` era una fórmula heurística nunca calibrada
(`sets_esp * games_por_set`, con games_por_set fijo entre 10.0-10.375).
Un backtest walk-forward confirmó que sobreestimaba el total real en
+2.80 games (BO3) / +3.10 games (BO5), en 67.5% de los partidos — sesgo
sistemático, no ruido — y que el Brier score del modelo de Over/Under
en umbrales bajos (20.5 en BO3) era 0.31, PEOR que adivinar a ciegas
(0.25). Esto generaba EV artificialmente inflado (36-56% con cuotas
justas simétricas) en el mercado de Total Games. Ver
tennis_backtest_results.md.

Fix: en vez de siquiera intentar ajustar a mano los coeficientes de la
fórmula heurística, se reemplaza por una regresión lineal simple
calibrada contra el histórico real, walk-forward (reusa
`backtest_tennis.ejecutar_backtest(..., retornar_registros=True)`, que
ya corre la simulación completa con el mismo Elo/forma/decay/H2H que ve
producción hoy):

    total_esp = a + b * p_base*(1 - p_base)

p_base*(1-p_base) ("competitividad") es la misma variable que ya usaba
la fórmula vieja — se mantiene la simetría (no importa cuál jugador es
favorito, solo cuán parejo está el partido), solo se reemplazan los
coeficientes inventados por unos ajustados por mínimos cuadrados contra
games reales.

El std_dev (P1) usa la desviación estándar muestral directa de los
games reales — no se toca en este script, sigue siendo el estimador
correcto para el ancho.

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
from backtest_tennis import ejecutar_backtest

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


MIN_MUESTRAS_REGRESION = 100


def _total_esp_heuristico_original(p: float, formato: str) -> float:
    """
    Réplica AUTOCONTENIDA de la fórmula heurística original de
    total_esp (sets_esp*games_por_set), previa a esta calibración.

    Necesaria para que la comparación "antes/después" sea honesta al
    re-correr este script una segunda vez: si se leyera
    TennisImprovedEngine.prob_total_games() directamente, "antes" ya
    reflejaría los coeficientes calibrados de una corrida anterior (el
    archivo de salida ya existe) en vez de la fórmula vieja real —
    comparación circular. Esta copia no depende de ningún estado
    calibrado, así que "antes" siempre es la misma fórmula que estaba
    en producción hasta este fix, sin importar cuántas veces se corra.
    """
    q = 1.0 - p
    pq = p * q
    if formato == "best_of_5":
        sets_esp = 3.0 + 3.0 * pq
        competitiveness = min(2.0 * pq, 0.25)
        games_por_set = 10.5 + 2.0 * competitiveness
    else:
        sets_esp = 2.0 + 2.0 * pq
        competitiveness = min(2.0 * pq, 0.25)
        games_por_set = 10.0 + 1.5 * competitiveness
    return sets_esp * games_por_set


def calibrar_total_esp() -> tuple[dict, dict]:
    """
    Ajusta total_esp = a + b*p_base*(1-p_base) por mínimos cuadrados
    contra games reales, walk-forward (misma simulación de producción:
    Elo con burn-in + decay + H2H, todo ya validado).

    Returns:
        (coeficientes por formato, comparación antes/después por formato)
    """
    _log.info("Corriendo simulación walk-forward completa (Elo+forma+decay+H2H) "
              "para recolectar (p_base, total_games_real) por partido...")
    resultado_bt = ejecutar_backtest(evaluar_desde="2024-01-01", h2h_weight=0.18,
                                      h2h_min_partidos=2, retornar_registros=True)
    registros = resultado_bt["_registros"]

    coefs: dict = {}
    antes_despues: dict = {}
    for formato in ("best_of_3", "best_of_5"):
        rs = [r for r in registros if r["formato"] == formato and r["real_total_games"] is not None]
        n = len(rs)
        if n < MIN_MUESTRAS_REGRESION:
            _log.warning(f"{formato}: solo {n} partidos con p_base+total real, "
                          f"insuficiente para regresión (< {MIN_MUESTRAS_REGRESION}).")
            coefs[formato] = None
            continue

        pq = np.array([r["p_base"] * (1.0 - r["p_base"]) for r in rs])
        reales = np.array([r["real_total_games"] for r in rs], dtype=float)
        b, a = np.polyfit(pq, reales, 1)  # grado 1: [pendiente, ordenada]

        predichos_antes = np.array([_total_esp_heuristico_original(r["p_base"], formato) for r in rs])
        mae_antes = float(np.mean(np.abs(predichos_antes - reales)))
        sesgo_antes = float(np.mean(predichos_antes - reales))

        predichos_despues = a + b * pq
        mae_despues = float(np.mean(np.abs(predichos_despues - reales)))
        sesgo_despues = float(np.mean(predichos_despues - reales))

        _log.info(f"\n{formato} — n={n}")
        _log.info(f"  ANTES  (fórmula heurística nunca calibrada): "
                  f"sesgo={sesgo_antes:+.2f}  MAE={mae_antes:.2f}")
        _log.info(f"  DESPUÉS (regresión total_esp = {a:.2f} + {b:.2f}*p*q): "
                  f"sesgo={sesgo_despues:+.2f}  MAE={mae_despues:.2f}")

        coefs[formato] = {"a": round(float(a), 3), "b": round(float(b), 3)}
        antes_despues[formato] = {
            "n": n,
            "antes": {"sesgo": round(sesgo_antes, 3), "mae": round(mae_antes, 3)},
            "despues": {"sesgo": round(sesgo_despues, 3), "mae": round(mae_despues, 3)},
        }

    return coefs, antes_despues


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

    _log.info("\n" + "=" * 70)
    _log.info("CALIBRANDO total_esp (MEDIA) POR REGRESIÓN CONTRA GAMES REALES")
    _log.info("=" * 70)
    coefs_total_esp, antes_despues = calibrar_total_esp()

    config = {
        "STD_DEV_BO3": resultado_bo3["std_dev"],
        "STD_DEV_BO5": resultado_bo5["std_dev"],
        "TOTAL_ESP_BO3": coefs_total_esp.get("best_of_3"),
        "TOTAL_ESP_BO5": coefs_total_esp.get("best_of_5"),
        "_meta": {
            "fecha": datetime.now(timezone.utc).isoformat(),
            "metodo_std_dev": "Desviación estándar muestral de games totales reales por "
                              "partido completo (RET/W/O/DEF excluidos), separado por best_of.",
            "metodo_total_esp": "Regresión lineal total_esp = a + b*p_base*(1-p_base), "
                                "ajustada por mínimos cuadrados contra games reales, "
                                "walk-forward (Elo+forma+decay+H2H, mismo config de producción).",
            "best_of_3": resultado_bo3,
            "best_of_5": resultado_bo5,
            "total_esp_antes_despues": antes_despues,
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    _log.info(f"\nGuardado: {OUTPUT_FILE}")
    _log.info(f"STD_DEV_BO3: {config['STD_DEV_BO3']} (antes: {DEFAULTS['best_of_3']})")
    _log.info(f"STD_DEV_BO5: {config['STD_DEV_BO5']} (antes: {DEFAULTS['best_of_5']})")
    _log.info(f"TOTAL_ESP_BO3: {config['TOTAL_ESP_BO3']}")
    _log.info(f"TOTAL_ESP_BO5: {config['TOTAL_ESP_BO5']}")

    return True


if __name__ == "__main__":
    success = calibrar_std_dev()
    sys.exit(0 if success else 1)
