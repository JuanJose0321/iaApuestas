#!/usr/bin/env python3
"""
Backtesting walk-forward del motor de tenis (TennisImprovedEngine) contra
el histórico real de partidos.

"Walk-forward" significa: para predecir el partido N, solo se usa el Elo
y la forma calculados con los partidos 1..N-1 — nunca información del
partido N ni de partidos futuros. Después de predecir, el Elo/forma se
actualiza con el resultado real de N antes de pasar al partido N+1.

Mide Brier score, log-loss y accuracy del modelo de producción (el mismo
TennisImprovedEngine que usa app.py, no una reimplementación aparte),
contra dos baselines ingenuos, y desglosa por superficie / nivel de
torneo / paridad del enfrentamiento.

Uso:
    python backtest_tennis.py                  # sin shrinkage de Elo
    python backtest_tennis.py --shrink-k 20     # con shrinkage, k=20
"""
import argparse
import json
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.tennis_elo import TennisEloCalculator
from src.providers.tennis_data_loader import combinar_archivos, extraer_sets
from src.engines.tennis_improved import TennisImprovedEngine
from calibrate_tennis_elo import mapear_nivel_torneo, FORMA_VENTANA, FORMA_MIN_PARTIDOS

SURFACES = ("hard", "clay", "grass", "carpet")

ELO_GAP_BUCKETS = [
    ("parejo (<50)", 0.0, 50.0),
    ("moderado (50-150)", 50.0, 150.0),
    ("desbalanceado (>150)", 150.0, float("inf")),
]


def _forma_desde_historial(hist: deque) -> Dict:
    total = len(hist)
    ganados = sum(1 for w in hist if w)
    return {
        "ganados": ganados, "perdidos": total - ganados,
        "porcentaje": round(100.0 * ganados / total, 1) if total else 0.0,
    }


def _brier(preds: List[float]) -> float:
    """y_actual es siempre 1 (se predice P(gana el que realmente ganó))."""
    return float(np.mean([(p - 1.0) ** 2 for p in preds]))


def _log_loss(preds: List[float], eps: float = 1e-15) -> float:
    ps = np.clip(np.array(preds), eps, 1 - eps)
    return float(-np.mean(np.log(ps)))


def _accuracy(preds: List[float]) -> float:
    return float(np.mean([1.0 if p > 0.5 else (0.5 if p == 0.5 else 0.0) for p in preds]))


def ejecutar_backtest(shrink_k: Optional[float] = None,
                       evaluar_desde: Optional[str] = None,
                       usar_superficie: bool = False,
                       min_games_superficie: Optional[int] = None) -> Dict:
    """
    Corre el backtest walk-forward completo.

    shrink_k=None: sin regresión a la media (comportamiento original).
    shrink_k=X: aplica shrink_elo con esa constante k antes de predecir.

    evaluar_desde="YYYY-MM-DD": si se pasa, TODOS los partidos disponibles
    se usan para actualizar Elo/forma en orden cronológico (walk-forward),
    pero solo se puntúan (Brier/accuracy) los partidos con fecha >= este
    valor. Sirve para medir el efecto de darle al Elo más tiempo de
    converger ("burn-in") antes de la ventana que realmente importa,
    manteniendo la comparación sobre el mismo conjunto de partidos
    evaluados que sin burn-in.

    usar_superficie: si True, mantiene un TennisEloCalculator por
    superficie además del overall (walk-forward, cada uno solo se
    actualiza con partidos de su propia superficie) y usa el Elo de
    superficie en la predicción cuando hay suficiente muestra (ver
    elegir_elo_superficie/SURFACE_MIN_GAMES), con fallback al overall.
    """
    matches = combinar_archivos()
    hoy = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    matches = [m for m in matches if m["date"] != "0000-00-00" and m["date"] <= hoy]

    elo_calc = TennisEloCalculator()
    elo_calc_surf: Dict[str, TennisEloCalculator] = {s: TennisEloCalculator() for s in SURFACES}
    recientes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=FORMA_VENTANA))
    engine = TennisImprovedEngine()

    registros = []
    con_rank = 0
    rank_acierta = 0

    for m in matches:
        winner, loser = m["winner"], m["loser"]
        evaluar_este = evaluar_desde is None or m["date"] >= evaluar_desde

        elo_w = elo_calc.get_elo(winner)
        elo_l = elo_calc.get_elo(loser)
        games_w = elo_calc.players[winner].games_played if winner in elo_calc.players else 0
        games_l = elo_calc.players[loser].games_played if loser in elo_calc.players else 0

        # Forma walk-forward: solo lo visto ANTES de este partido
        hist_w = recientes.get(winner)
        hist_l = recientes.get(loser)
        engine.form_stats = {}
        if hist_w and len(hist_w) >= FORMA_MIN_PARTIDOS:
            engine.form_stats[winner] = _forma_desde_historial(hist_w)
        if hist_l and len(hist_l) >= FORMA_MIN_PARTIDOS:
            engine.form_stats[loser] = _forma_desde_historial(hist_l)

        formato = "best_of_5" if m.get("best_of") == "5" else "best_of_3"
        superficie = m.get("surface", "hard")
        surf_calc = elo_calc_surf.setdefault(superficie, TennisEloCalculator())

        if evaluar_este:
            if usar_superficie:
                elo_w_s = surf_calc.get_elo(winner)
                elo_l_s = surf_calc.get_elo(loser)
                games_w_s = surf_calc.players[winner].games_played if winner in surf_calc.players else 0
                games_l_s = surf_calc.players[loser].games_played if loser in surf_calc.players else 0
                kwargs = {}
                if min_games_superficie is not None:
                    kwargs["min_games_superficie"] = min_games_superficie
                mw = engine.prob_match_winner_ensemble(
                    elo_w, elo_l, winner, loser, superficie, formato,
                    elo1_superficie=elo_w_s, elo2_superficie=elo_l_s,
                    games1_superficie=games_w_s, games2_superficie=games_l_s,
                    **kwargs,
                )
            elif shrink_k is not None:
                mw = _predecir_con_k(engine, elo_w, elo_l, winner, loser, superficie, formato,
                                      games_w, games_l, shrink_k)
            else:
                mw = engine.prob_match_winner_ensemble(elo_w, elo_l, winner, loser, superficie, formato)

            pred = mw["prob_j1"]  # P(gana "winner", el ganador real)

            # Baseline de ranking oficial (accuracy, no probabilidad — el
            # rank no se puede convertir a probabilidad calibrada sin
            # inventar una fórmula arbitraria)
            try:
                wr = float(m.get("winner_rank") or "")
                lr = float(m.get("loser_rank") or "")
                con_rank += 1
                if wr < lr:  # menor ranking = mejor
                    rank_acierta += 1
            except (ValueError, TypeError):
                pass

            elo_gap = abs(elo_w - elo_l)
            bucket = next(b[0] for b in ELO_GAP_BUCKETS if b[1] <= elo_gap < b[2])

            registros.append({
                "pred": pred,
                "surface": superficie,
                "level": mapear_nivel_torneo(m.get("level", "")),
                "elo_gap_bucket": bucket,
            })

        # Actualizar Elo/forma DESPUÉS de predecir (walk-forward real)
        winner_sets, loser_sets = extraer_sets(m.get("score", ""))
        if winner_sets == 0 and loser_sets == 0:
            winner_sets = 2
        level_mapped = mapear_nivel_torneo(m.get("level", "ATP 250"))
        elo_calc.update_elo(winner, loser, tournament_level=level_mapped,
                             winner_sets=winner_sets, loser_sets=loser_sets)
        surf_calc.update_elo(winner, loser, tournament_level=level_mapped,
                              winner_sets=winner_sets, loser_sets=loser_sets)
        recientes[winner].append(True)
        recientes[loser].append(False)

    preds = [r["pred"] for r in registros]

    resultado = {
        "n_partidos": len(registros),
        "shrink_k": shrink_k,
        "usar_superficie": usar_superficie,
        "brier": round(_brier(preds), 5),
        "log_loss": round(_log_loss(preds), 5),
        "accuracy": round(_accuracy(preds) * 100, 2),
        "baseline_coinflip": {"brier": 0.25, "log_loss": round(float(np.log(2)), 5), "accuracy": 50.0},
        "baseline_ranking_oficial": {
            "n_con_rank": con_rank,
            "accuracy": round(100.0 * rank_acierta / con_rank, 2) if con_rank else None,
        },
        "por_superficie": _desglose(registros, "surface"),
        "por_nivel_torneo": _desglose(registros, "level"),
        "por_paridad": _desglose(registros, "elo_gap_bucket"),
    }
    return resultado


def _predecir_con_k(engine, elo_w, elo_l, winner, loser, superficie, formato, games_w, games_l, k):
    """Permite probar un k de shrink distinto al default del módulo sin mutar estado global."""
    from src.engines.tennis_improved import shrink_elo
    elo_w_s = shrink_elo(elo_w, games_w, k=k)
    elo_l_s = shrink_elo(elo_l, games_l, k=k)
    return engine.prob_match_winner_ensemble(elo_w_s, elo_l_s, winner, loser, superficie, formato)


def _desglose(registros: List[Dict], campo: str) -> Dict:
    grupos: Dict[str, List[float]] = defaultdict(list)
    for r in registros:
        grupos[r[campo]].append(r["pred"])
    return {
        k: {"n": len(v), "brier": round(_brier(v), 5), "log_loss": round(_log_loss(v), 5),
            "accuracy": round(_accuracy(v) * 100, 2)}
        for k, v in sorted(grupos.items())
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shrink-k", type=float, default=None,
                         help="Constante k de shrinkage de Elo (default: sin shrinkage)")
    parser.add_argument("--evaluar-desde", type=str, default=None,
                         help="YYYY-MM-DD: usa partidos anteriores solo para calentar el Elo "
                              "(burn-in), puntúa solo desde esta fecha")
    parser.add_argument("--usar-superficie", action="store_true",
                         help="Usa Elo específico por superficie (con fallback al overall)")
    args = parser.parse_args()

    resultado = ejecutar_backtest(shrink_k=args.shrink_k, evaluar_desde=args.evaluar_desde,
                                   usar_superficie=args.usar_superficie)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
