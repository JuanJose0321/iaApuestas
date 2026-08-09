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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.tennis_elo import TennisEloCalculator
from src.providers.tennis_data_loader import combinar_archivos, extraer_sets, contar_games_totales
from src.engines.tennis_improved import TennisImprovedEngine
from calibrate_tennis_elo import mapear_nivel_torneo, FORMA_VENTANA, FORMA_MIN_PARTIDOS

SURFACES = ("hard", "clay", "grass", "carpet")

ELO_GAP_BUCKETS = [
    ("parejo (<50)", 0.0, 50.0),
    ("moderado (50-150)", 50.0, 150.0),
    ("desbalanceado (>150)", 150.0, float("inf")),
]

TOTAL_GAMES_THRESHOLDS = {
    "best_of_3": [20.5, 22.5, 24.5, 26.5],
    "best_of_5": [26.5, 28.5, 30.5, 32.5],
}


def _meses_entre(fecha_anterior: str, fecha_actual: str) -> float:
    """Meses (aproximados, 30.44 días) entre dos fechas ISO YYYY-MM-DD."""
    d1 = date.fromisoformat(fecha_anterior)
    d2 = date.fromisoformat(fecha_actual)
    return max(0.0, (d2 - d1).days / 30.44)


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


def _mae_total_games(predichos: List[float], reales: List[int]) -> float:
    return float(np.mean([abs(p - r) for p, r in zip(predichos, reales)]))


def _sesgo_total_games(predichos: List[float], reales: List[int]) -> float:
    """Sesgo medio: predicho - real. Positivo = el modelo sobreestima."""
    return float(np.mean([p - r for p, r in zip(predichos, reales)]))


def _brier_over_bajo_umbral(total_esp_list: List[float], std_dev: float,
                             reales: List[int], formato: str) -> Dict[float, float]:
    """
    Brier score de P(Over umbral) contra el resultado real, para los
    umbrales convencionales de ese formato (mismos que
    calibrate_tennis_std_dev.py) — valida el modelo COMPLETO de
    distribución (media + std), no solo la media.
    """
    from scipy.stats import norm
    resultado = {}
    for th in TOTAL_GAMES_THRESHOLDS[formato]:
        errores = []
        for total_esp, real in zip(total_esp_list, reales):
            p_over = 1.0 - norm.cdf(th, loc=total_esp, scale=std_dev)
            y = 1.0 if real > th else 0.0
            errores.append((p_over - y) ** 2)
        resultado[th] = round(float(np.mean(errores)), 5)
    return resultado


def _bucket_games(games_min: int) -> str:
    """Bucket por tamaño de muestra del jugador con MENOS partidos previos
    de los dos (ej. un qualifier/ITF contra un top-100) — para ver si el
    modelo degrada con gracia o queda sobreconfiado con muestra chica."""
    if games_min < 10:
        return "1. <10 (muestra muy chica)"
    if games_min < 20:
        return "2. 10-20 (muestra chica)"
    if games_min < 50:
        return "3. 20-50"
    return "4. 50+"


def _bucket_certeza(pred: float) -> str:
    """Bucket por certeza del modelo en el lado que favoreció, sin
    importar si acertó (max(pred, 1-pred)) — sirve para validar si los
    umbrales verde/amarillo (UMBRAL_VERDE=0.65, UMBRAL_AMARILLO=0.50)
    realmente separan tramos de distinta accuracy real, o son arbitrarios."""
    certeza = max(pred, 1.0 - pred)
    if certeza >= 0.80:
        return "4. 0.80+"
    if certeza >= 0.70:
        return "3. 0.70-0.80"
    if certeza >= 0.65:
        return "2. 0.65-0.70 (equiv. verde)"
    if certeza >= 0.55:
        return "1. 0.55-0.65 (equiv. amarillo)"
    return "0. 0.50-0.55"


BASELINE_DECAY_POR_MES = 0.25
BASELINE_H2H_WEIGHT = 0.18
BASELINE_H2H_MIN_PARTIDOS = 2


def ejecutar_backtest(shrink_k: Optional[float] = None,
                       evaluar_desde: Optional[str] = None,
                       usar_superficie: bool = False,
                       min_games_superficie: Optional[int] = None,
                       decay_por_mes: Optional[float] = None,
                       h2h_weight: Optional[float] = None,
                       h2h_min_partidos: int = 3,
                       retornar_registros: bool = False,
                       sobre_baseline: bool = False) -> Dict:
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

    decay_por_mes: si se pasa (>0), acerca el Elo de cada jugador hacia
    la media según los meses transcurridos desde su partido anterior
    (walk-forward: la fecha de referencia es siempre la del último
    partido REALMENTE jugado antes de este, ver aplicar_decay_inactividad).

    h2h_weight: si se pasa (>0), mezcla la señal de head-to-head sobre
    el ensemble ya calculado (ver prob_from_h2h). El historial de
    enfrentamientos directos se arma walk-forward por par de jugadores
    (solo partidos previos a este, nunca futuros) y solo se usa si hay
    al menos h2h_min_partidos enfrentamientos previos entre ese par
    específico.

    sobre_baseline: si True, combina TODO lo que se pase (shrink_k y/o
    usar_superficie) ENCIMA del baseline ya validado y activo en
    producción (decay=0.25 + H2H weight=0.18/min=2), en vez de probarlo
    en aislamiento contra un baseline viejo sin esas mejoras — evita
    repetir el error de medir shrink_elo/superficie contra un punto de
    referencia que ya no es el que ve producción. decay_por_mes/
    h2h_weight explícitos siguen ganando si se pasan junto con esto.
    """
    matches = combinar_archivos()
    hoy = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    matches = [m for m in matches if m["date"] != "0000-00-00" and m["date"] <= hoy]

    elo_calc = TennisEloCalculator()
    elo_calc_surf: Dict[str, TennisEloCalculator] = {s: TennisEloCalculator() for s in SURFACES}
    recientes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=FORMA_VENTANA))
    ultima_fecha: Dict[str, str] = {}
    h2h_record: Dict[frozenset, Dict[str, int]] = {}
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

        meses_inactivo_w = _meses_entre(ultima_fecha[winner], m["date"]) if winner in ultima_fecha else None
        meses_inactivo_l = _meses_entre(ultima_fecha[loser], m["date"]) if loser in ultima_fecha else None

        # H2H walk-forward: solo enfrentamientos previos a este partido
        pair_key = frozenset({winner, loser})
        h2h_prev = h2h_record.get(pair_key, {})
        h2h_total_prev = sum(h2h_prev.values())
        h2h_ganados_winner = h2h_prev.get(winner, 0)

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
            if sobre_baseline:
                mw = _predecir_combinado(
                    engine, elo_w, elo_l, winner, loser, superficie, formato,
                    games_w, games_l, meses_inactivo_w, meses_inactivo_l,
                    h2h_ganados_winner, h2h_total_prev,
                    shrink_k=shrink_k,
                    decay_por_mes=decay_por_mes if decay_por_mes is not None else BASELINE_DECAY_POR_MES,
                    h2h_weight=h2h_weight if h2h_weight is not None else BASELINE_H2H_WEIGHT,
                    h2h_min_partidos=h2h_min_partidos,
                    usar_superficie=usar_superficie, surf_calc=surf_calc,
                    min_games_superficie=min_games_superficie,
                )
            elif usar_superficie:
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
            elif h2h_weight is not None:
                # El baseline actual ya incluye el decay validado (0.25) —
                # se prueba H2H ENCIMA de eso, no en aislamiento contra un
                # baseline viejo sin decay.
                mw = engine.prob_match_winner_ensemble(
                    elo_w, elo_l, winner, loser, superficie, formato,
                    meses_inactivo1=meses_inactivo_w, meses_inactivo2=meses_inactivo_l,
                    decay_por_mes=0.25,
                    h2h_ganados_j1=h2h_ganados_winner, h2h_total=h2h_total_prev,
                    h2h_weight=h2h_weight, h2h_min_partidos=h2h_min_partidos,
                )
            elif decay_por_mes is not None:
                mw = engine.prob_match_winner_ensemble(
                    elo_w, elo_l, winner, loser, superficie, formato,
                    meses_inactivo1=meses_inactivo_w, meses_inactivo2=meses_inactivo_l,
                    decay_por_mes=decay_por_mes,
                )
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

            p_base = mw["debug"]["ensemble"]
            dist = engine.prob_total_games(p_base, formato)
            real_total_games = contar_games_totales(m.get("score", ""))

            registros.append({
                "pred": pred,
                "surface": superficie,
                "level": mapear_nivel_torneo(m.get("level", "")),
                "elo_gap_bucket": bucket,
                "p_base": p_base,
                "formato": formato,
                "total_esp": dist["total_esp"],
                "std_dev": dist["std_dev"],
                "real_total_games": real_total_games,
                "genero": m.get("genero", "M"),
                "games_bucket": _bucket_games(min(games_w, games_l)),
                "certeza_bucket": _bucket_certeza(pred),
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
        ultima_fecha[winner] = m["date"]
        ultima_fecha[loser] = m["date"]
        h2h_record.setdefault(pair_key, {})
        h2h_record[pair_key][winner] = h2h_record[pair_key].get(winner, 0) + 1

    preds = [r["pred"] for r in registros]

    resultado = {
        "n_partidos": len(registros),
        "shrink_k": shrink_k,
        "usar_superficie": usar_superficie,
        "decay_por_mes": decay_por_mes,
        "h2h_weight": h2h_weight,
        "h2h_min_partidos": h2h_min_partidos if h2h_weight is not None else None,
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
        "por_genero": _desglose(registros, "genero"),
        "por_muestra": _desglose(registros, "games_bucket"),
        "por_certeza": _desglose(registros, "certeza_bucket"),
        "total_games": _validar_total_games(registros),
    }
    if retornar_registros:
        resultado["_registros"] = registros
    return resultado


def _validar_total_games(registros: List[Dict]) -> Dict:
    """Valida el mercado de Total Games (total_esp + std_dev actuales del
    motor) contra los games reales — completos únicamente, RET/W/O/DEF
    excluidos porque contar_games_totales ya devuelve None para esos."""
    resultado = {}
    for formato in ("best_of_3", "best_of_5"):
        rs = [r for r in registros if r["formato"] == formato and r["real_total_games"] is not None]
        if not rs:
            resultado[formato] = {"n": 0}
            continue
        predichos = [r["total_esp"] for r in rs]
        reales = [r["real_total_games"] for r in rs]
        std_dev = rs[0]["std_dev"]
        resultado[formato] = {
            "n": len(rs),
            "mae": round(_mae_total_games(predichos, reales), 3),
            "sesgo": round(_sesgo_total_games(predichos, reales), 3),
            "brier_por_umbral": _brier_over_bajo_umbral(predichos, std_dev, reales, formato),
        }
    return resultado


def _predecir_con_k(engine, elo_w, elo_l, winner, loser, superficie, formato, games_w, games_l, k):
    """Permite probar un k de shrink distinto al default del módulo sin mutar estado global."""
    from src.engines.tennis_improved import shrink_elo
    elo_w_s = shrink_elo(elo_w, games_w, k=k)
    elo_l_s = shrink_elo(elo_l, games_l, k=k)
    return engine.prob_match_winner_ensemble(elo_w_s, elo_l_s, winner, loser, superficie, formato)


def _predecir_combinado(engine, elo_w, elo_l, winner, loser, superficie, formato,
                         games_w, games_l, meses_inactivo_w, meses_inactivo_l,
                         h2h_ganados_winner, h2h_total_prev,
                         shrink_k, decay_por_mes, h2h_weight, h2h_min_partidos,
                         usar_superficie, surf_calc, min_games_superficie):
    """Combina shrink_k y/o usar_superficie ENCIMA del baseline de
    producción (decay + H2H) en una sola llamada al engine, en vez de
    medir cada mejora en aislamiento contra un baseline que ya no es el
    real (ver docstring de sobre_baseline en ejecutar_backtest)."""
    from src.engines.tennis_improved import shrink_elo

    elo_w_calc, elo_l_calc = elo_w, elo_l
    if shrink_k is not None:
        elo_w_calc = shrink_elo(elo_w, games_w, k=shrink_k)
        elo_l_calc = shrink_elo(elo_l, games_l, k=shrink_k)

    kwargs = dict(
        meses_inactivo1=meses_inactivo_w, meses_inactivo2=meses_inactivo_l,
        decay_por_mes=decay_por_mes,
        h2h_ganados_j1=h2h_ganados_winner, h2h_total=h2h_total_prev,
        h2h_weight=h2h_weight, h2h_min_partidos=h2h_min_partidos,
    )
    if usar_superficie:
        elo_w_s = surf_calc.get_elo(winner)
        elo_l_s = surf_calc.get_elo(loser)
        games_w_s = surf_calc.players[winner].games_played if winner in surf_calc.players else 0
        games_l_s = surf_calc.players[loser].games_played if loser in surf_calc.players else 0
        kwargs.update(elo1_superficie=elo_w_s, elo2_superficie=elo_l_s,
                       games1_superficie=games_w_s, games2_superficie=games_l_s)
        if min_games_superficie is not None:
            kwargs["min_games_superficie"] = min_games_superficie

    return engine.prob_match_winner_ensemble(
        elo_w_calc, elo_l_calc, winner, loser, superficie, formato, **kwargs
    )


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
    parser.add_argument("--decay-por-mes", type=float, default=None,
                         help="Activa el decay de Elo por inactividad con esta tasa mensual")
    parser.add_argument("--h2h-weight", type=float, default=None,
                         help="Activa H2H con este peso (se aplica junto al decay ya validado)")
    parser.add_argument("--h2h-min-partidos", type=int, default=3,
                         help="Enfrentamientos previos mínimos para confiar en el H2H")
    parser.add_argument("--sobre-baseline", action="store_true",
                         help="Combina shrink-k/usar-superficie ENCIMA del baseline de "
                              "producción (decay=0.25 + H2H weight=0.18/min=2) en vez de "
                              "probarlos en aislamiento")
    args = parser.parse_args()

    resultado = ejecutar_backtest(shrink_k=args.shrink_k, evaluar_desde=args.evaluar_desde,
                                   usar_superficie=args.usar_superficie,
                                   decay_por_mes=args.decay_por_mes,
                                   h2h_weight=args.h2h_weight,
                                   h2h_min_partidos=args.h2h_min_partidos,
                                   sobre_baseline=args.sobre_baseline)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
