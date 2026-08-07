"""
Motor unificado de análisis de fútbol.
Combina dos fuentes de probabilidad:
  1) Modelo XGBoost calibrado (aprende de histórico + cuotas)
  2) Poisson (derivado de cuotas para mercados OU/BTTS/AH)
"""
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from itertools import combinations

import joblib
import numpy as np
import pandas as pd

_log = logging.getLogger("betbrain.engine")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import CALIBRATOR_PATH, MIN_EV, KELLY_FRACTION
from src.core.probability import (
    eliminar_vig, estimar_lambdas_desde_cuotas,
    generar_matriz_poisson, derivar_mercados, calcular_ev,
    prob_conjunta, prob_marginal, prob_conjunta_n, son_compatibles,
)
from src.providers.loader import cargar_todo
from src.core.features import calculate_rolling_stats
from src.core.model import FEATURES


def _es_combo_uniforme(combo: list[tuple]) -> tuple[bool, str]:
    for (m, s), cnt in Counter(combo).items():
        if cnt >= 2:
            return True, f"{m} {s}"
    return False, ""


@dataclass
class ValueBet:
    mercado: str
    seleccion: str
    prob_modelo: float
    cuota: float
    prob_implicita: float
    ev: float
    kelly_stake_pct: float


def _kelly_fraction(p: float, cuota: float, fraction: float = KELLY_FRACTION) -> float:
    b = cuota - 1
    if b <= 0:
        return 0.0
    f_full = (p * (b + 1) - 1) / b
    return max(0.0, f_full * fraction)


def _ultimo_estado_equipo(df_stats: pd.DataFrame, equipo: str) -> dict | None:
    mask_h = df_stats["HomeTeam"] == equipo
    mask_a = df_stats["AwayTeam"] == equipo
    partidos = df_stats[mask_h | mask_a].sort_values("Date")
    if partidos.empty:
        return None
    row = partidos.iloc[-1]
    if row["HomeTeam"] == equipo:
        return {"form": row["Home_Form_5"], "gf": row["Home_GF_5"], "gc": row["Home_GC_5"]}
    return {"form": row["Away_Form_5"], "gf": row["Away_GF_5"], "gc": row["Away_GC_5"]}


class BettingEngine:
    def __init__(self):
        self.calibrator = None
        self.df_stats = None
        self._cargar()

    def _cargar(self):
        if CALIBRATOR_PATH.exists():
            self.calibrator = joblib.load(CALIBRATOR_PATH)
        else:
            print(f"[WARN] No existe {CALIBRATOR_PATH}. Corre: python src/core/model.py")
        try:
            df = cargar_todo()
            self.df_stats = calculate_rolling_stats(df, window=5)
        except FileNotFoundError:
            print("[WARN] No hay datos descargados. Corre: python src/providers/loader.py")

    def prob_ml(self, home: str, away: str,
                cuota_h: float, cuota_d: float, cuota_a: float) -> dict | None:
        if self.calibrator is None or self.df_stats is None:
            return None
        h = _ultimo_estado_equipo(self.df_stats, home)
        a = _ultimo_estado_equipo(self.df_stats, away)
        if h is None or a is None:
            return None
        X = pd.DataFrame([{
            "Home_Form_5": h["form"], "Away_Form_5": a["form"],
            "Home_GF_5": h["gf"], "Away_GF_5": a["gf"],
            "Home_GC_5": h["gc"], "Away_GC_5": a["gc"],
            "B365H": cuota_h, "B365D": cuota_d, "B365A": cuota_a,
        }])[FEATURES]
        probs = self.calibrator.predict_proba(X)[0]
        return {"1": float(probs[0]), "X": float(probs[1]), "2": float(probs[2])}

    def prob_poisson(self, cuota_h: float, cuota_d: float, cuota_a: float,
                     promedio_goles_liga: float = 2.7) -> dict:
        sin_vig = eliminar_vig({"1": cuota_h, "X": cuota_d, "2": cuota_a})
        lh, la = estimar_lambdas_desde_cuotas(sin_vig["1"], sin_vig["2"], promedio_goles_liga)
        matriz = generar_matriz_poisson(lh, la, max_goles=8)
        mercados = derivar_mercados(matriz)
        mercados["_lambdas"] = {"home": lh, "away": la}
        mercados["_sin_vig"] = sin_vig
        return mercados

    def analizar(self, home: str, away: str, cuotas: dict,
                 promedio_goles_liga: float = 2.7) -> dict:
        c_1x2 = cuotas["1X2"]
        poisson = self.prob_poisson(c_1x2["1"], c_1x2["X"], c_1x2["2"], promedio_goles_liga)
        ml = self.prob_ml(home, away, c_1x2["1"], c_1x2["X"], c_1x2["2"])

        if ml:
            final_1x2 = {k: 0.6 * ml[k] + 0.4 * poisson["1X2"][k] for k in ("1", "X", "2")}
            fuente = "ensemble ML+Poisson"
        else:
            final_1x2 = poisson["1X2"].copy()
            fuente = "solo Poisson (modelo ML no disponible)"

        value_bets: list[ValueBet] = []

        for sel, cuota in c_1x2.items():
            p = final_1x2[sel]
            ev = calcular_ev(p, cuota)
            if ev >= MIN_EV:
                value_bets.append(ValueBet(
                    mercado="1X2", seleccion=sel, prob_modelo=p, cuota=cuota,
                    prob_implicita=1 / cuota, ev=ev, kelly_stake_pct=_kelly_fraction(p, cuota),
                ))

        if "OU_2.5" in cuotas:
            for sel, cuota in cuotas["OU_2.5"].items():
                p = poisson["OU_2.5"][sel]
                ev = calcular_ev(p, cuota)
                if ev >= MIN_EV:
                    value_bets.append(ValueBet(
                        mercado="OU_2.5", seleccion=sel, prob_modelo=p, cuota=cuota,
                        prob_implicita=1 / cuota, ev=ev, kelly_stake_pct=_kelly_fraction(p, cuota),
                    ))

        if "BTTS" in cuotas:
            for sel, cuota in cuotas["BTTS"].items():
                p = poisson["BTTS"][sel]
                ev = calcular_ev(p, cuota)
                if ev >= MIN_EV:
                    value_bets.append(ValueBet(
                        mercado="BTTS", seleccion=sel, prob_modelo=p, cuota=cuota,
                        prob_implicita=1 / cuota, ev=ev, kelly_stake_pct=_kelly_fraction(p, cuota),
                    ))

        return {
            "partido": f"{home} vs {away}",
            "fuente_prob_1x2": fuente,
            "prob_1x2": final_1x2,
            "prob_poisson_full": {
                "OU_2.5": poisson["OU_2.5"],
                "BTTS": poisson["BTTS"],
                "AH_-1.5_local": poisson["AH_-1.5_local"],
                "lambdas": poisson["_lambdas"],
            },
            "value_bets": [asdict(v) for v in sorted(value_bets, key=lambda v: v.ev, reverse=True)],
        }

    def pick_simple(self, home: str, away: str, cuotas: dict,
                    cuota_min: float = 1.70, cuota_max: float = 2.50,
                    promedio_goles_liga: float = 2.7) -> dict:
        c_1x2 = cuotas["1X2"]
        sin_vig = eliminar_vig({"1": c_1x2["1"], "X": c_1x2["X"], "2": c_1x2["2"]})
        lh, la = estimar_lambdas_desde_cuotas(sin_vig["1"], sin_vig["2"], promedio_goles_liga)
        matriz = generar_matriz_poisson(lh, la, max_goles=8)
        ml = self.prob_ml(home, away, c_1x2["1"], c_1x2["X"], c_1x2["2"])

        catalogo: dict[tuple, tuple[float, float]] = {}
        for sel in ("1", "X", "2"):
            p_poisson = prob_marginal(matriz, "1X2", sel)
            p_final = 0.6 * ml[sel] + 0.4 * p_poisson if ml else p_poisson
            catalogo[("1X2", sel)] = (p_final, c_1x2[sel])
        if "OU_2.5" in cuotas:
            for sel, cuota in cuotas["OU_2.5"].items():
                catalogo[("OU_2.5", sel)] = (prob_marginal(matriz, "OU_2.5", sel), cuota)
        if "BTTS" in cuotas:
            for sel, cuota in cuotas["BTTS"].items():
                catalogo[("BTTS", sel)] = (prob_marginal(matriz, "BTTS", sel), cuota)

        directa_candidatas = []
        for (mercado, sel), (p, c) in catalogo.items():
            if cuota_min <= c <= cuota_max:
                ev = calcular_ev(p, c)
                if ev >= MIN_EV:
                    directa_candidatas.append({
                        "mercado": mercado, "seleccion": sel, "cuota": c,
                        "prob_modelo": p, "prob_implicita": 1 / c, "ev": ev,
                        "kelly_stake_pct": _kelly_fraction(p, c),
                    })
        directa_candidatas.sort(key=lambda x: x["ev"], reverse=True)
        directa = directa_candidatas[0] if directa_candidatas else None

        dupla_candidatas = []
        seleccs = list(catalogo.keys())
        for i in range(len(seleccs)):
            for j in range(i + 1, len(seleccs)):
                (m1, s1), (m2, s2) = seleccs[i], seleccs[j]
                if m1 == m2:
                    continue
                p1, c1 = catalogo[(m1, s1)]
                p2, c2 = catalogo[(m2, s2)]
                p_conj = prob_conjunta(matriz, (m1, s1), (m2, s2))
                cuota_total = c1 * c2
                if not (cuota_min <= cuota_total <= cuota_max):
                    continue
                ev = calcular_ev(p_conj, cuota_total)
                if ev < MIN_EV:
                    continue
                dupla_candidatas.append({
                    "picks": [
                        {"mercado": m1, "seleccion": s1, "cuota": c1, "prob_marginal": p1},
                        {"mercado": m2, "seleccion": s2, "cuota": c2, "prob_marginal": p2},
                    ],
                    "cuota_total": round(cuota_total, 3),
                    "prob_conjunta": round(p_conj, 4),
                    "prob_implicita": round(1 / cuota_total, 4),
                    "ev": round(ev, 4),
                    "kelly_stake_pct": round(_kelly_fraction(p_conj, cuota_total), 4),
                    "nota": "Same-game parley: muchas casas ajustan la cuota del producto.",
                })
        dupla_candidatas.sort(key=lambda x: x["ev"], reverse=True)
        dupla = dupla_candidatas[0] if dupla_candidatas else None

        return {
            "partido": f"{home} vs {away}",
            "rango_cuota": [cuota_min, cuota_max],
            "fuente_1x2": "ensemble ML+Poisson" if ml else "solo Poisson",
            "directa": directa,
            "dupla": dupla,
        }

    def pick_multileg(self, home: str, away: str, cuotas: dict,
                      cuota_min: float = 1.70, cuota_max: float = 2.50,
                      cuota_min_tripleta: float = 2.50,
                      cuota_max_tripleta: float = 6.00,
                      min_ev: float = MIN_EV,
                      promedio_goles_liga: float = 2.7) -> dict:
        c_1x2 = cuotas["1X2"]
        sin_vig = eliminar_vig({"1": c_1x2["1"], "X": c_1x2["X"], "2": c_1x2["2"]})
        lh, la = estimar_lambdas_desde_cuotas(sin_vig["1"], sin_vig["2"], promedio_goles_liga)
        matriz = generar_matriz_poisson(lh, la, max_goles=8)
        ml = self.prob_ml(home, away, c_1x2["1"], c_1x2["X"], c_1x2["2"])

        catalogo: dict[tuple, tuple[float, float]] = {}
        for sel in ("1", "X", "2"):
            p_poisson = prob_marginal(matriz, "1X2", sel)
            p_final = 0.6 * ml[sel] + 0.4 * p_poisson if ml else p_poisson
            catalogo[("1X2", sel)] = (p_final, c_1x2[sel])
        if "OU_2.5" in cuotas:
            for sel, cuota in cuotas["OU_2.5"].items():
                catalogo[("OU_2.5", sel)] = (prob_marginal(matriz, "OU_2.5", sel), cuota)
        if "BTTS" in cuotas:
            for sel, cuota in cuotas["BTTS"].items():
                catalogo[("BTTS", sel)] = (prob_marginal(matriz, "BTTS", sel), cuota)

        keys = list(catalogo.keys())

        directa_cands = []
        for (m, s) in keys:
            p, c = catalogo[(m, s)]
            if not (cuota_min <= c <= cuota_max):
                continue
            ev = calcular_ev(p, c)
            if ev < min_ev:
                continue
            directa_cands.append({
                "tipo": "directa",
                "legs": [{"mercado": m, "seleccion": s, "cuota": round(c, 3), "prob": round(p, 4)}],
                "cuota_total": round(c, 3), "prob_conjunta": round(p, 4),
                "prob_implicita": round(1 / c, 4), "ev": round(ev, 4),
                "kelly_stake_pct": round(_kelly_fraction(p, c), 4),
            })
        directa_cands.sort(key=lambda x: x["ev"], reverse=True)

        dupla_cands = []
        combos_bloqueados_uniforme = []
        for combo in combinations(keys, 2):
            if not son_compatibles(list(combo)):
                continue
            uniforme, leg_unif = _es_combo_uniforme(list(combo))
            if uniforme:
                _log.warning("Combo descartado: 2+ legs son %s", leg_unif)
                combos_bloqueados_uniforme.append({
                    "tipo": "dupla", "motivo": "uniforme", "leg_repetido": leg_unif,
                    "legs": [{"mercado": m, "seleccion": s} for m, s in combo],
                })
                continue
            p_conj = prob_conjunta_n(matriz, list(combo))
            if p_conj <= 0:
                continue
            cuota_total = 1.0
            legs = []
            for (m, s) in combo:
                p_ind, c_ind = catalogo[(m, s)]
                cuota_total *= c_ind
                legs.append({"mercado": m, "seleccion": s,
                             "cuota": round(c_ind, 3), "prob_marginal": round(p_ind, 4)})
            if not (cuota_min <= cuota_total <= cuota_max):
                continue
            ev = calcular_ev(p_conj, cuota_total)
            if ev < min_ev:
                continue
            dupla_cands.append({
                "tipo": "dupla", "legs": legs,
                "cuota_total": round(cuota_total, 3), "prob_conjunta": round(p_conj, 4),
                "prob_implicita": round(1 / cuota_total, 4), "ev": round(ev, 4),
                "kelly_stake_pct": round(_kelly_fraction(p_conj, cuota_total), 4),
                "nota": "Same game parlay — la casa puede ajustar la cuota real.",
            })
        dupla_cands.sort(key=lambda x: x["ev"], reverse=True)

        tripleta_cands = []
        for combo in combinations(keys, 3):
            if not son_compatibles(list(combo)):
                continue
            uniforme, leg_unif = _es_combo_uniforme(list(combo))
            if uniforme:
                _log.warning("Combo descartado: 2+ legs son %s", leg_unif)
                combos_bloqueados_uniforme.append({
                    "tipo": "tripleta", "motivo": "uniforme", "leg_repetido": leg_unif,
                    "legs": [{"mercado": m, "seleccion": s} for m, s in combo],
                })
                continue
            p_conj = prob_conjunta_n(matriz, list(combo))
            if p_conj <= 0:
                continue
            cuota_total = 1.0
            legs = []
            for (m, s) in combo:
                p_ind, c_ind = catalogo[(m, s)]
                cuota_total *= c_ind
                legs.append({"mercado": m, "seleccion": s,
                             "cuota": round(c_ind, 3), "prob_marginal": round(p_ind, 4)})
            if not (cuota_min_tripleta <= cuota_total <= cuota_max_tripleta):
                continue
            ev = calcular_ev(p_conj, cuota_total)
            if ev < min_ev:
                continue
            tripleta_cands.append({
                "tipo": "tripleta", "legs": legs,
                "cuota_total": round(cuota_total, 3), "prob_conjunta": round(p_conj, 4),
                "prob_implicita": round(1 / cuota_total, 4), "ev": round(ev, 4),
                "kelly_stake_pct": round(_kelly_fraction(p_conj, cuota_total), 4),
                "nota": "Same game parlay — la casa puede ajustar la cuota real.",
            })
        tripleta_cands.sort(key=lambda x: x["ev"], reverse=True)

        prob_1x2_final = {}
        for sel in ("1", "X", "2"):
            p_p = prob_marginal(matriz, "1X2", sel)
            prob_1x2_final[sel] = round(0.6 * ml[sel] + 0.4 * p_p if ml else p_p, 4)

        return {
            "partido": f"{home} vs {away}",
            "fuente_1x2": "ensemble ML+Poisson" if ml else "solo Poisson",
            "lambdas": {"home": round(lh, 2), "away": round(la, 2)},
            "prob_1x2_final": prob_1x2_final,
            "sin_vig": {"1": round(sin_vig["1"], 4),
                        "X": round(sin_vig["X"], 4),
                        "2": round(sin_vig["2"], 4)},
            "directa": directa_cands[0] if directa_cands else None,
            "dupla": dupla_cands[0] if dupla_cands else None,
            "tripleta": tripleta_cands[0] if tripleta_cands else None,
            "directa_alt": directa_cands[1:3],
            "dupla_alt": dupla_cands[1:3],
            "tripleta_alt": tripleta_cands[1:3],
            "combos_bloqueados_uniforme": combos_bloqueados_uniforme,
        }


MERCADO_LABEL = {"1X2": "Resultado", "OU_2.5": "Goles 2.5", "BTTS": "Ambos anotan"}
SELECCION_LABEL = {
    ("1X2", "1"): "Gana local", ("1X2", "X"): "Empate", ("1X2", "2"): "Gana visitante",
    ("OU_2.5", "Over"): "Más de 2.5", ("OU_2.5", "Under"): "Menos de 2.5",
    ("BTTS", "Yes"): "Sí anotan ambos", ("BTTS", "No"): "No anotan ambos",
}


def leg_legible(leg: dict, home: str = "", away: str = "") -> str:
    m, s = leg["mercado"], leg["seleccion"]
    base = SELECCION_LABEL.get((m, s), f"{m} {s}")
    if m == "1X2":
        if s == "1" and home:
            return f"Gana {home}"
        if s == "2" and away:
            return f"Gana {away}"
    return base
