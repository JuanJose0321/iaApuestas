"""
Motor mejorado de tenis con Elo calibrado + histórico.

Combine:
1. Elo dinámico (basado en resultados históricos)
2. Forma de jugador (últimos 5 partidos)
3. Distribución de probabilidades (Normal para games)
4. Ensemble simple (70% Elo + 30% Forma)
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.probability import norm_cdf

_log = logging.getLogger("betbrain.tennis_improved")

KELLY_FRACTION = 0.25
MIN_EV = 0.05

UMBRAL_VERDE = 0.65
UMBRAL_AMARILLO = 0.50

SURFACE_ELO_FACTOR = {
    "clay": 1.20,
    "hard": 1.00,
    "grass": 0.85,
    "carpet": 0.90,
}

_STD_GAMES_DEFAULT = {"best_of_3": 4.5, "best_of_5": 6.0}
_STD_DEV_PATH = Path(__file__).parent.parent / "data" / "tennis_std_dev_calibrated.json"


def _cargar_std_games() -> Dict[str, float]:
    """
    Carga std_dev de total de games calibrado contra partidos reales (ver
    calibrate_tennis_std_dev.py). Si el archivo no existe o falla la
    lectura, cae a las constantes originales — nunca crashea el motor por
    esto.
    """
    try:
        with open(_STD_DEV_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        std_games = {
            "best_of_3": float(data["STD_DEV_BO3"]),
            "best_of_5": float(data["STD_DEV_BO5"]),
        }
        _log.info(f"std_dev de total de games calibrado cargado: {std_games}")
        return std_games
    except Exception as e:
        _log.warning(f"No se pudo cargar std_dev calibrado, usando defaults "
                      f"{_STD_GAMES_DEFAULT}: {e}")
        return dict(_STD_GAMES_DEFAULT)


STD_GAMES = _cargar_std_games()


class TennisImprovedEngine:
    """Motor de tenis mejorado con datos históricos."""

    def __init__(self, elo_ratings: Optional[Dict[str, float]] = None,
                 form_stats: Optional[Dict[str, Dict]] = None):
        """
        Inicializa el motor.

        Args:
            elo_ratings: Dict con Elo actual de jugadores
            form_stats: Dict con estadísticas de forma
        """
        self.elo_ratings = elo_ratings or {}
        self.form_stats = form_stats or {}

    def get_elo(self, player_name: str) -> float:
        """Obtiene Elo del jugador (default 1500)."""
        return self.elo_ratings.get(player_name, 1500.0)

    def get_form(self, player_name: str) -> Optional[Dict]:
        """
        Obtiene estadísticas de forma (últimos N partidos).

        Returns:
            {'ganados': N, 'perdidos': M, 'porcentaje': X} si hay datos reales
            del jugador, o None si no hay forma calculada (no se debe inventar
            un 50% neutro: eso diluiría el Elo real con una constante sin
            información — ver AUDIT_REPORT / report_tennis_audit.md, hallazgo
            crítico P0-1).
        """
        return self.form_stats.get(player_name)

    def prob_from_elo(self, elo1: float, elo2: float,
                      superficie: str = "hard") -> float:
        """P(J1 gana) basada en Elo ajustado por superficie."""
        factor = SURFACE_ELO_FACTOR.get(superficie.lower(), 1.0)
        delta = (elo1 - elo2) * factor
        return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))

    def prob_from_form(self, player1_form: Dict, player2_form: Dict) -> float:
        """
        P(J1 gana) basada en forma reciente.

        Usa últimos 5 partidos.
        """
        p1_pct = player1_form.get('porcentaje', 50.0) / 100.0
        p2_pct = player2_form.get('porcentaje', 50.0) / 100.0

        # Normalizar a probabilidad (suavizar extremos)
        p1_smooth = 0.3 + 0.4 * p1_pct  # Rango [0.3, 0.7]
        p2_smooth = 0.3 + 0.4 * p2_pct

        # Probabilidad normalizada
        total = p1_smooth + p2_smooth
        return p1_smooth / total if total > 0 else 0.5

    def prob_match_winner_ensemble(self, elo1: float, elo2: float,
                                   player1: str, player2: str,
                                   superficie: str = "hard",
                                   formato: str = "best_of_3") -> Dict:
        """
        P(ganar partido) combinando Elo + Forma.

        Ensemble: 70% Elo + 30% Forma — pero SOLO cuando hay forma real
        para ambos jugadores. Si a alguno le falta, usar 100% Elo: mezclar
        con un 50% inventado sesga cada predicción hacia el empate sin
        ninguna base estadística (era el bug de P0-1).
        """
        # Probabilidad por Elo
        p_elo = self.prob_from_elo(elo1, elo2, superficie)

        # Probabilidad por Forma (solo si hay datos reales de ambos)
        form1 = self.get_form(player1)
        form2 = self.get_form(player2)
        usa_forma = form1 is not None and form2 is not None

        if usa_forma:
            p_form = self.prob_from_form(form1, form2)
            p_j1 = 0.70 * p_elo + 0.30 * p_form
        else:
            p_form = None
            p_j1 = p_elo

        # Combinatoria para sets (igual que antes)
        q = 1.0 - p_j1
        if formato == "best_of_5":
            p_win = p_j1**3 + 3*p_j1**3*q + 6*p_j1**3*q**2
        else:
            p_win = p_j1**2 + 2*p_j1**2*q

        return {
            "prob_j1": round(p_win, 4),
            "prob_j2": round(1.0 - p_win, 4),
            "debug": {
                "p_elo": round(p_elo, 4),
                "p_form": round(p_form, 4) if p_form is not None else None,
                "ensemble": round(p_j1, 4),
                "usa_forma": usa_forma,
            }
        }

    def prob_total_games(self, p_base: float,
                        formato: str = "best_of_3") -> Dict:
        """
        Distribución de total de games (Normal).

        p_base: probabilidad base (puede ser Elo o Ensemble)
        Modelo: E[games] basado en expectativa de sets + competitiveness
        """
        p, q = p_base, 1.0 - p_base

        # Calcular sets esperados
        if formato == "best_of_5":
            sets_esp = 3.0 + 3.0*p*q
            # Games por set: ~10-12 dependiendo de competitiveness
            competitiveness = min(2.0 * p*q, 0.25)  # Máximo en 0.5-0.5
            games_por_set = 10.5 + 2.0 * competitiveness
            total_esp = sets_esp * games_por_set
        else:  # best_of_3
            sets_esp = 2.0 + 2.0*p*q
            # Games por set más realista: 10-11 en sets pareados, menos en dominantes
            competitiveness = min(2.0 * p*q, 0.25)
            games_por_set = 10.0 + 1.5 * competitiveness
            # En BO3: total esperado = 19.5 a 23 games típicamente
            total_esp = sets_esp * games_por_set

        return {
            "total_esp": round(total_esp, 1),
            "std_dev": STD_GAMES.get(formato, _STD_GAMES_DEFAULT["best_of_3"]),
            "games_por_set": round(games_por_set, 1),
            "sets_esperados": round(sets_esp, 2),
            "competitiveness": round(p*q, 3),  # Para debug
        }

    def evaluar_value(self, prob: float, cuota: float) -> Dict:
        """Calcula EV y Kelly."""
        if cuota <= 1.0:
            return {
                "prob_implicita": 1.0, "ev": -1.0,
                "kelly_pct": 0.0, "tiene_value": False
            }
        prob_impl = 1.0 / cuota
        ev = prob * cuota - 1.0
        b = cuota - 1.0
        q = 1.0 - prob
        kelly_full = (prob * b - q) / b if b > 0 else 0.0
        kelly_frac = max(0.0, kelly_full * KELLY_FRACTION)

        return {
            "prob_implicita": round(prob_impl, 4),
            "ev": round(ev, 4),
            "kelly_pct": round(kelly_frac, 4),
            "tiene_value": ev >= MIN_EV,
        }

    def _calc_confianza(self, prob: float, ev: float) -> float:
        """Calcula score de confianza."""
        if ev <= 0:
            return 0.0
        certeza = abs(prob - 0.5) * 2.0
        ev_bonus = min(ev * 0.5, 0.15)
        return round(min(certeza + ev_bonus, 0.99), 3)

    def _nivel_confianza(self, score: float) -> str:
        """'verde' / 'amarillo' / 'rojo' según los umbrales del módulo."""
        if score >= UMBRAL_VERDE:
            return "verde"
        if score >= UMBRAL_AMARILLO:
            return "amarillo"
        return "rojo"

    def analizar(self, player1: str, player2: str,
                elo1: float, elo2: float,
                superficie: str, formato: str,
                cuotas: Dict,
                cuota_min: float = 1.20,
                cuota_max: float = 6.00) -> Dict:
        """
        Análisis completo con Elo + Forma.

        Returns análisis similar al motor anterior pero con mejores probabilidades.
        """
        # Obtener probabilidades (Elo + Forma)
        mw_result = self.prob_match_winner_ensemble(
            elo1, elo2, player1, player2, superficie, formato
        )
        p_base = mw_result["debug"]["ensemble"]  # Probabilidad ensemble

        # Modelo completo
        usa_forma = mw_result["debug"]["usa_forma"]
        modelo = {
            "p_base_j1": round(p_base, 4),
            "elo": {"j1": elo1, "j2": elo2},
            "forma": {
                "j1": self.get_form(player1),  # None si no hay datos reales
                "j2": self.get_form(player2),
            },
            "match_winner": mw_result,
            "total_games": self.prob_total_games(p_base, formato),
            "metodo": ("Ensemble Elo (70%) + Forma (30%)" if usa_forma
                       else "Elo puro (sin forma reciente disponible para uno o ambos jugadores)"),
        }

        # Generar picks
        picks_verdes = []
        picks_amarillos = []

        # Pick: Match Winner J1
        if "match_winner" in cuotas:
            mw = cuotas["match_winner"]
            if "1" in mw:
                cuota = float(mw["1"])
                if cuota_min <= cuota <= cuota_max:
                    val = self.evaluar_value(mw_result["prob_j1"], cuota)
                    if val["tiene_value"]:
                        confianza = self._calc_confianza(mw_result["prob_j1"], val["ev"])
                        pick = {
                            "mercado": "Match Winner",
                            "pick": f"{player1} gana",
                            "prob": mw_result["prob_j1"],
                            "cuota": cuota,
                            "ev": val["ev"],
                            "kelly_pct": val["kelly_pct"],
                            "confianza": confianza,
                            "confianza_nivel": self._nivel_confianza(confianza),
                        }
                        if confianza >= UMBRAL_VERDE:
                            picks_verdes.append(pick)
                        elif confianza >= UMBRAL_AMARILLO:
                            picks_amarillos.append(pick)

        # Pick: Total Games
        if "total_games" in cuotas:
            tg = cuotas["total_games"]
            if "linea" in tg:
                linea = float(tg["linea"])
                dist = modelo["total_games"]

                if "over" in tg:
                    p_over = float(1.0 - norm_cdf(
                        linea, loc=dist["total_esp"], scale=dist["std_dev"]
                    ))
                    cuota = float(tg["over"])
                    if cuota_min <= cuota <= cuota_max:
                        val = self.evaluar_value(p_over, cuota)
                        if val["tiene_value"]:
                            confianza = self._calc_confianza(p_over, val["ev"])
                            pick = {
                                "mercado": f"Total Games Over {linea}",
                                "pick": f"Over {linea} games",
                                "prob": p_over,
                                "cuota": cuota,
                                "ev": val["ev"],
                                "kelly_pct": val["kelly_pct"],
                                "confianza": confianza,
                                "confianza_nivel": self._nivel_confianza(confianza),
                                "total_esp": dist["total_esp"],
                            }
                            if confianza >= UMBRAL_VERDE:
                                picks_verdes.append(pick)
                            elif confianza >= UMBRAL_AMARILLO:
                                picks_amarillos.append(pick)

        return {
            "partido": f"{player1} vs {player2}",
            "superficie": superficie,
            "formato": formato,
            "modelo": modelo,
            "picks_verdes": picks_verdes,
            "picks_amarillos": picks_amarillos,
            "resumen": f"{len(picks_verdes)} verde, {len(picks_amarillos)} amarillo",
        }

    def stake_recomendado(self, bankroll: float, prob: float,
                         cuota: float) -> float:
        """Kelly fraccional (25%)."""
        b = cuota - 1.0
        if b <= 0:
            return 0.0
        q = 1.0 - prob
        kelly_full = (prob * b - q) / b
        return round(bankroll * max(0.0, kelly_full * KELLY_FRACTION), 2)
