"""
Motor de análisis NBA — Moneyline, Spread, Total Points.

Diferencias con fútbol:
- Poisson (goles) → Normal (puntos continuos)
- 1X2 (Moneyline) → Más cercano al 50/50 (NBA más equilibrado)
- xG estadísticas → Pace, ORtg, DRtg (eficiencia)
"""
import logging
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.probability import norm_cdf

_log = logging.getLogger("betbrain.nba")


class NBAEngine:
    """Motor de análisis para partidos NBA."""

    def __init__(self):
        self.model = None  # XGBoost (futuro)
        # Por ahora: predicción basada en stats recientes

    # ────────────────────────────────────────────────────────────
    # Core: Predicción de puntos
    # ────────────────────────────────────────────────────────────

    def predecir_puntos(self, home_stats: dict, away_stats: dict) -> dict:
        """
        Predice puntos esperados usando distribución normal.

        Args:
            home_stats: {"ortg": 110.5, "pace": 100.2, "pts_avg": 108, ...}
            away_stats: {"ortg": 108.2, "pace": 99.8, "pts_avg": 106, ...}

        Returns:
            {
                "home_points_esp": 108.5,
                "away_points_esp": 104.2,
                "std_dev": 8.5,
                "total_esp": 212.7
            }
        """
        # Factor de pace (velocidad de juego)
        pace_home = home_stats.get("pace", 100)
        pace_away = away_stats.get("pace", 100)
        pace_promedio = (pace_home + pace_away) / 2

        # Offensive rating (puntos por 100 posesiones)
        ortg_home = home_stats.get("ortg", 108)
        ortg_away = away_stats.get("ortg", 108)

        # Predicción lineal simple: (ORTG / 100) * Pace
        home_ptos_esp = (ortg_home / 100) * pace_promedio
        away_ptos_esp = (ortg_away / 100) * pace_promedio

        # Desviación estándar (empiricamente ~8.5 en NBA)
        std_dev = 8.5

        return {
            "home_points_esp": round(home_ptos_esp, 1),
            "away_points_esp": round(away_ptos_esp, 1),
            "std_dev": std_dev,
            "total_esp": round(home_ptos_esp + away_ptos_esp, 1),
        }

    # ────────────────────────────────────────────────────────────
    # Probabilidades — Moneyline
    # ────────────────────────────────────────────────────────────

    def prob_moneyline(self, home_points: float, away_points: float,
                       std_dev: float = 8.5) -> dict:
        """
        Calcula P(Home Win) y P(Away Win) usando distribución normal.

        Args:
            home_points: Puntos esperados local
            away_points: Puntos esperados visitante
            std_dev: Desviación estándar (defecto: 8.5)

        Returns:
            {
                "prob_home_win": 0.55,
                "prob_away_win": 0.45,
                "spread_esp": 4.3  # Diferencia esperada
            }
        """
        # Diferencia esperada
        spread_esp = home_points - away_points

        # Distribución de la diferencia: N(spread_esp, 2*std_dev^2)
        # Std combinada = sqrt(std_dev^2 + std_dev^2) = std_dev * sqrt(2)
        std_diff = std_dev * np.sqrt(2)

        # P(Home > Away) = P(diferencia > 0)
        prob_home = norm_cdf(0, loc=spread_esp, scale=std_diff)
        # Nota: norm.cdf invierte, por eso es P(X < 0)
        # Realmente queremos P(X > 0) = 1 - norm_cdf(0, loc, scale)
        prob_home = 1 - norm_cdf(0, loc=spread_esp, scale=std_diff)
        prob_away = 1 - prob_home

        return {
            "prob_home_win": round(prob_home, 4),
            "prob_away_win": round(prob_away, 4),
            "spread_esp": round(spread_esp, 1),
        }

    # ────────────────────────────────────────────────────────────
    # Evaluación de value — Moneyline
    # ────────────────────────────────────────────────────────────

    def evaluar_value_moneyline(self, prob_real: float,
                                cuota_americana: float) -> dict:
        """
        Evalúa si hay value en una cuota Moneyline.

        Args:
            prob_real: Probabilidad real (ej: 0.55)
            cuota_americana: Cuota en formato americano (ej: -150 para favorito)

        Returns:
            {
                "prob_implicita": 0.60,
                "ev": 0.05,  # +5%
                "kelly_pct": 0.08,  # 8% del bankroll
                "tiene_value": True
            }
        """
        # Convertir cuota americana a decimal
        if cuota_americana > 0:
            # Underdog: +200 → decimal = 1 + 200/100 = 3.00
            cuota_decimal = 1 + (cuota_americana / 100)
        else:
            # Favorito: -150 → decimal = 1 + 100/150 = 1.667
            cuota_decimal = 1 + (100 / abs(cuota_americana))

        # Probabilidad implícita en la cuota
        prob_implicita = 1 / cuota_decimal

        # EV: Expected Value = (prob_real * cuota) - 1
        ev = (prob_real * cuota_decimal) - 1

        # Kelly fraccionario (25%)
        kelly_full = (prob_real * cuota_decimal - 1) / (cuota_decimal - 1) if cuota_decimal > 1 else 0
        kelly_frac = kelly_full * 0.25

        return {
            "cuota_decimal": round(cuota_decimal, 2),
            "prob_implicita": round(prob_implicita, 4),
            "prob_real": round(prob_real, 4),
            "ev": round(ev, 4),
            "kelly_pct": round(max(0, kelly_frac), 4),
            "tiene_value": ev > 0.05,  # Umbral mínimo 5%
        }

    # ────────────────────────────────────────────────────────────
    # Probabilidades — Spread
    # ────────────────────────────────────────────────────────────

    def prob_spread(self, home_points: float, away_points: float,
                    spread: float, std_dev: float = 8.5) -> dict:
        """
        Calcula P(Home > Away + Spread) para líneas de spread.

        Args:
            home_points: Puntos esperados local
            away_points: Puntos esperados visitante
            spread: Línea de spread (ej: -4.5, +4.5)
            std_dev: Desviación estándar

        Returns:
            {
                "prob_home_covers": 0.48,  # Home bate el spread
                "prob_away_covers": 0.52   # Away bate el spread
            }
        """
        diff_esp = home_points - away_points
        std_diff = std_dev * np.sqrt(2)

        # P(Home - Away > Spread)
        prob_home_covers = 1 - norm_cdf(spread, loc=diff_esp, scale=std_diff)
        prob_away_covers = 1 - prob_home_covers

        return {
            "prob_home_covers": round(prob_home_covers, 4),
            "prob_away_covers": round(prob_away_covers, 4),
        }

    # ────────────────────────────────────────────────────────────
    # Probabilidades — Over/Under
    # ────────────────────────────────────────────────────────────

    def prob_ou(self, total_esp: float, total_linea: float,
                std_dev: float = 8.5) -> dict:
        """
        Calcula P(Total > Línea) y P(Total < Línea).

        Args:
            total_esp: Total esperado (ej: 212.5)
            total_linea: Línea de apuesta (ej: 210.5)
            std_dev: Desviación estándar del total

        Returns:
            {
                "prob_over": 0.55,
                "prob_under": 0.45
            }
        """
        # El total sigue N(total_esp, std_dev)
        prob_over = 1 - norm_cdf(total_linea, loc=total_esp, scale=std_dev)
        prob_under = 1 - prob_over

        return {
            "prob_over": round(prob_over, 4),
            "prob_under": round(prob_under, 4),
        }

    # ────────────────────────────────────────────────────────────
    # Pick generation
    # ────────────────────────────────────────────────────────────

    def generar_picks_nba(self, home: str, away: str,
                          prediccion: dict, cuotas: dict) -> list:
        """
        Genera picks NBA (Moneyline, Spread, Total).

        Args:
            home: Equipo local (ej: "Lakers")
            away: Equipo visitante (ej: "Celtics")
            prediccion: Output de predecir_puntos()
            cuotas: {
                "moneyline": {"home": -150, "away": +130},
                "spread": {"home": -4.5, "away": +4.5, "cuota": -110},
                "total": {"over": 210.5, "under": 210.5, "cuota_over": -110}
            }

        Returns:
            [pick1, pick2, pick3, ...]
        """
        picks = []

        home_pts = prediccion["home_points_esp"]
        away_pts = prediccion["away_points_esp"]
        total_esp = prediccion["total_esp"]
        std_dev = prediccion["std_dev"]

        # 1. MONEYLINE
        probs_ml = self.prob_moneyline(home_pts, away_pts, std_dev)

        # Home Moneyline
        if cuotas.get("moneyline", {}).get("home"):
            val_home_ml = self.evaluar_value_moneyline(
                probs_ml["prob_home_win"],
                cuotas["moneyline"]["home"]
            )
            if val_home_ml["tiene_value"] and val_home_ml["ev"] > 0.05:
                picks.append({
                    "tipo": "MONEYLINE",
                    "mercado": "Moneyline Home",
                    "pick": f"{home} Win",
                    "cuota": cuotas["moneyline"]["home"],
                    "prob": probs_ml["prob_home_win"],
                    "ev": val_home_ml["ev"],
                    "kelly_pct": val_home_ml["kelly_pct"],
                    "confianza": self._calc_confianza_nba(
                        probs_ml["prob_home_win"],
                        val_home_ml["ev"]
                    ),
                })

        # Away Moneyline
        if cuotas.get("moneyline", {}).get("away"):
            val_away_ml = self.evaluar_value_moneyline(
                probs_ml["prob_away_win"],
                cuotas["moneyline"]["away"]
            )
            if val_away_ml["tiene_value"] and val_away_ml["ev"] > 0.05:
                picks.append({
                    "tipo": "MONEYLINE",
                    "mercado": "Moneyline Away",
                    "pick": f"{away} Win",
                    "cuota": cuotas["moneyline"]["away"],
                    "prob": probs_ml["prob_away_win"],
                    "ev": val_away_ml["ev"],
                    "kelly_pct": val_away_ml["kelly_pct"],
                    "confianza": self._calc_confianza_nba(
                        probs_ml["prob_away_win"],
                        val_away_ml["ev"]
                    ),
                })

        # 2. SPREAD (si disponible)
        if "spread" in cuotas:
            spread_linea = cuotas["spread"]["home"]  # Línea para home
            probs_spread = self.prob_spread(home_pts, away_pts, spread_linea, std_dev)

            val_home_spread = self.evaluar_value_moneyline(
                probs_spread["prob_home_covers"],
                cuotas["spread"].get("cuota_home", -110)
            )

            if val_home_spread["tiene_value"] and val_home_spread["ev"] > 0.05:
                picks.append({
                    "tipo": "SPREAD",
                    "mercado": f"Spread {spread_linea}",
                    "pick": f"{home} {spread_linea}",
                    "cuota": cuotas["spread"].get("cuota_home", -110),
                    "prob": probs_spread["prob_home_covers"],
                    "ev": val_home_spread["ev"],
                    "confianza": self._calc_confianza_nba(
                        probs_spread["prob_home_covers"],
                        val_home_spread["ev"]
                    ),
                })

        # 3. OVER/UNDER (si disponible)
        if "total" in cuotas:
            total_linea = cuotas["total"]["linea"]
            probs_ou = self.prob_ou(total_esp, total_linea, std_dev)

            val_over = self.evaluar_value_moneyline(
                probs_ou["prob_over"],
                cuotas["total"].get("cuota_over", -110)
            )

            if val_over["tiene_value"] and val_over["ev"] > 0.05:
                picks.append({
                    "tipo": "TOTAL",
                    "mercado": f"O/U {total_linea}",
                    "pick": f"Over {total_linea}",
                    "cuota": cuotas["total"].get("cuota_over", -110),
                    "prob": probs_ou["prob_over"],
                    "ev": val_over["ev"],
                    "confianza": self._calc_confianza_nba(
                        probs_ou["prob_over"],
                        val_over["ev"]
                    ),
                })

        return picks

    # ────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────

    def _calc_confianza_nba(self, prob: float, ev: float) -> float:
        """
        Calcula confianza (0-1) para picks NBA.

        Fórmula:
        - Base: cuánto se desvía de 0.50 (certeza)
        - Bonus: EV alto
        """
        # Desviación de 50/50
        certeza = abs(prob - 0.5) * 2  # Escala 0-1

        # Bonus por EV
        ev_bonus = min(ev * 0.5, 0.15)  # Max +0.15 por EV alto

        confianza = min(certeza + ev_bonus, 0.99)
        return round(confianza, 3)

    def stake_recomendado_nba(self, bankroll: float, prob: float,
                              cuota: float, metodo: str = "kelly_frac") -> float:
        """
        Calcula stake recomendado para NBA (mismo Kelly que fútbol).

        Args:
            bankroll: Dinero disponible
            prob: Probabilidad real
            cuota: Cuota decimal
            metodo: "kelly_frac" (recomendado)

        Returns:
            Stake en dinero ($)
        """
        if metodo != "kelly_frac":
            return 0

        # Kelly full: (p*b - q) / b donde b = cuota - 1
        b = cuota - 1
        q = 1 - prob
        kelly_full = (prob * b - q) / b if b > 0 else 0

        # Kelly fraccionario (25%)
        kelly_frac = kelly_full * 0.25

        stake = bankroll * max(0, kelly_frac)
        return round(stake, 2)
