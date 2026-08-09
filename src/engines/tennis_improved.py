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

ELO_PRIOR = 1500.0   # mismo default usado en todo el resto del proyecto (Elo inicial sin historial)
SHRINK_K = 20.0       # constante de regresión a la media — ver report_tennis_audit.md sección 9/10 para el valor elegido empíricamente por backtest


def shrink_elo(elo: float, games: Optional[int],
               elo_prior: float = ELO_PRIOR, k: float = SHRINK_K) -> float:
    """
    Regresión a la media (empirical-Bayes shrinkage) para Elo calculado con
    pocos partidos. Un Elo de 1650 con 3 partidos es mucho más ruidoso
    (pudo ganarle a rivales débiles por azar) que el mismo 1650 con 200
    partidos — pero antes de este fix el motor los trataba con la misma
    confianza.

    factor = games / (games + k):
      - games=0    -> factor=0   -> devuelve elo_prior puro (1500)
      - games=k    -> factor=0.5 -> a mitad de camino entre el Elo y 1500
      - games>>k   -> factor->1  -> casi no se toca el Elo observado

    games=None desactiva el shrink (devuelve el Elo sin modificar) — así
    las llamadas existentes sin dato de "games" siguen funcionando igual
    que antes de este cambio.
    """
    if games is None:
        return elo
    games = max(0, games)
    factor = games / (games + k) if (games + k) > 0 else 0.0
    return elo_prior + (elo - elo_prior) * factor


SURFACE_MIN_GAMES = 5   # partidos mínimos en una superficie para confiar en su Elo específico


def elegir_elo_superficie(elo_overall: float, elo_superficie: Optional[float],
                           games_superficie: Optional[int],
                           min_games: int = SURFACE_MIN_GAMES) -> tuple[float, bool]:
    """
    Decide qué Elo usar para un partido: el específico de la superficie si
    hay suficiente muestra (>= min_games partidos jugados en esa
    superficie), o el overall si no.

    Returns:
        (elo_a_usar, se_uso_elo_de_superficie)
    """
    if (elo_superficie is not None and games_superficie is not None
            and games_superficie >= min_games):
        return elo_superficie, True
    return elo_overall, False


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
                      superficie: str = "hard",
                      aplicar_factor_superficie: bool = True) -> float:
        """
        P(J1 gana) basada en Elo.

        aplicar_factor_superficie=True (default): multiplica el delta por
        SURFACE_ELO_FACTOR — el ajuste genérico usado cuando elo1/elo2 son
        el Elo *overall* del jugador. Si elo1/elo2 ya son el Elo específico
        de esa superficie (ver elegir_elo_superficie), hay que pasar False
        — la superficie ya está reflejada en qué número se usó, aplicar el
        factor de nuevo sería contar el efecto dos veces.
        """
        factor = SURFACE_ELO_FACTOR.get(superficie.lower(), 1.0) if aplicar_factor_superficie else 1.0
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
                                   formato: str = "best_of_3",
                                   games1: Optional[int] = None,
                                   games2: Optional[int] = None,
                                   elo1_superficie: Optional[float] = None,
                                   elo2_superficie: Optional[float] = None,
                                   games1_superficie: Optional[int] = None,
                                   games2_superficie: Optional[int] = None,
                                   min_games_superficie: int = SURFACE_MIN_GAMES) -> Dict:
        """
        P(ganar partido) combinando Elo + Forma.

        Ensemble: 70% Elo + 30% Forma — pero SOLO cuando hay forma real
        para ambos jugadores. Si a alguno le falta, usar 100% Elo: mezclar
        con un 50% inventado sesga cada predicción hacia el empate sin
        ninguna base estadística (era el bug de P0-1).

        games1/games2 (opcional): si se pasan, el Elo de cada jugador se
        encoge hacia la media antes de usarlo (ver shrink_elo) — mitiga
        que un Elo calculado con pocos partidos se trate con la misma
        confianza que uno con historial largo. None (default) desactiva
        el shrink y preserva el comportamiento anterior a este cambio.

        elo{1,2}_superficie/games{1,2}_superficie (opcional): Elo y
        partidos jugados en la superficie específica de ESTE partido. Si
        hay suficiente muestra (ver elegir_elo_superficie/SURFACE_MIN_GAMES)
        se usa ese Elo en vez del overall, y se desactiva el multiplicador
        genérico SURFACE_ELO_FACTOR (la superficie ya está reflejada en
        qué Elo se eligió — aplicar el factor de nuevo la contaría dos
        veces). None (default) preserva el comportamiento anterior.
        """
        elo1_base, uso_surf1 = elegir_elo_superficie(elo1, elo1_superficie, games1_superficie, min_games_superficie)
        elo2_base, uso_surf2 = elegir_elo_superficie(elo2, elo2_superficie, games2_superficie, min_games_superficie)
        aplicar_factor = not (uso_surf1 or uso_surf2)

        elo1_calc = shrink_elo(elo1_base, games1)
        elo2_calc = shrink_elo(elo2_base, games2)

        # Probabilidad por Elo
        p_elo = self.prob_from_elo(elo1_calc, elo2_calc, superficie, aplicar_factor)

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
                "elo1_ajustado": round(elo1_calc, 1),
                "elo2_ajustado": round(elo2_calc, 1),
                "uso_elo_superficie": {"j1": uso_surf1, "j2": uso_surf2},
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
                cuota_max: float = 6.00,
                games1: Optional[int] = None,
                games2: Optional[int] = None,
                elo1_superficie: Optional[float] = None,
                elo2_superficie: Optional[float] = None,
                games1_superficie: Optional[int] = None,
                games2_superficie: Optional[int] = None) -> Dict:
        """
        Análisis completo con Elo + Forma.

        Returns análisis similar al motor anterior pero con mejores probabilidades.

        games1/games2 (opcional): activa la regresión a la media del Elo
        para jugadores con pocos partidos (ver shrink_elo). None = sin cambios.

        elo{1,2}_superficie/games{1,2}_superficie (opcional): activa el uso
        de Elo específico por superficie (ver elegir_elo_superficie).
        None = sin cambios.
        """
        # Obtener probabilidades (Elo + Forma)
        mw_result = self.prob_match_winner_ensemble(
            elo1, elo2, player1, player2, superficie, formato, games1, games2,
            elo1_superficie, elo2_superficie, games1_superficie, games2_superficie,
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
