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

# MIN_EV ya NO decide si se muestra un pick (ver MIN_PROB_PICK más abajo) —
# se sigue usando dentro de evaluar_value() para calcular tiene_value/ev,
# que se sigue guardando y trackeando (ev_predicho, predicciones_tenis),
# solo dejó de ser lo que filtra qué partidos se muestran como pick.
MIN_EV = 0.05

# Umbral de PROBABILIDAD que decide si un partido genera un pick real
# (verde/amarillo) — reemplaza al filtro de EV que usaba el motor hasta el
# 2026-08-25. Validado contra cuotas reales de Tennis-Data.co.uk (12,934
# partidos 2024-2026, ambos lados de cada partido) y contra los 117 picks
# reales de producción: filtrar por EV≥MIN_EV daba ~40% de accuracy (peor
# que cara o cruz, y empeoraba cuanto más exigente se ponía el EV: 10%→
# 37.7%, 20%→31.8%, 30%→29.5%), mientras que filtrar solo por probabilidad
# mejoró de forma monótona y limpia en todo el rango probado (prob≥60%→
# 64.1%, prob≥65%→66.3%, prob≥70%→69.1%, prob≥75%→71.3%). Ver
# tennis_validacion_filtro_ev.md para el detalle completo de la validación.
MIN_PROB_PICK = 0.60

# UMBRAL_VERDE/UMBRAL_AMARILLO se comparan contra `confianza`, que desde
# el 2026-08-25 es directamente certeza = abs(prob - 0.5) * 2.0, sin bonus
# de EV (ver _calc_confianza y tennis_validacion_filtro_ev.md — "Variante
# A" del backtest de fórmulas de confianza, la que rindió igual o mejor
# que las que combinaban EV, a volumen comparable). Con la fórmula vieja
# estos umbrales estaban en escala de probabilidad (0.65/0.50); ahora
# están en escala de certeza, así que se recalibraron para seguir
# representando las mismas zonas de accuracy validadas:
#   UMBRAL_AMARILLO = certeza en MIN_PROB_PICK (abs(0.60-0.5)*2 = 0.20) —
#     así ningún pick que cruce el umbral de entrada queda sin clasificar.
#   UMBRAL_VERDE = certeza en prob=0.75 (abs(0.75-0.5)*2 = 0.50) — la zona
#     "conservadora" de mayor accuracy validada (71.3% con cuota real).
UMBRAL_VERDE = 0.50
UMBRAL_AMARILLO = 0.20

# Señales informativas para picks manuales (prob < MIN_PROB_PICK en los dos
# lados, no llegaron al piso de pick real) — el EV ya no decide si hay
# pick, pero se sigue mostrando como contexto de si la cuota compensaba o
# no una probabilidad que de por sí no alcanzó el umbral. El "favorito" de
# un pick manual siempre tiene prob en [0.50, MIN_PROB_PICK) — una banda
# angosta — por eso estos dos umbrales viven adentro de esa banda (a
# diferencia de antes, donde UMBRAL_PROB_SOLIDO calzaba con el viejo
# UMBRAL_VERDE=0.65, un valor que ya no es alcanzable para un pick manual
# bajo el nuevo criterio de entrada).
UMBRAL_PROB_SOLIDO = 0.58      # "casi" llega al umbral de pick (0.60)
UMBRAL_PROB_RAZONABLE = 0.55   # por encima de 50/50 con margen real

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


def _cargar_total_esp_coefs() -> Dict[str, Optional[Dict[str, float]]]:
    """
    Carga los coeficientes (a, b) de total_esp = a + b*p*q calibrados por
    regresión contra partidos reales (ver calibrate_tennis_std_dev.py —
    reemplaza la fórmula heurística sets_esp*games_por_set, que
    sobreestimaba el total real en +2.8/+3.1 games, ver
    tennis_backtest_results.md). None por formato si el archivo no existe,
    es viejo (sin estos campos) o falla la lectura — ese formato cae a la
    fórmula heurística original, nunca crashea el motor por esto.
    """
    try:
        with open(_STD_DEV_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        coefs = {}
        for formato, clave in (("best_of_3", "TOTAL_ESP_BO3"), ("best_of_5", "TOTAL_ESP_BO5")):
            entry = data.get(clave)
            coefs[formato] = {"a": float(entry["a"]), "b": float(entry["b"])} if entry else None
        _log.info(f"Coeficientes de total_esp calibrado cargados: {coefs}")
        return coefs
    except Exception as e:
        _log.warning(f"No se pudieron cargar coeficientes de total_esp calibrado, "
                      f"usando fórmula heurística original: {e}")
        return {"best_of_3": None, "best_of_5": None}


TOTAL_ESP_COEFS = _cargar_total_esp_coefs()

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


DECAY_POR_MES = 0.0   # 0.0 = desactivado por default hasta validar por backtest (ver tennis_backtest_results.md)


def aplicar_decay_inactividad(elo: float, meses_inactivo: Optional[float],
                               elo_medio: float = ELO_PRIOR,
                               decay_por_mes: float = DECAY_POR_MES) -> float:
    """
    Acerca el Elo hacia la media general cuanto más tiempo lleva el
    jugador sin competir — sin esto, un jugador retirado o con una
    lesión larga queda con su Elo de pico "congelado" indefinidamente.

    factor = min(1, meses_inactivo * decay_por_mes):
      - meses_inactivo=0 o decay_por_mes=0 -> factor=0 -> Elo sin tocar
      - factor=1 (tope) -> devuelve elo_medio puro (1500)

    meses_inactivo=None (jugador sin partido previo conocido) desactiva
    el decay — no hay fecha de referencia desde la cual contar.
    """
    if meses_inactivo is None or decay_por_mes <= 0:
        return elo
    factor = min(1.0, max(0.0, meses_inactivo) * decay_por_mes)
    return elo + (elo_medio - elo) * factor


FORMA_MAX_MESES_INACTIVO = 3.0   # más de esto sin jugar, los "últimos 10 partidos" ya no son forma reciente


def forma_vigente(meses_inactivo: Optional[float],
                   max_meses: float = FORMA_MAX_MESES_INACTIVO) -> bool:
    """
    False si el jugador lleva más de max_meses sin competir — su forma
    calculada (últimos 10 partidos) quedó vieja y no debe tratarse como
    señal de forma actual (mismo problema que motivó el decay de Elo:
    un jugador retirado o lesionado mucho tiempo, ej. Federer, seguía
    mostrando la racha de su última temporada activa como si fuera
    vigente hoy).

    meses_inactivo=None (sin fecha de referencia, ej. Elo explícito en
    el request) no descarta la forma — no hay con qué evaluarlo.
    """
    if meses_inactivo is None:
        return True
    return meses_inactivo <= max_meses


H2H_MIN_PARTIDOS = 3   # enfrentamientos previos mínimos para confiar en el H2H — con menos, es ruido
H2H_WEIGHT = 0.0       # 0.0 = desactivado por default hasta validar por backtest (ver tennis_backtest_results.md)


def prob_from_h2h(h2h_ganados_j1: int, h2h_total: int) -> float:
    """
    P(J1 gana) basada en el historial de enfrentamientos directos.

    Mismo suavizado que prob_from_form (evita que un H2H perfecto tipo
    3-0 produzca una probabilidad extrema de 1.0): rango [0.3, 0.7].
    """
    pct = h2h_ganados_j1 / h2h_total
    return 0.3 + 0.4 * pct


def h2h_vigente(h2h_total: Optional[int], min_partidos: int = H2H_MIN_PARTIDOS) -> bool:
    """True si hay suficiente historial de enfrentamientos directos para usarlo."""
    return h2h_total is not None and h2h_total >= min_partidos


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
                                   min_games_superficie: int = SURFACE_MIN_GAMES,
                                   meses_inactivo1: Optional[float] = None,
                                   meses_inactivo2: Optional[float] = None,
                                   decay_por_mes: float = DECAY_POR_MES,
                                   max_meses_forma: float = FORMA_MAX_MESES_INACTIVO,
                                   h2h_ganados_j1: Optional[int] = None,
                                   h2h_total: Optional[int] = None,
                                   h2h_weight: float = H2H_WEIGHT,
                                   h2h_min_partidos: int = H2H_MIN_PARTIDOS) -> Dict:
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

        meses_inactivo{1,2}/decay_por_mes (opcional): acerca el Elo hacia
        la media cuanto más tiempo lleva el jugador sin competir (ver
        aplicar_decay_inactividad). decay_por_mes=0 (default del módulo)
        desactiva el decay y preserva el comportamiento anterior. Estos
        mismos meses_inactivo{1,2} también descartan la "forma" (últimos
        10 partidos) si superan max_meses_forma — un jugador inactivo
        hace mucho no tiene "forma reciente", tiene una racha vieja (ver
        forma_vigente). Esto aplica siempre que se pase meses_inactivo,
        independiente de decay_por_mes.

        h2h_ganados_j1/h2h_total/h2h_weight (opcional): mezcla la
        probabilidad del historial de enfrentamientos directos (ver
        prob_from_h2h) SOBRE el ensemble de Elo+Forma ya calculado, con
        peso h2h_weight — no reemplaza Elo/Forma, se aplica encima. Solo
        si hay al menos h2h_min_partidos enfrentamientos previos (ver
        h2h_vigente); si no, se ignora y el resultado es idéntico a no
        pasar estos parámetros. h2h_weight=0 (default) desactiva el H2H.
        """
        elo1_base, uso_surf1 = elegir_elo_superficie(elo1, elo1_superficie, games1_superficie, min_games_superficie)
        elo2_base, uso_surf2 = elegir_elo_superficie(elo2, elo2_superficie, games2_superficie, min_games_superficie)
        aplicar_factor = not (uso_surf1 or uso_surf2)

        elo1_base = aplicar_decay_inactividad(elo1_base, meses_inactivo1, decay_por_mes=decay_por_mes)
        elo2_base = aplicar_decay_inactividad(elo2_base, meses_inactivo2, decay_por_mes=decay_por_mes)

        elo1_calc = shrink_elo(elo1_base, games1)
        elo2_calc = shrink_elo(elo2_base, games2)

        # Probabilidad por Elo
        p_elo = self.prob_from_elo(elo1_calc, elo2_calc, superficie, aplicar_factor)

        # Probabilidad por Forma (solo si hay datos reales de ambos Y siguen vigentes)
        form1 = self.get_form(player1) if forma_vigente(meses_inactivo1, max_meses_forma) else None
        form2 = self.get_form(player2) if forma_vigente(meses_inactivo2, max_meses_forma) else None
        usa_forma = form1 is not None and form2 is not None

        if usa_forma:
            p_form = self.prob_from_form(form1, form2)
            p_j1 = 0.70 * p_elo + 0.30 * p_form
        else:
            p_form = None
            p_j1 = p_elo

        # H2H: se mezcla ENCIMA del ensemble de Elo+Forma ya calculado,
        # no lo reemplaza — solo si hay suficiente historial directo.
        usa_h2h = h2h_weight > 0 and h2h_vigente(h2h_total, h2h_min_partidos)
        if usa_h2h:
            p_h2h = prob_from_h2h(h2h_ganados_j1, h2h_total)
            p_j1 = (1.0 - h2h_weight) * p_j1 + h2h_weight * p_h2h
        else:
            p_h2h = None

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
                "p_h2h": round(p_h2h, 4) if p_h2h is not None else None,
                "usa_h2h": usa_h2h,
            }
        }

    def prob_total_games(self, p_base: float,
                        formato: str = "best_of_3") -> Dict:
        """
        Distribución de total de games (Normal).

        p_base: probabilidad base (puede ser Elo o Ensemble)

        total_esp usa la regresión calibrada contra partidos reales
        (total_esp = a + b*p*q, ver calibrate_tennis_std_dev.py /
        TOTAL_ESP_COEFS) cuando está disponible para ese formato — la
        fórmula heurística vieja (sets_esp*games_por_set) sobreestimaba
        el total real en +2.8/+3.1 games (ver tennis_backtest_results.md),
        generando EV inflado en el mercado de Total Games. games_por_set/
        sets_esperados siguen siendo la estimación heurística original,
        son solo informativos — no se usan en ningún cálculo de EV.
        """
        p, q = p_base, 1.0 - p_base
        pq = p * q

        # Calcular sets esperados (heurístico, solo para los campos de debug)
        if formato == "best_of_5":
            sets_esp = 3.0 + 3.0*pq
            competitiveness = min(2.0*pq, 0.25)
            games_por_set = 10.5 + 2.0 * competitiveness
        else:  # best_of_3
            sets_esp = 2.0 + 2.0*pq
            competitiveness = min(2.0*pq, 0.25)
            games_por_set = 10.0 + 1.5 * competitiveness

        coef = TOTAL_ESP_COEFS.get(formato)
        if coef is not None:
            total_esp = coef["a"] + coef["b"] * pq
        else:
            total_esp = sets_esp * games_por_set  # fallback: fórmula heurística original

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
        """
        Score de confianza = certeza del modelo en el lado evaluado
        (distancia de la probabilidad respecto de 50/50).

        `ev` se sigue recibiendo (compatibilidad de firma, todos los
        call-sites ya lo calculan de todas formas) pero no influye en el
        score — ver tennis_validacion_filtro_ev.md (2026-08-25): tanto el
        EV como filtro de entrada como el EV sumado/restado a la confianza
        rindieron igual o peor que la probabilidad sola contra cuotas
        reales, en todo el rango probado.
        """
        certeza = abs(prob - 0.5) * 2.0
        return round(min(certeza, 0.99), 3)

    def _nivel_confianza(self, score: float) -> str:
        """'verde' / 'amarillo' / 'rojo' según los umbrales del módulo."""
        if score >= UMBRAL_VERDE:
            return "verde"
        if score >= UMBRAL_AMARILLO:
            return "amarillo"
        return "rojo"

    def _senal_manual(self, prob: float, ev: float) -> str:
        """
        Clasifica un pick manual (prob < MIN_PROB_PICK en los dos lados, no
        llegó al umbral de pick real) por qué tan convencido está el
        modelo, no por el tamaño del EV — una cuota larga (2.50+) infla el
        EV aunque la probabilidad real ronde el 50/50, y eso no es lo mismo
        que el modelo casi seguro de quién gana pero sin llegar al piso.

        Como un pick manual por definición nunca cruzó MIN_PROB_PICK, la
        probabilidad del favorito siempre está en la banda angosta
        [0.50, MIN_PROB_PICK) — UMBRAL_PROB_SOLIDO/RAZONABLE viven dentro
        de esa banda (ver comentario de esos umbrales al inicio del
        archivo), no en la escala de "favorito claro" que tenían antes.

        Con EV <= 0 no hay ninguna base matemática para apostar, pero eso
        puede pasar por dos motivos bien distintos: el modelo no tiene una
        opinión fuerte (prob cerca de 50/50), o el modelo SÍ se inclina
        por un lado (prob >= UMBRAL_PROB_SOLIDO, aunque sin llegar al piso
        de pick) y la cuota simplemente no compensa ni esa probabilidad
        parcial (cuota corta — el mercado ya lo tiene como más favorito
        todavía). Separar ese segundo caso evita esconder que el modelo
        sigue prefiriendo ese lado, aunque no haya valor a ese precio ni
        probabilidad suficiente para un pick real.
        """
        if ev <= 0:
            return "favorito_sin_valor" if prob >= UMBRAL_PROB_SOLIDO else "sin_base"
        if prob >= UMBRAL_PROB_SOLIDO:
            return "solido"
        if prob >= UMBRAL_PROB_RAZONABLE:
            return "razonable"
        return "cuota_larga"

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
                games2_superficie: Optional[int] = None,
                meses_inactivo1: Optional[float] = None,
                meses_inactivo2: Optional[float] = None,
                decay_por_mes: float = DECAY_POR_MES,
                max_meses_forma: float = FORMA_MAX_MESES_INACTIVO,
                h2h_ganados_j1: Optional[int] = None,
                h2h_total: Optional[int] = None,
                h2h_weight: float = H2H_WEIGHT,
                h2h_min_partidos: int = H2H_MIN_PARTIDOS) -> Dict:
        """
        Análisis completo con Elo + Forma.

        Returns análisis similar al motor anterior pero con mejores probabilidades.

        games1/games2 (opcional): activa la regresión a la media del Elo
        para jugadores con pocos partidos (ver shrink_elo). None = sin cambios.

        elo{1,2}_superficie/games{1,2}_superficie (opcional): activa el uso
        de Elo específico por superficie (ver elegir_elo_superficie).
        None = sin cambios.

        meses_inactivo{1,2}/decay_por_mes (opcional): activa el decay de
        Elo por inactividad (ver aplicar_decay_inactividad). Default
        desactivado. Los mismos meses_inactivo{1,2} también descartan la
        forma si superan max_meses_forma (ver forma_vigente).

        h2h_ganados_j1/h2h_total/h2h_weight (opcional): activa la señal de
        head-to-head (ver prob_from_h2h). Default desactivado.
        """
        # Obtener probabilidades (Elo + Forma)
        mw_result = self.prob_match_winner_ensemble(
            elo1, elo2, player1, player2, superficie, formato, games1, games2,
            elo1_superficie, elo2_superficie, games1_superficie, games2_superficie,
            min_games_superficie=SURFACE_MIN_GAMES,
            meses_inactivo1=meses_inactivo1, meses_inactivo2=meses_inactivo2,
            decay_por_mes=decay_por_mes, max_meses_forma=max_meses_forma,
            h2h_ganados_j1=h2h_ganados_j1, h2h_total=h2h_total,
            h2h_weight=h2h_weight, h2h_min_partidos=h2h_min_partidos,
        )
        p_base = mw_result["debug"]["ensemble"]  # Probabilidad ensemble

        # Modelo completo
        usa_forma = mw_result["debug"]["usa_forma"]
        modelo = {
            "p_base_j1": round(p_base, 4),
            "elo": {"j1": elo1, "j2": elo2},
            "forma": {
                # None si no hay datos reales, o si el jugador lleva más de
                # max_meses_forma sin jugar (forma vieja, no reciente).
                "j1": self.get_form(player1) if forma_vigente(meses_inactivo1, max_meses_forma) else None,
                "j2": self.get_form(player2) if forma_vigente(meses_inactivo2, max_meses_forma) else None,
            },
            "match_winner": mw_result,
            "total_games": self.prob_total_games(p_base, formato),
            "metodo": ("Ensemble Elo (70%) + Forma (30%)" if usa_forma
                       else "Elo puro (sin forma reciente disponible para uno o ambos jugadores)"),
        }

        # Generar picks
        picks_verdes = []
        picks_amarillos = []

        # Pick: Match Winner — se evalúan los dos lados por separado (antes
        # solo se chequeaba jugador1; una apuesta con valor real a favor de
        # jugador2 nunca se detectaba, aunque el modelo lo tuviera como
        # favorito — ver report_tennis_audit.md). El criterio de entrada es
        # prob >= MIN_PROB_PICK (no EV — ver tennis_validacion_filtro_ev.md,
        # 2026-08-25): se sigue calculando el EV (val["ev"]) para mostrarlo
        # y trackearlo, pero ya no decide si el partido genera un pick.
        if "match_winner" in cuotas:
            mw = cuotas["match_winner"]
            for jugador, clave_cuota, prob in (
                (player1, "1", mw_result["prob_j1"]),
                (player2, "2", mw_result["prob_j2"]),
            ):
                if clave_cuota not in mw:
                    continue
                cuota = float(mw[clave_cuota])
                if not (cuota_min <= cuota <= cuota_max):
                    continue
                if prob < MIN_PROB_PICK:
                    continue
                val = self.evaluar_value(prob, cuota)
                confianza = self._calc_confianza(prob, val["ev"])
                pick = {
                    "mercado": "Match Winner",
                    "pick": f"{jugador} gana",
                    "prob": prob,
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

                # Over y Under se evalúan por separado (mismo patrón que
                # Match Winner con jugador1/jugador2, sección 12 del audit) —
                # antes solo se chequeaba "over"; una cuota de "under" con
                # valor real, aunque el frontend la mandara, se descartaba
                # en silencio sin generar pick. Criterio de entrada: prob >=
                # MIN_PROB_PICK, no EV (ver tennis_validacion_filtro_ev.md).
                if "over" in tg:
                    p_over = float(1.0 - norm_cdf(
                        linea, loc=dist["total_esp"], scale=dist["std_dev"]
                    ))
                    cuota = float(tg["over"])
                    if cuota_min <= cuota <= cuota_max and p_over >= MIN_PROB_PICK:
                        val = self.evaluar_value(p_over, cuota)
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

                if "under" in tg:
                    p_under = float(norm_cdf(
                        linea, loc=dist["total_esp"], scale=dist["std_dev"]
                    ))
                    cuota = float(tg["under"])
                    if cuota_min <= cuota <= cuota_max and p_under >= MIN_PROB_PICK:
                        val = self.evaluar_value(p_under, cuota)
                        confianza = self._calc_confianza(p_under, val["ev"])
                        pick = {
                            "mercado": f"Total Games Under {linea}",
                            "pick": f"Under {linea} games",
                            "prob": p_under,
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

        # Picks manuales: ningún lado de ningún mercado llegó a
        # MIN_PROB_PICK (antes del 2026-08-25 era "EV insuficiente"), pero
        # la decisión de apostar queda en manos del usuario — solo se arman
        # cuando no hay ningún pick verde/amarillo, con la cuota que el
        # usuario realmente cargó y la probabilidad real del modelo (nunca
        # inventada). Ver report_tennis_audit.md / FEATURE: registrar
        # manualmente, y tennis_validacion_filtro_ev.md para el criterio
        # de entrada actual.
        picks_manual: List[Dict] = []
        if not picks_verdes and not picks_amarillos:
            if "match_winner" in cuotas:
                mw = cuotas["match_winner"]
                favorito, clave, prob = (
                    (player1, "1", mw_result["prob_j1"])
                    if mw_result["prob_j1"] >= mw_result["prob_j2"]
                    else (player2, "2", mw_result["prob_j2"])
                )
                if clave in mw:
                    cuota = float(mw[clave])
                    val = self.evaluar_value(prob, cuota)
                    picks_manual.append({
                        "mercado":         "Match Winner",
                        "pick":            f"{favorito} gana",
                        "prob":            prob,
                        "cuota":           cuota,
                        "ev":              val["ev"],
                        "confianza":       self._calc_confianza(prob, val["ev"]),
                        "confianza_nivel": "manual",
                        "senal":           self._senal_manual(prob, val["ev"]),
                        "tiene_valor":     False,
                    })

            if "total_games" in cuotas and "linea" in cuotas["total_games"]:
                tg    = cuotas["total_games"]
                linea = float(tg["linea"])
                dist  = modelo["total_games"]
                p_over = float(1.0 - norm_cdf(
                    linea, loc=dist["total_esp"], scale=dist["std_dev"]
                ))
                lado = "over" if p_over >= 0.5 else "under"
                if lado not in tg:
                    lado = "under" if lado == "over" else "over"
                if lado in tg:
                    prob  = p_over if lado == "over" else 1.0 - p_over
                    cuota = float(tg[lado])
                    val   = self.evaluar_value(prob, cuota)
                    picks_manual.append({
                        "mercado":         f"Total Games {lado.capitalize()} {linea}",
                        "pick":            f"{lado.capitalize()} {linea} games",
                        "prob":            prob,
                        "cuota":           cuota,
                        "ev":              val["ev"],
                        "confianza":       self._calc_confianza(prob, val["ev"]),
                        "confianza_nivel": "manual",
                        "senal":           self._senal_manual(prob, val["ev"]),
                        "total_esp":       dist["total_esp"],
                        "tiene_valor":     False,
                    })

        return {
            "partido": f"{player1} vs {player2}",
            "superficie": superficie,
            "formato": formato,
            "modelo": modelo,
            "picks_verdes": picks_verdes,
            "picks_amarillos": picks_amarillos,
            "picks_manual": picks_manual,
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
