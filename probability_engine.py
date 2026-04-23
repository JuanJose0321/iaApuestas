"""
Motor de probabilidades basado en Poisson + derivación de mercados.
"""
import math
import numpy as np


def eliminar_vig(cuotas: dict) -> dict:
    """Quita el margen de la casa para obtener probs 'reales' implícitas."""
    implied = {k: 1 / v for k, v in cuotas.items()}
    margin = sum(implied.values())
    return {k: v / margin for k, v in implied.items()}


def _poisson_1x2_desde_lambdas(lh: float, la: float) -> tuple:
    """Calcula p1, pX, p2 de la matriz de Poisson para los lambdas dados."""
    p1 = px = p2 = 0.0
    for h in range(9):
        for a in range(9):
            p = (math.exp(-lh) * lh**h / math.factorial(h)) * \
                (math.exp(-la) * la**a / math.factorial(a))
            if h > a:    p1 += p
            elif h == a: px += p
            else:        p2 += p
    return p1, px, p2


def estimar_lambdas_desde_cuotas(prob_1: float, prob_2: float,
                                 promedio_goles_liga: float = 2.6):
    """
    Encuentra (λ_local, λ_visitante) resolviendo numéricamente para que
    la matriz de Poisson reproduzca las probs 1X2 objetivo con mínimo error.
    Reemplaza la antigua aproximación lineal.
    """
    from scipy.optimize import minimize

    def objetivo(params):
        lh, la = params
        if lh < 0.1 or la < 0.1:
            return 1e9
        p1, _, p2 = _poisson_1x2_desde_lambdas(lh, la)
        return (p1 - prob_1) ** 2 + (p2 - prob_2) ** 2

    # Punto de inicio: antigua fórmula lineal como estimación inicial
    supremacia = prob_1 - prob_2
    media = promedio_goles_liga / 2
    lh0 = max(0.3, media + supremacia * media)
    la0 = max(0.3, media - supremacia * media)

    result = minimize(objetivo, x0=[lh0, la0], method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 1000})
    lh, la = result.x
    return max(0.15, float(lh)), max(0.15, float(la))


def poisson_probability(l: float, x: int) -> float:
    return (math.exp(-l) * (l ** x)) / math.factorial(x)


def generar_matriz_poisson(lambda_h: float, lambda_a: float,
                           max_goles: int = 8) -> np.ndarray:
    m = np.zeros((max_goles + 1, max_goles + 1))
    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            m[i][j] = poisson_probability(lambda_h, i) * poisson_probability(lambda_a, j)
    return m / m.sum()


def derivar_mercados(matriz: np.ndarray) -> dict:
    """Deriva mercados 1X2, OU2.5, BTTS y handicaps asiáticos de la matriz."""
    n = matriz.shape[0]
    m = {
        "1X2": {"1": 0.0, "X": 0.0, "2": 0.0},
        "OU_2.5": {"Over": 0.0, "Under": 0.0},
        "BTTS": {"Yes": 0.0, "No": 0.0},
        "AH_-1.5_local": {"Home": 0.0, "Away": 0.0},
        "AH_-2.0_local": {"Home": 0.0, "Away": 0.0, "Push": 0.0},
    }
    for h in range(n):
        for a in range(n):
            p = matriz[h][a]
            if h > a:    m["1X2"]["1"] += p
            elif h == a: m["1X2"]["X"] += p
            else:        m["1X2"]["2"] += p

            if h + a > 2.5: m["OU_2.5"]["Over"] += p
            else:           m["OU_2.5"]["Under"] += p

            if h > 0 and a > 0: m["BTTS"]["Yes"] += p
            else:               m["BTTS"]["No"] += p

            if h - a > 1.5: m["AH_-1.5_local"]["Home"] += p
            else:           m["AH_-1.5_local"]["Away"] += p

            diff = h - a
            if diff > 2:   m["AH_-2.0_local"]["Home"] += p
            elif diff == 2: m["AH_-2.0_local"]["Push"] += p
            else:          m["AH_-2.0_local"]["Away"] += p
    return m


def calcular_ev(prob_real: float, cuota_mercado: float) -> float:
    """EV por unidad apostada. Positivo = value bet."""
    return (prob_real * cuota_mercado) - 1


# ----------------------------------------------------------------
# Probabilidad conjunta de 2 selecciones del MISMO partido
# (a partir de la matriz Poisson, capturando correlación real)
# ----------------------------------------------------------------

def _predicado(seleccion: tuple):
    """Devuelve función (h, a) -> bool que decide si la celda cumple la selección."""
    table = {
        ("1X2", "1"):     lambda h, a: h > a,
        ("1X2", "X"):     lambda h, a: h == a,
        ("1X2", "2"):     lambda h, a: h < a,
        ("OU_2.5", "Over"):  lambda h, a: (h + a) > 2,
        ("OU_2.5", "Under"): lambda h, a: (h + a) <= 2,
        ("BTTS", "Yes"):  lambda h, a: h > 0 and a > 0,
        ("BTTS", "No"):   lambda h, a: h == 0 or a == 0,
    }
    return table[seleccion]


def prob_marginal(matriz, mercado: str, seleccion: str) -> float:
    pred = _predicado((mercado, seleccion))
    n = matriz.shape[0]
    return float(sum(matriz[h][a] for h in range(n) for a in range(n) if pred(h, a)))


def prob_conjunta(matriz, sel1: tuple, sel2: tuple) -> float:
    """
    Probabilidad de que se cumplan AMBAS selecciones a la vez.
    sel1, sel2 = (mercado, seleccion), p.ej. ('1X2','1') y ('OU_2.5','Over').
    Captura la correlación real entre mercados (Poisson conjunto).
    """
    p1 = _predicado(sel1)
    p2 = _predicado(sel2)
    n = matriz.shape[0]
    return float(sum(
        matriz[h][a]
        for h in range(n) for a in range(n)
        if p1(h, a) and p2(h, a)
    ))


def prob_conjunta_n(matriz, selecciones: list) -> float:
    """
    Probabilidad conjunta de N selecciones al mismo tiempo (same game parlay).
    selecciones = [('1X2','1'), ('OU_2.5','Over'), ('BTTS','Yes'), ...]
    Se suman las celdas de la matriz donde TODAS las selecciones se cumplen.
    Esto captura la correlación real (por ejemplo: BTTS sí + Over 2.5 están
    correlacionados positivamente — la matriz lo refleja sola).
    """
    if not selecciones:
        return 1.0
    preds = [_predicado(s) for s in selecciones]
    n = matriz.shape[0]
    total = 0.0
    for h in range(n):
        for a in range(n):
            if all(p(h, a) for p in preds):
                total += matriz[h][a]
    return float(total)


def son_compatibles(selecciones: list) -> bool:
    """
    Verifica que no haya dos selecciones del mismo mercado (serían mutuamente
    excluyentes y la conjunta sería 0).
    """
    mercados = [m for (m, _) in selecciones]
    return len(mercados) == len(set(mercados))
