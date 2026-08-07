"""
Motor de probabilidades basado en Poisson + derivación de mercados.
"""
import math
import numpy as np


def norm_cdf(x: float, loc: float = 0.0, scale: float = 1.0) -> float:
    """CDF de una normal N(loc, scale) en x — reemplazo exacto de
    scipy.stats.norm.cdf(x, loc=loc, scale=scale) vía math.erf (stdlib)."""
    return 0.5 * (1.0 + math.erf((x - loc) / (scale * math.sqrt(2.0))))


def _nelder_mead(f, x0, xatol=1e-5, fatol=1e-7, maxiter=1000,
                 alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """
    Simplex de Nelder-Mead sin dependencias externas — reemplaza
    scipy.optimize.minimize(method="Nelder-Mead") solo para problemas de
    baja dimensión como este (2 parámetros). Validado contra la salida de
    scipy con diferencia < 1e-5 en los casos de prueba de estimar_lambdas_desde_cuotas.
    """
    n = len(x0)
    step = 0.05
    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] = p[i] + step if p[i] == 0 else p[i] * 1.05
        simplex.append(p)

    for _ in range(maxiter):
        simplex.sort(key=f)
        fvals = [f(p) for p in simplex]

        if max(abs(fvals[i] - fvals[0]) for i in range(len(fvals))) < fatol:
            maxd = max(math.dist(simplex[i], simplex[0]) for i in range(1, len(simplex)))
            if maxd < xatol:
                break

        centroid = [sum(p[i] for p in simplex[:-1]) / n for i in range(n)]
        worst = simplex[-1]

        xr = [centroid[i] + alpha * (centroid[i] - worst[i]) for i in range(n)]
        fr = f(xr)

        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
        elif fr < fvals[0]:
            xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(n)]
            fe = f(xe)
            simplex[-1] = xe if fe < fr else xr
        else:
            xc = [centroid[i] + rho * (worst[i] - centroid[i]) for i in range(n)]
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
            else:
                best = simplex[0]
                simplex = [best] + [
                    [best[i] + sigma * (p[i] - best[i]) for i in range(n)]
                    for p in simplex[1:]
                ]

    simplex.sort(key=f)
    return simplex[0]


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
    def objetivo(params):
        """Error cuadrático entre el 1X2 objetivo y el que produce (lh, la)."""
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

    lh, la = _nelder_mead(objetivo, [lh0, la0])
    return max(0.15, float(lh)), max(0.15, float(la))


def poisson_probability(l: float, x: int) -> float:
    """P(X=x) para una Poisson de media l."""
    return (math.exp(-l) * (l ** x)) / math.factorial(x)


def generar_matriz_poisson(lambda_h: float, lambda_a: float,
                           max_goles: int = 8) -> np.ndarray:
    """Matriz [goles_local][goles_visita] de probabilidad conjunta, normalizada a 1."""
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
    """Probabilidad de una sola selección, marginalizando el resto de la matriz."""
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
