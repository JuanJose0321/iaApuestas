"""
Sistema de confianza y validador anti-contradicciones para picks.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MIN_EV

UMBRAL_CONFIANZA = 0.70          # con datos API reales (factor_datos = 1.0)
UMBRAL_CONFIANZA_SIN_API = 0.55  # sin forma real disponible (factor_datos = 0.6)

# Pares de selecciones semánticamente contradictorias
_CONTRADICCIONES_SET = {
    # BTTS No + Over 2.5: si nadie anota, imposible tener 3+ goles
    frozenset({("BTTS", "No"), ("OU_2.5", "Over")}),
    # BTTS Yes + Under 2.5: si ambos anotan, mínimo 2 goles (1-1), combo muy raro
    frozenset({("BTTS", "Yes"), ("OU_2.5", "Under")}),
}


def calcular_confianza(prob: float, ev: float, factor_datos: float) -> float:
    """
    Devuelve un score 0.0-1.0 que refleja la confianza en el pick.

      prob:         probabilidad real del modelo (0-1)
      ev:           expected value (0.05 = 5%)
      factor_datos: 1.0 si hay datos API reales, 0.6 si solo cuotas

    Fórmula: prob × EV_normalizado × factor_datos
      EV_normalizado escala de 0 en EV=0 a 2.0 en EV >= 2*MIN_EV
    """
    if ev <= 0:
        return 0.0
    ev_factor = min(2.0, ev / max(MIN_EV, 0.01))
    return round(min(1.0, prob * ev_factor * factor_datos), 4)


def nivel_confianza(score: float) -> str:
    """'alta', 'media' o 'baja' a partir del score de confianza."""
    if score >= 0.80:
        return "alta"
    if score >= UMBRAL_CONFIANZA:
        return "media"
    return "baja"


def verificar_contradicciones_combo(legs: list) -> list:
    """
    Recibe lista de dicts {mercado, seleccion} o tuples (mercado, seleccion).
    Devuelve lista de mensajes de contradicción detectados.
    Si la lista está vacía, el combo es coherente.
    """
    sels = []
    for leg in legs:
        if isinstance(leg, dict):
            sels.append((leg.get("mercado", ""), leg.get("seleccion", "")))
        else:
            sels.append(tuple(leg))

    mensajes = []
    n = len(sels)
    for i in range(n):
        for j in range(i + 1, n):
            par = frozenset({sels[i], sels[j]})
            if par in _CONTRADICCIONES_SET:
                m1, s1 = sels[i]
                m2, s2 = sels[j]
                mensajes.append(
                    f"{m1} {s1} + {m2} {s2}: combinación estadísticamente "
                    f"incoherente (se excluyen mutuamente en términos prácticos)"
                )
    return mensajes
