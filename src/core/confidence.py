"""
Sistema de confianza y validador anti-contradicciones para picks.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import MIN_EV

UMBRAL_VERDE    = 0.75
UMBRAL_AMARILLO = 0.65
UMBRAL_ROJO     = 0.50

# Alias legacy (usados en imports de app.py)
UMBRAL_CONFIANZA = UMBRAL_VERDE
UMBRAL_CONFIANZA_SIN_API = 0.60

_CONTRADICCIONES_SET = {
    frozenset({("BTTS", "No"), ("OU_2.5", "Over")}),
    frozenset({("BTTS", "Yes"), ("OU_2.5", "Under")}),
}


def calcular_confianza(prob: float, ev: float, factor_datos: float) -> float:
    """
    Score 0.0-1.0 de confianza en el pick.
      prob:         probabilidad real del modelo (0-1)
      ev:           expected value (0.05 = 5%)
      factor_datos: 1.0 si hay datos API reales, 0.6 si solo cuotas
    """
    if ev <= 0:
        return 0.0
    ev_factor = min(2.0, ev / max(MIN_EV, 0.01))
    return round(min(1.0, prob * ev_factor * factor_datos), 4)


def nivel_confianza(score: float) -> str:
    """'verde', 'amarillo', 'rojo' o 'muy_baja' a partir del score."""
    if score >= UMBRAL_VERDE:
        return "verde"
    if score >= UMBRAL_AMARILLO:
        return "amarillo"
    if score >= UMBRAL_ROJO:
        return "rojo"
    return "muy_baja"


def verificar_contradicciones_combo(legs: list) -> list:
    """
    Recibe lista de dicts {mercado, seleccion} o tuples (mercado, seleccion).
    Devuelve lista de mensajes de contradicción detectados.
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
