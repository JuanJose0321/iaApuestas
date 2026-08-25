"""
Tests del fix: el motor de tenis solo evaluaba cuotas["match_winner"]["1"]
(jugador1) para generar el pick de "ganador del partido" — nunca chequeaba
si jugador2 tenía valor real, aunque el modelo lo tuviera como favorito.
Ver report_tennis_audit.md / commit 9a46433.

Desde el 2026-08-25 el criterio que decide si un lado genera pick es
MIN_PROB_PICK (probabilidad), no EV — ver tennis_validacion_filtro_ev.md:
con cuotas reales, el filtro de EV rindió peor que cara o cruz y empeoraba
cuanto más exigente se lo ponía, mientras que la probabilidad sola mejoró
de forma monótona en todo el rango probado. El patrón de evaluar los dos
lados por separado (no "el mejor de los dos") se mantiene igual — lo que
cambia es que, como prob_j1 + prob_j2 = 1, como mucho UNO de los dos lados
puede cruzar MIN_PROB_PICK a la vez (a diferencia del EV, donde con
cuotas generosas en los dos lados sí era posible que ambos tuvieran
valor).

Los casos arman las cuotas a partir de la probabilidad REAL que calcula
el motor (no números inventados a mano), y eligen elos que dejan esa
probabilidad claramente arriba o abajo de MIN_PROB_PICK, para no depender
de un valor arbitrario que cruce el umbral por casualidad.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.tennis_improved import TennisImprovedEngine, MIN_PROB_PICK


@pytest.fixture
def engine():
    return TennisImprovedEngine()


def _picks(resultado):
    return resultado["picks_verdes"] + resultado["picks_amarillos"]


def test_pick_para_jugador1_cuando_supera_el_umbral_comportamiento_previo(engine):
    """Caso que ya funcionaba antes del fix de ambos-lados — no debe romperse."""
    mw = engine.prob_match_winner_ensemble(1650.0, 1500.0, "A", "B", "hard", "best_of_3")
    assert mw["prob_j1"] >= MIN_PROB_PICK  # confirma el setup del caso

    resultado = engine.analizar(
        "A", "B", 1650.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": 1.50, "2": 3.00}},
    )

    picks = _picks(resultado)
    assert any(p["pick"] == "A gana" for p in picks)
    assert not any(p["pick"] == "B gana" for p in picks)


def test_pick_para_jugador2_cuando_supera_el_umbral_caso_antes_roto(engine):
    """Caso que antes del fix de ambos-lados NUNCA se detectaba, sin
    importar cuánto valor real tuviera jugador2 — solo se miraba
    cuotas['1']. Acá "valor" pasó a ser probabilidad, no EV."""
    mw = engine.prob_match_winner_ensemble(1500.0, 1650.0, "A", "B", "hard", "best_of_3")
    assert mw["prob_j2"] >= MIN_PROB_PICK  # confirma el setup del caso

    resultado = engine.analizar(
        "A", "B", 1500.0, 1650.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": 3.00, "2": 1.50}},
    )

    picks = _picks(resultado)
    assert any(p["pick"] == "B gana" for p in picks), (
        "Jugador2 tenía probabilidad real por encima del umbral y debería "
        "generar pick — este es el bug que se arregló en 9a46433"
    )
    assert not any(p["pick"] == "A gana" for p in picks)


def test_sin_pick_cuando_ningun_lado_supera_el_umbral(engine):
    """Partido parejo (los dos lados por debajo de MIN_PROB_PICK) -> sin
    pick, sin importar qué tan generosa sea la cuota — el EV ya no decide
    si hay pick (ver tennis_validacion_filtro_ev.md)."""
    mw = engine.prob_match_winner_ensemble(1500.0, 1500.0, "A", "B", "hard", "best_of_3")
    assert mw["prob_j1"] < MIN_PROB_PICK and mw["prob_j2"] < MIN_PROB_PICK

    resultado = engine.analizar(
        "A", "B", 1500.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": 5.00, "2": 5.00}},  # EV muy positivo para los dos
    )

    assert _picks(resultado) == []


def test_como_mucho_un_lado_genera_pick_de_match_winner(engine):
    """A diferencia del criterio de EV (donde con cuotas generosas en los
    dos lados podían generar pick simultáneamente — ver el test que este
    reemplaza en versiones anteriores de este archivo), con el criterio de
    probabilidad esto es estructuralmente imposible: prob_j1 + prob_j2 = 1,
    así que como mucho uno de los dos puede cruzar MIN_PROB_PICK. Cada lado
    se sigue evaluando de forma independiente (mismo patrón que el resto
    de los mercados, ej. Total Games Over/Under) — es la aritmética de
    probabilidades complementarias la que limita el resultado a un pick
    por partido, no que se compare "el mejor de los dos"."""
    mw = engine.prob_match_winner_ensemble(1650.0, 1500.0, "A", "B", "hard", "best_of_3")
    assert mw["prob_j1"] >= MIN_PROB_PICK
    assert mw["prob_j2"] < MIN_PROB_PICK

    resultado = engine.analizar(
        "A", "B", 1650.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": 5.00, "2": 5.00}},  # EV altísimo para los dos, a propósito
    )

    picks = _picks(resultado)
    nombres = {p["pick"] for p in picks}
    assert nombres == {"A gana"}
