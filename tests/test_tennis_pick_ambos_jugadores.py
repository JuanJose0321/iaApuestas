"""
Tests del fix: el motor de tenis solo evaluaba cuotas["match_winner"]["1"]
(jugador1) para generar el pick de "ganador del partido" — nunca chequeaba
si jugador2 tenía valor real, aunque el modelo lo tuviera como favorito.
Ver report_tennis_audit.md / commit del fix.

Los casos arman las cuotas a partir de la probabilidad REAL que calcula
el motor (no números inventados a mano) para garantizar de forma robusta
que un lado tiene valor y el otro no, sin depender de que un valor
arbitrario cruce el umbral MIN_EV por casualidad.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.tennis_improved import TennisImprovedEngine, MIN_EV


def _cuota_con_valor(prob: float, margen_extra: float = 0.20) -> float:
    """Cuota generosa: el modelo la ve con EV claramente positivo."""
    return round((1.0 / prob) * (1.0 + margen_extra), 2)


def _cuota_sin_valor(prob: float, recorte: float = 0.10) -> float:
    """Cuota ajustada: el modelo la ve con EV negativo (mala paga)."""
    return round((1.0 / prob) * (1.0 - recorte), 2)


@pytest.fixture
def engine():
    return TennisImprovedEngine()


def _picks(resultado):
    return resultado["picks_verdes"] + resultado["picks_amarillos"]


def test_pick_para_jugador1_cuando_tiene_valor_comportamiento_previo(engine):
    """Caso que ya funcionaba antes del fix — no debe romperse."""
    mw = engine.prob_match_winner_ensemble(1600.0, 1500.0, "A", "B", "hard", "best_of_3")
    p1, p2 = mw["prob_j1"], mw["prob_j2"]

    resultado = engine.analizar(
        "A", "B", 1600.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": _cuota_con_valor(p1), "2": _cuota_sin_valor(p2)}},
    )

    picks = _picks(resultado)
    assert any(p["pick"] == "A gana" for p in picks)
    assert not any(p["pick"] == "B gana" for p in picks)


def test_pick_para_jugador2_cuando_tiene_valor_caso_antes_roto(engine):
    """Caso que antes del fix NUNCA se detectaba, sin importar cuánto
    valor real tuviera jugador2 — solo se miraba cuotas['1']."""
    mw = engine.prob_match_winner_ensemble(1600.0, 1500.0, "A", "B", "hard", "best_of_3")
    p1, p2 = mw["prob_j1"], mw["prob_j2"]

    resultado = engine.analizar(
        "A", "B", 1600.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": _cuota_sin_valor(p1), "2": _cuota_con_valor(p2)}},
    )

    picks = _picks(resultado)
    assert any(p["pick"] == "B gana" for p in picks), (
        "Jugador2 tenía valor real y debería generar pick — este es el bug que se arregló"
    )
    assert not any(p["pick"] == "A gana" for p in picks)


def test_sin_pick_cuando_ninguno_tiene_valor(engine):
    mw = engine.prob_match_winner_ensemble(1600.0, 1500.0, "A", "B", "hard", "best_of_3")
    p1, p2 = mw["prob_j1"], mw["prob_j2"]

    resultado = engine.analizar(
        "A", "B", 1600.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": _cuota_sin_valor(p1), "2": _cuota_sin_valor(p2)}},
    )

    assert _picks(resultado) == []


def test_ambos_lados_pueden_generar_pick_si_ambos_superan_el_umbral(engine):
    """No es 'el mejor de los dos' — cada lado se evalúa independiente,
    mismo patrón que el resto de los mercados (ej. Total Games Over)."""
    mw = engine.prob_match_winner_ensemble(1600.0, 1500.0, "A", "B", "hard", "best_of_3")
    p1, p2 = mw["prob_j1"], mw["prob_j2"]

    resultado = engine.analizar(
        "A", "B", 1600.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": _cuota_con_valor(p1, 0.30), "2": _cuota_con_valor(p2, 0.30)}},
    )

    picks = _picks(resultado)
    nombres = {p["pick"] for p in picks}
    assert "A gana" in nombres
    assert "B gana" in nombres
