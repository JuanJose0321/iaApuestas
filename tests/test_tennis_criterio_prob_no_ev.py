"""
Tests del cambio de criterio de entrada de picks (2026-08-25): antes un
partido generaba pick si EV >= MIN_EV; ahora genera pick si prob >=
MIN_PROB_PICK, sin importar el EV. Ver tennis_validacion_filtro_ev.md
para la validación completa contra cuotas reales (12,934 partidos
históricos + 117 picks reales de producción) que motivó el cambio: el
filtro de EV rendía peor que cara o cruz y empeoraba cuanto más exigente
se lo ponía, mientras que la probabilidad sola mejoraba de forma monótona.

Estos dos tests cubren exactamente los casos que antes se comportaban al
revés de como se comportan ahora.
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


def test_probabilidad_alta_con_ev_negativo_ahora_genera_pick(engine):
    """Antes del 2026-08-25 esto NUNCA generaba pick (EV negativo no pasa
    el filtro de MIN_EV) — ahora sí, porque la probabilidad sola decide."""
    mw = engine.prob_match_winner_ensemble(1650.0, 1500.0, "A", "B", "hard", "best_of_3")
    assert mw["prob_j1"] >= MIN_PROB_PICK  # confirma el setup del caso

    cuota = 1.20  # cuota corta a propósito: EV negativo para esta probabilidad
    ev_esperado = mw["prob_j1"] * cuota - 1.0
    assert ev_esperado < 0  # confirma el setup del caso

    resultado = engine.analizar(
        "A", "B", 1650.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": cuota, "2": 50.0}},
    )

    picks = _picks(resultado)
    assert any(p["pick"] == "A gana" for p in picks), (
        "Probabilidad por encima de MIN_PROB_PICK debería generar pick "
        "aunque el EV sea negativo"
    )
    pick = next(p for p in picks if p["pick"] == "A gana")
    assert pick["ev"] < 0
    assert resultado["picks_manual"] == []


def test_ev_alto_con_probabilidad_baja_ya_no_genera_pick(engine):
    """Antes del 2026-08-25 esto SIEMPRE generaba pick (EV muy alto por
    cuota larga, aunque la probabilidad rondara 50/50) — ahora cae a
    picks_manual porque no llega a MIN_PROB_PICK."""
    mw = engine.prob_match_winner_ensemble(1510.0, 1500.0, "A", "B", "hard", "best_of_3")
    assert mw["prob_j1"] < MIN_PROB_PICK  # confirma el setup del caso

    cuota = 3.5  # cuota larga a propósito: EV alto pese a la probabilidad baja
    ev_esperado = mw["prob_j1"] * cuota - 1.0
    assert ev_esperado > 0.10  # EV bien positivo — confirma el setup del caso

    resultado = engine.analizar(
        "A", "B", 1510.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": cuota, "2": 50.0}},
    )

    assert _picks(resultado) == [], (
        "EV alto por cuota larga ya no debería alcanzar para generar pick "
        "si la probabilidad no llega a MIN_PROB_PICK"
    )
    assert resultado["picks_manual"], "debe seguir ofreciendo la tarjeta manual"
    manual = resultado["picks_manual"][0]
    assert manual["ev"] > 0.10
    assert manual["prob"] < MIN_PROB_PICK
