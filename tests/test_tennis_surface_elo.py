"""
Tests del Elo por superficie — ver report_tennis_audit.md / tennis_backtest_results.md.
Antes: SURFACE_ELO_FACTOR aplicaba un multiplicador genérico igual para
cualquier jugador. Estos tests cubren elegir_elo_superficie() (con
fallback al overall cuando falta muestra) y su integración retrocompatible
en el ensemble.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.tennis_improved import (
    TennisImprovedEngine, elegir_elo_superficie, SURFACE_MIN_GAMES,
)


def test_elegir_elo_superficie_usa_overall_sin_datos():
    elo, uso = elegir_elo_superficie(1600.0, None, None)
    assert elo == 1600.0 and uso is False


def test_elegir_elo_superficie_usa_overall_con_pocos_partidos():
    elo, uso = elegir_elo_superficie(1600.0, 1750.0, SURFACE_MIN_GAMES - 1)
    assert elo == 1600.0 and uso is False


def test_elegir_elo_superficie_usa_superficie_con_suficientes_partidos():
    elo, uso = elegir_elo_superficie(1600.0, 1750.0, SURFACE_MIN_GAMES)
    assert elo == 1750.0 and uso is True


def test_ensemble_sin_datos_superficie_preserva_comportamiento_anterior():
    """Sin elo_superficie/games_superficie, debe dar exactamente lo mismo
    que antes de agregar esta feature."""
    engine = TennisImprovedEngine()
    con_default = engine.prob_match_winner_ensemble(1650.0, 1500.0, "A", "B", "clay", "best_of_3")
    con_none_explicito = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "clay", "best_of_3",
        elo1_superficie=None, elo2_superficie=None,
        games1_superficie=None, games2_superficie=None,
    )
    assert con_default["prob_j1"] == con_none_explicito["prob_j1"]
    assert con_default["debug"]["uso_elo_superficie"] == {"j1": False, "j2": False}


def test_ensemble_usa_elo_de_superficie_cuando_hay_muestra():
    """Jugador A es mucho mejor en clay que su overall sugiere — con
    suficiente muestra en clay, el ensemble debe reflejar eso."""
    engine = TennisImprovedEngine()

    resultado_overall = engine.prob_match_winner_ensemble(
        1500.0, 1500.0, "A", "B", "clay", "best_of_3",
    )
    resultado_superficie = engine.prob_match_winner_ensemble(
        1500.0, 1500.0, "A", "B", "clay", "best_of_3",
        elo1_superficie=1700.0, games1_superficie=SURFACE_MIN_GAMES,
    )

    assert resultado_overall["prob_j1"] == pytest.approx(0.5, abs=0.02)
    assert resultado_superficie["prob_j1"] > resultado_overall["prob_j1"]
    assert resultado_superficie["debug"]["uso_elo_superficie"] == {"j1": True, "j2": False}


def test_factor_generico_se_desactiva_cuando_se_usa_elo_de_superficie():
    """No se debe contar el efecto de la superficie dos veces (una vía el
    Elo específico, otra vía SURFACE_ELO_FACTOR)."""
    engine = TennisImprovedEngine()
    r = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "clay", "best_of_3",
        elo1_superficie=1650.0, games1_superficie=SURFACE_MIN_GAMES,
        elo2_superficie=1500.0, games2_superficie=SURFACE_MIN_GAMES,
    )
    # Con el factor desactivado, p_elo debe coincidir con el cálculo sin
    # ajuste de superficie (factor=1.0) sobre el mismo delta de Elo.
    p_elo_sin_factor = engine.prob_from_elo(1650.0, 1500.0, "clay", aplicar_factor_superficie=False)
    assert r["debug"]["p_elo"] == pytest.approx(p_elo_sin_factor, abs=1e-4)


def test_analizar_acepta_parametros_de_superficie():
    engine = TennisImprovedEngine()
    resultado = engine.analizar(
        "A", "B", 1500.0, 1500.0, "clay", "best_of_3",
        cuotas={"match_winner": {"1": 1.9, "2": 1.9}},
        elo1_superficie=1700.0, games1_superficie=SURFACE_MIN_GAMES,
    )
    assert resultado["modelo"]["match_winner"]["debug"]["uso_elo_superficie"]["j1"] is True
