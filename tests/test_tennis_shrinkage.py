"""
Tests de la regresión a la media (shrinkage) del Elo de tenis — ver
report_tennis_audit.md sección 9/10: un Elo calculado con pocos partidos
es más ruidoso que el mismo Elo con historial largo, pero antes de este
fix el motor los trataba con la misma confianza.

Cubre:
  1. shrink_elo(): games=None no toca el Elo (retrocompatibilidad)
  2. shrink_elo(): games=0 devuelve el prior puro (1500)
  3. shrink_elo(): más partidos = menos shrink (monotonía)
  4. Mismo Elo nominal, jugador con pocos partidos da una probabilidad
     más conservadora (más cerca de 0.5) que uno con historial largo
  5. prob_match_winner_ensemble / analizar: games1/games2=None preserva
     el comportamiento anterior (no rompe nada de lo ya validado)
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.tennis_improved import TennisImprovedEngine, shrink_elo, ELO_PRIOR


def test_shrink_elo_sin_games_no_modifica():
    assert shrink_elo(1650.0, None) == 1650.0


def test_shrink_elo_cero_partidos_da_prior_puro():
    assert shrink_elo(1650.0, 0) == pytest.approx(ELO_PRIOR)


def test_shrink_elo_mas_partidos_menos_shrink():
    """A más partidos, el Elo ajustado debe acercarse más al Elo real (menos shrink)."""
    elo_pocos = shrink_elo(1650.0, 3)
    elo_medio = shrink_elo(1650.0, 20)
    elo_muchos = shrink_elo(1650.0, 500)

    dist_pocos = abs(1650.0 - elo_pocos)
    dist_medio = abs(1650.0 - elo_medio)
    dist_muchos = abs(1650.0 - elo_muchos)

    assert dist_pocos > dist_medio > dist_muchos
    # Con muchísimos partidos, el shrink debe ser casi despreciable
    assert dist_muchos < 10.0


def test_probabilidad_mas_conservadora_con_pocos_partidos():
    """Mismo Elo nominal (1650 vs 1500), pero J1 con 3 partidos vs 200 —
    la probabilidad de J1 con pocos partidos debe estar más cerca de 0.5
    (más conservadora) que la de J1 con historial largo."""
    engine = TennisImprovedEngine()

    mw_pocos = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "hard", "best_of_3", games1=3, games2=200
    )
    mw_muchos = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "hard", "best_of_3", games1=200, games2=200
    )

    p_pocos = mw_pocos["prob_j1"]
    p_muchos = mw_muchos["prob_j1"]

    # Ambos favorecen a A (elo nominal más alto), pero el de pocos partidos
    # debe ser menos extremo (más cerca de 0.5)
    assert 0.5 < p_pocos < p_muchos


def test_ensemble_sin_games_preserva_comportamiento_anterior():
    """games1/games2=None (default) debe dar exactamente el mismo resultado
    que antes de agregar el shrinkage — no debe romper nada ya validado."""
    engine = TennisImprovedEngine()
    con_default = engine.prob_match_winner_ensemble(1650.0, 1500.0, "A", "B", "hard", "best_of_3")
    con_none_explicito = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "hard", "best_of_3", games1=None, games2=None
    )
    assert con_default["prob_j1"] == con_none_explicito["prob_j1"]
    assert con_default["debug"]["elo1_ajustado"] == 1650.0
    assert con_default["debug"]["elo2_ajustado"] == 1500.0


def test_analizar_acepta_games_opcionales():
    engine = TennisImprovedEngine()
    resultado = engine.analizar(
        "A", "B", 1650.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": 1.5, "2": 2.7}},
        games1=3, games2=200,
    )
    assert resultado["modelo"]["match_winner"]["debug"]["elo1_ajustado"] < 1650.0
