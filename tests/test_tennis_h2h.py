"""
Tests de la señal de head-to-head (H2H) — ver report_tennis_audit.md /
tennis_backtest_results.md. Se mezcla ENCIMA del ensemble de Elo+Forma
ya existente, no lo reemplaza, y solo cuando hay suficiente historial
de enfrentamientos directos previos.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.tennis_improved import (
    TennisImprovedEngine, prob_from_h2h, h2h_vigente, H2H_MIN_PARTIDOS,
)


def test_prob_from_h2h_dominante():
    p = prob_from_h2h(h2h_ganados_j1=9, h2h_total=10)
    assert p == pytest.approx(0.66)  # 0.3 + 0.4*0.9


def test_prob_from_h2h_parejo():
    p = prob_from_h2h(h2h_ganados_j1=5, h2h_total=10)
    assert p == pytest.approx(0.5)


def test_h2h_vigente_insuficiente_muestra():
    assert h2h_vigente(H2H_MIN_PARTIDOS - 1) is False


def test_h2h_vigente_sin_datos():
    assert h2h_vigente(None) is False


def test_h2h_vigente_suficiente_muestra():
    assert h2h_vigente(H2H_MIN_PARTIDOS) is True


def test_ensemble_sin_h2h_preserva_comportamiento_anterior():
    engine = TennisImprovedEngine()
    con_default = engine.prob_match_winner_ensemble(1650.0, 1500.0, "A", "B", "hard", "best_of_3")
    con_none_explicito = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "hard", "best_of_3",
        h2h_ganados_j1=None, h2h_total=None,
    )
    assert con_default["prob_j1"] == con_none_explicito["prob_j1"]
    assert con_default["debug"]["usa_h2h"] is False


def test_ensemble_ignora_h2h_con_pocos_enfrentamientos():
    """Con menos de h2h_min_partidos, el H2H no debe afectar la predicción
    aunque se pase h2h_weight > 0 — mismo patrón de fallback que forma/superficie."""
    engine = TennisImprovedEngine()
    sin_h2h = engine.prob_match_winner_ensemble(1650.0, 1500.0, "A", "B", "hard", "best_of_3")
    con_poco_h2h = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "hard", "best_of_3",
        h2h_ganados_j1=1, h2h_total=2, h2h_weight=0.20, h2h_min_partidos=3,
    )
    assert sin_h2h["prob_j1"] == con_poco_h2h["prob_j1"]
    assert con_poco_h2h["debug"]["usa_h2h"] is False


def test_ensemble_aplica_h2h_con_suficiente_muestra():
    """H2H fuerte a favor de B (que en Elo es el menos favorito) debe
    reducir la ventaja de A frente al caso sin H2H."""
    engine = TennisImprovedEngine()
    sin_h2h = engine.prob_match_winner_ensemble(1650.0, 1500.0, "A", "B", "hard", "best_of_3")
    con_h2h = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "hard", "best_of_3",
        h2h_ganados_j1=1, h2h_total=5, h2h_weight=0.20, h2h_min_partidos=3,  # B le ganó 4 de 5
    )
    assert con_h2h["debug"]["usa_h2h"] is True
    assert con_h2h["prob_j1"] < sin_h2h["prob_j1"]


def test_h2h_weight_cero_no_activa_aunque_haya_datos():
    engine = TennisImprovedEngine()
    r = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "hard", "best_of_3",
        h2h_ganados_j1=5, h2h_total=5, h2h_weight=0.0,
    )
    assert r["debug"]["usa_h2h"] is False


def test_analizar_acepta_parametros_h2h():
    engine = TennisImprovedEngine()
    resultado = engine.analizar(
        "A", "B", 1650.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": 1.5, "2": 2.7}},
        h2h_ganados_j1=4, h2h_total=5, h2h_weight=0.15,
    )
    assert resultado["modelo"]["match_winner"]["debug"]["usa_h2h"] is True
