"""
Tests del decay de Elo por inactividad — ver report_tennis_audit.md /
tennis_backtest_results.md. Antes: un jugador retirado o lesionado
mucho tiempo (ej. Federer, Barty) quedaba con su Elo de pico "congelado"
indefinidamente, sin importar cuánto tiempo llevara sin competir.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.tennis_improved import (
    TennisImprovedEngine, aplicar_decay_inactividad, ELO_PRIOR,
)


def test_decay_sin_meses_no_modifica():
    assert aplicar_decay_inactividad(1900.0, None, decay_por_mes=0.05) == 1900.0


def test_decay_por_mes_cero_no_modifica():
    """decay_por_mes=0 (default del módulo) debe ser un no-op aunque haya meses_inactivo."""
    assert aplicar_decay_inactividad(1900.0, 24.0, decay_por_mes=0.0) == 1900.0


def test_decay_cero_meses_no_modifica():
    assert aplicar_decay_inactividad(1900.0, 0.0, decay_por_mes=0.05) == pytest.approx(1900.0)


def test_decay_acerca_a_la_media_con_el_tiempo():
    elo_reciente = aplicar_decay_inactividad(1900.0, 1.0, decay_por_mes=0.05)
    elo_medio_plazo = aplicar_decay_inactividad(1900.0, 10.0, decay_por_mes=0.05)
    elo_largo_plazo = aplicar_decay_inactividad(1900.0, 30.0, decay_por_mes=0.05)

    dist_reciente = abs(1900.0 - elo_reciente)
    dist_medio = abs(1900.0 - elo_medio_plazo)
    dist_largo = abs(1900.0 - elo_largo_plazo)

    assert dist_reciente < dist_medio < dist_largo


def test_decay_tiene_piso_en_elo_medio():
    """Con suficiente inactividad, el Elo no puede pasarse de la media (factor tope en 1.0)."""
    elo = aplicar_decay_inactividad(1900.0, 1000.0, decay_por_mes=0.05)
    assert elo == pytest.approx(ELO_PRIOR)


def test_decay_funciona_simetrico_para_elo_bajo():
    """Un Elo por debajo de la media también debe subir hacia ella, no solo bajar."""
    elo = aplicar_decay_inactividad(1300.0, 1000.0, decay_por_mes=0.05)
    assert elo == pytest.approx(ELO_PRIOR)


def test_ensemble_sin_decay_preserva_comportamiento_anterior():
    engine = TennisImprovedEngine()
    con_default = engine.prob_match_winner_ensemble(1900.0, 1500.0, "A", "B", "hard", "best_of_3")
    con_none_explicito = engine.prob_match_winner_ensemble(
        1900.0, 1500.0, "A", "B", "hard", "best_of_3",
        meses_inactivo1=None, meses_inactivo2=None,
    )
    assert con_default["prob_j1"] == con_none_explicito["prob_j1"]


def test_ensemble_con_decay_reduce_ventaja_del_inactivo():
    """Jugador A con Elo alto pero muy inactivo debe perder ventaja frente
    a B (activo, mismo Elo nominal) cuando se activa el decay."""
    engine = TennisImprovedEngine()

    sin_decay = engine.prob_match_winner_ensemble(1900.0, 1500.0, "A", "B", "hard", "best_of_3")
    con_decay = engine.prob_match_winner_ensemble(
        1900.0, 1500.0, "A", "B", "hard", "best_of_3",
        meses_inactivo1=24.0, meses_inactivo2=0.0, decay_por_mes=0.05,
    )

    assert con_decay["prob_j1"] < sin_decay["prob_j1"]
