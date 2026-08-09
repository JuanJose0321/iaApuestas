"""
Tests del fix: el mercado Total Games solo evaluaba "over" para generar
picks — "under" se descartaba en silencio aunque el frontend sí lo
manda si el usuario la completa (ver report_tennis_audit.md sección 13,
punto 5). Mismo patrón que el fix de Match Winner J1/J2
(tests/test_tennis_pick_ambos_jugadores.py): cada lado se evalúa
independiente, con cuotas derivadas de la probabilidad REAL del motor
para no depender de un número inventado que cruce MIN_EV por casualidad.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.tennis_improved import TennisImprovedEngine


def _cuota_con_valor(prob: float, margen_extra: float = 0.20) -> float:
    return round((1.0 / prob) * (1.0 + margen_extra), 2)


def _cuota_sin_valor(prob: float, recorte: float = 0.10) -> float:
    return round((1.0 / prob) * (1.0 - recorte), 2)


@pytest.fixture
def engine():
    return TennisImprovedEngine()


def _picks(resultado):
    return resultado["picks_verdes"] + resultado["picks_amarillos"]


def _probs_over_under(engine, elo1, elo2, formato, linea):
    mw = engine.prob_match_winner_ensemble(elo1, elo2, "A", "B", "hard", formato)
    dist = engine.prob_total_games(mw["debug"]["ensemble"], formato)
    from src.core.probability import norm_cdf
    p_over = 1.0 - norm_cdf(linea, loc=dist["total_esp"], scale=dist["std_dev"])
    return p_over, 1.0 - p_over, dist["total_esp"]


def test_pick_over_cuando_tiene_valor_comportamiento_previo(engine):
    """Caso que ya funcionaba antes del fix — no debe romperse.
    linea=18.5 (bien por debajo de total_esp~22.7) para que la certeza
    del modelo supere UMBRAL_AMARILLO, no solo el EV — con la línea al
    total_esp la probabilidad ronda 0.5 y _calc_confianza descarta el
    pick por baja certeza aunque el EV sea alto."""
    linea = 18.5
    p_over, p_under, _ = _probs_over_under(engine, 1600.0, 1500.0, "best_of_3", linea)

    resultado = engine.analizar(
        "A", "B", 1600.0, 1500.0, "hard", "best_of_3",
        cuotas={"total_games": {
            "linea": linea,
            "over": _cuota_con_valor(p_over),
            "under": _cuota_sin_valor(p_under),
        }},
    )

    picks = _picks(resultado)
    assert any(p["pick"] == f"Over {linea} games" for p in picks)
    assert not any(p["pick"] == f"Under {linea} games" for p in picks)


def test_pick_under_cuando_tiene_valor_caso_antes_roto(engine):
    """Caso que antes del fix NUNCA se detectaba, sin importar cuánto
    valor real tuviera "under" — solo se miraba cuotas['over']."""
    linea = 18.5
    p_over, p_under, _ = _probs_over_under(engine, 1600.0, 1500.0, "best_of_3", linea)

    resultado = engine.analizar(
        "A", "B", 1600.0, 1500.0, "hard", "best_of_3",
        cuotas={"total_games": {
            "linea": linea,
            "over": _cuota_sin_valor(p_over),
            "under": _cuota_con_valor(p_under),
        }},
    )

    picks = _picks(resultado)
    assert any(p["pick"] == f"Under {linea} games" for p in picks), (
        "Under tenía valor real y debería generar pick — este es el bug que se arregló"
    )
    assert not any(p["pick"] == f"Over {linea} games" for p in picks)


def test_sin_pick_cuando_ninguno_tiene_valor(engine):
    linea = 22.5
    p_over, p_under, _ = _probs_over_under(engine, 1600.0, 1500.0, "best_of_3", linea)

    resultado = engine.analizar(
        "A", "B", 1600.0, 1500.0, "hard", "best_of_3",
        cuotas={"total_games": {
            "linea": linea,
            "over": _cuota_sin_valor(p_over),
            "under": _cuota_sin_valor(p_under),
        }},
    )
    assert _picks(resultado) == []


def test_over_y_under_pueden_generar_pick_si_ambos_superan_el_umbral(engine):
    """No es 'el mejor de los dos' — cada lado se evalúa independiente."""
    linea = 18.5
    p_over, p_under, _ = _probs_over_under(engine, 1600.0, 1500.0, "best_of_3", linea)

    resultado = engine.analizar(
        "A", "B", 1600.0, 1500.0, "hard", "best_of_3",
        cuotas={"total_games": {
            "linea": linea,
            "over": _cuota_con_valor(p_over, 0.30),
            "under": _cuota_con_valor(p_under, 0.30),
        }},
    )

    picks = _picks(resultado)
    nombres = {p["pick"] for p in picks}
    assert f"Over {linea} games" in nombres
    assert f"Under {linea} games" in nombres


def test_under_ausente_en_cuotas_no_genera_pick_ni_rompe(engine):
    """Si el usuario solo carga 'over' (como antes del fix), el
    comportamiento sigue siendo válido: no debe intentar evaluar 'under'."""
    linea = 18.5
    p_over, _, _ = _probs_over_under(engine, 1600.0, 1500.0, "best_of_3", linea)

    resultado = engine.analizar(
        "A", "B", 1600.0, 1500.0, "hard", "best_of_3",
        cuotas={"total_games": {"linea": linea, "over": _cuota_con_valor(p_over)}},
    )
    picks = _picks(resultado)
    assert any(p["pick"] == f"Over {linea} games" for p in picks)
