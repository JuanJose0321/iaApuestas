"""
Tests de regresión para el fix P0-1 del motor de tenis (ver
report_tennis_audit.md): el ensemble Elo+Forma diluía cada predicción con
una "forma" constante de 50% (form_stats siempre vacío en producción), lo
que sesgaba las probabilidades hacia el empate sin ninguna base estadística.

Cubre:
  1. get_form() devuelve None (no un 50% inventado) cuando no hay datos reales
  2. Sin forma real para alguno de los dos jugadores, el ensemble cae a 100% Elo
  3. Con forma real para ambos jugadores, sí se aplica el blend 70/30
  4. El campo "metodo" de la respuesta refleja lo que realmente se usó
  5. Endpoint /api/analizar_tenis: con los ratings actuales (sin forma
     calibrada todavía) usa Elo puro, no un ensemble fantasma
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.tennis_improved import TennisImprovedEngine


# ── Unit: motor con form_stats vacío (estado real de producción hoy) ──────

def test_get_form_sin_datos_devuelve_none():
    engine = TennisImprovedEngine(elo_ratings={"A": 1600.0, "B": 1500.0})
    assert engine.get_form("A") is None
    assert engine.get_form("B") is None


def test_ensemble_sin_forma_cae_a_elo_puro():
    """Sin form_stats, el ensemble debe igualar la probabilidad de Elo puro,
    no diluirla con un 50% constante (el bug original)."""
    engine = TennisImprovedEngine(elo_ratings={"A": 1650.0, "B": 1500.0})

    p_elo = engine.prob_from_elo(1650.0, 1500.0, "hard")
    resultado = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "hard", "best_of_3"
    )

    assert resultado["debug"]["usa_forma"] is False
    assert resultado["debug"]["p_form"] is None
    assert resultado["debug"]["ensemble"] == pytest.approx(p_elo, abs=1e-3)


def test_ensemble_con_forma_real_aplica_blend_70_30():
    """Con datos reales de forma para AMBOS jugadores, sí se debe aplicar
    el ensemble 70% Elo / 30% Forma."""
    engine = TennisImprovedEngine(
        elo_ratings={"A": 1650.0, "B": 1500.0},
        form_stats={
            "A": {"ganados": 8, "perdidos": 2, "porcentaje": 80.0},
            "B": {"ganados": 3, "perdidos": 7, "porcentaje": 30.0},
        },
    )

    p_elo = engine.prob_from_elo(1650.0, 1500.0, "hard")
    resultado = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "hard", "best_of_3"
    )

    assert resultado["debug"]["usa_forma"] is True
    assert resultado["debug"]["p_form"] is not None
    # El ensemble con forma real debe diferir de Elo puro (A tiene mejor forma)
    assert resultado["debug"]["ensemble"] != pytest.approx(p_elo, abs=1e-3)


def test_ensemble_forma_parcial_tambien_cae_a_elo_puro():
    """Si solo UNO de los dos jugadores tiene forma real, no se debe mezclar
    forma real de uno con un 50% inventado del otro — cae a Elo puro."""
    engine = TennisImprovedEngine(
        elo_ratings={"A": 1650.0, "B": 1500.0},
        form_stats={"A": {"ganados": 8, "perdidos": 2, "porcentaje": 80.0}},
    )

    p_elo = engine.prob_from_elo(1650.0, 1500.0, "hard")
    resultado = engine.prob_match_winner_ensemble(
        1650.0, 1500.0, "A", "B", "hard", "best_of_3"
    )

    assert resultado["debug"]["usa_forma"] is False
    assert resultado["debug"]["ensemble"] == pytest.approx(p_elo, abs=1e-3)


def test_metodo_refleja_lo_que_realmente_se_uso():
    engine_sin_forma = TennisImprovedEngine(elo_ratings={"A": 1650.0, "B": 1500.0})
    resultado = engine_sin_forma.analizar(
        "A", "B", 1650.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": 1.5, "2": 2.7}},
    )
    assert "Elo puro" in resultado["modelo"]["metodo"]
    assert resultado["modelo"]["forma"]["j1"] is None
    assert resultado["modelo"]["forma"]["j2"] is None

    engine_con_forma = TennisImprovedEngine(
        elo_ratings={"A": 1650.0, "B": 1500.0},
        form_stats={
            "A": {"ganados": 8, "perdidos": 2, "porcentaje": 80.0},
            "B": {"ganados": 3, "perdidos": 7, "porcentaje": 30.0},
        },
    )
    resultado2 = engine_con_forma.analizar(
        "A", "B", 1650.0, 1500.0, "hard", "best_of_3",
        cuotas={"match_winner": {"1": 1.5, "2": 2.7}},
    )
    assert "Ensemble" in resultado2["modelo"]["metodo"]
    assert resultado2["modelo"]["forma"]["j1"] is not None


# ── Integración: endpoint real ──────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("FLASK_TESTING", "1")
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def test_endpoint_analizar_tenis_forma_es_coherente(client):
    """El endpoint nunca debe devolver p_form=0.5 fijo simulando una señal
    real (el bug original de P0-1). O bien usa_forma=False y p_form/forma
    son None (sin datos), o bien usa_forma=True con forma real de ambos
    jugadores — nunca una mezcla ni un valor inventado."""
    resp = client.post(
        "/api/analizar_tenis",
        json={
            "jugador1": "Novak Djokovic",
            "jugador2": "Jannik Sinner",
            "superficie": "hard",
            "formato": "best_of_3",
            "cuotas": {"match_winner": {"1": 2.10, "2": 1.75}},
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    mw = data["modelo"]["match_winner"]
    forma = data["modelo"]["forma"]

    if mw["debug"]["usa_forma"]:
        assert mw["debug"]["p_form"] is not None
        assert forma["j1"] is not None and forma["j2"] is not None
        assert "porcentaje" in forma["j1"] and "porcentaje" in forma["j2"]
    else:
        assert mw["debug"]["p_form"] is None
        assert forma["j1"] is None or forma["j2"] is None


def test_endpoint_analizar_tenis_jugador_desconocido(client):
    """Jugadores fuera de los ratings calibrados caen a Elo default 1500
    para ambos y no deben crashear."""
    resp = client.post(
        "/api/analizar_tenis",
        json={
            "jugador1": "Jugador Inexistente Uno",
            "jugador2": "Jugador Inexistente Dos",
            "superficie": "hard",
            "cuotas": {"match_winner": {"1": 1.9, "2": 1.9}},
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["modelo"]["elo"]["j1"] == 1500.0
    assert data["modelo"]["elo"]["j2"] == 1500.0
