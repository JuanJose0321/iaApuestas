"""
Tests del log de predicciones de tenis (src/services/tennis_predictions.py):
registra CADA análisis (con o sin pick), permite cargar el resultado real
y calcula acerto_ganador/acerto_total automáticamente.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services import tennis_predictions as tp


@pytest.fixture
def csv_path(tmp_path):
    return tmp_path / "predicciones_tenis_test.csv"


def _registrar(csv_path, **overrides):
    data = {
        "jugador1": "Carlos Alcaraz", "jugador2": "Novak Djokovic",
        "favorito": "Carlos Alcaraz", "prob_favorito": 0.62,
        "superficie": "hard", "formato": "best_of_3", "total_esp": 22.4,
        "tuvo_pick": True, "tipo_pick": "Match Winner",
    }
    data.update(overrides)
    return tp.registrar_prediccion(data, csv_path=csv_path)


def test_registrar_prediccion_asigna_id_incremental(csv_path):
    r1 = _registrar(csv_path)
    r2 = _registrar(csv_path)
    assert r1["id"] == 1
    assert r2["id"] == 2


def test_leer_predicciones_incluye_predicciones_sin_pick(csv_path):
    _registrar(csv_path, tuvo_pick=False, tipo_pick="")
    rows = tp.leer_predicciones(csv_path=csv_path)
    assert len(rows) == 1
    assert rows[0]["tuvo_pick"] == "False"


def test_cargar_resultado_ganador_correcto_marca_acerto_true(csv_path):
    r = _registrar(csv_path, favorito="Carlos Alcaraz")
    resultado = tp.cargar_resultado(r["id"], ganador_real="Carlos Alcaraz", csv_path=csv_path)
    assert resultado["acerto_ganador"] is True


def test_cargar_resultado_ganador_incorrecto_marca_acerto_false(csv_path):
    r = _registrar(csv_path, favorito="Carlos Alcaraz")
    resultado = tp.cargar_resultado(r["id"], ganador_real="Novak Djokovic", csv_path=csv_path)
    assert resultado["acerto_ganador"] is False


def test_cargar_resultado_total_dentro_de_std_dev_marca_acerto_true(csv_path):
    r = _registrar(csv_path, formato="best_of_3", total_esp=22.0)
    # std_dev de best_of_3 ronda ~4.5 games (calibrado) — 23 está muy cerca de 22
    resultado = tp.cargar_resultado(r["id"], total_games_real=23, csv_path=csv_path)
    assert resultado["acerto_total"] is True


def test_cargar_resultado_total_lejos_marca_acerto_false(csv_path):
    r = _registrar(csv_path, formato="best_of_3", total_esp=22.0)
    resultado = tp.cargar_resultado(r["id"], total_games_real=40, csv_path=csv_path)
    assert resultado["acerto_total"] is False


def test_cargar_resultado_solo_ganador_no_toca_acerto_total(csv_path):
    r = _registrar(csv_path)
    resultado = tp.cargar_resultado(r["id"], ganador_real="Carlos Alcaraz", csv_path=csv_path)
    assert "acerto_total" not in resultado
    assert "total_games_real" not in resultado


def test_cargar_resultado_sin_ganador_ni_total_lanza_error(csv_path):
    r = _registrar(csv_path)
    with pytest.raises(ValueError):
        tp.cargar_resultado(r["id"], csv_path=csv_path)


def test_cargar_resultado_id_inexistente_lanza_error(csv_path):
    with pytest.raises(ValueError):
        tp.cargar_resultado(999, ganador_real="X", csv_path=csv_path)


def test_calcular_precision_sin_resultados_cargados_devuelve_none(csv_path):
    _registrar(csv_path)
    precision = tp.calcular_precision(csv_path=csv_path)
    assert precision["n_con_ganador_cargado"] == 0
    assert precision["precision_ganador_pct"] is None


def test_calcular_precision_separa_ganador_y_total(csv_path):
    r1 = _registrar(csv_path, favorito="Carlos Alcaraz", formato="best_of_3", total_esp=22.0)
    tp.cargar_resultado(r1["id"], ganador_real="Carlos Alcaraz", total_games_real=23, csv_path=csv_path)

    r2 = _registrar(csv_path, favorito="Carlos Alcaraz", formato="best_of_3", total_esp=22.0)
    tp.cargar_resultado(r2["id"], ganador_real="Novak Djokovic", total_games_real=40, csv_path=csv_path)

    precision = tp.calcular_precision(csv_path=csv_path)
    assert precision["n_total_predicciones"] == 2
    assert precision["n_con_ganador_cargado"] == 2
    assert precision["precision_ganador_pct"] == 50.0
    assert precision["n_con_total_cargado"] == 2
    assert precision["precision_total_pct"] == 50.0
