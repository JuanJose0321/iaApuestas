"""
Tests de /api/analizar_tenis: el log de predicciones (tennis_predictions.py)
ahora guarda también las cuotas reales que cargó el usuario -- antes solo
tenía la probabilidad del modelo, así que era imposible reconstruir el EV
real de una predicción pasada para validar con datos históricos si un EV
alto predice mejor performance en tenis (ver report_tennis_audit.md).

Cubre específicamente que app.py elige la cuota del lado correcto
(match_winner "1" o "2") según quién sea el favorito real del modelo, no
siempre jugador1.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services import tennis_predictions as tp


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "CSV_PATH", tmp_path / "predicciones_tenis.csv")

    import app as flask_app
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def _analizar(client, **cuotas_extra):
    body = {
        "jugador1": "A", "jugador2": "B", "elo1": 1900, "elo2": 1700,
        "superficie": "hard", "formato": "best_of_3",
        "cuotas": {"match_winner": {"1": 1.30, "2": 3.40}, **cuotas_extra},
    }
    return client.post("/api/analizar_tenis", json=body)


def test_guarda_cuota_del_favorito_j1(client):
    # elo1 (1900) >> elo2 (1700) -> favorito real es jugador1 ("A")
    r = _analizar(client)
    assert r.status_code == 200
    pred_id = r.get_json()["prediccion_id"]

    rows = tp.leer_predicciones(csv_path=tp.CSV_PATH)
    fila = next(row for row in rows if row["id"] == str(pred_id))
    assert fila["favorito"] == "A"
    assert float(fila["cuota_favorito"]) == 1.30  # la cuota "1", no la "2"


def test_guarda_cuota_del_favorito_j2(client):
    body = {
        "jugador1": "A", "jugador2": "B", "elo1": 1700, "elo2": 1900,  # ahora B es favorito
        "superficie": "hard", "formato": "best_of_3",
        "cuotas": {"match_winner": {"1": 3.40, "2": 1.30}},
    }
    r = client.post("/api/analizar_tenis", json=body)
    assert r.status_code == 200
    pred_id = r.get_json()["prediccion_id"]

    rows = tp.leer_predicciones(csv_path=tp.CSV_PATH)
    fila = next(row for row in rows if row["id"] == str(pred_id))
    assert fila["favorito"] == "B"
    assert float(fila["cuota_favorito"]) == 1.30  # la cuota "2" (la de B), no la "1"


def test_guarda_linea_y_cuotas_de_total_games_si_se_cargaron(client):
    r = _analizar(client, total_games={"linea": 20.5, "over": 1.92, "under": 1.85})
    pred_id = r.get_json()["prediccion_id"]

    rows = tp.leer_predicciones(csv_path=tp.CSV_PATH)
    fila = next(row for row in rows if row["id"] == str(pred_id))
    assert float(fila["total_games_linea"]) == 20.5
    assert float(fila["cuota_total_games_over"]) == 1.92
    assert float(fila["cuota_total_games_under"]) == 1.85


def test_sin_total_games_deja_esos_campos_en_blanco(client):
    r = _analizar(client)  # sin total_games en el body
    pred_id = r.get_json()["prediccion_id"]

    rows = tp.leer_predicciones(csv_path=tp.CSV_PATH)
    fila = next(row for row in rows if row["id"] == str(pred_id))
    assert fila["total_games_linea"] == ""
    assert fila["cuota_total_games_over"] == ""
    assert fila["cuota_total_games_under"] == ""
