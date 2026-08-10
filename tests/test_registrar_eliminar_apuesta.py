"""
Tests de /api/registrar_apuesta con los picks nuevos (Under en Total Games,
J2 como ganador) — regresión sobre los fixes de tennis_improved.py que
generan estos picks (ver test_tennis_total_games_under.py y
test_tennis_pick_ambos_jugadores.py) — y de /api/eliminar_apuesta, que
permite borrar filas del historial (ej. duplicados registrados por error).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services import tracking


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "CSV_PATH", tmp_path / "apuestas.csv")
    monkeypatch.setattr(tracking, "CONFIG_PATH", tmp_path / "bankroll_config.json")
    monkeypatch.setattr(tracking, "BACKUP_DIR", tmp_path / "backup")

    import app as flask_app
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def _registrar(client, **overrides):
    body = {
        "liga": "Tenis", "local": "Iga Swiatek", "visitante": "Naomi Osaka",
        "pick_tipo": "TENIS", "pick_descripcion": "Over 22.5 games",
        "cuota": 1.90, "stake": 20.0, "prob_predicha": 0.6,
        "ev_predicho": 0.08, "confianza_score": 0.7, "confianza_badge": "amarillo",
    }
    body.update(overrides)
    return client.post("/api/registrar_apuesta", json=body)


# ── PROBLEMA 1: registro con picks Under / J2 ──────────────────────────────

def test_registrar_pick_under_total_games(client):
    r = _registrar(client, pick_descripcion="Under 22.5 games")
    assert r.status_code == 200
    j = r.get_json()
    assert "error" not in j
    assert j["id"] == 1


def test_registrar_pick_ganador_jugador2(client):
    r = _registrar(client, pick_descripcion="Naomi Osaka gana")
    assert r.status_code == 200
    j = r.get_json()
    assert "error" not in j
    assert j["id"] == 1


def test_registrar_persiste_pick_descripcion_tal_cual(client):
    _registrar(client, pick_descripcion="Under 22.5 games")
    rows = tracking.leer_historial(tracking.CSV_PATH)
    assert rows[0]["pick_descripcion"] == "Under 22.5 games"


def test_registrar_funciona_con_las_cuatro_combinaciones(client):
    descripciones = [
        "Over 22.5 games", "Under 22.5 games",
        "Iga Swiatek gana", "Naomi Osaka gana",
    ]
    ids = []
    for desc in descripciones:
        r = _registrar(client, pick_descripcion=desc)
        assert r.status_code == 200, desc
        j = r.get_json()
        assert "error" not in j, desc
        ids.append(j["id"])
    assert ids == [1, 2, 3, 4]


# ── PROBLEMA 2: eliminar apuestas del historial ────────────────────────────

def test_eliminar_apuesta_quita_fila_del_historial(client):
    id1 = _registrar(client, pick_descripcion="Over 22.5 games").get_json()["id"]
    id2 = _registrar(client, pick_descripcion="Over 22.5 games").get_json()["id"]  # duplicado

    assert len(tracking.leer_historial(tracking.CSV_PATH)) == 2

    r = client.delete(f"/api/eliminar_apuesta/{id2}")
    assert r.status_code == 200
    j = r.get_json()
    assert j["id"] == id2
    assert j["total_apuestas"] == 1

    rows = tracking.leer_historial(tracking.CSV_PATH)
    assert len(rows) == 1
    assert rows[0]["id"] == str(id1)


def test_eliminar_apuesta_inexistente_devuelve_error(client):
    r = client.delete("/api/eliminar_apuesta/999")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_eliminar_apuesta_no_afecta_otras_filas(client):
    ids = [_registrar(client, pick_descripcion=f"Over {n}.5 games").get_json()["id"]
           for n in range(20, 23)]

    r = client.delete(f"/api/eliminar_apuesta/{ids[1]}")
    assert r.status_code == 200

    rows = tracking.leer_historial(tracking.CSV_PATH)
    ids_restantes = {row["id"] for row in rows}
    assert ids_restantes == {str(ids[0]), str(ids[2])}
