"""
Tests de generar_jugadores_por_genero.py — el selector de jugadores de
tenis se deriva de tennis_elo_ratings.json (campo "genero", derivado de
si el jugador jugó en archivos atp_matches_* o wta_matches_*), no se
mantiene más a mano. Antes, una lista curada de 226 jugadores nunca se
sincronizó con los 2,467 jugadores reales calibrados.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from generar_jugadores_por_genero import generar


@pytest.fixture
def ratings_fixture(tmp_path):
    ratings = {
        "jugadores": {
            "Rafael Nadal": {"elo": 1857.4, "games": 471, "genero": "M"},
            "Roger Federer": {"elo": 1924.6, "games": 309, "genero": "M"},
            "Serena Williams": {"elo": 1757.4, "games": 218, "genero": "F"},
            "Naomi Osaka": {"elo": 1863.7, "games": 383, "genero": "F"},
            "Jugador Sin Genero": {"elo": 1500.0, "games": 1},  # ratings viejos, sin el campo
        },
        "_meta": {},
    }
    path = tmp_path / "tennis_elo_ratings.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ratings, f)
    return path


def test_genera_separando_por_genero_real(tmp_path, ratings_fixture):
    output = tmp_path / "jugadores_por_genero.json"
    ok = generar(elo_ratings_path=ratings_fixture, output_path=output)
    assert ok is True

    with open(output, encoding="utf-8") as f:
        data = json.load(f)

    assert "Rafael Nadal" in data["masculino"]
    assert "Roger Federer" in data["masculino"]
    assert "Serena Williams" in data["femenino"]
    assert "Naomi Osaka" in data["femenino"]


def test_jugador_sin_genero_no_aparece_en_ninguna_lista(tmp_path, ratings_fixture):
    """Un registro de una calibración vieja sin el campo 'genero' no debe
    romper la generación — simplemente no aparece en ningún género."""
    output = tmp_path / "jugadores_por_genero.json"
    generar(elo_ratings_path=ratings_fixture, output_path=output)

    with open(output, encoding="utf-8") as f:
        data = json.load(f)

    assert "Jugador Sin Genero" not in data["masculino"]
    assert "Jugador Sin Genero" not in data["femenino"]


def test_listas_ordenadas_alfabeticamente(tmp_path, ratings_fixture):
    output = tmp_path / "jugadores_por_genero.json"
    generar(elo_ratings_path=ratings_fixture, output_path=output)

    with open(output, encoding="utf-8") as f:
        data = json.load(f)

    assert data["masculino"] == sorted(data["masculino"])
    assert data["femenino"] == sorted(data["femenino"])


def test_sin_archivo_de_ratings_falla_ordenadamente(tmp_path):
    output = tmp_path / "jugadores_por_genero.json"
    ok = generar(elo_ratings_path=tmp_path / "no_existe.json", output_path=output)
    assert ok is False
    assert not output.exists()


def test_meta_incluye_conteos_correctos(tmp_path, ratings_fixture):
    output = tmp_path / "jugadores_por_genero.json"
    generar(elo_ratings_path=ratings_fixture, output_path=output)

    with open(output, encoding="utf-8") as f:
        data = json.load(f)

    assert data["_meta"]["masculino_count"] == len(data["masculino"])
    assert data["_meta"]["femenino_count"] == len(data["femenino"])


def test_archivo_real_del_proyecto_incluye_jugadores_conocidos():
    """Test end-to-end contra el jugadores_por_genero.json real del
    repo (ya regenerado) — confirma que la sincronización funcionó."""
    path = Path(__file__).parent.parent / "src" / "data" / "jugadores_por_genero.json"
    if not path.exists():
        pytest.skip("jugadores_por_genero.json no existe en este entorno")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    conocidos = {
        "Rafael Nadal": "masculino", "Roger Federer": "masculino",
        "Serena Williams": "femenino", "Naomi Osaka": "femenino",
        "Ashleigh Barty": "femenino",
    }
    for nombre, genero in conocidos.items():
        assert nombre in data[genero], f"{nombre} debería estar en {genero}"
