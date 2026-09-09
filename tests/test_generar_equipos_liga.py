"""
Tests de generar_equipos_liga.py: la regeneración automática de equipos
vigentes para las 5 ligas domésticas con CSV propio (ver commit del fix de
raíz -- antes tomaba la unión de TODAS las temporadas históricas y nunca
"sacaba" a un equipo ya descendido).

Todo corre contra un data/raw y un equipos_por_liga.json de prueba
(monkeypatch de las rutas del módulo) -- nunca toca los archivos reales del
repo.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generar_equipos_liga as gen


def _escribir_csv(path: Path, filas: list[tuple[str, str]]) -> None:
    """filas = [(HomeTeam, AwayTeam), ...] -- solo las columnas que el script usa."""
    lineas = ["Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR"]
    for home, away in filas:
        lineas.append(f"15/08/2026,{home},{away},1,0,H")
    path.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _json_base(**ligas_extra) -> dict:
    return {
        "_meta": {"temporada": "vieja", "nota": "nota original", "actualizado": "2020-01-01", "verificado": True},
        "Champions League": ["Equipo Intacto"],
        **ligas_extra,
    }


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    output = tmp_path / "src" / "data" / "equipos_por_liga.json"
    output.parent.mkdir(parents=True)
    monkeypatch.setattr(gen, "DATA_DIR", data_dir)
    monkeypatch.setattr(gen, "OUTPUT", output)
    return data_dir, output


def test_toma_solo_la_temporada_mas_reciente_no_la_union_historica(entorno):
    """El bug de raíz: un equipo descendido en una temporada vieja no debe
    sobrevivir en el resultado final aunque haya jugado en el histórico."""
    data_dir, output = entorno
    # Temporada vieja: incluye "Malaga" en LaLiga (descendido hace años en la
    # realidad, pero acá lo que importa es que NO esté en la temporada nueva).
    _escribir_csv(data_dir / "SP1_2425.csv", [("Real Madrid", "Malaga"), ("Barcelona", "Real Madrid")])
    # Temporada vigente: Malaga ya no juega, aparece Villarreal en su lugar.
    _escribir_csv(data_dir / "SP1_2627.csv", [("Real Madrid", "Villarreal"), ("Barcelona", "Real Madrid")])

    resultado = gen.generar_equipos_liga()

    assert resultado["LaLiga"] == sorted(["Real Madrid", "Villarreal", "Barcelona"])
    assert "Malaga" not in resultado["LaLiga"]


def test_mapea_codigos_csv_a_nombres_de_display(entorno):
    data_dir, output = entorno
    _escribir_csv(data_dir / "E0_2627.csv", [("Man City", "Nott'm Forest"), ("Arsenal", "Man United")])

    resultado = gen.generar_equipos_liga()

    assert resultado["Premier League"] == sorted(
        ["Manchester City", "Nottingham Forest", "Arsenal", "Manchester United"]
    )


def test_codigo_sin_mapear_corta_la_ejecucion(entorno, capsys):
    data_dir, output = entorno
    _escribir_csv(data_dir / "SP1_2627.csv", [("Real Madrid", "Equipo Marciano FC")])

    with pytest.raises(SystemExit) as exc:
        gen.generar_equipos_liga()

    assert exc.value.code == 1
    assert "Equipo Marciano FC" in capsys.readouterr().out


def test_actualizar_json_preserva_otras_ligas_y_solo_commitea_si_cambia(entorno):
    data_dir, output = entorno
    output.write_text(json.dumps(_json_base(LaLiga=["Barcelona"])), encoding="utf-8")

    cambio1 = gen.actualizar_json({"LaLiga": ["Barcelona", "Real Madrid"]})
    assert cambio1 is True

    guardado = json.loads(output.read_text(encoding="utf-8"))
    assert guardado["LaLiga"] == ["Barcelona", "Real Madrid"]
    assert guardado["Champions League"] == ["Equipo Intacto"]  # intacta
    assert guardado["_meta"]["nota"] == "nota original"        # intacta
    assert guardado["_meta"]["actualizado"] != "2020-01-01"    # se actualizó

    # Segunda corrida con el mismo resultado: no debe reescribir nada.
    cambio2 = gen.actualizar_json({"LaLiga": ["Barcelona", "Real Madrid"]})
    assert cambio2 is False
