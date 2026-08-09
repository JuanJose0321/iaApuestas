"""
Tests de regresión para el fix P1 del motor de tenis (ver
report_tennis_audit.md): el std_dev de la distribución de total de games
estaba hardcodeado (4.5 BO3, 6.0 BO5), más angosto que la variabilidad
real de los partidos, lo que inflaba el EV de picks en ese mercado.

Cubre:
  1. El archivo de calibración existe y tiene la forma esperada
  2. Los valores calibrados son positivos y en un rango físicamente
     razonable para tenis profesional
  3. El motor (TennisImprovedEngine) efectivamente usa los valores
     calibrados, no los defaults hardcodeados
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

STD_DEV_PATH = Path(__file__).parent.parent / "src" / "data" / "tennis_std_dev_calibrated.json"


def test_std_dev_calibrado_existe_y_tiene_forma_esperada():
    assert STD_DEV_PATH.exists(), "No se encontró tennis_std_dev_calibrated.json"

    with open(STD_DEV_PATH, encoding="utf-8") as f:
        config = json.load(f)

    assert "STD_DEV_BO3" in config
    assert "STD_DEV_BO5" in config
    assert config["STD_DEV_BO3"] > 0
    assert config["STD_DEV_BO5"] > 0


def test_std_dev_bo5_mayor_que_bo3():
    """Un partido a 5 sets acumula más varianza en games totales que uno a
    3 sets — el std_dev de BO5 debe ser mayor."""
    with open(STD_DEV_PATH, encoding="utf-8") as f:
        config = json.load(f)
    assert config["STD_DEV_BO5"] > config["STD_DEV_BO3"]


def test_std_dev_en_rango_fisicamente_razonable():
    with open(STD_DEV_PATH, encoding="utf-8") as f:
        config = json.load(f)
    # Un partido de tenis normalmente tiene entre 12 y 40 games en total;
    # un std_dev fuera de [2, 15] indicaría un bug en el cálculo, no una
    # calibración real.
    assert 2.0 <= config["STD_DEV_BO3"] <= 15.0
    assert 2.0 <= config["STD_DEV_BO5"] <= 15.0


def test_engine_usa_std_dev_calibrado_no_el_hardcode_original():
    """El motor debe reflejar los valores del JSON de calibración, no los
    4.5/6.0 originales (a menos que la calibración real haya coincidido
    con esos valores por casualidad, lo cual es estadísticamente
    improbable con miles de partidos reales)."""
    from src.engines.tennis_improved import STD_GAMES

    with open(STD_DEV_PATH, encoding="utf-8") as f:
        config = json.load(f)

    assert STD_GAMES["best_of_3"] == pytest.approx(config["STD_DEV_BO3"])
    assert STD_GAMES["best_of_5"] == pytest.approx(config["STD_DEV_BO5"])


def test_prob_total_games_usa_std_dev_calibrado():
    from src.engines.tennis_improved import TennisImprovedEngine, STD_GAMES

    engine = TennisImprovedEngine(elo_ratings={"A": 1600.0, "B": 1500.0})
    dist_bo3 = engine.prob_total_games(0.6, "best_of_3")
    dist_bo5 = engine.prob_total_games(0.6, "best_of_5")

    assert dist_bo3["std_dev"] == STD_GAMES["best_of_3"]
    assert dist_bo5["std_dev"] == STD_GAMES["best_of_5"]
