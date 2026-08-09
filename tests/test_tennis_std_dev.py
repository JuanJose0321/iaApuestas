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


# ── total_esp calibrado por regresión — fix del sesgo sistemático de +2.8/+3.1 games ──

def test_total_esp_bo3_bo5_existen_y_son_positivos():
    with open(STD_DEV_PATH, encoding="utf-8") as f:
        config = json.load(f)
    for clave in ("TOTAL_ESP_BO3", "TOTAL_ESP_BO5"):
        assert clave in config and config[clave] is not None, f"Falta {clave}"
        assert "a" in config[clave] and "b" in config[clave]
        assert config[clave]["a"] > 0


def test_total_esp_en_rango_fisicamente_razonable():
    """Para p*q entre 0 (partido totalmente desparejo) y 0.25 (perfectamente
    parejo), total_esp debe caer en un rango realista de games de tenis."""
    with open(STD_DEV_PATH, encoding="utf-8") as f:
        config = json.load(f)
    for clave, rango in (("TOTAL_ESP_BO3", (12, 35)), ("TOTAL_ESP_BO5", (18, 55))):
        a, b = config[clave]["a"], config[clave]["b"]
        for pq in (0.0, 0.25):
            total_esp = a + b * pq
            assert rango[0] <= total_esp <= rango[1], f"{clave} en pq={pq}: {total_esp}"


def test_engine_usa_total_esp_calibrado_no_la_formula_heuristica_vieja():
    """La fórmula heurística vieja (sets_esp*games_por_set) daba ~25-26
    games para partidos competitivos — el sesgo confirmado era de
    +2.8/+3.1 games sobre el real. El valor calibrado debe ser
    consistentemente más bajo en ese mismo rango de competitividad."""
    from src.engines.tennis_improved import TennisImprovedEngine

    engine = TennisImprovedEngine()
    dist = engine.prob_total_games(0.6, "best_of_3")  # partido moderadamente competitivo

    # La fórmula vieja daba total_esp≈25.7 para p=0.6 en BO3 (ver
    # report_tennis_audit.md) — el valor calibrado debe ser claramente menor.
    assert dist["total_esp"] < 25.0


def test_prob_total_games_total_esp_varia_con_competitividad():
    """total_esp debe seguir siendo mayor cuanto más parejo el partido
    (p cerca de 0.5) que cuando es un partido desparejo (p cerca de 1.0) —
    la regresión usa p*q, que es simétrica y máxima en p=0.5."""
    from src.engines.tennis_improved import TennisImprovedEngine

    engine = TennisImprovedEngine()
    dist_parejo = engine.prob_total_games(0.5, "best_of_3")
    dist_desparejo = engine.prob_total_games(0.95, "best_of_3")

    assert dist_parejo["total_esp"] > dist_desparejo["total_esp"]


def test_total_esp_cae_a_formula_heuristica_si_falta_calibracion(monkeypatch):
    """Si TOTAL_ESP_COEFS no tiene coeficientes para un formato (archivo
    viejo sin estos campos, o corrupto), debe caer a la fórmula
    heurística original — nunca crashear el motor por esto."""
    import src.engines.tennis_improved as ti

    monkeypatch.setitem(ti.TOTAL_ESP_COEFS, "best_of_3", None)
    engine = ti.TennisImprovedEngine()
    dist = engine.prob_total_games(0.6, "best_of_3")

    # Fórmula heurística original para p=0.6, best_of_3
    p, q = 0.6, 0.4
    pq = p * q
    sets_esp = 2.0 + 2.0 * pq
    competitiveness = min(2.0 * pq, 0.25)
    games_por_set = 10.0 + 1.5 * competitiveness
    esperado = round(sets_esp * games_por_set, 1)

    assert dist["total_esp"] == pytest.approx(esperado)
