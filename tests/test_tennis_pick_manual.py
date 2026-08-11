"""
Tests del FEATURE: registrar manualmente cualquier análisis de tenis,
tenga o no valor detectado por el sistema.

Cuando ni match_winner ni total_games generan un pick verde/amarillo
(EV/confianza insuficientes), el motor debe seguir devolviendo la
probabilidad real del modelo y la cuota que el usuario cargó, marcadas
como "picks_manual" / tiene_valor=False — para que el frontend arme una
tarjeta registrable en vez de solo un texto plano de "sin picks".
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.tennis_improved import TennisImprovedEngine
from src.services import tracking


def _cuota_sin_valor(prob: float, recorte: float = 0.10) -> float:
    """Cuota recortada respecto de la justa: EV negativo, nunca cruza MIN_EV."""
    return round((1.0 / prob) * (1.0 - recorte), 2)


@pytest.fixture
def engine():
    return TennisImprovedEngine()


def _cuotas_sin_valor(engine, elo1, elo2, formato, linea):
    """Arma cuotas 'justas' (recortadas) para match_winner y total_games,
    de modo que ningún lado de ningún mercado cruce MIN_EV."""
    mw = engine.prob_match_winner_ensemble(elo1, elo2, "A", "B", "hard", formato)
    dist = engine.prob_total_games(mw["debug"]["ensemble"], formato)
    from src.core.probability import norm_cdf
    p_over = 1.0 - norm_cdf(linea, loc=dist["total_esp"], scale=dist["std_dev"])
    return {
        "match_winner": {
            "1": _cuota_sin_valor(mw["prob_j1"]),
            "2": _cuota_sin_valor(mw["prob_j2"]),
        },
        "total_games": {
            "linea": linea,
            "over":  _cuota_sin_valor(p_over),
            "under": _cuota_sin_valor(1.0 - p_over),
        },
    }, mw


# ── Motor: análisis sin valor devuelve datos para la tarjeta manual ───────

def test_sin_valor_devuelve_picks_manual_con_datos_reales(engine):
    cuotas, mw = _cuotas_sin_valor(engine, 1600.0, 1500.0, "best_of_3", 22.5)

    resultado = engine.analizar(
        "A", "B", 1600.0, 1500.0, "hard", "best_of_3", cuotas,
    )

    assert resultado["picks_verdes"] == []
    assert resultado["picks_amarillos"] == []
    assert resultado["picks_manual"], "debe ofrecer al menos un pick manual"

    for p in resultado["picks_manual"]:
        assert p["tiene_valor"] is False
        assert p["confianza_nivel"] == "manual"
        assert p["cuota"] > 1.0
        assert 0.0 < p["prob"] < 1.0
        # cuotas recortadas -> EV negativo en ambos mercados -> sin_base
        assert p["senal"] == "sin_base"

    mercados = {p["mercado"] for p in resultado["picks_manual"]}
    assert "Match Winner" in mercados


# ── _senal_manual: clasifica por qué tan convencido está el modelo ────────

@pytest.mark.parametrize("prob,ev,esperado", [
    (0.70, 0.02, "solido"),       # prob >= UMBRAL_PROB_SOLIDO (0.65)
    (0.65, 0.001, "solido"),      # límite inclusive
    (0.60, 0.02, "razonable"),    # entre 0.55 y 0.65
    (0.55, 0.001, "razonable"),   # límite inclusive
    (0.52, 0.02, "cuota_larga"),  # EV positivo (cuota larga) pero prob ~50/50
    (0.50, 0.001, "cuota_larga"),
    (0.90, 0.0, "sin_base"),      # EV <= 0 manda, sin importar la prob
    (0.90, -0.05, "sin_base"),
])
def test_senal_manual_clasifica_por_umbral_de_probabilidad(engine, prob, ev, esperado):
    assert engine._senal_manual(prob, ev) == esperado


def test_pick_manual_match_winner_usa_favorito_real_y_cuota_cargada(engine):
    cuotas, mw = _cuotas_sin_valor(engine, 1600.0, 1500.0, "best_of_3", 22.5)

    resultado = engine.analizar("A", "B", 1600.0, 1500.0, "hard", "best_of_3", cuotas)

    mw_manual = next(p for p in resultado["picks_manual"] if p["mercado"] == "Match Winner")
    favorito_esperado = "A" if mw["prob_j1"] >= mw["prob_j2"] else "B"
    cuota_esperada = cuotas["match_winner"]["1" if favorito_esperado == "A" else "2"]

    assert mw_manual["pick"] == f"{favorito_esperado} gana"
    assert mw_manual["cuota"] == cuota_esperada
    assert mw_manual["prob"] == max(mw["prob_j1"], mw["prob_j2"])


def test_con_pick_de_valor_no_se_generan_picks_manual(engine):
    """Si ya hay un pick verde/amarillo real, no hace falta ofrecer manual."""
    linea = 18.5
    mw = engine.prob_match_winner_ensemble(1600.0, 1500.0, "A", "B", "hard", "best_of_3")
    dist = engine.prob_total_games(mw["debug"]["ensemble"], "best_of_3")
    from src.core.probability import norm_cdf
    p_over = 1.0 - norm_cdf(linea, loc=dist["total_esp"], scale=dist["std_dev"])

    cuotas = {"total_games": {
        "linea": linea,
        "over": round((1.0 / p_over) * 1.20, 2),   # con valor real
        "under": _cuota_sin_valor(1.0 - p_over),
    }}
    resultado = engine.analizar("A", "B", 1600.0, 1500.0, "hard", "best_of_3", cuotas)

    assert resultado["picks_verdes"] or resultado["picks_amarillos"]
    assert resultado["picks_manual"] == []


def test_sin_total_games_en_cuotas_no_rompe(engine):
    """Si el usuario no cargó total_games, solo debe faltar ese pick manual,
    sin romper el análisis."""
    cuotas = {"match_winner": {
        "1": _cuota_sin_valor(0.5), "2": _cuota_sin_valor(0.5),
    }}
    resultado = engine.analizar("A", "B", 1500.0, 1500.0, "hard", "best_of_3", cuotas)
    mercados = {p["mercado"] for p in resultado["picks_manual"]}
    assert "Match Winner" in mercados
    assert not any(m.startswith("Total Games") for m in mercados)


# ── Registro de un pick manual: guarda el flag distintivo ─────────────────

@pytest.fixture
def csv_path(tmp_path, monkeypatch):
    # actualizar_resultado() escribe bankroll vía leer_config()/guardar_config(),
    # que no toman csv_path — hay que aislar también CONFIG_PATH o el test
    # termina pisando el bankroll_config.json real del proyecto.
    monkeypatch.setattr(tracking, "CONFIG_PATH", tmp_path / "bankroll_config.json")
    monkeypatch.setattr(tracking, "BACKUP_DIR", tmp_path / "backup")
    return tmp_path / "apuestas.csv"


def test_registrar_pick_manual_guarda_confianza_badge_manual(csv_path):
    resultado = tracking.registrar_apuesta({
        "liga": "Tenis", "local": "A", "visitante": "B",
        "pick_tipo": "TENIS", "pick_descripcion": "A gana",
        "cuota": 1.75, "stake": 20.0, "prob_predicha": 0.55,
        "ev_predicho": -0.04, "confianza_score": 0.5,
        "confianza_badge": "manual",
    }, csv_path=csv_path)

    rows = tracking.leer_historial(csv_path)
    assert len(rows) == 1
    assert rows[0]["confianza_badge"] == "manual"
    assert resultado["id"] == 1


def test_endpoint_analizar_tenis_incluye_picks_manual_con_stake_fijo(engine, monkeypatch, tmp_path):
    monkeypatch.setattr(tracking, "CSV_PATH", tmp_path / "apuestas.csv")
    monkeypatch.setattr(tracking, "CONFIG_PATH", tmp_path / "bankroll_config.json")
    monkeypatch.setattr(tracking, "BACKUP_DIR", tmp_path / "backup")

    mw = engine.prob_match_winner_ensemble(1600.0, 1500.0, "A", "B", "hard", "best_of_3")

    import app as flask_app
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        cfg = tracking.leer_config()
        body = {
            "jugador1": "A", "jugador2": "B", "elo1": 1600, "elo2": 1500,
            "superficie": "hard", "formato": "best_of_3",
            "cuotas": {
                "match_winner": {
                    "1": _cuota_sin_valor(mw["prob_j1"]),
                    "2": _cuota_sin_valor(mw["prob_j2"]),
                },
            },
        }
        r = c.post("/api/analizar_tenis", json=body)
        assert r.status_code == 200
        j = r.get_json()
        assert j["picks_verdes"] == [] and j["picks_amarillos"] == []
        assert j["picks_manual"], "debe ofrecer la tarjeta manual"
        for p in j["picks_manual"]:
            assert p["stake_sugerido"] == float(cfg.get("stake_fijo", 20.0))


def test_metricas_no_mezclan_aciertos_manual_con_sistema(csv_path):
    # Pick del sistema: gana
    tracking.registrar_apuesta({
        "local": "A", "visitante": "B", "pick_descripcion": "A gana",
        "cuota": 1.90, "stake": 20.0, "confianza_badge": "verde",
    }, csv_path=csv_path)
    # Pick manual: pierde
    tracking.registrar_apuesta({
        "local": "C", "visitante": "D", "pick_descripcion": "C gana",
        "cuota": 1.75, "stake": 20.0, "confianza_badge": "manual",
    }, csv_path=csv_path)

    rows = tracking.leer_historial(csv_path)
    tracking.actualizar_resultado(int(rows[0]["id"]), "ganada", csv_path=csv_path)
    tracking.actualizar_resultado(int(rows[1]["id"]), "perdida", csv_path=csv_path)

    m = tracking.calcular_metricas(csv_path=csv_path)
    assert m["tasa_acierto_sistema"] == 100.0
    assert m["tasa_acierto_manual"] == 0.0
    assert m["n_manual"] == 1
