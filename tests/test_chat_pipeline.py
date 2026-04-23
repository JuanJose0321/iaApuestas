"""
Tests de los puntos de fallo silencioso del endpoint /chat,
y del endpoint /api/teams (dropdown de equipos por liga).

Cubre:
  1. Picks descartados por confianza baja (factor_datos=0.6 sin API)
  2. Picks descartados por contradicciones
  3. Engine devuelve None en todas las categorías (no hay EV ni rango)
  4. Narrativa: fallback cuando GROQ_API_KEY está ausente
  5. API-Football caída → degradación elegante (factor_datos=0.6)
  6. Formato JSON de /chat: claves obligatorias siempre presentes
  7. Motor ML: equipos desconocidos caen a solo Poisson sin crash
  8. calcular_confianza: factor_datos bajo hace que picks válidos no pasen umbral
  9. /api/teams: Liga MX devuelve Cruz Azul y Tigres UANL
 10. /api/teams: liga inexistente → 404 con cuerpo de error claro
 11. /api/teams: sin ?liga → dict completo con todas las ligas
 12. /api/teams: el filtro de texto (Mon→Monterrey) está en la lista
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Fixture: app en modo test ───────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("FLASK_TESTING", "1")
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


CUOTAS_BASE = {
    "1X2":    {"1": 2.30, "X": 3.40, "2": 3.10},
    "OU_2.5": {"Over": 1.85, "Under": 2.00},
    "BTTS":   {"Yes": 1.75, "No": 2.10},
}

PAYLOAD_BASE = {
    "home": "Real Madrid",
    "away": "Barcelona",
    "liga": "LaLiga",
    "cuotas": CUOTAS_BASE,
}


def _post_chat(client, payload=None):
    p = payload or PAYLOAD_BASE
    return client.post("/chat", data=json.dumps(p),
                       content_type="application/json")


# ── Test 1: respuesta siempre devuelve 200 y claves obligatorias ────────
def test_chat_keys_always_present(client):
    resp = _post_chat(client)
    assert resp.status_code == 200
    body = resp.get_json()
    for key in ("partido", "picks", "mensaje", "narrativa", "coherencia",
                "contexto_api", "bankroll", "lambdas", "fuente", "debug_filtrado"):
        assert key in body, f"Clave '{key}' ausente en respuesta"


# ── Test 2: picks es lista (puede estar vacía, pero NUNCA None) ─────────
def test_chat_picks_is_list(client):
    resp = _post_chat(client)
    body = resp.get_json()
    assert isinstance(body["picks"], list), "picks debe ser una lista, no None"


# ── Test 3: narrativa nunca es None (fallback cubre ausencia de LLM) ────
def test_chat_narrativa_not_none(client):
    resp = _post_chat(client)
    body = resp.get_json()
    assert body["narrativa"] is not None, "narrativa es None — fallback_sin_llm falló"
    assert len(body["narrativa"]) > 10, "narrativa está prácticamente vacía"


# ── Test 4: umbrales dinámicos (Fix C) ─────────────────────────────────
def test_umbral_sin_api_es_menor():
    from src.confidence import UMBRAL_CONFIANZA, UMBRAL_CONFIANZA_SIN_API
    assert UMBRAL_CONFIANZA_SIN_API < UMBRAL_CONFIANZA, (
        "El umbral sin API debe ser más permisivo que el umbral con API"
    )


def test_calcular_confianza_factor_bajo_pasa_umbral_sin_api():
    from src.confidence import calcular_confianza, UMBRAL_CONFIANZA_SIN_API
    # EV=12%, prob=0.50, sin datos reales → factor=0.6
    # Con el umbral reducido debe pasar
    score = calcular_confianza(prob=0.50, ev=0.12, factor_datos=0.6)
    assert score >= UMBRAL_CONFIANZA_SIN_API, (
        f"Con factor=0.6 y EV=12%, score={score:.4f} debería pasar "
        f"UMBRAL_CONFIANZA_SIN_API={UMBRAL_CONFIANZA_SIN_API}"
    )


def test_calcular_confianza_factor_alto():
    from src.confidence import calcular_confianza, UMBRAL_CONFIANZA
    # EV=15%, prob=0.55, con datos reales → factor=1.0
    score = calcular_confianza(prob=0.55, ev=0.15, factor_datos=1.0)
    assert score >= UMBRAL_CONFIANZA, (
        f"Con factor_datos=1.0 y EV alto, confianza={score:.4f} debería "
        f"pasar el umbral {UMBRAL_CONFIANZA}"
    )


# ── Test 5: contradicciones detectadas correctamente ────────────────────
def test_contradicciones_btts_no_over():
    from src.confidence import verificar_contradicciones_combo
    legs = [
        {"mercado": "BTTS", "seleccion": "No"},
        {"mercado": "OU_2.5", "seleccion": "Over"},
    ]
    result = verificar_contradicciones_combo(legs)
    assert len(result) > 0, "BTTS No + Over 2.5 debería detectarse como contradicción"


def test_sin_contradicciones():
    from src.confidence import verificar_contradicciones_combo
    legs = [
        {"mercado": "1X2", "seleccion": "1"},
        {"mercado": "OU_2.5", "seleccion": "Over"},
    ]
    result = verificar_contradicciones_combo(legs)
    assert result == [], f"1X2 + Over 2.5 no debería tener contradicciones: {result}"


# ── Test 6: API-Football caída → degradación elegante ──────────────────
def test_chat_api_football_down(client):
    ctx_vacio = {
        "api_disponible": False, "fuente": "thesportsdb",
        "home": "Real Madrid", "away": "Barcelona",
        "home_id": None, "away_id": None,
        "forma_home": None, "forma_away": None, "h2h": None,
        "injuries_home": [], "injuries_away": [], "notas": [],
    }
    with (
        patch("src.api_football.contexto_partido_completo",
              side_effect=Exception("timeout simulado")),
        patch("src.sportsmonk.disponible", return_value=False),
        patch("src.thesportsdb.contexto_partido_completo", return_value=ctx_vacio),
    ):
        resp = _post_chat(client)
    assert resp.status_code == 200
    body = resp.get_json()
    ctx = body["contexto_api"]
    assert ctx.get("api_disponible") is False, (
        "Con API caida, api_disponible deberia ser False"
    )


# ── Test 7: motor con equipos desconocidos cae a solo Poisson ───────────
def test_engine_equipos_desconocidos(client):
    payload = {**PAYLOAD_BASE,
               "home": "Equipo_XYZ_Desconocido",
               "away": "Equipo_ABC_Desconocido"}
    resp = _post_chat(client, payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert "solo Poisson" in body["fuente"], (
        f"Equipos desconocidos deberían usar 'solo Poisson', got: {body['fuente']}"
    )


# ── Test 8: validación de entrada — faltan cuotas ───────────────────────
def test_chat_sin_cuotas_1x2(client):
    payload = {"home": "Real Madrid", "away": "Barcelona",
               "cuotas": {"OU_2.5": {"Over": 1.85, "Under": 2.00}}}
    resp = _post_chat(client, payload)
    assert resp.status_code == 400
    body = resp.get_json()
    assert "error" in body


def test_chat_cuota_invalida(client):
    payload = {**PAYLOAD_BASE,
               "cuotas": {"1X2": {"1": 0.5, "X": 3.40, "2": 3.10}}}
    resp = _post_chat(client, payload)
    assert resp.status_code == 400


# ── Test 9: lambdas son positivos y razonables ──────────────────────────
def test_chat_lambdas_razonables(client):
    resp = _post_chat(client)
    body = resp.get_json()
    lh = body["lambdas"]["home"]
    la = body["lambdas"]["away"]
    assert 0.1 < lh < 8.0, f"lambda_home={lh} fuera de rango razonable"
    assert 0.1 < la < 8.0, f"lambda_away={la} fuera de rango razonable"


# ── Test 10: contexto_api siempre tiene las claves esperadas ────────────
def test_chat_ctx_api_estructura(client):
    resp = _post_chat(client)
    ctx = resp.get_json()["contexto_api"]
    for key in ("api_disponible", "forma_home", "forma_away", "h2h"):
        assert key in ctx, f"contexto_api falta clave '{key}'"


# ── Test 11: debug_filtrado siempre presente y bien formado (Fix B) ─────
def test_chat_debug_filtrado_estructura(client):
    resp = _post_chat(client)
    body = resp.get_json()
    dbg = body.get("debug_filtrado")
    assert dbg is not None, "debug_filtrado ausente en respuesta"
    for key in ("factor_datos", "umbral_usado", "datos_api_reales",
                "picks_totales", "picks_pasaron", "descartados"):
        assert key in dbg, f"debug_filtrado falta clave '{key}'"
    assert isinstance(dbg["descartados"], list)
    assert dbg["picks_pasaron"] == len(body["picks"])


# ── Tests /api/teams — dropdown de equipos ──────────────────────────────
def test_teams_liga_mx_equipos_presentes(client):
    r = client.get("/api/teams?liga=Liga%20MX")
    assert r.status_code == 200
    j = r.get_json()
    equipos = j["equipos"]
    assert "Cruz Azul" in equipos,   f"Cruz Azul no encontrado en Liga MX: {equipos}"
    assert "Tigres UANL" in equipos, f"Tigres UANL no encontrado en Liga MX: {equipos}"
    assert "Monterrey" in equipos,   f"Monterrey no encontrado en Liga MX: {equipos}"
    assert j["total"] == len(equipos)


def test_teams_liga_mx_placeholder_order(client):
    r = client.get("/api/teams?liga=Liga%20MX")
    equipos = r.get_json()["equipos"]
    assert len(equipos) >= 2, "Liga MX debe tener al menos 2 equipos para los placeholders"
    # el placeholder de home = equipos[0], away = equipos[1]
    assert equipos[0] == "Club America",  f"Primer equipo Liga MX esperado 'Club America', got '{equipos[0]}'"
    assert equipos[1] == "Cruz Azul",     f"Segundo equipo Liga MX esperado 'Cruz Azul', got '{equipos[1]}'"


def test_teams_filtro_mon_monterrey(client):
    r = client.get("/api/teams?liga=Liga%20MX")
    equipos = r.get_json()["equipos"]
    # Simula el filtro nativo del datalist: contiene "Mon"
    sugerencias = [e for e in equipos if "Mon" in e]
    assert any("Monterrey" in s for s in sugerencias), (
        f"Escribir 'Mon' debería sugerir Monterrey. Sugerencias: {sugerencias}"
    )


def test_teams_liga_inexistente_404(client):
    r = client.get("/api/teams?liga=Liga%20Inexistente")
    assert r.status_code == 404
    j = r.get_json()
    assert "error" in j
    assert j["equipos"] == []


def test_teams_sin_param_devuelve_todas_ligas(client):
    r = client.get("/api/teams")
    assert r.status_code == 200
    j = r.get_json()
    ligas = j["ligas"]
    assert "Liga MX" in ligas
    assert "LaLiga" in ligas
    assert "Premier League" in ligas
    assert j["total_ligas"] == len(ligas)
    assert "_meta" not in ligas


def test_teams_liga_mx_chat_round_trip(client):
    """Cruz Azul vs Tigres UANL en /chat no debería crashear."""
    payload = {
        "home": "Cruz Azul",
        "away": "Tigres UANL",
        "liga": "Liga MX",
        "cuotas": {
            "1X2":    {"1": 2.10, "X": 3.20, "2": 3.50},
            "OU_2.5": {"Over": 1.90, "Under": 1.95},
        }
    }
    r = client.post("/chat", data=json.dumps(payload),
                    content_type="application/json")
    assert r.status_code == 200
    body = r.get_json()
    assert body["partido"] == "Cruz Azul vs Tigres UANL"
    assert isinstance(body["picks"], list)
    assert body["lambdas"]["home"] > 0
    assert body["lambdas"]["away"] > 0


# ── Tests filtro OU_2.5 marginal ────────────────────────────────────────
def test_check_marginal_ou_descarta_sin_datos():
    """xg=2.66, margen=0.16 < 0.30, sin datos → descarte marginal_sin_datos."""
    import app as _app
    raw = {"legs": [{"mercado": "OU_2.5", "seleccion": "Under"}], "ev": 0.06}
    motivo = _app._check_marginal_ou(raw, xg_total=2.66, tiene_datos_reales=False)
    assert motivo is not None, "Debería descartar: margen 0.16 < 0.30 sin datos"
    assert "marginal_sin_datos" in motivo
    assert "2.66" in motivo
    assert "0.16" in motivo


def test_check_marginal_ou_pasa_margen_suficiente():
    """xg=3.10, margen=0.60 >= 0.30 → no descarta aunque no haya datos."""
    import app as _app
    raw = {"legs": [{"mercado": "OU_2.5", "seleccion": "Over"}], "ev": 0.06}
    motivo = _app._check_marginal_ou(raw, xg_total=3.10, tiene_datos_reales=False)
    assert motivo is None, f"No debería descartar con margen=0.60: {motivo}"


def test_check_marginal_ou_con_datos_ev_bajo():
    """xg=2.66, margen=0.16, CON datos pero EV=6% < 10% → descarte ev_insuficiente."""
    import app as _app
    raw = {"legs": [{"mercado": "OU_2.5", "seleccion": "Under"}], "ev": 0.06}
    motivo = _app._check_marginal_ou(raw, xg_total=2.66, tiene_datos_reales=True)
    assert motivo is not None, "Con datos y EV bajo debería descartar en zona marginal"
    assert "ev_insuficiente" in motivo
    assert "6.0%" in motivo or "6%" in motivo


def test_check_marginal_ou_con_datos_ev_suficiente():
    """xg=2.66, margen=0.16, CON datos y EV=12% >= 10% → pasa."""
    import app as _app
    raw = {"legs": [{"mercado": "OU_2.5", "seleccion": "Under"}], "ev": 0.12}
    motivo = _app._check_marginal_ou(raw, xg_total=2.66, tiene_datos_reales=True)
    assert motivo is None, f"Con datos y EV=12% debería pasar: {motivo}"


def test_check_marginal_ou_ignora_picks_sin_ou():
    """Pick 1X2 puro → filtro marginal no aplica."""
    import app as _app
    raw = {"legs": [{"mercado": "1X2", "seleccion": "1"}], "ev": 0.06}
    motivo = _app._check_marginal_ou(raw, xg_total=2.66, tiene_datos_reales=False)
    assert motivo is None, "Picks 1X2 no deben filtrarse por marginal_ou"


def test_chat_debug_filtrado_incluye_xg(client):
    """debug_filtrado siempre expone xg_total y margen_ou."""
    resp = _post_chat(client)
    dbg = resp.get_json()["debug_filtrado"]
    assert "xg_total" in dbg, "debug_filtrado debe incluir xg_total"
    assert "margen_ou" in dbg, "debug_filtrado debe incluir margen_ou"
    assert dbg["xg_total"] > 0
    assert dbg["margen_ou"] >= 0


def test_chat_marginal_ou_sin_datos_aparece_en_descartados(client):
    """
    Cuotas que produzcan xg~2.66 (cerca de 2.5) con API caída
    deben mostrar motivo marginal_ou en debug_filtrado.
    """
    # Cuotas 1X2 muy equilibradas → lambdas bajos (~1.3 + 1.3 = 2.6)
    payload = {
        "home": "Equipo_XYZ_Desconocido", "away": "Equipo_ABC_Desconocido",
        "liga": "Default",
        "cuotas": {
            "1X2":    {"1": 2.50, "X": 3.20, "2": 2.80},
            "OU_2.5": {"Over": 1.95, "Under": 1.90},  # cuotas muy parejas → cerca de 2.5
        }
    }
    with patch("src.api_football.contexto_partido_completo",
               side_effect=Exception("timeout")):
        resp = client.post("/chat", data=json.dumps(payload),
                           content_type="application/json")
    assert resp.status_code == 200
    dbg = resp.get_json()["debug_filtrado"]
    # Si el xg está en zona marginal, debe aparecer en descartados
    if dbg["margen_ou"] < 0.30:
        motivos = [d["motivo"] for d in dbg["descartados"]]
        assert "marginal_ou" in motivos, (
            f"xg_total={dbg['xg_total']}, margen={dbg['margen_ou']} < 0.30 "
            f"sin datos → esperado marginal_ou. Motivos: {motivos}"
        )


# ── Tests bloqueo de parlay uniforme (Mejora 3) ─────────────────────────
def test_dupla_uniforme_descartada():
    """[1X2 Local, 1X2 Local] → _es_combo_uniforme detecta el duplicado."""
    from src.engine import _es_combo_uniforme
    combo = [("1X2", "1"), ("1X2", "1")]
    es_unif, leg = _es_combo_uniforme(combo)
    assert es_unif, "Dupla con 2× 1X2 Local debe ser uniforme"
    assert "1X2" in leg and "1" in leg


def test_tripleta_uniforme_descartada():
    """[OU_2.5 Under × 3] → uniforme."""
    from src.engine import _es_combo_uniforme
    combo = [("OU_2.5", "Under"), ("OU_2.5", "Under"), ("OU_2.5", "Under")]
    es_unif, leg = _es_combo_uniforme(combo)
    assert es_unif, "Tripleta triple Under debe ser uniforme"
    assert "OU_2.5" in leg


def test_combo_mixto_pasa():
    """[1X2 Local, OU_2.5 Under, BTTS Yes] → no uniforme."""
    from src.engine import _es_combo_uniforme
    combo = [("1X2", "1"), ("OU_2.5", "Under"), ("BTTS", "Yes")]
    es_unif, _ = _es_combo_uniforme(combo)
    assert not es_unif, "Combo con mercados distintos NO debe ser uniforme"


def test_directa_no_afectada():
    """_es_combo_uniforme con un solo leg nunca devuelve True."""
    from src.engine import _es_combo_uniforme
    combo = [("OU_2.5", "Under")]
    es_unif, _ = _es_combo_uniforme(combo)
    assert not es_unif, "Un solo leg nunca puede ser uniforme"


def test_dupla_2_mercados_distintos_misma_seleccion_pasa():
    """[OU_2.5 Under, BTTS No] — distinto mercado, pasa aunque ambos sean 'negativos'."""
    from src.engine import _es_combo_uniforme
    combo = [("OU_2.5", "Under"), ("BTTS", "No")]
    es_unif, _ = _es_combo_uniforme(combo)
    assert not es_unif, "OU Under + BTTS No son mercados distintos: NO uniforme"


def test_chat_debug_filtrado_tiene_combos_descartados(client):
    """debug_filtrado siempre expone combos_descartados (lista, puede estar vacía)."""
    resp = _post_chat(client)
    dbg = resp.get_json()["debug_filtrado"]
    assert "combos_descartados" in dbg, "debug_filtrado debe tener 'combos_descartados'"
    assert isinstance(dbg["combos_descartados"], list)


# ── Test 12: umbral menor cuando API caída (Fix C) ──────────────────────
def test_chat_umbral_reducido_sin_api(client):
    from src.confidence import UMBRAL_CONFIANZA, UMBRAL_CONFIANZA_SIN_API
    ctx_vacio = {
        "api_disponible": False, "fuente": "thesportsdb",
        "home": "Real Madrid", "away": "Barcelona",
        "home_id": None, "away_id": None,
        "forma_home": None, "forma_away": None, "h2h": None,
        "injuries_home": [], "injuries_away": [], "notas": [],
    }
    with (
        patch("src.api_football.contexto_partido_completo",
              side_effect=Exception("timeout")),
        patch("src.sportsmonk.disponible", return_value=False),
        patch("src.thesportsdb.contexto_partido_completo", return_value=ctx_vacio),
    ):
        resp = _post_chat(client)
    dbg = resp.get_json()["debug_filtrado"]
    assert dbg["datos_api_reales"] is False
    assert abs(dbg["umbral_usado"] - UMBRAL_CONFIANZA_SIN_API) < 0.001, (
        f"Sin API, umbral debería ser {UMBRAL_CONFIANZA_SIN_API}, "
        f"got {dbg['umbral_usado']}"
    )
    assert dbg["umbral_usado"] < UMBRAL_CONFIANZA
