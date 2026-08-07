"""
Suite de pruebas del sistema BetBrain.
Ejecutar: python test_sistema.py
"""
import sys
import os
import json
from pathlib import Path

# Forzar UTF-8 en la salida del terminal (necesario en Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
results = []


def test(nombre, fn):
    msg = ""
    try:
        msg = fn() or "ok"
        estado = PASS
    except AssertionError as e:
        estado = FAIL
        msg = str(e)
    except Exception as e:
        estado = FAIL
        msg = f"{type(e).__name__}: {e}"
    # Imprimir fuera del try para no confundir errores de print con fallos de test
    print(f"  [{estado}] {nombre}: {msg}")
    results.append((nombre, estado, msg))


# ──────────────────────────────────────────────
# TEST 1: Carga del .env
# ──────────────────────────────────────────────
def _test_dotenv():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.getenv("API_FOOTBALL_KEY", "")
    host = os.getenv("API_FOOTBALL_HOST", "")
    assert host, ".env no tiene API_FOOTBALL_HOST"
    if not key:
        return "API_FOOTBALL_KEY vacío (introduce tu key)"
    return f"API_FOOTBALL_KEY presente ({key[:6]}...)"


# ──────────────────────────────────────────────
# TEST 2: config.py — PROMEDIO_GOLES_LIGA
# ──────────────────────────────────────────────
def _test_config_ligas():
    from config import PROMEDIO_GOLES_LIGA
    assert "LaLiga" in PROMEDIO_GOLES_LIGA, "Falta LaLiga"
    assert "Default" in PROMEDIO_GOLES_LIGA, "Falta Default"
    assert PROMEDIO_GOLES_LIGA["Bundesliga"] == 3.1, "Bundesliga debería ser 3.1"
    return f"{len(PROMEDIO_GOLES_LIGA)} ligas configuradas"


# ──────────────────────────────────────────────
# TEST 3: Solver numérico de lambdas (scipy)
# ──────────────────────────────────────────────
def _test_solver_lambdas():
    from probability_engine import (
        estimar_lambdas_desde_cuotas,
        eliminar_vig,
        generar_matriz_poisson,
        derivar_mercados,
    )
    cuotas_1x2 = {"1": 2.30, "X": 3.40, "2": 3.10}
    sv = eliminar_vig(cuotas_1x2)
    lh, la = estimar_lambdas_desde_cuotas(sv["1"], sv["2"], promedio_goles_liga=2.5)
    assert lh > 0, f"lambda_h debe ser positivo, es {lh}"
    assert la > 0, f"lambda_a debe ser positivo, es {la}"
    # Verificar que la matriz reproduce bien las probs originales
    mat = generar_matriz_poisson(lh, la)
    mercados = derivar_mercados(mat)
    p1_mat = mercados["1X2"]["1"]
    err = abs(p1_mat - sv["1"])
    assert err < 0.02, f"Error en P(local gana): {err:.4f} (esperado < 0.02)"
    return f"λ_h={lh:.3f}, λ_a={la:.3f}, error_p1={err:.4f}"


# ──────────────────────────────────────────────
# TEST 4: Validación de cuotas inválidas
# ──────────────────────────────────────────────
def _test_validacion_cuotas():
    import app as flask_app
    from app import _validar_cuotas_1x2

    # Cuota <= 1.0 debe rechazarse
    err = _validar_cuotas_1x2({"1": 0.9, "X": 3.40, "2": 3.10})
    assert err is not None, "Debería rechazar cuota <= 1.0"

    # Cuota 0 debe rechazarse
    err = _validar_cuotas_1x2({"1": 0.0, "X": 3.40, "2": 3.10})
    assert err is not None, "Debería rechazar cuota 0"

    # Clave faltante
    err = _validar_cuotas_1x2({"1": 2.30, "2": 3.10})
    assert err is not None, "Debería rechazar cuotas sin 'X'"

    # Cuotas válidas — no debe dar error
    err = _validar_cuotas_1x2({"1": 2.30, "X": 3.40, "2": 3.10})
    assert err is None, f"Cuotas válidas dieron error: {err}"

    return "4 casos de validación correctos"


# ──────────────────────────────────────────────
# TEST 5: Detección de contradicciones en parlay
# ──────────────────────────────────────────────
def _test_contradicciones():
    from src.confidence import verificar_contradicciones_combo

    # BTTS No + Over 2.5 → debe detectarse
    legs_contradict = [
        {"mercado": "BTTS", "seleccion": "No"},
        {"mercado": "OU_2.5", "seleccion": "Over"},
    ]
    msgs = verificar_contradicciones_combo(legs_contradict)
    assert len(msgs) > 0, "Debería detectar BTTS_No + Over_2.5 como contradicción"

    # Under 2.5 + BTTS Yes → debe detectarse
    legs_contradict2 = [
        {"mercado": "OU_2.5", "seleccion": "Under"},
        {"mercado": "BTTS", "seleccion": "Yes"},
    ]
    msgs2 = verificar_contradicciones_combo(legs_contradict2)
    assert len(msgs2) > 0, "Debería detectar Under_2.5 + BTTS_Yes como contradicción"

    # Local gana + Over 2.5 → NO contradicción
    legs_ok = [
        {"mercado": "1X2", "seleccion": "1"},
        {"mercado": "OU_2.5", "seleccion": "Over"},
    ]
    msgs_ok = verificar_contradicciones_combo(legs_ok)
    assert len(msgs_ok) == 0, f"No debería dar contradicción: {msgs_ok}"

    return "3 casos de contradicción correctos"


# ──────────────────────────────────────────────
# TEST 6: Sistema de confianza
# ──────────────────────────────────────────────
def _test_confianza():
    from src.confidence import calcular_confianza, nivel_confianza, UMBRAL_CONFIANZA

    # Pick de alta confianza
    c_alta = calcular_confianza(prob=0.65, ev=0.12, factor_datos=1.0)
    assert c_alta >= UMBRAL_CONFIANZA, f"Debería ser alta confianza: {c_alta}"
    assert nivel_confianza(c_alta) in ("alta", "media"), "Nivel incorrecto"

    # Pick de baja confianza (solo cuotas, EV mínimo)
    c_baja = calcular_confianza(prob=0.35, ev=0.05, factor_datos=0.6)
    assert c_baja < UMBRAL_CONFIANZA, f"Debería ser baja confianza: {c_baja}"
    assert nivel_confianza(c_baja) == "baja", "Debería ser 'baja'"

    return f"alta={c_alta:.3f}, baja={c_baja:.3f}, umbral={UMBRAL_CONFIANZA}"


# ──────────────────────────────────────────────
# TEST 7: Generación de pick completo (sin API)
# ──────────────────────────────────────────────
def _test_pick_completo():
    from src.engine import BettingEngine

    eng = BettingEngine()
    res = eng.pick_multileg(
        home="Real Madrid", away="Barcelona",
        cuotas={
            "1X2":    {"1": 2.30, "X": 3.40, "2": 3.10},
            "OU_2.5": {"Over": 1.75, "Under": 2.10},
            "BTTS":   {"Yes": 1.70, "No": 2.10},
        },
        cuota_min=1.70, cuota_max=2.50,
        promedio_goles_liga=2.5,
    )
    assert "lambdas" in res, "Falta 'lambdas' en el resultado"
    assert "prob_1x2_final" in res, "Falta 'prob_1x2_final'"
    lh = res["lambdas"]["home"]
    la = res["lambdas"]["away"]
    assert lh > 0 and la > 0, "Lambdas deben ser positivos"
    picks_found = sum(1 for k in ("directa", "dupla", "tripleta") if res.get(k))
    return f"λ=({lh},{la}), picks encontrados={picks_found}"


# ──────────────────────────────────────────────
# TEST 8: Conexión a API-Football
# ──────────────────────────────────────────────
def _test_api_football():
    from config import API_FOOTBALL_KEY
    if not API_FOOTBALL_KEY:
        raise AssertionError("API_FOOTBALL_KEY no configurada — skip")

    from src.api_football import search_team
    team_id = search_team("Barcelona")
    if team_id is None:
        raise AssertionError("API respondió pero no encontró 'Barcelona' — verificar key")
    return f"team_id de Barcelona = {team_id}"


# ──────────────────────────────────────────────
# TEST 9: registrar_apuesta guarda en CSV
# ──────────────────────────────────────────────
def _test_registrar_apuesta():
    import tempfile, csv
    from src.tracking import registrar_apuesta, leer_historial, COLUMNS

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                     newline="", encoding="utf-8") as tf:
        tmp = tf.name
        csv.DictWriter(tf, fieldnames=COLUMNS).writeheader()

    from pathlib import Path
    p = Path(tmp)
    data = {
        "liga": "LaLiga", "local": "Real Madrid", "visitante": "Barcelona",
        "pick_tipo": "DIRECTA", "pick_descripcion": "Gana Real Madrid",
        "cuota": 2.1, "stake": 20.0,
        "prob_predicha": 0.55, "ev_predicho": 0.155,
        "confianza_score": 0.78, "confianza_badge": "alta",
    }
    res = registrar_apuesta(data, csv_path=p)
    assert res["id"] == 1, f"id esperado 1, obtenido {res['id']}"
    rows = leer_historial(csv_path=p)
    assert len(rows) == 1, f"Debería haber 1 apuesta, hay {len(rows)}"
    assert rows[0]["resultado"] == "pendiente", "resultado inicial debe ser 'pendiente'"
    p.unlink()
    return f"Apuesta #{res['id']} registrada — {res['mensaje']}"


# ──────────────────────────────────────────────
# TEST 10: actualizar_resultado recalcula bankroll
# ──────────────────────────────────────────────
def _test_actualizar_resultado():
    import tempfile, csv
    from src.tracking import (registrar_apuesta, actualizar_resultado,
                               leer_historial, leer_config, guardar_config, COLUMNS)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                     newline="", encoding="utf-8") as tf:
        tmp = tf.name
        csv.DictWriter(tf, fieldnames=COLUMNS).writeheader()

    from pathlib import Path
    p = Path(tmp)
    cfg_backup = leer_config()
    try:
        registrar_apuesta({
            "liga": "Premier League", "local": "Arsenal", "visitante": "Chelsea",
            "pick_tipo": "DIRECTA", "pick_descripcion": "Gana Arsenal",
            "cuota": 2.0, "stake": 10.0, "prob_predicha": 0.6,
            "ev_predicho": 0.20, "confianza_score": 0.80, "confianza_badge": "alta",
        }, csv_path=p)
        res = actualizar_resultado(1, "ganada", csv_path=p)
    finally:
        guardar_config(cfg_backup)
        p.unlink()

    assert res["resultado"] == "ganada", "resultado debe ser 'ganada'"
    assert res["ganancia_neta"] == 10.0, f"ganancia_neta esperada 10.0, obtenida {res['ganancia_neta']}"
    assert res["bankroll_despues"] == res["bankroll_antes"] + 10.0, "bankroll incorrecto"
    return f"bankroll {res['bankroll_antes']} -> {res['bankroll_despues']} (+{res['ganancia_neta']})"


# ──────────────────────────────────────────────
# TEST 11: leer_historial filtra correctamente
# ──────────────────────────────────────────────
def _test_leer_historial():
    import tempfile, csv
    from src.tracking import registrar_apuesta, leer_historial, COLUMNS

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                     newline="", encoding="utf-8") as tf:
        tmp = tf.name
        csv.DictWriter(tf, fieldnames=COLUMNS).writeheader()

    from pathlib import Path
    p = Path(tmp)

    for i in range(3):
        registrar_apuesta({
            "liga": "Bundesliga", "local": f"Equipo{i}", "visitante": "Rival",
            "pick_tipo": "DIRECTA", "pick_descripcion": f"Gana Equipo{i}",
            "cuota": 1.9, "stake": 5.0, "prob_predicha": 0.5,
            "ev_predicho": 0.05, "confianza_score": 0.70, "confianza_badge": "media",
        }, csv_path=p)

    rows = leer_historial(csv_path=p)
    assert len(rows) == 3, f"Deberían ser 3 apuestas, hay {len(rows)}"
    assert all(r["resultado"] == "pendiente" for r in rows), "Todas deben ser pendientes"
    p.unlink()
    return f"{len(rows)} apuestas leídas correctamente"


# ──────────────────────────────────────────────
# TEST 12: calcular_metricas retorna estructura completa
# ──────────────────────────────────────────────
def _test_calcular_metricas():
    import tempfile, csv
    from src.tracking import (registrar_apuesta, actualizar_resultado,
                               calcular_metricas, leer_config, guardar_config, COLUMNS)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                     newline="", encoding="utf-8") as tf:
        tmp = tf.name
        csv.DictWriter(tf, fieldnames=COLUMNS).writeheader()

    from pathlib import Path
    p = Path(tmp)
    cfg_backup = leer_config()
    try:
        registrar_apuesta({
            "liga": "Serie A", "local": "Juventus", "visitante": "Milan",
            "pick_tipo": "DIRECTA", "pick_descripcion": "Gana Juventus",
            "cuota": 2.0, "stake": 10.0, "prob_predicha": 0.55,
            "ev_predicho": 0.10, "confianza_score": 0.75, "confianza_badge": "alta",
        }, csv_path=p)
        registrar_apuesta({
            "liga": "Serie A", "local": "Napoli", "visitante": "Roma",
            "pick_tipo": "DIRECTA", "pick_descripcion": "Gana Napoli",
            "cuota": 1.8, "stake": 10.0, "prob_predicha": 0.60,
            "ev_predicho": 0.08, "confianza_score": 0.72, "confianza_badge": "alta",
        }, csv_path=p)
        actualizar_resultado(1, "ganada",  csv_path=p)
        actualizar_resultado(2, "perdida", csv_path=p)
        m = calcular_metricas(csv_path=p)
    finally:
        guardar_config(cfg_backup)
        p.unlink()

    assert m["total"] == 2, f"total esperado 2, obtenido {m['total']}"
    assert m["ganadas"] == 1 and m["perdidas"] == 1, "ganadas/perdidas incorrectas"
    assert m["tasa_acierto"] == 50.0, f"tasa esperada 50.0, obtenida {m['tasa_acierto']}"
    assert m["ganancia_total"] == 0.0, f"ganancia_total esperada 0.0, obtenida {m['ganancia_total']}"
    assert "grafica_bankroll" in m, "Falta grafica_bankroll"
    assert m["roi"] == 0.0, f"ROI esperado 0.0, obtenido {m['roi']}"
    return f"total={m['total']}, tasa={m['tasa_acierto']}%, roi={m['roi']}%"


# ──────────────────────────────────────────────
# Ejecutar todos los tests
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  BetBrain — Suite de pruebas del sistema")
    print("=" * 55)

    test("Carga .env",            _test_dotenv)
    test("Config ligas",          _test_config_ligas)
    test("Solver de lambdas",     _test_solver_lambdas)
    test("Validación cuotas",     _test_validacion_cuotas)
    test("Contradicciones parlay",_test_contradicciones)
    test("Sistema de confianza",  _test_confianza)
    test("Pick completo (sin API)",_test_pick_completo)
    test("Conexión API-Football", _test_api_football)
    test("Tracking: registrar_apuesta",    _test_registrar_apuesta)
    test("Tracking: actualizar_resultado", _test_actualizar_resultado)
    test("Tracking: leer_historial",       _test_leer_historial)
    test("Tracking: calcular_metricas",    _test_calcular_metricas)

    print("\n" + "=" * 55)
    total  = len(results)
    pasado = sum(1 for _, e, _ in results if e == PASS)
    fallado = sum(1 for _, e, _ in results if e == FAIL)
    print(f"  Resultado: {pasado}/{total} tests pasaron "
          f"({'OK' if fallado == 0 else f'{fallado} FALLADOS'})")
    print("=" * 55 + "\n")
    sys.exit(0 if fallado == 0 else 1)
