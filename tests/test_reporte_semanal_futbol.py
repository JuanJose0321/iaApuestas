"""
Tests del reporte semanal automático de fútbol (src/scripts/reporte_semanal_futbol.py).

Mismo patrón que tests/test_reporte_semanal_tenis.py -- todo se testea
contra `generar_reporte_md` (función pura, sin Supabase) para no depender
de red. Cubre además lo específico de fútbol: filtro por liga (no por
liga="Tenis" fija), desglose por pick_tipo, y la comparación modelo vs.
probabilidad implícita de la cuota.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scripts.reporte_semanal_futbol import (
    accuracy,
    chequear_calidad,
    comparacion_mercado,
    detectar_hitos_muestra,
    generar_reporte_md,
    obtener_datos_futbol,
    resumen_badges,
    resumen_ev,
    resumen_por_liga,
    resumen_por_tipo,
    resumen_umbrales_prob,
)


def _fila(**overrides):
    base = {
        "id": 1, "fecha_registro": "25/08/2026 17:49", "liga": "LaLiga",
        "local": "Barcelona", "visitante": "Sevilla",
        "pick_tipo": "directa", "pick_descripcion": "1X2: Barcelona",
        "cuota": 1.8, "prob_predicha": 0.62, "ev_predicho": 0.05,
        "confianza_score": 0.68, "confianza_badge": "amarillo",
        "resultado": "ganada",
    }
    base.update(overrides)
    return base


# ── Filtro fútbol vs. tenis (comparten la misma tabla apuestas) ────────

def test_obtener_datos_futbol_excluye_tenis_y_default(monkeypatch):
    filas = [
        _fila(id=1, liga="LaLiga"),
        _fila(id=2, liga="Tenis"),
        _fila(id=3, liga="Default"),
        _fila(id=4, liga=""),
        _fila(id=5, liga="Champions League"),
    ]
    import src.scripts.reporte_semanal_futbol as mod
    monkeypatch.setattr(mod._sb, "leer_apuestas", lambda: filas)
    resultado = obtener_datos_futbol()
    assert {f["id"] for f in resultado} == {1, 5}


# ── Cero apuestas resueltas ─────────────────────────────────────────────

def test_cero_resueltas_no_rompe_nada():
    todas = [_fila(id=1, resultado="pendiente"), _fila(id=2, resultado="pendiente")]
    md, estado = generar_reporte_md(todas, estado_anterior=None, hoy=date(2026, 9, 1))
    assert "**0**" in md
    assert estado["resueltas_totales"] == 0
    n, aciertos, pct = accuracy([])
    assert (n, aciertos, pct) == (0, 0, 0.0)


def test_lista_vacia_de_apuestas():
    md, estado = generar_reporte_md([], estado_anterior=None, hoy=date(2026, 9, 1))
    assert "Reporte semanal de fútbol" in md
    assert estado["resueltas_totales"] == 0


# ── Desglose por liga y por tipo de pick ────────────────────────────────

def test_resumen_por_liga():
    filas = [
        _fila(id=1, liga="LaLiga", resultado="ganada"),
        _fila(id=2, liga="LaLiga", resultado="perdida"),
        _fila(id=3, liga="Premier League", resultado="ganada"),
    ]
    resumen = resumen_por_liga(filas)
    assert resumen["LaLiga"] == (2, 1, 50.0)
    assert resumen["Premier League"] == (1, 1, 100.0)


def test_resumen_por_tipo():
    filas = [
        _fila(id=1, pick_tipo="directa", resultado="ganada"),
        _fila(id=2, pick_tipo="dupla", resultado="perdida"),
        _fila(id=3, pick_tipo="dupla", resultado="perdida"),
    ]
    resumen = resumen_por_tipo(filas)
    assert resumen["directa"] == (1, 1, 100.0)
    assert resumen["dupla"] == (2, 0, 0.0)


# ── Hitos de muestra (mismo mecanismo que tenis, sin rango de alerta) ──

def test_hito_de_muestra_se_detecta_al_cruzar_el_umbral():
    filas = [_fila(id=i, confianza_badge="verde", resultado="ganada") for i in range(22)]
    resumen = resumen_badges(filas)
    estado_anterior = {"badge_n": {"verde": 10}}
    hitos = detectar_hitos_muestra(resumen, estado_anterior)
    assert len(hitos) == 1
    assert "verde" in hitos[0]


def test_sin_estado_anterior_no_reporta_hitos_falsos():
    filas = [_fila(id=i, confianza_badge="manual", resultado="ganada") for i in range(50)]
    resumen = resumen_badges(filas)
    assert detectar_hitos_muestra(resumen, None) == []


# ── EV / umbrales de probabilidad no rompen con datos faltantes ────────

def test_prob_o_ev_none_no_rompe():
    filas = [_fila(id=1, prob_predicha=None, ev_predicho=None, resultado="ganada")]
    umbrales = resumen_umbrales_prob(filas)
    ev = resumen_ev(filas)
    assert umbrales["todos"][0] == 1
    assert ev["ev_positivo"][0] + ev["ev_negativo_o_cero"][0] == 1


# ── Comparación modelo vs. probabilidad implícita de la cuota ──────────

def test_comparacion_mercado_sin_datos_no_rompe():
    assert comparacion_mercado([])["n"] == 0
    filas = [_fila(id=1, cuota=None), _fila(id=2, prob_predicha=None), _fila(id=3, cuota=0)]
    assert comparacion_mercado(filas)["n"] == 0


def test_comparacion_mercado_calcula_brier_y_accuracy():
    # cuota=2.0 -> implicita=0.5; modelo dice 0.8 y gana -- modelo "mas
    # confiado" que la cuota implicita y tenia razon.
    filas = [_fila(id=1, cuota=2.0, prob_predicha=0.8, resultado="ganada")]
    r = comparacion_mercado(filas)
    assert r["n"] == 1
    assert r["acc_modelo"] == 100.0
    assert r["acc_mercado_implicito"] == 100.0  # 0.5 >= 0.5 y gano
    assert r["acc_cuando_modelo_mas_confiado"][0] == 1  # 0.8 > 0.5


def test_comparacion_mercado_distingue_mas_y_menos_confiado():
    filas = [
        _fila(id=1, cuota=2.0, prob_predicha=0.9, resultado="ganada"),   # modelo mas confiado (0.9>0.5), gano
        _fila(id=2, cuota=1.2, prob_predicha=0.3, resultado="perdida"),  # modelo menos confiado (0.3<0.833), perdio
    ]
    r = comparacion_mercado(filas)
    assert r["n"] == 2
    assert r["acc_cuando_modelo_mas_confiado"][0] == 1
    assert r["acc_cuando_modelo_menos_confiado_o_igual"][0] == 1


# ── Chequeos de calidad ─────────────────────────────────────────────────

def test_duplicados_exactos_se_detectan():
    fila = _fila(id=1)
    fila_dup = _fila(id=2)
    otra = _fila(id=3, local="Betis", visitante="Alaves", pick_descripcion="1X2: Betis")
    calidad = chequear_calidad([fila, fila_dup, otra])
    assert len(calidad["duplicados"]) == 1
    assert calidad["duplicados"][0]["copias"] == 2


def test_campos_vacios_en_resueltas_se_detectan():
    fila = _fila(id=1, cuota=None, resultado="ganada")
    fila_pendiente = _fila(id=2, cuota=None, resultado="pendiente")
    calidad = chequear_calidad([fila, fila_pendiente])
    assert calidad["campos_vacios"]["cuota"] == [1]


def test_sin_problemas_de_calidad_no_reporta_nada():
    filas = [_fila(id=1), _fila(id=2, local="X", visitante="Y", pick_descripcion="1X2: X")]
    calidad = chequear_calidad(filas)
    assert calidad["duplicados"] == []
    assert calidad["campos_vacios"] == {}


# ── Reporte completo con datos mezclados no revienta ────────────────────

def test_reporte_completo_con_datos_mezclados():
    todas = (
        [_fila(id=i, liga="LaLiga", pick_tipo="directa", resultado="ganada", confianza_badge="verde") for i in range(10)]
        + [_fila(id=i + 10, liga="Premier League", pick_tipo="dupla", resultado="perdida", confianza_badge="amarillo") for i in range(5)]
        + [_fila(id=i + 20, resultado="pendiente") for i in range(3)]
    )
    md, estado = generar_reporte_md(todas, estado_anterior={"resueltas_totales": 12, "badge_n": {}}, hoy=date(2026, 9, 1))
    assert "Reporte semanal de fútbol" in md
    assert estado["resueltas_totales"] == 15
    assert "Nuevas resueltas desde el último reporte" in md
    assert "LaLiga" in md and "Premier League" in md
