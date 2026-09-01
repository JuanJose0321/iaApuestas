"""
Tests del reporte semanal automático de tenis (src/scripts/reporte_semanal_tenis.py).

Cubre casos "raros" a propósito -- son justo los que rompen scripts de
reporte en producción: cero filas resueltas, badges con muestra
insuficiente, datos con duplicados/incoherencias. Todo se testea contra
`generar_reporte_md` (función pura, sin Supabase) para no depender de red.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scripts.reporte_semanal_tenis import (
    accuracy,
    chequear_calidad,
    detectar_hitos_muestra,
    evaluar_alertas_badges,
    generar_reporte_md,
    resumen_badges,
    resumen_ev,
    resumen_umbrales_prob,
)


def _fila(**overrides):
    base = {
        "id": 1, "fecha_registro": "25/08/2026 17:49", "liga": "Tenis",
        "local": "Jugador A", "visitante": "Jugador B",
        "pick_descripcion": "Jugador A gana", "cuota": 1.8,
        "prob_predicha": 0.62, "ev_predicho": 0.05, "confianza_score": 0.3,
        "confianza_badge": "amarillo", "resultado": "ganada",
    }
    base.update(overrides)
    return base


# ── Cero apuestas resueltas ─────────────────────────────────────────────

def test_cero_resueltas_no_rompe_nada():
    todas = [_fila(id=1, resultado="pendiente"), _fila(id=2, resultado="pendiente")]
    md, estado = generar_reporte_md(todas, estado_anterior=None, hoy=date(2026, 9, 1))
    assert "0/0" in md or "**0**" in md
    assert estado["resueltas_totales"] == 0
    # no debe reventar ninguna división por cero en los cortes
    n, aciertos, pct = accuracy([])
    assert (n, aciertos, pct) == (0, 0, 0.0)


def test_lista_vacia_de_apuestas():
    md, estado = generar_reporte_md([], estado_anterior=None, hoy=date(2026, 9, 1))
    assert "Reporte semanal de tenis" in md
    assert estado["resueltas_totales"] == 0


# ── Badges con muestra insuficiente ─────────────────────────────────────

def test_badge_con_poca_muestra_no_genera_alerta_aunque_este_lejos_del_rango():
    # 5 "verde" con 0% de acierto -- lejísimos del ~71%+ esperado, pero
    # n=5 < MIN_MUESTRA_ALERTA=20, no debe alertar.
    filas = [_fila(id=i, confianza_badge="verde", resultado="perdida") for i in range(5)]
    resumen = resumen_badges(filas)
    alertas = evaluar_alertas_badges(resumen)
    assert alertas == []


def test_badge_con_muestra_suficiente_y_fuera_de_rango_si_alerta():
    filas = [_fila(id=i, confianza_badge="verde", resultado="perdida") for i in range(25)]
    resumen = resumen_badges(filas)
    alertas = evaluar_alertas_badges(resumen)
    assert len(alertas) == 1
    assert "verde" in alertas[0]


def test_badge_dentro_de_rango_no_alerta_con_muestra_grande():
    filas = [_fila(id=i, confianza_badge="amarillo", resultado="ganada" if i % 3 else "perdida") for i in range(30)]
    resumen = resumen_badges(filas)
    alertas = evaluar_alertas_badges(resumen)
    assert alertas == []


def test_hito_de_muestra_se_detecta_al_cruzar_el_umbral():
    filas = [_fila(id=i, confianza_badge="verde", resultado="ganada") for i in range(22)]
    resumen = resumen_badges(filas)
    estado_anterior = {"badge_n": {"verde": 10}}
    hitos = detectar_hitos_muestra(resumen, estado_anterior)
    assert len(hitos) == 1
    assert "verde" in hitos[0]

    # si ya venía con >=20 antes, no es un hito nuevo
    estado_anterior_2 = {"badge_n": {"verde": 21}}
    assert detectar_hitos_muestra(resumen, estado_anterior_2) == []


def test_sin_estado_anterior_primer_reporte_no_reporta_hitos_falsos():
    # Primer reporte de la serie (estado_anterior=None): un badge que ya
    # arranca con n>=20 no debe leerse como "cruzó el umbral esta semana".
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


# ── Chequeos de calidad ─────────────────────────────────────────────────

def test_duplicados_exactos_se_detectan():
    fila = _fila(id=1)
    fila_dup = _fila(id=2)
    otra = _fila(id=3, local="Jugador C", visitante="Jugador D", pick_descripcion="Jugador C gana")
    calidad = chequear_calidad([fila, fila_dup, otra])
    assert len(calidad["duplicados"]) == 1
    assert calidad["duplicados"][0]["copias"] == 2


def test_incoherencia_local_visitante_se_detecta():
    fila = _fila(id=1, pick_descripcion="Jugador Z gana")  # Z no es local ni visitante
    calidad = chequear_calidad([fila])
    assert len(calidad["incoherencias"]) == 1
    assert calidad["incoherencias"][0]["id"] == 1


def test_campos_vacios_en_resueltas_se_detectan():
    fila = _fila(id=1, cuota=None, resultado="ganada")
    fila_pendiente = _fila(id=2, cuota=None, resultado="pendiente")
    calidad = chequear_calidad([fila, fila_pendiente])
    # solo la resuelta (id=1) debe aparecer -- pendientes no cuentan
    assert calidad["campos_vacios"]["cuota"] == [1]


def test_sin_problemas_de_calidad_no_reporta_nada():
    filas = [_fila(id=1), _fila(id=2, local="X", visitante="Y", pick_descripcion="X gana")]
    calidad = chequear_calidad(filas)
    assert calidad["duplicados"] == []
    assert calidad["incoherencias"] == []
    assert calidad["campos_vacios"] == {}


# ── Reporte completo con datos "reales" mezclados no revienta ──────────

def test_reporte_completo_con_datos_mezclados():
    todas = (
        [_fila(id=i, resultado="ganada", confianza_badge="manual") for i in range(10)]
        + [_fila(id=i + 10, resultado="perdida", confianza_badge="amarillo", prob_predicha=0.61) for i in range(5)]
        + [_fila(id=i + 20, resultado="pendiente") for i in range(3)]
    )
    md, estado = generar_reporte_md(todas, estado_anterior={"resueltas_totales": 12, "badge_n": {}}, hoy=date(2026, 9, 1))
    assert "Reporte semanal de tenis" in md
    assert estado["resueltas_totales"] == 15
    assert "Nuevas resueltas desde el último reporte" in md
