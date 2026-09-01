"""
Reporte semanal automático de accuracy/patrones de tenis.

Automatiza el análisis que se venía pidiendo a mano cada vez (accuracy
global, por badge verde/amarillo/manual, por umbral de probabilidad, chequeo
del EV, comparación contra `tennis_validacion_filtro_ev.md`, chequeos de
calidad de datos). Solo LEE de Supabase (`apuestas`) y escribe un archivo de
reporte en `reportes/` -- no toca `tennis_improved.py` ni ninguna otra ruta
de producción, y no decide ni activa nada por su cuenta. La decisión de
actuar sobre lo que el reporte muestra sigue siendo manual, como hasta
ahora.

Uso
---
    python src/scripts/reporte_semanal_tenis.py

Guarda `reportes/reporte_tenis_YYYY-MM-DD.md` (no pisa reportes anteriores)
y actualiza `reportes/estado.json` (bookkeeping interno: cuántas apuestas
resueltas y de qué tamaño era cada badge en el último reporte, para poder
calcular "nuevas desde la última vez" y avisar cuando una muestra cruza el
umbral de 20 casos).
"""
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.services import supabase_client as _sb

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REPORTES_DIR = ROOT_DIR / "reportes"
ESTADO_PATH = REPORTES_DIR / "estado.json"

# Rangos validados en tennis_validacion_filtro_ev.md (Paso 4, contra los
# picks reales de producción) -- referencia para las alertas de badge, no
# un umbral que decida nada del motor.
RANGO_ESPERADO = {
    "verde": (71.0, 100.0),      # "~71%+"
    "amarillo": (60.0, 67.0),
}
DESVIO_ALERTA_PP = 10.0   # puntos porcentuales de tolerancia antes de avisar
MIN_MUESTRA_ALERTA = 20   # no alertar con menos de esta cantidad de resueltas
UMBRALES_PROB = (0.55, 0.60, 0.65)
DIAS_TENDENCIA_RECIENTE = 14


# ──────────────────────────────────────────────
# Utilidades de fecha / accuracy (funciones puras, testeables sin Supabase)
# ──────────────────────────────────────────────

def _parse_fecha_registro(valor: str) -> Optional[datetime]:
    """`fecha_registro` viene como "dd/mm/YYYY HH:MM". None si viene vacío
    o con un formato inesperado -- una fila así no debe tumbar el reporte,
    solo queda fuera de los cortes que dependen de fecha."""
    if not valor:
        return None
    try:
        return datetime.strptime(valor, "%d/%m/%Y %H:%M")
    except ValueError:
        return None


def filas_resueltas(filas: list[dict]) -> list[dict]:
    return [f for f in filas if f.get("resultado") in ("ganada", "perdida")]


def accuracy(filas: list[dict]) -> tuple[int, int, float]:
    """(n, aciertos, % acierto). 0.0% si n == 0 (no 1/0)."""
    n = len(filas)
    if n == 0:
        return 0, 0, 0.0
    aciertos = sum(1 for f in filas if f.get("resultado") == "ganada")
    return n, aciertos, aciertos / n * 100


# ──────────────────────────────────────────────
# Paso 2 -- accuracy por badge
# ──────────────────────────────────────────────

def resumen_badges(resueltas: list[dict]) -> dict[str, tuple[int, int, float]]:
    por_badge: dict[str, list[dict]] = {}
    for f in resueltas:
        badge = f.get("confianza_badge") or "sin_badge"
        por_badge.setdefault(badge, []).append(f)
    return {badge: accuracy(lst) for badge, lst in por_badge.items()}


def evaluar_alertas_badges(resumen: dict[str, tuple[int, int, float]]) -> list[str]:
    """Alertas simples: badge con muestra >= MIN_MUESTRA_ALERTA cuya accuracy
    cae más de DESVIO_ALERTA_PP puntos fuera del rango validado."""
    alertas = []
    for badge, (rango_min, rango_max) in RANGO_ESPERADO.items():
        n, aciertos, pct = resumen.get(badge, (0, 0, 0.0))
        if n < MIN_MUESTRA_ALERTA:
            continue
        if pct < rango_min - DESVIO_ALERTA_PP:
            alertas.append(
                f"`{badge}` (n={n}) dio {pct:.2f}% -- más de {DESVIO_ALERTA_PP:.0f}pp "
                f"por DEBAJO del rango esperado ({rango_min:.0f}-{rango_max:.0f}%)."
            )
        elif pct > rango_max + DESVIO_ALERTA_PP:
            alertas.append(
                f"`{badge}` (n={n}) dio {pct:.2f}% -- más de {DESVIO_ALERTA_PP:.0f}pp "
                f"por ENCIMA del rango esperado ({rango_min:.0f}-{rango_max:.0f}%)."
            )
    return alertas


def detectar_hitos_muestra(
    resumen: dict[str, tuple[int, int, float]], estado_anterior: Optional[dict]
) -> list[str]:
    """Avisa cuando un badge cruza el umbral de MIN_MUESTRA_ALERTA desde el
    último reporte -- es el punto en el que una cifra que antes era "n
    insuficiente, no concluyente" ya empieza a poder leerse en serio.

    Sin `estado_anterior` (primer reporte de la serie) no hay "antes" contra
    el cual comparar -- reportar hitos ahí sería un falso positivo para
    cualquier badge que ya arranque con n>=20, así que se omite."""
    if estado_anterior is None:
        return []
    hitos = []
    badge_n_anterior = estado_anterior.get("badge_n", {})
    for badge, (n, aciertos, pct) in resumen.items():
        n_antes = badge_n_anterior.get(badge, 0)
        if n >= MIN_MUESTRA_ALERTA and n_antes < MIN_MUESTRA_ALERTA:
            rango = RANGO_ESPERADO.get(badge)
            ref = f" (esperado: {rango[0]:.0f}-{rango[1]:.0f}%)" if rango else ""
            hitos.append(
                f"`{badge}` ya junta {n} resueltas (antes tenía {n_antes}, por debajo "
                f"de {MIN_MUESTRA_ALERTA}) -- accuracy con muestra ya razonable: "
                f"{pct:.2f}%{ref}."
            )
    return hitos


# ──────────────────────────────────────────────
# Paso 3 -- accuracy por umbral de probabilidad
# ──────────────────────────────────────────────

def resumen_umbrales_prob(resueltas: list[dict]) -> dict[str, tuple[int, int, float]]:
    resumen = {"todos": accuracy(resueltas)}
    for umbral in UMBRALES_PROB:
        filtradas = [f for f in resueltas if (f.get("prob_predicha") or 0) >= umbral]
        resumen[f"prob>={umbral:.0%}"] = accuracy(filtradas)
    return resumen


# ──────────────────────────────────────────────
# Paso 4 -- chequeo del EV
# ──────────────────────────────────────────────

def resumen_ev(resueltas: list[dict]) -> dict[str, tuple[int, int, float]]:
    positivo = [f for f in resueltas if (f.get("ev_predicho") or 0) > 0]
    neg_o_cero = [f for f in resueltas if (f.get("ev_predicho") or 0) <= 0]
    return {"ev_positivo": accuracy(positivo), "ev_negativo_o_cero": accuracy(neg_o_cero)}


# ──────────────────────────────────────────────
# Paso 5 -- tendencia reciente vs histórico
# ──────────────────────────────────────────────

def tendencia_reciente(
    resueltas: list[dict], hoy: Optional[date] = None, dias: int = DIAS_TENDENCIA_RECIENTE
) -> dict:
    hoy = hoy or date.today()
    corte = hoy - timedelta(days=dias)
    recientes = []
    for f in resueltas:
        fr = _parse_fecha_registro(f.get("fecha_registro", ""))
        if fr is not None and fr.date() >= corte:
            recientes.append(f)
    return {
        "historico": accuracy(resueltas),
        "reciente": accuracy(recientes),
        "dias": dias,
    }


# ──────────────────────────────────────────────
# Paso 6 -- chequeos de calidad de datos
# ──────────────────────────────────────────────

def chequear_calidad(todas: list[dict]) -> dict:
    from collections import Counter

    clave = lambda f: (
        f.get("local"), f.get("visitante"), f.get("pick_descripcion"),
        f.get("cuota"), f.get("prob_predicha"), f.get("ev_predicho"),
    )
    conteo = Counter(clave(f) for f in todas)
    duplicados = [
        {"local": k[0], "visitante": k[1], "pick_descripcion": k[2], "copias": v}
        for k, v in conteo.items() if v > 1
    ]

    incoherencias = []
    for f in todas:
        desc = f.get("pick_descripcion") or ""
        if desc.endswith(" gana"):
            jugador = desc[: -len(" gana")].strip()
            if jugador and jugador not in (f.get("local"), f.get("visitante")):
                incoherencias.append({
                    "id": f.get("id"), "pick_descripcion": desc,
                    "local": f.get("local"), "visitante": f.get("visitante"),
                })

    campos_obligatorios = ["cuota", "prob_predicha", "ev_predicho", "confianza_score"]
    resueltas = filas_resueltas(todas)
    campos_vacios: dict[str, list] = {c: [] for c in campos_obligatorios}
    for f in resueltas:
        for campo in campos_obligatorios:
            if f.get(campo) is None or f.get(campo) == "":
                campos_vacios[campo].append(f.get("id"))
    campos_vacios = {c: ids for c, ids in campos_vacios.items() if ids}

    return {
        "duplicados": duplicados,
        "incoherencias": incoherencias,
        "campos_vacios": campos_vacios,
    }


# ──────────────────────────────────────────────
# Estado entre corridas (para "nuevas desde el último reporte" e hitos de muestra)
# ──────────────────────────────────────────────

def cargar_estado(path: Path = ESTADO_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def guardar_estado(estado: dict, path: Path = ESTADO_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")


# ──────────────────────────────────────────────
# Composición del reporte
# ──────────────────────────────────────────────

def _fmt_tabla(filas: list[tuple]) -> str:
    lineas = ["| Categoría | n | Aciertos | Accuracy |", "|---|---|---|---|"]
    for nombre, (n, aciertos, pct) in filas:
        lineas.append(f"| {nombre} | {n} | {aciertos} | {pct:.2f}% |")
    return "\n".join(lineas)


def generar_reporte_md(
    todas: list[dict],
    estado_anterior: Optional[dict],
    hoy: Optional[date] = None,
) -> tuple[str, dict]:
    """Devuelve (markdown, estado_nuevo). No toca disco ni Supabase --
    función pura para poder testearla con datos sintéticos."""
    hoy = hoy or date.today()
    resueltas = filas_resueltas(todas)
    n_total, aciertos_total, pct_total = accuracy(resueltas)

    resueltas_antes = (estado_anterior or {}).get("resueltas_totales")
    if resueltas_antes is None:
        linea_nuevas = "N/A (primer reporte, no hay corrida anterior para comparar)"
    else:
        nuevas = n_total - resueltas_antes
        linea_nuevas = f"{nuevas} (de {resueltas_antes} a {n_total})"

    badges = resumen_badges(resueltas)
    alertas_badges = evaluar_alertas_badges(badges)
    hitos = detectar_hitos_muestra(badges, estado_anterior)
    umbrales = resumen_umbrales_prob(resueltas)
    ev = resumen_ev(resueltas)
    tendencia = tendencia_reciente(resueltas, hoy=hoy)
    calidad = chequear_calidad(todas)

    partes = [
        f"# Reporte semanal de tenis -- {hoy.isoformat()}",
        "",
        "Generado automáticamente por `src/scripts/reporte_semanal_tenis.py` a partir de "
        "`apuestas` (Supabase, `liga = \"Tenis\"`). Solo lectura -- no cambia nada del "
        "motor ni de la config de producción. Cualquier decisión sobre lo que sigue es "
        "manual.",
        "",
        "## Panorama general",
        "",
        f"- Apuestas de tenis resueltas (ganada/perdida) a la fecha: **{n_total}**",
        f"- Nuevas resueltas desde el último reporte: **{linea_nuevas}**",
        f"- Accuracy global acumulada: **{pct_total:.2f}%** ({aciertos_total}/{n_total})",
        "",
        "## Accuracy por badge",
        "",
        "Rangos de referencia validados en `tennis_validacion_filtro_ev.md`: "
        "`verde` ~71%+, `amarillo` ~60-67%. Solo se alerta con `n >= "
        f"{MIN_MUESTRA_ALERTA}` para no generar alarmas falsas con poca muestra.",
        "",
        _fmt_tabla(sorted(badges.items())),
        "",
    ]

    if alertas_badges:
        partes.append("**Alertas de badge:**")
        partes.extend(f"- ⚠️ {a}" for a in alertas_badges)
    else:
        partes.append("Sin alertas de badge (todo dentro de rango, o sin muestra suficiente todavía).")
    partes.append("")

    partes += [
        "## Accuracy por umbral de probabilidad",
        "",
        _fmt_tabla(list(umbrales.items())),
        "",
        "## Chequeo del EV (solo seguimiento -- ya no decide qué se muestra)",
        "",
        _fmt_tabla(list(ev.items())),
        "",
        "## Tendencia reciente",
        "",
        f"Últimos {tendencia['dias']} días (por `fecha_registro`) vs. acumulado histórico:",
        "",
        _fmt_tabla([
            ("histórico (todo)", tendencia["historico"]),
            (f"últimos {tendencia['dias']} días", tendencia["reciente"]),
        ]),
        "",
        "## Chequeos de calidad de datos",
        "",
    ]

    if calidad["duplicados"]:
        partes.append(f"- ⚠️ **{len(calidad['duplicados'])} grupo(s) de filas duplicadas exactas:**")
        for d in calidad["duplicados"]:
            partes.append(
                f"  - {d['local']} vs {d['visitante']} ({d['pick_descripcion']}) -- {d['copias']} copias"
            )
    else:
        partes.append("- Duplicados exactos: ninguno.")

    if calidad["incoherencias"]:
        partes.append(f"- ⚠️ **{len(calidad['incoherencias'])} incoherencia(s) local/visitante vs. pick_descripcion:**")
        for inc in calidad["incoherencias"]:
            partes.append(f"  - id={inc['id']}: \"{inc['pick_descripcion']}\" no matchea con {inc['local']}/{inc['visitante']}")
    else:
        partes.append("- Coherencia local/visitante vs. pick_descripcion: OK.")

    if calidad["campos_vacios"]:
        partes.append("- ⚠️ **Campos numéricos vacíos en filas resueltas:**")
        for campo, ids in calidad["campos_vacios"].items():
            partes.append(f"  - `{campo}` vacío en ids: {ids}")
    else:
        partes.append("- Campos numéricos en filas resueltas: sin vacíos.")

    partes.append("")
    partes.append("## Para decidir")
    partes.append("")

    hay_algo_que_revisar = bool(alertas_badges) or bool(hitos) or bool(calidad["duplicados"]) or bool(calidad["incoherencias"]) or bool(calidad["campos_vacios"])
    if hay_algo_que_revisar:
        partes.append("**Esto podría valer la pena revisar con Juan:**")
        partes.extend(f"- {a}" for a in alertas_badges)
        partes.extend(f"- {h}" for h in hitos)
        if calidad["duplicados"]:
            partes.append(f"- {len(calidad['duplicados'])} grupo(s) de duplicados exactos (ver Chequeos de calidad).")
        if calidad["incoherencias"]:
            partes.append(f"- {len(calidad['incoherencias'])} incoherencia(s) local/visitante (ver Chequeos de calidad).")
        if calidad["campos_vacios"]:
            partes.append("- Campos numéricos vacíos en filas resueltas (ver Chequeos de calidad).")
    else:
        partes.append("Nada para decidir todavía -- todo dentro de lo esperado y sin problemas de datos.")

    md = "\n".join(partes) + "\n"

    estado_nuevo = {
        "fecha": hoy.isoformat(),
        "resueltas_totales": n_total,
        "badge_n": {badge: n for badge, (n, _, _) in badges.items()},
    }
    return md, estado_nuevo


# ──────────────────────────────────────────────
# Orquestación (I/O real -- Supabase + disco)
# ──────────────────────────────────────────────

def obtener_datos_tenis() -> list[dict]:
    filas = _sb.leer_apuestas()
    return [f for f in filas if f.get("liga") == "Tenis"]


def main() -> int:
    if not _sb.disponible():
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY no configurados -- nada que hacer.", file=sys.stderr)
        return 1

    todas = obtener_datos_tenis()
    estado_anterior = cargar_estado()
    hoy = date.today()
    md, estado_nuevo = generar_reporte_md(todas, estado_anterior, hoy=hoy)

    REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    destino = REPORTES_DIR / f"reporte_tenis_{hoy.isoformat()}.md"
    if destino.exists():
        print(f"{destino} ya existe -- no se pisa, se sale sin escribir de nuevo.")
        return 0

    destino.write_text(md, encoding="utf-8")
    guardar_estado(estado_nuevo)
    print(f"Reporte generado: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
