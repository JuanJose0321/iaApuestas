"""
Reporte semanal automático de accuracy/patrones de fútbol.

Mismo patrón que `reporte_semanal_tenis.py` (mismo archivo como base, ver
`Claude outputs/prompt_reporte_semanal_futbol.txt`), adaptado a fútbol:

- Fútbol y tenis comparten la misma tabla `apuestas` de Supabase --
  se distinguen por la columna `liga`: tenis siempre guarda `liga="Tenis"`
  (`tRegistrar()` en templates/index.html), fútbol guarda el nombre real
  de la competición seleccionada (`liga=currentLiga`, ej. "LaLiga",
  "Premier League", "Champions League", etc., o "Default" si no se
  eligió ninguna). Por eso el filtro de fútbol es "liga != Tenis y liga
  no vacía/Default", no una lista fija de nombres de liga -- así no hay
  que tocar este script cada vez que se agrega una liga nueva a
  `equipos_por_liga.json`.
- Fútbol usa `pick_tipo` = "directa" / "dupla" / "tripleta" (no "TENIS"
  como tenis) -- se reporta por separado porque la fórmula de confianza
  de cada tipo se corrigió por separado (ver commits fee1a01/5ddb82d,
  septiembre 2026): la confianza de "directa" y de las combinadas no
  deben ponderar por EV.
- A diferencia de tenis, fútbol NO tiene un rango de accuracy "bueno"
  validado para alertar -- el propio diagnóstico
  (`futbol_diagnostico_completo.md`) encontró que el modelo de producción
  **no le gana al mercado** (53.94% vs 54.14% de accuracy, Brier peor:
  0.5827 vs 0.5767). Por eso este reporte no tiene alertas de rango de
  badge como el de tenis -- su función es otra: confirmar con el tiempo
  si ese hallazgo sigue siendo cierto con datos reales de producción, o
  si algo cambia. La comparación central es modelo vs. probabilidad
  implícita de la cuota real de cada pick (misma idea que el backtest,
  aplicada a las apuestas reales ya resueltas).

Solo LEE de Supabase (`apuestas`) y escribe un archivo de reporte en
`reportes/` -- no toca `src/engines/football.py` ni ninguna otra ruta de
producción, y no decide ni activa nada por su cuenta.

Uso
---
    python src/scripts/reporte_semanal_futbol.py

Guarda `reportes/reporte_futbol_YYYY-MM-DD.md` (no pisa reportes
anteriores) y actualiza `reportes/estado_futbol.json` (archivo de estado
propio, separado del de tenis -- mismo bookkeeping: resueltas totales y
tamaño de cada badge en el último reporte).
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
ESTADO_PATH = REPORTES_DIR / "estado_futbol.json"

MIN_MUESTRA_ALERTA = 20   # no reportar hitos/calibración con menos de esta cantidad
# Mismos umbrales que src/core/confidence.py (UMBRAL_ROJO/AMARILLO/VERDE) --
# no los de tenis, que están en otra escala (certeza, no probabilidad).
UMBRALES_PROB = (0.50, 0.65, 0.75)
DIAS_TENDENCIA_RECIENTE = 14

# Referencia de futbol_diagnostico_completo.md (Paso 3.1, backtest walk-forward
# 2021-2026, n=7,008) -- NO es un rango "esperado" para alertar (a diferencia
# de tenis): es el punto de comparación para saber si la realidad de
# producción sigue pareciéndose a lo que dio el backtest o se despegó.
REFERENCIA_BACKTEST = {
    "accuracy_modelo": 53.94, "brier_modelo": 0.5827,
    "accuracy_mercado": 54.14, "brier_mercado": 0.5767,
}


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
# Paso 2 -- accuracy por liga y por tipo de pick
# ──────────────────────────────────────────────

def resumen_por_liga(resueltas: list[dict]) -> dict[str, tuple[int, int, float]]:
    por_liga: dict[str, list[dict]] = {}
    for f in resueltas:
        liga = f.get("liga") or "sin_liga"
        por_liga.setdefault(liga, []).append(f)
    return {liga: accuracy(lst) for liga, lst in por_liga.items()}


def resumen_por_tipo(resueltas: list[dict]) -> dict[str, tuple[int, int, float]]:
    por_tipo: dict[str, list[dict]] = {}
    for f in resueltas:
        tipo = f.get("pick_tipo") or "sin_tipo"
        por_tipo.setdefault(tipo, []).append(f)
    return {tipo: accuracy(lst) for tipo, lst in por_tipo.items()}


def resumen_badges(resueltas: list[dict]) -> dict[str, tuple[int, int, float]]:
    por_badge: dict[str, list[dict]] = {}
    for f in resueltas:
        badge = f.get("confianza_badge") or "sin_badge"
        por_badge.setdefault(badge, []).append(f)
    return {badge: accuracy(lst) for badge, lst in por_badge.items()}


def detectar_hitos_muestra(
    resumen: dict[str, tuple[int, int, float]], estado_anterior: Optional[dict]
) -> list[str]:
    """Avisa cuando un badge cruza el umbral de MIN_MUESTRA_ALERTA desde el
    último reporte -- el punto en el que una cifra empieza a poder leerse
    en serio. Sin estado_anterior (primer reporte) no hay "antes" contra el
    cual comparar, así que se omite (evita falsos positivos)."""
    if estado_anterior is None:
        return []
    hitos = []
    badge_n_anterior = estado_anterior.get("badge_n", {})
    for badge, (n, aciertos, pct) in resumen.items():
        n_antes = badge_n_anterior.get(badge, 0)
        if n >= MIN_MUESTRA_ALERTA and n_antes < MIN_MUESTRA_ALERTA:
            hitos.append(
                f"`{badge}` ya junta {n} resueltas (antes tenía {n_antes}, por debajo "
                f"de {MIN_MUESTRA_ALERTA}) -- accuracy con muestra ya razonable: {pct:.2f}%."
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
# Paso 5 -- modelo vs. probabilidad implícita del mercado
# ──────────────────────────────────────────────

def comparacion_mercado(resueltas: list[dict]) -> dict:
    """Compara la calibración del modelo (prob_predicha) contra la
    probabilidad implícita cruda (1/cuota, CON vig -- no se puede
    remover sin la cuota del lado contrario, que no se guarda por pick)
    de la cuota real con la que se registró cada apuesta. Mismo espíritu
    que la comparación modelo-vs-mercado del backtest, pero sobre
    apuestas reales ya resueltas en vez de datos históricos."""
    pares = []
    for f in resueltas:
        cuota = f.get("cuota")
        prob = f.get("prob_predicha")
        if not cuota or cuota <= 0 or prob is None:
            continue
        y = 1 if f.get("resultado") == "ganada" else 0
        pares.append((float(prob), 1.0 / float(cuota), y))

    n = len(pares)
    if n == 0:
        return {"n": 0}

    brier_modelo = sum((p - y) ** 2 for p, _, y in pares) / n
    brier_mercado = sum((m - y) ** 2 for _, m, y in pares) / n
    acc_modelo = sum(1 for p, _, y in pares if (p >= 0.5) == bool(y)) / n * 100
    acc_mercado = sum(1 for _, m, y in pares if (m >= 0.5) == bool(y)) / n * 100

    mas_confiado = [(p, m, y) for p, m, y in pares if p > m]
    menos_confiado = [(p, m, y) for p, m, y in pares if p <= m]
    acc_mas_confiado = accuracy([{"resultado": "ganada" if y else "perdida"} for _, _, y in mas_confiado])
    acc_menos_confiado = accuracy([{"resultado": "ganada" if y else "perdida"} for _, _, y in menos_confiado])

    return {
        "n": n,
        "acc_modelo": acc_modelo, "brier_modelo": brier_modelo,
        "acc_mercado_implicito": acc_mercado, "brier_mercado_implicito": brier_mercado,
        "acc_cuando_modelo_mas_confiado": acc_mas_confiado,
        "acc_cuando_modelo_menos_confiado_o_igual": acc_menos_confiado,
    }


# ──────────────────────────────────────────────
# Paso 6 -- tendencia reciente vs histórico
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
# Paso 7 -- chequeos de calidad de datos
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

    campos_obligatorios = ["cuota", "prob_predicha", "ev_predicho", "confianza_score"]
    resueltas = filas_resueltas(todas)
    campos_vacios: dict[str, list] = {c: [] for c in campos_obligatorios}
    for f in resueltas:
        for campo in campos_obligatorios:
            if f.get(campo) is None or f.get(campo) == "":
                campos_vacios[campo].append(f.get("id"))
    campos_vacios = {c: ids for c, ids in campos_vacios.items() if ids}

    return {"duplicados": duplicados, "campos_vacios": campos_vacios}


# ──────────────────────────────────────────────
# Estado entre corridas
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

    por_liga = resumen_por_liga(resueltas)
    por_tipo = resumen_por_tipo(resueltas)
    badges = resumen_badges(resueltas)
    hitos = detectar_hitos_muestra(badges, estado_anterior)
    umbrales = resumen_umbrales_prob(resueltas)
    ev = resumen_ev(resueltas)
    mercado = comparacion_mercado(resueltas)
    tendencia = tendencia_reciente(resueltas, hoy=hoy)
    calidad = chequear_calidad(todas)

    partes = [
        f"# Reporte semanal de fútbol -- {hoy.isoformat()}",
        "",
        "Generado automáticamente por `src/scripts/reporte_semanal_futbol.py` a partir de "
        "`apuestas` (Supabase, todo lo que no tiene `liga = \"Tenis\"`). Solo lectura -- no "
        "cambia nada de `src/engines/football.py` ni de la config de producción. Cualquier "
        "decisión sobre lo que sigue es manual.",
        "",
        "## Panorama general",
        "",
        f"- Apuestas de fútbol resueltas (ganada/perdida) a la fecha: **{n_total}**",
        f"- Nuevas resueltas desde el último reporte: **{linea_nuevas}**",
        f"- Accuracy global acumulada: **{pct_total:.2f}%** ({aciertos_total}/{n_total})",
        "",
        "## Accuracy por liga",
        "",
        _fmt_tabla(sorted(por_liga.items())),
        "",
        "## Accuracy por tipo de pick (directa / dupla / tripleta)",
        "",
        _fmt_tabla(sorted(por_tipo.items())),
        "",
        "## Accuracy por badge de confianza",
        "",
        "Sin rango \"esperado\" para alertar (a diferencia de tenis) -- "
        "`futbol_diagnostico_completo.md` encontró que el modelo no le gana "
        "al mercado, así que no hay un piso de accuracy validado como bueno. "
        "Esta tabla es solo descriptiva.",
        "",
        _fmt_tabla(sorted(badges.items())),
        "",
    ]

    if hitos:
        partes.append("**Hitos de muestra:**")
        partes.extend(f"- {h}" for h in hitos)
        partes.append("")

    partes += [
        "## Accuracy por umbral de probabilidad",
        "",
        _fmt_tabla(list(umbrales.items())),
        "",
        "## Chequeo del EV (solo seguimiento -- ya no decide qué se muestra, "
        "ver commits fee1a01/5ddb82d)",
        "",
        _fmt_tabla(list(ev.items())),
        "",
        "## Modelo vs. probabilidad implícita de la cuota real",
        "",
        f"Referencia del backtest walk-forward (`futbol_diagnostico_completo.md`, n=7,008): "
        f"modelo {REFERENCIA_BACKTEST['accuracy_modelo']:.2f}% / Brier "
        f"{REFERENCIA_BACKTEST['brier_modelo']:.4f} -- mercado "
        f"{REFERENCIA_BACKTEST['accuracy_mercado']:.2f}% / Brier "
        f"{REFERENCIA_BACKTEST['brier_mercado']:.4f}. El backtest concluyó que el modelo "
        "**no le gana al mercado**. Esta sección compara lo mismo, pero con apuestas "
        "reales de producción ya resueltas -- para ver si eso sigue siendo así.",
        "",
    ]
    if mercado["n"] == 0:
        partes.append("Sin apuestas resueltas con cuota y probabilidad válidas todavía.")
    else:
        partes += [
            f"- n con cuota y probabilidad válidas: **{mercado['n']}**",
            f"- Accuracy modelo: **{mercado['acc_modelo']:.2f}%** -- Brier modelo: "
            f"**{mercado['brier_modelo']:.4f}**",
            f"- Accuracy implícita de la cuota (1/cuota, con vig): "
            f"**{mercado['acc_mercado_implicito']:.2f}%** -- Brier: "
            f"**{mercado['brier_mercado_implicito']:.4f}**",
            f"- Accuracy cuando el modelo está MÁS confiado que la cuota implícita: "
            f"{mercado['acc_cuando_modelo_mas_confiado'][2]:.2f}% "
            f"(n={mercado['acc_cuando_modelo_mas_confiado'][0]})",
            f"- Accuracy cuando el modelo está IGUAL o MENOS confiado: "
            f"{mercado['acc_cuando_modelo_menos_confiado_o_igual'][2]:.2f}% "
            f"(n={mercado['acc_cuando_modelo_menos_confiado_o_igual'][0]})",
        ]
    partes.append("")

    partes += [
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

    if calidad["campos_vacios"]:
        partes.append("- ⚠️ **Campos numéricos vacíos en filas resueltas:**")
        for campo, ids in calidad["campos_vacios"].items():
            partes.append(f"  - `{campo}` vacío en ids: {ids}")
    else:
        partes.append("- Campos numéricos en filas resueltas: sin vacíos.")

    partes.append("")
    partes.append("## Para decidir")
    partes.append("")

    hay_algo_que_revisar = bool(hitos) or bool(calidad["duplicados"]) or bool(calidad["campos_vacios"])
    if hay_algo_que_revisar:
        partes.append("**Esto podría valer la pena revisar con Juan:**")
        partes.extend(f"- {h}" for h in hitos)
        if calidad["duplicados"]:
            partes.append(f"- {len(calidad['duplicados'])} grupo(s) de duplicados exactos (ver Chequeos de calidad).")
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

def obtener_datos_futbol() -> list[dict]:
    """Todo lo que NO es tenis y tiene una liga real asignada -- fútbol es
    la única otra rama de producción (MLB todavía no existe en la app,
    ver mlb_diagnostico_y_diseño.md), así que no hace falta enumerar
    ligas a mano ni tocar este filtro cuando se agregue una liga nueva."""
    filas = _sb.leer_apuestas()
    return [f for f in filas if f.get("liga") not in ("Tenis", "", None, "Default")]


def main() -> int:
    if not _sb.disponible():
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY no configurados -- nada que hacer.", file=sys.stderr)
        return 1

    todas = obtener_datos_futbol()
    estado_anterior = cargar_estado()
    hoy = date.today()
    md, estado_nuevo = generar_reporte_md(todas, estado_anterior, hoy=hoy)

    REPORTES_DIR.mkdir(parents=True, exist_ok=True)
    destino = REPORTES_DIR / f"reporte_futbol_{hoy.isoformat()}.md"
    if destino.exists():
        print(f"{destino} ya existe -- no se pisa, se sale sin escribir de nuevo.")
        return 0

    destino.write_text(md, encoding="utf-8")
    guardar_estado(estado_nuevo)
    print(f"Reporte generado: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
