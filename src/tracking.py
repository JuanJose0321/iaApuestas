"""
Sistema de tracking de apuestas.
Gestiona CSV de historial, bankroll_config.json, backups y métricas.
"""
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Paths sobreescribibles para tests
CSV_PATH    = DATA_DIR / "apuestas_registradas.csv"
CONFIG_PATH = DATA_DIR / "bankroll_config.json"
BACKUP_DIR  = DATA_DIR / "backup"

COLUMNS = [
    "id", "fecha_registro", "fecha_partido", "liga", "local", "visitante",
    "pick_tipo", "pick_descripcion", "cuota", "stake", "prob_predicha",
    "ev_predicho", "confianza_score", "confianza_badge",
    "resultado", "ganancia_neta", "bankroll_antes", "bankroll_despues",
    "fixture_id_api", "notas",
]

RESULTADOS_VALIDOS = {"ganada", "perdida", "void", "cashout", "pendiente"}
BACKUP_CADA = 10   # hacer backup cada N apuestas nuevas


# ──────────────────────────────────────────────
# Inicialización / persistencia
# ──────────────────────────────────────────────

def _init():
    """Asegura que el CSV y el directorio de backup existen."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def leer_config() -> dict:
    """Lee bankroll_config.json. Lo crea con valores por defecto si no existe."""
    default = {
        "bankroll_inicial": 800.0,
        "bankroll_actual":  800.0,
        "stake_modo":       "conservador",
        "stake_fijo":       20.0,
        "max_stake_pct":    0.03,
        "racha_negativa_alerta": 5,
        "fecha_inicio":     datetime.now().strftime("%Y-%m-%d"),
    }
    if not CONFIG_PATH.exists():
        guardar_config(default)
        return default
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default


def guardar_config(cfg: dict):
    CONFIG_PATH.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def leer_historial(csv_path: Path | None = None) -> list[dict]:
    """Lee todas las apuestas del CSV. Devuelve lista vacía si no hay datos."""
    path = csv_path or CSV_PATH
    _init()
    # Intentar leer; si falla, usar el último backup
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception:
        return _leer_desde_backup() or []


def _leer_desde_backup() -> list[dict] | None:
    backups = sorted(BACKUP_DIR.glob("apuestas_*.csv"), reverse=True)
    for bk in backups:
        try:
            with open(bk, "r", newline="", encoding="utf-8") as f:
                rows = [dict(r) for r in csv.DictReader(f)]
            shutil.copy2(bk, CSV_PATH)  # restaurar
            return rows
        except Exception:
            continue
    return None


def _escribir_todas(rows: list[dict], csv_path: Path | None = None):
    path = csv_path or CSV_PATH
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _hacer_backup():
    if not CSV_PATH.exists():
        return
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"apuestas_{ts}.csv"
    shutil.copy2(CSV_PATH, dest)


def _siguiente_id(rows: list[dict]) -> int:
    if not rows:
        return 1
    try:
        return max(int(r["id"]) for r in rows if r.get("id")) + 1
    except ValueError:
        return len(rows) + 1


# ──────────────────────────────────────────────
# Operaciones principales
# ──────────────────────────────────────────────

def registrar_apuesta(data: dict, csv_path: Path | None = None) -> dict:
    """
    Registra una nueva apuesta como 'pendiente'.
    Devuelve {id, mensaje, total_apuestas}.
    """
    _init()
    rows   = leer_historial(csv_path)
    cfg    = leer_config()
    nid    = _siguiente_id(rows)
    bankroll_antes = round(float(cfg.get("bankroll_actual", 800)), 2)

    fila: dict = {
        "id":               nid,
        "fecha_registro":   datetime.now().strftime("%d/%m/%Y %H:%M"),
        "fecha_partido":    data.get("fecha_partido", ""),
        "liga":             data.get("liga", ""),
        "local":            data.get("local", ""),
        "visitante":        data.get("visitante", ""),
        "pick_tipo":        data.get("pick_tipo", ""),
        "pick_descripcion": data.get("pick_descripcion", ""),
        "cuota":            round(float(data.get("cuota", 0)), 2),
        "stake":            round(float(data.get("stake", 0)), 2),
        "prob_predicha":    round(float(data.get("prob_predicha", 0)), 4),
        "ev_predicho":      round(float(data.get("ev_predicho", 0)), 4),
        "confianza_score":  round(float(data.get("confianza_score", 0)), 4),
        "confianza_badge":  data.get("confianza_badge", ""),
        "resultado":        "pendiente",
        "ganancia_neta":    "",
        "bankroll_antes":   bankroll_antes,
        "bankroll_despues": "",
        "fixture_id_api":   data.get("fixture_id_api", ""),
        "notas":            data.get("notas", ""),
    }

    rows.append(fila)

    # Backup automático cada BACKUP_CADA apuestas
    if nid % BACKUP_CADA == 0:
        _hacer_backup()

    _escribir_todas(rows, csv_path)
    return {"id": nid, "mensaje": f"Apuesta #{nid} registrada", "total_apuestas": len(rows)}


def actualizar_resultado(id_apuesta: int, resultado: str,
                          notas: str = "", csv_path: Path | None = None) -> dict:
    """
    Actualiza resultado y recalcula bankroll. Devuelve el registro actualizado.
    """
    if resultado not in RESULTADOS_VALIDOS:
        raise ValueError(f"Resultado inválido: '{resultado}'. Válidos: {RESULTADOS_VALIDOS}")

    rows = leer_historial(csv_path)
    cfg  = leer_config()

    target = next((r for r in rows if str(r.get("id")) == str(id_apuesta)), None)
    if target is None:
        raise ValueError(f"Apuesta #{id_apuesta} no encontrada")

    cuota  = float(target["cuota"])
    stake  = float(target["stake"])
    br_antes = (
        float(target["bankroll_antes"])
        if target.get("bankroll_antes") not in ("", None)
        else float(cfg.get("bankroll_actual", 800))
    )

    # Calcular ganancia_neta
    if resultado == "ganada":
        ganancia_neta = round(stake * (cuota - 1), 2)
    elif resultado == "perdida":
        ganancia_neta = round(-stake, 2)
    elif resultado in ("void", "cashout"):
        ganancia_neta = 0.0
    else:  # pendiente (vuelta atrás)
        ganancia_neta = None

    if ganancia_neta is not None:
        br_despues = round(br_antes + ganancia_neta, 2)
        cfg["bankroll_actual"] = br_despues
        guardar_config(cfg)
    else:
        br_despues = None

    target["resultado"]        = resultado
    target["ganancia_neta"]    = "" if ganancia_neta is None else ganancia_neta
    target["bankroll_despues"] = "" if br_despues is None else br_despues
    if notas:
        target["notas"] = notas

    _escribir_todas(rows, csv_path)
    return {
        "id":             int(id_apuesta),
        "resultado":      resultado,
        "ganancia_neta":  ganancia_neta,
        "bankroll_antes": br_antes,
        "bankroll_despues": br_despues,
    }


def calcular_metricas(csv_path: Path | None = None) -> dict:
    """
    Calcula métricas completas del historial.
    """
    rows = leer_historial(csv_path)
    cfg  = leer_config()

    total     = len(rows)
    ganadas   = sum(1 for r in rows if r["resultado"] == "ganada")
    perdidas  = sum(1 for r in rows if r["resultado"] == "perdida")
    pendientes = sum(1 for r in rows if r["resultado"] == "pendiente")
    void_ct   = sum(1 for r in rows if r["resultado"] in ("void", "cashout"))

    decididas = ganadas + perdidas
    tasa_acierto = round(ganadas / decididas * 100, 1) if decididas > 0 else None

    ganancias = [
        float(r["ganancia_neta"])
        for r in rows
        if r.get("ganancia_neta") not in ("", None)
    ]
    stakes_val = [float(r["stake"]) for r in rows if r.get("stake") not in ("", None)]

    ganancia_total = round(sum(ganancias), 2)
    stake_total    = round(sum(stakes_val), 2)
    roi = round(ganancia_total / stake_total * 100, 1) if stake_total > 0 else 0.0

    bankroll_actual  = round(float(cfg.get("bankroll_actual", 800)), 2)
    bankroll_inicial = round(float(cfg.get("bankroll_inicial", 800)), 2)
    variacion_pct    = round((bankroll_actual - bankroll_inicial) / bankroll_inicial * 100, 1)

    # ── Tasa acierto por nivel de confianza ──
    def _tasa(badge):
        subset = [r for r in rows
                  if r.get("confianza_badge") == badge
                  and r["resultado"] in ("ganada", "perdida")]
        if not subset:
            return None
        return round(sum(1 for r in subset if r["resultado"] == "ganada") / len(subset) * 100, 1)

    # ── Calibración ──
    rows_cal = [r for r in rows if r.get("prob_predicha") and r["resultado"] in ("ganada", "perdida")]
    prob_prom = None
    calibracion = None
    if rows_cal:
        prob_prom = round(sum(float(r["prob_predicha"]) for r in rows_cal) / len(rows_cal) * 100, 1)
        if tasa_acierto is not None:
            calibracion = round(prob_prom - tasa_acierto, 1)

    # ── Racha actual ──
    decididas_orden = [r for r in reversed(rows) if r["resultado"] in ("ganada", "perdida")]
    racha, racha_tipo = 0, None
    if decididas_orden:
        racha_tipo = decididas_orden[0]["resultado"]
        for r in decididas_orden:
            if r["resultado"] == racha_tipo:
                racha += 1
            else:
                break

    # ── Alertas ──
    alerta_bankroll = bankroll_actual < bankroll_inicial * 0.80
    alerta_racha    = (racha >= cfg.get("racha_negativa_alerta", 5)
                       and racha_tipo == "perdida")

    # ── Datos para gráfica de bankroll ──
    puntos_grafica = [{"x": 0, "y": bankroll_inicial, "label": "Inicio"}]
    for i, r in enumerate(rows):
        if r.get("bankroll_despues") not in ("", None):
            puntos_grafica.append({
                "x":     int(r["id"]),
                "y":     round(float(r["bankroll_despues"]), 2),
                "label": r.get("fecha_registro", ""),
            })

    return {
        "total":              total,
        "ganadas":            ganadas,
        "perdidas":           perdidas,
        "pendientes":         pendientes,
        "void":               void_ct,
        "tasa_acierto":       tasa_acierto,
        "ganancia_total":     ganancia_total,
        "stake_total":        stake_total,
        "roi":                roi,
        "profit_neto":        ganancia_total,
        "bankroll_actual":    bankroll_actual,
        "bankroll_inicial":   bankroll_inicial,
        "variacion_pct":      variacion_pct,
        "tasa_alta_confianza":  _tasa("alta"),
        "tasa_media_confianza": _tasa("media"),
        "prob_promedio_pct":    prob_prom,
        "calibracion":          calibracion,
        "racha_actual":         racha,
        "racha_tipo":           racha_tipo,
        "alerta_bankroll":      alerta_bankroll,
        "alerta_racha":         alerta_racha,
        "grafica_bankroll":     puntos_grafica,
    }


def determinar_resultado_pick(pick_desc: str, local: str, visitante: str,
                               goles_local: int, goles_visitante: int) -> str:
    """
    Determina 'ganada', 'perdida' o 'unknown' a partir de la descripción del pick
    y el marcador final.  Soporta multi-leg (separados por ' + ').
    """
    condiciones = [s.strip() for s in pick_desc.split("+")]
    for cond in condiciones:
        c = cond.lower().strip()
        if "menos de 2.5" in c:
            ok = (goles_local + goles_visitante) <= 2
        elif "más de 2.5" in c or "mas de 2.5" in c:
            ok = (goles_local + goles_visitante) >= 3
        elif "sí anotan ambos" in c or "si anotan ambos" in c:
            ok = goles_local > 0 and goles_visitante > 0
        elif "no anotan ambos" in c:
            ok = goles_local == 0 or goles_visitante == 0
        elif "empate" in c:
            ok = goles_local == goles_visitante
        elif local and f"gana {local.lower()}" in c:
            ok = goles_local > goles_visitante
        elif visitante and f"gana {visitante.lower()}" in c:
            ok = goles_visitante > goles_local
        else:
            return "unknown"
        if not ok:
            return "perdida"
    return "ganada"
