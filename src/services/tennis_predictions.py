"""
Log de predicciones de tenis: registra CADA análisis que corre el motor
(tenga o no pick con value), independiente del historial de apuestas
(tracking.py), para poder medir la precisión real del modelo — quién
gana y cuántos games totales — contra lo que de verdad pasó, no solo
contra las apuestas que el usuario decidió hacer.

Mismo patrón dual CSV/Supabase que tracking.py: si hay credenciales de
Supabase configuradas, la persistencia usa la tabla `predicciones_tenis`
en su lugar (ver supabase/schema.sql).
"""
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.services import supabase_client as _sb
from src.engines.tennis_improved import STD_GAMES, _STD_GAMES_DEFAULT

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CSV_PATH = DATA_DIR / "predicciones_tenis.csv"

COLUMNS = [
    "id", "fecha_registro", "fecha_partido", "jugador1", "jugador2",
    "favorito", "prob_favorito", "cuota_favorito", "superficie", "formato",
    "total_esp", "total_games_linea", "cuota_total_games_over", "cuota_total_games_under",
    "tuvo_pick", "tipo_pick",
    "ganador_real", "total_games_real", "acerto_ganador", "acerto_total",
    "resultado_cargado",
]


def _init(csv_path: Path | None = None):
    """Crea el CSV (con headers) si no existe todavía. Respeta el
    csv_path explícito del caller — nunca toca el CSV_PATH por defecto
    cuando el caller pidió otro archivo (ej. en tests)."""
    path = csv_path or CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()


def _siguiente_id(rows: list[dict]) -> int:
    if not rows:
        return 1
    try:
        return max(int(r["id"]) for r in rows if r.get("id")) + 1
    except ValueError:
        return len(rows) + 1


def _escribir_todas(rows: list[dict], csv_path: Path | None = None):
    path = csv_path or CSV_PATH
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def leer_predicciones(csv_path: Path | None = None) -> list[dict]:
    """Todas las predicciones logueadas (Supabase si está configurado)."""
    if csv_path is None and _sb.disponible():
        try:
            return _sb.leer_predicciones_tenis()
        except Exception:
            return []

    path = csv_path or CSV_PATH
    _init(csv_path)
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception:
        return []


def _num_opcional(valor, decimales: int):
    """Redondea si viene un valor real; None si el mercado no se cargó —
    nunca inventa un 0.0 que se confundiría con una cuota real, y None
    (a diferencia de "") viaja bien tanto a CSV (queda en blanco) como a
    Supabase (NULL en vez de un string vacío que rompería una columna
    NUMERIC)."""
    if valor in (None, ""):
        return None
    try:
        return round(float(valor), decimales)
    except (TypeError, ValueError):
        return None


def registrar_prediccion(data: dict, csv_path: Path | None = None) -> dict:
    """
    Registra un análisis de tenis recién corrido (con o sin pick de value).
    Devuelve {"id": ...}.

    Guarda también las cuotas reales que cargó el usuario (cuota_favorito,
    total_games_linea/cuota_total_games_over/under) — hasta ahora el log
    solo tenía la probabilidad del modelo, sin la cuota de mercado, así
    que era imposible reconstruir el EV real de una predicción pasada
    para validar con datos históricos si un EV alto predice mejor
    performance (ver report_tennis_audit.md). No cambia ningún umbral ni
    comportamiento existente — es solo captura de datos hacia adelante.
    """
    fila: dict = {
        "fecha_registro":    datetime.now().isoformat(),
        "fecha_partido":     data.get("fecha_partido", ""),
        "jugador1":          data.get("jugador1", ""),
        "jugador2":          data.get("jugador2", ""),
        "favorito":          data.get("favorito", ""),
        "prob_favorito":     round(float(data.get("prob_favorito", 0)), 4),
        "cuota_favorito":    _num_opcional(data.get("cuota_favorito"), 3),
        "superficie":        data.get("superficie", ""),
        "formato":           data.get("formato", ""),
        "total_esp":         round(float(data.get("total_esp", 0)), 2),
        "total_games_linea":       _num_opcional(data.get("total_games_linea"), 1),
        "cuota_total_games_over":  _num_opcional(data.get("cuota_total_games_over"), 3),
        "cuota_total_games_under": _num_opcional(data.get("cuota_total_games_under"), 3),
        "tuvo_pick":         bool(data.get("tuvo_pick", False)),
        "tipo_pick":         data.get("tipo_pick", ""),
        "ganador_real":      "",
        "total_games_real":  "",
        "acerto_ganador":    "",
        "acerto_total":      "",
        "resultado_cargado": "",
    }

    if csv_path is None and _sb.disponible():
        payload = dict(fila)
        payload.pop("fecha_registro")  # deja que Postgres use su DEFAULT NOW()
        creada = _sb.insertar_prediccion(payload)
        return {"id": creada["id"]}

    _init(csv_path)
    rows = leer_predicciones(csv_path)
    nid  = _siguiente_id(rows)
    fila["id"] = nid
    rows.append(fila)
    _escribir_todas(rows, csv_path)
    return {"id": nid}


def cargar_resultado(id_prediccion: int, ganador_real: Optional[str] = None,
                      total_games_real: Optional[float] = None,
                      csv_path: Path | None = None) -> dict:
    """
    Carga el resultado real de un partido ya analizado y calcula
    automáticamente acerto_ganador (favorito == ganador_real) y
    acerto_total (|total_games_real - total_esp| <= std_dev del formato —
    dentro de una desviación estándar de la distribución que el propio
    modelo predijo, en vez de un umbral arbitrario). std_dev no se
    guarda por fila: es constante por formato, se toma de STD_GAMES.
    Cualquiera de los dos campos es opcional (un partido puede cargarse
    solo con ganador, o solo con total, o ambos).
    """
    rows = leer_predicciones(csv_path)
    target = next((r for r in rows if str(r.get("id")) == str(id_prediccion)), None)
    if target is None:
        raise ValueError(f"Predicción #{id_prediccion} no encontrada")

    cambios: dict = {}
    if ganador_real is not None:
        cambios["ganador_real"]   = ganador_real
        cambios["acerto_ganador"] = ganador_real == target.get("favorito")
    if total_games_real is not None:
        total_esp = float(target["total_esp"])
        formato   = target.get("formato", "best_of_3")
        std_dev   = STD_GAMES.get(formato, _STD_GAMES_DEFAULT["best_of_3"])
        cambios["total_games_real"] = float(total_games_real)
        cambios["acerto_total"]     = abs(float(total_games_real) - total_esp) <= std_dev
    if not cambios:
        raise ValueError("Debe pasarse ganador_real y/o total_games_real")

    cambios["resultado_cargado"] = datetime.now().isoformat()

    if csv_path is None and _sb.disponible():
        _sb.actualizar_prediccion(int(id_prediccion), cambios)
    else:
        target.update(cambios)
        _escribir_todas(rows, csv_path)

    return {"id": int(id_prediccion), **cambios}


def calcular_precision(csv_path: Path | None = None) -> dict:
    """
    Precisión acumulada del modelo sobre predicciones con resultado real
    cargado — separada en "quién gana" y "total de games" porque no
    siempre se cargan juntos.
    """
    rows = leer_predicciones(csv_path)

    con_ganador = [r for r in rows if r.get("acerto_ganador") not in ("", None)]
    aciertos_ganador = sum(1 for r in con_ganador if str(r["acerto_ganador"]) in ("True", "true", "1"))

    con_total = [r for r in rows if r.get("acerto_total") not in ("", None)]
    aciertos_total = sum(1 for r in con_total if str(r["acerto_total"]) in ("True", "true", "1"))

    return {
        "n_total_predicciones": len(rows),
        "n_con_ganador_cargado": len(con_ganador),
        "precision_ganador_pct": (
            round(100.0 * aciertos_ganador / len(con_ganador), 1) if con_ganador else None
        ),
        "n_con_total_cargado": len(con_total),
        "precision_total_pct": (
            round(100.0 * aciertos_total / len(con_total), 1) if con_total else None
        ),
    }
