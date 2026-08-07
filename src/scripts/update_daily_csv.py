"""
Actualización diaria de los CSVs de football-data.co.uk en data/raw/.

Por qué solo football-data.co.uk (y no api-football.com / thesportsdb.com)
────────────────────────────────────────────────────────────────────────
Los CSVs en data/raw/ tienen un schema muy específico de football-data.co.uk
(Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR + decenas de columnas de cuotas).
api-football.com y thesportsdb.com son APIs de fixtures/forma en vivo con un
modelo de datos completamente distinto (búsqueda por equipo/ID, no archivos
de temporada) — no hay forma directa de reconstruir este CSV desde ahí, y
usar API_FOOTBALL_KEY acá competiría por la cuota diaria (100 req/día) con
el tráfico real de /chat en producción. Este script reusa el mismo origen
con el que se descargaron los CSVs originalmente (src/providers/loader.py).

Qué actualiza
─────────────
Los archivos ya versionados en data/raw/ (2021-22 a 2024-25) son temporadas
cerradas: football-data.co.uk no los vuelve a tocar, así que no hay nada que
actualizar ahí. Lo único que cambia día a día es la temporada en curso. Este
script calcula el código de temporada activo a partir de la fecha actual
(sin hardcodear años) y descarga también la temporada previa por si el
archivo de la temporada en curso recién está apareciendo (arranque de
temporada a mediados de agosto) — football-data.co.uk devuelve 404/HTML de
error si una temporada todavía no tiene archivo, y eso se trata como
"todavía no disponible", no como error fatal.

Uso
───
    python src/scripts/update_daily_csv.py
"""
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import polars as pl

from config import LIGAS, RAW_DATA_DIR
from src.providers.loader import descargar_temporada

_log = logging.getLogger("betbrain.update_csv")
_log.setLevel(logging.INFO)
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("%(asctime)s [update_csv] %(levelname)s %(message)s"))
_log.addHandler(_console)

# Log a archivo además de consola -- best effort, nunca debe tumbar el script
# (mismo patrón que api_football.py: un filesystem de solo lectura no puede
# romper la ejecución, solo se pierde el log en disco).
try:
    _file_handler = logging.FileHandler(
        Path(__file__).resolve().parent.parent.parent / "logs" / "update_csv.log",
        encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [update_csv] %(levelname)s %(message)s"))
    _log.addHandler(_file_handler)
except OSError:
    _log.warning("No se pudo abrir logs/update_csv.log (filesystem de solo lectura) -- solo consola")

DIAS_FRESCURA_ESPERADA = 30


def temporadas_activas(hoy: date | None = None) -> list[str]:
    """
    Códigos de temporada football-data.co.uk (ej. "2526") relevantes a HOY:
    la temporada en curso y la anterior. Las ligas europeas corren de
    agosto a mayo, así que la temporada que "está en curso" depende del mes.
    """
    hoy = hoy or date.today()
    inicio_actual = hoy.year if hoy.month >= 8 else hoy.year - 1
    cod = lambda y: f"{y % 100:02d}{(y + 1) % 100:02d}"
    return [cod(inicio_actual - 1), cod(inicio_actual)]


def _fecha_mas_reciente(csv_path: Path) -> date | None:
    try:
        df = pl.read_csv(csv_path, encoding="latin-1", ignore_errors=True)
        if df.is_empty() or "Date" not in df.columns:
            return None
        fechas = df.select(
            pl.col("Date").str.strptime(pl.Date, "%d/%m/%Y", strict=False)
        ).drop_nulls()
        if fechas.is_empty():
            return None
        return fechas.to_series().max()
    except Exception as exc:
        _log.warning("No se pudo leer fecha de %s: %s", csv_path.name, exc)
        return None


def actualizar() -> dict:
    """
    Descarga la temporada en curso (y la previa) para todas las ligas en
    LIGAS. Nunca lanza excepción -- errores individuales se registran y
    se sigue con el resto, para que un fallo de red puntual no tumbe todo
    el workflow de GitHub Actions.
    """
    temporadas = temporadas_activas()
    _log.info("Ligas: %s | Temporadas activas: %s", LIGAS, temporadas)

    actualizados: list[str] = []
    sin_cambios: list[str] = []
    fallidos: list[str] = []

    for liga in LIGAS:
        for temporada in temporadas:
            destino = RAW_DATA_DIR / f"{liga}_{temporada}.csv"
            existia_antes = destino.exists()
            hash_antes = destino.read_bytes() if existia_antes else None

            try:
                resultado = descargar_temporada(liga, temporada)
            except Exception as exc:
                _log.error("%s %s: excepción inesperada: %s", liga, temporada, exc, exc_info=True)
                fallidos.append(f"{liga}_{temporada}")
                continue

            if resultado is None:
                if not existia_antes:
                    _log.info("%s %s: aun no publicado (normal si la temporada no empezo)", liga, temporada)
                else:
                    _log.warning("%s %s: descarga falló, se conserva el archivo existente", liga, temporada)
                    fallidos.append(f"{liga}_{temporada}")
                continue

            if hash_antes is not None and resultado.read_bytes() == hash_antes:
                sin_cambios.append(f"{liga}_{temporada}")
                continue

            actualizados.append(f"{liga}_{temporada}")

            ultima_fecha = _fecha_mas_reciente(resultado)
            if ultima_fecha is not None:
                dias = (date.today() - ultima_fecha).days
                if dias > DIAS_FRESCURA_ESPERADA:
                    _log.warning(
                        "%s %s: ultimo partido hace %d dias (%s) -- puede ser normal en pretemporada",
                        liga, temporada, dias, ultima_fecha,
                    )
                else:
                    _log.info("%s %s: ultimo partido %s (hace %d dias)", liga, temporada, ultima_fecha, dias)

    resumen = {
        "actualizados": actualizados,
        "sin_cambios": sin_cambios,
        "fallidos": fallidos,
    }
    _log.info(
        "Resumen: %d actualizados, %d sin cambios, %d fallidos",
        len(actualizados), len(sin_cambios), len(fallidos),
    )
    return resumen


if __name__ == "__main__":
    resumen = actualizar()
    # Falla el job solo si absolutamente nada se pudo procesar (posible
    # caída total de football-data.co.uk) -- un fallo parcial no debe
    # bloquear el resto del workflow.
    total_intentos = len(resumen["actualizados"]) + len(resumen["sin_cambios"]) + len(resumen["fallidos"])
    if total_intentos > 0 and len(resumen["fallidos"]) == total_intentos:
        _log.error("Todas las descargas fallaron")
        sys.exit(1)
    sys.exit(0)
