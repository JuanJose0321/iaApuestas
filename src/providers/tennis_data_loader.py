"""
Cargador de datos históricos ATP/WTA desde GitHub de Jeff Sackmann.

Fuente: https://github.com/JeffSackmann/tennis_atp y tennis_wta
"""
import logging
import csv
from pathlib import Path
from typing import List, Dict
import urllib.request
import urllib.error

_log = logging.getLogger("betbrain.tennis_loader")

# URLs base de Jeff Sackmann
GITHUB_ATP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
GITHUB_WTA_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"

# Datos locales
DATA_DIR = Path(__file__).parent.parent / "data" / "tennis"
MATCHES_ATP_FILE = DATA_DIR / "matches_atp.csv"
MATCHES_WTA_FILE = DATA_DIR / "matches_wta.csv"


def descargar_archivo(url: str, output_path: Path, max_years: int = 3) -> bool:
    """
    Descarga un archivo CSV de GitHub.

    Args:
        url: URL del archivo
        output_path: Dónde guardar
        max_years: Cuántos años descargar (últimos N años)

    Returns:
        True si éxito, False si falla
    """
    try:
        _log.info(f"Descargando {url}...")
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read().decode('utf-8')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(data)
            _log.info(f"✅ Guardado: {output_path}")
            return True
    except urllib.error.URLError as e:
        _log.error(f"❌ Error descargando {url}: {e}")
        return False
    except Exception as e:
        _log.error(f"❌ Error inesperado: {e}")
        return False


def descargar_datos_tennis(genero: str = "ambos") -> bool:
    """
    Descarga histórico de partidos ATP/WTA.

    Args:
        genero: "atp", "wta" o "ambos"

    Returns:
        True si al menos uno descarga exitosamente
    """
    success = False

    if genero in ("atp", "ambos"):
        # Descargar últimos 3 años de ATP
        for year in range(2024, 2027):  # 2024, 2025, 2026
            url = f"{GITHUB_ATP_BASE}/matches{year}.csv"
            filename = DATA_DIR / f"matches_atp_{year}.csv"
            if descargar_archivo(url, filename):
                success = True

    if genero in ("wta", "ambos"):
        # Descargar últimos 3 años de WTA
        for year in range(2024, 2027):  # 2024, 2025, 2026
            url = f"{GITHUB_WTA_BASE}/matches{year}.csv"
            filename = DATA_DIR / f"matches_wta_{year}.csv"
            if descargar_archivo(url, filename):
                success = True

    return success


def combinar_archivos(patron: str = "matches_") -> List[Dict]:
    """
    Combina todos los archivos CSV de matches.

    Returns:
        Lista de partidos como dicts
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    matches = []

    for csv_file in DATA_DIR.glob("*.csv"):
        if not csv_file.name.startswith(patron):
            continue
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('winner_name') and row.get('loser_name'):
                        matches.append({
                            'date': row.get('tourney_date', ''),
                            'tournament': row.get('tourney_name', ''),
                            'level': row.get('tourney_level', 'ATP'),
                            'winner': row.get('winner_name', '').strip(),
                            'loser': row.get('loser_name', '').strip(),
                            'winner_sets': int(row.get('score', '0-0').split()[0]) if '-' in row.get('score', '0-0') else 0,
                            'loser_sets': int(row.get('score', '0-0').split()[-1]) if '-' in row.get('score', '0-0') else 0,
                        })
            _log.info(f"✅ Cargados {len(matches)} partidos de {csv_file.name}")
        except Exception as e:
            _log.error(f"❌ Error leyendo {csv_file}: {e}")

    return matches


def contar_partidos_recientes(jugador: str, matches: List[Dict],
                             ultimos_n: int = 5) -> Dict[str, int]:
    """
    Cuenta partidos recientes de un jugador.

    Returns:
        {'ganados': N, 'perdidos': M, 'total': N+M}
    """
    ganados = 0
    perdidos = 0

    # Los matches ya deberían estar en orden cronológico
    for match in matches[-ultimos_n:]:  # Últimos N partidos
        if match['winner'].lower() == jugador.lower():
            ganados += 1
        elif match['loser'].lower() == jugador.lower():
            perdidos += 1

    return {
        'ganados': ganados,
        'perdidos': perdidos,
        'total': ganados + perdidos,
        'porcentaje': round(100 * ganados / (ganados + perdidos), 1) if (ganados + perdidos) > 0 else 0
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _log.info("Descargando datos históricos de tenis...")
    if descargar_datos_tennis("ambos"):
        _log.info("✅ Descarga completada")
        matches = combinar_archivos()
        _log.info(f"Total de partidos cargados: {len(matches)}")
    else:
        _log.error("❌ No se pudieron descargar archivos")
