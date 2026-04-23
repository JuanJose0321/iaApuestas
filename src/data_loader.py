"""
Descarga de datos históricos de fútbol desde football-data.co.uk
Soporta múltiples ligas y temporadas.
"""
import pandas as pd
import requests
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, LIGAS, TEMPORADAS


BASE_URL = "https://www.football-data.co.uk/mmz4281/"


def descargar_temporada(liga: str, temporada: str) -> Path | None:
    """
    Descarga una temporada concreta (ej. liga='SP1', temporada='2425').
    Devuelve el path del CSV o None si falló.
    """
    url = f"{BASE_URL}{temporada}/{liga}.csv"
    destino = RAW_DATA_DIR / f"{liga}_{temporada}.csv"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and len(r.content) > 100:
            destino.write_bytes(r.content)
            print(f"  ✅ {liga} {temporada} → {destino.name} ({len(r.content) // 1024} KB)")
            return destino
        print(f"  ❌ {liga} {temporada}: HTTP {r.status_code}")
    except Exception as e:
        print(f"  ❌ {liga} {temporada}: {e}")
    return None


def descargar_todo(ligas=None, temporadas=None) -> list[Path]:
    """Descarga todas las combinaciones liga x temporada configuradas."""
    ligas = ligas or LIGAS
    temporadas = temporadas or TEMPORADAS
    print(f"🚀 Descargando {len(ligas)} ligas × {len(temporadas)} temporadas...")
    paths = []
    for liga in ligas:
        for t in temporadas:
            p = descargar_temporada(liga, t)
            if p:
                paths.append(p)
    print(f"✅ Total descargado: {len(paths)} archivos")
    return paths


def cargar_todo() -> pd.DataFrame:
    """
    Carga y concatena todos los CSV descargados en RAW_DATA_DIR.
    Normaliza fechas y conserva solo columnas útiles.
    """
    frames = []
    for csv in sorted(RAW_DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv, encoding="latin-1", on_bad_lines="skip")
            liga_val, temp_val = csv.stem.split("_")[0], csv.stem.split("_")[1]
            meta = pd.DataFrame(
                {"Liga": liga_val, "Temporada": temp_val}, index=df.index
            )
            frames.append(pd.concat([df, meta], axis=1))
        except Exception as e:
            print(f"⚠️ {csv.name}: {e}")

    if not frames:
        raise FileNotFoundError(f"No hay CSVs en {RAW_DATA_DIR}. Corre descargar_todo() primero.")

    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    # Columnas mínimas requeridas
    requeridas = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                  "B365H", "B365D", "B365A"]
    df = df.dropna(subset=requeridas)
    return df.sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    descargar_todo()
    df = cargar_todo()
    print(f"\n📊 Total partidos cargados: {len(df)}")
    print(f"   Ligas: {df['Liga'].unique().tolist()}")
    print(f"   Temporadas: {sorted(df['Temporada'].unique().tolist())}")
    print(f"   Rango fechas: {df['Date'].min().date()} → {df['Date'].max().date()}")
