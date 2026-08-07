#!/usr/bin/env python3
"""
Script para generar equipos_por_liga.json desde los CSVs descargados.
"""
import json
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT = ROOT / "src" / "data" / "equipos_por_liga.json"

# Mapeo de código de liga a nombre
LIGA_NAMES = {
    "SP1": "LaLiga",
    "E0": "Premier League",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

def generar_equipos_liga():
    """Lee CSVs y extrae equipos únicos por liga."""
    equipos_por_liga = defaultdict(set)

    # Buscar todos los CSVs con patrón código_temporada (ej: SP1_2122.csv)
    for csv_file in DATA_DIR.glob("*_*.csv"):
        # Extraer código liga (SP1, E0, D1, etc.)
        codigo = csv_file.name.split("_")[0]
        liga = LIGA_NAMES.get(codigo)

        if not liga:
            continue

        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    home = row.get("HomeTeam", "").strip()
                    away = row.get("AwayTeam", "").strip()
                    if home:
                        equipos_por_liga[liga].add(home)
                    if away:
                        equipos_por_liga[liga].add(away)
        except Exception as e:
            print(f"⚠️  Error leyendo {csv_file.name}: {e}")

    # Convertir sets a listas ordenadas
    resultado = {}
    for liga, equipos in equipos_por_liga.items():
        resultado[liga] = sorted(list(equipos))
        print(f"✅ {liga}: {len(equipos)} equipos")

    # Agregar metadata
    resultado["_meta"] = {
        "total_ligas": len(resultado) - 1,
        "generado": "generar_equipos_liga.py"
    }

    # Guardar JSON
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Archivo guardado: {OUTPUT}")
    print(f"📊 Total: {len(resultado) - 1} ligas cargadas")

if __name__ == "__main__":
    generar_equipos_liga()
