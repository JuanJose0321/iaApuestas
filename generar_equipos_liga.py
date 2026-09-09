#!/usr/bin/env python3
"""
Regenera, dentro de src/data/equipos_por_liga.json, la lista de equipos de
las 5 ligas domésticas que sí tienen histórico real en data/raw/
(LaLiga/SP1, Premier League/E0, Bundesliga/D1, Serie A/I1, Ligue 1/F1).

Toma SOLO la temporada más reciente disponible por liga (ej. "2627"), nunca
la unión de todas las temporadas históricas -- esa unión era el bug de raíz
de la versión anterior de este script: nunca "sacaba" a un equipo ya
descendido, porque una vez que aparecía en algún CSV quedaba para siempre
en el set. Ver commit 971d331 y `futbol_diagnostico_completo.md` para el
contexto de cómo se detectó.

Los CSV de football-data.co.uk usan códigos cortos/abreviados como nombre
de equipo ("Ath Madrid", "Man City", "Nott'm Forest"), no los nombres
"lindos" que ya usa el dropdown del frontend ("Atletico Madrid",
"Manchester City", "Nottingham Forest"). NOMBRES_EQUIPOS traduce cada
código visto en la temporada vigente a ese nombre de display. Si
football-data.co.uk empieza a usar un código nuevo (típicamente: un equipo
recién ascendido que nunca jugó estas 5 ligas antes), el script lo señala
explícitamente en vez de inventar un nombre o fallar en silencio -- hay que
sumar una línea a NOMBRES_EQUIPOS a mano esa única vez.

Solo toca las claves de estas 5 ligas dentro del JSON -- preserva intacto
todo lo demás (Champions League, Europa League, Liga MX, MLS, Brasileirao,
Eredivisie, Primeira Liga, Championship, Liga Profesional Argentina), que
no tienen CSV propio en el repo (ver Paso 3 de la investigación de
2026-09-09: no se encontró una fuente gratuita y confiable para
automatizarlas también).

Uso
---
    python generar_equipos_liga.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT = ROOT / "src" / "data" / "equipos_por_liga.json"

# Código football-data.co.uk -> nombre de la liga en equipos_por_liga.json
LIGA_NAMES = {
    "SP1": "LaLiga",
    "E0": "Premier League",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

# Código de equipo tal como aparece en HomeTeam/AwayTeam del CSV -> nombre
# de display usado en el dropdown. Cubre todos los códigos vistos en la
# temporada 2026-27 (2627) al momento de esta corrección (2026-09-09);
# sumar acá cualquier código nuevo que aparezca en temporadas futuras.
NOMBRES_EQUIPOS = {
    # LaLiga (SP1)
    "Alaves": "Alaves", "Ath Bilbao": "Athletic Club", "Ath Madrid": "Atletico Madrid",
    "Barcelona": "Barcelona", "Betis": "Real Betis", "Celta": "Celta Vigo",
    "Elche": "Elche", "Espanol": "Espanyol", "Getafe": "Getafe",
    "La Coruna": "Deportivo La Coruna", "Levante": "Levante", "Malaga": "Malaga",
    "Osasuna": "Osasuna", "Real Madrid": "Real Madrid", "Santander": "Racing Santander",
    "Sevilla": "Sevilla", "Sociedad": "Real Sociedad", "Valencia": "Valencia",
    "Vallecano": "Rayo Vallecano", "Villarreal": "Villarreal",
    # Premier League (E0)
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton": "Brighton", "Chelsea": "Chelsea",
    "Coventry": "Coventry", "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Hull": "Hull", "Ipswich": "Ipswich", "Leeds": "Leeds",
    "Liverpool": "Liverpool", "Man City": "Manchester City", "Man United": "Manchester United",
    "Newcastle": "Newcastle", "Nott'm Forest": "Nottingham Forest", "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    # Bundesliga (D1)
    "Augsburg": "FC Augsburg", "Bayern Munich": "Bayern Munich", "Dortmund": "Borussia Dortmund",
    "Ein Frankfurt": "Eintracht Frankfurt", "Elversberg": "SV Elversberg",
    "FC Koln": "1. FC Koln", "Freiburg": "SC Freiburg", "Hamburg": "Hamburger SV",
    "Hoffenheim": "1899 Hoffenheim", "Leverkusen": "Bayer Leverkusen",
    "M'gladbach": "Borussia Monchengladbach", "Mainz": "FSV Mainz 05",
    "Paderborn": "SC Paderborn", "RB Leipzig": "RB Leipzig", "Schalke 04": "Schalke 04",
    "Stuttgart": "VfB Stuttgart", "Union Berlin": "Union Berlin", "Werder Bremen": "Werder Bremen",
    # Serie A (I1)
    "Atalanta": "Atalanta", "Bologna": "Bologna", "Cagliari": "Cagliari", "Como": "Como",
    "Fiorentina": "Fiorentina", "Frosinone": "Frosinone", "Genoa": "Genoa", "Inter": "Inter",
    "Juventus": "Juventus", "Lazio": "Lazio", "Lecce": "Lecce", "Milan": "AC Milan",
    "Monza": "Monza", "Napoli": "Napoli", "Parma": "Parma", "Roma": "Roma",
    "Sassuolo": "Sassuolo", "Torino": "Torino", "Udinese": "Udinese", "Venezia": "Venezia",
    # Ligue 1 (F1)
    "Angers": "Angers", "Auxerre": "Auxerre", "Brest": "Brest", "Le Havre": "Le Havre",
    "Le Mans": "Le Mans", "Lens": "Lens", "Lille": "Lille", "Lorient": "Lorient",
    "Lyon": "Lyon", "Marseille": "Marseille", "Monaco": "Monaco", "Nice": "Nice",
    "Paris FC": "Paris FC", "Paris SG": "Paris Saint Germain", "Rennes": "Rennes",
    "Strasbourg": "Strasbourg", "Toulouse": "Toulouse", "Troyes": "Troyes",
}


def _temporada_mas_reciente(codigo_liga: str) -> str | None:
    """Código de temporada más alto (ej. "2627") entre los CSV descargados
    para esta liga. None si no hay ningún archivo."""
    temporadas = []
    for csv_file in DATA_DIR.glob(f"{codigo_liga}_*.csv"):
        temporada = csv_file.stem.split("_")[1]
        if temporada.isdigit():
            temporadas.append(temporada)
    return max(temporadas, key=int) if temporadas else None


def generar_equipos_liga() -> dict:
    """Lee, para cada una de las 5 ligas domésticas, SOLO el CSV de su
    temporada más reciente, y devuelve {liga: sorted(equipos)}. Corta la
    ejecución si encuentra un código de equipo sin traducción conocida en
    NOMBRES_EQUIPOS -- mejor fallar ruidosamente acá que guardar un código
    crudo de football-data.co.uk en el dropdown."""
    resultado: dict[str, list[str]] = {}
    codigos_sin_mapear: dict[str, set[str]] = defaultdict(set)

    for codigo_liga, liga in LIGA_NAMES.items():
        temporada = _temporada_mas_reciente(codigo_liga)
        if temporada is None:
            print(f"⚠️  {liga} ({codigo_liga}): no hay ningún CSV en {DATA_DIR}, se omite")
            continue

        csv_path = DATA_DIR / f"{codigo_liga}_{temporada}.csv"
        df = pl.read_csv(csv_path, encoding="latin-1", ignore_errors=True)
        codigos = set(df["HomeTeam"].drop_nulls().to_list()) | set(df["AwayTeam"].drop_nulls().to_list())

        equipos = set()
        for codigo_equipo in codigos:
            nombre = NOMBRES_EQUIPOS.get(codigo_equipo)
            if nombre is None:
                codigos_sin_mapear[liga].add(codigo_equipo)
            else:
                equipos.add(nombre)

        resultado[liga] = sorted(equipos)
        print(f"✅ {liga} (temporada {temporada}): {len(equipos)} equipos")

    if codigos_sin_mapear:
        print("\n❌ Códigos de equipo sin traducción en NOMBRES_EQUIPOS (no se puede continuar):")
        for liga, codigos in codigos_sin_mapear.items():
            print(f"   {liga}: {sorted(codigos)}")
        print("\nSumá cada código nuevo a NOMBRES_EQUIPOS (nombre de display para el dropdown) y corré de nuevo.")
        sys.exit(1)

    return resultado


def actualizar_json(nuevas_ligas: dict) -> bool:
    """Fusiona nuevas_ligas (solo las 5 claves domésticas) dentro del JSON
    existente, preservando intactas las otras 9 competiciones y el resto de
    _meta. Devuelve True si algo cambió de verdad (para no commitear ruido
    cuando la temporada no tuvo movimientos)."""
    with open(OUTPUT, encoding="utf-8") as f:
        actual = json.load(f)

    cambio = any(actual.get(liga) != equipos for liga, equipos in nuevas_ligas.items())
    if not cambio:
        print("\nSin cambios reales respecto del JSON actual -- no se reescribe el archivo.")
        return False

    actual.update(nuevas_ligas)
    actual["_meta"]["actualizado"] = actual["_meta"].get("actualizado")  # se pisa abajo
    from datetime import date
    actual["_meta"]["actualizado"] = date.today().isoformat()
    actual["_meta"]["ultima_regeneracion_automatica"] = (
        "Las 5 ligas domesticas (LaLiga/Premier League/Bundesliga/Serie A/Ligue 1) se "
        "regeneran automaticamente desde data/raw/*_2627.csv via generar_equipos_liga.py "
        "-- no requieren research manual. El resto de las competiciones de este archivo "
        "siguen curadas a mano (ver 'nota')."
    )

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(actual, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n💾 Actualizado: {OUTPUT}")
    return True


if __name__ == "__main__":
    ligas = generar_equipos_liga()
    cambio = actualizar_json(ligas)
    sys.exit(0)
