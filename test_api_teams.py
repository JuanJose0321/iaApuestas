#!/usr/bin/env python3
"""
Test para verificar qué devuelve /api/teams
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.providers.league_manager import get_teams

# Probar cada liga
ligas = ["LaLiga", "Premier League", "Bundesliga", "Serie A", "Ligue 1"]

print("="*60)
print("PROBANDO get_teams() directamente")
print("="*60)

for liga in ligas:
    equipos = get_teams(liga)
    print(f"\n🔍 {liga}:")
    print(f"   Total: {len(equipos)} equipos")
    if equipos:
        print(f"   Primeros 5: {equipos[:5]}")
    else:
        print(f"   ❌ SIN EQUIPOS")

# Verificar archivo JSON
print("\n" + "="*60)
print("VERIFICANDO equipos_por_liga.json")
print("="*60)

json_path = Path(__file__).parent / "src" / "data" / "equipos_por_liga.json"
print(f"\nArchivo: {json_path}")
print(f"Existe: {json_path.exists()}")

if json_path.exists():
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for liga in ligas:
        equipos_json = data.get(liga, [])
        print(f"\n📄 {liga} en JSON: {len(equipos_json)} equipos")
        if equipos_json:
            print(f"   Primeros 5: {equipos_json[:5]}")
