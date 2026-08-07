#!/usr/bin/env python3
"""
Script para probar la API /api/teams directamente
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.league_manager import get_league_manager

print("\n" + "="*80)
print("🔍 TEST API /api/teams")
print("="*80 + "\n")

manager = get_league_manager()

# Ligas a probar
ligas_test = [
    "Premier League",
    "Championship",
    "LaLiga",
    "Bundesliga",
]

for liga in ligas_test:
    equipos = manager.get_teams(liga)
    print(f"\n{liga}:")
    print(f"  Total: {len(equipos)} equipos")
    print(f"  Primeros 5:")
    for eq in equipos[:5]:
        print(f"    • {eq}")
    print(f"  Últimos 3:")
    for eq in equipos[-3:]:
        print(f"    • {eq}")

# Verificar si hay equipos duplicados entre ligas
print("\n" + "="*80)
print("🔍 BÚSQUEDA DE DUPLICADOS")
print("="*80 + "\n")

todas_ligas = manager.listar_ligas()
print(f"Ligas disponibles: {len(todas_ligas)}")
print(f"  {', '.join(sorted(todas_ligas))}\n")

# Buscar Ipswich y Luton
print("Equipos críticos:")
for eq_nombre in ["Ipswich", "Luton", "Sunderland"]:
    encontrado_en = []
    for liga in todas_ligas:
        equipos = manager.get_teams(liga)
        if eq_nombre in equipos:
            encontrado_en.append(liga)

    if encontrado_en:
        print(f"  {eq_nombre}: {', '.join(encontrado_en)}")
    else:
        print(f"  {eq_nombre}: NO ENCONTRADO")

print("\n" + "="*80)
