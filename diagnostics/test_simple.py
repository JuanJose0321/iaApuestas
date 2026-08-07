#!/usr/bin/env python3
"""
Test SIMPLE: Real Betis vs Real Madrid
Motor puro SIN lesiones, SIN datos API
Solo: cuotas → probabilidades → picks
"""
import requests
import json

API_URL = "http://localhost:5000/chat"

BETIS_MADRID = {
    "home": "Real Betis",
    "away": "Real Madrid",
    "liga": "LaLiga",
    "cuotas": {
        "1X2": {
            "1": 3.74,    # Betis
            "X": 4.04,    # Empate
            "2": 1.84     # Madrid
        },
        "OU_2.5": {
            "Under": 1.49,
            "Over": 2.43
        },
        "BTTS": {
            "No": 1.5,
            "Yes": 2.43
        }
    },
    "promedio": 2.75
}

print("\n" + "="*100)
print(f"🧪 TEST SIMPLE: Real Betis vs Real Madrid")
print("="*100)

print(f"\n📋 PARTIDO:")
print(f"  {BETIS_MADRID['home']} vs {BETIS_MADRID['away']}")
print(f"  Liga: {BETIS_MADRID['liga']}")

print(f"\n💰 CUOTAS:")
print(f"  1X2: {BETIS_MADRID['cuotas']['1X2']}")
print(f"  OU 2.5: {BETIS_MADRID['cuotas']['OU_2.5']}")
print(f"  BTTS: {BETIS_MADRID['cuotas']['BTTS']}")

try:
    print(f"\n🔄 Enviando petición...")
    response = requests.post(API_URL, json=BETIS_MADRID, timeout=15)

    if response.status_code != 200:
        print(f"❌ Error HTTP {response.status_code}")
        print(response.text)
        exit(1)

    result = response.json()
    picks = result.get("picks", [])

    print(f"\n✅ Respuesta recibida")
    print(f"\n🎯 PICKS GENERADOS: {len(picks)}")

    if not picks:
        print("❌ Sin picks generados")
        print(f"\nDebug: {result.get('debug_filtrado', {})}")
        exit(1)

    for i, pick in enumerate(picks, 1):
        print(f"\n  ──────────────────────────────────────────")
        print(f"  PICK #{i}: {pick.get('tipo', '?').upper()}")
        print(f"  ──────────────────────────────────────────")

        legs = pick.get("legs", [])
        print(f"  Legs ({len(legs)}):")
        for leg in legs:
            print(f"    • {leg['texto']} @ {leg['cuota']}")

        print(f"  Cuota total: {pick.get('cuota_pick', 0):.2f}")
        print(f"  Probabilidad: {pick.get('prob_pick', 0):.1%}")
        print(f"  EV: {pick.get('ev_pick', 0):.1%}")
        print(f"  Confianza: {pick.get('confianza_pick', 0):.1%}")
        print(f"  Nivel: {pick.get('nivel_confianza_pick', '?')}")

    print(f"\n" + "="*100)
    print(f"✅ TEST COMPLETADO - {len(picks)} picks generados por el motor puro")
    print("="*100 + "\n")

except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
