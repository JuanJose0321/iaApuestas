#!/usr/bin/env python3
"""
Simula exactamente lo que hace el frontend JavaScript
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, request, jsonify
from src.providers.league_manager import get_teams

app = Flask(__name__)

@app.route("/api/teams")
def api_teams():
    """Exactamente la misma ruta que en app.py"""
    liga = request.args.get("liga", "").strip()
    print(f"\n🔍 DEBUG: Liga recibida: '{liga}'")

    if not liga or liga == "Default":
        print(f"❌ Liga inválida")
        return jsonify({"error": "Liga no especificada"})

    try:
        equipos = get_teams(liga)
        print(f"✅ Backend devolvió {len(equipos)} equipos")
        if equipos:
            print(f"   Primeros 5: {equipos[:5]}")

        respuesta = {"liga": liga, "equipos": sorted(equipos)}
        print(f"📤 Respuesta JSON:")
        print(json.dumps(respuesta, indent=2, ensure_ascii=False)[:300])
        return jsonify(respuesta)
    except Exception as exc:
        print(f"❌ Error: {exc}")
        return jsonify({"error": str(exc)})

if __name__ == "__main__":
    print("="*60)
    print("TEST: Simulando peticiones del frontend")
    print("="*60)

    with app.test_client() as client:
        # Test 1: LaLiga
        print("\n\n🧪 TEST 1: GET /api/teams?liga=LaLiga")
        print("-"*60)
        r = client.get("/api/teams?liga=LaLiga&t=123")
        print(f"Status: {r.status_code}")
        data = r.get_json()
        print(f"Equipos recibidos: {len(data.get('equipos', []))}")

        # Test 2: Premier League
        print("\n\n🧪 TEST 2: GET /api/teams?liga=Premier League")
        print("-"*60)
        r = client.get("/api/teams?liga=Premier League&t=123")
        print(f"Status: {r.status_code}")
        data = r.get_json()
        print(f"Equipos recibidos: {len(data.get('equipos', []))}")
