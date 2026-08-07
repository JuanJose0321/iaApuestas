#!/usr/bin/env python3
"""
Script de prueba para el partido: Real Betis vs Real Madrid
Ejecutar: python test_partido.py
"""
import requests
import json
from pprint import pprint

# Configuración
API_URL = "http://127.0.0.1:5000/chat"
TIMEOUT = 30

# Datos del partido
payload = {
    "liga": "LaLiga",
    "home": "Real Betis",
    "away": "Real Madrid",
    "cuotas": {
        "1X2": {
            "1": 3.74,      # Victoria Betis
            "X": 4.04,      # Empate
            "2": 1.84       # Victoria Real Madrid
        },
        "OU_2.5": {
            "O": 1.49,      # Over 2.5
            "U": 2.43       # Under 2.5
        },
        "BTTS": {
            "Si": 1.50,     # Ambos anotan
            "No": 2.43      # No ambos anotan
        }
    }
}

def main():
    print("=" * 80)
    print("🧪 PRUEBA DEL SISTEMA: Real Betis vs Real Madrid")
    print("=" * 80)
    print("\n📊 DATOS ENVIADOS:")
    pprint(payload, width=100)

    try:
        print(f"\n⏳ Conectando a {API_URL}...")
        response = requests.post(API_URL, json=payload, timeout=TIMEOUT)

        print(f"✅ Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print("\n" + "=" * 80)
            print("✅ RESPUESTA EXITOSA")
            print("=" * 80)

            # Datos de la API
            print("\n🔄 DATOS OBTENIDOS DE FUENTES:")
            print("-" * 80)
            if "api_data" in data:
                api = data["api_data"]
                print(f"  ✓ Forma Betis:        {api.get('forma_home')}")
                print(f"  ✓ Forma Real Madrid:  {api.get('forma_away')}")
                print(f"  ✓ H2H:                {api.get('h2h')}")
                print(f"  ✓ Lesiones Betis:     {len(api.get('injuries_home', []))} jugadores")
                print(f"  ✓ Lesiones Real Mad:  {len(api.get('injuries_away', []))} jugadores")
                print(f"  ✓ Fuentes usadas:     {api.get('fuentes_usadas')}")
                print(f"  ✓ API disponible:     {api.get('api_disponible')}")

            # Picks recomendados
            print("\n🎯 PICKS RECOMENDADOS:")
            print("-" * 80)
            if "picks" in data:
                for i, pick in enumerate(data["picks"], 1):
                    print(f"\n  {i}. {pick.get('tipo')}:")
                    print(f"     Predicción:   {pick.get('prediccion')}")
                    print(f"     Cuota:        {pick.get('cuota')}")
                    print(f"     Probabilidad: {pick.get('probabilidad', 'N/A')}")
                    print(f"     EV:           {pick.get('ev', 'N/A')}")
                    print(f"     Confianza:    {pick.get('confianza')}")
                    print(f"     Kelly %:      {pick.get('kelly_pct', 'N/A')}")
                    print(f"     Stake:        ${pick.get('stake_recomendado', 'N/A')}")
            else:
                print("  (Sin picks generados)")

            # Coherencia
            print("\n📈 ANÁLISIS DE COHERENCIA:")
            print("-" * 80)
            if "coherencia" in data:
                coh = data["coherencia"]
                print(f"  • Confianza:  {coh.get('confianza_modelo')}")
                print(f"  • Flags:      {coh.get('flags', [])}")
                if coh.get('mensajes'):
                    for msg in coh['mensajes']:
                        print(f"    → {msg}")

            # Resumen final
            print("\n📝 RESUMEN FINAL:")
            print("-" * 80)
            print(f"  • Total picks:     {len(data.get('picks', []))}")
            print(f"  • Confianza gral:  {data.get('confianza_general', 'N/A')}")
            print(f"  • Recomendación:   {data.get('recomendacion', 'N/A')}")

            print("\n" + "=" * 80)
            print("✅ PRUEBA COMPLETADA")
            print("=" * 80)

        else:
            print(f"\n❌ Error {response.status_code}:")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("\n❌ NO SE PUEDE CONECTAR")
        print("   Asegúrate de que Flask está corriendo:")
        print("   → python app.py")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
