"""
Diagnóstico de qué modelos de Google Gemini tiene acceso tu API key.
Corre: python test_modelos.py
"""
import google.generativeai as genai
from config import GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    print("❌ Falta GOOGLE_API_KEY en tu .env")
    raise SystemExit(1)

genai.configure(api_key=GOOGLE_API_KEY)

print("--- Modelos Gemini disponibles ---")
try:
    encontrado = False
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"✅ {m.name.replace('models/', '')}")
            encontrado = True
    if not encontrado:
        print("❌ Tu llave no tiene acceso a ningún modelo de texto.")
except Exception as e:
    print(f"❌ Error al consultar la API: {e}")
