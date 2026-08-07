#!/usr/bin/env python3
"""
Test rápido de importaciones en app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    print("✓ Importando config...")
    from config import FLASK_DEBUG, FLASK_PORT
    print(f"  FLASK_PORT={FLASK_PORT}, FLASK_DEBUG={FLASK_DEBUG}")

    print("\n✓ Importando engines...")
    from src.engines.football import BettingEngine
    from src.engines.tennis import TennisEngine
    from src.engines.tennis_validator import validar_entrada_tenis
    print("  Engines OK")

    print("\n✓ Importando core modules...")
    from src.core.confidence import calcular_confianza, nivel_confianza
    from src.core.coherence import evaluar_coherencia
    print("  Core modules OK")

    print("\n✓ Importando providers...")
    from src.providers.manager import dsm
    print("  Providers OK")

    print("\n✓ Importando services...")
    from src.services.analyst import analizar, fallback_sin_llm
    from src.services.tracking import (
        calcular_metricas, registrar_apuesta,
        leer_historial, leer_config
    )
    print("  Services OK")

    print("\n✓ Importando Flask...")
    from flask import Flask
    print("  Flask OK")

    print("\n" + "="*60)
    print("✅ Todas las importaciones exitosas!")
    print("="*60)

except Exception as e:
    print(f"\n❌ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
