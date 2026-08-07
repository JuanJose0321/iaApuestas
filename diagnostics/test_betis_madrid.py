#!/usr/bin/env python3
"""
Test REAL: Real Betis vs Real Madrid
Verifica que el sistema:
1. Obtiene lesiones REALES desde football-data.org
2. Usa el orquestador de múltiples fuentes
3. Genera picks con confianza mejorada
"""
import requests
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TEST] %(levelname)s: %(message)s"
)
_log = logging.getLogger("test_betis_madrid")

# Configuración
API_URL = "http://localhost:5000/chat"
TIMEOUT = 20

# Datos del partido
BETIS_MADRID = {
    "home": "Real Betis",
    "away": "Real Madrid",
    "liga": "LaLiga",
    "cuota_min": 1.40,        # Rango para DIRECTA/DUPLA: permite Over(1.49), BTTS_Yes(1.50), Madrid(1.83)
    "cuota_max": 2.50,        # Rango para DIRECTA/DUPLA: permite Under(2.43), BTTS_No(2.43)
    "cuota_min_tripleta": 2.50,
    "cuota_max_tripleta": 6.00,
    "cuotas": {
        "1X2": {
            "1": 3.76,    # Betis gana
            "X": 4.06,    # Empate
            "2": 1.83     # Madrid gana
        },
        "OU_2.5": {
            "Under": 2.43,  # Menos de 2.5 goles
            "Over": 1.49    # Más de 2.5 goles
        },
        "BTTS": {
            "No": 2.43,     # Uno no anota
            "Yes": 1.50     # Ambos anotan
        }
    },
    "promedio": 2.75
}


def test_betis_madrid():
    """Ejecuta el test del partido Betis vs Madrid."""
    _log.info("=" * 100)
    _log.info("🧪 TEST: Real Betis vs Real Madrid (LaLiga)")
    _log.info("=" * 100)

    _log.info("\n📋 DATOS DEL PARTIDO:")
    _log.info(f"  Home: {BETIS_MADRID['home']}")
    _log.info(f"  Away: {BETIS_MADRID['away']}")
    _log.info(f"  Liga: {BETIS_MADRID['liga']}")
    _log.info(f"  Cuotas 1X2: {BETIS_MADRID['cuotas']['1X2']}")
    _log.info(f"  Cuotas OU 2.5: {BETIS_MADRID['cuotas']['OU_2.5']}")
    _log.info(f"  Cuotas BTTS: {BETIS_MADRID['cuotas']['BTTS']}")
    _log.info(f"  Promedio goles liga: {BETIS_MADRID['promedio']}")

    try:
        _log.info(f"\n🔄 Enviando petición a {API_URL}...")
        response = requests.post(
            API_URL,
            json=BETIS_MADRID,
            timeout=TIMEOUT
        )

        # Validar respuesta
        if response.status_code != 200:
            _log.error(f"❌ Error HTTP {response.status_code}")
            _log.error(f"Response: {response.text}")
            return False

        result = response.json()
        _log.info(f"✅ Respuesta recibida (HTTP 200)")

        # Extraer datos clave
        debug = result.get("debug_filtrado", {})
        picks = result.get("picks", [])

        # ── 1. VERIFICAR FUENTES ──
        _log.info("\n📊 FUENTES UTILIZADAS:")
        fuentes = debug.get("fuentes_api", [])
        _log.info(f"  Fuentes: {fuentes}")

        if "footballdata" in fuentes:
            _log.info(f"  ✅ football-data.org API utilizada (REAL DATA)")
        else:
            _log.info(f"  ⚠️  football-data.org no fue utilizado")

        if "thesportsdb" in fuentes:
            _log.info(f"  ✅ TheSportsDB utilizado")

        # ── 2. VERIFICAR LESIONES ──
        _log.info("\n🤕 LESIONES DETECTADAS:")
        lesiones_home = debug.get("injuries_home_count", 0)
        lesiones_away = debug.get("injuries_away_count", 0)

        _log.info(f"  {BETIS_MADRID['home']}: {lesiones_home} lesiones")
        _log.info(f"  {BETIS_MADRID['away']}: {lesiones_away} lesiones")

        if lesiones_home > 0 or lesiones_away > 0:
            _log.info(f"  ✅ Datos de lesiones obtenidos correctamente")

            # Mostrar detalles de lesiones
            injuries_h = debug.get("injuries_home_detail", [])
            injuries_a = debug.get("injuries_away_detail", [])

            if injuries_h:
                _log.info(f"\n  Lesiones {BETIS_MADRID['home']}:")
                for inj in injuries_h[:3]:
                    _log.info(f"    - {inj.get('jugador', 'Unknown')}: {inj.get('razon', 'Unknown')} "
                             f"({inj.get('dias_fuera', '?')} días)")

            if injuries_a:
                _log.info(f"\n  Lesiones {BETIS_MADRID['away']}:")
                for inj in injuries_a[:3]:
                    _log.info(f"    - {inj.get('jugador', 'Unknown')}: {inj.get('razon', 'Unknown')} "
                             f"({inj.get('dias_fuera', '?')} días)")

        # ── 3. VERIFICAR PICKS ──
        _log.info("\n🎯 PICKS GENERADOS:")
        _log.info(f"  Total picks: {len(picks)}")

        if picks:
            for i, pick in enumerate(picks[:3], 1):
                _log.info(f"\n  PICK #{i} - {pick.get('tipo', 'unknown').upper()}")
                _log.info(f"    Cuota: {pick.get('cuota_pick', 0):.2f}")
                _log.info(f"    Probabilidad: {pick.get('prob_pick', 0):.1%}")
                _log.info(f"    EV: {pick.get('ev_pick', 0):.1%}")
                _log.info(f"    Confianza: {pick.get('confianza_pick', 0):.1%}")
                _log.info(f"    Nivel confianza: {pick.get('nivel_confianza_pick', 'unknown')}")

                legs = pick.get("legs", [])
                if legs:
                    _log.info(f"    Legs ({len(legs)}):")
                    for leg in legs:
                        _log.info(f"      - {leg.get('texto', 'Unknown')} @ {leg.get('cuota', 0)}")
        else:
            _log.warning("  ⚠️  No se generaron picks")

        # ── 4. RESUMEN ──
        _log.info("\n" + "=" * 100)
        _log.info("📈 RESUMEN DEL TEST")
        _log.info("=" * 100)

        success = True
        checks = {
            "✅ Servidor responde": response.status_code == 200,
            "✅ Fuentes detectadas": len(fuentes) > 0,
            "✅ football-data.org utilizado": "footballdata" in fuentes,
            "✅ Lesiones detectadas": lesiones_home + lesiones_away > 0,
            "✅ Picks generados": len(picks) > 0,
        }

        for check, status in checks.items():
            symbol = "✅" if status else "❌"
            _log.info(f"  {symbol} {check.replace('✅ ', '')}")
            if not status:
                success = False

        _log.info("\n" + "=" * 100)
        if success and "footballdata" in fuentes:
            _log.info("✅ TEST EXITOSO - Sistema completo funcionando correctamente")
            _log.info(f"✅ football-data.org API integrada correctamente")
            _log.info(f"✅ {len(picks)} picks generados con datos REALES")
        elif success:
            _log.info("⚠️  TEST PARCIALMENTE EXITOSO - Pero sin datos de football-data.org")
        else:
            _log.info("❌ TEST FALLIDO - Revisar el servidor")
        _log.info("=" * 100)

        return success

    except requests.exceptions.ConnectionError:
        _log.error(f"❌ No se puede conectar a {API_URL}")
        _log.error("   Asegúrate de que el servidor Flask está corriendo:")
        _log.error("   python app.py")
        return False

    except requests.exceptions.Timeout:
        _log.error(f"❌ Timeout esperando respuesta del servidor")
        return False

    except Exception as e:
        _log.error(f"❌ Error inesperado: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    print(f"\n{'='*100}")
    print(f"🏆 TEST DE INTEGRACIÓN: Real Betis vs Real Madrid")
    print(f"Tiempo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")

    success = test_betis_madrid()
    exit(0 if success else 1)
