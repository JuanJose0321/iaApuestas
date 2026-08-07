#!/usr/bin/env python3
"""
Test script para verificar integración de football-data.org API.

Prueba:
  1. Lesiones REALES para ligas europeas (football-data.org)
  2. Mock data para ligas latinoamericanas
  3. Que los picks tengan mejor confianza con datos reales
"""
import requests
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
_log = logging.getLogger("test")

API_URL = "http://localhost:5000/chat"

# Test cases: diferentes ligas y tipos de datos
TEST_MATCHES = [
    {
        "name": "Premier League (REAL DATA)",
        "data": {
            "home": "Manchester City",
            "away": "Chelsea",
            "liga": "Premier League",
            "cuotas": {"1": 1.45, "X": 4.20, "2": 7.50},
            "promedio": 2.80
        }
    },
    {
        "name": "LaLiga (REAL DATA)",
        "data": {
            "home": "Real Madrid",
            "away": "Barcelona",
            "liga": "LaLiga",
            "cuotas": {"1": 1.80, "X": 3.80, "2": 4.50},
            "promedio": 2.75
        }
    },
    {
        "name": "Bundesliga (REAL DATA)",
        "data": {
            "home": "Bayern Munich",
            "away": "Borussia Dortmund",
            "liga": "Bundesliga",
            "cuotas": {"1": 1.65, "X": 4.00, "2": 5.50},
            "promedio": 2.80
        }
    },
    {
        "name": "Liga MX (MOCK DATA)",
        "data": {
            "home": "UNAM",
            "away": "Guadalajara",
            "liga": "Liga MX",
            "cuotas": {"1": 1.95, "X": 3.50, "2": 3.80},
            "promedio": 2.60
        }
    },
    {
        "name": "Brasileirao (MOCK DATA)",
        "data": {
            "home": "Flamengo",
            "away": "Palmeiras",
            "liga": "Brasileirao",
            "cuotas": {"1": 2.10, "X": 3.40, "2": 3.60},
            "promedio": 2.65
        }
    },
    {
        "name": "Liga Profesional Argentina (MOCK DATA)",
        "data": {
            "home": "River Plate",
            "away": "Boca Juniors",
            "liga": "Liga Profesional Argentina",
            "cuotas": {"1": 1.90, "X": 3.60, "2": 4.00},
            "promedio": 2.70
        }
    }
]


def test_match(match_info: dict) -> dict:
    """Envía un partido para análisis y retorna el resultado."""
    _log.info(f"\n{'='*80}")
    _log.info(f"🧪 TEST: {match_info['name']}")
    _log.info(f"{'='*80}")

    match_data = match_info["data"]
    _log.info(f"📋 Datos: {match_data['home']} vs {match_data['away']} ({match_data['liga']})")

    try:
        response = requests.post(API_URL, json=match_data, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Extraer información clave
        debug_info = result.get("debug_filtrado", {})
        picks = result.get("picks", [])

        _log.info(f"✅ Respuesta recibida")
        _log.info(f"   Status: HTTP {response.status_code}")
        _log.info(f"   Fuentes usadas: {debug_info.get('fuentes_api', [])}")
        _log.info(f"   Lesiones home: {debug_info.get('injuries_home_count', 0)}")
        _log.info(f"   Lesiones away: {debug_info.get('injuries_away_count', 0)}")

        # Mostrar picks generados
        if picks:
            for pick in picks[:3]:  # Primeros 3 picks
                _log.info(f"\n   🎯 PICK:")
                _log.info(f"      Tipo: {pick.get('tipo')}")
                _log.info(f"      Legs: {pick.get('legs')}")
                _log.info(f"      Cuota: {pick.get('cuota_pick'):.2f}")
                _log.info(f"      Prob: {pick.get('prob_pick'):.2%}")
                _log.info(f"      EV: {pick.get('ev_pick'):.2%}")
                _log.info(f"      Confianza: {pick.get('confianza_pick'):.2%}")
                _log.info(f"      Nivel: {pick.get('nivel_confianza_pick')}")
        else:
            _log.warning(f"⚠️  Sin picks generados")

        return {
            "success": True,
            "fuentes": debug_info.get("fuentes_api", []),
            "lesiones": {
                "home": debug_info.get("injuries_home_count", 0),
                "away": debug_info.get("injuries_away_count", 0)
            },
            "picks_count": len(picks),
            "picks": picks[:1] if picks else []
        }

    except requests.exceptions.RequestException as e:
        _log.error(f"❌ Error en request: {e}")
        return {"success": False, "error": str(e)}

    except json.JSONDecodeError as e:
        _log.error(f"❌ Error parsing JSON: {e}")
        return {"success": False, "error": str(e)}

    except Exception as e:
        _log.error(f"❌ Error inesperado: {e}")
        return {"success": False, "error": str(e)}


def main():
    _log.info("\n" + "="*80)
    _log.info("🔬 PRUEBA DE INTEGRACIÓN: football-data.org API")
    _log.info("="*80)

    results = []
    real_data_count = 0
    mock_data_count = 0
    total_picks = 0

    for match_info in TEST_MATCHES:
        result = test_match(match_info)
        results.append({
            "match": match_info["name"],
            "result": result
        })

        if result["success"]:
            total_picks += result["picks_count"]
            if "REAL DATA" in match_info["name"]:
                real_data_count += 1
            else:
                mock_data_count += 1

    # Resumen
    _log.info(f"\n{'='*80}")
    _log.info("📊 RESUMEN DE PRUEBAS")
    _log.info(f"{'='*80}")

    successful = sum(1 for r in results if r["result"]["success"])
    failed = len(results) - successful

    _log.info(f"✅ Exitosas: {successful}/{len(results)}")
    _log.info(f"❌ Fallidas: {failed}/{len(results)}")
    _log.info(f"🎯 Picks generados: {total_picks}")
    _log.info(f"📍 Ligas con datos REALES (football-data.org): {real_data_count}")
    _log.info(f"📍 Ligas con mock data (latinoamericanas): {mock_data_count}")

    # Detalles por tipo de liga
    _log.info(f"\n{'─'*80}")
    _log.info("LIGAS CON DATOS REALES (football-data.org):")
    for r in results:
        if "REAL DATA" in r["match"] and r["result"]["success"]:
            _log.info(
                f"  • {r['match']:40s} → "
                f"Fuentes: {r['result']['fuentes']} | "
                f"Lesiones: {r['result']['lesiones']['home']}+{r['result']['lesiones']['away']} | "
                f"Picks: {r['result']['picks_count']}"
            )

    _log.info(f"\n{'─'*80}")
    _log.info("LIGAS CON MOCK DATA (Latinoamericanas):")
    for r in results:
        if "MOCK DATA" in r["match"] and r["result"]["success"]:
            _log.info(
                f"  • {r['match']:40s} → "
                f"Fuentes: {r['result']['fuentes']} | "
                f"Lesiones: {r['result']['lesiones']['home']}+{r['result']['lesiones']['away']} | "
                f"Picks: {r['result']['picks_count']}"
            )

    _log.info(f"\n{'='*80}")
    _log.info("✅ Prueba completada")
    _log.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
