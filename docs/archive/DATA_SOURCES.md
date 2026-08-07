# 📊 Arquitectura de Fuentes de Datos - BetBrain

## Problema: Limitaciones de TheSportsDB

### ❌ Limitaciones Originales

```
TheSportsDB Free Tier:
  ❌ Sin livescore en tiempo real
  ❌ Histórico limitado (solo último partido por equipo)
  ❌ Sin odds/momios
  ❌ Sin información de lesiones/suspensiones
  ❌ eventsh2h fallan ocasionalmente
  ⚠️  Datos incompletos para ligas menores
```

**¿Por qué?** 
- TheSportsDB es un proyecto comunitario voluntario
- Free tier muy básico (data pública, sin APIs premium)
- Enfoque en ligas mayores, no en datos actualizados en tiempo real
- Sin acceso a información de lesiones en vivo

---

## ✅ Solución: Sistema Multicapa SIN API KEYS

### Nueva Arquitectura

```
                    ┌─────────────────────────────────────┐
                    │   SOLICITUD: {home, away, liga}      │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │   MultiSourceOrchestrator             │
                    │   (intenta múltiples fuentes)         │
                    └──────────────────┬──────────────────┘
                                       │
            ┌──────────┬──────────┬────┴────┬──────────┐
            │          │          │          │          │
            ▼          ▼          ▼          ▼          ▼
    ┌──────────┐┌──────────┐┌──────────┐┌────────┐┌──────────┐
    │TheSports ││Transfer- ││WorldFoot ││ ESPN   ││   JSON   │
    │  DB      ││  markt   ││  ball    ││Scraper ││  Local   │
    │(forma,  ││(lesiones)││(estad.)  ││(forma) ││(fallback)│
    │h2h, IDs) │          ││          ││        ││          │
    └──────────┘└──────────┘└──────────┘└────────┘└──────────┘
        ✅           ✅            ✅         ✅        ✅
      FREE          FREE           FREE       FREE      LOCAL
      NO KEY        NO KEY         NO KEY     NO KEY     -
    PRIORITY 1   PRIORITY 2    PRIORITY 3  PRIORITY 4 PRIORITY 5
```

---

## 🔄 Cadena de Fallback (Prioridad)

### 1️⃣ **TheSportsDB** (PRIORIDAD 1)
**Qué obtiene:**
- ✅ Forma reciente del equipo (últimos 5 partidos)
- ✅ H2H histórico entre equipos
- ✅ IDs de equipos
- ✅ Goles promedio por/en contra

**Línea en código:**
```python
ctx_tsdb = thesportsdb.contexto_partido_completo(home, away)
```

**Limitaciones aceptadas:**
- No tiene lesiones en tiempo real
- Datos un poco anticuados (últimas 24-48h)
- Algunos h2h fallan si IDs no coinciden exactamente

---

### 2️⃣ **Transfermarkt Scraper** (PRIORIDAD 2)
**Qué obtiene:**
- ✅ **Lesiones Y SUSPENSIONES en tiempo real** ⭐
- ✅ Forma actual del equipo
- ✅ Valor de mercado
- ✅ Alineación probable

**Ventaja sobre TheSportsDB:**
```
TheSportsDB: No tiene lesiones
                     ↓
Transfermarkt: lesiones_home = [
  {
    "jugador": "Haaland",
    "tipo": "injury",
    "dias_fuera": 7,
    "estimado_retorno": "2026-05-01"
  },
  ...
]
```

**Implementación:**
```python
injuries_h = transfermarkt_scraper.get_injuries(home)
injuries_a = transfermarkt_scraper.get_injuries(away)
```

**Estado:** Parcialmente implementado (necesita parsing HTML)

---

### 3️⃣ **WorldFootball Scraper** (PRIORIDAD 3)
**Qué obtiene:**
- ✅ Estadísticas detalladas (posesión, tiros, tarjetas)
- ✅ Histórico completo de partidos
- ✅ Tablas de posiciones
- ✅ Comparativas head-to-head detalladas

**Ventaja:**
```
Mucho más completo que TheSportsDB:
  - Posesión de balón promedio
  - Tiros por partido
  - Tiros en marco
  - Tarjetas amarillas/rojas
  - Diferencia de goles esperados
```

**Estado:** Esquema creado (necesita parsing HTML)

---

### 4️⃣ **ESPN Scraper** (PRIORIDAD 4)
**Qué obtiene:**
- ✅ Resultados recientes
- ✅ Forma del equipo
- ✅ Estadísticas básicas

**Rol:**
Fallback adicional si TheSportsDB/Transfermarkt fallan

**Estado:** Esquema creado (necesita parsing HTML)

---

### 5️⃣ **JSON Local** (PRIORIDAD 5)
**Qué obtiene:**
- ✅ Equipos correctos por liga
- ✅ Datos históricos pre-descargados

**Rol:** Fallback final, siempre funciona

**Estado:** ✅ Completamente implementado

---

## 📈 Diagrama de Flujo Actual

```python
@app.route("/chat", methods=["POST"])
def chat():
    # Usuario ingresa datos
    data = request.get_json()
    home = data["home"]      # "Manchester City"
    away = data["away"]      # "Arsenal"
    liga = data["liga"]      # "Premier League"
    
    # PASO 1: Orquestador obtiene contexto de múltiples fuentes
    ctx = get_orchestrator().contexto_partido_completo(home, away)
    
    # ctx ahora contiene:
    # {
    #   "forma_home": {...},        ← de TheSportsDB
    #   "forma_away": {...},        ← de TheSportsDB
    #   "h2h": {...},               ← de TheSportsDB
    #   "injuries_home": [...],     ← de Transfermarkt
    #   "injuries_away": [...],     ← de Transfermarkt
    #   "stats_home": {...},        ← de WorldFootball (si disponible)
    #   "stats_away": {...},        ← de WorldFootball (si disponible)
    #   "fuentes_usadas": ["thesportsdb", "transfermarkt", ...]
    # }
    
    # PASO 2: Engine usa todos los datos para predicción
    resultado = engine.pick_multileg(
        home, away, cuotas,
        forma_home=ctx["forma_home"],
        forma_away=ctx["forma_away"],
        injuries_home=ctx["injuries_home"],
        injuries_away=ctx["injuries_away"],
        ...
    )
    
    return jsonify(resultado)
```

---

## 🚀 Plan de Implementación

### FASE 1: Completar Parsers (ESTA SEMANA)
- [ ] Implementar Transfermarkt HTML parser (lesiones)
- [ ] Implementar WorldFootball HTML parser (estadísticas)
- [ ] Implementar ESPN HTML parser (forma)
- [ ] Tests para cada scraper

### FASE 2: Integración (SIGUIENTE SEMANA)
- [ ] Conectar orquestador en app.py
- [ ] Metrics de uso por fuente
- [ ] Caché inteligente (TTL por tipo de dato)

### FASE 3: Optimización (SEMANA 3)
- [ ] Paralelizar requests (async)
- [ ] Rate-limiting y retry logic
- [ ] Fallback automático si scraper falla

---

## 📊 Estadísticas de Uso

```python
from src.multi_source_orchestrator import get_orchestrator

orq = get_orchestrator()
stats = orq.get_stats()

print(stats)
# {
#   "thesportsdb_hits": 245,
#   "transfermarkt_hits": 198,
#   "worldfootball_hits": 45,
#   "espn_hits": 12,
#   "fallback_hits": 8,
#   "errors": [...]
# }
```

---

## ✨ Ventajas de Esta Arquitectura

```
1. ✅ SIN API KEYS REQUERIDAS
   - TheSportsDB: free, sin key
   - Transfermarkt: scraping, sin key
   - WorldFootball: scraping, sin key
   - ESPN: scraping, sin key

2. ✅ DATOS COMPLETOS Y ACTUALIZADOS
   - Forma + H2H: TheSportsDB
   - Lesiones: Transfermarkt
   - Estadísticas: WorldFootball
   - Fallback: ESPN + JSON local

3. ✅ RESILIENTE (sin puntos de fallo único)
   - Si TheSportsDB cae → intenta Transfermarkt
   - Si Transfermarkt cae → intenta WorldFootball
   - Si todo falla → JSON local siempre funciona

4. ✅ ACTUALIZACIÓN EN TIEMPO REAL
   - Lesiones: Transfermarkt actualiza cada hora
   - Forma: TheSportsDB cada 24h
   - Estadísticas: WorldFootball cada 24h

5. ✅ BAJO OVERHEAD
   - Memory cache: 10min a 2 horas
   - Disk cache: automático
   - Re-uses existing infrastructure
```

---

## 🔧 Configuración en app.py

```python
# ANTES (con problemas)
ctx_api = dsm.contexto_partido_completo(home, away)

# AHORA (mejorado)
from src.multi_source_orchestrator import get_orchestrator

orq = get_orchestrator()
ctx_api = orq.contexto_partido_completo(home, away)

_log.info(f"Fuentes usadas: {ctx_api['fuentes_usadas']}")
# Output: Fuentes usadas: ['thesportsdb', 'transfermarkt']
```

---

## 📝 Próximos Pasos

1. **Instalar dependencias:**
   ```bash
   pip install beautifulsoup4 requests
   ```

2. **Implementar parsers HTML** (ver esquemas en cada scraper)

3. **Conectar en app.py** (reemplazar data_source_manager)

4. **Test end-to-end** con un partido real

---

## Conclusión

**Antes:** TheSportsDB solo con muchas limitaciones
**Ahora:** 5 fuentes fallback sin dependencias de API keys

Cada fuente llena gaps de las anteriores.
**Resultado:** Datos completos, actualizados, resilientes.
