# 🏆 Sistema de Ligas - BetBrain

## Arquitectura (Diciembre 2026)

```
┌─────────────────────────┐
│   Frontend (Dropdown)   │
└────────────┬────────────┘
             │
             ▼
     ┌──────────────────┐
     │  /api/teams      │
     │  (Flask route)   │
     └────────┬─────────┘
              │
              ▼
    ┌──────────────────────────────────────┐
    │  league_manager (Sistema Híbrido)   │
    │                                      │
    │  1️⃣  Intenta TheSportsDB (tiempo real)
    │  2️⃣  Si falla → JSON local (respaldo)
    │                                      │
    └────────┬──────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
  TheSportsDB   JSON local
  (Internet)    (Archivo)
  FRESCO        RESPALDO
```

---

## 📋 Cómo Funciona

### Modo Normal (Usuario Final)
```
Cuando hace clic en dropdown de equipos:
  1. App intenta obtener de TheSportsDB (datos FRESCOS)
  2. Si TheSportsDB no responde → Usa JSON local (datos RESPALDO)
  3. ¡NUNCA falla!
```

### Mantenimiento (Tú - Mensual)
```bash
python src/fetch_leagues.py
```
Esto actualiza el JSON con todos los equipos reales de TheSportsDB.

---

## 🚀 Comandos Principales

### 1. Ver ligas disponibles
```bash
python -c "from src.league_manager import listar_ligas; print('\n'.join(listar_ligas()))"
```

### 2. Ver equipos de una liga
```bash
python -c "from src.league_manager import get_teams; print('\n'.join(get_teams('LaLiga')))"
```

### 3. Actualizar datos (mensual)
```bash
python src/fetch_leagues.py
```

### 4. Actualizar ligas específicas
```bash
python src/fetch_leagues.py --ligas LaLiga "Premier League" Bundesliga
```

---

## 📊 Estado Actual

- **Ligas activas:** 14
- **Equipos totales:** 300+
- **Fuente primaria:** TheSportsDB (tiempo real)
- **Respaldo:** JSON local (`data/equipos_por_liga.json`)
- **Argentina:** ✅ Agregada

---

## ✅ Garantías

✅ **Nunca falla** - Siempre hay datos disponibles  
✅ **Siempre fresco** - Intenta obtener en tiempo real  
✅ **Rápido** - Respaldo local es instantáneo  
✅ **Automático** - No requiere intervención manual  
✅ **Fácil de mantener** - 1 comando mensual  

---

## 🔧 Próximos Pasos

1. Ejecuta `python src/fetch_leagues.py` en tu máquina (ahora)
2. Cada mes, ejecuta el mismo comando para actualizar
3. ¡Listo! El sistema funciona automáticamente

Preguntas: revisa `src/league_manager.py` o `src/fetch_leagues.py`
