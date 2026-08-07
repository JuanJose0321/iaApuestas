# 🚀 QUICK START - Generar Predicciones

**Tu sistema está 100% listo.** Solo necesitas ejecutar 2 comandos.

---

## ✅ Requisitos Cumplidos
- ✅ Joblib instalado (1.5.3)
- ✅ Motor XGBoost cargado
- ✅ 14 ligas disponibles (300+ equipos)
- ✅ Sistema de fallback activo
- ✅ Datos frescos garantizados

---

## 🎯 Paso 1: Generar Predicciones (ahora mismo)

Abre terminal en tu carpeta `iaApuestas` y ejecuta:

```bash
python src/run_predictions.py
```

**Esto te mostrará:**
- ✅ Todos los picks válidos ordenados por EV
- ✅ Confianza, cuota, y valor esperado para cada uno
- ✅ Estadísticas agregadas

### Ejemplos de Uso

```bash
# Solo LaLiga y Premier
python src/run_predictions.py --ligas LaLiga "Premier League"

# EV mínimo de 0.10 (más restrictivo)
python src/run_predictions.py --min-ev 0.10

# Guardar resultado a JSON
python src/run_predictions.py --output mis_picks.json

# Modo verboso (más detalles)
python src/run_predictions.py --verbose
```

---

## 📊 Paso 2: Registrar Tus Picks (para validar)

Una vez que ves los picks y quieres registrarlos para seguimiento:

```bash
python src/prediction_tracker.py --generate
```

**Esto:**
1. Genera las predicciones (iguales al paso 1)
2. Las registra en `data/prediction_tracking.json` con timestamp
3. Crea IDs únicos para cada pick

---

## ✅ Paso 3: Después que Juegue la Apuesta

Cuando tengas el resultado, registra si ganó o perdió:

```bash
python src/prediction_tracker.py --result "PICK_ID" --outcome won
python src/prediction_tracker.py --result "PICK_ID" --outcome lost
```

Ejemplo real:
```bash
python src/prediction_tracker.py --result "2026-04-23T15:30:45.123456_001" --outcome won
python src/prediction_tracker.py --result "2026-04-23T15:30:45.123456_002" --outcome lost --notas "Equipo suspendido"
```

---

## 📈 Paso 4: Ver Tus Estadísticas

Después de registrar varios picks:

```bash
python src/prediction_tracker.py --stats
```

**Te mostrará:**
- Total de predicciones
- Win rate actual
- Cuota promedio
- EV promedio
- ROI estimado

---

## 📋 Ver Pendientes

Para ver qué picks aún no han jugado:

```bash
python src/prediction_tracker.py --pending
```

---

## 🔄 Actualizar Datos (semanal)

Cada semana, actualiza los equipos desde TheSportsDB:

```bash
python src/fetch_leagues.py
```

---

## 🗂️ Archivos de Salida

- **Predicciones guardadas**: `mis_picks.json`
- **Tracking de picks**: `data/prediction_tracking.json`
- **Datos de equipos**: `data/equipos_por_liga.json`

---

## ⚡ TL;DR

```bash
# Hoy: Ver predicciones
python src/run_predictions.py

# Registrar para follow-up
python src/prediction_tracker.py --generate

# Cuando jueguen: Registrar resultado
python src/prediction_tracker.py --result "ID" --outcome won

# Próxima semana: Actualizar equipos
python src/fetch_leagues.py
```

---

## 📞 Troubleshooting

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: joblib` | `pip install joblib` |
| `No hay predicciones` | Verifica que hay partidos en tu dataset |
| `TheSportsDB timeout` | Sistema automáticamente usa JSON local |
| `Permission denied` | En Linux/Mac: `chmod +x src/*.py` |

---

## 🎯 Tu Objetivo

1. **Genera predicciones diarias** con `run_predictions.py`
2. **Registra picks** con `prediction_tracker.py --generate`
3. **Valida resultados** cuando jueguen
4. **Alcanza 30-50 picks** para validar estadísticamente

Si mantienes 52-55%+ win rate, el motor probó ser rentable. ✅

---

**¿Listo? Ejecuta ahora:**
```bash
python src/run_predictions.py
```

**¡Sistema 100% operacional!** 🔥
