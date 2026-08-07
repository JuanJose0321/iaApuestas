# 🔥 Estado del Motor - BetBrain
**Última actualización:** 2026-04-23

---

## ✅ SISTEMA OPERACIONAL

| Componente | Estado | Detalles |
|-----------|--------|---------|
| **Joblib** | ✅ Instalado | Versión 1.5.3 (instalado en tu máquina) |
| **XGBoost** | ✅ Cargado | Modelo calibrado en `models/xgboost_calibrador.pkl` |
| **Poisson** | ✅ Listo | Generador de matrices de probabilidad |
| **League Manager** | ✅ Activo | Fallback: TheSportsDB → JSON local |
| **Data Loader** | ✅ Funcional | Carga datos desde CSV históricos |
| **Feature Engineering** | ✅ Listo | Estadísticas rolling incorporadas |

---

## 🚀 CÓMO EJECUTAR PREDICCIONES

### 1. Predicción Rápida (5 picks principales)
```bash
python -c "
from src.engine import Betting_Motor
motor = Betting_Motor()
picks = motor.generar_picks()
print(f'✅ {len(picks)} picks generados')
for p in picks[:5]:
    print(f'{p[\"match\"]}: {p[\"forecast\"]} @ {p[\"odds\"]:.2f} (EV: {p[\"ev\"]:.3f})')
"
```

### 2. Predicción Completa (todos los picks)
```bash
python src/run_predictions.py
```

### 3. Con Opciones Avanzadas
```bash
python src/run_predictions.py --ligas "LaLiga" "Premier League" --min-ev 0.05 --min-confidence 0.65
```

---

## 📊 LÓGICA DE FILTRADO (Dual-Filter)

El motor ACEPTA un pick si cumple CUALQUIERA de estas condiciones:

```python
# Condición 1: Alta confianza
if confidence >= 0.70:
    ✅ PICK ACEPTADO

# Condición 2: Alto valor esperado
elif ev >= 0.08:
    ✅ PICK ACEPTADO

# Si no cumple ninguna:
else:
    ❌ PICK RECHAZADO
```

**Explicación:**
- **Confianza >= 70%**: Pick matemáticamente sólido
- **EV >= 0.08**: Oportunidad con valor incluso si confianza es menor
- Esta combinación balancea precisión con aprovechamiento de oportunidades

---

## 📈 ARQUITECTURA DEL MOTOR

```
INPUT: Partidos disponibles
   ↓
   ├─→ [POISSON] Análisis OU/BTTS/AH
   │   └→ Genera matrices de probabilidad
   │
   ├─→ [XGBOOST] Análisis clasificación
   │   └→ Carga modelo desde joblib
   │
   └─→ [COMPARATIVA]
       ├─ prob_modelo vs cuota_usuario
       └─ Calcula EV (Expected Value)
          ↓
      [DUAL-FILTER]
      if confidence >= 0.70 OR ev >= 0.08:
          ✅ PICK VÁLIDO
OUTPUT: Lista ordenada por EV descendente
```

---

## 🔄 CICLO DE ACTUALIZACIÓN RECOMENDADO

| Tarea | Frecuencia | Comando |
|-------|-----------|---------|
| **Actualizar datos de equipos** | Semanal | `python src/fetch_leagues.py` |
| **Generar predicciones** | Diaria | `python src/run_predictions.py` |
| **Re-entrenar modelo** | Mensual | `python src/train_model.py` |
| **Validar resultados** | Diaria | Comparar picks vs resultados reales |

---

## 🎯 PRÓXIMAS PREDICCIONES (Tú)

Basado en tus 4 aciertos previos (100% en muestra pequeña):
1. Ejecuta `python src/run_predictions.py` en tu máquina
2. Registra los picks con fecha/hora
3. Después que jueguen, anota si acertó o falló
4. Necesitas ~30-50 predicciones para validar si el motor mantiene 52-55%+ WR

---

## 📝 NOTAS IMPORTANTES

✅ **Nunca falla**: Sistema tiene respaldo local garantizado  
✅ **Siempre fresco**: Intenta TheSportsDB primero  
✅ **Datos completos**: 14 ligas, 300+ equipos  
✅ **Joblib instalado**: Motor listo en tu máquina  
⚠️  **Beta activo**: Registra predicciones para validar estadísticas  

---

## 🐛 Si algo falla

1. **"ModuleNotFoundError: joblib"**
   ```bash
   pip install joblib
   ```

2. **"TheSportsDB no responde"**
   → Automáticamente usa respaldo JSON local ✅

3. **"Modelo XGBoost corrupto"**
   → Regenerar: `python src/train_model.py`

---

## 📍 Archivos Clave

- `src/engine.py` - Motor principal
- `src/fetch_leagues.py` - Actualizar equipos
- `data/equipos_por_liga.json` - Teams respaldo
- `models/xgboost_calibrador.pkl` - Modelo XGBoost (joblib)
- `config.py` - Parámetros del sistema

**¡Sistema operacional y listo para picks en vivo!** 🚀
