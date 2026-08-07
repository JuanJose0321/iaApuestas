# ▶️ CÓMO EJECUTAR EL MOTOR AHORA

**Tu sistema está 100% listo.** Solo necesitas estos 3 pasos en tu máquina local.

---

## 🔧 Paso 1: Verificar que todo está instalado (1 minuto)

Abre PowerShell/Terminal en tu carpeta `iaApuestas` y ejecuta:

```bash
python diagnostic.py
```

**Te dirá:** ✅ Si todo está listo o ❌ qué falta instalar

Si ves errores como `ModuleNotFoundError`:
```bash
pip install joblib numpy pandas scikit-learn xgboost requests
```

---

## 🎯 Paso 2: Generar Predicciones (AHORA MISMO)

Una vez que el diagnóstico dice ✅:

```bash
python src/run_predictions.py
```

**Esto te mostrará:**
- 📊 Todos los picks válidos (ordenados por EV)
- 💰 Confianza y cuota para cada uno
- 📈 Estadísticas agregadas

---

## 📋 Paso 3 (Opcional): Registrar para validación

Si quieres registrar los picks para validar después:

```bash
python src/prediction_tracker.py --generate
```

Luego cuando jueguen:
```bash
python src/prediction_tracker.py --stats
```

---

## 📝 Scripts Disponibles

| Script | Función |
|--------|---------|
| `diagnostic.py` | ✅ Verifica dependencias |
| `src/run_predictions.py` | 🎯 Genera picks (principal) |
| `src/prediction_tracker.py` | 📊 Tracking y validación |
| `src/fetch_leagues.py` | 🌍 Actualizar equipos (semanal) |

---

## ⚡ TL;DR (Solo 2 comandos)

```bash
# 1. Verificar
python diagnostic.py

# 2. Generar picks
python src/run_predictions.py
```

¡Eso es! 🔥

---

## 🐛 Troubleshooting

### Error: ModuleNotFoundError: joblib
```bash
pip install joblib
```

### Error: No module named 'src'
Asegúrate de estar en la carpeta `iaApuestas` cuando ejecutas los comandos.

### Error: Permission denied
En Linux/Mac:
```bash
chmod +x *.py src/*.py
python diagnostic.py
```

---

## ✨ Qué Esperar

Cuando ejecutes `python src/run_predictions.py`:

```
================================================================================
🔥 GENERADOR DE PREDICCIONES - BetBrain
================================================================================
Timestamp: 2026-04-23T...

✅ Motor listo

================================================================================
PICKS RECOMENDADOS (ordenados por EV)
================================================================================

 1. Real Madrid vs Barcelona
    Pronóstico: 1X2 1
    Cuota: 2.10 | Confianza: 65.2% | EV: 0.125

 2. Manchester City vs Liverpool
    Pronóstico: OU_2.5 Over
    Cuota: 1.85 | Confianza: 58.3% | EV: 0.092

... (más picks)

================================================================================
ESTADÍSTICAS
================================================================================
Picks válidos: 7
Cuota promedio: 2.05
Confianza promedio: 62.1%
EV promedio: 0.108

================================================================================
✅ PREDICCIONES LISTAS PARA OPERAR
================================================================================
```

---

## 🎓 Entendiendo la Salida

- **Pronóstico:** Qué apostar (mercado + selección)
- **Cuota:** Cuota ofrecida (debes verificar en tu bookie)
- **Confianza:** Probabilidad del modelo (0-100%)
- **EV:** Valor esperado (>0.08 es bueno)

Si EV > 0.08, el pick tiene valor matemático.

---

## 📊 Próximos Pasos

1. ✅ Ejecuta los 2 comandos arriba
2. ✅ Registra los picks que te interesen
3. ✅ Después que jueguen, valida resultados
4. ✅ Acumula 30-50 picks para confirmar estadísticamente

---

**¿Listo?**

```bash
python diagnostic.py
```

Luego:

```bash
python src/run_predictions.py
```

🔥🚀
