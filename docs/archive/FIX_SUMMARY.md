# 🔧 Fix Summary: Picks Descartados Display Issues

## ✅ What Was Fixed

The rejected picks (picks_descartados) were showing "undefined" and "NaN%" values in the frontend because they were missing critical fields.

### **Problem Identified**
Three rejection points in `app.py` were not including the required fields:
- `legs` - The individual bets that make up the pick
- `cuota_total` - The total odds for the combined bet
- `prob` - The calculated probability
- `ev` - The expected value

This caused the frontend to fail when trying to display pick details in the right panel.

---

## 🛠️ Changes Made to `app.py`

### **1. Marginal OU Rejection** (Lines 413-421)
✅ Added fields to picks_descartados:
```python
picks_descartados.append({
    "tipo": etiqueta,
    "motivo": "marginal_ou",
    "detalle": motivo_marginal,
    "legs": [],              # ← NEW
    "cuota_total": 0,        # ← NEW
    "prob": 0,               # ← NEW
    "ev": 0,                 # ← NEW
})
```

### **2. Contradicción Rejection** (Lines 434-444)
✅ Added missing fields with `or` operator pattern:
```python
picks_descartados.append({
    "tipo": etiqueta,
    "motivo": "contradiccion",
    "detalle": fmt["contradicciones"],
    "confianza": fmt["confianza"],
    "umbral": umbral,
    "legs": fmt.get("legs") or [],           # ← NEW
    "cuota_total": fmt.get("cuota_total") or 0,  # ← NEW
    "prob": fmt.get("prob") or 0,            # ← NEW
    "ev": fmt.get("ev") or 0,                # ← NEW
})
```

### **3. Confianza Baja Rejection** (Lines 451-462)
✅ Added missing fields with `or` operator pattern:
```python
picks_descartados.append({
    "tipo": etiqueta,
    "motivo": "confianza_baja",
    "detalle": f"confianza={fmt['confianza']:.4f} < umbral={umbral:.2f} ...",
    "confianza": fmt["confianza"],
    "umbral": umbral,
    "legs": fmt.get("legs") or [],           # ← NEW
    "cuota_total": fmt.get("cuota_total") or 0,  # ← NEW
    "prob": fmt.get("prob") or 0,            # ← NEW
    "ev": fmt.get("ev") or 0,                # ← NEW
})
```

### **4. Debug Logging** (Lines 467-469)
✅ Added detailed logging to verify what data is in each rejected pick:
```python
for dp in picks_descartados:
    _log.info(f"Pick descartado: tipo={dp.get('tipo')} motivo={dp.get('motivo')} legs={len(dp.get('legs', []))} cuota={dp.get('cuota_total')} prob={dp.get('prob')} ev={dp.get('ev')}")
```

---

## 🧪 How to Test

### **Step 1: Restart the Flask server**
```bash
cd C:\Users\regal\iaApuestas
python app.py
```

You should see:
```
=== BetBrain arrancando ===
...
Running on http://localhost:5000
```

### **Step 2: Run the test in another terminal**
```bash
cd C:\Users\regal\iaApuestas
python test_betis_madrid.py
```

### **Step 3: Check the console output**
Look for the debug logs:
```
Pick descartado: tipo=DIRECTA motivo=confianza_baja legs=2 cuota=1.95 prob=0.625 ev=0.187
```

If you see `legs > 0` and numeric values (not None), the fix is working! ✅

---

## 📊 Expected Results

### **In the Flask Server Console:**
```
Pick descartado: tipo=DIRECTA motivo=confianza_baja legs=2 cuota=1.95 prob=0.625 ev=0.187
Pick descartado: tipo=PARLAY_2 motivo=marginal_ou legs=2 cuota=2.43 prob=0.412 ev=0.003
Pick descartado: tipo=TRIPLETA motivo=contradiccion legs=3 cuota=4.50 prob=0.222 ev=-0.150
```

### **In the Frontend (Right Panel):**
When you click on rejected picks, you should now see:
- ✅ Legs displayed (with player names and odds)
- ✅ Cuota total showing numeric value (not "undefined")
- ✅ Probabilidad showing percentage (not "NaN%")
- ✅ EV showing percentage (not "NaN%")
- ✅ Motivo showing why it was rejected

---

## 🔍 Key Technical Details

### **Why `or` operator?**
We use `fmt.get("key") or default` instead of `fmt.get("key", default)` because:
- `dict.get(key, default)` only returns the default if the **key is missing**
- If the key exists with a `None` value, it returns `None` (not the default)
- The `or` operator treats `None` as falsy, so it returns the default

This ensures proper defaults even when the value is explicitly `None`.

### **Data Flow**
1. Motor generates pick with `legs`, `cuota_total`, `prob`, `ev` ✅
2. Pick is rejected for various reasons (marginal_ou, contradiccion, confianza_baja)
3. **Before fix**: Rejected pick lost these fields → frontend showed "undefined"
4. **After fix**: Rejected pick includes these fields → frontend displays properly

---

## 📝 Files Modified
- `app.py`: Lines 413-469 (pick rejection logic and debug logging)

## 📌 No Changes Needed For
- `templates/index.html` - Already handles the fields correctly
- `src/footballdata_api.py` - Still consulting all sources
- `.env` - FOOTBALL_DATA_API_KEY still active
- Database/cache - No changes

---

**Status:** ✅ Ready to test locally
**Next Step:** Restart server → Run test → Verify debug logs show numeric values
