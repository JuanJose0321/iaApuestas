# Scripts de diagnóstico

Scripts manuales para probar partes del sistema a mano — no son tests de
pytest: usan `print()` en vez de `assert`, y varios requieren un servidor
Flask corriendo en `localhost:5000` o llaves de API externas configuradas
en `.env` (Groq, Gemini). Se ejecutan directamente:

```bash
python diagnostics/test_simple.py        # requiere `python app.py` corriendo aparte
python diagnostics/test_api_teams.py
python diagnostics/test_app_imports.py   # smoke test de imports, no requiere servidor
```

Los tests automatizados (con `assert`, sin dependencias externas) están en `tests/` y corren con `pytest`.
