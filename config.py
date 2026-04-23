"""
Configuración central del proyecto.
Lee variables de entorno desde .env si existe.
"""
import os
from pathlib import Path

# Cargar .env si python-dotenv está disponible (no es obligatorio)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Sin dotenv, usamos solo variables de entorno del sistema

# ---- Paths del proyecto ----
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
NBA_DIR = ROOT_DIR / "nba"

# Crear si no existen
for d in (RAW_DATA_DIR, MODELS_DIR, NBA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- API keys (NUNCA hardcodear) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# API-Football (api-football.com) — free tier: 100 req/día
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_HOST = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")
# Sportmonks (sportmonks.com) — fuente secundaria opcional, solo fallback
SPORTMONKS_TOKEN = os.getenv("SPORTMONKS_TOKEN", "")

# Caché de API-Football en disco (evita quemar quota)
CACHE_DIR = ROOT_DIR / "data" / "api_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HORAS = int(os.getenv("CACHE_TTL_HORAS", "12"))

# ---- Flask ----
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

# ---- Bankroll y reglas de apuesta ----
BANKROLL_INICIAL = float(os.getenv("BANKROLL_INICIAL", "1000"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))   # Kelly fraccional 1/4
MIN_EV = float(os.getenv("MIN_EV", "0.05"))                    # 5% EV mínimo para apostar

# ---- Modelo ----
MODEL_PATH = MODELS_DIR / "betting_model.pkl"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.pkl"

# ---- Promedios de goles por liga (para estimar lambdas con mayor precisión) ----
PROMEDIO_GOLES_LIGA = {
    "LaLiga": 2.5,
    "Premier League": 2.8,
    "Bundesliga": 3.1,
    "Serie A": 2.6,
    "Ligue 1": 2.7,
    "Liga MX": 2.9,
    "MLS": 3.0,
    "Brasileirao": 2.4,
    "Champions League": 2.8,
    "Europa League": 2.7,
    "Eredivisie": 3.2,
    "Primeira Liga": 2.5,
    "Championship": 2.5,
    "Default": 2.6,
}

# ---- Datos: ligas y temporadas a descargar ----
# Códigos de football-data.co.uk:
#   E0=Premier, E1=Champ, SP1=LaLiga, SP2=LaLiga2,
#   D1=Bundesliga, I1=SerieA, F1=Ligue1
LIGAS = ["SP1", "E0", "D1", "I1", "F1"]
TEMPORADAS = ["2122", "2223", "2324", "2425"]
