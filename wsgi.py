"""
Entry point de producción — sirve BetBrain con waitress en vez del
servidor de desarrollo de Flask (app.run()).

Uso:
    python wsgi.py
"""
from waitress import serve

from app import app, _log
from config import FLASK_PORT

if __name__ == "__main__":
    _log.info("=== BetBrain (waitress) arrancando en puerto %d ===", FLASK_PORT)
    serve(app, host="127.0.0.1", port=FLASK_PORT)
