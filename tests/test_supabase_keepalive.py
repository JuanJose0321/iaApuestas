"""
Tests del keep-alive de Supabase (src/scripts/supabase_keepalive.py).

Todo con mocks: el punto del keep-alive es no depender de nada durante el
test -- si estos tests necesitaran red, fallarían justo cuando Supabase esté
pausado, que es el problema que el script viene a evitar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scripts import supabase_keepalive as ka


def test_ping_ok_imprime_conteo_y_sale_cero(monkeypatch, capsys):
    monkeypatch.setattr(ka._sb, "disponible", lambda: True)
    monkeypatch.setattr(ka._sb, "contar_apuestas", lambda: 208)

    assert ka.main() == 0

    salida = capsys.readouterr().out
    assert "Ping OK" in salida
    assert "208 filas" in salida
    assert "UTC" in salida   # el mensaje lleva fecha/hora


def test_error_de_conexion_sale_distinto_de_cero(monkeypatch, capsys):
    def _explota():
        raise ConnectionError("Name or service not known")

    monkeypatch.setattr(ka._sb, "disponible", lambda: True)
    monkeypatch.setattr(ka._sb, "contar_apuestas", _explota)

    assert ka.main() == 1

    err = capsys.readouterr().err
    assert "Ping FALLO" in err
    assert "ConnectionError" in err
    assert "Name or service not known" in err


def test_sin_credenciales_sale_distinto_de_cero(monkeypatch, capsys):
    monkeypatch.setattr(ka._sb, "disponible", lambda: False)
    # Si igual intentara consultar, el test falla con este error en vez de
    # pasar por casualidad.
    monkeypatch.setattr(
        ka._sb, "contar_apuestas",
        lambda: (_ for _ in ()).throw(AssertionError("no debería consultar sin credenciales")),
    )

    assert ka.main() == 1
    assert "no configurados" in capsys.readouterr().err


def test_el_ping_no_escribe_nada_en_supabase(monkeypatch):
    """El script solo puede tocar `contar_apuestas` (lectura); si algún día
    alguien mete un insert/update/delete acá, este test lo caza."""
    for escritura in ("insertar_apuesta", "actualizar_apuesta", "eliminar_apuesta",
                      "insertar_prediccion", "actualizar_prediccion", "guardar_config"):
        monkeypatch.setattr(
            ka._sb, escritura,
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("el keep-alive no debe escribir")),
        )
    monkeypatch.setattr(ka._sb, "disponible", lambda: True)
    monkeypatch.setattr(ka._sb, "contar_apuestas", lambda: 1)

    assert ka.main() == 0
