"""
Validador de cuotas NBA — Moneyline, Spread, Total Points.

Reglas específicas:
- Moneyline: formato americano (-150, +130)
- Spread: debe terminar en .5 (no permitir empates)
- Total: rango típico 190-230 pts
"""
import logging

_log = logging.getLogger("betbrain.nba")


def validar_entrada_nba(data: dict) -> tuple:
    """
    Valida los campos comunes del análisis NBA.

    Args:
        data: {
            "home": "Lakers",
            "away": "Celtics",
            "cuotas": {
                "moneyline": {"home": -150, "away": 130},
                "spread": {"home": -4.5, "away": 4.5, "cuota": -110},
                "total": {"linea": 210.5, "cuota_over": -110, "cuota_under": -110}
            }
        }

    Returns:
        (home, away, cuotas, error_msg)
        error_msg es None si todo es válido.
    """
    home = (data.get("home") or "").strip()
    away = (data.get("away") or "").strip()
    cuotas = data.get("cuotas") or {}

    if not home:
        return None, None, None, "Falta el equipo local (home)"

    if not away:
        return None, None, None, "Falta el equipo visitante (away)"

    if home.lower() == away.lower():
        return None, None, None, "No puede ser el mismo equipo"

    # Validar que haya al menos un mercado
    if not cuotas:
        return None, None, None, "Falta cuotas para analizar"

    # Validar Moneyline si existe
    if "moneyline" in cuotas:
        err = validar_moneyline(cuotas["moneyline"])
        if err:
            return None, None, None, err

    # Validar Spread si existe
    if "spread" in cuotas:
        err = validar_spread(cuotas["spread"])
        if err:
            return None, None, None, err

    # Validar Total si existe
    if "total" in cuotas:
        err = validar_total(cuotas["total"])
        if err:
            return None, None, None, err

    return home, away, cuotas, None


def validar_moneyline(ml: dict) -> str | None:
    """
    Valida cuotas Moneyline (formato americano).

    Ejemplos válidos:
    - Favorito: -150, -200, -110
    - Underdog: +150, +200, +110
    """
    if not ml:
        return "Moneyline vacío"

    # Debe tener home y away
    if "home" not in ml or "away" not in ml:
        return "Moneyline debe tener home y away"

    home_ml = ml.get("home")
    away_ml = ml.get("away")

    # Validar que sean números
    try:
        home_ml = float(home_ml) if home_ml else None
        away_ml = float(away_ml) if away_ml else None
    except (TypeError, ValueError):
        return "Moneyline debe ser números válidos"

    if home_ml is None or away_ml is None:
        return "Moneyline no puede estar vacío"

    # En NBA, moneyline típicamente está entre -300 y +300
    if abs(home_ml) > 500:
        return f"Moneyline Home {home_ml} muy extremo"

    if abs(away_ml) > 500:
        return f"Moneyline Away {away_ml} muy extremo"

    # Verificar que sean opuestos (lógica de libro)
    # Si home es favorito (-150), away debe ser underdog (+120 aprox)
    home_fav = home_ml < 0
    away_fav = away_ml < 0

    if home_fav and away_fav:
        return "Ambos no pueden ser favoritos en Moneyline"

    if not home_fav and not away_fav:
        return "Ambos no pueden ser underdogs en Moneyline"

    return None


def validar_spread(spread: dict) -> str | None:
    """
    Valida cuotas de Spread.

    Ejemplos válidos:
    - home: -4.5, away: +4.5
    - home: -3.0, away: +3.0 (sin .5 también se permite)
    """
    if not spread:
        return "Spread vacío"

    if "home" not in spread or "away" not in spread:
        return "Spread debe tener home y away"

    home_spread = spread.get("home")
    away_spread = spread.get("away")

    # Validar que sean números
    try:
        home_spread = float(home_spread)
        away_spread = float(away_spread)
    except (TypeError, ValueError):
        return "Spread debe ser números válidos"

    # Spread no debe ser 0 (eso sería una apuesta inválida)
    if home_spread == 0 or away_spread == 0:
        return "Spread no puede ser 0"

    # Spread debe ser opuesto (si home -4.5, away debe ser +4.5)
    if abs(abs(home_spread) - abs(away_spread)) > 0.1:
        return "Spread home y away deben ser opuestos"

    # Rango típico NBA: -15 a +15
    if abs(home_spread) > 20:
        return f"Spread {home_spread} muy extremo para NBA"

    # Preferencia: debe terminar en .5 (para no permitir empates)
    # Pero también aceptamos .0 por si acaso
    home_decimal = home_spread % 1
    away_decimal = away_spread % 1

    # Aceptar .0 o .5, rechazar otras decimales
    valid_decimals = [0.0, 0.5]
    if home_decimal not in valid_decimals:
        return f"Spread home debe terminar en .0 o .5 (recibido: {home_spread})"

    if away_decimal not in valid_decimals:
        return f"Spread away debe terminar en .0 o .5 (recibido: {away_spread})"

    return None


def validar_total(total: dict) -> str | None:
    """
    Valida línea de Total Points (Over/Under).

    Ejemplos válidos:
    - linea: 210.5
    - linea: 215.0
    """
    if not total:
        return "Total vacío"

    if "linea" not in total:
        return "Total debe tener 'linea'"

    linea = total.get("linea")

    # Validar que sea número
    try:
        linea = float(linea)
    except (TypeError, ValueError):
        return "Total debe ser número válido"

    # Rango típico NBA: 185-235 puntos
    # Casos extremos: puede ser 175 (equipos muy lentos) o 245 (High Scoring)
    if linea < 150 or linea > 270:
        return f"Total {linea} está fuera del rango típico NBA (150-270)"

    # Preferencia: debe terminar en .5 (para no permitir empates)
    decimal_part = linea % 1
    valid_decimals = [0.0, 0.5]

    if decimal_part not in valid_decimals:
        return f"Total debe terminar en .0 o .5 (recibido: {linea})"

    return None


def validar_cuota_americana(cuota: float) -> str | None:
    """
    Valida una cuota en formato americano.

    Formatos válidos:
    - Favorito: -110, -150, -200, etc.
    - Underdog: +110, +150, +200, etc.
    """
    try:
        cuota = float(cuota)
    except (TypeError, ValueError):
        return "Cuota debe ser número válido"

    # Cuota americana típicamente está entre -1000 y +1000
    if abs(cuota) < 100 or abs(cuota) > 1000:
        return f"Cuota {cuota} fuera de rango típico (-1000 a +1000)"

    # Debe ser número entero (aunque permitimos float por flexibilidad)
    # No hay problema si no es entero, pero es raro

    return None


def convertir_cuota_americana_a_decimal(cuota_americana: float) -> float:
    """
    Convierte cuota americana a decimal.

    Ejemplos:
    - -150 → 1.67
    - +130 → 2.30
    - -110 → 1.91
    """
    if cuota_americana > 0:
        # Underdog: +200 → 1 + 200/100 = 3.00
        return 1 + (cuota_americana / 100)
    else:
        # Favorito: -150 → 1 + 100/150 = 1.667
        return 1 + (100 / abs(cuota_americana))
