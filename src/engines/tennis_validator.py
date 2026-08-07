"""
Validador de entradas para el motor de tenis.
"""
import logging

_log = logging.getLogger("betbrain.tennis")

SUPERFICIES_VALIDAS = {"clay", "hard", "grass", "carpet"}
FORMATOS_VALIDOS    = {"best_of_3", "best_of_5"}
ELO_MIN, ELO_MAX    = 500.0, 3000.0
ELO_DEFAULT         = 1500.0


def validar_entrada_tenis(data: dict) -> tuple:
    """
    Valida y extrae todos los campos de una solicitud de análisis de tenis.

    Returns:
        (jugador1, jugador2, elo1, elo2, superficie, formato, cuotas, error_msg)
    """
    j1 = (data.get("jugador1") or "").strip()
    j2 = (data.get("jugador2") or "").strip()

    if not j1:
        return (None,) * 7 + ("Falta el jugador 1",)
    if not j2:
        return (None,) * 7 + ("Falta el jugador 2",)
    if j1.lower() == j2.lower():
        return (None,) * 7 + ("Los jugadores no pueden ser el mismo",)

    # Cargar Elos del formulario O desde ratings calibrados
    try:
        elo1 = float(data.get("elo1")) if data.get("elo1") else None  # type: ignore[arg-type]
        elo2 = float(data.get("elo2")) if data.get("elo2") else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return (None,) * 7 + ("Elo debe ser un número",)

    # Si los Elos no se proporcionan, cargar desde ratings
    if elo1 is None or elo2 is None:
        try:
            import json
            from pathlib import Path
            elo_path = Path(__file__).parent.parent / "data" / "tennis_elo_ratings.json"
            if elo_path.exists():
                with open(elo_path, 'r', encoding='utf-8') as f:
                    ratings = json.load(f).get("jugadores", {})
                    if elo1 is None:
                        elo1 = ratings.get(j1, {}).get("elo", ELO_DEFAULT)
                    if elo2 is None:
                        elo2 = ratings.get(j2, {}).get("elo", ELO_DEFAULT)
            else:
                elo1 = elo1 or ELO_DEFAULT
                elo2 = elo2 or ELO_DEFAULT
        except Exception as e:
            _log.warning(f"Error cargando Elos calibrados: {e}")
            elo1 = elo1 or ELO_DEFAULT
            elo2 = elo2 or ELO_DEFAULT

    err = _validar_elo(elo1, "elo1")
    if err:
        return (None,) * 7 + (err,)
    err = _validar_elo(elo2, "elo2")
    if err:
        return (None,) * 7 + (err,)

    superficie = (data.get("superficie") or "hard").lower().strip()
    if superficie not in SUPERFICIES_VALIDAS:
        return (None,) * 7 + (
            f"Superficie '{superficie}' inválida. Opciones: {sorted(SUPERFICIES_VALIDAS)}",
        )

    formato = (data.get("formato") or "best_of_3").lower().strip()
    if formato not in FORMATOS_VALIDOS:
        return (None,) * 7 + (
            f"Formato '{formato}' inválido. Opciones: {sorted(FORMATOS_VALIDOS)}",
        )

    cuotas = data.get("cuotas") or {}
    if not cuotas:
        return (None,) * 7 + ("Falta al menos un mercado en cuotas",)

    validators = {
        "match_winner":  validar_match_winner,
        "primer_set":    validar_primer_set,
    }
    for mercado, fn in validators.items():
        if mercado in cuotas:
            err = fn(cuotas[mercado])
            if err:
                return (None,) * 7 + (err,)

    if "set_handicap" in cuotas:
        err = validar_set_handicap(cuotas["set_handicap"], formato)
        if err:
            return (None,) * 7 + (err,)

    if "total_games" in cuotas:
        err = validar_total_games(cuotas["total_games"], formato)
        if err:
            return (None,) * 7 + (err,)

    if "game_handicap" in cuotas:
        err = validar_game_handicap(cuotas["game_handicap"], formato)
        if err:
            return (None,) * 7 + (err,)

    if "sets_winners" in cuotas:
        err = validar_sets_winners(cuotas["sets_winners"], formato)
        if err:
            return (None,) * 7 + (err,)

    return j1, j2, elo1, elo2, superficie, formato, cuotas, None


# ── Validadores por mercado ───────────────────────────────────────────────────

def validar_match_winner(mw: dict) -> str | None:
    if not mw:
        return "match_winner vacío"
    if "1" not in mw or "2" not in mw:
        return "match_winner debe tener claves '1' y '2'"
    for k in ("1", "2"):
        err = _validar_cuota_decimal(mw[k], f"match_winner[{k}]")
        if err:
            return err
    margen = 1.0 / float(mw["1"]) + 1.0 / float(mw["2"])
    if not (1.0 <= margen <= 1.30):
        return f"Margen de match_winner fuera de rango ({margen:.3f})"
    return None


def validar_primer_set(ps: dict) -> str | None:
    if not ps:
        return "primer_set vacío"
    if "1" not in ps or "2" not in ps:
        return "primer_set debe tener claves '1' y '2'"
    for k in ("1", "2"):
        err = _validar_cuota_decimal(ps[k], f"primer_set[{k}]")
        if err:
            return err
    margen = 1.0 / float(ps["1"]) + 1.0 / float(ps["2"])
    if not (1.0 <= margen <= 1.35):
        return f"Margen de primer_set fuera de rango ({margen:.3f})"
    return None


def validar_set_handicap(sh: dict, formato: str = "best_of_3") -> str | None:
    if not sh:
        return "set_handicap vacío"
    lineas_validas_bo3 = {-1.5, +1.5}
    lineas_validas_bo5 = {-2.5, -1.5, -0.5, +0.5, +1.5, +2.5}
    lineas_ok = lineas_validas_bo5 if formato == "best_of_5" else lineas_validas_bo3

    for linea_str, cuota in sh.items():
        try:
            linea = float(linea_str)
        except ValueError:
            return f"set_handicap: clave '{linea_str}' no es un número"
        if linea not in lineas_ok:
            return (f"set_handicap: línea {linea_str} no válida para {formato}. "
                    f"Válidas: {sorted(lineas_ok)}")
        err = _validar_cuota_decimal(cuota, f"set_handicap[{linea_str}]")
        if err:
            return err
    return None


def validar_total_games(tg: dict, formato: str = "best_of_3") -> str | None:
    if not tg:
        return "total_games vacío"
    if "linea" not in tg:
        return "total_games debe tener 'linea'"
    try:
        linea = float(tg["linea"])
    except (TypeError, ValueError):
        return "total_games.linea debe ser número"
    rng = (20.0, 65.0) if formato == "best_of_5" else (15.0, 40.0)
    if not (rng[0] <= linea <= rng[1]):
        return f"total_games.linea {linea} fuera de rango ({rng[0]}–{rng[1]})"
    if linea % 0.5 != 0:
        return f"total_games.linea debe terminar en .0 o .5 (recibido: {linea})"
    if "over" not in tg and "under" not in tg:
        return "total_games debe tener al menos 'over' o 'under'"
    for lado in ("over", "under"):
        if lado in tg:
            err = _validar_cuota_decimal(tg[lado], f"total_games.{lado}")
            if err:
                return err
    return None


def validar_game_handicap(gh: dict, formato: str = "best_of_3") -> str | None:
    """
    Valida hándicap de games totales.

    Estructura: {"linea": 4.5, "j1": 2.10, "j2": 1.75}
    """
    if not gh:
        return "game_handicap vacío"
    if "linea" not in gh:
        return "game_handicap debe tener 'linea'"
    try:
        linea = float(gh["linea"])
    except (TypeError, ValueError):
        return "game_handicap.linea debe ser número"
    rng = (1.5, 25.0) if formato == "best_of_5" else (1.5, 16.0)
    if not (rng[0] <= linea <= rng[1]):
        return f"game_handicap.linea {linea} fuera de rango ({rng[0]}–{rng[1]})"
    if linea % 0.5 != 0:
        return "game_handicap.linea debe terminar en .0 o .5"
    if "j1" not in gh and "j2" not in gh:
        return "game_handicap debe tener al menos 'j1' o 'j2'"
    for lado in ("j1", "j2"):
        if lado in gh:
            err = _validar_cuota_decimal(gh[lado], f"game_handicap.{lado}")
            if err:
                return err
    return None


def validar_sets_winners(sw: dict, formato: str = "best_of_3") -> str | None:
    """
    Valida ganadores por set.

    Estructura: {"2": {"1": 1.80, "2": 2.00}, "3": {"1": 1.75, "2": 2.05}}
    """
    if not sw:
        return "sets_winners vacío"
    max_set = 5 if formato == "best_of_5" else 3
    for set_str, opciones in sw.items():
        try:
            n = int(set_str)
        except ValueError:
            return f"sets_winners: clave '{set_str}' no es un número entero"
        if not (1 <= n <= max_set):
            return f"sets_winners: set {n} fuera de rango (1–{max_set})"
        if not isinstance(opciones, dict):
            return f"sets_winners[{n}] debe ser un dict {{\"1\": cuota, \"2\": cuota}}"
        if "1" not in opciones or "2" not in opciones:
            return f"sets_winners[{n}] debe tener claves '1' y '2'"
        for k in ("1", "2"):
            err = _validar_cuota_decimal(opciones[k], f"sets_winners[{n}][{k}]")
            if err:
                return err
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validar_cuota_decimal(cuota, campo: str) -> str | None:
    try:
        c = float(cuota)
    except (TypeError, ValueError):
        return f"{campo} debe ser número decimal"
    if c <= 1.01:
        return f"{campo} debe ser > 1.01 (recibido: {c})"
    if c > 100.0:
        return f"{campo} parece demasiado alta ({c})"
    return None


def _validar_elo(elo: float, campo: str) -> str | None:
    if not (ELO_MIN <= elo <= ELO_MAX):
        return f"{campo} = {elo} fuera del rango ({ELO_MIN}–{ELO_MAX})"
    return None
