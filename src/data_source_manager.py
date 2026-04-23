"""
DataSourceManager — orquestador multi-fuente de datos externos.

Fuentes disponibles
───────────────────
  "api-football"  : api-sports.io (live, 100 req/día, requiere API_FOOTBALL_KEY)
  "sportsmonk"    : sportmonks.com (live, opcional, requiere SPORTMONKS_TOKEN)
  "football-data" : CSVs locales de football-data.co.uk (sin límite, sin key)
  "merged"        : api-football + CSV combinados

Cadena de fallback en modo "auto"
──────────────────────────────────
  1. api-football  → si tiene forma, devuelve (api_disponible=True)
  2. sportsmonk    → si api falla Y hay token, intenta SM (api_disponible=True)
  3. football-data → fallback final siempre disponible  (api_disponible=False)

Garantías de resiliencia
────────────────────────
  - Si api-football falla → el sistema sigue funcionando (SM o CSV)
  - Si sportsmonk falla   → el sistema sigue funcionando (CSV)
  - Si CSV falla          → devuelve ctx vacío, nunca lanza excepción
  - SPORTMONKS_TOKEN vacío → sportsmonk se salta silenciosamente

Otras responsabilidades
───────────────────────
  - Deduplicación de requests concurrentes (request coalescing via threading.Event)
  - Estadísticas de uso por fuente (delegated, fallback, errors, coalesced)
  - Singleton `dsm` compartido por toda la app

Uso
───
    from src.data_source_manager import dsm

    ctx = dsm.contexto_partido_completo("Real Madrid", "Barcelona")
    ctx = dsm.contexto_partido_completo("Arsenal", "Chelsea", fuente="sportsmonk")
    ctx = dsm.contexto_partido_completo("PSG", "Lyon",       fuente="merged")

    fixtures = dsm.get_fixtures_today()
    result   = dsm.get_fixture_result(fixture_id)
    stats    = dsm.stats()
"""
import logging
import threading
from typing import Any, Literal

_log = logging.getLogger("betbrain.dsm")

Fuente = Literal["api-football", "sportsmonk", "thesportsdb", "football-data", "merged", "auto"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de merge (api-football + CSV)
# ──────────────────────────────────────────────────────────────────────────────

def _merge_forma(forma_primary: dict | None, forma_secondary: dict | None) -> dict | None:
    """
    Prioriza la fuente primaria (más reciente).
    Si la primaria falla pero la secundaria tiene datos, usa la secundaria con nota.
    """
    if forma_primary:
        return forma_primary
    if forma_secondary:
        result = dict(forma_secondary)
        result["_fuente"] = f"{result.get('_fuente', 'secondary')}-fallback"
        return result
    return None


def _merge_h2h(h2h_primary: dict | None, h2h_secondary: dict | None) -> dict | None:
    """
    Para H2H elige el que tenga más partidos (mayor profundidad histórica).
    Si solo uno tiene datos, usa ese.
    """
    if h2h_primary and h2h_secondary:
        if h2h_secondary.get("n", 0) >= h2h_primary.get("n", 0):
            merged = dict(h2h_secondary)
            merged["_fuente"] = f"{h2h_secondary.get('_fuente','sec')}+primary"
            merged["_n_primary"] = h2h_primary.get("n", 0)
        else:
            merged = dict(h2h_primary)
            merged["_fuente"] = f"{h2h_primary.get('_fuente','pri')}+secondary"
            merged["_n_secondary"] = h2h_secondary.get("n", 0)
        return merged
    return h2h_primary or h2h_secondary


def _merge_contextos(ctx_api: dict, ctx_csv: dict) -> dict:
    """Fusiona api-football + CSV: forma de API, H2H del que tenga más partidos."""
    notas = ctx_api.get("notas", []) + [
        n for n in ctx_csv.get("notas", [])
        if n not in ctx_api.get("notas", [])
    ]
    return {
        "api_disponible": ctx_api.get("api_disponible") or ctx_csv.get("api_disponible"),
        "fuente":         "merged",
        "home":           ctx_api["home"],
        "away":           ctx_api["away"],
        "home_id":        ctx_api.get("home_id") or ctx_csv.get("home_id"),
        "away_id":        ctx_api.get("away_id") or ctx_csv.get("away_id"),
        "forma_home":     _merge_forma(ctx_api.get("forma_home"), ctx_csv.get("forma_home")),
        "forma_away":     _merge_forma(ctx_api.get("forma_away"), ctx_csv.get("forma_away")),
        "h2h":            _merge_h2h(ctx_api.get("h2h"), ctx_csv.get("h2h")),
        "injuries_home":  ctx_api.get("injuries_home", []),
        "injuries_away":  ctx_api.get("injuries_away", []),
        "notas":          notas,
    }


def _ctx_tiene_forma(ctx: dict) -> bool:
    """True si el contexto tiene al menos una forma de equipo válida."""
    return (ctx.get("api_disponible") and
            (ctx.get("forma_home") is not None or
             ctx.get("forma_away") is not None))


# ──────────────────────────────────────────────────────────────────────────────
# DataSourceManager
# ──────────────────────────────────────────────────────────────────────────────

class DataSourceManager:
    """
    Punto de acceso único a datos externos. Thread-safe.

    `fuente_default` controla la estrategia cuando el caller no especifica
    fuente explícita. Valores: "auto" | "api-football" | "sportsmonk" |
    "football-data" | "merged".

    El __init__ NO inicializa ninguna fuente externa — todos los imports
    son lazy (dentro de métodos) para respetar el orden de importación
    y facilitar el mockeo en tests.
    """

    def __init__(self, fuente_default: Fuente = "auto"):
        self.fuente_default: Fuente = fuente_default

        # Deduplicación concurrente — ningún import externo aquí
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self._inflight_results: dict[str, Any]     = {}

        # Contadores de diagnóstico por fuente
        self._stats = {
            "coalesced":       0,  # requests que esperaron a otro igual
            "delegated_api":   0,  # llamadas delegadas a api-football
            "delegated_sm":    0,  # llamadas delegadas a sportsmonk
            "delegated_tsdb":  0,  # llamadas delegadas a thesportsdb
            "delegated_csv":   0,  # llamadas delegadas a CSV provider
            "fallback_sm":     0,  # api-football fallo -> intento sportsmonk
            "fallback_tsdb":   0,  # sportsmonk fallo -> intento thesportsdb
            "fallback_csv":    0,  # todo live fallo -> uso CSV
            "merged":          0,  # respuestas merged
            "errors_api":      0,
            "errors_sm":       0,
            "errors_tsdb":     0,
            "errors_csv":      0,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Deduplicación interna (request coalescing)
    # ──────────────────────────────────────────────────────────────────────────

    def _call(self, key: str, fn, *args, **kwargs) -> Any:
        """
        Ejecuta fn(*args, **kwargs) garantizando que si dos threads piden
        la misma `key` simultáneamente, solo uno lanza la llamada real;
        el otro espera y reutiliza el resultado.
        """
        with self._lock:
            if key in self._inflight:
                event = self._inflight[key]
                self._stats["coalesced"] += 1
                wait_for_result = True
            else:
                event = threading.Event()
                self._inflight[key] = event
                wait_for_result = False

        if wait_for_result:
            event.wait(timeout=30)
            return self._inflight_results.get(key)

        result = None
        try:
            result = fn(*args, **kwargs)
        finally:
            with self._lock:
                self._inflight_results[key] = result
                del self._inflight[key]
            event.set()

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # contexto_partido_completo — punto de entrada principal
    # ──────────────────────────────────────────────────────────────────────────

    def contexto_partido_completo(
        self,
        home: str,
        away: str,
        fuente: Fuente | None = None,
    ) -> dict:
        """
        Devuelve contexto completo (forma, H2H, lesiones) para analizar el partido.

        fuente:
          "auto"          → api-football → sportsmonk → football-data (CSV)
          "api-football"  → solo api-football
          "sportsmonk"    → solo sportsmonk
          "football-data" → solo CSVs locales
          "merged"        → api-football + CSV combinados
          None            → usa self.fuente_default
        """
        fuente = fuente or self.fuente_default

        if fuente == "api-football":
            return self._ctx_api(home, away)
        if fuente == "sportsmonk":
            return self._ctx_sm(home, away)
        if fuente == "thesportsdb":
            return self._ctx_tsdb(home, away)
        if fuente == "football-data":
            return self._ctx_csv(home, away)
        if fuente == "merged":
            return self._ctx_merged(home, away)
        # "auto": cadena api-football -> sportsmonk -> thesportsdb -> CSV
        return self._ctx_auto(home, away)

    # ──────────────────────────────────────────────────────────────────────────
    # Proveedores individuales
    # ──────────────────────────────────────────────────────────────────────────

    def _ctx_api(self, home: str, away: str) -> dict:
        """Fuente: api-football (api-sports.io)."""
        from src.api_football import contexto_partido_completo as _fn
        key = f"api:ctx:{home.lower()}:{away.lower()}"

        def _wrap():
            with self._lock:
                self._stats["delegated_api"] += 1
            try:
                return _fn(home, away)
            except Exception as exc:
                with self._lock:
                    self._stats["errors_api"] += 1
                _log.error("api-football error: %s", exc, exc_info=True)
                return None

        result = self._call(key, _wrap)
        if result is None:
            return _empty_ctx(home, away, fuente="api-football",
                              nota="api-football: error o no disponible")
        result["fuente"] = "api-football"
        return result

    def _ctx_sm(self, home: str, away: str) -> dict:
        """Fuente: Sportmonks (sportmonks.com). Solo activo si SPORTMONKS_TOKEN está configurado."""
        from src.sportsmonk import contexto_partido_completo as _fn
        key = f"sm:ctx:{home.lower()}:{away.lower()}"

        def _wrap():
            with self._lock:
                self._stats["delegated_sm"] += 1
            try:
                return _fn(home, away)
            except Exception as exc:
                with self._lock:
                    self._stats["errors_sm"] += 1
                _log.error("sportsmonk error: %s", exc, exc_info=True)
                return None

        result = self._call(key, _wrap)
        if result is None:
            return _empty_ctx(home, away, fuente="sportsmonk",
                              nota="sportsmonk: error o no disponible")
        return result

    def _ctx_tsdb(self, home: str, away: str) -> dict:
        """Fuente: TheSportsDB v1 (gratuita, sin key). Cubre Liga MX, MLS, etc."""
        from src.thesportsdb import contexto_partido_completo as _fn
        key = f"tsdb:ctx:{home.lower()}:{away.lower()}"

        def _wrap():
            with self._lock:
                self._stats["delegated_tsdb"] += 1
            try:
                return _fn(home, away)
            except Exception as exc:
                with self._lock:
                    self._stats["errors_tsdb"] += 1
                _log.error("thesportsdb error: %s", exc, exc_info=True)
                return None

        result = self._call(key, _wrap)
        if result is None:
            return _empty_ctx(home, away, fuente="thesportsdb",
                              nota="thesportsdb: error o equipo no encontrado")
        return result

    def _ctx_csv(self, home: str, away: str) -> dict:
        """Fuente: CSVs locales de football-data.co.uk. Sin límite, sin key."""
        from src.football_data_api import contexto_partido_completo as _fn
        key = f"csv:ctx:{home.lower()}:{away.lower()}"

        def _wrap():
            with self._lock:
                self._stats["delegated_csv"] += 1
            try:
                return _fn(home, away)
            except Exception as exc:
                with self._lock:
                    self._stats["errors_csv"] += 1
                _log.error("CSV error: %s", exc, exc_info=True)
                return None

        result = self._call(key, _wrap)
        if result is None:
            return _empty_ctx(home, away, fuente="football-data",
                              nota="CSV provider: error interno")
        return result

    def _ctx_auto(self, home: str, away: str) -> dict:
        """
        Cadena de fallback: api-football -> sportsmonk -> thesportsdb -> CSV.

        api_disponible semantica:
          True  -> una fuente live respondio con datos de forma reales
          False -> solo hay datos locales (CSV) o ninguno
        """
        # Paso 1: api-football
        ctx = self._ctx_api(home, away)
        if _ctx_tiene_forma(ctx):
            return ctx

        # Paso 2: sportsmonk (solo si hay token)
        from src.sportsmonk import disponible as _sm_disponible
        if _sm_disponible():
            _log.info("DSM auto: api-football sin datos -> intentando Sportmonks")
            with self._lock:
                self._stats["fallback_sm"] += 1
            ctx_sm = self._ctx_sm(home, away)
            if _ctx_tiene_forma(ctx_sm):
                ctx_sm["fuente"] = "sportsmonk-fallback"
                return ctx_sm
            _log.info("DSM auto: Sportmonks sin datos -> intentando TheSportsDB")
        else:
            _log.debug("DSM auto: SPORTMONKS_TOKEN no configurado -> saltando SM")

        # Paso 3: TheSportsDB (siempre disponible, sin key, cubre Liga MX)
        with self._lock:
            self._stats["fallback_tsdb"] += 1
        ctx_tsdb = self._ctx_tsdb(home, away)
        if _ctx_tiene_forma(ctx_tsdb):
            ctx_tsdb["fuente"] = "thesportsdb-fallback"
            return ctx_tsdb
        _log.info("DSM auto: TheSportsDB sin datos -> fallback CSV")

        # Paso 4: football-data CSV (ultimo recurso, solo ligas europeas)
        with self._lock:
            self._stats["fallback_csv"] += 1
        ctx_csv = self._ctx_csv(home, away)

        # api_disponible=False: ningun live API respondio con datos utiles.
        ctx_csv["api_disponible"] = False

        notas_fallido = ctx.get("notas", [])
        if notas_fallido:
            ctx_csv["notas"] = (notas_fallido +
                                ["auto-fallback: live APIs sin datos -> usando CSV local"] +
                                ctx_csv.get("notas", []))
        ctx_csv["fuente"] = "csv-fallback"
        return ctx_csv

    def _ctx_merged(self, home: str, away: str) -> dict:
        """Fusiona api-football + CSV."""
        ctx_api = self._ctx_api(home, away)
        ctx_csv = self._ctx_csv(home, away)
        with self._lock:
            self._stats["merged"] += 1
        merged = _merge_contextos(ctx_api, ctx_csv)
        _log.info("DSM merged '%s vs %s': api_forma=%s csv_forma=%s h2h=%s",
                  home, away,
                  ctx_api.get("forma_home") is not None,
                  ctx_csv.get("forma_home") is not None,
                  merged.get("h2h") is not None)
        return merged

    # ──────────────────────────────────────────────────────────────────────────
    # Wrappers de funciones puntuales
    # ──────────────────────────────────────────────────────────────────────────

    def factor_ajuste_lesiones(self, injuries: list) -> float:
        """Factor multiplicativo del lambda según lesionados. Sin IO."""
        from src.api_football import factor_ajuste_lesiones as _fn
        return _fn(injuries)

    def get_fixtures_today(self, league_id: int | None = None) -> list:
        """Partidos del día (solo api-football dispone de este endpoint)."""
        from src.api_football import get_fixtures_today as _fn
        key = f"api:fixtures_today:{league_id}"

        def _wrap():
            with self._lock:
                self._stats["delegated_api"] += 1
            try:
                return _fn(league_id)
            except Exception as exc:
                with self._lock:
                    self._stats["errors_api"] += 1
                _log.error("get_fixtures_today error: %s", exc)
                return None

        result = self._call(key, _wrap)
        return result if result is not None else []

    def get_fixture_result(self, fixture_id: int) -> dict | None:
        """Resultado de un partido terminado (solo api-football)."""
        from src.api_football import get_fixture_result as _fn
        key = f"api:fixture_result:{fixture_id}"

        def _wrap():
            with self._lock:
                self._stats["delegated_api"] += 1
            try:
                return _fn(fixture_id)
            except Exception as exc:
                with self._lock:
                    self._stats["errors_api"] += 1
                _log.error("get_fixture_result error: %s", exc)
                return None

        return self._call(key, _wrap)

    def get_team_form(self, team_name: str, last: int = 5,
                      fuente: Fuente | None = None) -> dict | None:
        """Forma reciente. Cadena: api → sportsmonk → CSV según fuente."""
        fuente = fuente or self.fuente_default

        if fuente in ("api-football", "auto"):
            from src.api_football import search_team, get_team_form as _fn_api
            tid = self._call(f"api:search:{team_name.lower()}", search_team, team_name)
            if tid:
                result = self._call(f"api:form:{tid}:{last}", _fn_api, tid, last)
                if result:
                    return result
            if fuente == "api-football":
                return None

        if fuente in ("sportsmonk", "auto"):
            from src.sportsmonk import disponible as _sm_disponible, search_team as _sm_search
            from src.sportsmonk import get_team_form as _fn_sm
            if _sm_disponible():
                tid = self._call(f"sm:search:{team_name.lower()}", _sm_search, team_name)
                if tid:
                    result = self._call(f"sm:form:{tid}:{last}", _fn_sm, tid, last)
                    if result:
                        return result
            if fuente == "sportsmonk":
                return None

        from src.football_data_api import get_team_form_csv
        return get_team_form_csv(team_name, last)

    def get_h2h(self, home: str, away: str, last: int = 10,
                fuente: Fuente | None = None) -> dict | None:
        """H2H entre dos equipos. Cadena: api → sportsmonk → CSV según fuente."""
        fuente = fuente or self.fuente_default

        if fuente in ("api-football", "auto"):
            from src.api_football import search_team, get_head_to_head as _fn_api
            hid = self._call(f"api:search:{home.lower()}", search_team, home)
            aid = self._call(f"api:search:{away.lower()}", search_team, away)
            if hid and aid:
                id_a, id_b = sorted([hid, aid])
                result = self._call(f"api:h2h:{id_a}:{id_b}:{last}", _fn_api, hid, aid, last)
                if result:
                    return result
            if fuente == "api-football":
                return None

        if fuente in ("sportsmonk", "auto"):
            from src.sportsmonk import disponible as _sm_disponible, search_team as _sm_search
            from src.sportsmonk import get_head_to_head as _fn_sm
            if _sm_disponible():
                hid = self._call(f"sm:search:{home.lower()}", _sm_search, home)
                aid = self._call(f"sm:search:{away.lower()}", _sm_search, away)
                if hid and aid:
                    result = self._call(f"sm:h2h:{hid}:{aid}:{last}", _fn_sm, hid, aid, last)
                    if result:
                        return result
            if fuente == "sportsmonk":
                return None

        from src.football_data_api import get_h2h_csv
        return get_h2h_csv(home, away, last)

    def search_team_csv(self, name: str) -> str | None:
        """Nombre normalizado del equipo en los CSVs."""
        from src.football_data_api import buscar_nombre_equipo
        return buscar_nombre_equipo(name)

    def csv_info(self) -> dict:
        """Cobertura de los CSVs (ligas, fechas, nº partidos)."""
        from src.football_data_api import info_cobertura
        return info_cobertura()

    def sportsmonk_disponible(self) -> bool:
        """True si SPORTMONKS_TOKEN esta configurado."""
        from src.sportsmonk import disponible
        return disponible()

    def thesportsdb_fixtures_hoy(self, league_id: str = "4350") -> list:
        """Partidos de hoy de una liga via TheSportsDB (gratis). Default: Liga MX."""
        from src.thesportsdb import get_fixtures_today
        try:
            return get_fixtures_today(league_id)
        except Exception as exc:
            _log.error("thesportsdb fixtures_hoy error: %s", exc)
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnóstico
    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Contadores de uso del manager (por fuente)."""
        with self._lock:
            return dict(self._stats)

    def reset_stats(self) -> None:
        """Reinicia contadores (útil en tests)."""
        with self._lock:
            for k in self._stats:
                self._stats[k] = 0


# ──────────────────────────────────────────────────────────────────────────────
# Helper privado
# ──────────────────────────────────────────────────────────────────────────────

def _empty_ctx(home: str, away: str, fuente: str, nota: str) -> dict:
    return {
        "api_disponible": False,
        "fuente":         fuente,
        "home": home, "away": away,
        "home_id": None, "away_id": None,
        "forma_home": None, "forma_away": None,
        "h2h": None,
        "injuries_home": [], "injuries_away": [],
        "notas": [nota],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Singleton — toda la app comparte esta instancia
# ──────────────────────────────────────────────────────────────────────────────

dsm = DataSourceManager(fuente_default="auto")
