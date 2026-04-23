"""
Tests del sistema multi-fuente de datos.

Cubre:
  - football_data_api (CSV provider)
  - DataSourceManager: fuente "auto", "api-football", "sportsmonk",
                       "football-data", "merged"
  - Cadena de fallback: api-football → sportsmonk → CSV
  - Sportmonks: graceful degradation si no hay token
  - Deduplicación concurrente del DSM
  - Merge de contextos

Ejecutar:
    python -m pytest tests/test_multi_source.py -v
    python -m pytest tests/test_multi_source.py -v -k csv         # solo CSV
    python -m pytest tests/test_multi_source.py -v -k sportsmonk  # solo SM
    python -m pytest tests/test_multi_source.py -v -k merge       # solo merge
"""
import threading
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures compartidos
# ──────────────────────────────────────────────────────────────────────────────

CTX_API_COMPLETO = {
    "api_disponible": True,
    "fuente": "api-football",
    "home": "Real Madrid", "away": "Barcelona",
    "home_id": 541, "away_id": 529,
    "forma_home": {
        "partidos": 5, "W": 3, "D": 1, "L": 1,
        "gf_promedio": 2.0, "gc_promedio": 1.0,
        "btts_rate": 0.6, "over_25_rate": 0.6,
        "secuencia": "WWDLW",
    },
    "forma_away": {
        "partidos": 5, "W": 4, "D": 0, "L": 1,
        "gf_promedio": 2.4, "gc_promedio": 0.8,
        "btts_rate": 0.4, "over_25_rate": 0.6,
        "secuencia": "WWWLW",
    },
    "h2h": {
        "n": 5, "goles_promedio": 3.2,
        "btts_rate": 0.8, "over_25_rate": 0.8,
        "wins_local_actual": 2, "empates": 1, "wins_visit_actual": 2,
    },
    "injuries_home": [], "injuries_away": [],
    "notas": [],
}

CTX_API_VACIO = {
    "api_disponible": True,
    "fuente": "api-football",
    "home": "Real Madrid", "away": "Barcelona",
    "home_id": None, "away_id": None,
    "forma_home": None, "forma_away": None,
    "h2h": None,
    "injuries_home": [], "injuries_away": [],
    "notas": ["No se encontró team_id"],
}

CTX_CSV_COMPLETO = {
    "api_disponible": True,
    "fuente": "csv",
    "home": "Real Madrid", "away": "Barcelona",
    "home_id": "Real Madrid", "away_id": "Barcelona",
    "forma_home": {
        "partidos": 5, "W": 2, "D": 2, "L": 1,
        "gf_promedio": 1.6, "gc_promedio": 1.2,
        "btts_rate": 0.6, "over_25_rate": 0.4,
        "secuencia": "WWDDL", "_fuente": "csv",
    },
    "forma_away": {
        "partidos": 5, "W": 3, "D": 1, "L": 1,
        "gf_promedio": 2.0, "gc_promedio": 1.0,
        "btts_rate": 0.6, "over_25_rate": 0.6,
        "secuencia": "WWWDL", "_fuente": "csv",
    },
    "h2h": {
        "n": 10, "goles_promedio": 3.0,
        "btts_rate": 0.7, "over_25_rate": 0.7,
        "wins_local_actual": 4, "empates": 2, "wins_visit_actual": 4,
        "_fuente": "csv",
    },
    "injuries_home": [], "injuries_away": [],
    "notas": [],
}


# ──────────────────────────────────────────────────────────────────────────────
# Tests: football_data_api (CSV provider)
# ──────────────────────────────────────────────────────────────────────────────

class TestCSVProvider:

    def test_normalizar_nombre_exacto(self):
        from src.football_data_api import _normalizar
        assert _normalizar("Real Madrid") == "real madrid"
        assert _normalizar("Atlético Madrid") == "atletico madrid"
        assert _normalizar("  PSG  ") == "psg"

    def test_buscar_nombre_equipo_hit(self):
        """Con CSVs reales cargados, Real Madrid debería encontrarse."""
        from src.football_data_api import buscar_nombre_equipo, _cargar_df
        df = _cargar_df()
        if df.empty:
            pytest.skip("CSVs no descargados — ejecuta data_loader.py")
        resultado = buscar_nombre_equipo("Real Madrid")
        assert resultado is not None
        assert "Real Madrid" in resultado or "madrid" in resultado.lower()

    def test_buscar_nombre_equipo_miss(self):
        from src.football_data_api import buscar_nombre_equipo
        resultado = buscar_nombre_equipo("Equipo Ficticio XYZ 99999")
        assert resultado is None

    def test_form_csv_retorna_schema_correcto(self):
        """Verifica que el schema de forma coincide con el esperado por el engine."""
        from src.football_data_api import _cargar_df, get_team_form_csv
        df = _cargar_df()
        if df.empty:
            pytest.skip("CSVs no disponibles")
        # Busca el primer equipo que tenga partidos
        equipos = df["HomeTeam"].dropna().unique()[:5]
        for equipo in equipos:
            forma = get_team_form_csv(equipo, last=5)
            if forma:
                assert "partidos"    in forma
                assert "W" in forma and "D" in forma and "L" in forma
                assert "gf_promedio" in forma
                assert "gc_promedio" in forma
                assert "btts_rate"   in forma
                assert "over_25_rate" in forma
                assert "secuencia"   in forma
                assert isinstance(forma["secuencia"], str)
                assert all(c in "WDL" for c in forma["secuencia"])
                break

    def test_h2h_csv_schema_correcto(self):
        from src.football_data_api import _cargar_df, get_h2h_csv
        df = _cargar_df()
        if df.empty:
            pytest.skip("CSVs no disponibles")
        # Busca un par con enfrentamientos reales
        grupos = df.groupby(["HomeTeam", "AwayTeam"]).size()
        pares = [(h, a) for (h, a), n in grupos.items() if n >= 2]
        if not pares:
            pytest.skip("Sin enfrentamientos repetidos en el CSV")
        home, away = pares[0]
        h2h = get_h2h_csv(home, away, last=10)
        if h2h:
            assert "n"                 in h2h
            assert "goles_promedio"    in h2h
            assert "btts_rate"         in h2h
            assert "over_25_rate"      in h2h
            assert "wins_local_actual" in h2h
            assert "empates"           in h2h
            assert "wins_visit_actual" in h2h

    def test_contexto_completo_schema(self):
        from src.football_data_api import _cargar_df, contexto_partido_completo
        df = _cargar_df()
        if df.empty:
            pytest.skip("CSVs no disponibles")
        ctx = contexto_partido_completo("Real Madrid", "Barcelona")
        assert "api_disponible" in ctx
        assert "fuente"         in ctx
        assert ctx["fuente"]    == "csv"
        assert "forma_home"     in ctx
        assert "forma_away"     in ctx
        assert "h2h"            in ctx
        assert "injuries_home"  in ctx   # vacío pero presente
        assert "notas"          in ctx

    def test_info_cobertura_estructura(self):
        from src.football_data_api import info_cobertura
        info = info_cobertura()
        assert "partidos"     in info
        assert "equipos"      in info
        assert "ligas"        in info
        assert "rango_fechas" in info
        assert isinstance(info["ligas"], list)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: DataSourceManager
# ──────────────────────────────────────────────────────────────────────────────

class TestDataSourceManager:

    @pytest.fixture
    def dsm_fresco(self):
        """DSM nuevo sin estado previo para cada test."""
        from src.data_source_manager import DataSourceManager
        return DataSourceManager(fuente_default="auto")

    # --- fuente explícita "football-data" (CSV) ---

    def test_fuente_csv_retorna_schema(self, dsm_fresco):
        from src.football_data_api import _cargar_df
        if _cargar_df().empty:
            pytest.skip("CSVs no disponibles")
        ctx = dsm_fresco.contexto_partido_completo(
            "Real Madrid", "Barcelona", fuente="football-data"
        )
        assert ctx["fuente"] == "csv"
        assert "forma_home"  in ctx
        assert "h2h"         in ctx

    # --- fuente "api-football" mockeada ---

    def test_fuente_api_llama_api_football(self, dsm_fresco):
        with patch("src.api_football.contexto_partido_completo",
                   return_value=CTX_API_COMPLETO) as mock_fn:
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="api-football"
            )
        mock_fn.assert_called_once_with("Real Madrid", "Barcelona")
        assert ctx["fuente"] == "api-football"
        assert ctx["forma_home"] is not None

    def test_fuente_api_error_retorna_empty(self, dsm_fresco):
        with patch("src.api_football.contexto_partido_completo",
                   side_effect=Exception("timeout")):
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="api-football"
            )
        assert ctx["api_disponible"] is False
        assert ctx["forma_home"] is None
        assert len(ctx["notas"]) > 0

    # --- fuente "auto": api ok ---

    def test_auto_usa_api_si_hay_forma(self, dsm_fresco):
        with patch("src.api_football.contexto_partido_completo",
                   return_value=CTX_API_COMPLETO):
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="auto"
            )
        assert ctx["fuente"] == "api-football"
        assert ctx["forma_home"] is not None

    # --- fuente "auto": api sin datos → fallback CSV ---

    def test_auto_fallback_csv_si_api_sin_forma(self, dsm_fresco):
        ctx_tsdb_vacio = {**CTX_API_VACIO, "fuente": "thesportsdb"}
        with (
            patch("src.api_football.contexto_partido_completo",
                  return_value=CTX_API_VACIO),
            patch("src.sportsmonk.disponible", return_value=False),
            patch("src.thesportsdb.contexto_partido_completo",
                  return_value=ctx_tsdb_vacio),
            patch("src.football_data_api.contexto_partido_completo",
                  return_value=CTX_CSV_COMPLETO) as mock_csv,
        ):
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="auto"
            )
        mock_csv.assert_called_once()
        assert ctx["fuente"] == "csv-fallback"
        s = dsm_fresco.stats()
        assert s["fallback_csv"] == 1

    # --- fuente "merged" ---

    def test_merged_combina_forma_api_h2h_csv(self, dsm_fresco):
        ctx_api_sin_h2h = dict(CTX_API_COMPLETO)
        ctx_api_sin_h2h["h2h"] = None

        with (
            patch("src.api_football.contexto_partido_completo",
                  return_value=ctx_api_sin_h2h),
            patch("src.football_data_api.contexto_partido_completo",
                  return_value=CTX_CSV_COMPLETO),
        ):
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="merged"
            )

        assert ctx["fuente"] == "merged"
        # La forma viene de api (tiene precedencia)
        assert ctx["forma_home"]["gf_promedio"] == CTX_API_COMPLETO["forma_home"]["gf_promedio"]
        # El H2H viene del CSV (api no tenía)
        assert ctx["h2h"] is not None
        assert ctx["h2h"].get("_fuente") == "csv"

    def test_merged_h2h_csv_gana_si_mas_partidos(self, dsm_fresco):
        """CSV tiene n=10, api tiene n=5 → se elige CSV."""
        with (
            patch("src.api_football.contexto_partido_completo",
                  return_value=CTX_API_COMPLETO),          # h2h n=5
            patch("src.football_data_api.contexto_partido_completo",
                  return_value=CTX_CSV_COMPLETO),          # h2h n=10
        ):
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="merged"
            )
        assert ctx["h2h"]["n"] == 10   # CSV tiene más historia

    def test_merged_api_h2h_gana_si_mas_partidos(self, dsm_fresco):
        """API tiene n=12, CSV tiene n=5 → se elige API."""
        ctx_api_h2h_grande = dict(CTX_API_COMPLETO)
        ctx_api_h2h_grande["h2h"] = dict(CTX_API_COMPLETO["h2h"])
        ctx_api_h2h_grande["h2h"]["n"] = 12

        ctx_csv_h2h_chico = dict(CTX_CSV_COMPLETO)
        ctx_csv_h2h_chico["h2h"] = dict(CTX_CSV_COMPLETO["h2h"])
        ctx_csv_h2h_chico["h2h"]["n"] = 5

        with (
            patch("src.api_football.contexto_partido_completo",
                  return_value=ctx_api_h2h_grande),
            patch("src.football_data_api.contexto_partido_completo",
                  return_value=ctx_csv_h2h_chico),
        ):
            ctx = dsm_fresco.contexto_partido_completo(
                "Arsenal", "Chelsea", fuente="merged"
            )
        assert ctx["h2h"]["n"] == 12

    # --- Estadísticas ---

    def test_stats_se_incrementan(self, dsm_fresco):
        with patch("src.api_football.contexto_partido_completo",
                   return_value=CTX_API_COMPLETO):
            dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="api-football"
            )
        s = dsm_fresco.stats()
        assert s["delegated_api"] >= 1

    def test_reset_stats(self, dsm_fresco):
        with patch("src.api_football.contexto_partido_completo",
                   return_value=CTX_API_COMPLETO):
            dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="api-football"
            )
        dsm_fresco.reset_stats()
        s = dsm_fresco.stats()
        assert all(v == 0 for v in s.values())

    # --- Deduplicación concurrente ---

    def test_coalescing_llama_api_una_sola_vez(self, dsm_fresco):
        """
        10 threads pidiendo el mismo partido al mismo tiempo
        → solo 1 delegated_api (los otros 9 son coalesced).
        """
        call_count = 0
        barrier = threading.Barrier(10)

        def mock_api(home, away):
            nonlocal call_count
            call_count += 1
            import time; time.sleep(0.05)  # simula latencia
            return CTX_API_COMPLETO

        resultados = []

        def worker():
            barrier.wait()  # todos arrancan al mismo tiempo
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="api-football"
            )
            resultados.append(ctx)

        with patch("src.api_football.contexto_partido_completo", side_effect=mock_api):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads: t.start()
            for t in threads: t.join()

        assert len(resultados) == 10
        assert all(r["fuente"] == "api-football" for r in resultados)
        # La API se llamó UNA sola vez (el resto fueron coalesced)
        assert call_count == 1
        s = dsm_fresco.stats()
        assert s["coalesced"] == 9
        assert s["delegated_api"] == 1

    # --- Cambio de fuente en caliente ---

    def test_cambio_fuente_default(self, dsm_fresco):
        assert dsm_fresco.fuente_default == "auto"
        dsm_fresco.fuente_default = "merged"
        assert dsm_fresco.fuente_default == "merged"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: helpers de merge (unitarios, sin IO)
# ──────────────────────────────────────────────────────────────────────────────

class TestMergeHelpers:

    def test_merge_forma_api_tiene_precedencia(self):
        from src.data_source_manager import _merge_forma
        api = {"gf_promedio": 2.0, "secuencia": "WWW"}
        csv = {"gf_promedio": 1.5, "secuencia": "DLD"}
        assert _merge_forma(api, csv) is api

    def test_merge_forma_fallback_a_csv(self):
        from src.data_source_manager import _merge_forma
        csv = {"gf_promedio": 1.5, "secuencia": "DLD", "_fuente": "csv"}
        result = _merge_forma(None, csv)
        assert result is not None
        assert result["_fuente"] == "csv-fallback"

    def test_merge_forma_ambos_none(self):
        from src.data_source_manager import _merge_forma
        assert _merge_forma(None, None) is None

    def test_merge_h2h_csv_mayor(self):
        from src.data_source_manager import _merge_h2h
        api = {"n": 3, "goles_promedio": 2.5}
        csv = {"n": 8, "goles_promedio": 3.0, "_fuente": "csv"}
        result = _merge_h2h(api, csv)
        assert result["n"] == 8

    def test_merge_h2h_api_mayor(self):
        from src.data_source_manager import _merge_h2h
        api = {"n": 12, "goles_promedio": 2.5}
        csv = {"n": 5,  "goles_promedio": 3.0}
        result = _merge_h2h(api, csv)
        assert result["n"] == 12

    def test_merge_h2h_solo_api(self):
        from src.data_source_manager import _merge_h2h
        api = {"n": 5}
        assert _merge_h2h(api, None) is api

    def test_merge_h2h_solo_csv(self):
        from src.data_source_manager import _merge_h2h
        csv = {"n": 5}
        assert _merge_h2h(None, csv) is csv

    def test_merge_contextos_combina_notas(self):
        from src.data_source_manager import _merge_contextos
        ctx_a = dict(CTX_API_COMPLETO); ctx_a["notas"] = ["nota-api"]
        ctx_c = dict(CTX_CSV_COMPLETO); ctx_c["notas"] = ["nota-csv"]
        merged = _merge_contextos(ctx_a, ctx_c)
        assert "nota-api" in merged["notas"]
        assert "nota-csv" in merged["notas"]

    def test_merge_contextos_no_duplica_notas(self):
        from src.data_source_manager import _merge_contextos
        misma_nota = "misma nota"
        ctx_a = dict(CTX_API_COMPLETO); ctx_a["notas"] = [misma_nota]
        ctx_c = dict(CTX_CSV_COMPLETO); ctx_c["notas"] = [misma_nota]
        merged = _merge_contextos(ctx_a, ctx_c)
        assert merged["notas"].count(misma_nota) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Sportmonks
# ──────────────────────────────────────────────────────────────────────────────

# Contexto que devuelve Sportmonks cuando tiene datos
CTX_SM_COMPLETO = {
    "api_disponible": True,
    "fuente": "sportsmonk",
    "home": "Real Madrid", "away": "Barcelona",
    "home_id": 1,  "away_id": 2,
    "forma_home": {
        "partidos": 5, "W": 3, "D": 1, "L": 1,
        "gf_promedio": 2.2, "gc_promedio": 0.8,
        "btts_rate": 0.4, "over_25_rate": 0.6,
        "secuencia": "WWDWL", "_fuente": "sportsmonk",
    },
    "forma_away": {
        "partidos": 5, "W": 4, "D": 0, "L": 1,
        "gf_promedio": 2.6, "gc_promedio": 0.6,
        "btts_rate": 0.4, "over_25_rate": 0.8,
        "secuencia": "WWWWL", "_fuente": "sportsmonk",
    },
    "h2h": {
        "n": 6, "goles_promedio": 3.0,
        "btts_rate": 0.8, "over_25_rate": 0.8,
        "wins_local_actual": 3, "empates": 1, "wins_visit_actual": 2,
        "_fuente": "sportsmonk",
    },
    "injuries_home": [], "injuries_away": [],
    "notas": [],
}


class TestSportsmonkClient:
    """Tests del cliente sportsmonk.py — todos usan mocks, sin red real."""

    def test_sin_token_devuelve_api_no_disponible(self):
        """Sin SPORTMONKS_TOKEN, contexto_partido_completo devuelve api_disponible=False."""
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", ""):
            from src.sportsmonk import contexto_partido_completo
            ctx = contexto_partido_completo("Real Madrid", "Barcelona")
        assert ctx["api_disponible"] is False
        assert ctx["fuente"] == "sportsmonk"
        assert ctx["forma_home"] is None
        assert ctx["forma_away"] is None
        assert any("SPORTMONKS_TOKEN" in n for n in ctx["notas"])

    def test_disponible_sin_token(self):
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", ""):
            from src.sportsmonk import disponible
            assert disponible() is False

    def test_disponible_con_token(self):
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", "fake-token-123"):
            from src.sportsmonk import disponible
            assert disponible() is True

    def test_search_team_sin_token_devuelve_none(self):
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", ""):
            from src.sportsmonk import search_team
            result = search_team("Real Madrid")
        assert result is None

    def test_search_team_respuesta_vacia(self):
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", "fake-token"):
            with patch("src.sportsmonk._get", return_value={"data": []}):
                from src.sportsmonk import search_team
                result = search_team("Equipo Inexistente")
        assert result is None

    def test_search_team_coincidencia_exacta(self):
        mock_resp = {
            "data": [
                {"id": 99, "name": "Real Valladolid"},
                {"id": 541, "name": "Real Madrid"},
            ]
        }
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", "fake-token"):
            with patch("src.sportsmonk._get", return_value=mock_resp):
                from src.sportsmonk import search_team
                result = search_team("Real Madrid")
        assert result == 541

    def test_search_team_fallback_primer_resultado(self):
        mock_resp = {"data": [{"id": 42, "name": "Real Zaragoza"}]}
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", "fake-token"):
            with patch("src.sportsmonk._get", return_value=mock_resp):
                from src.sportsmonk import search_team
                result = search_team("Real Madrid")  # no coincide exacto
        assert result == 42

    def test_get_team_form_schema_correcto(self):
        """Verifica que get_team_form devuelve el schema esperado."""
        fixtures_mock = [
            {
                "scores": [
                    {"description": "CURRENT", "score": {"goals": 2, "participant": "home"}},
                    {"description": "CURRENT", "score": {"goals": 1, "participant": "away"}},
                ],
                "participants": [
                    {"id": 541, "meta": {"location": "home"}},
                    {"id": 529, "meta": {"location": "away"}},
                ],
            },
            {
                "scores": [
                    {"description": "CURRENT", "score": {"goals": 0, "participant": "home"}},
                    {"description": "CURRENT", "score": {"goals": 0, "participant": "away"}},
                ],
                "participants": [
                    {"id": 100, "meta": {"location": "home"}},
                    {"id": 541, "meta": {"location": "away"}},
                ],
            },
        ]
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", "fake-token"):
            with patch("src.sportsmonk._get", return_value={"data": fixtures_mock}):
                from src.sportsmonk import get_team_form
                forma = get_team_form(541, last=5)

        assert forma is not None
        assert "partidos"    in forma
        assert "W" in forma and "D" in forma and "L" in forma
        assert "gf_promedio" in forma
        assert "gc_promedio" in forma
        assert "btts_rate"   in forma
        assert "over_25_rate" in forma
        assert "secuencia"   in forma
        assert forma["_fuente"] == "sportsmonk"
        assert forma["partidos"] == 2

    def test_get_team_form_sin_scores_devuelve_none(self):
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", "fake-token"):
            with patch("src.sportsmonk._get", return_value={"data": [{"scores": []}]}):
                from src.sportsmonk import get_team_form
                result = get_team_form(541, last=5)
        assert result is None

    def test_get_h2h_schema_correcto(self):
        fixture_h2h = [
            {
                "starting_at": "2024-04-01",
                "scores": [
                    {"description": "CURRENT", "score": {"goals": 3, "participant": "home"}},
                    {"description": "CURRENT", "score": {"goals": 1, "participant": "away"}},
                ],
                "participants": [
                    {"id": 541, "meta": {"location": "home"}},
                    {"id": 529, "meta": {"location": "away"}},
                ],
            }
        ]
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", "fake-token"):
            with patch("src.sportsmonk._get", return_value={"data": fixture_h2h}):
                from src.sportsmonk import get_head_to_head
                h2h = get_head_to_head(541, 529, last=10)

        assert h2h is not None
        assert h2h["n"] == 1
        assert h2h["wins_local_actual"] == 1
        assert h2h["empates"] == 0
        assert h2h["wins_visit_actual"] == 0
        assert h2h["_fuente"] == "sportsmonk"

    def test_contexto_completo_sin_api_error_graceful(self):
        """Si la API HTTP devuelve error, contexto_partido_completo no lanza excepción."""
        with patch("src.sportsmonk.SPORTMONKS_TOKEN", "fake-token"):
            with patch("src.sportsmonk._get", return_value=None):
                from src.sportsmonk import contexto_partido_completo
                ctx = contexto_partido_completo("Arsenal", "Chelsea")
        # No explota, devuelve ctx con notas
        assert "api_disponible" in ctx
        assert isinstance(ctx["notas"], list)


class TestDataSourceManagerSportsmonk:
    """Tests del DSM con Sportmonks integrado."""

    @pytest.fixture
    def dsm_fresco(self):
        from src.data_source_manager import DataSourceManager
        return DataSourceManager(fuente_default="auto")

    # --- fuente explícita "sportsmonk" ---

    def test_fuente_sportsmonk_llama_sm(self, dsm_fresco):
        with patch("src.sportsmonk.contexto_partido_completo",
                   return_value=CTX_SM_COMPLETO) as mock_sm:
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="sportsmonk"
            )
        mock_sm.assert_called_once_with("Real Madrid", "Barcelona")
        assert ctx["fuente"] == "sportsmonk"

    def test_fuente_sportsmonk_error_retorna_empty(self, dsm_fresco):
        with patch("src.sportsmonk.contexto_partido_completo",
                   side_effect=Exception("SM timeout")):
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="sportsmonk"
            )
        assert ctx["api_disponible"] is False
        assert ctx["forma_home"] is None

    # --- cadena auto: api falla → SM rescata ---

    def test_auto_sm_rescata_cuando_api_falla(self, dsm_fresco):
        """api-football sin datos + SM con datos → SM rescata, api_disponible=True."""
        from tests.test_multi_source import CTX_API_VACIO
        with (
            patch("src.api_football.contexto_partido_completo",
                  return_value=CTX_API_VACIO),
            patch("src.sportsmonk.disponible", return_value=True),
            patch("src.sportsmonk.contexto_partido_completo",
                  return_value=CTX_SM_COMPLETO),
        ):
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="auto"
            )

        assert ctx["api_disponible"] is True
        assert ctx["fuente"] == "sportsmonk-fallback"
        assert ctx["forma_home"] is not None
        s = dsm_fresco.stats()
        assert s["fallback_sm"] == 1

    # --- cadena auto: api falla + SM falla → CSV ---

    def test_auto_csv_cuando_api_y_sm_fallan(self, dsm_fresco):
        """api falla + SM falla + TSDB falla → CSV, api_disponible=False."""
        from tests.test_multi_source import CTX_API_VACIO, CTX_CSV_COMPLETO
        ctx_sm_vacio = {**CTX_SM_COMPLETO, "api_disponible": False,
                        "forma_home": None, "forma_away": None}
        ctx_tsdb_vacio = {**CTX_API_VACIO, "fuente": "thesportsdb"}
        with (
            patch("src.api_football.contexto_partido_completo",
                  return_value=CTX_API_VACIO),
            patch("src.sportsmonk.disponible", return_value=True),
            patch("src.sportsmonk.contexto_partido_completo",
                  return_value=ctx_sm_vacio),
            patch("src.thesportsdb.contexto_partido_completo",
                  return_value=ctx_tsdb_vacio),
            patch("src.football_data_api.contexto_partido_completo",
                  return_value=CTX_CSV_COMPLETO),
        ):
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="auto"
            )

        assert ctx["api_disponible"] is False
        assert ctx["fuente"] == "csv-fallback"
        s = dsm_fresco.stats()
        assert s["fallback_sm"] == 1
        assert s["fallback_tsdb"] == 1
        assert s["fallback_csv"] == 1

    # --- cadena auto: SM sin token → se salta ---

    def test_auto_sin_token_sm_se_salta(self, dsm_fresco):
        """Sin SPORTMONKS_TOKEN, el paso SM se salta; TSDB vacio → CSV."""
        from tests.test_multi_source import CTX_API_VACIO, CTX_CSV_COMPLETO
        ctx_tsdb_vacio = {**CTX_API_VACIO, "fuente": "thesportsdb"}
        with (
            patch("src.api_football.contexto_partido_completo",
                  return_value=CTX_API_VACIO),
            patch("src.sportsmonk.disponible", return_value=False),
            patch("src.thesportsdb.contexto_partido_completo",
                  return_value=ctx_tsdb_vacio),
            patch("src.football_data_api.contexto_partido_completo",
                  return_value=CTX_CSV_COMPLETO) as mock_csv,
        ):
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="auto"
            )

        mock_csv.assert_called_once()
        assert ctx["fuente"] == "csv-fallback"
        assert ctx["api_disponible"] is False
        # SM nunca fue llamado → fallback_sm sigue en 0
        assert dsm_fresco.stats()["fallback_sm"] == 0
        assert dsm_fresco.stats()["fallback_tsdb"] == 1

    # --- api_disponible=False cuando todo live falla ---

    def test_api_disponible_false_cuando_sm_sin_token_y_api_falla(self, dsm_fresco):
        """Garantia central: si live APIs no tienen datos, api_disponible=False."""
        from tests.test_multi_source import CTX_API_VACIO, CTX_CSV_COMPLETO
        ctx_tsdb_vacio = {**CTX_API_VACIO, "fuente": "thesportsdb"}
        with (
            patch("src.api_football.contexto_partido_completo",
                  side_effect=Exception("timeout")),
            patch("src.sportsmonk.disponible", return_value=False),
            patch("src.thesportsdb.contexto_partido_completo",
                  return_value=ctx_tsdb_vacio),
            patch("src.football_data_api.contexto_partido_completo",
                  return_value=CTX_CSV_COMPLETO),
        ):
            ctx = dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="auto"
            )
        assert ctx["api_disponible"] is False

    # --- sportsmonk_disponible() en DSM ---

    def test_dsm_sportsmonk_disponible_false_sin_token(self, dsm_fresco):
        with patch("src.sportsmonk.disponible", return_value=False):
            assert dsm_fresco.sportsmonk_disponible() is False

    def test_dsm_sportsmonk_disponible_true_con_token(self, dsm_fresco):
        with patch("src.sportsmonk.disponible", return_value=True):
            assert dsm_fresco.sportsmonk_disponible() is True

    # --- stats incluyen contadores SM ---

    def test_stats_tienen_contadores_sm(self, dsm_fresco):
        s = dsm_fresco.stats()
        assert "delegated_sm" in s
        assert "fallback_sm"  in s
        assert "errors_sm"    in s

    def test_reset_stats_limpia_sm(self, dsm_fresco):
        with (
            patch("src.api_football.contexto_partido_completo",
                  return_value={**CTX_API_VACIO}),
            patch("src.sportsmonk.disponible", return_value=True),
            patch("src.sportsmonk.contexto_partido_completo",
                  return_value=CTX_SM_COMPLETO),
        ):
            dsm_fresco.contexto_partido_completo(
                "Real Madrid", "Barcelona", fuente="auto"
            )
        dsm_fresco.reset_stats()
        s = dsm_fresco.stats()
        assert all(v == 0 for v in s.values())
