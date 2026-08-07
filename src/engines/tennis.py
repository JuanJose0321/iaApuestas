"""
Motor de análisis de tenis — Match Winner, Set Handicap, Total Games,
Game Handicap, Set-by-Set Winner, Multi-leg picks.

Modelo: Elo superficial (Bradley-Terry) — sin Poisson.
Cuotas: formato decimal europeo.
"""
import logging
from itertools import combinations
from scipy.stats import norm
import numpy as np

_log = logging.getLogger("betbrain.tennis")

# ── Constantes ────────────────────────────────────────────────────────────────

SURFACE_ELO_FACTOR = {
    "clay":   1.20,
    "hard":   1.00,
    "grass":  0.85,
    "carpet": 0.90,
}

KELLY_FRACTION = 0.25
MIN_EV         = 0.05

STD_GAMES    = {"best_of_3": 4.5,  "best_of_5": 6.0}
STD_NET_DIFF = {"best_of_3": 3.5,  "best_of_5": 5.0}

UMBRAL_VERDE    = 0.65
UMBRAL_AMARILLO = 0.50

# Categorías incompatibles para multi-leg (muy correlacionadas)
_INCOMPATIBLE = {
    frozenset({"match_winner", "set_handicap"}),
    frozenset({"match_winner", "game_handicap"}),
    frozenset({"game_handicap", "total_games"}),
}


class TennisEngine:
    """Motor de análisis para partidos de tenis."""

    # ── Probabilidad de set ───────────────────────────────────────────────────

    def prob_ganar_set(self, elo1: float, elo2: float,
                       superficie: str = "hard") -> float:
        """P(jugador 1 gana un set) usando Elo ajustado por superficie."""
        factor = SURFACE_ELO_FACTOR.get(superficie.lower(), 1.0)
        delta  = (elo1 - elo2) * factor
        return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))

    # ── Match Winner ──────────────────────────────────────────────────────────

    def prob_match_winner(self, p_set: float,
                          formato: str = "best_of_3") -> dict:
        """P(ganar partido) — combinatoria exacta para BO3 y BO5."""
        q = 1.0 - p_set
        if formato == "best_of_5":
            p_win = p_set**3 + 3*p_set**3*q + 6*p_set**3*q**2
        else:
            p_win = p_set**2 + 2*p_set**2*q
        return {"prob_j1": round(p_win, 4), "prob_j2": round(1.0 - p_win, 4)}

    # ── Set Handicap (línea variable) ─────────────────────────────────────────

    def prob_set_handicap(self, p_set: float, handicap: float,
                          formato: str = "best_of_3") -> dict:
        """
        P(cubrir hándicap de sets).

        Hándicap negativo = barrida (ganar sin ceder set).
        Hándicap positivo = no ser barrido (ganar al menos un set).

        Líneas soportadas: ±0.5, ±1.5, ±2.5.
        """
        q = 1.0 - p_set
        if handicap <= -2.5:
            # BO5 únicamente: ganar 3-0
            p_cubre = p_set**3
        elif handicap <= -1.5:
            # BO3: ganar 2-0 | BO5: ganar 3-0 o 3-1
            if formato == "best_of_5":
                p_cubre = p_set**3 + 3*p_set**3*q
            else:
                p_cubre = p_set**2
        elif handicap <= -0.5:
            # Ganar el partido (ya contemplado en match_winner)
            p_cubre = self.prob_match_winner(p_set, formato)["prob_j1"]
        elif handicap >= 2.5:
            # BO5: no ser barrido 3-0 ni 3-1
            p_no_cubre = q**3 + 3*q**3*p_set
            p_cubre = 1.0 - p_no_cubre
        elif handicap >= 1.5:
            # BO3: ganar al menos 1 set | BO5: ganar al menos 1 set
            p_no_cubre = q**3 if formato == "best_of_5" else q**2
            p_cubre = 1.0 - p_no_cubre
        else:
            # +0.5: no ser barrido (= ganar el partido o perder 0-2 en BO3)
            p_no_cubre = q**2 if formato == "best_of_3" else q**3
            p_cubre = 1.0 - p_no_cubre

        return {
            "prob_cubre":    round(p_cubre, 4),
            "prob_no_cubre": round(1.0 - p_cubre, 4),
            "handicap":      handicap,
        }

    # ── Total Games ───────────────────────────────────────────────────────────

    def prob_total_games(self, p_set: float,
                         formato: str = "best_of_3") -> dict:
        """Distribución Normal del total de games."""
        dominancia    = abs(p_set - 0.5) * 2.0
        games_por_set = 10.0 - 3.0 * dominancia
        p, q = p_set, 1.0 - p_set
        sets_esp = (3.0 + 3.0*p*q) if formato == "best_of_5" else (2.0 + 2.0*p*q)
        return {
            "total_esp":      round(sets_esp * games_por_set, 1),
            "std_dev":        STD_GAMES.get(formato, 4.5),
            "games_por_set":  round(games_por_set, 1),
            "sets_esperados": round(sets_esp, 2),
        }

    # ── Game Handicap ─────────────────────────────────────────────────────────

    def prob_game_handicap(self, p_set: float, formato: str,
                           handicap: float) -> dict:
        """
        P(j1 gana por más de `handicap` games en total).

        Modelo: diferencia neta de games ≈ Normal(
            mean = (2p-1) * total_games/2,
            std  = std_net_diff
        )

        Handicap positivo → j1 debe ganar por X+ games.
        Handicap negativo → j2 debe ganar por |X|+ games (convenio apuesta j2).
        """
        dist    = self.prob_total_games(p_set, formato)
        total   = dist["total_esp"]
        net_esp = (2.0 * p_set - 1.0) * total * 0.5
        std     = STD_NET_DIFF.get(formato, 3.5)

        prob_j1 = float(1.0 - norm.cdf(handicap, loc=net_esp, scale=std))
        prob_j2 = float(1.0 - norm.cdf(-handicap, loc=-net_esp, scale=std))

        return {
            "net_esp":     round(net_esp, 1),
            "prob_j1_cubre": round(prob_j1, 4),
            "prob_j2_cubre": round(prob_j2, 4),
            "handicap":    handicap,
        }

    # ── Primer Set ────────────────────────────────────────────────────────────

    def prob_primer_set(self, p_set: float) -> dict:
        return {"prob_j1": round(p_set, 4), "prob_j2": round(1.0 - p_set, 4)}

    # ── Set N (winner de un set específico) ───────────────────────────────────

    def prob_set_n_winner(self, p_set: float, n: int,
                          formato: str = "best_of_3") -> dict:
        """
        P(j1 gana el set N) — asume independencia entre sets.

        También devuelve P(el partido llega al set N), útil para validar
        que la apuesta tiene sentido.
        """
        p_win  = self.prob_match_winner(p_set, formato)["prob_j1"]
        q      = 1.0 - p_set

        # P(partido llega al set N)
        if formato == "best_of_5":
            p_llega = {
                1: 1.0,
                2: 1.0,
                3: 1.0,
                4: 1.0 - p_set**3 - q**3,          # alguien no barró 3-0
                5: 6.0 * p_set**2 * q**2,           # todos los 3-2 llegan al 5
            }.get(n, 0.0)
        else:  # best_of_3
            p_llega = {
                1: 1.0,
                2: 1.0,
                3: 2.0 * p_set * q,                 # P(1-1 after 2 sets)
            }.get(n, 0.0)

        # P(j1 gana set N) ≈ p_set (independencia)
        return {
            "prob_j1":     round(p_set, 4),
            "prob_j2":     round(q, 4),
            "p_llega_set": round(p_llega, 4),
            "set_n":       n,
        }

    # ── Evaluación de Value ───────────────────────────────────────────────────

    def evaluar_value(self, prob: float, cuota: float) -> dict:
        """EV + Kelly fraccional para cuota decimal."""
        if cuota <= 1.0:
            return {"prob_implicita": 1.0, "ev": -1.0,
                    "kelly_pct": 0.0, "tiene_value": False}
        prob_impl = 1.0 / cuota
        ev        = prob * cuota - 1.0
        b         = cuota - 1.0
        q         = 1.0 - prob
        kelly_full = (prob * b - q) / b if b > 0 else 0.0
        kelly_frac = max(0.0, kelly_full * KELLY_FRACTION)
        return {
            "prob_implicita": round(prob_impl, 4),
            "ev":             round(ev, 4),
            "kelly_pct":      round(kelly_frac, 4),
            "tiene_value":    ev >= MIN_EV,
        }

    # ── Pick Generation ───────────────────────────────────────────────────────

    def generar_picks(self,
                      jugador1: str, jugador2: str,
                      elo1: float, elo2: float,
                      superficie: str, formato: str,
                      cuotas: dict,
                      cuota_min: float = 1.20,
                      cuota_max: float = 6.00) -> list:
        """
        Genera picks simples (un mercado cada uno) con value.

        Cuotas aceptadas:
            match_winner:  {"1": 1.65, "2": 2.20}
            set_handicap:  {"-1.5": 2.10, "+1.5": 1.70, "-2.5": 3.20, ...}
            total_games:   {"linea": 22.5, "over": 1.85, "under": 1.95}
            primer_set:    {"1": 1.55, "2": 2.40}
            game_handicap: {"linea": 4.5, "j1": 2.10, "j2": 1.75}
            sets_winners:  {"2": {"1": 1.80, "2": 2.00}, "3": {...}}
        """
        picks  = []
        p_set  = self.prob_ganar_set(elo1, elo2, superficie)

        def _add(mercado_key, mercado_label, prob, cuota, pick_desc, extra=None):
            c = float(cuota)
            if not (cuota_min <= c <= cuota_max):
                return
            val = self.evaluar_value(prob, c)
            if not val["tiene_value"]:
                return
            confianza = self._calc_confianza(prob, val["ev"])
            p = {
                "mercado":        mercado_label,
                "mercado_key":    mercado_key,
                "pick":           pick_desc,
                "prob":           round(prob, 4),
                "cuota":          c,
                "ev":             val["ev"],
                "kelly_pct":      val["kelly_pct"],
                "confianza":      confianza,
                "confianza_nivel": self._nivel_confianza(confianza),
            }
            if extra:
                p.update(extra)
            picks.append(p)

        # 1. Match Winner
        mw = cuotas.get("match_winner", {})
        probs_mw = self.prob_match_winner(p_set, formato)
        for jid, pk in [("1", "prob_j1"), ("2", "prob_j2")]:
            if jid in mw:
                nombre = jugador1 if jid == "1" else jugador2
                _add("match_winner", "Match Winner",
                     probs_mw[pk], mw[jid],
                     f"{nombre} gana el partido",
                     {"jugador": jid})

        # 2. Set Handicap (líneas variables)
        sh = cuotas.get("set_handicap", {})
        for linea_str, cuota_sh in sh.items():
            try:
                linea = float(linea_str)
            except ValueError:
                continue
            pr = self.prob_set_handicap(p_set, linea, formato)
            nombre = jugador1 if linea < 0 else jugador2
            sentido = jugador1 if linea < 0 else jugador2
            _add("set_handicap", "Set Handicap",
                 pr["prob_cubre"], cuota_sh,
                 f"{sentido} {linea_str:+} sets",
                 {"handicap": linea})

        # 3. Total Games
        tg = cuotas.get("total_games", {})
        if "linea" in tg:
            dist = self.prob_total_games(p_set, formato)
            linea_tg = float(tg["linea"])
            p_over  = float(1.0 - norm.cdf(linea_tg, loc=dist["total_esp"],
                                            scale=dist["std_dev"]))
            p_under = 1.0 - p_over
            if "over" in tg:
                _add("total_games", f"Total Games Over {linea_tg}",
                     p_over, tg["over"], f"Over {linea_tg} games",
                     {"total_esp": dist["total_esp"]})
            if "under" in tg:
                _add("total_games", f"Total Games Under {linea_tg}",
                     p_under, tg["under"], f"Under {linea_tg} games",
                     {"total_esp": dist["total_esp"]})

        # 4. Primer Set
        ps = cuotas.get("primer_set", {})
        probs_ps = self.prob_primer_set(p_set)
        for jid, pk in [("1", "prob_j1"), ("2", "prob_j2")]:
            if jid in ps:
                nombre = jugador1 if jid == "1" else jugador2
                _add("primer_set", "Primer Set",
                     probs_ps[pk], ps[jid],
                     f"{nombre} gana el primer set",
                     {"jugador": jid})

        # 5. Game Handicap
        gh = cuotas.get("game_handicap", {})
        if "linea" in gh:
            linea_gh = float(gh["linea"])
            pr_gh    = self.prob_game_handicap(p_set, formato, linea_gh)
            if "j1" in gh:
                _add("game_handicap", f"Game Handicap -{linea_gh}",
                     pr_gh["prob_j1_cubre"], gh["j1"],
                     f"{jugador1} gana por más de {linea_gh} games",
                     {"net_esp": pr_gh["net_esp"]})
            if "j2" in gh:
                _add("game_handicap", f"Game Handicap -{linea_gh}",
                     pr_gh["prob_j2_cubre"], gh["j2"],
                     f"{jugador2} gana por más de {linea_gh} games",
                     {"net_esp": pr_gh["net_esp"]})

        # 6. Sets Winners (set 2, set 3, …)
        sw = cuotas.get("sets_winners", {})
        for set_str, opciones in sw.items():
            try:
                n = int(set_str)
            except ValueError:
                continue
            pr_sn = self.prob_set_n_winner(p_set, n, formato)
            for jid, pk in [("1", "prob_j1"), ("2", "prob_j2")]:
                if jid in opciones:
                    nombre = jugador1 if jid == "1" else jugador2
                    _add("set_n_winner", f"Ganador Set {n}",
                         pr_sn[pk], opciones[jid],
                         f"{nombre} gana el set {n}",
                         {"set_n": n, "p_llega": pr_sn["p_llega_set"]})

        return picks

    # ── Multi-leg picks ───────────────────────────────────────────────────────

    def generar_multileg(self,
                         picks_singles: list,
                         cuota_min: float = 2.00,
                         cuota_max: float = 8.00) -> list:
        """
        Combina picks simples en duplas (2 legs) con value.

        Solo combina mercados compatibles (no correlacionados en exceso).
        Usa independencia de probabilidades — la casa puede ajustar la cuota.
        """
        multileg = []
        for p1, p2 in combinations(picks_singles, 2):
            cat1 = p1["mercado_key"]
            cat2 = p2["mercado_key"]
            if frozenset({cat1, cat2}) in _INCOMPATIBLE:
                continue
            # No combinar mismo mercado
            if cat1 == cat2:
                continue

            cuota_total = round(p1["cuota"] * p2["cuota"], 3)
            if not (cuota_min <= cuota_total <= cuota_max):
                continue

            prob_conj = round(p1["prob"] * p2["prob"], 4)
            val       = self.evaluar_value(prob_conj, cuota_total)
            if not val["tiene_value"]:
                continue

            confianza = self._calc_confianza(prob_conj, val["ev"])
            multileg.append({
                "tipo":             "DUPLA",
                "legs": [
                    {"mercado": p1["mercado"], "pick": p1["pick"],
                     "cuota": p1["cuota"], "prob": p1["prob"]},
                    {"mercado": p2["mercado"], "pick": p2["pick"],
                     "cuota": p2["cuota"], "prob": p2["prob"]},
                ],
                "cuota_total":      cuota_total,
                "prob":             prob_conj,
                "ev":               val["ev"],
                "kelly_pct":        val["kelly_pct"],
                "confianza":        confianza,
                "confianza_nivel":  self._nivel_confianza(confianza),
                "nota":             "Parlay — casas pueden ajustar cuota real.",
            })

        multileg.sort(key=lambda x: x["ev"], reverse=True)
        return multileg

    # ── Análisis completo ─────────────────────────────────────────────────────

    def analizar(self,
                 jugador1: str, jugador2: str,
                 elo1: float, elo2: float,
                 superficie: str, formato: str,
                 cuotas: dict,
                 cuota_min: float = 1.20,
                 cuota_max: float = 6.00) -> dict:
        """
        Análisis completo: modelo + picks clasificados + multi-leg.

        Returns:
            {
              partido, superficie, formato, elo, modelo,
              picks_verdes, picks_amarillos, picks_descartados,
              picks_multileg, resumen
            }
        """
        p_set  = self.prob_ganar_set(elo1, elo2, superficie)
        modelo = {
            "p_set_j1":     round(p_set, 4),
            "match_winner": self.prob_match_winner(p_set, formato),
            "total_games":  self.prob_total_games(p_set, formato),
            "primer_set":   self.prob_primer_set(p_set),
        }

        todos = self.generar_picks(
            jugador1, jugador2, elo1, elo2,
            superficie, formato, cuotas, cuota_min, cuota_max,
        )

        verdes      = [p for p in todos if p["confianza"] >= UMBRAL_VERDE]
        amarillos   = [p for p in todos if UMBRAL_AMARILLO <= p["confianza"] < UMBRAL_VERDE]
        descartados = [p for p in todos if p["confianza"] < UMBRAL_AMARILLO]

        multileg = self.generar_multileg(todos)

        n_valor = len(verdes) + len(amarillos)
        return {
            "partido":          f"{jugador1} vs {jugador2}",
            "superficie":       superficie,
            "formato":          formato,
            "elo":              {"j1": elo1, "j2": elo2},
            "modelo":           modelo,
            "picks_verdes":     verdes,
            "picks_amarillos":  amarillos,
            "picks_descartados": descartados,
            "picks_multileg":   multileg,
            "resumen":          f"{n_valor} pick(s) con value ({len(verdes)} verde, "
                                f"{len(amarillos)} amarillo)",
        }

    # ── Stake ─────────────────────────────────────────────────────────────────

    def stake_recomendado(self, bankroll: float, prob: float,
                          cuota: float) -> float:
        """Stake via Kelly fraccional (25%)."""
        b = cuota - 1.0
        if b <= 0:
            return 0.0
        q          = 1.0 - prob
        kelly_full = (prob * b - q) / b
        return round(bankroll * max(0.0, kelly_full * KELLY_FRACTION), 2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _calc_confianza(self, prob: float, ev: float) -> float:
        if ev <= 0:
            return 0.0
        certeza  = abs(prob - 0.5) * 2.0
        ev_bonus = min(ev * 0.5, 0.15)
        return round(min(certeza + ev_bonus, 0.99), 3)

    def _nivel_confianza(self, score: float) -> str:
        if score >= UMBRAL_VERDE:
            return "verde"
        if score >= UMBRAL_AMARILLO:
            return "amarillo"
        return "rojo"
