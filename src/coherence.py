"""
Detector de coherencia entre:
  - Lo que dicen las cuotas del MERCADO (distintos mercados entre sí)
  - Lo que dice nuestro modelo Poisson derivado del 1X2

La idea es flagear automáticamente los casos peligrosos tipo:
  "El 1X2 es equilibrado pero Over 2.5 está a 1.46 → la casa espera más
  goles de los que el Poisson derivado puede saber → desconfía del EV
  que salga en OU/BTTS."

Uso:
    flags = evaluar_coherencia(cuotas, lambdas_modelo, forma_home, forma_away, h2h)
"""
from probability_engine import eliminar_vig


def _sin_vig_2way(c1: float, c2: float) -> tuple[float, float]:
    imp1, imp2 = 1 / c1, 1 / c2
    s = imp1 + imp2
    return imp1 / s, imp2 / s


def evaluar_coherencia(cuotas: dict,
                       lambdas_modelo: dict,
                       forma_home: dict | None = None,
                       forma_away: dict | None = None,
                       h2h: dict | None = None) -> dict:
    """
    Devuelve:
      {
        "flags":   ["market_over_sesgado", "1x2_muy_equilibrado", ...],
        "mensajes": [texto legible por cada flag],
        "market_xg_total": <float | None>,
        "model_xg_total":  <float>,
        "discrepancia_xg": <float>,
        "confianza_modelo": <"alta" | "media" | "baja">,
      }
    """
    flags = []
    mensajes = []

    model_xg_total = (lambdas_modelo["home"] + lambdas_modelo["away"])
    market_xg_total = None

    # -------- 1X2 -----------------------------------------------------
    c_1x2 = cuotas.get("1X2") or {}
    if c_1x2:
        sv = eliminar_vig({"1": c_1x2["1"], "X": c_1x2["X"], "2": c_1x2["2"]})
        max_p = max(sv.values())
        if max_p < 0.45:
            flags.append("1x2_muy_equilibrado")
            mensajes.append(
                "El 1X2 está muy parejo (ningún favorito claro), por lo que "
                "el xG derivado puede no reflejar bien el potencial real de ataque."
            )
        if max_p > 0.65:
            flags.append("1x2_favorito_claro")
            mensajes.append(
                "Hay un favorito muy claro en 1X2, los goles esperados suelen "
                "concentrarse en el favorito."
            )

    # -------- Over 2.5 vs Poisson -------------------------------------
    c_ou = cuotas.get("OU_2.5")
    if c_ou:
        over, under = _sin_vig_2way(c_ou["Over"], c_ou["Under"])
        # Aproximación: si el mercado da Over > 0.60, implica xG total ≈ 3.2+
        # Si Over < 0.45, implica xG total ≈ 2.2-
        if over >= 0.60:
            market_xg_total_est = 3.2
        elif over >= 0.55:
            market_xg_total_est = 2.9
        elif over >= 0.48:
            market_xg_total_est = 2.7
        elif over >= 0.42:
            market_xg_total_est = 2.5
        else:
            market_xg_total_est = 2.2
        market_xg_total = market_xg_total_est

        diff = market_xg_total_est - model_xg_total
        if diff > 0.4:
            flags.append("market_espera_mas_goles")
            mensajes.append(
                f"El mercado espera ~{market_xg_total_est:.1f} goles totales "
                f"(Over 2.5 a {c_ou['Over']}), pero nuestro modelo solo deriva "
                f"{model_xg_total:.2f} del 1X2. DESCONFÍA de los EVs que apunten "
                f"a Under — probablemente son artefactos del modelo."
            )
        elif diff < -0.4:
            flags.append("market_espera_menos_goles")
            mensajes.append(
                f"El mercado espera pocos goles (Under alto, ~{market_xg_total_est:.1f} "
                f"totales) pero el modelo deriva {model_xg_total:.2f}. "
                f"Desconfía de EVs que apunten a Over."
            )
        else:
            flags.append("ou_coherente")
            mensajes.append(
                f"Mercado OU y Poisson están alineados (~{market_xg_total_est:.1f} "
                f"goles esperados). El EV en OU/BTTS es confiable."
            )

    # -------- Datos reales (forma + H2H) ------------------------------
    if forma_home and forma_away:
        gf_tot = forma_home.get("gf_promedio", 0) + forma_away.get("gf_promedio", 0)
        gc_tot = forma_home.get("gc_promedio", 0) + forma_away.get("gc_promedio", 0)
        # Proxy simple de goles esperados = (GF home + GC away)/2 + (GF away + GC home)/2
        xg_real = (
            (forma_home["gf_promedio"] + forma_away["gc_promedio"]) / 2 +
            (forma_away["gf_promedio"] + forma_home["gc_promedio"]) / 2
        )
        diff_real = xg_real - model_xg_total
        if diff_real > 0.4:
            flags.append("forma_sugiere_mas_goles")
            mensajes.append(
                f"La forma reciente sugiere ~{xg_real:.1f} goles esperados "
                f"vs {model_xg_total:.2f} del modelo derivado de cuotas."
            )
        if diff_real < -0.4:
            flags.append("forma_sugiere_menos_goles")
            mensajes.append(
                f"La forma reciente sugiere ~{xg_real:.1f} goles vs "
                f"{model_xg_total:.2f} del modelo."
            )

    if h2h and h2h.get("n", 0) >= 5:
        if h2h["over_25_rate"] >= 0.70:
            flags.append("h2h_goleado")
            mensajes.append(
                f"El H2H histórico ha pasado de 2.5 goles en "
                f"{int(h2h['over_25_rate']*100)}% de los últimos {h2h['n']} cruces."
            )
        elif h2h["over_25_rate"] <= 0.30:
            flags.append("h2h_cerrado")
            mensajes.append(
                f"Los últimos {h2h['n']} cruces han sido cerrados: solo "
                f"{int(h2h['over_25_rate']*100)}% pasaron de 2.5 goles."
            )
        if h2h["btts_rate"] >= 0.70:
            flags.append("h2h_btts_si")
        elif h2h["btts_rate"] <= 0.30:
            flags.append("h2h_btts_no")

    # -------- Confianza general ---------------------------------------
    peligros = {"market_espera_mas_goles", "market_espera_menos_goles",
                "1x2_muy_equilibrado"}
    n_peligros = sum(1 for f in flags if f in peligros)
    if n_peligros >= 2:
        confianza = "baja"
    elif n_peligros == 1:
        confianza = "media"
    else:
        confianza = "alta"

    return {
        "flags": flags,
        "mensajes": mensajes,
        "market_xg_total": market_xg_total,
        "model_xg_total": round(model_xg_total, 2),
        "confianza_modelo": confianza,
    }
