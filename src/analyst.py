"""
Analista LLM grounded.
Recibe datos duros (math + API + coherencia) y produce un veredicto en prose.

REGLA FUNDAMENTAL: el LLM solo puede usar los datos que le pasamos en el prompt.
Está prohibido inventar estadísticas, lesiones, alineaciones o H2H.
Si un dato no está en el prompt, el LLM debe decir "no lo sé" o ignorarlo.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GROQ_API_KEY, GROQ_MODEL


_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY)
    except ImportError:
        pass


SYSTEM_PROMPT = """Eres un analista experto en apuestas deportivas de fútbol.

REGLAS INQUEBRANTABLES:
1. Solo puedes usar los datos que el usuario te pasa explícitamente.
2. PROHIBIDO inventar: lesiones, alineaciones, historial que no te dieron,
   estadísticas que no te dieron, nombres de jugadores, partidos específicos
   que no están en los datos, rumores, clima.
3. Si algo no te dieron, di "no tengo ese dato" o ignóralo — NUNCA te lo inventes.
4. Tu trabajo es SINTETIZAR: combinar las probabilidades del modelo, la señal
   del mercado, los datos de forma reciente y H2H, y las banderas de coherencia
   en un veredicto claro y accionable.
5. Escribe en español, tono directo y honesto, sin hype.
6. Si hay banderas de baja confianza, DI CLARAMENTE que no apueste.
7. No uses emojis excesivamente (como mucho 1-2).
8. Longitud: 4-7 frases. Ni más ni menos. Denso y útil.
9. Acaba con un VEREDICTO en una línea: "🟢 Apuesta: <pick>", "🟡 Prueba con stake mínimo", o "🔴 No apuestes".
"""


def construir_prompt(partido: str,
                     cuotas: dict,
                     prob_modelo: dict,
                     picks: list,
                     contexto_api: dict,
                     coherencia: dict) -> str:
    """Arma el user-prompt con todos los datos duros."""
    lineas = [f"PARTIDO: {partido}\n"]

    # Cuotas del mercado
    lineas.append("CUOTAS DEL MERCADO:")
    for mercado, sels in cuotas.items():
        for sel, c in sels.items():
            lineas.append(f"  {mercado} {sel}: {c}")

    # Probabilidades del modelo
    lineas.append("\nPROBABILIDADES DEL MODELO (Poisson derivado del 1X2):")
    for k, v in prob_modelo.items():
        lineas.append(f"  {k}: {v:.1%}")

    # Coherencia
    lineas.append(f"\nCOHERENCIA MODELO ↔ MERCADO: confianza={coherencia['confianza_modelo']}")
    lineas.append(f"  xG modelo: {coherencia['model_xg_total']}")
    if coherencia.get("market_xg_total"):
        lineas.append(f"  xG que el mercado sugiere: ~{coherencia['market_xg_total']}")
    if coherencia["mensajes"]:
        lineas.append("  Observaciones:")
        for m in coherencia["mensajes"]:
            lineas.append(f"   - {m}")

    # Datos reales de API-Football
    if contexto_api.get("api_disponible"):
        fh = contexto_api.get("forma_home")
        fa = contexto_api.get("forma_away")
        if fh:
            lineas.append(
                f"\nFORMA LOCAL (últimos {fh['partidos']}): {fh['secuencia']}, "
                f"GF={fh['gf_promedio']}, GC={fh['gc_promedio']}, "
                f"BTTS={int(fh['btts_rate']*100)}%, O2.5={int(fh['over_25_rate']*100)}%"
            )
        if fa:
            lineas.append(
                f"FORMA VISITANTE (últimos {fa['partidos']}): {fa['secuencia']}, "
                f"GF={fa['gf_promedio']}, GC={fa['gc_promedio']}, "
                f"BTTS={int(fa['btts_rate']*100)}%, O2.5={int(fa['over_25_rate']*100)}%"
            )
        h2 = contexto_api.get("h2h")
        if h2:
            lineas.append(
                f"H2H últimos {h2['n']} cruces: {h2['wins_local_actual']} victorias "
                f"del local actual, {h2['empates']} empates, "
                f"{h2['wins_visit_actual']} del visitante actual. "
                f"Goles promedio={h2['goles_promedio']}, "
                f"BTTS={int(h2['btts_rate']*100)}%, O2.5={int(h2['over_25_rate']*100)}%"
            )
        for nota in contexto_api.get("notas", []):
            lineas.append(f"(nota: {nota})")
    else:
        lineas.append("\n(sin datos de API — no tengo forma reciente ni H2H)")

    # Picks detectados por el motor
    if picks:
        lineas.append("\nPICKS CON EV≥5% QUE DETECTÓ EL MOTOR:")
        for p in picks:
            legs = " + ".join(l["texto"] for l in p["legs"])
            lineas.append(
                f"  [{p['tipo']}] {legs} @ cuota {p['cuota_total']}, "
                f"prob={p['prob']:.1%}, EV={p['ev']:+.1%}, stake={p['stake_sugerido']:.2f}"
            )
    else:
        lineas.append("\nEl motor NO encontró picks con EV≥5% en estas cuotas.")

    lineas.append(
        "\n\nAhora dame tu análisis (4-7 frases) y tu veredicto final. "
        "Si la confianza del modelo es 'baja' y el mercado sugiere lo contrario "
        "al pick, SÉ MUY CLARO diciendo que no apueste. Si hay coherencia y el "
        "EV es real, recomienda el pick concreto."
    )
    return "\n".join(lineas)


def analizar(partido: str, cuotas: dict, prob_modelo: dict,
             picks: list, contexto_api: dict, coherencia: dict) -> str | None:
    """Devuelve el veredicto del analista LLM. None si no hay LLM disponible."""
    if _client is None:
        return None
    prompt = construir_prompt(partido, cuotas, prob_modelo, picks,
                              contexto_api, coherencia)
    try:
        resp = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(El analista LLM falló: {e})"


def fallback_sin_llm(coherencia: dict, picks: list) -> str:
    """
    Si no hay LLM configurado, generamos un veredicto basado en reglas
    sobre coherencia. No es tan rico como el LLM pero sigue siendo útil.
    """
    confianza = coherencia["confianza_modelo"]

    if not picks:
        base = ("El motor no encontró picks con EV≥5% en estas cuotas. "
                "Normalmente eso significa que la casa tiene precio eficiente; "
                "mejor no apostar.")
        return base + "\n\n🔴 No apuestes."

    if confianza == "baja":
        mensaje = (
            "El motor encontró picks con EV positivo, PERO la coherencia entre "
            "modelo y mercado es baja. Eso significa que el EV probablemente "
            "es un artefacto del modelo, no edge real.\n\n"
        )
        for m in coherencia["mensajes"]:
            mensaje += f"• {m}\n"
        return mensaje + "\n🔴 No apuestes estos picks."

    # Confianza media o alta
    top = picks[0]
    legs = " + ".join(l["texto"] for l in top["legs"])
    mensaje = (
        f"El motor encontró {len(picks)} pick(s) con EV positivo y la "
        f"coherencia modelo-mercado es {confianza}. "
        f"Mejor pick: {top['tipo']} → {legs} @ {top['cuota_total']} "
        f"(EV {top['ev']*100:+.1f}%, stake {top['stake_sugerido']:.2f}).\n\n"
    )
    for m in coherencia["mensajes"][:2]:
        mensaje += f"• {m}\n"

    if confianza == "alta":
        mensaje += f"\n🟢 Apuesta: {legs} @ {top['cuota_total']}"
    else:
        mensaje += f"\n🟡 Considera con stake reducido (medio Kelly o menos)."
    return mensaje
