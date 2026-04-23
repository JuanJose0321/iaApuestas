from itertools import combinations

def optimizar_parleys(lista_picks, tamano=2):
    """
    Maximiza la probabilidad conjunta de combinaciones.
    """
    mejores_parleys = []
    
    for combo in combinations(lista_picks, tamano):
        prob_conjunta = 1.0
        cuota_total = 1.0
        nombres = []
        
        for pick in combo:
            prob_conjunta *= pick['prob_real']
            cuota_total *= pick['cuota']
            nombres.append(f"{pick['equipo']} ({pick['cuota']})")
        
        mejores_parleys.append({
            "picks": " + ".join(nombres),
            "prob_conjunta": prob_conjunta,
            "cuota_total": cuota_total
        })
    
    # Ordenar por mayor probabilidad conjunta
    return sorted(mejores_parleys, key=lambda x: x['prob_conjunta'], reverse=True)