# montecarlo.py

import numpy as np

def simular_partido_montecarlo(lambda_h, lambda_a, iteraciones=50000):
    goles_h = np.random.poisson(lambda_h, iteraciones)
    goles_a = np.random.poisson(lambda_a, iteraciones)
    
    resultados = {
        "1X2": {
            "1": float(np.sum(goles_h > goles_a) / iteraciones),
            "X": float(np.sum(goles_h == goles_a) / iteraciones),
            "2": float(np.sum(goles_h < goles_a) / iteraciones)
        },
        "OU_2.5": {
            "Over": float(np.sum((goles_h + goles_a) > 2) / iteraciones),
            "Under": float(np.sum((goles_h + goles_a) <= 2) / iteraciones)
        },
        "BTTS": {
            "Yes": float(np.sum((goles_h > 0) & (goles_a > 0)) / iteraciones),
            "No": float(np.sum((goles_h == 0) | (goles_a == 0)) / iteraciones)
        },
        "AH_1.5": {
            "Home": float(np.sum((goles_h - goles_a) > 1) / iteraciones),
            "Away": float(np.sum((goles_h - goles_a) <= 1) / iteraciones)
        },
        "AH_2.0": {
            "Home": float(np.sum((goles_h - goles_a) > 2) / iteraciones),
            "Push": float(np.sum((goles_h - goles_a) == 2) / iteraciones),
            "Away": float(np.sum((goles_h - goles_a) < 2) / iteraciones)
        }
    }
    return resultados