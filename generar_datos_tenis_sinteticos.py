#!/usr/bin/env python3
"""
Genera datos sintéticos de partidos de tenis basados en rankings reales.

Crea un CSV con partidos simulados usando Elo/ranking actual.
"""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# Top jugadores ATP 2026
ATP_TOP = [
    "Jannik Sinner", "Carlos Alcaraz", "Alexander Zverev",
    "Novak Djokovic", "Daniil Medvedev", "Ben Shelton",
    "Taylor Fritz", "Alex de Minaur", "Lorenzo Musetti",
    "Daniil Medvedev", "Andriy Rublev", "Jakub Mensik",
    "Casper Ruud", "Arthur Fils", "Grigor Dimitrov",
    "Tommy Paul", "Felix Auger-Aliassime", "Sebastian Korda",
    "Stefanos Tsitsipas", "Hubert Hurkacz"
]

# Top jugadores WTA 2026
WTA_TOP = [
    "Aryna Sabalenka", "Elena Rybakina", "Iga Swiatek",
    "Coco Gauff", "Jessica Pegula", "Amanda Anisimova",
    "Mirra Andreeva", "Jasmine Paolini", "Victoria Azarenka",
    "Elina Svitolina", "Karolina Muchova", "Belinda Bencic",
    "Madison Keys", "Marketa Vondrousova", "Jeļena Ostapenko",
    "Magdalena Fręch", "Anna Karolina Schmiedlova", "Liudmila Samsonova",
    "Beatriz Haddad Maia", "Diana Shnaider"
]

TOURNAMENTS = [
    "Australian Open", "Wimbledon", "US Open", "French Open",
    "Miami Masters", "Monte Carlo Masters", "Madrid Masters", "Rome Masters",
    "Cincinnati Masters", "Canada Masters", "Shanghai Masters", "Paris Masters",
]

TOURNAMENTS_LEVEL = {
    "Australian Open": "Grand Slam",
    "Wimbledon": "Grand Slam",
    "US Open": "Grand Slam",
    "French Open": "Grand Slam",
    "Miami Masters": "Masters 1000",
    "Monte Carlo Masters": "Masters 1000",
    "Madrid Masters": "Masters 1000",
    "Rome Masters": "Masters 1000",
    "Cincinnati Masters": "Masters 1000",
    "Canada Masters": "Masters 1000",
    "Shanghai Masters": "Masters 1000",
    "Paris Masters": "Masters 1000",
}


def generar_matches_sinteticos(num_matches=500):
    """Genera matches sintéticos pero realistas."""
    matches = []
    start_date = datetime(2024, 1, 1)

    for i in range(num_matches):
        # Seleccionar género aleatorio
        is_atp = random.choice([True, False])
        players = ATP_TOP if is_atp else WTA_TOP

        # Seleccionar dos jugadores diferentes
        p1, p2 = random.sample(players, 2)

        # Ranking (posición aproximada)
        rank1 = players.index(p1) + 1
        rank2 = players.index(p2) + 1

        # Mayor probabilidad de ganar para mejor ranking
        prob_p1 = rank2 / (rank1 + rank2)

        # Simular resultado
        if random.random() < prob_p1:
            winner, loser = p1, p2
            sets_w, sets_l = random.choice([(2, 0), (2, 1)])
        else:
            winner, loser = p2, p1
            sets_w, sets_l = random.choice([(2, 0), (2, 1)])

        # Generar score ficticio
        games = []
        for _ in range(sets_w):
            games.append(str(random.randint(6, 7)))
        for _ in range(sets_l):
            games.append(str(random.randint(3, 5)))

        # Si es 3er set (tie-break)
        if sets_w == 2 and sets_l == 1:
            games.append(str(random.randint(10, 13)))

        score = " ".join(games)

        # Fecha y torneo
        date = start_date + timedelta(days=i * 2)
        tournament = random.choice(TOURNAMENTS)
        level = TOURNAMENTS_LEVEL.get(tournament, "ATP 250")

        matches.append({
            "tourney_date": date.strftime("%Y%m%d"),
            "tourney_name": tournament,
            "tourney_level": level,
            "winner_name": winner,
            "loser_name": loser,
            "score": score,
            "gender": "M" if is_atp else "W",
        })

    return matches


def guardar_matches(matches, filename):
    """Guarda matches en CSV."""
    fieldnames = ["tourney_date", "tourney_name", "tourney_level", "winner_name", "loser_name", "score", "gender"]

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)

    print(f"✅ Guardado: {filename}")
    print(f"   Total: {len(matches)} matches")


if __name__ == "__main__":
    print("Generando datos sintéticos de tenis...")
    matches = generar_matches_sinteticos(500)

    # Guardar para ATP y WTA
    guardar_matches(matches, "src/data/tennis/matches_atp_sintetic.csv")
    print("\n✅ Datos sintéticos generados correctamente")
