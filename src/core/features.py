"""
Feature engineering para fútbol.
Calcula rachas (forma) y promedios de goles anotados/recibidos por equipo,
correctamente separando local y visitante sin data-leakage.
"""
import pandas as pd
import numpy as np


def _stats_por_equipo(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Construye una tabla larga (una fila por equipo-partido) con puntos y goles,
    calcula rolling means SHIFTADAS (sin el partido actual) y devuelve el dataframe.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    # Construir dos DataFrames (uno por cada "lado") y concatenar
    home = pd.DataFrame({
        "match_idx": df.index,
        "Date": df["Date"],
        "team": df["HomeTeam"],
        "rival": df["AwayTeam"],
        "is_home": 1,
        "goles_a_favor": df["FTHG"],
        "goles_en_contra": df["FTAG"],
        "pts": df["FTR"].map({"H": 3, "D": 1, "A": 0}),
    })
    away = pd.DataFrame({
        "match_idx": df.index,
        "Date": df["Date"],
        "team": df["AwayTeam"],
        "rival": df["HomeTeam"],
        "is_home": 0,
        "goles_a_favor": df["FTAG"],
        "goles_en_contra": df["FTHG"],
        "pts": df["FTR"].map({"A": 3, "D": 1, "H": 0}),
    })
    largo = pd.concat([home, away], ignore_index=True).sort_values(["team", "Date"])

    # Rolling por equipo, SHIFT(1) para no ver el partido actual
    g = largo.groupby("team", group_keys=False)
    largo[f"form_{window}"] = g["pts"].apply(lambda s: s.shift(1).rolling(window).mean())
    largo[f"gf_{window}"] = g["goles_a_favor"].apply(lambda s: s.shift(1).rolling(window).mean())
    largo[f"gc_{window}"] = g["goles_en_contra"].apply(lambda s: s.shift(1).rolling(window).mean())

    return largo


def calculate_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Añade al dataframe original las columnas de forma y goles rolling
    para equipo local y visitante. Elimina filas sin historial suficiente.
    """
    largo = _stats_por_equipo(df, window=window)

    # Pivotar de vuelta: una fila por partido con columnas de home y away
    home_stats = (
        largo[largo["is_home"] == 1]
        .set_index("match_idx")[[f"form_{window}", f"gf_{window}", f"gc_{window}"]]
        .rename(columns={
            f"form_{window}": f"Home_Form_{window}",
            f"gf_{window}": f"Home_GF_{window}",
            f"gc_{window}": f"Home_GC_{window}",
        })
    )
    away_stats = (
        largo[largo["is_home"] == 0]
        .set_index("match_idx")[[f"form_{window}", f"gf_{window}", f"gc_{window}"]]
        .rename(columns={
            f"form_{window}": f"Away_Form_{window}",
            f"gf_{window}": f"Away_GF_{window}",
            f"gc_{window}": f"Away_GC_{window}",
        })
    )

    out = df.copy().sort_values("Date").reset_index(drop=True)
    out = out.join(home_stats).join(away_stats)
    return out.dropna(subset=[
        f"Home_Form_{window}", f"Away_Form_{window}",
        f"Home_GF_{window}", f"Away_GF_{window}",
    ]).reset_index(drop=True)


if __name__ == "__main__":
    import os
    path = "data/raw/SP1_2425.csv"
    if os.path.exists(path):
        data = pd.read_csv(path)
        data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
        out = calculate_rolling_stats(data)
        print(f"✅ Partidos con features: {len(out)}")
        print(out[["Date", "HomeTeam", "AwayTeam",
                   "Home_Form_5", "Away_Form_5",
                   "Home_GF_5", "Away_GF_5"]].tail())
    else:
        print("❌ Corre primero: python src/data_loader.py")
