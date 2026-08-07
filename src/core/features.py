"""
Feature engineering para fútbol.
Calcula rachas (forma) y promedios de goles anotados/recibidos por equipo,
correctamente separando local y visitante sin data-leakage.
"""
import polars as pl


def _stats_por_equipo(df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """
    Construye una tabla larga (una fila por equipo-partido) con puntos y goles,
    calcula rolling means SHIFTADAS (sin el partido actual) y devuelve el dataframe.
    """
    df = df.sort("Date").with_row_index("match_idx")

    pts_home = (
        pl.when(pl.col("FTR") == "H").then(3)
        .when(pl.col("FTR") == "D").then(1)
        .otherwise(0)
    )
    pts_away = (
        pl.when(pl.col("FTR") == "A").then(3)
        .when(pl.col("FTR") == "D").then(1)
        .otherwise(0)
    )

    home = df.select(
        pl.col("match_idx"), pl.col("Date"),
        pl.col("HomeTeam").alias("team"), pl.col("AwayTeam").alias("rival"),
        pl.lit(1).alias("is_home"),
        pl.col("FTHG").alias("goles_a_favor"), pl.col("FTAG").alias("goles_en_contra"),
        pts_home.alias("pts"),
    )
    away = df.select(
        pl.col("match_idx"), pl.col("Date"),
        pl.col("AwayTeam").alias("team"), pl.col("HomeTeam").alias("rival"),
        pl.lit(0).alias("is_home"),
        pl.col("FTAG").alias("goles_a_favor"), pl.col("FTHG").alias("goles_en_contra"),
        pts_away.alias("pts"),
    )
    largo = pl.concat([home, away]).sort(["team", "Date"])

    # Rolling por equipo, SHIFT(1) para no ver el partido actual
    largo = largo.with_columns([
        pl.col("pts").cast(pl.Float64).shift(1)
          .rolling_mean(window_size=window).over("team").alias(f"form_{window}"),
        pl.col("goles_a_favor").cast(pl.Float64).shift(1)
          .rolling_mean(window_size=window).over("team").alias(f"gf_{window}"),
        pl.col("goles_en_contra").cast(pl.Float64).shift(1)
          .rolling_mean(window_size=window).over("team").alias(f"gc_{window}"),
    ])

    return largo


def calculate_rolling_stats(df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """
    Añade al dataframe original las columnas de forma y goles rolling
    para equipo local y visitante. Elimina filas sin historial suficiente.
    """
    df = df.sort("Date").with_row_index("match_idx")
    largo = _stats_por_equipo(df.drop("match_idx"), window=window)

    home_stats = largo.filter(pl.col("is_home") == 1).select(
        "match_idx",
        pl.col(f"form_{window}").alias(f"Home_Form_{window}"),
        pl.col(f"gf_{window}").alias(f"Home_GF_{window}"),
        pl.col(f"gc_{window}").alias(f"Home_GC_{window}"),
    )
    away_stats = largo.filter(pl.col("is_home") == 0).select(
        "match_idx",
        pl.col(f"form_{window}").alias(f"Away_Form_{window}"),
        pl.col(f"gf_{window}").alias(f"Away_GF_{window}"),
        pl.col(f"gc_{window}").alias(f"Away_GC_{window}"),
    )

    out = df.join(home_stats, on="match_idx", how="left").join(away_stats, on="match_idx", how="left")
    out = out.drop_nulls(subset=[
        f"Home_Form_{window}", f"Away_Form_{window}",
        f"Home_GF_{window}", f"Away_GF_{window}",
    ])
    return out.drop("match_idx")


if __name__ == "__main__":
    import os
    path = "data/raw/SP1_2425.csv"
    if os.path.exists(path):
        data = pl.read_csv(path, encoding="latin-1", ignore_errors=True)
        data = data.with_columns(pl.col("Date").str.strptime(pl.Date, "%d/%m/%Y"))
        out = calculate_rolling_stats(data)
        print(f"✅ Partidos con features: {len(out)}")
        print(out.select(["Date", "HomeTeam", "AwayTeam",
                          "Home_Form_5", "Away_Form_5",
                          "Home_GF_5", "Away_GF_5"]).tail())
    else:
        print("❌ Corre primero: python src/providers/loader.py")
