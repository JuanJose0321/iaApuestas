"""
Entrena un modelo XGBoost con calibración isotónica para predecir 1X2.
- Split temporal (no aleatorio, para evitar data-leakage)
- Calibración de probabilidades
- Guarda modelo + calibrador en /models
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_PATH, CALIBRATOR_PATH
from src.data_loader import cargar_todo
from src.feature_engineering import calculate_rolling_stats


FEATURES = [
    "Home_Form_5", "Away_Form_5",
    "Home_GF_5", "Away_GF_5",
    "Home_GC_5", "Away_GC_5",
    "B365H", "B365D", "B365A",
]
TARGET = "FTR"
MAPPING = {"H": 0, "D": 1, "A": 2}
INV_MAPPING = {0: "H", 1: "D", 2: "A"}


def preparar_dataset() -> pd.DataFrame:
    df = cargar_todo()
    df = calculate_rolling_stats(df, window=5)
    df["y"] = df[TARGET].map(MAPPING)
    df = df.dropna(subset=FEATURES + ["y"])
    return df


def split_temporal(df: pd.DataFrame, test_frac: float = 0.2):
    """Ordena por fecha y deja el último test_frac como test."""
    df = df.sort_values("Date").reset_index(drop=True)
    split = int(len(df) * (1 - test_frac))
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train, test


def entrenar():
    print("🧠 Cargando y preparando datos...")
    df = preparar_dataset()
    print(f"   Partidos con features: {len(df)}")

    train, test = split_temporal(df, test_frac=0.2)
    X_train, y_train = train[FEATURES], train["y"]
    X_test, y_test = test[FEATURES], test["y"]

    print(f"   Train: {len(train)} | Test: {len(test)}")

    # Modelo base
    base = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )
    base.fit(X_train, y_train)

    # Calibración isotónica usando el propio train via CV
    print("📐 Calibrando probabilidades (isotonic CV=3)...")
    calibrated = CalibratedClassifierCV(
        estimator=XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42,
        ),
        method="isotonic",
        cv=3,
    )
    calibrated.fit(X_train, y_train)

    # Evaluar
    preds_base = base.predict(X_test)
    probs_cal = calibrated.predict_proba(X_test)
    preds_cal = np.argmax(probs_cal, axis=1)

    acc_base = accuracy_score(y_test, preds_base)
    acc_cal = accuracy_score(y_test, preds_cal)
    ll_cal = log_loss(y_test, probs_cal, labels=[0, 1, 2])

    print(f"\n📊 Accuracy base      : {acc_base * 100:.2f}%")
    print(f"📊 Accuracy calibrado : {acc_cal * 100:.2f}%")
    print(f"📊 Log-loss calibrado : {ll_cal:.4f}  (más bajo = mejor)")

    # Benchmark: log-loss si apostamos a la cuota implícita del mercado
    impl = test[["B365H", "B365D", "B365A"]].apply(lambda r: 1 / r, axis=0)
    impl = impl.div(impl.sum(axis=1), axis=0)
    ll_mkt = log_loss(y_test, impl.values, labels=[0, 1, 2])
    print(f"📊 Log-loss mercado   : {ll_mkt:.4f}  (a batir)")

    # Guardar
    joblib.dump(base, MODEL_PATH)
    joblib.dump(calibrated, CALIBRATOR_PATH)
    print(f"\n💾 Guardados:\n   {MODEL_PATH}\n   {CALIBRATOR_PATH}")


if __name__ == "__main__":
    entrenar()
