"""
Entrena un modelo XGBoost (API nativa) con calibración isotónica propia
para predecir 1X2 — sin scikit-learn/scipy (dependencias duras que se
sacaron del bundle de despliegue).

- Split temporal (no aleatorio, para evitar data-leakage) para train/test.
- K-fold (3) sobre el train para obtener probabilidades out-of-fold y
  calibrar sin overfitting, igual que hacía sklearn.CalibratedClassifierCV.
- Calibración isotónica por clase (one-vs-rest) vía PAVA (Pool Adjacent
  Violators), reimplementada en src/core/calibration.py — validada contra
  sklearn.isotonic.IsotonicRegression con diferencia 0.0 en datos sintéticos.
- Guarda un CalibratedBooster (booster final + calibradores) en /models.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import polars as pl
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import MODEL_PATH, CALIBRATOR_PATH
from src.providers.loader import cargar_todo
from src.core.features import calculate_rolling_stats
from src.core.calibration import CalibratedBooster, IsotonicCalibrator


FEATURES = [
    "Home_Form_5", "Away_Form_5",
    "Home_GF_5", "Away_GF_5",
    "Home_GC_5", "Away_GC_5",
    "B365H", "B365D", "B365A",
]
TARGET = "FTR"
MAPPING = {"H": 0, "D": 1, "A": 2}
INV_MAPPING = {0: "H", 1: "D", 2: "A"}
N_CLASES = 3

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": N_CLASES,
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "seed": 42,
}
N_ESTIMATORS = 300


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _log_loss(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    """Log-loss multiclase estándar: -mean(log(p de la clase real))."""
    probs = np.clip(probs, eps, 1 - eps)
    n = len(y_true)
    return float(-np.mean(np.log(probs[np.arange(n), y_true])))


def preparar_dataset() -> pl.DataFrame:
    df = cargar_todo()
    df = calculate_rolling_stats(df, window=5)
    y = (
        pl.when(pl.col(TARGET) == "H").then(0)
        .when(pl.col(TARGET) == "D").then(1)
        .when(pl.col(TARGET) == "A").then(2)
        .otherwise(None)
        .alias("y")
    )
    df = df.with_columns(y)
    df = df.drop_nulls(subset=FEATURES + ["y"])
    return df


def split_temporal(df: pl.DataFrame, test_frac: float = 0.2):
    """Ordena por fecha y deja el último test_frac como test."""
    df = df.sort("Date")
    split = int(len(df) * (1 - test_frac))
    train = df[:split]
    test = df[split:]
    return train, test


def _kfold_indices(n: int, k: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """K folds aleatorios (índices) — igual de espíritu que StratifiedKFold(shuffle=True)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    resultado = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        resultado.append((train_idx, val_idx))
    return resultado


def _entrenar_booster(X: np.ndarray, y: np.ndarray) -> xgb.Booster:
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(XGB_PARAMS, dtrain, num_boost_round=N_ESTIMATORS)


def entrenar():
    print("🧠 Cargando y preparando datos...")
    df = preparar_dataset()
    print(f"   Partidos con features: {len(df)}")

    train, test = split_temporal(df, test_frac=0.2)
    X_train = train.select(FEATURES).to_numpy()
    y_train = train["y"].to_numpy().astype(int)
    X_test = test.select(FEATURES).to_numpy()
    y_test = test["y"].to_numpy().astype(int)

    print(f"   Train: {len(train)} | Test: {len(test)}")

    # ── 1. Out-of-fold: probabilidades sin overfitting para calibrar ──
    print("📐 Generando probabilidades out-of-fold (3-fold)...")
    oof_probs = np.zeros((len(X_train), N_CLASES))
    for fold_train_idx, fold_val_idx in _kfold_indices(len(X_train), k=3):
        booster_fold = _entrenar_booster(X_train[fold_train_idx], y_train[fold_train_idx])
        oof_probs[fold_val_idx] = booster_fold.predict(xgb.DMatrix(X_train[fold_val_idx]))

    # ── 2. Calibración isotónica por clase (one-vs-rest) sobre las OOF ──
    print("📐 Calibrando probabilidades (isotonic, PAVA propio)...")
    calibradores = []
    for c in range(N_CLASES):
        y_bin = (y_train == c).astype(float)
        calibradores.append(IsotonicCalibrator().fit(oof_probs[:, c], y_bin))

    # ── 3. Booster final sobre TODO el train (el que de verdad se despliega) ──
    booster_final = _entrenar_booster(X_train, y_train)
    modelo = CalibratedBooster(booster=booster_final, calibradores=calibradores)

    # ── 4. Evaluación en test ──
    probs_sin_cal = booster_final.predict(xgb.DMatrix(X_test))
    probs_cal = modelo.predict(xgb.DMatrix(X_test))
    preds_cal = np.argmax(probs_cal, axis=1)

    acc = _accuracy(y_test, preds_cal)
    ll_sin_cal = _log_loss(y_test, probs_sin_cal)
    ll_cal = _log_loss(y_test, probs_cal)

    print(f"\n📊 Accuracy (calibrado)   : {acc * 100:.2f}%")
    print(f"📊 Log-loss sin calibrar  : {ll_sin_cal:.4f}")
    print(f"📊 Log-loss calibrado     : {ll_cal:.4f}  (más bajo = mejor)")

    odds = test.select(["B365H", "B365D", "B365A"]).to_numpy()
    implied = 1.0 / odds
    implied = implied / implied.sum(axis=1, keepdims=True)
    ll_mkt = _log_loss(y_test, implied)
    print(f"📊 Log-loss mercado       : {ll_mkt:.4f}  (a batir)")

    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(modelo, CALIBRATOR_PATH)
    print(f"\n💾 Guardados:\n   {MODEL_PATH}\n   {CALIBRATOR_PATH}")


if __name__ == "__main__":
    entrenar()
