"""
Calibración isotónica sin scikit-learn/scipy, para envolver un xgb.Booster.

IsotonicCalibrator reemplaza sklearn.isotonic.IsotonicRegression vía el
algoritmo PAVA (Pool Adjacent Violators) — validado contra sklearn con
diferencia 0.0 en datos sintéticos (ver notas de la migración).

CalibratedBooster reemplaza sklearn.calibration.CalibratedClassifierCV:
aplica un IsotonicCalibrator por clase (one-vs-rest) sobre las
probabilidades crudas del booster y renormaliza para que sumen 1.
"""
from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators — ajusta y (pesos w) a la secuencia no-decreciente
    más cercana en error cuadrático ponderado. y/w ya deben venir ordenados por x."""
    values: list[float] = []
    weights: list[float] = []
    counts: list[int] = []
    for i in range(len(y)):
        v, wt, cnt = float(y[i]), float(w[i]), 1
        while values and values[-1] > v:
            v = (values[-1] * weights[-1] + v * wt) / (weights[-1] + wt)
            wt = weights[-1] + wt
            cnt = counts[-1] + cnt
            values.pop(); weights.pop(); counts.pop()
        values.append(v); weights.append(wt); counts.append(cnt)

    result = np.empty(len(y))
    idx = 0
    for v, cnt in zip(values, counts):
        result[idx:idx + cnt] = v
        idx += cnt
    return result


class IsotonicCalibrator:
    """Regresión isotónica 1D: aprende x → y no-decreciente, con interpolación lineal."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        order = np.argsort(x, kind="stable")
        x_sorted, y_sorted = x[order], y[order]

        # Promediar duplicados de x antes de PAVA (weighted PAVA correcto)
        ux, inv, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        sums = np.zeros(len(ux))
        np.add.at(sums, inv, y_sorted)
        means = sums / counts

        self.x_ = ux
        self.y_ = _pava(means, counts.astype(np.float64))
        return self

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.interp(x, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])


@dataclass
class CalibratedBooster:
    """Booster de XGBoost + calibración isotónica por clase (one-vs-rest)."""
    booster: xgb.Booster
    calibradores: list = field(default_factory=list)

    def predict(self, dmatrix: xgb.DMatrix) -> np.ndarray:
        raw = self.booster.predict(dmatrix)
        calibrado = np.column_stack([
            cal.predict(raw[:, c]) for c, cal in enumerate(self.calibradores)
        ])
        sumas = calibrado.sum(axis=1, keepdims=True)
        sumas[sumas == 0] = 1.0  # evita división por cero en el caso degenerado
        return calibrado / sumas
