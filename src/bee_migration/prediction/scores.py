from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SMALLEST_DENOMINATOR = 1e-8


@dataclass(frozen=True)
class RegressionScore:
    """How closely predictions match the observed values."""

    rmse: float
    mae: float
    r2: float
    mape: float

    @classmethod
    def of(cls, actual: np.ndarray, predicted: np.ndarray) -> RegressionScore:
        scale = np.maximum(np.abs(actual), SMALLEST_DENOMINATOR)
        return cls(
            rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
            mae=float(mean_absolute_error(actual, predicted)),
            r2=float(r2_score(actual, predicted)),
            mape=float(np.mean(np.abs(actual - predicted) / scale) * 100),
        )

    def labelled(self, prefix: str) -> dict[str, float]:
        return {f"{prefix}_{name}": value for name, value in asdict(self).items()}

    def describe(self, label: str) -> list[str]:
        return [
            f"  {label} R2: {self.r2:.4f}",
            f"  {label} RMSE: {self.rmse:.4f}",
            f"  {label} MAE: {self.mae:.4f}",
            f"  {label} MAPE: {self.mape:.2f}%",
        ]


@dataclass(frozen=True)
class Evaluation:
    """Scores on the years a model learned from and on the years held out."""

    train: RegressionScore
    test: RegressionScore

    def labelled(self) -> dict[str, float]:
        return {**self.train.labelled("train"), **self.test.labelled("test")}
