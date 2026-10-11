from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.inspection import permutation_importance

from bee_migration.climate.observations import BEE_MEASUREMENTS, Observations
from bee_migration.prediction.scores import RegressionScore

DEFAULT_TARGET = "bee_abundance"
IMPORTANCE_REPEATS = 10
IMPORTANCE_SEED = 42
CLIMATE_FEATURES = [
    "year_normalized",
    "latitude",
    "longitude",
    "temperature",
    "precipitation",
    "temperature_precipitation_ratio",
    "temperature_by_latitude",
    "precipitation_by_latitude",
    "temperature_previous_year",
    "precipitation_previous_year",
    "temperature_change",
    "precipitation_change",
]


@dataclass(frozen=True)
class Sample:
    """Feature rows paired with the value to predict for each."""

    features: np.ndarray
    target: np.ndarray

    def score(self, model: RegressorMixin) -> RegressionScore:
        return RegressionScore.of(self.target, model.predict(self.features))

    def importance(self, model: RegressorMixin, names: Sequence[str]) -> pd.Series:
        """Loss of accuracy when each feature is shuffled."""
        shuffled = permutation_importance(
            model,
            self.features,
            self.target,
            n_repeats=IMPORTANCE_REPEATS,
            random_state=IMPORTANCE_SEED,
        )
        return pd.Series(shuffled.importances_mean, index=list(names))


@dataclass(frozen=True)
class TrainTestSplit:
    train: Sample
    test: Sample


class FeatureTable:
    """Predictors known before a year's bee measurements are taken."""

    def __init__(
        self, observations: Observations, target: str = DEFAULT_TARGET
    ) -> None:
        if target not in BEE_MEASUREMENTS:
            raise ValueError(
                f"Unknown target '{target}', expected one of {BEE_MEASUREMENTS}"
            )
        self._target = target
        self._table = _engineered(observations.table(), target)

    def names(self) -> list[str]:
        regions = [name for name in self._table.columns if name.startswith("region_")]
        return [*CLIMATE_FEATURES, *regions, f"{self._target}_previous_year"]

    def split(self, holdout_years: int) -> TrainTestSplit:
        """Train on the earlier years, test on the most recent `holdout_years`."""
        recent = self._recent(holdout_years)
        return TrainTestSplit(self._sample(~recent), self._sample(recent))

    def holdout(self, holdout_years: int) -> pd.DataFrame:
        """Year, region and observed target of every held-out row, in test order."""
        rows = self._table[self._recent(holdout_years)]
        labels = rows[["year", "region", self._target]]
        return labels.rename(columns={self._target: "actual"}).reset_index(drop=True)

    def _recent(self, holdout_years: int) -> pd.Series:
        years = sorted(self._table["year"].unique())
        if not 0 < holdout_years < len(years):
            raise ValueError(
                f"Cannot holdout {holdout_years} of {len(years)} usable years"
            )
        return self._table["year"] >= years[-holdout_years]

    def _sample(self, selected: pd.Series) -> Sample:
        rows = self._table[selected]
        features = rows[self.names()].to_numpy(dtype=float)
        return Sample(features, rows[self._target].to_numpy(dtype=float))


def _engineered(table: pd.DataFrame, target: str) -> pd.DataFrame:
    by_region = table.groupby("region")
    first_year = table["year"].min()
    elapsed = table["year"].max() - first_year
    lagged = table.assign(
        year_normalized=(table["year"] - first_year) / max(elapsed, 1),
        temperature_precipitation_ratio=table["temperature"]
        / (table["precipitation"] + 1),
        temperature_by_latitude=table["temperature"] * table["latitude"],
        precipitation_by_latitude=table["precipitation"] * table["latitude"],
        temperature_previous_year=by_region["temperature"].shift(1),
        precipitation_previous_year=by_region["precipitation"].shift(1),
        **{f"{target}_previous_year": by_region[target].shift(1)},
    )
    changed = lagged.assign(
        temperature_change=lagged["temperature"] - lagged["temperature_previous_year"],
        precipitation_change=lagged["precipitation"]
        - lagged["precipitation_previous_year"],
    )
    flags = pd.get_dummies(changed["region"], prefix="region", dtype=float)
    encoded = pd.concat([changed, flags], axis=1)
    return encoded.dropna().sort_values("year", kind="stable").reset_index(drop=True)
