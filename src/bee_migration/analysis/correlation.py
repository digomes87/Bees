from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from bee_migration.climate.observations import MEASUREMENTS, Observations

CORRELATED_COLUMNS = [
    *MEASUREMENTS,
    "temperature_anomaly",
    "precipitation_anomaly",
    "climate_stress_index",
    "bee_health_index",
]
MODERATE_THRESHOLD = 0.3
NOTABLE_THRESHOLD = MODERATE_THRESHOLD
STRONG_THRESHOLD = 0.7
YEAR_LAGS = (0, 1, 2, 3)


@dataclass(frozen=True)
class CorrelationPair:
    variables: tuple[str, str]
    coefficient: float

    def magnitude(self) -> float:
        return abs(self.coefficient)

    def strength(self) -> str:
        if self.magnitude() >= STRONG_THRESHOLD:
            return "strong"
        if self.magnitude() >= MODERATE_THRESHOLD:
            return "moderate"
        return "weak"

    def record(self) -> dict[str, str | float]:
        first, second = self.variables
        return {
            "first": first,
            "second": second,
            "coefficient": self.coefficient,
            "strength": self.strength(),
        }

    def describe(self) -> str:
        first, second = self.variables
        return f"{first} <-> {second}: {self.coefficient:+.3f} ({self.strength()})"


class Correlations:
    """Linear relationships between climate and bee measurements."""

    def __init__(self, observations: Observations) -> None:
        self._observations = observations

    def matrix(self) -> pd.DataFrame:
        return self._observations.table()[CORRELATED_COLUMNS].corr()

    def by_region(self) -> dict[str, pd.DataFrame]:
        return {
            region: rows[CORRELATED_COLUMNS].corr()
            for region, rows in self._observations.by_region()
        }

    def notable_pairs(
        self, threshold: float = NOTABLE_THRESHOLD
    ) -> list[CorrelationPair]:
        """Distinct variable pairs correlated at least `threshold`, strongest first."""
        matrix = self.matrix()
        pairs = [
            CorrelationPair((first, second), float(matrix.loc[first, second]))
            for first, second in combinations(CORRELATED_COLUMNS, 2)
        ]
        notable = [pair for pair in pairs if pair.magnitude() >= threshold]
        return sorted(notable, key=CorrelationPair.magnitude, reverse=True)

    def lagged(self, lags: Sequence[int] = YEAR_LAGS) -> pd.DataFrame:
        """Temperature against bee abundance `lag` years later, by region."""
        delayed = {
            region: _delayed_response(rows, lags)
            for region, rows in self._observations.by_region()
        }
        return pd.DataFrame.from_dict(delayed, orient="index")


def _delayed_response(rows: pd.DataFrame, lags: Sequence[int]) -> dict[int, float]:
    temperature = rows["temperature"]
    abundance = rows["bee_abundance"]
    return {lag: float(temperature.corr(abundance.shift(-lag))) for lag in lags}
