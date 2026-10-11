from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

CLIMATE_MEASUREMENTS = ("temperature", "precipitation")
BEE_MEASUREMENTS = ("bee_abundance", "species_diversity", "honey_production")
MEASUREMENTS = (*CLIMATE_MEASUREMENTS, *BEE_MEASUREMENTS)
NUMERIC_COLUMNS = ("year", "latitude", "longitude", *MEASUREMENTS)
REQUIRED_COLUMNS = ("year", "region", "latitude", "longitude", *MEASUREMENTS)
HEALTH_WEIGHTS = {
    "bee_abundance": 0.4,
    "species_diversity": 0.3,
    "honey_production": 0.3,
}


class Observations:
    """Yearly climate and bee measurements per region, with derived indices."""

    def __init__(self, table: pd.DataFrame) -> None:
        missing = [name for name in REQUIRED_COLUMNS if name not in table.columns]
        if missing:
            raise ValueError(f"Observations are missing columns: {', '.join(missing)}")
        if table.empty:
            raise ValueError("Observations cannot be empty")
        self._table = _with_indices(_consistent(_ordered(_numeric(table))))

    @classmethod
    def from_csv(cls, path: Path) -> Observations:
        return cls(pd.read_csv(path))

    def __len__(self) -> int:
        return len(self._table)

    def table(self) -> pd.DataFrame:
        return self._table.copy()

    def region_names(self) -> list[str]:
        return sorted(self._table["region"].unique())

    def by_region(self) -> Iterator[tuple[str, pd.DataFrame]]:
        for region, rows in self._table.groupby("region"):
            yield str(region), rows.copy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._table.to_csv(path, index=False)


def _numeric(table: pd.DataFrame) -> pd.DataFrame:
    try:
        numbers = {name: pd.to_numeric(table[name]) for name in NUMERIC_COLUMNS}
    except (TypeError, ValueError) as error:
        raise ValueError(f"Observations hold non-numeric values: {error}") from error
    return table.assign(**numbers)


def _skips_years(years: pd.Series) -> bool:
    return years.max() - years.min() + 1 != len(years)


def _consistent(table: pd.DataFrame) -> pd.DataFrame:
    """Reject tables the yearly lags and trends cannot be computed from."""
    if table.isna().any().any():
        raise ValueError("Observations contain missing values")
    if table.duplicated(["region", "year"]).any():
        raise ValueError("Observations repeat a year within a region")
    if table.groupby("region")["year"].agg(_skips_years).any():
        raise ValueError("Observations skip years within a region")
    return table


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    raw = table[list(REQUIRED_COLUMNS)]
    return raw.sort_values(["region", "year"]).reset_index(drop=True)


def _anomaly(table: pd.DataFrame, column: str) -> pd.Series:
    return table[column] - table.groupby("region")[column].transform("mean")


def _standardised(table: pd.DataFrame, column: str) -> pd.Series:
    return _anomaly(table, column) / table.groupby("region")[column].transform("std")


def _stress(table: pd.DataFrame) -> pd.Series:
    """Mean absolute departure from the regional norm, in standard deviations."""
    temperature = _standardised(table, "temperature").abs()
    precipitation = _standardised(table, "precipitation").abs()
    return (temperature + precipitation) / 2


def _health(table: pd.DataFrame) -> pd.Series:
    weighted = [table[name] * weight for name, weight in HEALTH_WEIGHTS.items()]
    return sum(weighted[1:], start=weighted[0])


def _with_indices(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(
        temperature_anomaly=_anomaly(table, "temperature"),
        precipitation_anomaly=_anomaly(table, "precipitation"),
        climate_stress_index=_stress(table),
        bee_health_index=_health(table),
    )
