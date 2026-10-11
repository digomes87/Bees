"""Synthetic continent-wide series and range maps used by the charts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bee_migration.climate.period import StudyPeriod
from bee_migration.climate.region import Coordinates
from bee_migration.climate.simulation import DEFAULT_SEED
from bee_migration.climate.trend import Drift, Trend

SERIES_TRENDS = {
    "temperature": Trend(Drift(22.5, 2.5), spread=0.5),
    "latitude": Trend(Drift(-15.0, -3.0), spread=0.3),
    "species_count": Trend(Drift(45.0, -8.0), spread=2.0, floor=20.0),
    "honey_production": Trend(Drift(25.0, -5.0), spread=1.5, floor=15.0),
}
RANGE_SPREAD = Coordinates(latitude=4.0, longitude=8.0)


@dataclass(frozen=True)
class SpeciesRange:
    """Where bees are found during a period, and how abundant they are."""

    period: str
    centre: Coordinates
    mean_abundance: float

    def points(self, count: int, generator: np.random.Generator) -> pd.DataFrame:
        positions = self.centre.scattered(RANGE_SPREAD, count, generator)
        abundance = generator.exponential(self.mean_abundance, count)
        return positions.assign(period=self.period, abundance=abundance)


RANGES = (
    SpeciesRange("Current (2020-2023)", Coordinates(-12.0, -55.0), 2.0),
    SpeciesRange("Projected (2040-2050)", Coordinates(-18.0, -58.0), 1.5),
)


def continental_series(period: StudyPeriod, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    columns = {
        name: trend.sample(period.length(), generator)
        for name, trend in SERIES_TRENDS.items()
    }
    return pd.DataFrame({"year": period.years(), **columns})


def distribution_shift(
    seed: int = DEFAULT_SEED, points_per_period: int = 100
) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    clouds = [species.points(points_per_period, generator) for species in RANGES]
    return pd.concat(clouds, ignore_index=True)
