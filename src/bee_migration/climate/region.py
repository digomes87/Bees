from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bee_migration.climate.observations import MEASUREMENTS
from bee_migration.climate.period import StudyPeriod
from bee_migration.climate.trend import Drift, Trend


@dataclass(frozen=True)
class Coordinates:
    latitude: float
    longitude: float

    def scattered(
        self, spread: "Coordinates", count: int, generator: np.random.Generator
    ) -> pd.DataFrame:
        """Random positions around this point, `spread` being the deviations."""
        return pd.DataFrame(
            {
                "latitude": generator.normal(self.latitude, spread.latitude, count),
                "longitude": generator.normal(self.longitude, spread.longitude, count),
            }
        )


SAMPLING_SPREAD = Coordinates(latitude=1.0, longitude=2.0)


@dataclass(frozen=True)
class RegionProfile:
    """How a region's climate and bee populations evolve over a study."""

    name: str
    centre: Coordinates
    trends: Mapping[str, Trend]

    def simulate(
        self, period: StudyPeriod, generator: np.random.Generator
    ) -> pd.DataFrame:
        length = period.length()
        positions = self.centre.scattered(SAMPLING_SPREAD, length, generator)
        measurements = {
            name: trend.sample(length, generator) for name, trend in self.trends.items()
        }
        return positions.assign(year=period.years(), region=self.name, **measurements)


# Change over the whole study, shared by every region: warming, drying, decline.
STUDY_WIDE_CHANGE = {
    "temperature": Trend(Drift(0.0, 2.8), spread=0.6),
    "precipitation": Trend(Drift(0.0, -150.0), spread=100.0, floor=300.0),
    "bee_abundance": Trend(Drift(0.0, -20.0), spread=5.0, floor=20.0),
    "species_diversity": Trend(Drift(0.0, -8.0), spread=2.0, floor=15.0),
    "honey_production": Trend(Drift(0.0, -6.0), spread=2.0, floor=12.0),
}


def _profile(
    name: str, centre: Coordinates, baselines: tuple[float, ...]
) -> RegionProfile:
    starts = dict(zip(MEASUREMENTS, baselines, strict=True))
    trends = {
        name: STUDY_WIDE_CHANGE[name].starting_at(start)
        for name, start in starts.items()
    }
    return RegionProfile(name, centre, trends)


# Baselines follow MEASUREMENTS: °C, mm/year, abundance index, species, kg/hive.
BRAZILIAN_REGIONS = (
    _profile("North", Coordinates(-3.0, -60.0), (26.5, 2200.0, 85.0, 45.0, 28.0)),
    _profile("Northeast", Coordinates(-9.0, -40.0), (25.8, 800.0, 65.0, 35.0, 22.0)),
    _profile(
        "Center-West", Coordinates(-15.0, -56.0), (24.2, 1400.0, 75.0, 40.0, 26.0)
    ),
    _profile("Southeast", Coordinates(-20.0, -44.0), (22.1, 1300.0, 70.0, 38.0, 24.0)),
    _profile("South", Coordinates(-27.0, -51.0), (19.8, 1600.0, 60.0, 32.0, 30.0)),
)
