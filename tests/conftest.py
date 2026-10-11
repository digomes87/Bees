import pandas as pd
import pytest

from bee_migration.climate.observations import Observations
from bee_migration.climate.period import StudyPeriod
from bee_migration.climate.region import BRAZILIAN_REGIONS
from bee_migration.climate.simulation import ClimateSimulation


@pytest.fixture(scope="session")
def observations() -> Observations:
    simulation = ClimateSimulation(BRAZILIAN_REGIONS, StudyPeriod(2000, 2023))
    return simulation.observations(seed=42)


def region_table(region: str, **columns: list[float]) -> pd.DataFrame:
    """Build a raw single-region table, filling unspecified measurements."""
    length = len(next(iter(columns.values())))
    defaults = {
        "temperature": [20.0 + step for step in range(length)],
        "precipitation": [1000.0 - 10 * step for step in range(length)],
        "bee_abundance": [80.0 - step for step in range(length)],
        "species_diversity": [40.0 - step for step in range(length)],
        "honey_production": [25.0 - step for step in range(length)],
    }
    return pd.DataFrame(
        {
            "year": range(2000, 2000 + length),
            "region": region,
            "latitude": -10.0,
            "longitude": -50.0,
            **defaults,
            **columns,
        }
    )
