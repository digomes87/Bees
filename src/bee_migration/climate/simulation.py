from collections.abc import Sequence

import numpy as np
import pandas as pd

from bee_migration.climate.observations import Observations
from bee_migration.climate.period import StudyPeriod
from bee_migration.climate.region import RegionProfile

DEFAULT_SEED = 42


class ClimateSimulation:
    """Generates synthetic observations for demonstrating the analysis."""

    def __init__(self, profiles: Sequence[RegionProfile], period: StudyPeriod) -> None:
        self._profiles = tuple(profiles)
        self._period = period

    def observations(self, seed: int = DEFAULT_SEED) -> Observations:
        generator = np.random.default_rng(seed)
        tables = [
            profile.simulate(self._period, generator) for profile in self._profiles
        ]
        return Observations(pd.concat(tables, ignore_index=True))
