import numpy as np
import pandas as pd

from bee_migration.climate.observations import MEASUREMENTS, Observations

CHANGE_POINT_COLUMNS = ("temperature", "bee_abundance")
CHANGE_POINT_SIGMAS = 2.0


class TemporalPatterns:
    """How measurements evolve over the years within each region."""

    def __init__(self, observations: Observations) -> None:
        self._observations = observations

    def trends(self) -> pd.DataFrame:
        """Linear change per year of every measurement, by region."""
        slopes = {
            region: _yearly_slopes(rows)
            for region, rows in self._observations.by_region()
        }
        return pd.DataFrame.from_dict(slopes, orient="index")

    def change_points(self) -> pd.DataFrame:
        """Years where a series bends unusually sharply, by region."""
        changes = {
            region: _change_years(rows)
            for region, rows in self._observations.by_region()
        }
        return pd.DataFrame.from_dict(changes, orient="index")


def _yearly_slopes(rows: pd.DataFrame) -> dict[str, float]:
    return {
        name: float(np.polyfit(rows["year"], rows[name], 1)[0]) for name in MEASUREMENTS
    }


def _change_years(rows: pd.DataFrame) -> dict[str, list[int]]:
    return {name: _sharp_bends(rows, name) for name in CHANGE_POINT_COLUMNS}


def _sharp_bends(rows: pd.DataFrame, column: str) -> list[int]:
    curvature = np.diff(rows[column].to_numpy(), n=2)
    threshold = CHANGE_POINT_SIGMAS * curvature.std()
    later_years = rows["year"].to_numpy()[2:]
    return later_years[np.abs(curvature) > threshold].tolist()
