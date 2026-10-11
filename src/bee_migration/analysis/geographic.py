import pandas as pd

from bee_migration.climate.observations import MEASUREMENTS, Observations

SUMMARY_STATISTICS = ["mean", "std", "min", "max"]


class GeographicPatterns:
    """How measurements differ between regions and along latitude."""

    def __init__(self, observations: Observations) -> None:
        self._observations = observations

    def summary(self) -> pd.DataFrame:
        measurements = self._observations.table().groupby("region")[list(MEASUREMENTS)]
        return measurements.agg(SUMMARY_STATISTICS).round(2)

    def latitudinal_gradients(self) -> pd.Series:
        """Correlation of every measurement with latitude."""
        table = self._observations.table()
        return table[list(MEASUREMENTS)].corrwith(table["latitude"])

    def hotspots(self) -> pd.DataFrame:
        """Regions ranked by how much temperature and bee abundance swung."""
        swings = {
            region: _swings(rows) for region, rows in self._observations.by_region()
        }
        ranked = pd.DataFrame.from_dict(swings, orient="index")
        return ranked.sort_values("total_change_score", ascending=False)


def _range(series: pd.Series) -> float:
    return float(series.max() - series.min())


def _swings(rows: pd.DataFrame) -> dict[str, float]:
    temperature = _range(rows["temperature"])
    abundance = _range(rows["bee_abundance"])
    return {
        "temperature_change": temperature,
        "bee_abundance_change": abundance,
        "total_change_score": temperature + abundance,
    }
